# Restarting a generated CC run from persisted lower-rank amplitudes

This scopes one capability: **write the converged amplitudes of a generated
arbitrary-order RCC solve to disk, and on a later run read them back to seed the
iteration** — so a `cc4` run can start from a `cc4`/`ccsdt` solve done in a
previous process, not only from a lower-rank solve done in the *same* process
(which is W6).

It is the persistence sibling of W6. W6 warm-starts amplitudes **in memory**
within one run (`seed_arbitrary_order_amplitudes`, landed). This work reuses that
exact seed hook, feeding it amplitudes loaded from a file instead of from a
lower-rank solve held in RAM.

Everything below is grounded in the current tree; nothing here is landed.

---

## What exists today (the reuse surface)

- **The seed hook.** `seed_arbitrary_order_amplitudes(state, seed)`
  (`generated_arbitrary_prepare.cpp`) overwrites the lowest ranks of a prepared
  state's zero amplitudes with a supplied `ArbitraryOrderRCCAmplitudes`,
  validating per-rank dims. This is the single injection point — a restart lands
  here identically to a W6 in-memory seed. **No new seeding machinery is needed.**
- **The amplitude container is trivially serializable.**
  `ArbitraryOrderRCCAmplitudes` is `std::vector<TensorND> by_rank`; each
  `TensorND` is `{std::vector<int> dims; std::vector<double> data;}`
  (`common.h`). Flat dims + flat doubles — the whole payload is two length-prefixed
  arrays per rank.
- **The write/read primitives.** `checkpoint.cpp` already has `write_matrix` /
  `write_string` / scalar reads/writes and a magic+version framing convention.
  The sidecar reuses the same little-endian, length-prefixed style (not the same
  file).
- **The dims come from the reference.** A prepared state's expected rank dims are
  `[n_occ…, n_virt…]` from `reference.orbital_partition` — the loader validates a
  read-back seed against the live state's slots (the seed hook already does this),
  so a stale/mismatched sidecar is rejected, not silently misapplied.

## Why a separate sidecar, not a bumped `.hfchk` version

- Amplitudes are `O(o^n v^n)` — orders of magnitude larger than the SCF matrices.
  Appending them to the SCF checkpoint would bloat **every** `.hfchk`, including
  the vast majority of runs that never touch CC.
- They are method/rank-specific (a `cc4` sidecar means nothing to a `cc5` run
  except as a partial seed) and reference-specific (canonical RHF MO basis).
  That metadata belongs next to the amplitudes, not in the SCF header.
- The SCF checkpoint is the SCF-restart contract; keeping amplitudes out of it
  means this feature can't regress SCF restart. `ponytail:` one file, one concern.

Sidecar path: derive from the SCF checkpoint path (`<stem>.ccamp`) or an explicit
`cc_checkpoint <path>` keyword. Default: alongside the SCF checkpoint.

---

## Scope (small verifiable steps, persistence first)

- **X0 — the sidecar format + writer (~S).** A `cc_amplitude_checkpoint.{h,cpp}`
  (or a section in `checkpoint.cpp`) with `save_cc_amplitudes(path, amps, meta)`.
  Format, little-endian:
  ```
  [8]  magic  "PLNKCCA\0"
  [4]  version u32 = 1
  [4]  max_rank i32
  [4+len] method/rank tag (e.g. "cc4")   — string
  [4+len] basis name                      — validated against the run's basis
  [8]  n_occ u64,  [8] n_virt u64          — validated against the live reference
  for r in 1..=max_rank:
    [4]  order (=2r) i32
    [order×4] dims i32[]
    [8]  count u64
    [count×8] data f64[]  (TensorND.data, native storage order)
  ```
  *Gate:* a unit test round-trips a hand-built 2-rank `ArbitraryOrderRCCAmplitudes`
  through save→load and asserts bytewise-equal `dims`/`data`; a truncated file and
  a bad magic each error, not crash.

- **X1 — the loader (~S given X0).** `load_cc_amplitudes(path) ->
  expected<{amps, meta}>`. Pure read; no state coupling yet. *Gate:* covered by
  X0's round-trip; add a version-mismatch error case.

- **X2 — write on a successful generated CC solve (~S).** In `run_rccsdtq`, after
  `solve_res->converged`, call `save_cc_amplitudes` with the converged
  `solve_res->state.amplitudes` and the reference's `n_occ`/`n_virt`/basis, gated
  on a `cc_checkpoint`/`checkpoint` keyword being active (mirror how SCF
  `save()` is gated). Failure to write is a **warning**, not a run failure — the
  energy is already computed. *Gate:* running `be_rccsdtq_sto3g` with checkpointing
  on produces a `.ccamp` file whose `load_cc_amplitudes` returns rank-4 amps with
  the Be dims.

- **X3 — read + seed on restart (~S given the seed hook).** In `run_rccsdtq`
  (inside `solve_generated_rcc`, before the iteration loop), if a sidecar is
  present and enabled: `load_cc_amplitudes` → `seed_arbitrary_order_amplitudes`.
  The hook's per-rank dim check already rejects a mismatched file; on any load or
  seed error, log a warning and fall through to the cold/W6 path (a restart is an
  optimization, never a correctness gate). This composes with W6: a restart seeds
  whatever ranks the sidecar has; W6's in-memory recursion still fills higher
  ranks if the sidecar is partial. *Gate:* a two-run test — run `be_rccsdtq_sto3g`
  once to write the sidecar, then again reading it; the second run reaches the same
  FCI energy (-14.4036550465) in **materially fewer iterations**, and a corrupt
  sidecar still converges (falls back).

- **X4 — the `cc4`-from-`ccsdt` cross-rank restart (~M, the user's ask).** X3
  restarts `cc4` from a prior `cc4`. To restart `cc4` from a prior **`ccsdt`** run
  (seed T1/T2/T3, iterate mostly T4) the sidecar must be written by an
  *arbitrary-order* rank-3 solve. Two sub-parts:
  - **X4.0 — emit the rank-3 kernel against `ArbitraryOrderRCCAmplitudes`.** Today
    `ccsdt_planck_generated.cpp` targets `RCCSDTAmplitudes` (the fixed rank-3
    tensor_backend type), so its kernels don't match the arbitrary runtime
    signature and can't run through `run_generated_arbitrary_order_iterations`.
    The codegen already emits the arbitrary variant for rank ≥ 4 (W0.1); extend
    the CMake codegen to also emit a rank-3 arbitrary TU, and lower the
    `make_generated_rcc_kernels` floor from 4 to 3 so it registers. This is the
    same blocker W6 hit for in-memory `cc4` seeding — X4.0 unblocks **both**.
  - **X4.1 — write the `ccsdt` sidecar / read it in a `cc4` run.** With X4.0, a
    `correlation ccsdt` run (routed through the arbitrary path) writes a rank-3
    `.ccamp`; a later `cc4` run reads it via X3 and seeds T1/T2/T3. *Gate:* a
    `ccsdt`→`cc4` two-run test on Be converges in fewer T4-dominated iterations
    than cold `cc4`, same energy.

**Sequencing / risk.** X0–X3 are the persistence spine and are all ~S — they
give same-rank restart (`cc4` from `cc4`, `cc5` from `cc5`) reusing the landed
seed hook end-to-end. X4 is the cross-rank (`cc4` from `ccsdt`) case the user
asked for; its cost is entirely in X4.0 (re-emitting rank-3 as arbitrary-order +
lowering the registry floor), which is shared with W6's own `cc4` gap. Do X0–X3
first (self-contained, gated on Be), then X4.0 once (unblocking W6 and X4
together), then X4.1.

**What NOT to do.** Do not put amplitudes in the SCF `.hfchk` (bloats every run,
couples SCF restart to CC). Do not add a bespoke reader per rank — the sidecar is
rank-generic (`max_rank` + per-rank loop), same as the runtime. Do not fail the
run when a sidecar is absent, stale, or corrupt — restart is an optimization;
degrade to cold/W6 with a warning.

- **X5 — sidecar for the hand-written ccsd/ccsdt solvers (~M, the layout
  barrier). NOT STARTED.** X0–X4 wire the sidecar only into the generated
  `run_rccsdtq` path (ranks ≥ 4). To restart a `ccsdt` (rank 3) run from a `ccsd`
  (rank 2) run — or to feed a hand-written ccsd/ccsdt result into a generated
  `cc4` — the hand-written solvers must also read/write the sidecar. The blocker
  is **layout, not plumbing**:
  - The hand-written `run_rccsd` / `run_rccsdt` iterate amplitudes in
    **spin-orbital** form (`ccsd.cpp`: `so.n_occ = 2·reference.n_occ`), and route
    through backend selectors (`determinant` / `tensor` / `optimized`) — the
    determinant-space backend does not even hold dense `t1`/`t2`.
  - The sidecar / seed hook / generated runtime are all **spatial RCC**
    (`by_rank` stored `[n_occ…, n_virt…]`, no spin factor).
  - So a hand-written→generated seed needs a **spin-orbital → spatial-RCC
    projection** (for a closed-shell RHF reference the α/β blocks are redundant;
    the projection collapses them). This is the same conversion W6/X4 flagged and
    is the real content of X5 — a byte copy will silently produce wrong amplitudes.
  Sub-steps:
  - **X5.0 — a `so → spatial` amplitude projection (~M).** One function per rank
    that reads the spin-orbital `t` and writes the spatial `TensorND`, gated by a
    unit test that round-trips a spatial→so→spatial identity and checks a known
    closed-shell case.
  - **X5.1 — write from `run_rccsd`/`run_rccsdt` (~S given X5.0 and a dense
    backend).** Only the tensor backend holds dense amplitudes; the
    determinant-space backend must be excluded (or its amplitudes reconstructed),
    so gate the write on the backend actually producing `t1`/`t2`.
  - **X5.2 — read into the generated path.** The generated `cc4` run already reads
    a spatial sidecar (X3); once ccsd/ccsdt write a **spatial** sidecar via X5.0,
    the cross-method restart works with no further generated-side change.
  **Honest ceiling.** X5 is where "checkpoint for all methods" actually lives, and
  its cost is the spin-orbital↔spatial projection plus the determinant-backend
  exclusion — not the file format (X0 already covers that). Until X5.0 lands, the
  sidecar is generated-path-only, and a literal ccsd→ccsdt restart is not yet
  possible.

---

## What this reuses

- `seed_arbitrary_order_amplitudes` (W6.0) — the one and only seed injection point.
- `checkpoint.cpp` write/read primitives and magic+version framing convention.
- `run_rccsdtq` / `solve_generated_rcc` (`ccsdtq.cpp`) — the CC entry, already the
  W6 warm-start site; the restart read/write sits at the same two points.
- The W6/X4.0 shared unblock: emitting a rank-3 arbitrary-order kernel and
  lowering the registry floor (`make_generated_rcc_kernels`, `PLANCK_CC_MAXORDER`
  codegen).

See `ccgen_kernel_wiring_and_benchmark_scope.md` (W6, the in-memory warm-start
this persists) and `src/io/checkpoint.h` (the SCF checkpoint format this sits
beside).
