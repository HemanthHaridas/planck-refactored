# `.ccamp` dumping and reading — what remains

Follow-on scope to `CC_AMPLITUDE_CHECKPOINT_SCOPE.md`, which specified X0–X5.
X0–X3 are **landed**; this document scopes what is left, and records one
**correctness defect in the landed format** found while scoping.

Grounded in the current tree.

---

## Landed (verified in-tree)

| Step | Where | State |
|---|---|---|
| X0 format + writer | `src/post_hf/cc/cc_amplitude_checkpoint.{h,cpp}` (62 + 188 lines) | landed |
| X1 loader | same, `load_cc_amplitudes` | landed; errors (never crashes) on bad magic, bad version, truncation, negative dims, count/dims disagreement, overflow |
| X2 write on success | `ccsdtq.cpp:219-243`, gated on `_save_checkpoint && !_checkpoint_path.empty()`, path = `<stem>.ccamp`, write failure is a **warning** | landed |
| X3 read + seed | `ccsdtq.cpp:30-80`, `load_cc_amplitudes` → `seed_arbitrary_order_amplitudes`, falls through to cold/W6 on any error | landed |
| X4.0 rank-3 arbitrary emit | `PLANCK_CC_ARBITRARY_LOWER_RANKS` lowers the registry floor 4→3 (`ccsdtq.cpp:175`) | landed |
| Round-trip gate | `tests/cc_amplitude_checkpoint.cpp` (round-trip + bad-magic) | landed |

Memory records the X0–X3 validation: Be cc4 cold = 18 iterations → restart = 1
iteration at loose tolerance.

---

## C0 — DEFECT: the sidecar silently drops higher Sz sectors (~S, do first)

**This is a correctness bug in the landed format, not a missing feature.**

`save_cc_amplitudes` iterates only `amplitudes.by_rank`
(`cc_amplitude_checkpoint.cpp:73`). But an `ArbitraryOrderRCCAmplitudes` for
rank ≥ 4 also carries `amplitudes.sectors` — the higher independent Sz sector
blocks, keyed `(rank, tag)`, e.g. `(4, "aaabaaab")` for CCSDTQ
(`amplitudes.h:64-70`). R3.1.3 proved these sectors are **genuinely
independent**: `aaab` is not a signed-permutation combination of `aabb`, even
from one shared spatial amplitude. So the t4 `aaabaaab` block is real,
converged data.

The consequence: a `cc4` run writes a sidecar containing only the balanced t4
sector. A later `cc4` run seeds `by_rank` from it and leaves the `aaabaaab`
sector at **zero** — a *partially* seeded state that mixes converged and
zero amplitudes. It still converges (Jacobi/DIIS pulls the zero sector up, and
the fall-through logic is sound), so the observed "1 iteration" restart is not
wrong in its final energy — but the restart is weaker than reported, and the
metadata does not record that a sector is missing.

**The fix:**

- Bump `CCAMP_VERSION` to 2 and append, after the `by_rank` loop:
  ```
  [4]  n_sectors i32
  for each sector:
    [4]      excitation_rank i32
    [4+len]  tag string
    [4]      order i32
    [order×4] dims i32[]
    [8]      count u64
    [count×8] data f64[]
  ```
  The per-tensor body is byte-identical to the `by_rank` entry, so factor the
  existing tensor write/read into a helper and call it from both — `ponytail:`
  one tensor codec, two callers.
- Keep the loader accepting version 1 (treat as `n_sectors = 0`) so existing
  sidecars stay readable. This is cheap and it is the whole reason to version a
  format.
- On seed: apply sector blocks via `sector_tensor(rank, tag)`, and where a
  sector in the file has no counterpart in the live state (or vice versa), warn
  and skip that block rather than failing — same degradation policy X3 already
  uses.

*Gate:* extend `tests/cc_amplitude_checkpoint.cpp` to round-trip an amplitude
set with one `(4, "aaabaaab")` sector and assert bytewise-equal `dims`/`data`
for the sector as well as `by_rank`; assert a version-1 file still loads with
zero sectors. Then a two-run Be `cc4` test asserting the restart's **first**
residual RMS is at converged magnitude (which it cannot be while a sector is
zero) — that is the assertion that would have caught this.

**Why first:** it is ~S, it is a correctness issue in shipped code, and every
later step inherits the format.

---

## C1 — the sidecar is unvalidated against the live reference (~S)

`load_cc_amplitudes` deliberately does not validate against a live reference —
the header says so, and defers to `seed_arbitrary_order_amplitudes`'s per-rank
dim check. That check catches a wrong *shape*, but the metadata that would catch
a wrong *meaning* is written and then never read:

- `meta.basis_name` — a sidecar from a different basis can have identical dims
  (e.g. two bases with the same occ/virt counts) and be silently applied.
- `meta.method` — read only for a log message (`ccsdtq.cpp:80`).
- `meta.n_occ` / `meta.n_virt` — written from
  `state.reference.orbital_partition`, never compared back.

**The work:** at the X3 read site, compare `meta.basis_name` against
`calculator._basis._basis_name` and `meta.n_occ`/`n_virt` against the live
partition. On mismatch: warn, ignore the sidecar, cold-start. Not an error — the
whole feature is an optimization.

Also worth recording in the metadata and checking: whether the amplitudes came
from a **frozen-core** solve, and the SCF energy they were converged against.
A sidecar from a different SCF solution on the same geometry/basis is
dimensionally valid and physically wrong; the SCF energy is a one-`double`
fingerprint that catches it.

*Gate:* a unit test that hand-writes a sidecar with a mismatched basis name and
asserts the run cold-starts with a warning. Cheap, and it pins the degradation
contract.

**Priority note:** C1 is a robustness gap, not a live defect — a user has to
actively point a run at a stale sidecar to hit it. But the failure mode is a
*wrong-basin silent seed*, which is the expensive kind.

---

## C2 — sector-aware cross-rank restart (~S, given C0)

X4.1 (`ccsdt_gen` → `cc4`) works via the rank-3 arbitrary emit
(`PLANCK_CC_ARBITRARY_LOWER_RANKS`, landed). With C0 landed, cross-rank restart
also needs to handle the case where the *source* has fewer sectors than the
*target*: a rank-3 sidecar has no rank-4 sectors, and rank 3 carries only one
independent sector anyway. The seed applies what it has; W6's in-memory
recursion fills the rest. This should already fall out of C0's
warn-and-skip policy — **verify rather than build**.

*Gate:* a `ccsdt_gen` → `cc4` two-run test on Be: same FCI energy, fewer
T4-dominated iterations than cold `cc4`. This is X4.1's gate; it may already
pass, in which case record it and close X4.

---

## C3 — hand-written ccsd/ccsdt participation (~M, unchanged from X5)

Still the real barrier, and still **layout, not plumbing**. Restated only to
confirm it is unchanged:

- Hand-written `run_rccsd` / `run_rccsdt` iterate in **spin-orbital** form
  (`ccsd.cpp`: `so.n_occ = 2·reference.n_occ`); the sidecar and seed hook are
  **spatial RCC**. A byte copy produces wrong amplitudes.
- The determinant-space backend does not hold dense `t1`/`t2` at all, so the
  write must be gated on the tensor backend.

So C3 = X5.0 (a spin-orbital→spatial projection, per rank, unit-gated by a
spatial→so→spatial identity round-trip) + X5.1 (write from the dense backends
only) + X5.2 (free — the generated side already reads spatial).

**Recommendation: defer C3.** It is the only ~M item here and it buys a
`ccsd`→`ccsdt` restart. The generated path (`ccsdt_gen`) already reaches rank 3
through the arbitrary runtime and writes a spatial sidecar, which covers the
cross-rank case C2 gates. C3 is worth doing when the hand-written solvers are
still the production route — and per
`ccgen_generated_kernels_to_production`, the intent is that they are **not**,
long-term. Doing C3 is investing in the path being retired.

---

## C4 — UCC sidecar (~S, only once UCC lands)

If arbitrary-order UCC lands (`CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md`), its
amplitudes are stored in the **same** `(rank, tag)` sector machinery. So C0's
sector-aware format covers UCC with no format change — only the metadata needs a
reference-type field (`rhf` vs `uhf`) so an RCC sidecar cannot be seeded into a
UCC run or vice versa. Fold that field into C0's version-2 header now (one
`u8`), even before UCC exists, to avoid a version 3.

`ponytail:` one spare byte in the header beats a second version bump.

---

## Recommended order

1. **C0** — fix the sector drop, bump to version 2, include C4's reference-type
   byte while the header is open. ~S, correctness.
2. **C1** — validate basis / dims / SCF-energy fingerprint at the read site. ~S,
   robustness.
3. **C2** — verify the `ccsdt_gen`→`cc4` cross-rank restart gate passes; close X4.
   ~S, likely already works.
4. **C3** — defer. Revisit only if the hand-written solvers stay production.

---

## What NOT to do

- **Do not put amplitudes in `.hfchk`.** Unchanged from the original scope:
  `O(o^n v^n)` payload would bloat every run, and couples SCF restart to CC.
- **Do not fail a run on a missing, stale, or corrupt sidecar.** Restart is an
  optimization. The landed X3 fall-through is correct; every new check added by
  C0/C1 must degrade the same way.
- **Do not add a per-rank reader.** The format is rank-generic; C0 keeps it
  sector-generic with the same one tensor codec.
- **Do not skip the version-1 compatibility branch in C0.** It is a few lines
  and it is the entire justification for having a version field.
