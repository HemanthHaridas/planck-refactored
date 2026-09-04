# `.ccamp` dumping and reading — what remains

Follow-on scope to `CC_AMPLITUDE_CHECKPOINT_SCOPE.md`, which specified X0–X5.
X0–X4.1 are landed. This file's own C0, C1, and C2 are now also landed and
verified (PR #164, branch `ccamp-x5-spin-orbital-projection`). C3 and C4
remain open; this revision sharpens each with the detail needed to pick it up
without re-deriving context.

Grounded in the current tree.

---

## Landed (verified in-tree)

| Step | Where | State |
|---|---|---|
| X0 format + writer | `src/post_hf/cc/cc_amplitude_checkpoint.{h,cpp}` | landed |
| X1 loader | same, `load_cc_amplitudes` | landed; errors (never crashes) on bad magic, bad version, truncation, negative dims, count/dims disagreement, overflow |
| X2 write on success | `rccgen.cpp:415-445` (`run_rccgen`), gated on `_save_checkpoint && !_checkpoint_path.empty()`, path = `<stem>.ccamp`, write failure is a **warning** | landed |
| X3 read + seed | `rccgen.cpp:203-250` (`try_restart_from_sidecar`), `load_cc_amplitudes` → `seed_arbitrary_order_amplitudes`, falls through to cold/W6 on any error | landed |
| X4.0 rank-3 arbitrary emit | `PLANCK_CC_ARBITRARY_LOWER_RANKS` lowers the registry floor 4→3 (`rccgen.cpp:284`) | landed |
| X4.1 `ccsdt_gen`→`cc4` write + cross-rank restart | Same write site as X2, rank-generic (comment at `rccgen.cpp:420-421` names this explicitly) | landed and verified end-to-end (C2) |
| X5.0 spin-orbital→spatial projection | `project_rccsd_amplitudes_to_spatial`, `amplitudes.{h,cpp}:474-...`, gated by `tests/cc_spatial_amplitude_projection.cpp` | landed |
| X5.1 hand-written `run_rccsd` write | `ccsd.cpp:648-...`, same gating/degradation policy as X2 | landed, verified end-to-end on BH3/STO-3G |
| C0 sector round-trip (format) | `cc_amplitude_checkpoint.{h,cpp}`, `CCAMP_VERSION = 2`, shared `write_tensor`/`read_tensor` | landed, mutation-verified |
| C0 sector application (seed) | `seed_arbitrary_order_amplitudes`, `generated_arbitrary_prepare.cpp` | landed, mutation-verified |
| C4's reference-type byte | Folded into C0's version-2 header (`CCReferenceType`, `cc_amplitude_checkpoint.h`) | landed, but **not sufficient for a UCC sidecar on its own — see C4 below**, the metadata shape itself (a single `n_occ`/`n_virt` pair) still cannot represent UHF's four occupation counts |
| C1 basis/dims validation | `try_restart_from_sidecar`, `rccgen.cpp:229-...` | landed, verified end-to-end on LiH/STO-3G (two independent corruption cases + the positive case) |

Memory records the X0–X3 validation: Be cc4 cold = 18 iterations → restart = 1
iteration at loose tolerance. C1 was validated the same way this session
(LiH/STO-3G, `ccsdt_gen`, rank 3): a corrupted `basis_name` and a corrupted
`n_occ` each correctly warn-and-cold-start at 62 iterations (matching a
from-scratch run's converged energy exactly), while an untouched sidecar
still warm-starts in 1 iteration.

**X5.0/X5.1's own note, not in the original scope table**: the test suite
here builds `assert()`-based gates, and this Release build tree compiles with
`-DNDEBUG`, which silently disables every one of them — including the new
`cc_spatial_amplitude_projection` gate on first pass, where it hid a real
fixture bug. Verify any new gate in this area with a `-UNDEBUG` build before
trusting it; `ctest` alone in this tree is not sufficient. Pre-existing,
project-wide, not fixed here.

---

## C2 — sector-aware cross-rank restart: VERIFIED end to end, X4 closed

**Status changed by C0 landing: this is no longer "should already work,"
it is "now testable and still untested."** Before C0, the sector-drop defect
meant a `ccsdt_gen`→`cc4` restart could never carry sector data regardless —
there was nothing sector-aware to verify. Now that C0's format and seed hook
both handle sectors correctly, C2's actual claim (a rank-3 sidecar, which by
construction has zero sectors of its own, seeds correctly into a rank-4 run
whose kernel bundle *does* have sectors, with the seed hook's own
`ensure_amplitude_sectors`-then-seed ordering — see `rccgen.cpp:273` then
`:279` — leaving the target's sectors at their pre-allocated zero rather than
erroring or crashing) has never been exercised by an actual run.

**Verified 2026-09-04, and the reasoning above held exactly.** Ran the actual
gate on Be/STO-3G at loose tolerance (`tol_energy 1e-4`, `tol_density 1e-3`,
`PLANCK_CC_ARBITRARY_LOWER_RANKS=ON`), with `cc_warm_start .false.` on every
run to isolate the disk-restart claim from W6's in-memory recursion (which
would otherwise mask a broken disk path — a `cc4` run with `warm_start=on`
recurses to a converged rank-3 in memory regardless of whether a sidecar
exists, so a genuinely cold rank-4 baseline needs W6 off too):

| run | iterations | E_corr | total energy |
|---|---|---|---|
| cold `cc4` (W6 off, no sidecar) | **6** | -0.0518049788 | -14.4036853795 |
| `ccsdt_gen` write (rank 3) | 6 | -0.0517514344 | -14.4035... |
| `cc4` restart from the rank-3 sidecar (W6 off) | **1** | -0.0517592490 | -14.4036396497 |

- **1 iteration vs 6** — fewer, as the gate requires.
- **`Warm-started rank 4 from CC amplitude checkpoint '...' (seeded 3 rank(s), method 'cc3')`**
  logged, confirming the disk path fired, not W6 (which was off).
- No warnings emitted during the restart — the empty-sectors-seed-nothing
  path is a silent no-op, exactly as reasoned, not an error.
- Both the cold and restart runs land within loose-tolerance spread
  (~1.5e-5 to ~4.5e-5) of the tight-tolerance reference `-14.4036551081`
  (`be_rccsdtq_sto3g`'s own committed value) — confirming both are in the
  correct basin, not a coincidentally-plausible wrong answer.

The predicted failure mode (an ordering/dims mismatch between
`ensure_amplitude_sectors`'s target allocation and the source sidecar's
`by_rank.resize(rank)` cap) did not occur. **X4 is formally closed.**

**Not added as a committed regression case.** This was run as an ad hoc
verification (three hand-built `.hfinp` files, not committed) rather than
wired into `tests/regression_cases.json` — the loose-tolerance energies
above are not stable enough across engine/compiler changes to pin as exact
`metric_close` values without re-deriving the tolerance the way the original
X0–X3 validation did. If this is worth a permanent gate, the shape to copy is
`be_rccsdtq_sto3g`'s own tight-tolerance form, extended to a three-run
(write, cold, restart) comparison with `cc_warm_start .false.` on the cold
and restart legs — the single `cc_warm_start .false.` flag is what makes the
comparison mean anything; without it the "cold" baseline is silently warm
via W6 and the iteration-count comparison is vacuous.

---

## C3 — hand-written ccsd/ccsdt participation: X5 done for ccsd, ccsdt still open

**Rescoped by this session's work.** The original C3 bundled ccsd and ccsdt
together as one ~M item and recommended deferring both. That bundling is now
wrong: **X5.0/X5.1 landed for `run_rccsd` (rank 2)**, so half of C3 is done.
What remains is specifically **`run_rccsdt` (rank 3, the hand-written path,
distinct from `ccsdt_gen`)**.

- **Landed**: `project_rccsd_amplitudes_to_spatial` (X5.0) handles rank 2
  only — its signature is `RCCSDAmplitudes → ArbitraryOrderRCCAmplitudes`,
  hardcoded to `by_rank.size() == 2`. `run_rccsd`'s write site (X5.1) is
  wired in `ccsd.cpp`.
- **Still open**: `run_rccsdt` (`ccsdt.cpp`) has zero references to
  `cc_amplitude_checkpoint.h` — confirmed by grep this session, matching the
  original scope's claim. It still iterates spin-orbital `t1`/`t2`/`t3`
  (three ranks, not two), and — per the original C3's own caveat, still
  correct — three separate backends select at runtime (determinant-space,
  tensor, tensor-optimized; see `PLANCK_RCCSDT_BACKEND`), and only the tensor
  backend holds dense amplitudes suitable for projection. The
  determinant-space and tensor-optimized backends would need to be excluded
  from the write path or have their amplitudes reconstructed first.
- **The projection function itself needs a rank-3 sibling**, not a
  generalization of the rank-2 one in place — `project_rccsd_amplitudes_to_spatial`
  is deliberately concrete (dims are hardcoded to 2 ranks with compile-time
  known shapes for `t1`/`t2`), matching this codebase's stated preference
  (see the original scope's own "one function per rank" instruction). A
  `project_rccsdt_amplitudes_to_spatial(const RCCSDTAmplitudes&)` sibling
  would follow the exact same closed-shell relation this session derived and
  verified for rank 2, extended by one more excitation level: `t3`'s spatial
  form is recoverable from one specific spin-orbital block the same way
  `t2`'s was (the opposite-spin block), though the exact index pattern for
  rank 3 was not derived or verified this session and should be checked
  numerically against a real converged `run_rccsdt` solve the same way X5.0
  was, not assumed by extrapolation.

**Recommendation, reaffirmed**: still defer the `run_rccsdt` half. The
generated path (`ccsdt_gen`) already reaches rank 3 through the arbitrary
runtime and writes a spatial sidecar (X4.1), which is the route the project's
own stated intent
(`ccgen_generated_kernels_to_production` — generated kernels are meant to
replace the hand-written solvers in production, gated on diagram dressing)
points toward long-term. The rank-2 half was worth doing now because
`run_rccsd` has no arbitrary-order generated sibling at all
(`generated_floor` is 3, per project memory — "there is no rank-2 generated
RCC kernel"), so X5.0/X5.1 fill a real gap rather than duplicate an existing
route. `run_rccsdt` does not have that gap: `ccsdt_gen` already covers it.

---

## C4 — UCC sidecar: NOT blocked on UCC landing (it already has). The real
blocker is the metadata SHAPE, and UCC's own author already said so.

**This section was wrong in the previous revision and is corrected here.**
It previously said C4 was "blocked on arbitrary-order UCC landing at all."
**That premise is false — arbitrary-order UCC is already in the tree**,
complete and validated (`ucc2`/`ucc3`/`ucc4`, behind `-DPLANCK_CC_UCC=ON`;
see `vault/Status/Completion.md` and `CCGEN_UNRESTRICTED_CC.md`). The actual
blocker was found by reading `uccgen.cpp` (the UCC entry point) directly,
where its own author already diagnosed and documented it — this section had
simply never been checked against that file:

```
// Deliberately NOT carried over from the RCC path:
//   ...
//   - .ccamp persistence. The sidecar's meta carries a single (n_occ, n_virt)
//     pair, which cannot describe a spin-resolved amplitude set. Writing one
//     would produce a file that reloads into the wrong shape.
```

**Confirmed independently by reading the two structs.**
`CCAmplitudeCheckpointMeta` (`cc_amplitude_checkpoint.h`) carries exactly one
`n_occ`/`n_virt` pair. `UHFReference` (`common.h`) carries **four**:
`n_occ_alpha`, `n_occ_beta`, `n_virt_alpha`, `n_virt_beta` — genuinely
independent counts for an open-shell or spin-polarized system, not
derivable from each other or from a single pair. **C0's `reference_type` byte
does not fix this.** It lets a reader distinguish "this file claims to be
RHF" from "this file claims to be UHF," but even a correctly-tagged UHF
sidecar has nowhere in the current format to store its real occupation
counts — `n_occ` would have to silently mean something different
(alpha-only? total? a lossy sum?) depending on `reference_type`, which is
exactly the kind of format ambiguity a version field exists to prevent, not
create.

**So C4 is not "write the UCC call sites," it is "design the metadata
extension first."** Concretely unresolved, and each is a real decision, not
mechanical:

1. **How to represent four occupation counts.** The direct fix is adding
   `n_occ_alpha`/`n_occ_beta`/`n_virt_alpha`/`n_virt_beta` alongside (or
   instead of) the existing `n_occ`/`n_virt` pair, discriminated by
   `reference_type` at read time — an RHF sidecar keeps using the existing
   two fields, a UHF sidecar uses the new four. This needs another version
   bump (to 3), since version 2's header layout has no room for the extra
   fields and no version-1-style "read what's there, default the rest"
   compatibility path is possible without knowing in advance how many extra
   u64s to expect.
2. **Whether `by_rank` means anything for UCC at all.** Confirmed this
   session: `prepare_generated_ucc_state` explicitly leaves
   `state.amplitudes.by_rank` empty (comment: "No amplitudes at all... the
   sectors are filled by `ensure_amplitude_sectors`") — UCC has no privileged
   "balanced reference sector" the way RCC's `by_rank` is. So a UCC sidecar
   is **sectors-only**; a write site would emit `max_rank` and an empty (or
   omitted) `by_rank` loop, all real data living in the sector block C0
   already built. Worth confirming the loader's `max_rank < 1` rejection
   (`load_cc_amplitudes`, unchanged since X1) doesn't need an exception for
   this — a UCC file having zero-length `by_rank` but a real `max_rank` may
   already trip that check, since `max_rank` today is read as `by_rank.size()`
   equivalent at write time (`rccgen.cpp`'s write sites set
   `meta.max_rank = rank`, independent of `by_rank`'s actual length) —
   worth re-reading `save_cc_amplitudes`/`load_cc_amplitudes` against a
   UCC-shaped input specifically before assuming this composes for free.
3. **UCC's sector tags are confirmed compatible with C0's format as-is.**
   `ucc_amplitude_blocks(rank)` produces tags like `"aaaa"`, `"abab"`,
   `"bbbb"` (rank 2) via the same `(rank, tag) -> TensorND` keying C0's
   sector block already writes/reads — this part of the original note holds
   and needs no format change.
4. **The write/read call sites themselves** (setting `reference_type`,
   validating it symmetrically to C1's basis/dims check) are the easy,
   mechanical remainder, but they should come **after** items 1–2 are
   decided, not before — writing UCC-shaped data into a format that cannot
   yet represent it correctly is worse than not writing at all, and would
   repeat the exact mistake C0 was fixing (a sidecar that looks complete but
   silently discards or misrepresents real data).

**Recommendation: still don't build this now.** It is a real design decision
(the version-3 header shape) that deserves its own pass, not a same-session
add-on the way C0/C1/C2 were. `uccgen.cpp`'s own comment is doing its job —
it is accurate, and nothing here should be built until that comment's
condition (a sidecar format that can actually hold UHF's occupation counts)
is met.

---

## Recommended order, revised

1. ~~C0~~ — **DONE.**
2. ~~C1~~ — **DONE.**
3. ~~C2~~ — **DONE. X4 formally closed.**
4. **C3 (ccsdt half only)** — still recommended deferred; the rank-2 half is
   done. Revisit only if `run_rccsdt`'s hand-written backends stay production
   longer than `ccsdt_gen`'s adoption. ~M if picked up.
5. **C4 (UCC sidecar)** — NOT blocked on UCC landing (it already has); blocked
   on a real metadata-format decision (`CCAmplitudeCheckpointMeta` cannot
   represent UHF's four occupation counts, `uccgen.cpp`'s own comment says so
   correctly). Needs a version-3 design pass before any write/read code, not
   a mechanical follow-on to C0/C1/C2. ~S once the format question is
   answered; the format question itself is not ~S.

---

## What NOT to do

- **Do not put amplitudes in `.hfchk`.** Unchanged from the original scope:
  `O(o^n v^n)` payload would bloat every run, and couples SCF restart to CC.
- **Do not fail a run on a missing, stale, or corrupt sidecar.** Restart is an
  optimization. The landed X3 fall-through is correct; C0 and C1 both degrade
  the same way (warn, cold-start) — any future check here must too.
- **Do not add a per-rank reader.** The format is rank-generic; C0 keeps it
  sector-generic with the same one tensor codec (`write_tensor`/`read_tensor`
  in `cc_amplitude_checkpoint.cpp`).
- **Do not skip the version-1 compatibility branch if the format changes
  again.** C0 already proved this is a real trap, not a hypothetical one: a
  version-2 file with an empty sector block and a version-1 file both end the
  stream at the same logical point, and the difference between "ended, that's
  fine" and "ended, that's truncation" needed a `peek()`-based check plus a
  test that truncates the `n_sectors` field itself — not just deep inside a
  sector's data — because the deep-truncation case does not exercise that
  code path at all. Any future format bump inherits this exact hazard.
- **Do not assume C2 needs new code without running the gate first.** The
  reasoning above suggests it may already be correct by construction; writing
  code before checking risks solving a problem that does not exist.
- **Do not build a full `hartree-fock` rebuild reflexively to verify a small
  change in this area.** `amplitudes.{h,cpp}`, `cc_amplitude_checkpoint.{h,cpp}`,
  and `ccsd.cpp` all compile in seconds standalone; the slow part of this
  codebase's build is the generated CC kernel bundle
  (`generated_kernel_registry.cpp`), which none of C0/C1/C2's own files touch.
  A targeted object-file compile or a small standalone test binary (see
  `tests/cc_spatial_amplitude_projection.cpp` and
  `tests/cc_amplitude_sector_seed.cpp` for the pattern) verifies correctness
  without paying that cost.
