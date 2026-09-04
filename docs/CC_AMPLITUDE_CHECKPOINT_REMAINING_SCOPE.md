# `.ccamp` dumping and reading — what remains

Follow-on scope to `CC_AMPLITUDE_CHECKPOINT_SCOPE.md`, which specified X0–X5.
X0–X4.1 are landed. This file's own C0 and C1 are now also landed (PR #164,
branch `ccamp-x5-spin-orbital-projection`, commits `c83b4bd7` / `d5d68314` /
`c610fe12`). C2, C3, C4 remain open; this revision sharpens each with the
detail needed to pick it up without re-deriving context.

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
| X4.1 `ccsdt_gen`→`cc4` write | Same write site as X2, rank-generic (comment at `rccgen.cpp:420-421` names this explicitly) | landed, **but see C2 — never actually run end to end** |
| X5.0 spin-orbital→spatial projection | `project_rccsd_amplitudes_to_spatial`, `amplitudes.{h,cpp}:474-...`, gated by `tests/cc_spatial_amplitude_projection.cpp` | landed |
| X5.1 hand-written `run_rccsd` write | `ccsd.cpp:648-...`, same gating/degradation policy as X2 | landed, verified end-to-end on BH3/STO-3G |
| C0 sector round-trip (format) | `cc_amplitude_checkpoint.{h,cpp}`, `CCAMP_VERSION = 2`, shared `write_tensor`/`read_tensor` | landed, mutation-verified |
| C0 sector application (seed) | `seed_arbitrary_order_amplitudes`, `generated_arbitrary_prepare.cpp` | landed, mutation-verified |
| C4's reference-type byte | Folded into C0's version-2 header (`CCReferenceType`, `cc_amplitude_checkpoint.h`) — **only the byte is landed, not UCC wiring; see C4 below** | partial |
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

## C2 — sector-aware cross-rank restart: STILL UNVERIFIED, now that C0 is done

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

**What to check first, before writing new code — this may already be correct
by construction:**

1. `ensure_amplitude_sectors(*state_res, *kernels_res)` runs at
   `rccgen.cpp:273`, before `try_restart_from_sidecar` at `:279` — so
   `state.amplitudes.sectors` is populated with zero-valued tensors at the
   right `(rank, tag)` keys before the seed ever runs, for a rank-4 target.
2. A rank-3 sidecar's `chk->amplitudes.sectors` is empty (rank 3 has no
   sectors — CCSDTQ is the first rank with an independent second sector; see
   `amplitudes.h:87-92`), so `seed_arbitrary_order_amplitudes`'s
   `for (const auto &[seed_key, seed_tensor] : seed.sectors)` loop
   (`generated_arbitrary_prepare.cpp`, C0's fix) simply does nothing — no
   live sector is touched, none is expected to be.
3. This reasoning was NOT verified against a real run this session. The unit
   tests built for C0 (`tests/cc_amplitude_sector_seed.cpp`) construct the
   scenario synthetically; they do not exercise the real
   `ensure_amplitude_sectors` → `try_restart_from_sidecar` ordering inside
   `solve_generated_rcc`, and they do not use a real rank-3-sourced,
   rank-4-target pair.

**Gate** (per the original scope, unchanged): a `ccsdt_gen` → `cc4` two-run
test on Be — write via `ccsdt_gen` (rank 3), restart via `cc4`
(`PLANCK_CC_ARBITRARY_LOWER_RANKS=ON` required), assert the same FCI energy
and fewer T4-dominated iterations than a cold `cc4` run. This is X4.1's own
gate and closes X4 formally once it passes. Given the build-cost lesson from
this session (a `cc4`/rank-4 solve on Be is the expensive case flagged
repeatedly in project memory), size the gate at loose tolerance first, the
same way the original X0–X3 validation did (`tol_energy 1e-4`,
`tol_density 1e-3`), before attempting a tight-tolerance version.

**If it passes as-is**: record the result here and in
`CC_AMPLITUDE_CHECKPOINT_SCOPE.md`, mark X4 closed, done. **If it does not
pass**: the most likely failure mode, given the code above, is an ordering or
dims mismatch between what `ensure_amplitude_sectors` allocates for the
*target* kernel bundle and what a rank-3 *source* sidecar's `by_rank` caps to
via the `chk->amplitudes.by_rank.resize(rank)` line at
`rccgen.cpp:232-233` (unchanged by C0/C1) — that resize only touches
`by_rank`, never `sectors`, which is fine for a rank-3→rank-4 seed (the
source genuinely has none) but worth re-checking for a rank-4→rank-5+ seed
where the source DOES carry sectors that could outnumber or mis-key against
the target's.

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

## C4 — UCC sidecar: the header byte is landed, nothing else is

**Only the metadata field landed, not any UCC wiring — worth stating plainly
since "C4 done" would be the wrong read of the current tree.** C0 added
`CCReferenceType` (`RHF = 0, UHF = 1`) to `CCAmplitudeCheckpointMeta` and the
version-2 header writes/reads it. Every current write site
(`rccgen.cpp`'s X2/X4.1, `ccsd.cpp`'s X5.1) leaves it at the default `RHF`,
because no UCC write path exists yet — `save_cc_amplitudes` has never been
called with `reference_type = UHF` in this tree.

**What actually remains, once arbitrary-order UCC lands**
(tracked separately in `CCGEN_UNRESTRICTED_CC.md`):

1. A UCC write site must set `meta.reference_type = CCReferenceType::UHF`
   explicitly — nothing does this automatically today.
2. `try_restart_from_sidecar` (or its UCC-path sibling, once one exists) must
   **check** `chk->meta.reference_type` against the live reference kind and
   reject a mismatch the same way C1 rejects a basis/dims mismatch — this
   check does **not exist yet**. C1's basis/dims validation added this
   session says nothing about `reference_type`; a UHF sidecar seeding an RCC
   run (or vice versa) would currently pass C1's checks (if `n_occ`/`n_virt`
   happen to agree) and only fail, if at all, deeper in the seed hook's dims
   check — which is exactly the "wrong-basin silent seed" failure mode C1
   was built to close, reopened for this one axis.
3. UCC's `sectors` machinery is confirmed to reuse the same `(rank, tag)`
   keying (per the original C4 note and confirmed unchanged by this
   session's read of `amplitudes.h`), so C0's sector format needs no further
   change — only the write/read *call sites* are UCC's remaining work, not
   the file format.

---

## Recommended order, revised

1. ~~C0~~ — **DONE.**
2. ~~C1~~ — **DONE.**
3. **C2** — verify the `ccsdt_gen`→`cc4` cross-rank restart gate on Be at
   loose tolerance; likely correct by construction (see the reasoning above)
   but genuinely unverified. Closes X4 formally. ~S.
4. **C3 (ccsdt half only)** — still recommended deferred; the rank-2 half is
   done. Revisit only if `run_rccsdt`'s hand-written backends stay production
   longer than `ccsdt_gen`'s adoption. ~M if picked up.
5. **C4 (UCC wiring)** — blocked on arbitrary-order UCC landing at all; the
   header byte is ready, the write-site and read-site work is not started.
   ~S once UCC exists.

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
