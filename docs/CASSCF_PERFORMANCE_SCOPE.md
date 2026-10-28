# Why is CASSCF ~50x slower than PySCF, and what makes it fast?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This is an in-flight scope; when the work lands,
rewrite it as the answer to the question in the title.

Prior answer this builds on: `docs/CASSCF_FOCK_REBUILD_COST.md` (direct and
cached Fock builds were tried and are dead; the redundancy is in how often the
candidate search re-enters `evaluate`, not in how each Fock is built).

## Baseline (measured 2026-10-28)

Water CAS(4,4), Cartesian, C2v, geometry and active space of the
`water_cas44_*` PySCF gates. 4 threads, `build-full` (OpenMP on). PySCF 2.13.0
from `tests/pyscf/.venv`, `mcscf.CASSCF(mf, 4, 4)`, `symmetry=True`.

| basis | AOs | Planck | macros | PySCF | macros (JK builds) | ratio |
|---|---|---|---|---|---|---|
| 6-31G | 13 | 0.85 s | 13 | 0.29 s | 10 (148) | 2.9x |
| cc-pVDZ | 25 | 12.7 s | 16 | 0.26 s | 9 (161) | **49x** |
| cc-pVTZ | 65 | ~176 s **per macro** (not run to convergence) | | 1.00 s | 9 (153) | **~1600x per macro** |

Energies agree (cc-pVDZ -76.0440109051 vs -76.0440109018). Per macro the
ratio is 27x at cc-pVDZ and ~1600x at cc-pVTZ, so this is a scaling defect,
not a constant. The cc-pVTZ figure is from the first two macros of a run that
would take roughly 45 min to converge.

### Profile (cc-pVDZ, 1 thread, `sample`, 24 792 self-time samples)

| function | self time |
|---|---|
| `Correlation::transform_eri_active_cache` | **87.3 %** |
| `ObaraSaika::_compute_fock_rhf` | 8.3 % |
| `Correlation::transform_eri` | 3.9 % |
| everything else (CI, RDMs, Q, diagonalization) | < 1 % |

CI is irrelevant at this active space. Every second of the gap is in
AO-integral contractions inside the orbital optimizer.

## Where the work comes from

Two multipliers stack.

**1. Each `(p u|v w)` transform is O(nb^5 n_act), not O(nb^4 n_act).**
`transform_eri_active_cache` (`src/post_hf/integrals.cpp`) loops over the
general index `p` *outside* the first two-leg contraction, so each `p`
re-reads the whole AO tensor: `nb * (nb^4 n_act)`. Transforming the three
active legs once and the general leg last is the standard O(nb^4 n_act). The
missing factor is ~`nb`: 25x at cc-pVDZ, 65x at cc-pVTZ, on 87 % of runtime.

**2. The optimizer calls it hundreds of times per macro.** The transform runs
in every `evaluate` (`casscf.cpp`, via `build_active_integral_cache`) and in
every `fixed_ci_orbital_gradient` (`orbital.cpp`). At cc-pVDZ there are 50
symmetry-allowed rotation pairs (138 in C1). Per macro:

| source | gradient rebuilds per macro | where |
|---|---|---|
| dense orbital Hessian, built when `npairs <= 128` | 2 x npairs = 100 | `response.cpp` `build_orbital_linear_operator`, via `delta_g_sa_action` (central FD) |
| `numeric-newton` candidate, built when `npairs <= 256` | 2 x npairs = 100 full `evaluate`s, each with a CI solve | `casscf.cpp` `build_numeric_newton_step` |
| line-search scales x candidates | up to ~14 `evaluate`s | `casscf.cpp` candidate loop |

Each rebuild costs one transform plus two `nb^4` Fock contractions. That is
>= 200 transforms per macro x 16 macros, against PySCF's ~10 JK builds per
macro. Note that line 891 builds `numeric-newton` on **every** macro whenever
`npairs <= 256`, although the surrounding comments describe it as a
stagnation-only fallback. It won 6 of 16 macros on cc-pVDZ.

**3. Planck takes ~1.8x more macros** (16 vs 9 at cc-pVDZ). The step model is
poor: macro 13 predicted dE = +3.5e-2 and got -4.7e-4. This is the smallest
multiplier and the hardest to fix, so it comes last.

## Plan

Each step is gated on all 11 PySCF CASSCF gates, the two ROHF gates, the RI and
spherical CASSCF cases, and the four SA-2 cases. The basin canary
`water_casscf_sa2_sto3g_sad_guess_uphill` matters most: the A2 detour in
`CASSCF_FOCK_REBUILD_COST.md` showed `numeric-newton`'s `/2eps` amplifying a
1e-15 reorder into a basin flip. The 2026-10-23 selector noise floor should
absorb that now, but each step must show it does.

### P0: measurement harness (DONE 2026-10-28)

`PLANCK_CASSCF_CALL_PROBE=1` prints per-macro counts (`CallProbe` in
`orbital.h`; temporary, delete when this scope is answered). PySCF side:
`tests/benchmarks/casscf/pyscf_casscf_timing.py [--c1] [basis ...]`. 4 threads.

| case | npairs | Planck | macros | PySCF | macros (JK) | ratio |
|---|---|---|---|---|---|---|
| 6-31G | 22 | 1.3-1.9 s (run-to-run) | 13 | 0.34 s | 10 (148) | 4-6x |
| cc-pVDZ | 50 | 13.4 s | 16 | 0.27 s | 9 (161) | 50x |
| cc-pVDZ C1 | 138 | 44.5 s | 27 | 0.43 s | 17 (295) | **103x** |

C1 lands in a lower solution than C2v for both codes (Planck -76.0781256198,
PySCF -76.0781256214). Per macro:

| case | `evaluate` | FD gradient rebuilds | `(pu\|vw)` transforms | Fock | accepted `numeric-newton` |
|---|---|---|---|---|---|
| 6-31G | 73 | 74 | 147 | 293 | **11 / 13** |
| cc-pVDZ | 129 | 128 | 257 | 514 | 6 / 16 |
| cc-pVDZ C1 | 306 | 122 | 428 | 857 | **25 / 27** |

What the counts confirm:

- `evaluate` per macro = 2·npairs + ~29: `numeric-newton` runs every macro.
- FD gradient rebuilds = 2·npairs + ~28 under symmetry (the dense Hessian).
  In C1 (138 > 128) the dense build is skipped and the matrix-free solve uses
  ~122 instead. Either way, about one FD rebuild per pair per macro.
- transforms = `evaluate` + rebuilds; Fock = 2 x transforms. Every row of the
  call table above is now measured, not read off the code.

**What changes the plan: `numeric-newton` is the optimizer, not a fallback.**
It wins 11/13 macros (6-31G) and 25/27 (C1), against `sa-coupled` 2 and 1.
The "primary" coupled step rarely produces the best candidate, which fits the
wrong-sign predicted dE. So P3 cannot demote `numeric-newton` until P2 has
made the coupled step good enough to win; P2 is about step quality as much
as cost.

### P1: fix the transform's complexity (DONE 2026-10-28)

`transform_eri_active_cache` is now the generic quarter-transform with the
legs reordered: `(pu|vw) = (vw|up)` under the AO tensor's 8-fold symmetry, so
`transform_eri(eri, nb, C_act, C_act, C_act, C)` contracts the three active
legs first (O(nb^4 n_act)) and a permutation restores `(p,u,v,w)`. The
hand-rolled O(nb^5 n_act) routine and its thread-local scratch are deleted
(net -116 lines). `transform_eri_internal` already used the generic routine
with active legs only, so it never had the defect.

Gate: `planck-casscf-internal` compares against `transform_eri(C, C_act,
C_act, C_act)` (an independent leg order) on a 6-AO, 3-active fixture. The old
fixture was a tensor with **no** ERI symmetry, which no real integral tensor
has and which the new routine correctly rejects. It is now 8-fold symmetric,
non-square and dense; swapping `u` and `w` in the permutation fails it.
Energies are unchanged to every printed digit with identical macro counts; all
15 CASSCF/RASSCF/FCI regression cases (uphill SA-2 canary included) and the 11
PySCF-table inputs pass, the latter within 2e-9 Eh of their recorded Planck
values.

| case | before | after | speedup |
|---|---|---|---|
| 6-31G | 1.3-1.9 s | 0.66 s | 2-3x |
| cc-pVDZ | 13.4 s | 3.5 s | 3.8x |
| cc-pVDZ C1 | 44.5 s | 11.1 s | 4.0x |

PySCF is still 13x (cc-pVDZ) and 26x (C1) faster. The re-profile (cc-pVDZ, 1
thread) moved from 24 792 to 7 026 samples: generic `transform_eri` 57 %,
`_compute_fock_rhf` 40 %. Both scale with the rebuild count, so the
multiplicity (P2/P3) is now the lever, as predicted.

Follow-ups, not done:

- `transform_eri`'s first quarter has `mu` innermost at stride `nb^3`, which
  is cache-hostile. Making it GEMM-shaped would speed this and MP2/CC; its own
  gate is `planck-transform-eri`. Worth less than P2, which removes calls.
- The RI twin `transform_eri_active_cache_ri` is `transform_eri_ri(C, C_act,
  C_act, C_act)`, full leg first. `water_casscf_ri_ccpvdz` takes 14 s against
  3.6 s for the tensor cc-pVDZ case. Measure before assuming the same defect;
  the RI cost model differs.

### P2: analytic orbital Hessian-vector product

Replace `delta_g_sa_action`'s central FD with the analytic fixed-CI orbital
Hessian action (PySCF `mc1step.gen_g_hop`): built from `F_I`, `F_A`, the
1-/2-RDMs, the cached `(p u|v w)`, and **one** JK build on a trial-rotated
density per product. This removes the two full gradient rebuilds per product,
so the dense-Hessian build drops from 100 rebuilds to ~50 JK.

- Gate: analytic vs current FD action on random `R` to FD precision, on RHF,
  ROHF and SA-2 references and in C1. Then the suite.
- Keep FD as a debug oracle (`mcscf_debug_fd_hessian`), not a second
  production path.
- Open question: does PySCF's `h_op` include the orbital-CI coupling block
  the shared-kappa SA solve handles separately in `response.cpp`? Map the
  blocks before writing code.

### P3: is `numeric-newton` still needed?

`numeric-newton` is an FD of the fully CI-relaxed gradient, i.e. the exact
Hessian including CI relaxation, costing 2 x npairs CI solves. Once P2 gives an
analytic orbital Hessian and the coupled solve supplies the CI response, it may
be redundant.

- Measure, don't assume: rerun the accepted-candidate sweep that kept
  `numeric-newton` (Completion, "Stagnation-cascade trim") after P2. If it
  still wins, restrict it to the stagnation branch its comments already
  describe rather than every macro.
- Remember it is load-bearing today (dominant accepted fallback; part of the
  uphill canary's path), so demoting it is a measured decision, not a cleanup.

### P4: macro count (research)

Why is the predicted dE often the wrong sign? Candidates: the level-shift
schedule, the trust cap `mcscf_max_rot`, or a model inconsistency between
what `sa-coupled` predicts and what the candidate selector scores. Compare
macro-by-macro against PySCF's 1-step log at cc-pVDZ. Only start once P1-P3
have made a macro cheap, because then the macro count is the remaining
multiplier.

## Out of scope

- CI cost. Irrelevant at CAS(4,4); the large-CAS sigma build is already done
  (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`).
- Direct or cached Fock builds (disproven, see the prior doc).
- RI. It removes the `nb^4` tensor (memory), and exists (`mp2_use_ri`). Once
  P1/P2 land, recheck whether the RI path has the same loop-shape defect.

## Stop conditions

- If P1's re-profile shows the remaining time is not in rebuild multiplicity,
  re-rank P2/P3 from the new profile instead of following this order.
- If any step moves a gated energy by more than 1e-8 Eh or flips the uphill
  canary's basin, stop and diagnose before continuing.
