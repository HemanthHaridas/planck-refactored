---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-06-04

This is the canonical open-work document for the repository.
Use it with `vault/Status/Completion.md`. Older status snapshots and handoff
notes may still exist for design history, but they are no longer the source of
truth for what remains.

## Highest-priority correctness and robustness work

- Replace the large thread-local Rys scratch allocation with a size-aware
  heap or lighter scratch strategy. `_rys_sum_buf` in `src/integrals/rys.cpp`
  is a thread-local `double[2·MAX_L+1]^6 = [13]^6 = 38.5 MB` sized off the
  global `MAX_L=6` (H shells). Allocation semantics (g++-15 / emulated TLS on
  this build): the buffer is **not** allocated at engine selection; only a
  small `___emutls_v.` control descriptor sits in `__DATA`, and
  `__emutls_get_address` `calloc`s the full 38.5 MB lazily the first time each
  thread *accesses* the symbol — i.e. the first time that thread runs the Rys
  primitive kernel. So `engine os` / `engine hgp` never pay it. But under
  `engine auto` every basis has s-shells → (ss|ss) quartets always reach Rys,
  so every multithreaded SCF worker thread does `calloc(38.5 MB)` once and
  reuses it; emutls allocates the whole declared size (no lazy page-commit
  rescue). So the per-Rys-thread cost is real for both `auto` (common) and
  explicit `rys`. The reachable angular momentum is far lower than the
  declared size, and is bounded by **which quartets actually reach the Rys
  buffer**, not by the basis:
  - Auto dispatch (`_auto_prefers_rys`, `rys.cpp`) sends a quartet to Rys only
    when `L_AB + L_CD <= 1` — i.e. (ss|ss)/(ss|sp)/(sp|ss); everything else
    goes to HGP. So under Auto the per-axis index never exceeds 1.
  - Explicit `engine rys` routes every quartet through Rys, but the highest-L
    basis used with explicit Rys in the suite is cc-pVDZ (**F, L=3**). cc-pVTZ
    (which has g) is only in the auto-dispatch benchmark, where g quartets go
    to HGP, never to the Rys buffer.

  So even the explicit-rys worst case (F+F, per-axis index `2·3 = 6`) needs
  only a `7^6 = 0.94 MB` slice, and the common `auto` path needs `~[2]^6 ≈ 64`
  doubles. `MAX_L` is global (8 files) so it must not move; the fix is local to
  `rys.cpp`.

  **Design: mirror the HGP `EriScratch` model** (`src/integrals/hgp.cpp`),
  which already solves exactly this — a thread-local struct of
  `std::vector<double>` with `resize_for_quartet(lAB*, lCD*, …)` and flat
  `spatial_index()` accessors, reused across quartets and only reallocated when
  the dimension actually changes (`if (vrr.size() != needed) resize()`). Replace
  the fixed `[13]^6` `_rys_sum_buf` with a `RysScratch` struct sized per
  quartet, and rewrite `_rys_hrr_ab` to take the flat buffer + strides instead
  of the `double[VRR_DIM]^6` array parameter. This is strictly better than a
  fixed `[7]^6` bound: no L ceiling (explicit `engine rys` keeps working at g/h,
  just allocates more), no rejection guard, and the `auto` path allocates ~KB
  not MB. Under `auto` the dimension is constant (`L_AB+L_CD≤1`) so it resizes
  once per thread then pure-reuses — no hot-path allocation churn, same as HGP.
  Bitwise-gated by `planck-compute-2e` + `planck-hgp-engine-smoke`; spot-check
  that `auto` allocates ~KB/thread and that explicit rys on cc-pVTZ (g) now
  succeeds instead of relying on the oversized fixed buffer.
- Resolve the ROHF MO-energy bookkeeping inconsistency between effective, alpha, and beta eigenvalue sets

## Verification and regression gaps

- Strengthen the end-to-end spherical full-symmetry direct-SCF regression ladder beyond the current focused infrastructure tests and committed NH3/CH4 ladder
- Add durable regression coverage for remaining full-symmetry edge cases called out in the design notes:
  D3h, Oh, linear-group interplay, and lone-atom behavior
- Revalidate the CASSCF/PySCF gate suite after future optimizer work; the current tree matches the documented state, but the 11/11 suite was not freshly rerun during the May 25 consolidation review
- Keep documentation comments aligned with the implemented spherical symmetry representation; stale comments have already drifted once

## Spherical-basis work still intentionally guarded off

- Spherical analytic gradients (and therefore geomopt / freq) for ROHF and
  for the post-HF correlated paths (RMP2 / UMP2). RHF/UHF spherical gradients,
  geomopt, and frequencies are landed; ROHF gradients remain unimplemented
  Cartesian-side too, and MP2 gradients still need the response-machinery
  audit before the same lift adapter (Phase 1) can be wired through
  `compute_rmp2_gradient` / `compute_ump2_gradient`. Boundary markers:
  `water_rmp2_spherical_{gradient,geomopt}_rejected`.
- Spherical PCM
- Spherical DFT and TDDFT
- Any additional spherical workflows not already covered by the landed
  single-point, RHF/UHF-gradient, and RHF/UHF-geomopt-and-freq allow-list

## Symmetry follow-up

- Conventional-path symmetry-unique ERI storage remains out of scope; current full-group reduction is a direct-SCF feature
- ROHF is still outside the full-symmetry direct-SCF implementation scope
- The full-symmetry performance story still has room to improve even after the persisted-skeleton and monomial-operator wins; the remaining major option is a true memory-direct contraction that avoids materializing the dense `nb^4` buffer

## DFT and response-method gaps

- Double-hybrid functionals remain single-point only; analytic gradients,
  geometry optimization, frequencies, and TDDFT are still unimplemented there
- For range-separated functionals, `ImaginaryFollow` and `LinearResponse`
  (TDDFT) remain gated / unvalidated even though gradient-driven workflows are
  now landed
- Analytic Hessian remains unimplemented; frequencies are currently semi-numerical
- DFT imaginary-mode following is not implemented
- DFT grid-layer loops (`evaluate_density_on_grid` and the `xc_grid.cpp`
  density/XC evaluation) are still serial. Parallelizing them is the residual
  DFT load-imbalance win (~12% idle after the J/K-build parallelization), but it
  means adding the first parallel region to the grid layer, which is exactly
  where the historical XC-reduction jitter lived (see DFT XC Reduction
  Determinism) — any reduction there must use fixed thread-index order, not
  `omp critical`. Deferred for that reason.
- Coarse/low-quality DFT grids can still show noticeable orientation sensitivity
  under symmetry reorientation; the validated symmetry-on gradient regression is
  intentionally pinned to `grid ultrafine`

## SCF, post-HF, and workflow gaps

- ROHF post-HF: FCI, CASSCF, and RASSCF now accept ROHF references; RMP2/UMP2
  and the coupled-cluster paths remain RHF/UHF only for ROHF inputs
- ROHF CASSCF/RASSCF only support a closed, doubly-occupied inactive core; a
  spin-polarized open inactive core (distinct alpha/beta core orbitals, with the
  unrestricted core Fock, core energy, and response-block changes it implies)
  is out of scope and stays rejected by the parity guard
- ROHF analytic gradients, stability analysis, and PCM remain incomplete
- The ccgen `TensorOptimized` RCCSDT backend is still treated in-tree as an experimental / phase-4 path
- The triplet UHF state-selection / convergence gap noted during the UMP2 gradient check is still open as an SCF issue
- The isolated-small-atom SAD false-convergence issue surfaced by the BSSE work is still open; the counterpoise driver currently works around it by forcing HCore

## BSSE follow-up

- DFT ghost / counterpoise support
- N-body counterpoise beyond two fragments
- Counterpoise-corrected gradients and geometry optimization
- Post-HF ghost-reference verification beyond the current SCF-level validated scope

## CASSCF

### Remaining work

#### P2: Optimizer simplification pass

`numeric-newton` is still a production escape hatch for spaces with `<= 64`
orbital-rotation pairs. The shared-kappa state-averaged solve is now mature
enough that this path should likely be demoted or removed from normal
production flow.

Deliverables:

- Demote `numeric-newton` to debug-only behind `mcscf_debug_numeric_newton`
- Remove per-root candidates and pair probes from the stagnation family
- Keep `sa-diag-fallback` as the sole explicit fallback path
- Make every transcript step label uniquely identify the path taken

Gate:

- All 11 PySCF reference cases continue to pass
- Stagnation logging becomes simpler and easier to audit

### Future hardening

- Plateau-escape convergence path (`casscf.cpp`, the `Treating the stationary
  orbital plateau as converged` branch) is **correct and load-bearing**, not a
  hack to retire. It is the only exit for a genuinely converged
  state-averaged solution: at an SA stationary point the gating quantity
  `sa_g = Σ_I w_I g_I` goes to ~1e-10 while the per-root screens
  (`root_screen_g` / `max_root_g`) plateau at an O(1e-2) nonzero value, because
  state-averaging makes only the *weighted* gradient stationary, not each
  individual root. With `mcscf_accept_uphill` the per-root convergence screen
  then never passes, so the plateau branch is the correct way to recognize "SA
  gradient converged, energy and step flat → done." This is exercised by
  `water_casscf_sa2_sto3g_sad_guess_uphill` (the only one of the four SA-2
  cases that uses it; the other three converge through the normal gate at
  `sa_g < 1e-5`).
- Narrow hardening worth doing (NOT a correctness fix): replace the literal
  `reported_gnorm < 100·tol_mcscf_grad` screen in the plateau branch with an
  explicit `sa_g`-stationarity assertion (the uphill case already satisfies it
  at ~1e-10, so this only tightens against a future regression where the branch
  could fire while `sa_g` is not actually small), and add a
  `casscf_converged_via_plateau` diagnostic the runner asserts is `false` for
  the three normal SA-2 cases and `true` only for the SAD-uphill case. Keep the
  uphill SA-2 case green as the acceptance gate.
- Keep the two water SA-2 SAD-start regressions, because they intentionally protect two distinct optimizer policies

## Performance and maintenance opportunities

- The DFT Coulomb/exchange (`build_coulomb_from_eri` / `build_exchange_from_eri`)
  contractions are now parallel and verified thread-count-invariant (see the
  Integral Engine note). The remaining DFT parallelization target is the grid
  layer, tracked above under the DFT gaps.
- Rework shell-pair construction to operate at shell granularity rather than per Cartesian AO component
- Eliminate remaining reversed-shell-pair reconstruction churn in gradient paths outside the already-fixed RHF path
- Deduplicate the full-group AO-transform machinery that still exists in both `group_operations.cpp` and `mo_symmetry.cpp`
- Refactor `Calculator` only where it buys real safety or clarity: the leading candidates are grouping the loose MP2/UMP2 result cache and introducing a geometry-derived working-state object with a single invalidation point
