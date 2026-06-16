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
- Extract a shared `SpatialQuartetLayout` (6-axis dims + strides +
  `spatial_index` + `resize_for_quartet`) and retrofit the OS, HGP, and Rys
  per-quartet scratch onto it. All three now carry near-duplicate per-quartet
  scratch structs — OS's `_eri_scratch`, HGP's `g_hgp_scratch`, and (as of
  PR #126) Rys's `RysScratch` — so the three concrete call sites exist to shape
  the shared interface. Only the spatial-layout core is common; the Boys `m`
  axis, HGP's `a0c0_accum`, OS's no-zero-init policy, and the differing
  accessors stay engine-specific. Bitwise-gate across all three engines
  (`planck-compute-2e`, `planck-hgp-engine-smoke`, plus the OS path via the
  existing ERI gates).
- Refactor `Calculator` only where it buys real safety or clarity: the leading candidates are grouping the loose MP2/UMP2 result cache and introducing a geometry-derived working-state object with a single invalidation point
