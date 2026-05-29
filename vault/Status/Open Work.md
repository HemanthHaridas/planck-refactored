---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-05-29

This is the canonical open-work document for the repository.
Use it with `vault/Status/Completion.md`. Older status snapshots and handoff
notes may still exist for design history, but they are no longer the source of
truth for what remains.

## Highest-priority correctness and robustness work

- Remove the committed developer-specific absolute basis path from `src/base/basis.h`
- Fix the Mayer bond-order convention mismatch between closed-shell and unrestricted paths
- Replace the large thread-local Rys scratch allocation with a size-aware heap or lighter scratch strategy
- Add a real stationarity guard to the CASSCF plateau-escape convergence path
- Add warning or fallback behavior when the CASSCF orbital-action solver heavily clamps negative curvature
- Add DIIS rank/conditioning guards shared across RHF, UHF, and ROHF
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
- DFT-side Coulomb and exchange contractions still lack a stable parallel implementation
- Coarse/low-quality DFT grids can still show noticeable orientation sensitivity
  under symmetry reorientation; the validated symmetry-on gradient regression is
  intentionally pinned to `grid ultrafine`

## SCF, post-HF, and workflow gaps

- ROHF post-HF beyond FCI remains incomplete
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

- Consider adding a regression assertion that state-averaged runs do not declare convergence through the plateau-escape warning path unless that behavior is explicitly intended
- Keep the two water SA-2 SAD-start regressions, because they intentionally protect two distinct optimizer policies

## Performance and maintenance opportunities

- Parallelize or redesign the DFT Coulomb/exchange contractions without perturbing tight regression outputs
- Rework shell-pair construction to operate at shell granularity rather than per Cartesian AO component
- Eliminate remaining reversed-shell-pair reconstruction churn in gradient paths outside the already-fixed RHF path
- Deduplicate the full-group AO-transform machinery that still exists in both `group_operations.cpp` and `mo_symmetry.cpp`
- Refactor `Calculator` only where it buys real safety or clarity: the leading candidates are grouping the loose MP2/UMP2 result cache and introducing a geometry-derived working-state object with a single invalidation point
- Route MP2 / UMP2 gradient derivative-ERI calls in
  `src/post_hf/mp2_gradient.cpp` (lines 371, 525, 726) through
  `compute_eri_deriv_dispatch` instead of hardcoding
  `ObaraSaika::_compute_eri_deriv_elem`. Today the MP2 response
  intermediates always use OS even when the user picked HGP — values agree
  to ~1e-15 so this is performance/coverage, not correctness.
