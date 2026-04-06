# Unified SA-CASSCF Plan (Merged Implementation + Theory)

Last updated: 2026-04-05 (PySCF cross-check results added)

This document merges:
1. SA-CASSCF theory and equations
2. Current repository implementation status

---

## Core Problem

The SA objective is:

```
E_SA = Σ_I w_I E_I
```

Stationarity requires:

```
g_SA = Σ_I w_I g_I = 0
```

The original implementation mixed:
- state-averaged objective
- root-resolved per-root solves
- averaged orbital updates
- rootwise convergence criteria (g_I = 0 for all I)

This caused plateau behavior for nroots > 1: the code would fail to declare
convergence even when g_SA was already small, because individual roots with small
weights can have large per-root gradients without violating SA stationarity.

---

## Phase 1: Fix Convergence Semantics — DONE (c97a754, 2026-04-04)

The stopping condition is now:

```
||g_SA||_inf < tol
```

where g_SA = st.g_orb is the already-built state-averaged gradient
`Σ_I w_I g_I` (built in `build_weighted_root_orbital_gradient`).

Per-root gradient norms `weighted_root_gnorm` and `max_root_gnorm` are retained
as diagnostics and as a safety guard, but they do not gate SA convergence.

The iteration table column is now labeled "SA Grad"; the per-root screen appears
as "MaxRootG".

Stagnation detection uses the SA gradient via `sa_gradient_progress_flat`.

Validated: both 2-root SA fixtures (water_cas44_sto3g_sa2,
ethylene_cas44_sto3g_sa2) converge and all 11 benchmark cases pass.

---

## Phase 2: True SA Coupled Solve with One Shared κ — NOT DONE

### Current implementation (what exists)

`build_root_resolved_coupled_step_set` solves a per-root coupled orbital/CI step
for each root independently using `solve_coupled_orbital_ci_step`, then reduces
to a weighted average:

```
kappa_SA = Σ_I w_I kappa_I
```

Each `solve_coupled_orbital_ci_step` call minimizes:

```
R_I(kappa_I, c1_I) = (g_I + H_oo kappa_I + G_oc c1_I,
                       (H - E_I) c1_I + Q sigma_I)
```

independently for root I. The shared step is an average of independent solutions,
not a solution to the coupled SA system.

### What a true shared-κ SA solve requires

The SA stationarity system is:

```
R_SA(kappa, {c1_I}) = g_SA + H_oo kappa + Σ_I w_I G_oc c1_I(kappa) = 0
```

subject to each root's CI response equation:

```
(H - E_I) c1_I = -Q sigma_I(kappa)    for each I
```

A shared-κ solve holds one κ, solves all roots' CI responses to that shared κ,
and then evaluates the SA orbital residual. Iterating on κ with the SA residual
as the driving force is the correct structure.

### Deliverables for Phase 2

- Implement a matrix-free shared-κ SA coupled solve:
  1. Given a trial κ, solve all roots' CI response equations `(H - E_I) c1_I = -Q sigma_I(kappa)`.
  2. Form the SA orbital residual `g_SA + H_oo kappa + Σ_I w_I G_oc c1_I`.
  3. Update κ using the SA orbital residual with diagonal preconditioning.
  4. Repeat until the SA orbital residual is small.
- Replace `build_root_resolved_coupled_step_set` with the shared-κ SA path as the
  production `sa-coupled` direction.
- Fix the acceptance merit function: replace `weighted_root_gnorm` with the SA
  gradient `gnorm` in the merit comparison (line 1169–1170 in casscf.cpp).

---

## Phase 3: Orbital Hessian Action Upgrade — NOT DONE

### Current state

`hessian_action()` in `orbital.cpp:178` is diagonal:

```
(HR)_pq = (F_sum[p,p] - F_sum[q,q]) * R_pq
```

This is a one-index energy denominator approximation. It serves as a
preconditioner, not a faithful second-order OO block model.

### What the true δg_SA[R] action requires

The full SA Hessian action includes:
- Core and active density response to the rotation R
- Q-matrix derivative
- Commutator terms

Schematically:

```
(delta g_SA)[R] = Σ_I w_I {
    [F_I + F_A^I, R]_pq (inactive-active Fock response)
    + active density response terms
    + Q-matrix derivative terms
}
```

### Deliverables for Phase 3

- Implement `delta_g_SA_action(R, ...)` returning the SA Hessian action.
- Use it inside the shared-κ SA coupled solve (Krylov solve with diagonal
  preconditioner as inner preconditioner).
- Validate with a finite-difference check in `tests/casscf_internal.cpp`.

---

## Phase 4: CI RHS Consistency — DONE (2c3bf20)

The CI response RHS is now built from the exact orbital-derivative active-space
Hamiltonian by default (`ResponseRHSMode::ExactOrbitalDerivative`).

The commutator-only shortcut remains available behind the
`mcscf_debug_commutator_rhs` debug flag.

`tests/casscf_internal.cpp` covers the analytic exact RHS vs finite-difference
active-space Hamiltonian rotation agreement on a small reference problem.

---

## Phase 5: Simplify Optimizer — PARTIAL (2c5038a, 2026-04-03)

### Done in 2c5038a

The large always-on candidate family has been removed. The normal candidate screen
now contains:
- `sa-coupled` (primary production step)
- `sa-grad-fallback`
- `numeric-newton` as small-space escape hatch (npairs ≤ 64)
- `sa-diag-fallback`, per-root candidates, pair probes under stagnation only

### Still remaining

- `numeric-newton` is still a production escape hatch for small spaces, not yet
  debug-only. Removing it without regressing the solver gate requires M2 to be
  robust first.
- Stagnation rescue family (`sa-diag-fallback`, per-root candidates, pair probes)
  is still present. Removal depends on M2+M3 providing reliable globalization.
- The acceptance merit function uses `weighted_root_gnorm` instead of the SA
  gradient `gnorm`. This should be fixed together with M2.

---

## Phase 6: Integrate with Existing Code — DONE

The root-resolved data structures, coupled response blocks, CI solver, Davidson
path, and active integral cache are all in place. The driver logic and convergence
checks reflect the SA stationarity condition (Phase 1). The step construction
(Phase 2) is the main remaining integration target.

---

## Phase 7: Testing — PARTIAL

### Done

- `planck-casscf-internal` covers: direct-sigma vs dense CI agreement, exported
  vs reference RDM agreement, analytic exact RHS vs finite-difference agreement,
  coupled block invariants, coupled orbital/CI residual seeding.
- Manual fixture gate: 11 cases, all passing, including 2 SA fixtures.
- `pyscf_reference_energies.md` skeleton created with PySCF script template.

### Still needed

- Fill PySCF reference values (all SA entries currently TBD).
- Automated PySCF-backed ctest comparisons.
- Regression runner (`run_regressions.py`) SA metric parsing: `sa_g`,
  `root_screen_g`, `max_root_g`.
- Strict SA stationarity assertion after M2: verify convergence was declared
  via `||g_SA||_inf < tol`, not the plateau escape path.
- At least one root-crossing stress case in the fixture suite.

---

## Bugs Confirmed by PySCF Cross-Check (2026-04-05)

PySCF 2.12.1 (Cartesian basis, correct SAD HF starting point) was run against all
9 CASSCF fixture cases. Results (full details in `pyscf_reference_energies.md`):

**Passing (Δ < 1e-7 Eh):** H₂ CAS(2,2), LiH CAS(2,2), ethylene CAS(2,2)/3-21G,
ethylene CAS(2,2)/cc-pVDZ — all CAS(2,2) cases are correct.

**Bug 1 — `guess hcore` + `use_symm true` → wrong HF (~0.5 Eh error):**
Water CAS(4,4) inputs use `guess hcore` + `use_symm true`. Planck HF converges to
−74.4581 Eh instead of the correct −74.9629 Eh. The CASSCF energies for
water_cas44_sto3g/631g/ccpvdz are wrong by ~0.5 Eh. Fix: remove `guess hcore`
from those inputs, re-run, update gate values.

**Bug 2 — SA optimizer finds wrong local minimum (water SA-2, Δ = 0.013 Eh):**
With the correct HF starting point, PySCF finds E_SA = −74.7877 Eh while Planck
finds −74.7751 Eh. PySCF is finding a better SA minimum. Consistent with the
per-root averaged κ issue (Phase 2 below).

**Bug 3 — Twisted ethylene SA-2 active space uncertain (Δ = 0.010 Eh):**
Ethylene SA-2 uses `guess hcore` (HF wrong by 0.112 Eh). Planck finds lower SA
energy than PySCF, but likely because the wrong HF starting point leads to different
active orbitals. Re-run without `guess hcore` to confirm.

---

## Acceptance Criteria

| Criterion | Status |
|---|---|
| nroots=1 results unchanged | Done (CAS(2,2) pass to < 1e-7 Eh) |
| Water CAS(4,4) SS energies correct | **NOT MET** — hcore+symm bug (Bug 1) |
| SA gradient `\|\|g_SA\|\|` used for stopping | Done (Phase 1) |
| nroots>1 converges to correct SA minimum | **NOT MET** — local-min gap (Bug 2) |
| Shared-κ coupled orbital solve implemented | Not done (Phase 2) |
| True Hessian action `δg_SA[R]` in production | Not done (Phase 3) |
| Exact CI response RHS as default | Done (Phase 4) |
| Optimizer simplified to one main path | Partial (Phase 5 first pass) |
| PySCF reference tests passing | 4/9 — bugs above block remaining 5 |

---

## Key Insight

The issue is NOT just the Hessian approximation.

The convergence plateau for nroots>1 had two distinct causes:
1. Wrong convergence criterion (rootwise `g_I = 0` instead of SA `g_SA = 0`) — FIXED
2. Wrong step construction (per-root coupled solves averaged, not shared-κ SA solve) — NOT FIXED

Fixing (1) unblocked the current 2-root benchmarks. Fixing (2) is required for
correct SA second-order behavior and for systems where the per-root average step
diverges from the true SA Newton step.
