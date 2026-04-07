# Unified SA-CASSCF Plan (Theory + Implementation)

Last updated: 2026-04-06

---

## Core Problem

The SA-CASSCF objective is:

```
E_SA = Σ_I w_I E_I
```

Stationarity of E_SA with respect to orbital rotations requires:

```
g_SA = Σ_I w_I g_I = 0
```

The original implementation mixed a state-averaged objective with rootwise
convergence criteria and a per-root averaged orbital update. This caused:
1. False non-convergence when `g_SA` was small but individual `g_I` were not.
2. Steps that minimized per-root residuals independently rather than the
   SA residual simultaneously.

Both issues have now been fixed.

---

## Phase 1: Fix Convergence Semantics — ✅ DONE (c97a754, 2026-04-04)

The stopping condition is:
```
||g_SA||_inf < tol
```
where `g_SA = st.g_orb = Σ_I w_I g_I` (built in `build_weighted_root_orbital_gradient`).

Per-root gradient norms `weighted_root_gnorm` and `max_root_gnorm` are retained
as diagnostics and the SA gradient gates all convergence and stagnation checks.

Iteration table: "SA Grad" (gating norm), "MaxRootG" (diagnostic).
Stagnation: `sa_gradient_progress_flat` uses the SA norm.

**Validated:** CAS(2,2) cases and water B1 pass. Water CAS(4,4) SS stalls (see
Known Bug below). SA cases agree from hcore start but fail against SAD reference.
Current suite: 6/11 passing.

---

## Phase 2: True SA Coupled Solve with One Shared κ — ✅ DONE (4fbb1b1, 2026-04-05)

### What existed before

`build_root_resolved_coupled_step_set` solved per-root coupled orbital/CI steps
independently and averaged the orbital step:
```
κ_SA = Σ_I w_I κ_I
```
Each root minimized its own residual. This was not a solution to the SA
stationarity system.

### What exists now

`solve_sa_coupled_orbital_ci_step` in `response.cpp` is the production
`sa-coupled` candidate. It holds one shared κ and iterates:

```
Given κ:
  For each root I: solve (H - E_I) c1_I ≈ -Q sigma_I(κ)  (CI response)
  Form SA orbital residual: r = g_SA + H_oo κ + Σ_I w_I G_oc c1_I
  If ||r|| < tol: return κ
  Update κ ← κ - preconditioner(r)   [diagonal H_oo preconditioner]
  Update c1_I ← c1_I - preconditioner(CI_residual_I)
```

Line-search scaling {1.0, 0.5, 0.25, 0.125} is applied to the (κ, {c1_I})
update before accepting. The residual metric is `max(|r_orb|, max_I |r_CI_I|)`.

`build_root_resolved_coupled_step_set` is retained as a stagnation-only fallback.

### Phase 2b: Acceptance merit

The acceptance merit function at `casscf.cpp:1255` uses:
```cpp
trial.E_cas + merit_weight * trial.gnorm * trial.gnorm
```
where `trial.gnorm = ||g_SA||_inf`. Per-root weighted screen `weighted_root_gnorm`
appears only as a diagnostic in the iteration transcript.

**Validated:** SA water and ethylene agree with PySCF from the hcore starting
point. Water CAS(4,4) SS still fails (orbital stalling, unrelated to shared-κ).
Suite status: 6/11 passing.

---

## Phase 3: Orbital Hessian Action Upgrade — ❌ NOT DONE

### Current state

`hessian_action()` in `orbital.cpp:178` is a diagonal energy-denominator model:
```
(HR)_pq = (F_sum[p,p] - F_sum[q,q]) * R_pq
```
This is used as the OO block preconditioner inside the shared-κ coupled solve.

### The true δg_SA[R] action

The full SA Hessian-vector product includes:
```
(δg_SA)[R]_pq = Σ_I w_I {
    [F_I + F_A^I, R]_pq            (inactive-active Fock response)
    + 2 Σ_rs (pq|rs) δD_rs[R]     (active density response)
    + δQ_pq[R]                     (Q-matrix derivative)
    + commutator terms
}
```
where `δD[R]` is the first-order response of the active 1-RDM and `δQ[R]` is
the Q-matrix derivative, both driven by the CI response `c1_I(R)` of each root
to the orbital rotation R.

### Deliverables for Phase 3

- Implement `delta_g_SA_action(R, ...)` in `orbital.cpp` computing the above.
- Use it inside `solve_sa_coupled_orbital_ci_step` to replace the diagonal OO
  block. The inner CI response calls for `δD[R]` can reuse `build_coupled_response_blocks`.
- Add a finite-difference Hessian-vector check in `tests/casscf_internal.cpp`:
  compare `delta_g_SA_action(R)` against `(g_SA(κ + ε*R) - g_SA(κ)) / ε`.

### Acceptance

- Finite-difference check passes to within 1e-6 relative error on H₂O CAS(4,4).
- Convergence rate for the coupled solve is no worse than the diagonal baseline.
- All 11 gate cases continue to pass.

---

## Phase 4: CI Response RHS Consistency — ✅ DONE (2c3bf20)

The CI response RHS is built from the exact orbital-derivative active-space
Hamiltonian by default (`ResponseRHSMode::ExactOrbitalDerivative`).

The commutator-only shortcut is available behind `mcscf_debug_commutator_rhs`.

`tests/casscf_internal.cpp` verifies the analytic exact RHS against a
finite-difference active-space Hamiltonian rotation.

---

## Phase 5: Optimizer Simplification — PARTIAL

### Done (2c5038a, 2026-04-03)

The large always-on candidate family was removed. The normal candidate screen
is now:
- `sa-coupled` — primary production step (shared-κ SA coupled solve)
- `sa-grad-fallback` — diagonal gradient step backup

Under stagnation (stagnation_streak ≥ 2):
- `numeric-newton` — small-space escape hatch (npairs ≤ 64)
- `sa-diag-fallback`, per-root candidates, single-pair probes

### Still remaining (after Phase 3 is stable)

- Demote `numeric-newton` to debug-only (`mcscf_debug_numeric_newton`).
  The shared-κ solve should handle small spaces without needing a separate
  numerical-Newton path.
- Remove per-root candidates and pair probes from the stagnation family.
  With a robust Hessian action, the shared-κ solve should not stagnate.
- Keep `sa-diag-fallback` as the one explicit diagnostic fallback.
- Ensure every step label in the transcript uniquely identifies the path taken.

---

## Phase 6: Active Orbital Selection — ✅ DONE (991ace6, 2026-04-06)

`select_active_orbitals` in `strings.cpp` provides symmetry-aware selection:

- **Explicit permutation:** validates and normalises a user-supplied full-column
  permutation from `mo_permutation` input keyword.
- **Explicit irrep quotas:** builds core/active blocks from `core_irrep_counts`
  and `active_irrep_counts` keywords, energy-ordered per irrep.
- **Automatic inference:** when no quotas are given and no prior CASSCF guess
  exists, infers the irrep pattern from the current RHF ordering and re-selects
  using energy-sorted representatives, so the active block is always contiguous.
- **Identity fallback:** used when symmetry labels are absent.

`reorder_mo_coefficients` applies the permutation before the MCSCF loop.

Parser additions: `IrrepCount` struct; `core_irrep_counts`, `active_irrep_counts`,
`mo_permutation` keywords in the `[scf]` section.

---

## Phase 7: Testing — PARTIAL

### Done

- `planck-casscf-internal` covers: direct-sigma vs dense CI, RDM agreement,
  analytic exact RHS vs finite-difference, coupled block invariants, coupled
  orbital/CI residual seeding, `select_active_orbitals` (4 cases), single-step
  CI response convergence detection.
- Manual fixture gate: 11 cases; 6/11 passing (5 fail — see Known Bug).
- PySCF reference suite: `tests/pyscf/run_all.py` with 11 scripts; 6/11 pass.
- Regression runner: parses `casscf_sa_gnorm`, `casscf_root_screen_gnorm`,
  `casscf_max_root_gnorm`.

### Still needed

- Strict SA stationarity assertion in `run_regressions.py`: assert `sa_g <
  tol_mcscf_grad` for SA cases; verify no plateau escape in final iteration.
- Active-active rotation invariance check after convergence.
- Anti-symmetry checks for `g_orb` and `kappa`.
- Root-crossing stress test (geometry distortion near avoided crossing).
- Checkpoint/restart consistency for CASSCF MO coefficients.
- Larger CI dimension CI-response restart/truncation cases (≥100 determinants).
- Water SA-2 validation from the default (SAD) starting point vs
  PySCF minimum −74.7877865139 Eh.
- Finite-difference Hessian-vector test for Phase 3 (not yet written).

---

## Known Bug — Water CAS(4,4) SS Orbital Stalling (3 cases failing)

Water CAS(4,4) RHF is now correct (Bug 1 fixed). Despite identical RHF starting
points, Planck's CASSCF recovers 3–3.5× less correlation energy than PySCF:

| Case | Planck / Eh | PySCF / Eh | ΔE_corr ratio |
|---|---|---|---|
| STO-3G | −74.9760 | −75.0084 | 3.5× |
| 6-31G | −75.9999 | −76.0370 | 3.4× |
| cc-pVDZ | −76.0440 | −76.0781 | 3.0× |

The basis-independent ratio indicates the orbital optimizer is stalling at a
higher-energy stationary point, not a geometry or basis issue. This is the
primary open correctness bug (see `casscf_remaining_work.md` item 0).

## Known Bug — Per-Root Total Energy (introduced 2026-04-06)

`calc._cas_root_energies` stores CI eigenvalues, not total molecular energies.
The correct per-root total energy requires adding the inactive energy offset:

```
E_total(r) = E_nuc + E_core + ci_energy(r)
```

Fix at `casscf.cpp:1418`. See `SA_CASSCF_PATH_FORWARD.md` for details.

---

## Acceptance Criteria Summary

| Criterion | Status |
|---|---|
| `\|\|g_SA\|\|` used for SA stopping | ✅ Done (Phase 1) |
| Shared-κ coupled orbital solve | ✅ Done (Phase 2) |
| Merit function uses SA gradient | ✅ Done (Phase 2b) |
| Exact CI response RHS as default | ✅ Done (Phase 4) |
| Symmetry-aware MO selection | ✅ Done (Phase 6) |
| CAS(2,2) cases match PySCF < 1e-7 Eh | ✅ Done |
| Water/ethylene CAS(4,4) SS correct | ❌ RHF fixed; CASSCF stalls (3 cases fail) |
| SA cases agree with PySCF (same start) | Partial — hcore-start agrees; SAD-start fails |
| SA water validated at global minimum | ❌ Not verified |
| True Hessian action `δg_SA[R]` | ❌ Not done (Phase 3) |
| Optimizer reduced to one main path | Partial (Phase 5) |
| Per-root energy display correct | ❌ Bug (2026-04-06) |
| Strict SA stationarity assertion | ❌ Not done |
| Active-active/anti-symmetry invariance | ❌ Not done |
