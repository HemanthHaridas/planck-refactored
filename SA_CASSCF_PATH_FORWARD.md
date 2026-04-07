# SA-CASSCF Path Forward (Milestones + Regression Gates)

Date: 2026-04-06

---

## Milestone Summary

| Milestone | Description | Status |
|---|---|---|
| M1 | SA convergence semantics (`\|\|g_SA\|\|`) | ✅ Done |
| M2 | Shared-κ SA coupled orbital solve | ✅ Done |
| M2b | Acceptance merit uses SA gradient | ✅ Done |
| M3 | Orbital Hessian action upgrade (`δg_SA[R]`) | ❌ Not done |
| M4 | Optimizer second simplification pass | Partial |
| M5 | PySCF reference suite | ✅ Done |
| M6 | Symmetry-aware active orbital selection | ✅ Done |
| —  | Per-root energy display bug | ⚠ New bug (2026-04-06) |

---

## Milestone 1 — SA Convergence Semantics ✅ DONE (c97a754)

The stopping condition is `||g_SA||_inf < tol` where `g_SA = Σ_I w_I g_I`.
This replaced the previous rootwise `g_I = 0 for all I` condition that caused
plateau behavior for nroots > 1.

- `sa_gradient_converged()` tests `sa_gnorm < tol`.
- `st.gnorm` is set to `st.g_orb.cwiseAbs().maxCoeff()` where `st.g_orb` is
  the state-averaged gradient built in `build_weighted_root_orbital_gradient`.
- Stagnation detection uses `sa_gradient_progress_flat`.
- Iteration table columns: "SA Grad" (gating), "MaxRootG" (diagnostic).
- Per-macroiteration wall-clock timing.

**Validated:** CAS(2,2) cases and water B1 pass. Water CAS(4,4) SS fails (orbital
stalling — see Open Bug below). SA cases fail against PySCF SAD-start reference.
Current suite status: 6/11 passing.

---

## Milestone 2 — Shared-κ SA Coupled Orbital Solve ✅ DONE (4fbb1b1)

`solve_sa_coupled_orbital_ci_step` in `response.cpp` implements a true
shared-κ SA coupled solve:

1. Seed the orbital step κ with the diagonal-preconditioned step.
2. For each root, compute the CI response to the shared κ via
   `build_coupled_response_blocks`.
3. Evaluate the SA orbital residual:
   `R(κ) = g_SA + H_oo κ + Σ_I w_I G_oc c1_I(κ)`
4. Update κ using `diagonal_preconditioned_orbital_step` on the residual.
5. Update each `c1_I` using the diagonal CI response preconditioner.
6. Repeat with line-search scaling {1.0, 0.5, 0.25, 0.125} until
   `max(|orbital_residual|, max_I |CI_residual_I|) < tol` or `max_iter`.

`build_root_resolved_coupled_step_set` (per-root averaged κ) remains as a
stagnation-only fallback candidate.

**Milestone 2b — Acceptance merit:** the acceptance merit at
`casscf.cpp:1255` uses `trial.gnorm * trial.gnorm` (SA gradient squared), not
`weighted_root_gnorm`. Acceptance conditions (merit improved, energy improved,
gradient worsen window) are all based on the SA gradient `trial.gnorm`.

**Validated:** SA water and ethylene agree with PySCF from the same (hcore)
starting point. Water CAS(4,4) SS still fails — orbital stalling unrelated to
the shared-κ step. Suite status: 6/11 passing.

---

## Milestone 3 — Orbital Hessian Action Upgrade ❌ NOT DONE

`hessian_action()` in `orbital.cpp:178` is a diagonal energy-denominator model:
```
(HR)_pq = (F_sum[p,p] - F_sum[q,q]) * R_pq
```
This serves as a preconditioner, not a faithful second-order OO block.

### What the true δg_SA[R] action requires

```
(δg_SA)[R] = Σ_I w_I {
    [F_I + F_A^I, R]_pq            (inactive-active Fock response)
    + active density response terms
    + Q-matrix derivative terms
    + commutator terms
}
```

### Deliverables

- Implement `delta_g_SA_action(R, ...)` in `orbital.cpp`.
- Add a finite-difference Hessian-vector check in `tests/casscf_internal.cpp`.
- Wire into `solve_sa_coupled_orbital_ci_step` to replace the diagonal OO block.

### Acceptance

- Finite-difference check passes on the H₂O CAS(4,4)/STO-3G reference problem.
- Convergence rate for the SA coupled solve is no worse than the diagonal baseline.
- All 11 fixture cases continue to pass.

---

## Milestone 4 — Optimizer Simplification Second Pass (PARTIAL)

### Done (2c5038a)

- Large always-on AH/mix/gradient-variant candidate family removed.
- Normal candidate screen: `sa-coupled` (primary), `sa-grad-fallback`.
- Stagnation-only rescue: `numeric-newton` (≤64 pairs), `sa-diag-fallback`,
  per-root candidates, pair probes.

### Still remaining (after M3 is stable)

- Demote `numeric-newton` to debug-only (`mcscf_debug_numeric_newton`).
- Remove the per-root candidate and pair-probe paths from the stagnation rescue.
- Keep one explicit diagnostic escape hatch (`sa-diag-fallback`), clearly labelled.
- Ensure the macro-iteration transcript attributable to a single main path.

### Acceptance

- All 11 fixture cases continue to pass.
- Production convergence is driven by `sa-coupled` in every case.

---

## Milestone 5 — PySCF Reference Suite ✅ DONE (infrastructure); 6/11 passing

`tests/pyscf/` contains 11 individual scripts and `run_all.py`. Suite expanded
from 9 → 11 cases by adding `water_cas44_b1` and `ethylene_casscf_321g_nroot2`.

**Current pass/fail (2026-04-06):**

| Case | PySCF / Eh | Planck / Eh | Delta | Status |
|---|---|---|---|---|
| h2_cas22_sto3g | −1.1372838345 | −1.1372838351 | 6.0e-10 | PASS |
| lih_cas22_sto3g | −7.8811184639 | −7.8811184797 | 1.6e-08 | PASS |
| water_cas44_sto3g | −75.0084054420 | −74.9760171760 | 3.2e-02 | **FAIL** |
| water_cas44_631g | −76.0370099226 | −75.9998609785 | 3.7e-02 | **FAIL** |
| water_cas44_ccpvdz | −76.0781256226 | −76.0440109052 | 3.4e-02 | **FAIL** |
| water_cas44_b1 | −74.5856164512 | −74.5856163677 | 8.4e-08 | PASS |
| ethylene_casscf_321g | −77.5145223959 | −77.5145223872 | 2.8e-07 | PASS |
| ethylene_casscf_321g_nroot2 | −77.5145223959 | −77.5145223872 | 8.7e-09 | PASS |
| ethylene_casscf_ccpvdz | −77.9524856210 | −77.9524855977 | 2.3e-08 | PASS |
| water_cas44_sto3g_sa2 | −74.7877865139 | −74.7751377977 | 1.3e-02 | **FAIL** |
| ethylene_cas44_sto3g_sa2 | −76.9930595776 | −77.0034974301 | 1.0e-02 | **FAIL** |

**Notes on failing cases:**
- Water CAS(4,4) SS (3 cases): RHF now matches PySCF to < 2e-8 Eh (Bug 1 fixed).
  CASSCF stalls at a higher-energy stationary point — Planck recovers 3–3.5× less
  correlation energy than PySCF despite identical starting points. See Open Bug below.
- Water SA-2: from the hcore start both codes find −74.7751377977 Eh. The PySCF
  SAD-start minimum (−74.7877865139 Eh) is 0.013 Eh lower. Planck has not been
  tested from a SAD-equivalent start.
- Ethylene SA-2: Planck and PySCF converge to different RHF stationary points
  (5.9e-2 Eh apart) regardless of guess. Comparison is not apples-to-apples;
  FAIL is flagged but the case is not a simple correctness failure.

### Open Bug — Water CAS(4,4) SS orbital stalling

Despite identical RHF starting points, Planck recovers only ~3× less correlation
energy than PySCF across all three bases. The active orbital characters at Planck's
converged solution likely differ from PySCF's, indicating the optimizer is at a
different (higher-energy) stationary point of the CAS energy surface.

---

## Milestone 6 — Symmetry-Aware Active Orbital Selection ✅ DONE (991ace6)

`select_active_orbitals` in `strings.cpp`:
- Accepts `core_irrep_counts`, `active_irrep_counts`, `mo_permutation` quotas.
- Falls back to energy-sorted inference from existing MO symmetry labels when no
  quotas are given and no prior CASSCF guess is present.
- Falls back to the identity permutation when symmetry labels are absent.
- Returns an `ActiveOrbitalSelection` with a full column permutation.

`reorder_mo_coefficients` applies the permutation before the MCSCF loop.
Unit tests in `tests/casscf_internal.cpp` cover 4 cases including error paths.

---

## Known Bug — Per-Root Energy Display ⚠ (introduced 2026-04-06)

`calc._cas_root_energies(r) = fst.roots[r].ci_energy` stores the CI eigenvalue
of the active-space Hamiltonian. This does not include the core Fock energy or
nuclear repulsion. The correct per-root total energy is:

```
E_total(r) = E_nuc + E_core + ci_energy(r)
```

The log currently shows nonsense values (e.g., −5.27 Eh for ethylene when the
correct value is −77.00 Eh).

**Fix:** at `casscf.cpp:1418`, store `calc._nuclear_repulsion + fst.E_core +
fst.roots[r].ci_energy` instead of `fst.roots[r].ci_energy`. This requires
`fst.E_core` to be accessible at that point. Alternatively, pass `E_nuc +
E_core` through the `SolverState` and accumulate it there.

---

## Regression Gates

### Mandatory (run before every merge)

```bash
cmake --build build --target hartree-fock planck-casscf-internal -j4
./build/planck-casscf-internal
for input in tests/inputs/casscf_tests/*.hfinp; do
    ./build/hartree-fock "$input" | tee "${input%.hfinp}.rerun.log"
done
```

### PySCF-backed (requires tests/pyscf/.venv)

```bash
tests/pyscf/.venv/bin/python3 tests/pyscf/run_all.py
```

Current result: 6/11 passing. The 5 failing cases are:
- water_cas44_sto3g, water_cas44_631g, water_cas44_ccpvdz: orbital stalling
- water_cas44_sto3g_sa2: SA optimizer does not reach SAD-start PySCF minimum
- ethylene_cas44_sto3g_sa2: different HF stationary points

### Still needed: strict SA stationarity assertion

`run_regressions.py` parses `casscf_sa_gnorm` but does not assert it is
`< tol_mcscf_grad`. Add this assertion for SA test cases so regressions where
the plateau escape fires instead of true SA convergence are caught automatically.
