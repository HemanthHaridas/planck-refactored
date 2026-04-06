# CASSCF Remaining Work

## Purpose

This note is the current handoff for the next CASSCF implementation pass. It
reflects the repository state on 2026-04-05 (branch `codex/casscf-correctness-pass2`)
after the SA convergence criterion fix and the first optimizer simplification pass.

---

## Current Status

The code has correct SA convergence semantics and a production coupled orbital/CI
step, but it is still not a true SA second-order CASSCF implementation.

**Three bugs confirmed by PySCF reference run (2026-04-05) — fix before other work:**

1. **`guess hcore` + `use_symm true` gives wrong HF for water CAS(4,4)** — the SCF
   converges to E(HF) = −74.4581 Eh instead of the correct −74.9629 Eh. This
   propagates ~0.5 Eh into CASSCF. The benchmark gate values for
   `water_cas44_sto3g`, `water_cas44_631g`, and `water_cas44_ccpvdz` are wrong.
   PySCF confirms the correct energies: −75.0084, −76.0370, −76.0781 Eh respectively.

2. **SA optimizer converges to wrong local minimum for water SA-2** — both Planck
   and PySCF start from the correct RHF (E ≈ −74.9629 Eh), same active space, but
   PySCF finds E_SA = −74.7877 Eh while Planck finds −74.7751 Eh (0.013 Eh higher).
   Consistent with the per-root averaged κ issue (Milestone 2 below).

3. **Twisted ethylene SA-2 active space unclear** — Planck uses `guess hcore` giving
   wrong HF by 0.112 Eh, yet finds E_SA = −77.0034 Eh vs PySCF −76.9930 Eh (0.010
   Eh lower). The lower energy may reflect a different active-orbital basis from the
   wrong HF starting point rather than a better CASSCF minimum. Needs re-run without
   `guess hcore` to confirm.

**Two structural correctness issues (not yet bugs, but known limitations):**

4. **Step construction**: the `sa-coupled` direction is built by averaging per-root
   coupled steps, not by solving the SA stationarity system with one shared `κ`.
5. **Orbital Hessian**: the OO block model is a diagonal energy-denominator
   preconditioner, not a faithful `δg_SA[R]` action.

---

## What Has Landed

- The old monolith split into dedicated `strings`, `ci`, `rdm`, `response`,
  `orbital`, and driver modules.
- RASSCF determinant screening enforced on combined alpha+beta determinant.
- CI symmetry screening uses explicit Abelian irrep product table.
- Root tracking uses overlap-based Hungarian maximum-overlap match.
- CI bitstring helpers guard full-width shifts; active space limit enforced.
- One-body CI sigma and Slater-Condon Hamiltonian share the same ket-to-bra
  operator convention, with internal agreement coverage.
- CI-response Davidson solve has projection, restart/collapse, bounded-subspace
  handling, and best-residual reporting on truncated exits.
- Single-step CI-response fallback no longer reports false convergence.
- Active-space `Q` construction reuses a cached transformed-integral container.
- Direct sigma-vector Davidson CI path with dense fallback, plus agreement tests.
- Per-root `StateSpecificData` carried through the macro loop (CI vectors/energies,
  1-RDMs, 2-RDMs, active Fock, Q, orbital gradient, CI-response data, first-order
  2-RDMs, Q1, CI-driven orbital corrections).
- SA `gamma`, `Gamma`, `F_A`, `g_orb` rebuilt as explicit weighted sums of per-root
  records.
- Response-side bilinear `Gamma1` accumulation expressed as explicit weighted sum.
- Exact orbital-derivative CI response RHS is the default
  (`ResponseRHSMode::ExactOrbitalDerivative`). Commutator-only shortcut is behind
  `mcscf_debug_commutator_rhs` flag.
- Exact RHS tested vs finite-difference active-space Hamiltonian rotation in
  `tests/casscf_internal.cpp`.
- Production matrix-free coupled orbital/CI step via `solve_coupled_orbital_ci_step`:
  seeds from diagonal-preconditioned + CI-response Schur step, refines with explicit
  OO/OC/CO/CC block residuals.
- `sa-coupled` direction promoted to the normal candidate screen.
- First optimizer simplification pass (2c5038a): large always-on AH/mix/gradient-
  variant family removed. Normal candidate screen contains `sa-coupled`,
  `sa-grad-fallback`, small-space `numeric-newton` escape hatch, and stagnation-only
  rescue candidates.
- **SA convergence criterion fixed (c97a754)**: convergence gate now uses
  `||g_SA||_inf < tol` where `g_SA = Σ_I w_I g_I`. Per-root gradient norms
  retained as diagnostics only. Iteration table column renamed "SA Grad"/"MaxRootG".
- Per-macroiteration wall-clock timing in the iteration table.
- SA end-to-end fixtures `water_cas44_sto3g_sa2` and `ethylene_cas44_sto3g_sa2`
  added and passing.
- `calc._cas_mo_coefficients` stores the optimized orbital basis, not natural
  orbitals.
- `tests/casscf_internal.cpp` covers: direct-sigma vs dense CI agreement, exported
  vs reference RDM agreement, exact RHS vs finite-difference agreement, coupled block
  invariants, coupled residual seeding.

---

## What Is Still True

- The `sa-coupled` step is built by per-root coupled solves averaged by weight, not
  by a shared-κ SA coupled solve (the correct SA second-order structure).
- The acceptance merit function uses `weighted_root_gnorm` (a per-root gradient
  screen) instead of the SA gradient `gnorm`. This should be fixed with the shared-κ
  solve (they are part of the same correctness gap).
- The orbital Hessian OO block is a diagonal preconditioner only; no `δg_SA[R]`
  action.
- The optimizer still has more than one production escape hatch: `numeric-newton` for
  small spaces and stagnation-only rescue candidates while the coupled solve hardens.
- PySCF reference values for SA cases are all TBD in `pyscf_reference_energies.md`.
- The regression runner (`run_regressions.py`) does not parse SA gradient diagnostics.

---

## Remaining Work (Priority Order)

### 0. Fix `guess hcore` + `use_symm true` HF convergence bug (BLOCKER)

**Symptoms:** `water_cas44_sto3g`, `water_cas44_631g`, `water_cas44_ccpvdz` all give
E(HF) ≈ −74.4581 Eh instead of the correct ≈ −74.9629 Eh. The CASSCF energies are
wrong by ~0.5 Eh. The SA-2 water input (which uses `use_symm false`, no guess) gives
the correct HF energy, confirming the bug is in the `hcore + symm` combination.

**Investigation targets in the driver / SCF path:**
- `src/driver.cpp` or `src/dft/driver.cpp`: how is `guess hcore` implemented under
  symmetry? Is the initial density matrix built in the symmetry-adapted AO frame?
- Does `detectSymmetry()` reorder or transform basis functions in a way that breaks
  the hcore diagonal guess?

**Immediate actions:**
- Update `water_cas44_sto3g.hfinp`, `water_cas44_631g.hfinp`,
  `water_cas44_ccpvdz.hfinp`: remove `guess hcore` (or switch to `guess sad` if
  supported) and re-run.
- Update the gate energies in `casscf_benchmark_results.txt` to match the corrected
  runs.
- Update `pyscf_reference_energies.md` once Planck values are corrected.

**Reference PySCF values (Cartesian basis, correct RHF starting point):**
- `water_cas44_sto3g`: −75.0084054420 Eh
- `water_cas44_631g`: −76.0370099226 Eh
- `water_cas44_ccpvdz`: −76.0781256226 Eh

### 0b. Investigate twisted ethylene SA-2 without `guess hcore`

The ethylene_cas44_sto3g_sa2 input also uses `guess hcore`. Planck gives E_SA =
−77.0034 Eh while PySCF (correct HF) gives −76.9930 Eh. The lower Planck energy is
suspicious — it likely reflects a different (wrong) orbital basis from the hcore
start rather than a genuinely better SA minimum.

**Immediate actions:**
- Re-run `ethylene_cas44_sto3g_sa2.hfinp` without `guess hcore`.
- Compare resulting E(HF) and E_SA against PySCF.
- If the energies still differ by > 1e-3 Eh after fixing the HF starting point,
  investigate active orbital selection.

**PySCF reference:**
- E_SA = −76.9930595776 Eh (root 0: −76.9940, root 1: −76.9921)

### 1. Implement shared-κ SA coupled solve (Phase 2)

The most important remaining correctness item.

**What to do:**
- Add a `solve_sa_coupled_orbital_ci_step` function that holds one shared `κ`,
  simultaneously solves all roots' CI response equations `(H - E_I) c1_I = -Q sigma_I(κ)`,
  forms the SA orbital residual `g_SA + H_oo κ + Σ_I w_I G_oc c1_I`, and iterates on κ.
- Replace `build_root_resolved_coupled_step_set` with this shared-κ path as the
  production `sa-coupled` direction.
- Fix the acceptance merit function to use `trial.gnorm` (SA gradient) instead of
  `trial.weighted_root_gnorm`.

**Acceptance:**
- 2-root SA ethylene converges via SA stationarity, not the plateau escape.
- `nroots=1` results unchanged.
- All 11 manual fixture inputs continue to pass.

### 2. Fix the acceptance merit function (needed independently of M2)

The merit at `casscf.cpp:1169` is:
```cpp
trial.E_cas + merit_weight * trial.weighted_root_gnorm * trial.weighted_root_gnorm
```
This should be:
```cpp
trial.E_cas + merit_weight * trial.gnorm * trial.gnorm
```
where `trial.gnorm` is `||g_SA||_inf`. The acceptance conditions at lines 1170–1187
use per-root gradient screens that are inconsistent with the SA stationarity condition
fixed in M1.

### 3. Add PySCF-backed reference suite

The most important missing safety net.

**Systems to cover:**
- H₂ CAS(2,2)/STO-3G (SS and SA-2)
- LiH CAS(2,2)/STO-3G (SS)
- H₂O CAS(4,4)/STO-3G (SS and SA-2)
- Twisted ethylene CAS(4,4)/STO-3G (SA-2)

**Script template is in** `tests/inputs/casscf_tests/pyscf_reference_energies.md`.

### 4. Make regression runner SA-aware

Add to `METRIC_PATTERNS` in `tests/run_regressions.py`:
```python
"casscf_sa_gnorm": re.compile(r"sa_g=([-+0-9Ee\.]+)", re.MULTILINE),
"casscf_root_screen_gnorm": re.compile(r"root_screen_g=([-+0-9Ee\.]+)", re.MULTILINE),
"casscf_max_root_gnorm": re.compile(r"max_root_g=([-+0-9Ee\.]+)", re.MULTILINE),
```
Assert `sa_g <= tol_mcscf_grad`; keep root metrics as diagnostics only.

### 5. Add Hessian action upgrade (Phase 3)

After M2 is stable. Add `delta_g_SA_action(R, ...)` with full OO response terms.
Validate with finite-difference check in `tests/casscf_internal.cpp`.
Use inside the shared-κ SA coupled solve.

### 6. Simplify optimizer further (Phase 5 second pass)

After M2+M3:
- Decide if `numeric-newton` can become debug-only for small spaces.
- Remove or narrow the stagnation-only rescue family.
- Move any remaining experiment-only heuristics behind debug flags.

### 7. Expand invariance and robustness coverage

Still missing:
- Active-active rotation invariance checks after convergence.
- Anti-symmetry checks for `g_orb` and `kappa`.
- Root-crossing stress tests (geometry distortion near avoided crossing).
- Response-solver restart and truncation cases at larger CI dimensions.
- Checkpoint/restart consistency for stored CASSCF orbitals.

### 8. Revisit RDM performance (lower priority)

The RDM module is isolated and cross-checked. Optimized 1-RDM, 2-RDM, and
bilinear 2-RDM accumulation kernels are still on the backlog but are lower
priority than theory and validation work.

### 9. Profile direct CI sigma before more CI restructuring

The direct Davidson path is correct and tested. Do not reopen major CI-driver
refactors until profiling identifies actual hot spots in
`apply_ci_hamiltonian`.

---

## Mandatory Change Gate

Every CASSCF solver-facing change must be validated against the full manual fixture
folder. Run:

```bash
cmake --build build --target hartree-fock -j4
for input in tests/inputs/casscf_tests/*.hfinp; do
    ./build/hartree-fock "$input" | tee "${input%.hfinp}.rerun.log"
done
```

Current gate energies (2026-04-04). Entries marked ⚠ are known wrong and must be
updated after the hcore+symm bug is fixed (see item 0 above):

| Input | E(CASSCF) / Eh | PySCF ref / Eh | Status |
|---|---|---|---|
| ethylene_cas44_sto3g_sa2 (SA-2) | −77.0034974301 | −76.9930595776 | ⚠ hcore bug + local-min gap |
| ethylene_casscf_321g | −77.5145223872 | −77.5145223959 | OK (Δ = 8.7e-9) |
| ethylene_casscf_321g_nroot2 | −77.5145223872 | — | OK |
| ethylene_casscf_ccpvdz | −77.9524855976 | −77.9524856210 | OK (Δ = 2.3e-8) |
| h2_cas22_sto3g | −1.1372838351 | −1.1372838345 | OK (Δ = 6.0e-10) |
| lih_cas22_sto3g | −7.8811184797 | −7.8811184639 | OK (Δ = 1.6e-8) |
| water_cas44_631g | −75.5497490402 | −76.0370099226 | ⚠ wrong (hcore+symm, Δ = 0.49) |
| water_cas44_b1 | −74.2879452324 | — | not yet cross-checked |
| water_cas44_ccpvdz | −75.6045806122 | −76.0781256226 | ⚠ wrong (hcore+symm, Δ = 0.47) |
| water_cas44_sto3g | −74.4700757755 | −75.0084054420 | ⚠ wrong (hcore+symm, Δ = 0.54) |
| water_cas44_sto3g_sa2 (SA-2) | −74.7751279351 | −74.7877865139 | ⚠ SA local-min gap (Δ = 0.013) |

Do not call work done if one case improves while another regresses.
The ⚠ entries must not be used as convergence targets until re-run with correct inputs.

---

## What Not To Do Next

- Do not relabel the current `sa-coupled` direction as full SA second-order; it is
  per-root averaged, not shared-κ.
- Do not add more heuristic step mixing to the production path.
- Do not reopen P0 items (RAS filtering, symmetry-table wiring, root tracking,
  bitstring guards) unless a new regression appears.
- Do not overload `_cas_mo_coefficients` with another orbital representation.
- Do not prioritize more file splitting; the important work is theory and validation.

---

## Relevant Files For The Next Pass

- `src/post_hf/casscf/casscf.cpp` — driver, convergence gate, candidate screen,
  acceptance merit function
- `src/post_hf/casscf/response.cpp` — `solve_coupled_orbital_ci_step`,
  `build_coupled_response_blocks`
- `src/post_hf/casscf/orbital.cpp` — `hessian_action`, `hess_diag`,
  `diagonal_preconditioned_orbital_step`
- `tests/casscf_internal.cpp` — internal unit and invariance coverage
- `tests/run_regressions.py` — regression runner (add SA metrics)
- `tests/inputs/casscf_tests/pyscf_reference_energies.md` — PySCF reference values
  (fill TBD entries)
