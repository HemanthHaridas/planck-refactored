# CASSCF Remaining Work

## Purpose

This is the live handoff document for CASSCF implementation. It reflects the
repository state on 2026-04-06 (branch `codex/casscf-correctness-pass2`) after
the shared-κ SA coupled solve landed and the symmetry-aware MO selection pass.

---

## What Has Landed (Complete)

### Theory and convergence
- SA convergence gated on `||g_SA||_inf < tol` where `g_SA = Σ_I w_I g_I`.
- Stagnation detection uses the SA gradient via `sa_gradient_progress_flat`.
- Iteration table prints "SA Grad" and "MaxRootG" as separate columns.
- Exact CI response RHS (`ResponseRHSMode::ExactOrbitalDerivative`) as default.
  Commutator-only shortcut behind `mcscf_debug_commutator_rhs` flag.
- **Shared-κ SA coupled solve** (`solve_sa_coupled_orbital_ci_step`): holds one
  orbital step κ, solves all roots' CI responses to that shared κ, iterates on
  the SA orbital residual `g_SA + H_oo κ + Σ_I w_I G_oc c1_I(κ)`. This replaced
  the per-root averaged κ approach as the production `sa-coupled` candidate.
- Acceptance merit uses `trial.gnorm` (SA gradient), not `weighted_root_gnorm`.

### Active space and orbital setup
- Symmetry-aware active orbital picker (`select_active_orbitals` in `strings.cpp`):
  supports explicit `core_irrep_counts`/`active_irrep_counts`/`mo_permutation`
  input keywords; falls back to energy-sorted inference when no quotas are given.
- `reorder_mo_coefficients` applies the picker's column permutation before the
  MCSCF loop, ensuring the active block is always contiguous.
- `IrrepCount` struct and three new `[scf]` keywords added to the parser.

### Bugs fixed
- **`guess hcore` + `use_symm true` → wrong HF**: water CAS(4,4) inputs now
  converge to the correct RHF (−74.9629 Eh). All three water SS gate values
  updated. Root cause: d2d RHF branch preservation fix (commit 46aa199).
- **Twisted ethylene SA-2 active space**: input still uses `guess hcore`; the
  correct comparison is now hcore-vs-hcore with PySCF. Energies agree at the
  same starting point. Both codes find −77.0034974301 Eh from the hcore start.

### Validation
- Per-root CI energies stored in `calc._cas_root_energies` and printed in the
  CASSCF summary. NOTE: this field stores CI eigenvalues, not total energies
  — **the displayed per-root numbers are wrong** (see Known Bugs below).
- PySCF reference suite fully operational: `tests/pyscf/` has individual scripts
  plus `run_all.py`. Current status: **6/11 pass**. The 5 failing cases are
  documented in Known Bugs 2 and 3 below.
- Regression runner (`run_regressions.py`) parses `sa_g`, `root_screen_g`,
  `max_root_g` SA diagnostics.
- CI-response single-step fallback now correctly reports `converged = true` when
  the diagonal preconditioner solves the equation exactly.
- `casscf_internal` unit harness covers: direct-sigma vs dense CI, RDM
  agreement, exact RHS vs finite-difference, coupled block invariants, coupled
  residual seeding, `select_active_orbitals` (4 cases), single-step CI
  response convergence.

### Optimizer
- First simplification pass: large always-on AH/mix/gradient-variant family
  removed. Production candidate screen: `sa-coupled` (primary), `sa-grad-fallback`,
  `numeric-newton` (small-space, ≤64 pairs), stagnation-only rescue family.
- `build_root_resolved_coupled_step_set` remains as a stagnation-only fallback
  candidate (not the production path).

---

## Known Bugs

### 2. Water CAS(4,4) SS: orbital optimizer stalls (3 cases failing)

Water_cas44_sto3g, water_cas44_631g, water_cas44_ccpvdz all start from the
correct RHF (Bug 1 fixed; RHF matches PySCF to < 2e-8 Eh). Despite this,
Planck's CASSCF recovers only ~3× less correlation energy than PySCF:

| Case | PySCF CASSCF / Eh | Planck CASSCF / Eh | Delta |
|---|---|---|---|
| STO-3G | −75.0084054420 | −74.9760171760 | 3.2e-02 |
| 6-31G | −76.0370099226 | −75.9998609785 | 3.7e-02 |
| cc-pVDZ | −76.0781256226 | −76.0440109052 | 3.4e-02 |

The basis-independent ratio (~3–3.5×) rules out geometry/basis mismatch. Planck's
orbital optimizer is stalling at a higher-energy stationary point of the CAS energy.

**Fix:** investigate whether the `sa-coupled` candidate is correctly driving
orbital rotations in the core-active and active-virtual blocks for this system.
Compare active orbital characters at convergence (Planck vs PySCF).

### 3. Water SA-2 and ethylene SA-2: SA optimizer at suboptimal minimum

- Water SA-2: from hcore start Planck agrees with PySCF. PySCF from SAD start finds
  −74.7877865139 Eh (0.013 Eh lower). Planck has not been tested from a SAD start.
- Ethylene SA-2: Planck and PySCF converge to different RHF stationary points
  (5.9e-2 Eh apart); comparison is blocked.

### 1. Per-root energy display is wrong (NEW, introduced 2026-04-06)

`calc._cas_root_energies(r)` stores `fst.roots[r].ci_energy` — the CI
eigenvalue of the active-space Hamiltonian. This does NOT include the core
Fock energy or nuclear repulsion. The correct per-root total energy is:

```
E_total(r) = E_nuc + E_core + ci_energy(r)
```

where `E_core = compute_core_energy(h_mo, F_I_mo, n_core)`.

Currently the log shows nonsense values (−5.27, −5.27 Eh for ethylene SA-2;
−6.07, −5.68 Eh for water SA-2) instead of the expected ~−77.00/~−74.77 Eh.

**Fix:** store `E_nuc + E_core + root.ci_energy` at `casscf.cpp:1418`, or pass
`E_nuc + E_core` to `casscf_summary` and add it there.

### 2. Water SA-2: not validated against best-known SA minimum

Planck and PySCF (both from hcore start) agree on −74.7751377977 Eh. The
original PySCF SAD run found −74.7877865139 Eh. The SAD minimum may be the true
global SA minimum. This is not a regression, but it means the SA optimizer has
not been validated against the lowest-energy SA solution from a neutral start.

---

## Remaining Work (Priority Order)

### 0. Fix water CAS(4,4) SS orbital stalling (BLOCKER — 3 cases fail)

See Known Bug 2. Despite correct RHF, Planck's orbital optimizer stalls at a
higher-energy CASSCF solution (3–3.5× less correlation energy than PySCF).

Suggested investigation steps:
- Compare active orbital characters at Planck's converged solution against PySCF.
- Check whether the `sa-coupled` primary candidate is producing non-zero orbital
  steps in the core-active and active-virtual blocks for this system.
- Verify that the stagnation rescue does not prematurely freeze the optimization.

**File:** `src/post_hf/casscf/casscf.cpp`, `src/post_hf/casscf/response.cpp`

### 1. Fix per-root total energy display (BLOCKER for next release)

See Known Bug 1 above. The fix is simple: before storing `_cas_root_energies`,
add the inactive energy (`E_nuc + E_core`). All downstream consumers of this
field (logging) will then display correct total energies.

**File:** `src/post_hf/casscf/casscf.cpp:1415–1419`

### 2. Orbital Hessian action upgrade

`hessian_action()` in `orbital.cpp:178` is a diagonal energy-denominator
preconditioner. It is used inside the shared-κ coupled solve as the OO block
model. A faithful `δg_SA[R]` action would include core/active density response,
Q-matrix derivative, and commutator terms.

**Why it matters:** the shared-κ solve converges with the diagonal model, but
a better Hessian would improve convergence rate and reduce reliance on the
stagnation rescue candidates.

**Deliverables:**
- Implement `delta_g_SA_action(R, ...)` in `orbital.cpp`.
- Add a finite-difference check in `tests/casscf_internal.cpp`.
- Wire into `solve_sa_coupled_orbital_ci_step` as the OO block action.

**Acceptance:** finite-difference check passes; convergence rate for the SA
coupled solve is at least as good as the diagonal baseline.

### 3. SA stationarity assertion in the regression runner

Currently `run_regressions.py` parses `sa_g` but does not assert that
convergence was declared via `||g_SA||_inf < tol` rather than the stagnation
plateau escape. This should be a hard assertion for SA cases.

**Deliverables:**
- In `run_regressions.py`, after parsing `casscf_sa_gnorm`, assert it is
  `< tol_mcscf_grad` (default 1e-5) for all SA test cases.
- Also verify the final macro iteration did not fire the plateau escape path
  (check that no `[WRN]` line about plateau appears in the last pass).

### 4. Optimizer second simplification pass

`numeric-newton` is still a production escape hatch for spaces with ≤64 pairs.
The stagnation rescue family (`sa-diag-fallback`, per-root candidates, pair
probes) is still active. With the shared-κ solve stable, these can be narrowed.

**Deliverables (after verifying the shared-κ solve is robust):**
- Demote `numeric-newton` to a debug-only option (behind `mcscf_debug_numeric_newton`).
- Narrow the stagnation rescue family: keep one clear diagnostic escape hatch,
  remove the redundant `per-root candidates` and pair-probe paths.
- Ensure the macro-iteration transcript clearly shows which path was taken.

**Gate:** all 11 fixture cases continue to pass.

### 5. Validate water SA-2 from SAD starting point

The Planck SA optimizer has not been validated against the lowest known SA
minimum for water SA-2. The SAD-start PySCF result (−74.7877865139 Eh) is
0.013 Eh lower than the hcore-start result both codes currently agree on.

**Deliverables:**
- Add a water SA-2 input that uses the default guess (no `guess hcore`).
- Run Planck and compare to the SAD-start PySCF value −74.7877865139 Eh.
- If Planck finds the lower minimum, update the gate value and document.
- If Planck still finds the higher minimum, record it as a known optimizer
  limitation and track as a future correctness item.

### 6. Invariance and robustness coverage

Still missing from the unit harness:
- Active-active rotation invariance after convergence (rotate within the active
  block, verify energy is unchanged and `g_orb` block is zero).
- Anti-symmetry checks for `g_orb[p,q]` and `kappa[p,q]`.
- Root-crossing stress test (geometry distortion near avoided crossing; verify
  root tracking does not swap roots mid-optimization).
- Checkpoint/restart consistency: store CASSCF MO coefficients, reload them,
  verify the energy and gradient reproduce within 1e-10 Eh.
- CI-response restart and truncation cases at larger CI dimension (≥100 dets).

### 7. RDM performance (lower priority)

The RDM module is correct and cross-checked. Optimized 1-RDM, 2-RDM, and
bilinear 2-RDM accumulation kernels are still on the backlog but are lower
priority than theory and validation work.

### 8. Profile direct CI sigma before restructuring (lower priority)

The direct Davidson path is correct and tested. Do not reopen major CI-driver
refactors until profiling of `apply_ci_hamiltonian` identifies actual hot spots.

---

## Current Gate Energies (2026-04-06) — 6/11 passing

Water CAS(4,4) SS RHF is now correct (Bug 1 fixed), but CASSCF stalls (see Known Bug 2).
SA cases agree with PySCF from the hcore start, but fail against the SAD-start reference.

| Input | Planck / Eh | PySCF ref / Eh | Delta | Status |
|---|---|---|---|---|
| h2_cas22_sto3g | −1.1372838351 | −1.1372838345 | 6.0e-10 | OK |
| lih_cas22_sto3g | −7.8811184797 | −7.8811184639 | 1.6e-08 | OK |
| water_cas44_sto3g | −74.9760171760 | −75.0084054420 | 3.2e-02 | **FAIL** |
| water_cas44_631g | −75.9998609785 | −76.0370099226 | 3.7e-02 | **FAIL** |
| water_cas44_ccpvdz | −76.0440109052 | −76.0781256226 | 3.4e-02 | **FAIL** |
| water_cas44_b1 | −74.5856163677 | −74.5856164512 | 8.4e-08 | OK |
| water_cas44_sto3g_sa2 (SA-2) | −74.7751377977 | −74.7877865139 | 1.3e-02 | **FAIL** |
| ethylene_cas44_sto3g_sa2 (SA-2) | −77.0034974301 | −76.9930595776 | 1.0e-02 | **FAIL** (blocked) |
| ethylene_casscf_321g | −77.5145223872 | −77.5145223959 | 8.7e-09 | OK |
| ethylene_casscf_321g_nroot2 | −77.5145223872 | −77.5145223959 | 8.7e-09 | OK |
| ethylene_casscf_ccpvdz | −77.9524855977 | −77.9524856210 | 2.3e-08 | OK |

PySCF references for water SA-2 use the SAD-start value (−74.7877865139 Eh).
From the hcore start, Planck and PySCF agree to < 1e-10 Eh. The ethylene SA-2
comparison is blocked by different HF stationary points (see Known Bug 3).

Do not call work done if one case improves while another regresses.

---

## Mandatory Verification Loop

Run after every solver-facing change:

```bash
cmake --build build --target hartree-fock planck-casscf-internal -j4
./build/planck-casscf-internal
for input in tests/inputs/casscf_tests/*.hfinp; do
    ./build/hartree-fock "$input" | tee "${input%.hfinp}.rerun.log"
done
```

---

## What Not To Do Next

- Do not add more heuristic step mixing to the production path.
- Do not reopen P0 items (RAS filtering, symmetry-table wiring, root tracking,
  bitstring guards) unless a new regression appears.
- Do not overload `_cas_mo_coefficients` with another orbital representation.
- Do not prioritize more file splitting or abstraction; the important work is
  theory, validation, and the Hessian upgrade.
- Do not change the PySCF reference energies without re-running the PySCF
  scripts and documenting why the reference changed.

---

## Key Files

| File | Purpose |
|---|---|
| `src/post_hf/casscf/casscf.cpp` | Driver, convergence gate, candidate screen, acceptance merit |
| `src/post_hf/casscf/response.cpp` | `solve_sa_coupled_orbital_ci_step`, `build_coupled_response_blocks` |
| `src/post_hf/casscf/orbital.cpp` | `hessian_action`, `hess_diag`, `diagonal_preconditioned_orbital_step` |
| `src/post_hf/casscf/strings.cpp` | `select_active_orbitals`, `reorder_mo_coefficients`, CI string algebra |
| `tests/casscf_internal.cpp` | Internal unit and invariance coverage |
| `tests/pyscf/run_all.py` | PySCF-backed energy validation suite |
| `tests/run_regressions.py` | Binary output regression runner (parses SA metrics) |
| `tests/inputs/casscf_tests/pyscf_reference_energies.md` | Reference energy table with notes |
