# CASSCF / SA-CASSCF Implementation Status

Last updated: 2026-04-10 (branch `codex/casscf-active-cache-patch1`)

---

## Overview

Full CASSCF and SA-CASSCF are implemented and validated. The macro-optimizer
uses a shared-κ SA coupled orbital/CI solve as the primary step, with a
finite-difference Newton escape hatch for small active spaces (≤64 non-redundant
pairs). The dedicated active-integral-cache transform is landed, the true
FD-based SA orbital Hessian action (`delta_g_sa_action`) is implemented and
wired into all response call sites, and the per-root SA energy display bug is
fixed. The PySCF reference suite passes 11/11. Open work is limited to
regression tightening (SA gnorm assertions) and optimizer simplification.

---

## What Is Done

### Theory and convergence
- SA convergence gated on `‖g_SA‖∞ < tol` where `g_SA = Σ_I w_I g_I`
  (`build_weighted_root_orbital_gradient`). Not per-root.
- Stagnation detection via `sa_gradient_progress_flat` uses the SA norm.
- Iteration table prints "SA Grad" (gating) and "MaxRootG" (diagnostic).
- Exact CI-response RHS (`ResponseRHSMode::ExactOrbitalDerivative`) as default;
  commutator-only shortcut behind `mcscf_debug_commutator_rhs`.
- **Shared-κ SA coupled solve** (`solve_sa_coupled_orbital_ci_step`,
  `response.cpp`): holds one orbital step κ, solves all roots' CI responses to
  that shared κ, iterates on the SA orbital residual
  `g_SA + H_oo κ + Σ_I w_I G_oc c1_I(κ)`.
- Acceptance merit uses `trial.gnorm` (SA gradient norm), not per-root screens.
- `delta_g_sa_action` (`orbital.cpp`): true FD-based SA orbital Hessian-vector
  product — central-difference of `fixed_ci_orbital_gradient` at ±ε·R.
  Falls back to diagonal energy-denominator model when context is incomplete.
  `matrix_free_hessian_action` is a thin alias that delegates to it.
  Wired into all orbital/CI response call sites in `response.cpp`.
- Plateau escape: if stagnation streak ≥ 2 and energy + step are flat,
  convergence is declared with a `[WRN]` line.

### Active space and orbital setup
- Symmetry-aware active orbital picker (`select_active_orbitals`, `strings.cpp`):
  explicit `core_irrep_counts`/`active_irrep_counts`/`mo_permutation` keywords;
  falls back to energy-sorted inference; identity fallback when symmetry absent.
- `reorder_mo_coefficients` applies the permutation before the MCSCF loop.
- `IrrepCount` struct; three `[scf]` parser keywords added.

### Optimizer candidates (current production screen)

Normal macroiteration:
- `sa-coupled` — primary: shared-κ SA coupled solve
- `sa-grad-fallback` — diagonal gradient step backup

Under stagnation (`stagnation_streak ≥ 2`):
- `numeric-newton` — small-space escape (npairs ≤ 64); exact FD Hessian
- `sa-diag-fallback`, per-root candidates, single-pair probes

Step scales tried: `{1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625}`;
lowest merit `E_cas + 0.1·‖g_orb‖²` wins.

### Active-integral-cache path
- `transform_eri_active_cache(...)` (`src/post_hf/integrals.cpp`) is the
  dedicated builder for the mixed-basis active cache used by CASSCF orbital
  gradients and CI response.
- Cache layout is row-major `(p,u,v,w)` with one contiguous `n_act^3` slab per
  fixed `p`.
- OpenMP parallelism is over disjoint `p` slabs with static scheduling; scratch
  buffers are thread-local and reused across repeated cache rebuilds.
- `contract_q_matrix` / `compute_Q_matrix` consume the cache as contiguous slab
  dot products against `Γ[t,u,v,w]`, matching the producer layout exactly.

### Reporting and bookkeeping
- `Calculator::_cas_root_energies` now stores per-root **total** CASSCF
  energies (`E_nuc + E_core + E_CI(root)`), so the printed SA root table is
  numerically consistent with the final SA total energy.

### Validation
- `planck-casscf-internal` unit harness covers: direct-sigma vs dense CI,
  RDM agreement, exact RHS vs finite-difference, coupled block invariants,
  coupled residual seeding, `select_active_orbitals` (4 cases), single-step
  CI response convergence.
- PySCF reference suite (`tests/pyscf/run_all.py`): **11/11 passing**.
- Regression runner (`run_regressions.py`) parses `sa_g`, `root_screen_g`,
  `max_root_g` SA diagnostics.
- Fresh regression after the active-cache work:
  - `water_cas44_sto3g` matches the checked-in convergence trace after
    normalizing away timing-only lines.
  - `water_cas44_sto3g_sa2` prints physically meaningful per-root total
    energies whose weighted average exactly matches the reported SA total.
- Fresh benchmark after the active-cache work:
  - `ethylene_cas44_sto3g_sa2`, `OMP_NUM_THREADS=4`
  - preserved baseline wall time: `165.791 s`
  - optimized rebuilt wall time: `46.974 s`
  - speedup: `3.53x` (~`71.7%` reduction)

### Historical bugs fixed
- **Reversed Cayley sign** (`apply_orbital_rotation`): was applying `exp(-κ)`
  instead of `exp(+κ)`, causing the line search to test uphill directions and
  reject every candidate. Fixed (see Appendix B).
- **`guess hcore` + `use_symm true` → wrong RHF**: d2d RHF branch preservation
  fix (commit 46aa199). All three water SS gate values pass from the correct
  RHF starting point.
- **Per-root SA energy display**: the summary was printing raw CI eigenvalues
  instead of full per-root CASSCF total energies. Fixed by storing
  `E_nuc + E_core + E_CI(root)` in `Calculator::_cas_root_energies`.

---

## PySCF Gate Table (2026-04-08, 11/11 passing)

**Suite status:** 11/11 passing  
PySCF version: 2.12.1. All scripts use `mol.cart = True` (Cartesian basis).  
Tolerance: 1e-5 Eh (all pass well within this).

| Case | Active space | Basis | PySCF / Eh | Planck / Eh | Delta / Eh | Status |
|---|---|---|---|---|---|---|
| h2_cas22_sto3g | CAS(2e,2o) | STO-3G | −1.1372838345 | −1.1372838351 | 6.0e-10 | PASS |
| lih_cas22_sto3g | CAS(2e,2o) | STO-3G | −7.8811184639 | −7.8811184797 | 1.6e-08 | PASS |
| water_cas44_sto3g | CAS(4e,4o) | STO-3G | −74.9760171635 | −74.9760171760 | 1.2e-08 | PASS |
| water_cas44_631g | CAS(4e,4o) | 6-31G | −75.9998609866 | −75.9998609785 | 8.1e-09 | PASS |
| water_cas44_ccpvdz | CAS(4e,4o) | cc-pVDZ | −76.0440109036 | −76.0440109052 | 1.6e-09 | PASS |
| water_cas44_b1 | CAS(4e,4o) | STO-3G | −74.5856164513 | −74.5856163677 | 8.4e-08 | PASS |
| ethylene_casscf_321g | CAS(2e,2o) | 3-21G | −77.5145223959 | −77.5145223872 | 8.7e-09 | PASS |
| ethylene_casscf_321g_nroot2 | CAS(2e,2o) | 3-21G | −77.5145223959 | −77.5145223872 | 8.7e-09 | PASS |
| ethylene_casscf_ccpvdz | CAS(2e,2o) | cc-pVDZ | −77.9524856209 | −77.9524855977 | 2.3e-08 | PASS |
| water_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | −74.7751378317 | −74.7751377977 | 3.4e-08 | PASS |
| ethylene_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | −77.0034974774 | −77.0034974301 | 4.7e-08 | PASS |

Note: PySCF references for water and ethylene were updated to use the
hcore-start converged values (both codes agree from this starting point).
The water SA-2 SAD-start minimum (−74.7877865139 Eh) is now matched in Planck
when uphill acceptance is enabled (`mcscf_accept_uphill .true.`) — see P3.

---

## Remaining Work (Priority Order)

### P0: SA stationarity assertion in the regression runner — DONE (2026-05-01)

Two SA cases now gate on `casscf_sa_gnorm ≤ 1e-5` via the existing
`metric_le` check type, alongside `casscf_total_energy` against the PySCF
reference:

- `water_casscf_sa2_sto3g` — water CAS(4,4)/STO-3G, 2 roots; final `sa_g ≈ 1.6e-08`.
- `ethylene_casscf_sa2_sto3g` — ethylene CAS(4,4)/STO-3G, 2 roots, D2d; final `sa_g ≈ 3.9e-07`.

A future polish item is asserting that the final macro iteration did not
fire the plateau escape path (no `[WRN]` about plateau). This would need a
`files_not_contain` (or `not_contains_in_last_n_lines`) check type added to
the runner.

**File:** `tests/regression_cases.json`

---

### P1: True SA orbital Hessian action — DONE (2026-04-08)

`delta_g_sa_action` (`orbital.cpp:241`) is now the production Hessian-vector
product: central-difference of `fixed_ci_orbital_gradient` at ±ε·R, falling
back to the diagonal energy-denominator model only when context is incomplete.
All call sites in `response.cpp` use `delta_g_sa_action` directly.
`matrix_free_hessian_action` is a thin backward-compat alias.

FD-check coverage added to `tests/casscf_internal.cpp` (270 lines).

---

### P2: Optimizer second simplification pass

`numeric-newton` is still a production escape hatch for spaces with ≤64 pairs.
P1 is now stable, so the shared-κ solve should handle these without a separate
FD-Newton path.

**Deliverables:**
- Demote `numeric-newton` to debug-only behind `mcscf_debug_numeric_newton`.
- Remove per-root candidates and pair probes from the stagnation family; keep
  `sa-diag-fallback` as the sole explicit fallback.
- Ensure every step label in the transcript uniquely identifies the path taken.

**Gate:** all 11 gate cases continue to pass; stagnation log is cleaner.

---

### P3: Water SA-2 SAD-start validation — CLOSED (2026-05-02)

We keep **both** SAD-start fixtures in regressions:

1. `water_cas44_sto3g_sa2_sad.hfinp` (original behavior, no uphill)
   - converges to **−74.7751377977 Eh** (upper/local SA basin).
2. `water_cas44_sto3g_sa2_sad_uphill.hfinp` (`mcscf_accept_uphill .true.`)
   - converges to **−74.7877864784 Eh**, matching the PySCF SAD-start value
     **−74.7877865139 Eh** within `3.6e-08 Eh` (deeper basin).

Both are gated in `tests/regression_cases.json` as:
- `water_casscf_sa2_sto3g_sad_guess`
- `water_casscf_sa2_sto3g_sad_guess_uphill`

### Why both are kept

- They validate **two intentional optimizer modes** (strict monotone vs
  uphill-enabled basin escape), both of which are currently supported.
- They protect against accidental regressions that would either:
  - break the historical monotone landing (`−74.7751377977 Eh`), or
  - break the PySCF-matching uphill landing (`−74.7877864784 Eh`).
- They provide a stable A/B harness for future optimizer changes, so we can
  distinguish "basin policy changed" from "numerical bug introduced".

### Learnings

- The water SA-2 SAD case is genuinely **multi-basin** under the current
  orbital parameterization and trust-region controls.
- The core blocker was not Hessian quality alone; it was **acceptance policy**:
  allowing bounded uphill steps enables the barrier-crossing move.
- Matching PySCF on this case requires not just AH/CIAH machinery, but also a
  compatible step-acceptance strategy.
- Keeping both fixtures codifies this: same method family, different accepted
  basin depending on acceptance mode.

---

### P4: Invariance and robustness test coverage

Still missing from the unit harness:
- Active-active rotation invariance: rotate within the active block after
  convergence, verify energy is unchanged and `g_orb` active-active block
  is zero.
- Anti-symmetry checks: `g_orb[p,q] = −g_orb[q,p]`, `kappa[p,q] = −kappa[q,p]`.
- Root-crossing stress test: geometry distortion near an avoided crossing;
  verify root tracking does not swap roots mid-optimization.
- Checkpoint/restart consistency: store CASSCF MO coefficients, reload, verify
  energy and gradient reproduce to 1e-10 Eh.
- CI-response restart and truncation cases at larger CI dimension (≥100 dets).

---

## What Not To Do

- Do not add more heuristic step mixing to the production path.
- Do not reopen active-space enumeration or bitstring guard work unless a new
  regression appears.
- Do not overload `_cas_mo_coefficients` with another orbital representation.
- Do not change PySCF reference energies without re-running the PySCF scripts
  and documenting why the reference changed.
- P1 (Hessian upgrade) is done; P2 (optimizer simplification) is now unblocked.

---

## Key Files

| File | Purpose |
|---|---|
| `src/post_hf/casscf/casscf.cpp` | Driver, convergence gate, candidate screen, acceptance merit |
| `src/post_hf/casscf/casscf_utils.h` | Shared single-root helpers and fermionic operator utilities |
| `src/post_hf/casscf/response.cpp` | `solve_sa_coupled_orbital_ci_step`, CI response RHS |
| `src/post_hf/casscf/orbital.cpp` | `delta_g_sa_action` (FD Hessian), `hessian_action` (diagonal fallback), gradient |
| `src/post_hf/casscf/rdm.cpp` | 1-RDM, 2-RDM, bilinear 2-RDM accumulation |
| `src/post_hf/casscf/ci.cpp` | Davidson CI driver, direct sigma |
| `src/post_hf/casscf/strings.cpp` | `select_active_orbitals`, `reorder_mo_coefficients`, CI string algebra |
| `tests/casscf_internal.cpp` | Internal unit and invariance coverage |
| `tests/pyscf/run_all.py` | PySCF-backed energy validation suite |
| `tests/run_regressions.py` | Binary output regression runner (parses SA metrics) |
| `tests/regression_cases.json` | Regression case specs (needs SA gnorm assertions — see P0) |
| `tests/benchmarks/casscf/pyscf_reference/` | PySCF reference inputs and energy table with diagnostic notes |

---

## Regression Loop

Run after every solver-facing change:

```bash
# Internal unit tests
cmake --build build --target planck-casscf-internal -j4
./build/planck-casscf-internal

# PySCF validation suite (requires .venv in tests/pyscf/)
tests/pyscf/.venv/bin/python tests/pyscf/run_all.py

# Fixture regressions
python3 tests/run_regressions.py
```

Do not call work done if one case improves while another regresses.

---

## Appendix A: Why `numeric-newton` wins for small water active spaces

For water CAS(4,4)/STO-3G the active space exhausts all 7 basis functions
(3 core + 4 active + 0 virtual). All 12 non-redundant pairs are core–active.

The shared-κ coupled solve uses a diagonal OO-block preconditioner. The
core–active block has the strongest off-diagonal Hessian coupling in CASSCF
(rotating a core orbital mixes directly into the active space). The diagonal
model systematically underestimates this coupling.

`numeric-newton` constructs the exact 12×12 fixed-CI Hessian column-by-column
via central finite differences (24 extra full evaluations per macroiteration —
negligible for 7-function STO-3G). This captures full off-diagonal coupling
and converges quadratically:

```
Macro 1  candidate=numeric-newton  sa_g=1.12e-02
Macro 2  candidate=numeric-newton  sa_g=1.24e-02
Macro 3  candidate=numeric-newton  sa_g=1.22e-02
Macro 4  candidate=numeric-newton  sa_g=7.55e-03
Macro 5  candidate=numeric-newton  sa_g=1.61e-03
Macro 6  candidate=numeric-newton  sa_g=5.13e-05
Macro 7  candidate=numeric-newton  sa_g=4.59e-08
```

For larger systems (ethylene CAS(4,4)/cc-pVDZ: 110 pairs), the FD cost is
prohibitive and `numeric-newton` is disabled; `sa-coupled` is the sole path.

The 64-pair gate (`numeric_newton_pair_limit`) is set at `casscf.cpp:650`.

| System | Basis | Pairs | numeric-newton active? |
|---|---|---|---|
| Water CAS(4,4) | STO-3G | 12 | yes |
| Water CAS(4,4) | 6-31G | 32 | yes |
| Water CAS(4,4) | cc-pVDZ | 62 | yes |
| Ethylene CAS(4,4) | STO-3G | 44 | yes |
| Ethylene CAS(4,4) | cc-pVDZ | 110 | no |

---

## Appendix B: Fixed Cayley rotation sign bug (historical)

**Symptom:** orbital gradient plateau at ~1e-2, `dE = 0` for all subsequent
macroiterations. Every trial step rejected.

**Root cause:** `apply_orbital_rotation` used the Cayley map
`U = (I + κ/2)⁻¹(I − κ/2)` ≈ `exp(−κ)`, while the optimizer constructs
descent steps for the convention `C' = C exp(+κ)`. The rotation direction was
inverted, so the line search always tested uphill and rejected every candidate.

**Fix:** flipped to `U = (I − κ/2)⁻¹(I + κ/2)` ≈ `exp(+κ)`, consistent with
the optimizer parameterization. Added accepted-state bookkeeping so the
logged gradient and convergence test are tied to the accepted state.

**Affected inputs at time of fix:** `water_cas22.hfinp`, `water_cas44_b1.hfinp`.
Both now converge correctly.
