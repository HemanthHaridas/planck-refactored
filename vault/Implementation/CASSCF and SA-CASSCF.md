---
name: CASSCF and SA-CASSCF
description: Full CASSCF/SA-CASSCF/RASSCF implementation details, optimizer, validation
type: implementation
priority: high
include_in_claude: true
tags: [casscf, sa-casscf, rasscf, mcscf, ci, active-space]
---

# CASSCF / SA-CASSCF / RASSCF

## Status: 11/11 PySCF gate cases passing (2026-04-08)

## Key Files

| File | Responsibility |
|------|---------------|
| `src/post_hf/casscf/casscf.cpp` | Macro-iteration loop, convergence gating, output |
| `src/post_hf/casscf/ci.cpp` | FCI / CI diagonalization (direct-sigma) |
| `src/post_hf/casscf/orbital.cpp` | Orbital rotation, `matrix_free_hessian_action` |
| `src/post_hf/casscf/rdm.cpp` | 1-RDM and 2-RDM computation |
| `src/post_hf/casscf/response.cpp` | SA coupled solve (`solve_sa_coupled_orbital_ci_step`) |
| `src/post_hf/casscf/strings.cpp` | Active space setup, `select_active_orbitals`, `IrrepCount` |

## Convergence Gating

Convergence gated on `‖g_SA‖∞ < tol` where `g_SA = Σ_I w_I g_I` (weighted sum of per-root orbital gradients). Not per-root. Function: `build_weighted_root_orbital_gradient`.

Iteration table prints:
- "SA Grad": the gating quantity
- "MaxRootG": diagnostic (max over roots, not used for convergence)

## Macro-Optimizer Cascade

**Normal iteration:**
1. `sa-coupled` — shared-κ SA coupled solve (primary)
2. `sa-grad-fallback` — diagonal gradient step backup

**Under stagnation** (`stagnation_streak ≥ 2`):
3. `numeric-newton` — exact FD Hessian (only when npairs ≤ 64)
4. `sa-diag-fallback`, per-root candidates, single-pair probes

**Step scales tried**: `{1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625}`
**Merit function**: `E_cas + 0.1·‖g_orb‖²` (lowest wins)

**Plateau escape**: if stagnation_streak ≥ 2 and both energy and step are flat, convergence is declared with a `[WRN]` line.

## Shared-κ SA Coupled Solve

`solve_sa_coupled_orbital_ci_step` in `response.cpp`:
- Holds one orbital step κ shared across all state-averaged roots
- Solves each root's CI response to that shared κ
- Iterates on SA orbital residual: `g_SA + H_oo κ + Σ_I w_I G_oc c1_I(κ)`
- Acceptance merit uses `trial.gnorm` (SA gradient norm), not per-root screens

## CI Response RHS

Default mode: `ResponseRHSMode::ExactOrbitalDerivative` — exact CI-response RHS from orbital derivative. Debug/shortcut: `mcscf_debug_commutator_rhs` keyword switches to commutator-only approximation.

## Orbital Hessian

`matrix_free_hessian_action` in `orbital.cpp` delegates to `delta_g_sa_action`: full finite-difference fixed-CI Hessian-vector product (rotates MOs by ±ε·R, central-difference of `fixed_ci_orbital_gradient`). Falls back to diagonal energy-denominator model (M3) only when `OrbitalHessianContext` is incomplete.

## Active Space Setup

`select_active_orbitals` in `strings.cpp`:
- Reads explicit `core_irrep_counts` / `active_irrep_counts` / `mo_permutation` keywords
- Falls back to energy-sorted inference
- Identity fallback when symmetry is absent
`reorder_mo_coefficients` applies the permutation before the MCSCF loop.

## Historical Bugs Fixed

- **Reversed Cayley sign** (`apply_orbital_rotation`): was applying `exp(-κ)` instead of `exp(+κ)`. Caused line search to test uphill directions and reject every candidate. Now fixed.
- **`guess hcore` + `use_symm true` → wrong RHF**: d2d RHF branch preservation fix (commit 46aa199).

## Open Work

- SA stationarity assertion: `tests/regression_cases.json` has no `lte` check on `casscf_sa_gnorm` for any SA case — add `{ "metric": "casscf_sa_gnorm", "type": "lte", "value": 1e-5 }` to all SA entries

## RASSCF

Active space partitioned into RAS1 (hole excitations allowed), RAS2 (full CAS), RAS3 (particle excitations allowed). Built on top of CASSCF orbital machinery with modified CI string generation.
