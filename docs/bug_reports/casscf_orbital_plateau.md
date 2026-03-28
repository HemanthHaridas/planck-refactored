# Bug Report: CASSCF Orbital Gradient Plateau from Reversed Cayley Rotation

## Summary

The second-order CASSCF optimizer could stall with a nonzero orbital gradient
while reporting `dE = 0` for every subsequent macroiteration. The immediate
symptom was that no trial orbital update was ever accepted, so the accepted
state stayed frozen and the printed gradient never changed.

The primary implementation bug was a sign inconsistency in the orbital rotation
map: the optimizer generated descent directions for the parameterization
`C' = C exp(kappa)`, but the Cayley transform applied approximately
`exp(-kappa)`. This inverted AH/Newton orbital steps at application time, so
the line search repeatedly tested the wrong direction and rejected all
candidate steps.

## Affected Cases

- `inputs/water_cas22.hfinp`
- `inputs/water_cas44_b1.hfinp`

Observed pre-fix behavior:

- Water CAS(2,2): gradient plateau near `7.643e-03`, `dE = 0` thereafter
- Water CAS(4,4), target irrep B1: gradient plateau near `7.903e-03` as
  reported by the user; in the current code path before the final fix the run
  reproduced a larger frozen residual around `4.517e-01`, again with `dE = 0`
  and no accepted macro step

## Root Cause

### 1. Reversed orbital rotation

In `src/post_hf/casscf.cpp`, `apply_orbital_rotation()` used the Cayley map

`U = (I + kappa/2)^(-1) (I - kappa/2)`

which is a second-order approximation to `exp(-kappa)`.

The rest of the optimizer, including the AH step and Newton-like step builders,
assumes the update convention

`C' = C exp(+kappa)`

and therefore constructs descent steps with the usual negative-gradient sign.
Because the actual applied rotation had the opposite sign, the optimizer
evaluated uphill or otherwise inconsistent trial states, causing the
backtracking loop to reject every candidate.

### 2. Plateau masking from accepted-state bookkeeping

Once every trial step was rejected, the accepted state never changed. That made
the printed `dE` exactly zero on later macroiterations even though the orbital
gradient remained nonzero. The accepted-state residual handling also needed to
be tightened so the logged gradient and convergence test were tied to the
accepted state instead of a stale pre-step state.

This was not the fundamental cause of the stall, but it made the failure mode
look like a false convergence plateau instead of a rejected-step loop.

## Fix

### Correct the Cayley sign

`apply_orbital_rotation()` now uses

`U = (I - kappa/2)^(-1) (I + kappa/2)`

which is consistent with `exp(+kappa)` and with the optimizer’s orbital
parameterization.

### Keep convergence metrics tied to the accepted state

The macroiteration logic keeps the accepted `McscfState`, allows near-flat
energy acceptance when the orbital gradient is reduced, and reports the
gradient from the accepted state’s `g_orb`.

## Validation

After the fix:

- `./build/hartree-fock inputs/water_cas22.hfinp`
  - converges in 1 macroiteration
  - `E(CASSCF) = -74.9641867710 Eh`
  - final orbital gradient `1.526e-06`
- `./build/hartree-fock inputs/water_cas44_b1.hfinp`
  - converges in 8 macroiterations
  - `E(CASSCF) = -74.2879452324 Eh`
  - final orbital gradient `1.315e-07`
- `./build/hartree-fock inputs/h2_cas22.hfinp`
  - still converges
  - `E(CASSCF) = -1.1372744062 Eh`

## Files Changed

- `src/post_hf/casscf.cpp`
- `docs/bug_reports/casscf_orbital_plateau.md`

## Notes

- The local binary was relinked manually after recompiling `casscf.cpp`
  because the full `cmake --build build` path still runs into an unrelated
  offline `libmsym` fetch/update issue.
- The second-order microiteration model still uses a frozen-state approximation
  inside each macro step, so there is room for future robustness improvements.
  However, the plateau reported here was caused by the reversed orbital
  rotation sign.
