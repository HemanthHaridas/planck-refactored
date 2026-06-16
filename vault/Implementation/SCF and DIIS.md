---
name: SCF and DIIS
description: RHF/UHF/ROHF SCF loop implementation, guesses, DIIS, stability follow-up
type: implementation
priority: medium
include_in_claude: true
tags: [scf, rhf, uhf, rohf, diis, convergence]
---

# SCF and DIIS

## Loop Structure

Standard SCF flow:
1. Build H_core = T + V_ne (one-electron integrals, computed once)
2. Form orthogonalizer X from S (canonical or symmetric)
3. Build initial guess density P (core Hamiltonian diagonalization or SAD)
4. Iterate:
   - Build J (Coulomb) and K (exchange) from P + ERIs
   - Build the effective Fock operator for the active reference:
     - RHF: `F = H_core + 2J - K`
     - UHF: `F_α = H_core + J - K_α`, `F_β = H_core + J - K_β`
     - ROHF: Roothaan-type effective Fock assembled from α/β blocks
   - DIIS: push (F, e=FPS-SPF) onto queue, extrapolate
   - F' = X†FX → diagonalize → {C', ε}
   - C = XC', recompute P from occupied MOs
   - Check ΔE and ‖ΔP‖_max convergence

`src/scf/scf.cpp` owns the RHF/UHF/ROHF implementations; `src/scf/sad.cpp` owns the SAD initial guess.

## SAD Guess

Added after commit 733fb31. Projects stored minimal-basis atomic densities onto the working basis to form a reasonable starting P without solving H_core first. Greatly improves convergence for heavier atoms and larger bases.

The checkpoint system also supports `guess density` and `guess full`, including cross-basis projection through the shared checkpoint machinery.

## DIIS

DIIS queue size configurable (default 8, set via `diis_dim`). Error vector `e = FPS - SPF` in the AO basis. Builds the B matrix from error inner products, solves the Pulay system for weights, and extrapolates the Fock matrix. `diis_restart` can clear the subspace when the error spikes.

## UHF

Uses two `SpinChannel` objects (alpha, beta). Each has its own Fock matrix, density, MO coefficients. J built from total density P_α + P_β; K built separately per spin. DIIS runs independently per spin channel but uses the same queue index.

## ROHF

`run_rohf` in `src/scf/scf.cpp` uses:
- separate α/β densities for the occupied spaces,
- a shared spatial-orbital diagonalization path,
- a Roothaan effective Fock for DIIS and orbital updates,
- an orbital reordering helper to keep closed/open/virtual spaces in Aufbau order.

ROHF is available for SCF and checkpoint/restart workflows, but downstream support is narrower than RHF/UHF: FCI works, while most other post-HF, gradients, and PCM paths still reject ROHF explicitly.

## Stability Analysis

`src/scf/stability.{cpp,h}` implements post-SCF orbital-Hessian checks for converged RHF and UHF references:
- RHF: real-internal singlet, complex-internal singlet, and external triplet channels
- UHF: spin-conserving internal plus a diagonal-approximation external GHF check

With `stability_follow .true.`, the driver can rotate along the lowest unstable mode and re-run SCF. For RHF external triplet instabilities, the restart promotes the reference to UHF before re-converging.

## Key Files

- `src/scf/scf.cpp` — RHF/UHF/ROHF loops
- `src/scf/sad.cpp` + `sad.h` — SAD guess
- `src/scf/diis.cpp` — DIIS implementation
- `src/scf/stability.cpp` + `stability.h` — wavefunction stability analysis / follow
- `src/integrals/os.cpp` — ERI computation (Obara-Saika)
- `src/integrals/rys.cpp` — ERI computation (Rys quadrature)
