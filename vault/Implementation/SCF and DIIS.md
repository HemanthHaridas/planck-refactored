---
name: SCF and DIIS
description: RHF/UHF SCF loop implementation, DIIS extrapolation, convergence
type: implementation
priority: medium
include_in_claude: true
tags: [scf, rhf, uhf, diis, convergence]
---

# SCF and DIIS

## Loop Structure

Standard Roothaan-Hall loop:
1. Build H_core = T + V_ne (one-electron integrals, computed once)
2. Form orthogonalizer X from S (canonical or symmetric)
3. Build initial guess density P (core Hamiltonian diagonalization or SAD)
4. Iterate:
   - Build J (Coulomb) and K (exchange) from P + ERIs
   - F = H_core + 2J - K (RHF) or F_α = H_core + J - K_α (UHF)
   - DIIS: push (F, e=FPS-SPF) onto queue, extrapolate
   - F' = X†FX → diagonalize → {C', ε}
   - C = XC', recompute P from occupied MOs
   - Check ΔE and ‖ΔP‖_max convergence

## SAD Guess

Added after commit 733fb31. Projects stored minimal-basis atomic densities onto the working basis to form a reasonable starting P without solving H_core first. Greatly improves convergence for heavier atoms and larger bases.

## DIIS

DIIS queue size configurable (default 8, set via `diis_size` keyword). Error vector e = FPS - SPF in the AO basis. Builds B matrix (error inner products), solves for Lagrange weights, extrapolates Fock matrix. Falls back to straight iteration if B is singular.

## UHF

Uses two `SpinChannel` objects (alpha, beta). Each has its own Fock matrix, density, MO coefficients. J built from total density P_α + P_β; K built separately per spin. DIIS runs independently per spin channel but uses the same queue index.

## Key Files

- `src/scf/scf.cpp` — main loop
- `src/scf/diis.cpp` — DIIS implementation
- `src/integrals/os.cpp` — ERI computation (Obara-Saika)
- `src/integrals/rys.cpp` — ERI computation (Rys quadrature)
