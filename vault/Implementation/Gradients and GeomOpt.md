---
name: Gradients and GeomOpt
description: Analytic gradient implementation, geometry optimization (L-BFGS, IC-BFGS), frequencies
type: implementation
priority: medium
include_in_claude: true
tags: [gradient, geomopt, lbfgs, hessian, frequency]
---

# Analytic Gradients and Geometry Optimization

## Analytic Gradients

`src/gradient/gradient.cpp` — computes ∂E/∂R for RHF, UHF, **RMP2, and UMP2**.

Components:
- One-electron gradient: ∂H_core/∂R (kinetic + nuclear attraction derivatives)
- Two-electron gradient: ∂ERI/∂R (Obara-Saika derivative integrals)
- Nuclear repulsion gradient: ∂V_nn/∂R
- Orbital response (Pulay terms): couples density matrix response to basis function derivatives

For MP2: requires orbital response (Z-vector / coupled-perturbed HF) to handle the response of the HF orbitals to the nuclear displacement.

## UMP2 Gradient (commit 22c0645)

`src/post_hf/mp2_gradient.{cpp,h}` — spin-resolved UMP2 gradient intermediates for canonical UHF references:
- Same-spin and opposite-spin MP2 amplitude handling
- Correlated alpha/beta density corrections
- Spin-summed energy-weighted density
- Explicit AO pair-density contributions

Wired into the gradient driver (`src/driver.cpp`) so `correlation ump2` with `calculation gradient` now produces the correlated nuclear gradient instead of exiting as unimplemented. Regression: `water_radical_cation_ump2_gradient_smoke`.

## Geometry Optimization

`src/opt/opt.cpp` — two optimizers selectable:

### Cartesian L-BFGS
- Standard L-BFGS in Cartesian displacement coordinates
- History size configurable
- Simple but can be inefficient for bond angles/torsions

### IC-BFGS (Internal Coordinate BFGS)
- Builds internal coordinates: bonds, angles, dihedrals
- BFGS update in internal coordinate space
- Backtransformation from internal → Cartesian displacements
- Better convergence for molecular geometry optimization

Convergence criteria: max force < threshold AND RMS displacement < threshold (ORCA/Gaussian defaults).

## Frequencies (Vibrational Analysis)

`src/freq/freq.cpp` — semi-numerical Hessian:
1. Displace each atom in ±x, ±y, ±z by δ = 0.001 bohr
2. Compute analytic gradient at each displaced geometry
3. Finite-difference second derivative: H_ij = (g_+(δ) - g_-(δ)) / 2δ
4. Diagonalize mass-weighted Hessian → normal modes + frequencies
5. Imaginary frequencies flagged (negative eigenvalues)

## ImaginaryFollow

`CalculationType::ImaginaryFollow` — follows the lowest imaginary frequency mode downhill to find transition state or escape a saddle point. Used in DFT driver as well.

## Key Files

- `src/gradient/gradient.cpp` — analytic gradient (RHF/UHF/RMP2/UMP2)
- `src/post_hf/mp2_gradient.cpp` + `mp2_gradient.h` — UMP2 gradient intermediates
- `src/post_hf/rhf_response.cpp` + `rhf_response.h` — RHF Z-vector / CPHF machinery shared with RMP2 gradient
- `src/opt/geomopt.cpp` + `opt/intcoords.cpp` — L-BFGS and IC-BFGS optimizers
- `src/freq/hessian.cpp` — semi-numerical Hessian and vibrational analysis
