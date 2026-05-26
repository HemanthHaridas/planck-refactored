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

`src/gradient/gradient.cpp` computes ∂E/∂R for RHF, UHF, RMP2, and UMP2.

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

ROHF analytic gradients are still explicitly unimplemented, so ROHF frequency and geometry-optimization paths fail once they need the gradient.

## Spherical-basis gradients, geomopt, and frequencies (RHF / UHF)

The Cartesian-basis derivative integral engine in `src/integrals/os.cpp` is reused unchanged for the spherical basis via a transform-at-the-skin pattern:

1. SCF in spherical mode stores the density `P_sph` and energy-weighted density `W_sph` in the `(2L+1)`-per-shell basis.
2. At the gradient entry point, each AO matrix is lifted back to the Cartesian basis with `M_cart = Cᵀ M_sph C`, where `C` is the block-diagonal `[n_sph × n_cart]` transform held on `Basis._cart_to_sph`. Helper: `BasisFunctions::lift_density_sph_to_cart` in `src/basis/spherical.{h,cpp}`. In Cartesian mode the lift is a no-op pass-through.
3. The Cartesian derivative kernel then contracts `P_cart` / `W_cart` against the unchanged Cartesian shell-pair derivative blocks. The output is a real-space `natoms × 3` matrix, so nothing transforms back.

The lift is exact: for any Cartesian operator `X_cart` emitted by the integral engine, `tr(M_sph · X_sph) = tr(M_cart · X_cart)`. This is the energy-invariance contract, pinned by the `planck-spherical-density-lift` unit test in `tests/spherical_density_lift.cpp` for L = 0…4.

### Geomopt + freq inner-loop rebuild

The geometry-optimization (`src/opt/geomopt.cpp`) and semi-numerical Hessian (`src/freq/hessian.cpp`) inner loops each rebuild the basis from scratch at every displaced geometry. In the spherical basis this requires two non-obvious steps that the driver does once at startup:

1. **`_cart_to_sph` row-renormalization** against the new `S_cart`. The transform directions depend only on `L` (geometry-independent), but the row scaling depends on the Cartesian overlap, which moves with geometry. A stale transform silently corrupts every spherical AO matrix element.
2. **`_overlap = C·S·Cᵀ`**, **`_hcore = C·(T+V)·Cᵀ`** — store the spherical 1e matrices, not the Cartesian ones, so SCF sees the right working basis.

Both steps are packaged in `HartreeFock::SCF::rebuild_basis_dependent_state` (`src/scf/working_state.{h,cpp}`), which the geomopt and freq inner loops call after each `read_gbs_basis`. In Cartesian mode the helper degrades to the original `build_shellpairs → _compute_1e → store as _overlap, _hcore` sequence, so the Cartesian regression set is byte-identical. The post-geomopt "Final Symmetry SCF" block in `src/driver.cpp` also calls the same helper for the same reason. Both inner loops were updated to use `working_nbasis()` instead of `nbasis()` for the `DataSCF::initialize` and `set_scf_mode_auto` calls — these were a latent bug in spherical mode where SCF state was being sized to the Cartesian count.

Contract test: `planck-working-state-rebuild` (`tests/working_state_rebuild.cpp`) verifies `diag(_overlap) = 1` and idempotency for both Cartesian water/STO-3G and spherical water/6-31g* (the d-shell case that catches missing normalization).

### Driver gate

`src/driver.cpp` admits `Gradient`, `GeomOpt`, `Frequency`, `GeomOptFrequency`, and `ImaginaryFollow` for RHF and UHF in the spherical basis. ROHF analytic gradients (unimplemented Cartesian-side too) and MP2/UMP2 gradients (need the response-machinery audit before the same lift adapter can be wired in) remain rejected explicitly for every gradient-consuming workflow; each rejection names the specific feature.

### Regression coverage

Positive PySCF-gated cases (water/6-31g*, exercises the d-shell transform):

| ID | What it tests | PySCF tolerance |
|---|---|---|
| `water_rhf_spherical_gradient_631gd` | single-shot analytic gradient | `1e-7` Eh / Ha-Bohr |
| `water_rhf_spherical_geomopt_631gd` | IC-BFGS to convergence, post-opt symmetry SCF | `1e-7` Eh, `≤5e-4` Ha/Bohr final force, point group `C2v` |
| `water_rhf_spherical_freq_631gd` | semi-numerical Hessian, three vibrational frequencies | `1.0` cm⁻¹ per mode |
| `water_rhf_spherical_geomoptfreq_631gd` | sequenced opt+freq at the converged geometry | `1e-7` Eh, `2.0` cm⁻¹ per mode |

Negative boundary markers:

| ID | Rejected by | Asserts |
|---|---|---|
| `water_rmp2_spherical_gradient_rejected` | driver spherical gate | `exit 1` + "MP2 analytic gradient is not yet supported" |
| `water_rmp2_spherical_geomopt_rejected` | same, via gradient-needing-workflow propagation | same message |

## Geometry Optimization

`src/opt/geomopt.cpp` orchestrates two optimizers:

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

`src/freq/hessian.cpp` — semi-numerical Hessian:
1. Displace each atom in ±x, ±y, ±z by δ = 0.001 bohr
2. Compute analytic gradient at each displaced geometry
3. Finite-difference second derivative: H_ij = (g_+(δ) - g_-(δ)) / 2δ
4. Diagonalize mass-weighted Hessian → normal modes + frequencies
5. Imaginary frequencies flagged (negative eigenvalues)

## ImaginaryFollow

`CalculationType::ImaginaryFollow` in `src/driver.cpp` follows the lowest imaginary frequency mode downhill using the semi-numerical Hessian eigenvector. The HF/post-HF driver supports this workflow; the DFT driver currently reports imaginary-mode following as unimplemented.

## Key Files

- `src/gradient/gradient.cpp` — analytic gradient (RHF/UHF/RMP2/UMP2)
- `src/post_hf/mp2_gradient.cpp` + `mp2_gradient.h` — UMP2 gradient intermediates
- `src/post_hf/rhf_response.cpp` + `rhf_response.h` — RHF Z-vector / CPHF machinery shared with RMP2 gradient
- `src/post_hf/uhf_response.cpp` + `uhf_response.h` — UHF response machinery used by UMP2 gradients
- `src/opt/geomopt.cpp` + `opt/intcoords.cpp` — L-BFGS and IC-BFGS optimizers
- `src/freq/hessian.cpp` — semi-numerical Hessian and vibrational analysis
