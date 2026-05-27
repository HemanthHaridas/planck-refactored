---
name: Completion Status
description: Canonical summary of what is implemented and validated in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, completion, validated, canonical]
---

# Completion Status

Last updated: 2026-05-27

This is the canonical completion-status document for the repository.
Subsystem handoff, plan, benchmark, and fix-summary notes may still exist for
historical design context, but they are no longer the source of truth for
"what is done". Use this file together with `vault/Status/Open Work.md`.

## Fully Implemented and Validated

### HF / SCF core

- RHF, UHF, and ROHF SCF with DIIS (`src/scf/scf.cpp`)
- H_core and SAD initial guesses
- Same-basis checkpoint restart and density restart
- Symmetry detection, MO irrep labeling, and SAO-blocked Fock diagonalization
- Wavefunction stability analysis for RHF/UHF, plus optional instability following
- Mulliken, Lowdin, Mayer, dipole, quadrupole, and related property reporting
- PCM solvation for single-point RHF/UHF runs

### Direct SCF and full point-group symmetry

- Obara-Saika and Rys direct Fock engines, with auto-dispatch by angular momentum
- Full point-group ERI reduction for direct RHF/UHF in the Cartesian basis
- Full point-group ERI reduction for direct RHF/UHF in the spherical-harmonic basis
- Metric-correct spherical group operators
  `O_sph = S_sph^{-1} (C S_cart O_cart C^T)`
- Focused validation of the full-symmetry machinery:
  `planck-group-operations`, `planck-fock-symmetrization`,
  `planck-symm-fock-equivalence`
- Committed direct full-symmetry regression ladder through Td, including spherical
  NH3/CH4 cases
- Persisted full-symmetry skeleton ERI across SCF iterations (C1), so the
  density-independent skeleton is built once and reused during the SCF cycle
- Monomial-operator fast path in full-group symmetrization for operations that
  reduce to signed AO permutations

### Spherical-harmonic basis support

- Spherical single-point RHF, UHF, and ROHF
- Conventional and direct SCF in the spherical working basis
- Spherical property reporting
- Same-basis and cross-basis checkpoint restart in the spherical working basis
- Spherical MP2, CASSCF, RASSCF, FCI, FCIDUMP export, and coupled-cluster energies
- Spherical analytic gradients for RHF and UHF, via the
  `lift_density_sph_to_cart` adapter that maps the spherical SCF density and
  energy-weighted density back to the Cartesian basis so the Cartesian
  derivative-integral engine can be reused unchanged (the energy is invariant
  under the basis change, so the lift carries no approximation). PySCF-validated
  to ~1e-7 Ha/Bohr on water/6-31g* and OH/STO-3G
- Spherical geometry optimization (IC-BFGS + Cartesian L-BFGS) and
  semi-numerical frequencies for RHF and UHF, plus geomopt+frequency and
  imaginary-mode following. Driven by a shared
  `HartreeFock::SCF::rebuild_basis_dependent_state` helper that re-runs the
  spherical `_cart_to_sph` row-normalization and the `C·(T+V)·Cᵀ` working-basis
  lift at every displaced geometry, keeping the geomopt/freq inner loops in
  lockstep with the driver's startup setup. PySCF-validated to <0.1 cm⁻¹ on
  water/6-31g* vibrational frequencies and ~6e-8 Eh on the IC-BFGS optimized
  energy
- Spherical symmetry support at both the SAO-blocking level and the full-group
  direct-SCF level
- Regression coverage for the landed spherical single-point feature set plus
  RHF gradient, geomopt, frequency, geomopt+frequency cases (all PySCF-gated;
  see `tests/regression_cases.json` entries `water_rhf_spherical_{gradient,
  geomopt,freq,geomoptfreq}_631gd`), with unsupported workflows hard-gated
  rather than allowed to return wrong answers (boundary markers:
  `water_rmp2_spherical_{gradient,geomopt}_rejected`)

### Post-HF methods

- RMP2 and UMP2 correlation energies
- Analytic RHF, UHF, RMP2, and UMP2 gradients
- FCI over the full MO space for small RHF/ROHF references
- CASSCF, SA-CASSCF, and RASSCF
- Coupled-cluster support for RCCSD, UCCSD, RCCSDT, UCCSDT, RCCSDTQ
- Arbitrary-order RCC solver via ccgen-generated residuals
- Tensor-backed and determinant-space coupled-cluster paths, including the
  optimized RCCSDT warm-start route

### CASSCF / SA-CASSCF status

- Shared-kappa state-averaged coupled orbital/CI solve is the primary production path
- Exact CI-response RHS is the default
- FD-based SA orbital Hessian action (`delta_g_sa_action`) is implemented and wired
- Active-integral-cache transform is landed and benchmarked
- Per-root SA total-energy reporting is fixed
- SA diagnostics are parsed by the regression runner
- SAD-start uphill-enabled water SA-2 basin is validated and retained as a separate
  regression mode

### Gradients, optimization, and frequencies

- Analytic gradients for RHF, UHF, RMP2, and UMP2
- Geometry optimization in Cartesian and internal coordinates
- Semi-numerical Hessian / vibrational frequencies
- Imaginary-frequency following
- Constrained geometry optimization

### DFT

- RKS and UKS
- LDA, GGA, global hybrids, and arbitrary libxc functional selection
- Range-separated libxc functionals for single-point, analytic-gradient,
  geometry-optimization, frequency, and geomopt+frequency workflows
- Double-hybrid libxc functionals for single-point energies
- Treutler-Ahlrichs radial grid, Lebedev angular grid, and Becke partitioning
- Grid quality levels: Coarse, Normal, Fine, UltraFine
- Single-point PCM solvation for RKS/UKS
- Linear-response TDDFT / Casida and TDA excited states
- DFT single-point, gradient, geometry optimization, frequency, and geomopt+frequency workflows
- DFT checkpoint/restart and symmetry+SAO blocking
- Symmetry-enabled DFT gradient/frame handling fixed by synchronizing
  `_coordinates` to the symmetry-standardized frame before grid construction;
  covered by the `water_dft_hse06_gradient_symm_ultrafine_fd` regression
- HSE06 analytic-gradient validation against both finite differences and PySCF,
  including the long-range exchange contribution and a symmetry-on ultrafine
  finite-difference regression for water

### BSSE / counterpoise

- Ghost atoms, including multiple input syntaxes
- Automated two-fragment SCF-level Boys-Bernardi counterpoise driver
- Per-fragment charge and multiplicity handling
- PySCF-validated He2/cc-pVDZ counterpoise decomposition

### Recent fixes now considered landed

- RMP2 analytic gradient response-path fix, validated against finite differences
  and PySCF
- UMP2 gradient cross-check on the radical-cation path, with no code fix required
- BSSE / ghost-atom infrastructure and CP driver
- Full-symmetry direct-SCF performance improvements: persisted skeleton ERI and
  monomial-group-operator fast path

## CASSCF PySCF Gate Table

Suite status: **11/11 passing**

PySCF version: 2.12.1. All scripts use `mol.cart = True` to match Planck
Cartesian-basis references. Tolerance: `1e-5 Eh`.

| Case | Active space | Basis | PySCF / Eh | Planck / Eh | Delta / Eh | Status |
|---|---|---|---|---|---|---|
| h2_cas22_sto3g | CAS(2e,2o) | STO-3G | -1.1372838345 | -1.1372838351 | 6.0e-10 | PASS |
| lih_cas22_sto3g | CAS(2e,2o) | STO-3G | -7.8811184639 | -7.8811184797 | 1.6e-08 | PASS |
| water_cas44_sto3g | CAS(4e,4o) | STO-3G | -74.9760171635 | -74.9760171760 | 1.2e-08 | PASS |
| water_cas44_631g | CAS(4e,4o) | 6-31G | -75.9998609866 | -75.9998609785 | 8.1e-09 | PASS |
| water_cas44_ccpvdz | CAS(4e,4o) | cc-pVDZ | -76.0440109036 | -76.0440109052 | 1.6e-09 | PASS |
| water_cas44_b1 | CAS(4e,4o) | STO-3G | -74.5856164513 | -74.5856163677 | 8.4e-08 | PASS |
| ethylene_casscf_321g | CAS(2e,2o) | 3-21G | -77.5145223959 | -77.5145223872 | 8.7e-09 | PASS |
| ethylene_casscf_321g_nroot2 | CAS(2e,2o) | 3-21G | -77.5145223959 | -77.5145223872 | 8.7e-09 | PASS |
| ethylene_casscf_ccpvdz | CAS(2e,2o) | cc-pVDZ | -77.9524856209 | -77.9524855977 | 2.3e-08 | PASS |
| water_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | -74.7751378317 | -74.7751377977 | 3.4e-08 | PASS |
| ethylene_cas44_sto3g_sa2 | CAS(4e,4o) SA-2 | STO-3G | -77.0034974774 | -77.0034974301 | 4.7e-08 | PASS |

### CASSCF validation notes

- The committed gate suite is still the clearest validation point for the CASSCF stack.
- The water SA-2 SAD-start uphill-enabled case reaches the PySCF SAD-start basin
  within `3.6e-08 Eh`.
- The baseline monotone SAD-start landing is also intentionally preserved as a
  separate regression because it exercises a different optimizer policy.
