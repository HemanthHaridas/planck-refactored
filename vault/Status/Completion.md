---
name: Completion Status
description: What is fully implemented and validated in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, completion, validated]
---

# Completion Status

Last updated: 2026-05-22

## Fully Implemented and Validated

### HF/SCF
- RHF, UHF, and ROHF SCF with DIIS (`src/scf/scf.cpp`)
- SAD guess (commit 733fb31) and H_core guess
- Symmetry detection + MO irrep labeling (libmsym)
- Mulliken population analysis
- Checkpoint system: same-basis restart + cross-basis Löwdin projection
- Wavefunction stability analysis for RHF/UHF, plus optional instability following (`stability_check`, `stability_follow`)
- PCM solvation for single-point RHF/UHF runs (`src/solvation/pcm.{h,cpp}`)

### Post-HF
- RMP2 and UMP2 correlation energies
- FCI over the full MO space for small RHF/ROHF references
- **CASSCF**: fully implemented, 11/11 PySCF gate cases passing (2026-04-08)
- **SA-CASSCF**: shared-κ coupled solver, exact CI-response RHS, stagnation escape
- **RASSCF**: active space partitioning (RAS1/RAS2/RAS3)
- **Coupled cluster**: RCCSD, UCCSD, RCCSDT, UCCSDT, RCCSDTQ — teaching determinant-space prototypes plus tensor production backends. RCCSDT can also be forced onto the tensor-optimized ccgen warm-start path via `PLANCK_RCCSDT_BACKEND=optimized`. Arbitrary-order RCC solver via ccgen-generated residuals.

### ERI Engine
- Obara-Saika (`integrals/os.cpp`) — primary engine
- Rys quadrature (`integrals/rys.cpp`) — alternative
- Auto-dispatch based on angular momenta
- OpenMP parallelized

### Gradients and Geometry
- Analytic gradients: RHF, UHF, RMP2, **UMP2** (commit 22c0645)
- Geometry optimization: Cartesian L-BFGS + Internal Coordinate IC-BFGS
- Vibrational frequencies: semi-numerical Hessian (finite-difference analytic gradients)
- Imaginary frequency following
- Constrained geometry optimization via `%begin_constraints`

### DFT (`planck-dft` binary)
- RKS and UKS
- LDA (Slater, VWN5), GGA (B88, LYP, PBE, PW91), global hybrids (B3LYP, PBE0), and arbitrary libxc functional selection by name or ID
- Range-separated and double-hybrid libxc functionals for single-point energies (for example HSE06, B2PLYP)
- Treutler-Ahlrichs radial + Lebedev angular + Becke partitioning
- Grid quality levels: Coarse, Normal, Fine, UltraFine
- Single-point PCM solvation for RKS/UKS
- Linear-response TDDFT / Casida + TDA excited states with transition dipoles, oscillator strengths, and UV-Vis spectrum output
- SP, Gradient, GeomOpt, Frequency, GeomOptFrequency
- Checkpoint/restart, symmetry+SAO blocking

### Error Handling Hardening (commits 1593541, 6851a44, 6ca12ff, 6f4c220)
- All public tensor / grid / nuclear-repulsion APIs now return `std::expected<T, std::string>`; no silent-wrong-answer paths remain from CODE_REVIEW.md
- Bounds-checked tensor accessors (`Tensor2D/4D/6D/ND`, `DenseTensorView`, amplitude/residual/denominator tensor accessors)
- Lebedev grid, nuclear repulsion, AO gradient evaluation all error out explicitly instead of returning NaN or partial results

## CASSCF PySCF Gate Table (11/11 Passing)

See `docs/CASSCF_STATUS.md` for the full table with reference energies and deviations.

Suite: PySCF 2.12.1, `mol.cart = True` (Cartesian basis).
