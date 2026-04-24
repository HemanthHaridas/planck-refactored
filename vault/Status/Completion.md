---
name: Completion Status
description: What is fully implemented and validated in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, completion, validated]
---

# Completion Status

Last updated: 2026-04-24

## Fully Implemented and Validated

### HF/SCF
- RHF and UHF SCF with DIIS (queue size configurable)
- ROHF scaffolding (87a08ef)
- SAD guess (commit 733fb31) and H_core guess
- Symmetry detection + MO irrep labeling (libmsym)
- Mulliken population analysis
- Checkpoint system: same-basis restart + cross-basis Löwdin projection

### Post-HF
- RMP2 and UMP2 correlation energies
- **CASSCF**: fully implemented, 11/11 PySCF gate cases passing (2026-04-08)
- **SA-CASSCF**: shared-κ coupled solver, exact CI-response RHS, stagnation escape
- **RASSCF**: active space partitioning (RAS1/RAS2/RAS3)
- **Coupled cluster**: RCCSD, UCCSD, RCCSDT, UCCSDT, RCCSDTQ — teaching determinant-space prototypes plus tensor production backends. Arbitrary-order RCC solver via ccgen-generated residuals.

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
- LDA (Slater, VWN5), GGA (B88, LYP, PBE, PW91), and **global hybrids (B3LYP, PBE0)** via libxc (commit f208777)
- Treutler-Ahlrichs radial + Lebedev angular + Becke partitioning
- Grid quality levels: Coarse, Normal, Fine, UltraFine
- SP, Gradient, GeomOpt, Frequency, GeomOptFrequency, ImaginaryFollow
- Checkpoint/restart, symmetry+SAO blocking

### Error Handling Hardening (commits 1593541, 6851a44, 6ca12ff, 6f4c220)
- All public tensor / grid / nuclear-repulsion APIs now return `std::expected<T, std::string>`; no silent-wrong-answer paths remain from CODE_REVIEW.md
- Bounds-checked tensor accessors (`Tensor2D/4D/6D/ND`, `DenseTensorView`, amplitude/residual/denominator tensor accessors)
- Lebedev grid, nuclear repulsion, AO gradient evaluation all error out explicitly instead of returning NaN or partial results

## CASSCF PySCF Gate Table (11/11 Passing)

See `docs/CASSCF_STATUS.md` for the full table with reference energies and deviations.

Suite: PySCF 2.12.1, `mol.cart = True` (Cartesian basis).
