---
name: Completion Status
description: What is fully implemented and validated in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, completion, validated]
---

# Completion Status

Last updated: 2026-04-08

## Fully Implemented and Validated

### HF/SCF
- RHF and UHF SCF with DIIS (queue size configurable)
- SAD guess (commit 733fb31) and H_core guess
- Symmetry detection + MO irrep labeling (libmsym)
- Checkpoint system: same-basis restart + cross-basis Löwdin projection

### Post-HF
- RMP2 and UMP2 correlation energies
- **CASSCF**: fully implemented, 11/11 PySCF gate cases passing (2026-04-08)
- **SA-CASSCF**: shared-κ coupled solver, exact CI-response RHS, stagnation escape
- **RASSCF**: active space partitioning (RAS1/RAS2/RAS3)

### ERI Engine
- Obara-Saika (os.cpp) — primary engine
- Rys quadrature (rys.cpp) — alternative
- Auto-dispatch based on angular momenta
- OpenMP parallelized

### Gradients and Geometry
- Analytic gradients: RHF, UHF, RMP2
- Geometry optimization: Cartesian L-BFGS + Internal Coordinate IC-BFGS
- Vibrational frequencies: semi-numerical Hessian (finite-difference analytic gradients)
- Imaginary frequency following

### DFT (`planck-dft` binary)
- RKS and UKS
- LDA (Slater, VWN5) and GGA (B88, LYP, PBE, PW91) via libxc
- Treutler-Ahlrichs radial + Lebedev angular + Becke partitioning
- Grid quality levels: Coarse, Normal, Fine, UltraFine
- SP, Gradient, GeomOpt, Frequency, GeomOptFrequency, ImaginaryFollow
- Checkpoint/restart, symmetry+SAO blocking

## CASSCF PySCF Gate Table (11/11 Passing)

See `docs/CASSCF_STATUS.md` for the full table with reference energies and deviations.

Suite: PySCF 2.12.1, `mol.cart = True` (Cartesian basis).
