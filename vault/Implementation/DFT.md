---
name: DFT
description: Kohn-Sham DFT implementation — grid, libxc, KS matrix, planck-dft binary
type: implementation
priority: medium
include_in_claude: true
tags: [dft, ks-dft, rks, uks, libxc, grid]
---

# Kohn-Sham DFT (`planck-dft` binary)

## Entry Point

`DFT::Driver::run` in `src/dft/driver.cpp`. Handles: SinglePoint, Gradient, GeomOpt, Frequency, GeomOptFrequency, ImaginaryFollow.

## Grid Construction

Three-level partitioning:
1. **Radial**: Treutler-Ahlrichs scheme (`src/dft/base/radial.h`)
2. **Angular**: Lebedev quadrature (`src/dft/base/angular.h`)
3. **Partitioning**: Becke fuzzy-cell scheme (`src/dft/base/grid.h`)

Grid quality levels (`DFTGridQuality`):

| Level | Radial pts | Angular pts |
|-------|-----------|------------|
| Coarse | ~25 | ~110 |
| Normal | ~50 | ~302 |
| Fine | ~75 | ~590 |
| UltraFine | ~99 | ~974 |

## AO Evaluation

`src/dft/ao_grid.h` — evaluates all AOs (and optionally their gradients) on all grid points. Used for density and XC potential assembly.

## Density and XC

`src/dft/xc_grid.cpp` / `xc_grid.h`:
- Computes electron density ρ (and ∇ρ for GGA) on grid from P and AO values
- Calls libxc via `src/dft/base/wrapper.h` to get εxc and vxc
- Integrates vxc × AO products to form the XC contribution to KS matrix

## KS Matrix

`src/dft/ks_matrix.cpp` / `ks_matrix.h`:
- Assembles full KS Fock: F = H_core + J + V_xc
- J built from ERI the same way as HF (shared integral code)
- V_xc from grid integration
- Symmetry + SAO blocking supported

## Supported Functionals

**Exchange**: Slater (LDA), B88 (GGA), PW91, PBE, Custom  
**Correlation**: VWN5 (LDA), LYP (GGA), P86, PW91, PBE, Custom  
LDA and GGA are both supported. Hybrid (exact-exchange mixing) not yet implemented.

## Checkpoint / Restart

Same checkpoint system as HF — saves MO coefficients and energies. Cross-basis Löwdin projection works for warm-start across different basis sets.

## Key Files

- `src/dft/driver.cpp` + `driver.h` — entry
- `src/dft/base/grid.h`, `radial.h`, `angular.h` — grid
- `src/dft/base/wrapper.h` — libxc C API wrapper
- `src/dft/ao_grid.h` — AO-on-grid evaluation
- `src/dft/xc_grid.cpp` + `xc_grid.h` — density + XC
- `src/dft/ks_matrix.cpp` + `ks_matrix.h` — KS potential matrix
- `src/dft/main.cpp` — binary entry point
