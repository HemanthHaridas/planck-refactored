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

**Exchange**: Slater (LDA), B88 (GGA), PW91, PBE, **B3LYP**, **PBE0**, Custom
**Correlation**: VWN5 (LDA), LYP (GGA), P86, PW91, PBE, Custom
LDA, GGA, and **global hybrids** (B3LYP, PBE0) are supported. Range-separated and double-hybrid functionals are rejected at init with an explicit unsupported diagnostic.

## Global Hybrid XC (commit f208777)

`src/dft/base/wrapper.h` exposes `hybrid_type()`, `is_hybrid()`, `is_global_hybrid()`, and `exact_exchange_coefficient()` from libxc. When a global hybrid is selected:

1. `XCExchangeFunctional::B3LYP` / `PBE0` are named aliases; combined exchange-correlation libxc IDs are used without double-counting the correlation slot (see `src/dft/driver.cpp:999` and around line 1591).
2. The KS build assembles an AO exchange matrix `K` from the ERI tensor (both RKS and UKS) in `src/dft/driver.cpp` (scaled by `exact_exchange_coefficient`).
3. The scaled exact-exchange contribution is added to the KS potential via `build_ks_matrix` (see `src/dft/ks_matrix.h` — `exact_exchange_alpha` / `exact_exchange_beta` / `exact_exchange_energy` parameters).
4. The matching exchange energy is included in the reported DFT total energy.

Regression cases: `h2_dft_b3lyp_sto3g` and `h_dft_uks_b3lyp_sto3g`.

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
