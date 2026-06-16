---
name: DFT
description: Kohn-Sham DFT and TDDFT implementation — grid, libxc, PCM, KS matrix, `planck-dft`
type: implementation
priority: medium
include_in_claude: true
tags: [dft, ks-dft, tddft, rks, uks, libxc, grid, pcm]
---

# Kohn-Sham DFT (`planck-dft` binary)

## Entry Point

`DFT::Driver::run` in `src/dft/driver.cpp`. Handles:
- SinglePoint
- Gradient
- GeomOpt
- Frequency
- GeomOptFrequency
- TDDFT / linear response

`ImaginaryFollow` is parsed at the input level but currently returns an explicit unimplemented diagnostic in the DFT driver.

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

### J / K build parallelism (and why it does not jitter)

`build_coulomb_from_eri` and `build_exchange_from_eri` in
`src/dft/driver.cpp` are the per-iteration `nb⁴` AO contractions for the KS
Coulomb and exact-exchange (hybrid / range-separated) matrices. They were
fully serial — the dominant DFT load-imbalance (B3LYP profiled at ~60% of
samples idle at the barrier) — and are now `#pragma omp parallel for
schedule(static)` over the outer `mu`, mirroring
`HartreeFock::ObaraSaika::_compute_fock_rhf`.

Crucially this is **not** the kind of change that caused the historical DFT
jitter (see DFT XC Reduction Determinism). That jitter came from a
**cross-thread reduction** summed in non-deterministic completion order. The
J/K builds have **no cross-thread summation**: each thread owns a disjoint set
of output rows `coulomb(mu,·)` / `exchange(mu,·)` and computes them entirely
itself, with the inner `lam`/`sig` accumulation order unchanged. Verified
**bitwise-identical across `OMP_NUM_THREADS` = 1/2/4/8** (water-dimer/cc-pVTZ
B3LYP `-152.9317586225` to all digits), unlike the grid XC reduction which
still drifts ~1e-10 across thread counts.

The DFT **grid layer** (`evaluate_density_on_grid`, the `xc_grid.cpp`
density/XC loops) remains serial and is the residual DFT parallelization
target (~12% idle after the J/K change). It is deliberately deferred: adding a
parallel region there re-enters the grid-reduction jitter territory, so any
reduction must use fixed thread-index order.

## PCM Solvation

`planck-dft` can add a self-consistent C-PCM reaction field for single-point RKS/UKS calculations. The DFT driver builds a reusable `PCMState` once, then adds the reaction potential and solvation energy during each SCF iteration.

Key pieces:
- setup: `HartreeFock::Solvation::build_pcm_state`
- per-iteration reaction field: `HartreeFock::Solvation::evaluate_pcm_reaction_field`
- implementation: `src/solvation/pcm.{h,cpp}`

Analytic KS gradients currently warn/error if PCM geometry response would be required; PCM is not wired through DFT geometry optimization, frequencies, or TDDFT.

## Supported Functionals

**Exchange**: Slater (LDA), B88 (GGA), PW91, PBE, B3LYP, PBE0, HSE06, custom libxc names/IDs
**Correlation**: VWN5 (LDA), LYP (GGA), P86, PW91, PBE, B2PLYP-style combined XC entries, custom libxc names/IDs

Support is split by workflow:
- LDA/GGA/global-hybrid path: single points, gradients, geomopt, frequencies, and TDDFT
- Range-separated hybrids and double hybrids: single-point energies only

## Hybrid / Range-Separated / Double-Hybrid XC

`src/dft/base/wrapper.h` exposes `hybrid_type()`, `is_hybrid()`, `is_global_hybrid()`, and `exact_exchange_coefficient()` from libxc. When a global hybrid is selected:

1. `XCExchangeFunctional::B3LYP` / `PBE0` are named aliases; combined exchange-correlation libxc IDs are used without double-counting the correlation slot (see `src/dft/driver.cpp:999` and around line 1591).
2. The KS build assembles an AO exchange matrix `K` from the ERI tensor (both RKS and UKS) in `src/dft/driver.cpp` (scaled by `exact_exchange_coefficient`).
3. The scaled exact-exchange contribution is added to the KS potential via `build_ks_matrix` (see `src/dft/ks_matrix.h` — `exact_exchange_alpha` / `exact_exchange_beta` / `exact_exchange_energy` parameters).
4. The matching exchange energy is included in the reported DFT total energy.

Regression cases: `h2_dft_b3lyp_sto3g` and `h_dft_uks_b3lyp_sto3g`.

The same wrapper layer now also exposes `is_range_separated()` and `is_double_hybrid()`. The driver accepts those functionals for single-point calculations, computes the implemented exact-exchange share, and applies the additional perturbative correction path for double hybrids where supported. Non-single-point workflows still reject them explicitly.

## TDDFT / Linear Response

`src/dft/driver.cpp` contains a dense linear-response implementation with:
- full Casida and TDA solvers,
- RKS singlet/triplet support,
- UKS spin-conserving response,
- transition dipoles, oscillator strengths, and per-root reporting,
- Gaussian-broadened UV-Vis spectrum output.

Relevant input knobs live in `%begin_dft`:
- `lr_nstates` / `tddft_nstates` / `nroots`
- `root`
- `lr_method` / `tddft_method`
- `lr_spin` / `tddft_spin`

## Checkpoint / Restart

Same checkpoint system as HF — saves MO coefficients and energies. Cross-basis Löwdin projection works for warm-start across different basis sets.

## Key Files

- `src/dft/driver.cpp` + `driver.h` — entry
- `src/dft/base/grid.h`, `radial.h`, `angular.h` — grid
- `src/dft/base/wrapper.h` — libxc C API wrapper
- `src/dft/ao_grid.h` — AO-on-grid evaluation
- `src/dft/xc_grid.cpp` + `xc_grid.h` — density + XC
- `src/dft/ks_matrix.cpp` + `ks_matrix.h` — KS potential matrix
- `src/solvation/pcm.cpp` + `pcm.h` — shared PCM implementation used by HF and DFT
- `src/dft/main.cpp` — binary entry point
