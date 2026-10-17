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

Grid quality levels (`DFTGridQuality`). Radial counts come from PySCF's
`gen_grid._default_rad` period × level table (`RAD_GRIDS` in `radial.h`;
presets map to PySCF levels 1/3/5/7). The old ORCA-style row formula gave
only 44 radial points for a second-row atom where ~75 are needed, and the grid
did not converge. Angular sizes are the five-region pruning schedule's
largest (valence) region:

| Level | Radial pts (H / 2nd row) | Max angular pts |
|-------|--------------------------|-----------------|
| Coarse | 30 / 40 | 110 |
| Normal | 50 / 75 | 302 |
| Fine | 70 / 105 | 434 |
| UltraFine | 90 / 135 | 590 |

`grid_radial_points` / `grid_angular_points` override the counts, and
`grid_policy` selects a versioned pruning schedule (`GridPruningPolicy`).
`legacy` is the default. `g4-3-v4` is a qualified opt-in policy, and it is
mandatory (with `grid_radial_points 135` and no angular override) for UKS
double-hybrid gradients. See `docs/DFT_GRID_CONVERGENCE_SCOPE.md`.

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

### J / K build: memory-direct, and why it does not jitter

The KS Coulomb and exact-exchange matrices are built memory-direct through the
same fused Fock loop HF uses: `_compute_2e_j_direct` and `_compute_2e_k_direct`
/ `_compute_2e_k_uhf_direct` (`src/integrals/base.h`, accumulators in
`src/integrals/fock_accumulate.h`). No `nb⁴` ERI tensor is materialized for
the SCF; the old `build_coulomb_from_eri` / `build_exchange_from_eri` dense
contractions were deleted (2026-07-20). On 8 water/6-31g/B3LYP this cut peak
RSS from 1360 to 427 MB at an identical energy. TDDFT is the one remaining
dense-tensor consumer.

DFT needs J and K separately, with **unscaled** K: each call site applies its
own prefactor (`-0.5 × c` for RKS, `-c` for UKS), and the ½ must never be
folded into the shared accumulator or RKS hybrids get half their exact
exchange. Architecture and MPI distribution: `docs/HPC_MPI.md`.

None of this is the kind of change that caused the historical DFT jitter (see
DFT XC Reduction Determinism). That jitter came from a **cross-thread
reduction** summed in non-deterministic completion order; the fused loop sums
in fixed order, and MPI reductions sum in fixed rank order.

The grid layer is threaded too (`0453c80d`): the density contraction in
`xc_grid.cpp` and the libxc `exc_vxc` evaluation split grid points into
disjoint slabs with no cross-thread reduction, bitwise-identical across
1/2/3/4/6/8 threads. AO-on-grid evaluation (`src/dft/ao_grid.h`) and grid
construction remain serial; any future reduction there must use fixed order.

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
- LDA/GGA/global-hybrid: single points, gradients, geomopt, frequencies, geomopt+freq, TDDFT
- Range-separated hybrids (erf): the same set, including TDDFT
- RKS global double hybrids: single points, analytic gradients, geomopt, frequencies (FD of analytic gradients), geomopt+freq
- UKS global double hybrids: single points, analytic gradients only (conventional all-active UMP2, no solvent, qualified V4 grid)
- Range-separated double hybrids: single-point energies only
- Yukawa/Gaussian screened-exchange functionals: evaluated with the erf kernel; rejected for TDDFT (see Open Work)

## Hybrid / Range-Separated / Double-Hybrid XC

`src/dft/base/wrapper.h` exposes `hybrid_type()`, `is_hybrid()`, `is_global_hybrid()`, and `exact_exchange_coefficient()` from libxc. When a global hybrid is selected:

1. `XCExchangeFunctional::B3LYP` / `PBE0` are named aliases; combined exchange-correlation libxc IDs are used without double-counting the correlation slot (`src/dft/driver.cpp`).
2. The KS build assembles an AO exchange matrix `K` from the ERI tensor (both RKS and UKS) in `src/dft/driver.cpp` (scaled by `exact_exchange_coefficient`).
3. The scaled exact-exchange contribution is added to the KS potential via `build_ks_matrix` (see `src/dft/ks_matrix.h` — `exact_exchange_alpha` / `exact_exchange_beta` / `exact_exchange_energy` parameters).
4. The matching exchange energy is included in the reported DFT total energy.

Regression cases: `h2_dft_b3lyp_sto3g` and `h_dft_uks_b3lyp_sto3g`.

The same wrapper layer also exposes `is_range_separated()`, `is_double_hybrid()`, `fock_exchange_coefficient()` / `short_range_exchange_coefficient()` (the SCF, SOSCF, and TDDFT all read exact exchange from these), and `perturbative_correlation_coefficient()`. Double hybrids run the KS SCF, then a non-self-consistent RMP2/UMP2 on the KS orbitals scaled by the PT2 coefficient. Workflows outside the support list above are rejected explicitly.

## Double-Hybrid Analytic Gradients

The restricted path (`src/dft/dh_pt2_gradient.{h,cpp}`) follows the Neese–Schwabe–Grimme (2007) stationary Lagrangian. It uses a matrix-free GMRES Z-vector (`solve_dh_zvector`) and direct (streamed) derivative-integral and Eq. 47 pair-density contractions. Only the five MO-ERI sectors the RHS needs are formed. Analytic-vs-energy-FD error is ≤1.7e-8 Ha/Bohr on water and C1 H2O2 (B2PLYP/STO-3G). The unrestricted path (`src/dft/udh_*.cpp`, `build_udh_gradient_correction`) is a dense implementation: full spin MO ERIs, a dense joint α/β Jacobian solved by pivoted QR, and a dense AO pair density. Worst FD error on the three hard open-shell B2PLYP cases is 2.7e-9 Ha/Bohr on the V4 grid. User-facing scope: `docs/DOUBLE_HYBRID_GRADIENTS.md`. Architecture and remaining scalability work: `docs/DH_IMPLEMENTATION.md`. Teaching derivation: §15 of `docs/PLANCK_TEACHING_GUIDE.md`.

## TDDFT / Linear Response

`src/dft/driver.cpp` contains a dense linear-response implementation with:
- full Casida and TDA solvers,
- global-hybrid and erf range-separated exact exchange (`c_fr K + c_sr K[erf, ω]`, one shared helper `src/dft/tddft_exchange.h` owns the `(ib|ja)` B-matrix index; `docs/RS_TDDFT.md`),
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
