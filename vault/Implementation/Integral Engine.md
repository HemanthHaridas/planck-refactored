---
name: Integral Engine
description: Obara-Saika, Rys, and HGP ERI engines, shell pairs, dispatch
type: implementation
priority: high
include_in_claude: true
tags: [integrals, obara-saika, rys, hgp, eri, shell-pairs]
---

# Integral Engine

## Overview

Three ERI engines, selected via `IntegralMethod`:
- `ObaraSaika` — Obara-Saika horizontal/vertical recurrences
- `RysQuadrature` — Rys quadrature (alternative)
- `HeadGordonPople` — HGP variant with VRR built once per primitive pair and
  HRR hoisted to the contracted shell-quartet level (`hgp` / `head-gordon-pople`
  in input)
- `Auto` — dispatch based on angular momenta (picks faster engine per shell quartet)

## Shell Pairs

`src/integrals/shellpair.cpp` + `shellpair.h`

`ShellPair` stores:
- Gaussian product center P = (α_a * A + α_b * B) / (α_a + α_b)
- Prefactor K_ab = exp(-α_a * α_b / (α_a + α_b) * |A-B|²)
- Sum of exponents ζ = α_a + α_b
- References to shell A and shell B

`build_shellpairs` ordering: **row-major** with outer loop over ia, inner loop over ib ≥ ia (upper triangle). This is an important invariant — see [[Shell Pair Indexing]].

## Norm Factors

Contracted Gaussian norm Nc is folded into `Shell._coefficients` during GBS reading:
```
shell._coefficients *= Nc
```
No separate Nc factor needed anywhere in the integral code. See [[Norm Factors]].

## Obara-Saika Engine (`src/integrals/os.cpp`)

Standard OS scheme:
1. Compute primitive integrals via vertical recurrences (VRR)
2. Transfer angular momentum via horizontal recurrences (HRR)
3. Contract over primitives
4. Place result into AO matrix at positions given by `ContractedView._index`

OpenMP parallelized over shell-pair loops when `USE_OPENMP` is defined.

## HGP Engine (`src/integrals/hgp.cpp`)

The HGP engine implements the Head-Gordon-Pople rearrangement of the OS
recurrences: VRR runs per primitive pair to build the `(a0|c0; m)` block,
those blocks are contracted across primitives into a single `(a0|c0)`
accumulator, and the two HRR passes (CD then AB transfer) run **once per
contracted shell quartet** instead of once per primitive pair. Same final
ERI value as OS — the savings come from amortizing HRR over the primitive
loop. Thread-local scratch (`g_hgp_scratch`) is resized per quartet and
reused.

Public entry points mirror OS one-for-one:

- `_compute_2e` (full ERI tensor) and `_compute_2e_fock` / `_fock_uhf`
  (direct SCF builds)
- `_contracted_eri_elem` (single contracted quartet, used by Fock builders
  and the gradient lowering term)
- `_compute_eri_deriv_elem` (12-component derivative, used by the gradient
  dispatcher)
- `_build_skeleton_eri_symm` and `_compute_2e_fock_{symm,symm_spherical}` in
  `src/symmetry/hgp_symm.cpp` — full-point-group direct SCF, same Cartesian-
  skin pattern OS and Rys use

### Screened kernels (LongRange / ShortRange)

HGP serves screened kernels natively across every public entry — per-quartet
`_contracted_eri_elem`, derivative `_compute_eri_deriv_elem`, full-tensor
`_compute_2e`, and direct-SCF Fock builds `_compute_2e_fock` /
`_compute_2e_fock_uhf`. No OS detour anywhere. The earlier Fock-side
fallbacks that delegated screened kernels to OS were lifted once the
per-quartet sweep + end-to-end SCF-energy comparator validated the native
path; see the regression IDs `water_{rhf,hse06,uhf_triplet,uks_hse06,
hse06_symm}_scf_energy_engine_os_vs_hgp`.

The screened scaling is applied inside `hgp_vrr`:

- Boys-argument scale `T = boys_scale · rho · |P-Q|²`
- WP/WQ vectors scaled by `wpwq_scale = screen.rho / rho`
- Prefactor scaled by `screen.prefactor_scale`
- C-VRR bra/ket coupling term `inv_2_delta` scaled by `boys_scale` for
  non-Coulomb kernels — see [[HGP Screened inv_2_delta]] for the bug this
  fixed

### Dispatcher

- SCF Fock builds dispatch through `src/integrals/base.h`:
  `engine == HeadGordonPople` routes to the HGP entries.
- Full-symmetry direct SCF dispatches through `src/scf/scf.cpp`
  (`full_symmetry_fock_{rhf,uhf}` and `full_symmetry_build_skeleton`).
- Analytic gradient dispatches through `compute_eri_deriv_dispatch`, now a
  public entry in `HartreeFock::Gradient` (`src/gradient/gradient.{h,cpp}`) —
  engine-agnostic for all kernels (Coulomb, LongRange, ShortRange).
- MP2 / UMP2 analytic gradient (`src/post_hf/mp2_gradient.cpp`) now routes its
  three derivative-ERI sites through the same
  `HartreeFock::Gradient::compute_eri_deriv_dispatch`, so the MP2/UMP2 gradient
  response intermediates honor the selected engine (HGP when `engine hgp`)
  rather than always using OS. Cross-engine equality is gated by
  `water_rmp2_gradient_engine_os_vs_hgp` (OS↔HGP RMP2 gradient identical to the
  8-decimal print precision); the UMP2 radical-cation input shares the same
  routing and matches to `0.000e+00` when compared manually.

### Test hooks retained for historical gates

`_contracted_eri_elem_native_test` and `_compute_eri_deriv_elem_native_test`
are now thin aliases of the production entries. They predate the screened-
guard lift, when the test hooks were the only way to exercise the native
path. Kept so existing test code links unchanged.

## Index Placement

After computing a block of ERIs for shell quartet (μν|λσ), the result is placed at:
- Row: `sp_mn.A._index` to `sp_mn.A._index + n_a - 1`
- Col: `sp_mn.B._index` to `sp_mn.B._index + n_b - 1`

This requires `ContractedView._index` to be correctly set. **Never use `invert_pair_index`** to recover (ii, jj) — use `sp.A._index` / `sp.B._index` directly.

## Parallelization and Load Balance

### Flattened triangular `_compute_2e` loop

The one-shot full-tensor `_compute_2e` in all four engines (`ObaraSaika`,
`HeadGordonPople`, `RysQuad::_compute_2e`, and `RysQuad::_compute_2e_auto`)
iterates the upper triangle of shell-pair quartets `q >= p`. Parallelizing
`#pragma omp parallel for` over `p` alone is badly load-imbalanced: thread 0
gets a full triangle row (`npairs` quartets) while the last thread gets one, so
the long rows starve the others at the barrier — this was the dominant idle in
the conventional-SCF / MP2 / post-HF ERI build.

Each engine now flattens the triangle into a single linear index
`t ∈ [0, npairs(npairs+1)/2)` and distributes **that** with
`schedule(dynamic, 64)`. `(p,q)` is recovered from `t` by closed-form inversion
of the row-major triangle (row `p` starts at `p*npairs - p(p-1)/2`), with
`while`-guards that absorb `sqrt` drift (verified exact through the real
8515-pair case). This `t → (p,q)` map is used **only** to assign work to threads
— AO placement still uses `sp.A._index` / `sp.B._index`, so the
"never `invert_pair_index` for placement" rule above is intact.

The output is bitwise-identical to the serial-row form because the scatter
(`write_eri_permutations`) is **store-only** (`#pragma omp atomic write`, every
writer storing the same canonical value), so the tensor is independent of the
order in which `(p,q)` pairs are visited. Gated by `planck-compute-2e` (golden
checksum + 8-fold permutational symmetry + Rys/Auto-vs-OS cross-check) and
`planck-hgp-engine-smoke` (OS↔HGP to 1e-12).

The per-iteration `_compute_2e_fock{,_uhf}` Fock builds were left on the simpler
`parallel for` over `p`: they run every SCF iteration, so the triangular
imbalance amortizes across iterations (direct SCF profiled at ~1% idle).

### Parallel 4-index transforms

The two dense 4-index transforms were serial hotspots that stalled every other
thread:
- `Correlation::transform_eri` (`src/post_hf/integrals.cpp`) — AO→MO transform
  used by MP2, CC, CASSCF, stability, and hybrid-DFT exchange.
- `BasisFunctions::transform_eri_cart_to_sph` (`src/basis/spherical.cpp`) — the
  whole-system `n_cart⁴` Cartesian→spherical transform run once per conventional
  spherical SCF.

Both are now parallelized with `#pragma omp parallel for schedule(static)` over
the leading index of each quarter transform, which strides whole disjoint output
slabs (no shared writes, no reduction, inner accumulation order unchanged →
bitwise-identical). Neither is ever called from inside an OpenMP region, and the
project disables nesting by default, so there is no over-subscription risk.
Gated by `planck-transform-eri` and `planck-transform-eri-sph` (golden +
brute-force oracle, multi- and single-threaded).

## Key Files

- `src/integrals/os.cpp` + `os.h` — Obara-Saika ERI
- `src/integrals/rys.cpp` — Rys quadrature ERI. The 6D accumulator is a
  thread-local `RysScratch` sized per quartet (`resize_for_quartet` + flat
  `index()`/`at()`), reused across quartets — the same per-quartet scratch
  model as HGP's `g_hgp_scratch`/OS's `_eri_scratch`, but minimal (spatial-only,
  no Boys `m` axis, since Rys gets its angular dependence from quadrature
  roots). Replaced a fixed `[2·MAX_L+1]^6 = 38.5 MB`/thread buffer; see
  Completion. The small 1D VRR tables and 3D CD-HRR slice stay as fixed
  `VRR_DIM` stack arrays.
- `src/integrals/hgp.cpp` + `hgp.h` — HGP ERI (VRR-per-pair + HRR-outside)
- `src/integrals/base.h` — engine-dispatch wrappers for `_compute_2e` /
  `_compute_2e_fock` / `_compute_2e_fock_uhf`
- `src/symmetry/{os,rys,hgp}_symm.cpp` — full-point-group skeleton + Fock
  for each engine; SCF dispatches through `src/scf/scf.cpp`
- `src/integrals/shellpair.cpp` + `shellpair.h` — shell pair data and construction
