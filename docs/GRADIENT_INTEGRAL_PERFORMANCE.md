# Why was an analytic gradient 20–120× the SCF it follows, and what made it cheap?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`.

This file answers one architecture question:

**After the memory-direct Fock build was made fast
(`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`), an analytic gradient still cost
20–120× the SCF it followed, and the MP2 gradients took minutes. Where did
that time go, and which changes to the derivative integrals, the loops and the
MP2 response machinery actually paid?**

## Short answer

None of the cost was intrinsic, and none of it was one thing.
- **SCF gradients.** The derivative ERIs were evaluated one AO component at a
  time, over the full square of AO pairs, on one thread. One triangular,
  threaded loop shared by every reference, fed by shell-quartet derivative
  block kernels, removed that.
- **MP2 gradients.** The cost was never in the integrals. RMP2 built an O(N⁸)
  AO pair density that nothing read; UMP2 rebuilt an N⁵ intermediate inside a
  loop it did not depend on, then built its whole orbital Hessian to solve one
  linear system, and ran its T2 → AO back-transforms as scalar n_occ² sums.
- **The 1e term.** Pasted six times, serial, it re-ran the nuclear recurrence
  for every axis.

Every analytic gradient in the HF binary now costs 1.1–4.3× its SCF, and no
change moved a printed gradient.

| system | SCF | gradient run, 2026-10-25 | gradient run, now | now ÷ SCF |
|---|---|---|---|---|
| 4 H2O / 6-31G\* (nb 76) | 0.38 s | 53.9 s | **0.50 s** | 1.3× |
| 8 H2O / 6-31G (nb 104) | 0.56 s | 68.0 s | **0.78 s** | 1.4× |
| 1 H2O / cc-pVTZ (nb 65 Cart.) | 0.54 s | 114.9 s | **0.59 s** | 1.1× |
| 1 H2O / 6-31G\* RMP2 (conventional) | 0.08 s | 7.30 s | **0.12 s** | 1.5× |
| 2 H2O / 6-31G\* RMP2 (conventional) | 0.15 s | > 5 min | **0.65 s** | 4.3× |
| 2 H2O⁺ / 6-31G\* UMP2 (conventional) | 0.16 s | 136 s (after G3) | **0.53 s** | 3.3× |

Apple M3, `build-full` (g++-15, OpenMP), direct SCF, `engine auto`, SAD, no
symmetry, four threads, best of three, whole run including the SCF,
2026-10-28. The 2026-10-25 runs had a concurrent compile on the machine
(±20 %) and the SCF got faster in between, so compare gradient columns, not
ratios across dates. The starting profile (8 H2O / 6-31G, one thread):
`_compute_eri_deriv_elem` 98.5 % inclusive of `compute_rhf_gradient`, the 1e
terms ~5 %, no pragma anywhere in `src/gradient/` or `mp2_gradient.cpp`.

## Where the logic lives

- `src/gradient/two_electron_gradient.h`: `two_electron_gradient`, the one 2e
  loop for every reference; `detail::accumulate_block_eri_gradient` (shell
  quartets, `engine auto`), `detail::accumulate_shell_pair_eri_gradient` (AO
  pairs, explicit engines and screened kernels), `detail::pair_permuted_gamma`.
- `src/gradient/gradient.cpp`: `one_electron_gradient`, the one 1e block;
  the SCF-reference gradients (`compute_{rhf,rks,uhf,rohf,uks}_gradient`);
  `compute_eri_deriv_dispatch` (per-component, Auto → HGP).
- `src/integrals/auto_deriv_block_eri.h`: `auto_deriv_kernel()`, the
  Rys2D/HGP crossover chain; `hgp_deriv_block_eri.h` and
  `rys_deriv_block_eri.h`, the derivative block kernels.
- `src/integrals/os.cpp`: the 1e derivative integrals, including
  `_compute_nuclear_deriv_C_xyz`.
- `src/post_hf/mp2_gradient.cpp`: the RMP2/UMP2 gradient intermediates and the
  shared T2 → AO helpers (`pair_half_density`, `back_transform_half_density`,
  `add_symmetrized`).
- `src/post_hf/uhf_response.cpp`: `UHFOrbitalHessian`, `solve_uhf_cphf`
  (matrix-free GMRES), `build_uhf_cphf_matrix` (dense, for SOSCF).
- `src/dft/response_gmres.h`: `DFT::Response::gmres`, shared with the
  double-hybrid Z-vector.

## How the gradient works

**2e term.** RHF, RKS, UHF, ROHF, UKS, RMP2 and UMP2 all call
`two_electron_gradient(calc, shell_pairs, gamma_fn)` with their own Γ; the
result is ¼ Σ over ordered AO quartets of Γ(p,q,r,s) d(pq|rs)/dR. In every
index swap of a quartet each centre's derivative still goes to that centre's
own atom, so the loop visits each unordered pair of AO pairs once (`q ≥ p`),
`pair_permuted_gamma` sums Γ over the (ij)/(kl) and bra–ket swaps, and one
derivative evaluation serves them all. This is exact for any Γ, and a zero Γ skips the
integral.

Under `engine auto` with the Coulomb kernel the loop walks shell-group pairs
Q ≥ P and visits exactly the AO quartets the AO loop would. Each block's 12
derivatives per component come from one primitive pass of
`auto_deriv_kernel()`:
- `RysDerivBlockKernel`: per-axis differentiation of the 2D Rys tables,
  libcint's route. It handles one-primitive blocks at L ≥ 4 and three-quartet
  blocks at L ≥ 7.
- `HgpDerivBlockKernel`: everything else. One VRR to
  (L_A+L_B+1 | L_C+L_D+1), contracted with weights 1, 2α, 2β, 2γ, then the
  energy kernel's `hgp_block::hrr` on each raised and lowered quartet.

Both take D from translational invariance. A block the chain declines falls
back to the per-component derivative. Explicit engines and screened kernels
keep the AO-pair loop with per-component `compute_eri_deriv_dispatch`.

**1e term.** Every gradient with a one-electron part calls
`one_electron_gradient(calc, ao_pairs, D, W)`, which returns the kinetic,
basis-function nuclear, nucleus-position nuclear and overlap parts
(Σ D dh/dR, Σ W dS/dR) separately. Only the symmetric parts of D and W enter,
so each unordered AO pair is visited once. SCF gradients take `total()`; the
MP2 and double-hybrid paths take `core()` and build their own overlap term.

**MP2 orbital side.**
- RMP2's `dm2buf_full` and UMP2's D_α / D_β come from the same three helpers:
  a half density C_l T_ij C_rᵀ per occupied pair, then two reshaped GEMMs
  for Σ_ij C(p,i) H^{ij}_{qr} C'(s,j), then the index-image sums. D_β's αβ
  term is B_αβ read bra↔ket.
- The UHF Z-vector is solved by GMRES on the orbital-Hessian action. RHF
  still builds its matrix, but from MO integrals.

**Threads.** Every loop deals its work units into 64 fixed bins (unit
mod 64), each bin with its own `natoms × 3` partial, summed in bin order.

## What was found

### The derivative ERIs: per component, square, serial (G1–G3)

- **Routing (G1).** `compute_eri_deriv_dispatch` sent Auto to OS. Routed to
  HGP, 4 H2O / 6-31G\* went 56.4 → 29.7 s at four threads (1.90× against an
  estimated 1.5×). At one thread `engine os` took 71.4 s and `engine hgp`
  47.0 s.
- **The loop (G2).** UHF and ROHF had their own loop copies. One triangular
  loop, threaded over fixed bins, replaced all of them:

  | 4 H2O / 6-31G\* RHF, `engine auto` | 1 thread | 2 | 4 | 8 |
  |---|---|---|---|---|
  | G1 | | | 30.5 s | |
  | G2 | 12.8 s | 6.45 s | 3.42 s | 2.43 s |

  One thread beats G1 at four by 2.4×: G1 never used more than one core, and
  the triangle halves the work. Also removed:
  - the unused `assume_pair_exchange_symmetry = false` fallback (a full
    `nb²` ordered pair list);
  - UKS's long-range exchange sweep, which ran three times;
  - an RKS/UKS Coulomb-only sweep that ran on every gradient to fill a
    breakdown only a debug report reads (B3LYP 7.94 → 5.36 s).
- **The kernels (G3).** The per-component derivative evaluates up to 24
  contracted ERIs per AO quartet, each with its own primitive loop. The block
  kernels compute every component from one primitive pass. Accuracy against
  per-component OS on water, every component and all 12 derivatives (D
  included), relative to the sweep's largest derivative:

  | | 6-31G / 6-31G\* | cc-pVDZ | cc-pVTZ |
  |---|---|---|---|
  | HGP block | 2.6e-15 | 5.9e-15 | 1.8e-13 |
  | Rys2D block | 2.8e-15 | 4.9e-15 | 1.0e-13 |
  | per-component OS vs HGP | | | 2.7e-13 |

  The crossover was measured for derivatives, not carried over from the
  energy build (whole run, four threads, best of two):

  | | energy crossover rule | HGP only | Rys2D everywhere |
  |---|---|---|---|
  | 4 H2O / 6-31G\* | **0.57 s** | 0.58 s | 0.72 s |
  | 4 H2O / cc-pVDZ | **1.46 s** | 1.63 s | 2.02 s |
  | 1 H2O / cc-pVTZ | **0.62 s** | 0.68 s | 0.69 s |

  Against G2: 4 H2O / 6-31G\* 4.75 → 1.21 s, 1 H2O / cc-pVTZ 3.58 → 0.67 s
  (per-component OS 10.43 s, 12.23 s).

### The MP2 gradients: never the integrals (G4, G7)

- **An unread O(N⁸) tensor (G4).** A profile of the 2 H2O / 6-31G\* RMP2
  gradient put every busy sample in `build_rmp2_gradient_intermediates`' own
  loops. They built an AO pair density over μνλσ × ijab into `Gamma_pair_ao`,
  a field nothing read. Deleting it and moving the derivative sweep onto the
  shared loop took the run from 358 to 0.65 s. The MP2 terms fit that loop
  because they are a bra-only contraction Σ_{p on A} X ∂_p(pq|rs). X,
  symmetrized over r ↔ s, is 8-fold symmetric for both the pair density and
  the separable part (T̃ = 4t − 2t is symmetric under (i,a) ↔ (j,b)), so the
  loop's all-centre form with Γ = X is the same sum.
- **Loop-invariant work in a loop (G4).** The UMP2 sweep rebuilt
  D_σ[p,v,r,s] inside a q loop it does not depend on: O(nao⁵ nocc²). Built
  once, with Imat_σ as a separate contraction, 2 H2O⁺ went 136 → 3.5 s.
- **A matrix built to solve one system (G7).** The remaining UMP2 profile was:
  - 59 % `solve_uhf_cphf`. It formed the (n_ov^α + n_ov^β)² orbital Hessian
    one unit column at a time, one full direct UHF Fock build each:
    28·10 + 29·9 = 541 builds, 333 samples of them recomputing the Schwarz
    table.
  - 23 % D_σ and Imat.
  - 14 % serial `part_dm2`.

  GMRES on the same operator, preconditioned by ε_a − ε_i, converges in 16
  iterations, 33 Fock builds; run 3.5 → 1.83 s. The dense builder applies
  the same operator to unit vectors and stays for SOSCF.
- **Scalar back-transforms (G7).** `part_dm2` ran n_occ² n_ao² n_vir² on one
  thread, and D_σ did n_ao⁴ n_occ² work as four scalar terms per element. That
  was 83 % of the gradient once the Z-vector was fixed. As per-pair GEMMs, two
  half-transforms and image sums shared with RMP2 (whose `dm2buf_full` was
  also serial): 1.83 → 0.55 s. Imat as a GEMM was planned and dropped at
  1.3 %.

### The 1e term: six copies, one recurrence per axis (G5)

Priced before building: 14–32 % of the RHF gradient call (main-thread
samples 64/203 on 8 H2O / 6-31G, 29/108 on 4 H2O / 6-31G\*, 8/57 on cc-pVTZ).
Almost all of it was the nucleus-position derivative running the nuclear VRR
once per (atom, axis). The six copies were:
- four in `gradient.cpp` (RHF/RKS, UHF, ROHF, UKS);
- the MP2 helper, which allocated nine nao² matrices per atom and re-ran the
  C-derivative over all AO pairs per (atom, axis);
- the DH Eq. 33 contraction, which walked ordered pairs and recomputed the
  all-nuclei A-derivative per (atom, axis).

One threaded function and a three-axis C-derivative (Boys values and V
recurrence shared, bitwise per axis) took 8 H2O 0.87 → 0.78 s and 4 H2O
0.54 → 0.50 s; cc-pVTZ and UMP2 were unchanged within noise. The gain matched
the price. The unused public `contract_rmp2_one_electron_gradient` was
deleted.

### Defects found on the way

- **The SCF stored the DIIS extrapolation's orbitals.** G2's new summation
  order moved an H2 geometry-optimization step by 1e-16, and
  `h2_optfreq_smoke` started failing.
  - RHF, UHF, RKS and UKS stored the eigenpairs of the DIIS-extrapolated
    Fock at convergence. For H2/STO-3G every DIIS error vector is ~1e-16, so
    the weights are arbitrary, and the extrapolated matrix carried other
    iterations' eigenvalues with the right eigenvectors.
  - The gradient's `W = Σ εᵢ Cᵢ Cᵢᵀ` then had the wrong εᵢ: the 1e/Pulay
    term was off by 1.6e-3 Ha/Bohr. It had been hidden whenever the error
    came out exactly zero.
  - Every path now re-diagonalizes the true Fock at convergence, as PySCF
    does (teaching guide §7).
- **The regression runner tested a stale binary.** `run_regressions.py`
  passed only the input path to `.py` cases, and every comparator defaulted
  `--build-dir` to `./build`. Under `--build-dir build-full` the engine
  comparators had been running whatever sat in `./build`. The runner now
  passes `--build-dir` and a case's `args`.

## Where the time goes now

- **SCF gradients:** 1.1–1.4× the SCF on the benchmark systems; the 1e term
  is 19 of ~200 main-thread gradient samples on 8 H2O / 6-31G.
- **2 H2O⁺ / 6-31G\* UMP2** (~300 samples in the gradient):
  - Z-vector (33 direct Fock builds) ~35 %;
  - dense ERI tensor build ~16 %;
  - 2e derivative term ~7 %;
  - Imat ~7 %;
  - back-transforms ≤ 2 %.
- The conventional MP2 gradients are bounded by their dense nao⁴ tensors
  (D_α, D_β, the ERI) in memory, not time. RI-MP2 is the route past that.

## Not done

Tracked in `vault/Status/Open Work.md`; none is needed at these sizes.

- **G6: RI three-centre integrals.** `compute_3c_eri`, `compute_2c_eri` and
  `compute_3c_deriv_elem` (`src/post_hf/ri/ri_eri.cpp`) run per component,
  serial and unscreened. Measure their share on an RI-MP2 gradient and an
  RI-JK SCF first. D3 (RI pair factors recomputed per iteration,
  `docs/ERI_FOCK_BLAS_SCOPE.md`) is the larger RI-JK lever.
- **Block-level density screening** (`Q_ab Q_cd · max|Γ_block| < tol`) in the
  2e loop.
- **Screened kernels in the derivative block kernels.** HSE06's long-range
  term still takes the AO loop.
- **UMP2 Z-vector from the stored ERI tensor.** The conventional path already
  holds the AO ERI tensor, so the action could contract against it instead of
  running direct Fock builds.
- **The stored nb⁴ tensor.** `_compute_2e_auto`, used by MP2/CC/CASSCF/TDDFT
  and conventional SCF, still uses the shell-group hoist, not the block
  kernels. Measure its share of a real MP2/CC run first.
- **Analytic Hessians.**
- **Left alone on purpose: the two overlap-only MP2 loops.** Their weight
  includes `vhf_s1occ`, so routing them through a function that symmetrizes
  its weight is not obviously safe, and they are cheap.

## What invariants matter

### 1. Permutations change Γ, never which atom a derivative belongs to

Every index swap of a quartet keeps each centre's derivative on that centre's
atom. That is why one derivative evaluation per unordered pair of pairs serves
all eight orderings, for any Γ.

Design rule:

- Fold symmetry into Γ, not into the derivative. A loop that exploits
  permutational symmetry by assuming Γ is symmetric is wrong for the
  references whose Γ is not.

### 2. A gradient term must enter the shared loop through Γ alone

The MP2 terms reached the SCF loop by being rewritten as an all-centre Γ,
which needs X symmetrized over r ↔ s to be 8-fold symmetric. They needed no
MP2-specific integral code, and every later loop or kernel change reached them
for free.

Design rule:

- Before adding a new gradient's 2e term, show its coefficient has the 8-fold
  symmetry (or symmetrize it), then hand it to `two_electron_gradient`. A
  second derivative-integral loop is the regression this work removed six
  times.

### 3. A linear solve needs the operator, not the matrix

The UHF Z-vector was 59 % of the gradient purely because it materialized a
matrix whose only use was one solve. The cost was one Fock build per column,
growing as n_ov · (Fock build), faster than anything else.

Design rule:

- Build a dense response matrix only when the matrix itself is the product
  (SOSCF). Otherwise apply the operator inside a Krylov solve, and build the
  dense and matrix-free paths from one operator so they cannot drift.

### 4. A reference has its own noise floor

The GMRES Z-vector agreed with the dense QR to 1.2e-8 of max|z| at tolerance
1e-10, and tightening GMRES floored at ~5e-10. That floor is the dense
reference's own error: 541 screened direct builds. On STO-3G the two agree
to 1e-14.

Design rule:

- When tightening the method under test stops improving the agreement, the
  remaining difference belongs to the reference. Check that on a system where
  the reference is exact before chasing it.

### 5. Look for work nobody consumes before optimizing what is consumed

The largest single win (RMP2, 358 → 0.65 s) was deleting an O(N⁸) tensor no
caller read. The next (UMP2, 136 → 3.5 s) was moving an intermediate out of a
loop it did not depend on. Dead RMP2 and RKS/UKS sweeps fed only debug prints.

Design rule:

- Profile first, then for each hot loop check that its output is read and
  that its inputs depend on its index.

### 6. Price a term before building for it

The 1e term was priced at 14–32 % of the call before G5 was written, and the
measured gain matched it. The G3 crossover was measured for derivatives,
rather than assumed from the energy build (it held).

Design rule:

- Measure the share on the production path first, and state the expected gain
  with the change.

### 7. Determinism needs a fixed partition of the work

Every loop keys its partials by a bin fixed by the work unit (unit mod 64),
never by the thread, and sums them in bin order. Every step is identical at 1,
2, 4 and 8 threads (`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, invariant 4).

### 8. The derivative raise acts on the bare angular part

`Nc` is already in `_coefficients`, and raising A, B or C differentiates the
bare angular factor. So the component norm of the **unraised** function
applies. The RI `normC` bug was the same trap.

Design rule:

- In any new derivative kernel, apply the unraised component norm, and gate it
  on a basis with d or higher shells.

## Validation strategy that should remain in place

At close (2026-10-28, after G5): 72 gradient / geomopt / freq / FD / MP2 / DH
regression cases and 35 DH/MP2/gradient/derivative ctests pass. Three Eigen
`conjugate_gradient_*` tests report Not Run, the known Eigen test
registration issue.

- **`planck-eri-derivative-kernels`:** both block kernels against
  per-component OS for every component and all 12 derivatives on water
  6-31G … cc-pVTZ, gate 5e-13. The mutations 2β → β, a dropped lowered term,
  and D without C each fail on every basis.
- **Engine comparators** (`tests/engine_gradient_compare.py --engines
  os,auto`): eight `*_engine_os_vs_auto` cases covering
  - RHF on STO-3G and Cartesian 6-31G\*;
  - UHF;
  - ROHF on C1 CH3;
  - B3LYP;
  - HSE06/cc-pVTZ (block Coulomb, AO-loop long range);
  - RMP2 and UMP2.

  Explicit engines keep the AO loop, so these test the kernels against an
  independent path.
- **PySCF-gated cases:** the spherical 6-31G\* RHF/ROHF gradient, geomopt and
  freq cases, which run Auto; the RI-UMP2 FD and stationary-point cases.
- **`h2_optfreq_smoke`:** the guard on the DIIS-orbital defect.
- **What is missing:** no C++ test harness runs an SCF. The G7 dense-Z-vector
  and old-loop comparisons were temporary probes, and nothing permanently
  pins the GMRES Z-vector against the dense solve.

## Step index

Commits, tests and code comments cite these names.

| step | commit | what it did | outcome |
|---|---|---|---|
| G1 | `e851749e` | Auto derivative ERIs to HGP; dead RMP2 sweep; AO pairs built once | 1.90× (RHF), 6.78 → 3.17 s (RMP2) |
| fix | `ea251d62` | converged orbitals from the true Fock, not the DIIS extrapolation | correctness |
| G2 | `afafe332` | one threaded triangular 2e loop | 30.5 → 3.42 s at four threads |
| G2 follow-on | `4d1fe161` | RKS/UKS breakdown only on request | 7.94 → 5.36 s (B3LYP) |
| G3 | `ae5baf02`, `cbe1cd50` | HGP and Rys2D derivative block kernels; shell-quartet loop; runner `--build-dir` | 4.75 → 1.21 s |
| G4 | `4ca4d5ec` | MP2/UMP2 onto the shared loop; unread O(N⁸) density deleted | 358 → 0.65 s (RMP2), 136 → 3.5 s (UMP2) |
| G7 O1 | `7f436276` | matrix-free UHF Z-vector | 541 → 33 Fock builds; 3.5 → 1.83 s |
| G7 O2 | `d9cfba12` | T2 → AO back-transforms as shared GEMM helpers | 1.83 → 0.55 s |
| G7 O3 | — | Imat as a GEMM | dropped (1.3 %) |
| G5 | `32f0819f` | one threaded 1e block; three-axis C-derivative | 0.87 → 0.78 s (8 H2O) |
| G6 | — | RI three-centre integrals | not done |

Measuring: the benchmark inputs are water chains (`water_chain` in
`tests/benchmarks/scale_bench.py`, 3.0 Å spacing) with `calculation gradient`.
Subtract a `calculation energy` run of the same input to get the gradient's
share, and profile with `sample` on the running process. Build A/B baselines
as whole targets, and write shell loops in bash or with explicit arrays,
since zsh does not word-split an unquoted variable.
