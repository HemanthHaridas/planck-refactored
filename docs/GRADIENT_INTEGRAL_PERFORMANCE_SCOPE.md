# Why is an analytic gradient 20–120× the SCF it follows?

It no longer is. **Every analytic gradient in the HF binary (RHF, UHF, ROHF,
RKS/UKS through the same 2e loop, RMP2 and UMP2) now costs 1.1–4.3× the SCF
it follows, down from 20–120× and, for the MP2 cases, minutes.** The cause
was never one thing: the derivative ERIs were per AO component, untriangled
and single-threaded (G1–G3), the MP2 gradients carried an unread O(N⁸)
density and rebuilt N⁵ intermediates in their innermost loop (G4), and the
UMP2 Z-vector built its full orbital Hessian with one Fock build per column
while its back-transforms ran as scalar n_occ² sums (G7), and the serial 1e
block, pasted six times, re-ran the nuclear recurrence for every axis (G5).

Measured on M3, `build-full`, direct SCF, `engine auto`, SAD, no symmetry,
four threads, best of three, whole run including the SCF:

| system | SCF | gradient run 2026-10-25 | **now** | now ÷ SCF |
|---|---|---|---|---|
| 4 H2O / 6-31G\* (nb 76) | 0.38 s | 53.9 s | **0.50 s** | 1.3× |
| 8 H2O / 6-31G (nb 104) | 0.56 s | 68.0 s | **0.78 s** | 1.4× |
| 1 H2O / cc-pVTZ (nb 65 Cart.) | 0.54 s | 114.9 s | **0.59 s** | 1.1× |
| 1 H2O / 6-31G\* RMP2 (conventional) | 0.08 s | 7.30 s | **0.12 s** | 1.5× |
| 2 H2O / 6-31G\* RMP2 (conventional) | 0.15 s | > 5 min | **0.65 s** | 4.3× |
| 2 H2O⁺ / 6-31G\* UMP2 (conventional) | 0.16 s | 136 s (after G3) | **0.53 s** | 3.3× |

The 2026-10-25 runs had a concurrent compile on the machine (±20 %) and the
SCF got faster in between, so compare gradient columns, not ratios across
dates. Printed gradients are unchanged by every step, and identical at any
thread count.

Scope status: the 2e, MP2-orbital and 1e work is done; G6 and two G3
leftovers below are still open, so this file stays a scope until they land
or are declined. It follows `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, which
made the memory-direct Fock build fast and is closed.

## What is still open

In order:

1. **G6 — RI three-centre integrals (measure first).** `compute_3c_eri`,
   `compute_2c_eri`, `compute_3c_deriv_elem` (`src/post_hf/ri/ri_eri.cpp`):
   per component, serial, unscreened. Thread over aux shells (disjoint
   columns, bitwise). Price the share on an RI-MP2 gradient and an RI-JK SCF
   before building. D3 (RI pair factors recomputed per iteration,
   `docs/ERI_FOCK_BLAS_SCOPE.md`) is the larger RI-JK lever and stays separate.
2. **G3 leftovers, not yet needed at these sizes.** Block-level density
   screening `Q_ab Q_cd · max|Γ_block| < tol` (the Schwarz table itself stays
   unscreened, invariant 8); screened (range-separated) kernels in the block
   derivative kernels, since HSE06's long-range term still takes the AO loop.
3. **UMP2 residue, not worth a step yet.** After G7 the 2 H2O⁺ gradient is
   ~35 % Z-vector, ~16 % ERI tensor, ~7 % each 2e derivative term and Imat.
   The conventional path already holds the AO ERI tensor, so the Z-vector
   action could contract against it instead of 33 direct Fock builds. GMRES
   also spends a second action per iteration on its fresh-residual check, by
   design. The dense `nao⁴` tensors (D_α, D_β, the ERI) cap the conventional
   UMP2 gradient's system size; RI-MP2 is the route past that.

## How the 2e gradient runs now

The loop lives in `src/gradient/two_electron_gradient.h`; the SCF-reference
gradients are in `src/gradient/gradient.cpp`, the MP2 ones in
`src/post_hf/mp2_gradient.cpp`.

- **One loop for every reference.** RHF, RKS, UHF, ROHF, UKS, RMP2 and UMP2
  all call `two_electron_gradient(calc, shell_pairs, gamma_fn)` with their own
  Γ. The result is ¼ Σ over ordered AO quartets of Γ(p,q,r,s) d(pq|rs)/dR.
- **Permutations only change Γ.** In every index swap of a quartet, each
  centre's derivative still goes to that centre's own atom, so
  `pair_permuted_gamma` sums Γ over the (ij)/(kl) swaps and the bra–ket swap,
  and one derivative evaluation serves them all. Exact for any Γ. A zero Γ
  skips the derivative entirely.
- **MP2 fits the same loop.** The MP2 terms were written as a bra-only
  contraction Σ_{p on A} X ∂_p(pq|rs); with X symmetrized over r ↔ s 8-fold
  symmetric (true for both the pair density and the separable part), that
  equals the loop's all-centre form with Γ = X (teaching guide §15).
- **`engine auto`, Coulomb kernel: shell quartets.**
  `accumulate_block_eri_gradient` walks shell-group pairs Q ≥ P and visits
  exactly the AO quartets the AO loop would (i ≤ j within a group; each
  unordered AO pair of pairs once in a diagonal block). Each block's 12
  derivatives per component come from one primitive pass of
  `auto_deriv_kernel()` (`src/integrals/auto_deriv_block_eri.h`):
  - `RysDerivBlockKernel` (`rys_deriv_block_eri.h`) for one-primitive blocks
    at L ≥ 4 and three-quartet blocks at L ≥ 7 (the energy build's crossover,
    re-measured for derivatives). Per-axis differentiation of the 2D Rys
    tables, libcint's route.
  - `HgpDerivBlockKernel` (`hgp_deriv_block_eri.h`) for the rest: one VRR to
    (L_A+L_B+1 | L_C+L_D+1) contracted with weights 1, 2α, 2β, 2γ, then the
    shared `hgp_block::hrr` on each raised and lowered quartet.
  - D from translational invariance in both. A block the chain declines falls
    back to the per-component derivative.
- **Explicit engines and screened kernels: AO pairs.** The same `q ≥ p`
  triangle over AO pairs with per-component `compute_eri_deriv_dispatch`
  (HGP for `hgp`, OS otherwise). This keeps the `--engines os,auto`
  comparators independent of the kernels they check.
- **Threads.** Both loops deal bra units into 64 fixed bins (unit mod 64),
  each with its own `natoms × 3` partial summed in bin order. The result is
  identical at any thread count.
- **MP2 orbital side** (`mp2_gradient.cpp`, `uhf_response.cpp`): the UHF
  Z-vector is matrix-free GMRES; the T2 → AO back-transforms are the shared
  `pair_half_density` / `back_transform_half_density` / `add_symmetrized`
  helpers (G7).

## Record of what landed

### G5 — one threaded 1e derivative block (2026-10-28)

Priced first: on the RHF gradients the 1e term was 14–32 % of the gradient
call (main-thread samples: 8 H2O / 6-31G 64 of 203, 4 H2O / 6-31G\* 29 of
108, 1 H2O / cc-pVTZ 8 of 57), almost all of it the nucleus-position
derivative, which ran the whole nuclear VRR once per (atom, axis).

- **One function** `Gradient::one_electron_gradient(calc, ao_pairs, D, W)`
  returns the kinetic / basis-function nuclear / nucleus-position nuclear /
  overlap parts (Σ D dh/dR, Σ W dS/dR; only the symmetric parts of D and W
  enter). It replaces six copies: four in `gradient.cpp` (RHF/RKS, UHF,
  ROHF, UKS), the MP2 helper (which allocated nine nao² matrices per atom and
  re-ran the C-derivative over all AO pairs per (atom, axis)), and the DH
  Eq. 33 h/S contraction in `dft/driver.cpp` (ordered nao² pairs, the
  all-nuclei A-derivative recomputed per (atom, axis)). The unused public
  `contract_rmp2_one_electron_gradient` is deleted. The two overlap-only MP2
  loops stay: their weight includes `vhf_s1occ`, so symmetrizing it is not
  obviously safe, and they are cheap.
- **`_compute_nuclear_deriv_C_xyz`**: the Boys values and V VRR are shared by
  the three directions; only the dVRR and HRR run per direction. Same
  recurrences, so per direction bitwise the old kernel;
  `_compute_nuclear_deriv_C_elem` is now a wrapper.
- **Threaded** over AO pairs in 64 fixed bins (pair index mod 64) summed in
  order: identical at any thread count.

Whole run, four threads, best of three: 8 H2O / 6-31G 0.87 → 0.78 s,
4 H2O / 6-31G\* 0.54 → 0.50 s, 1 H2O / cc-pVTZ 0.60 → 0.59 s, 2 H2O⁺ UMP2
0.54 → 0.53 s. 1e main-thread samples on 8 H2O 64 → 19. Printed gradients
identical to the old binary on all four (one symmetry-zero component on
8 H2O changes the sign of its printed zero) and identical at 1 and 4
threads. 72 gradient / geomopt / freq / FD / MP2 / DH regression cases and
35 DH/MP2/gradient/derivative ctests pass (plus the three known Eigen Not
Run).

### G7 — UMP2 orbital side (2026-10-28)

Profile before it (2 H2O⁺ / 6-31G\*, ~3,040 main-thread samples in
`compute_ump2_gradient`): 59 % `solve_uhf_cphf` (1,443 samples in direct UHF
Fock builds, 333 recomputing the Schwarz table inside each), 23 % D_σ and
Imat, 14 % serial `part_dm2`, < 1 % the 2e derivative term.

- **O1 — matrix-free UHF Z-vector** (`7f436276`). `build_uhf_cphf_matrix`
  formed the (n_ov^α + n_ov^β)² orbital Hessian one unit column at a time,
  one full direct UHF Fock build each (28·10 + 29·9 = 541 here), then QR.
  The column body is now one operator (`UHFOrbitalHessian`,
  `uhf_response.cpp`); the dense builder still applies it to unit vectors
  (SOSCF needs the matrix), and `solve_uhf_cphf` hands it to
  `DFT::Response::gmres`, the solver the DH Z-vector already uses, with the
  ε_a − ε_i diagonal as preconditioner. No CG and no SPD check: GMRES checks
  its unpreconditioned residual with a fresh action, so one solver suffices.
  16 iterations, **33 Fock builds instead of 541**; run 3.5 → 1.83 s. z agrees
  with the dense QR to 1.2e-8 of max|z| at tolerance 1e-10; tightening to
  1e-12/1e-14 floors at ~5e-10, the dense reference's own screening noise,
  and the printed gradient is identical either way. STO-3G cation / triplet /
  RI: 1.6e-15 / 2.0e-15 / 1.3e-14. UHF SOSCF still converges superlinearly to
  the DIIS energy. RHF is left alone: `solve_rhf_cphf` builds its matrix from
  MO ERIs, no Fock build per column.
- **O2 — back-transforms as matrix products, one mechanism for both MP2
  flavours.** Three helpers in `mp2_gradient.cpp` replace RMP2's serial
  `dm2buf_full` loop and UMP2's `part_dm2` + D_σ loops:
  `pair_half_density` (C_l T_ij C_rᵀ per (i,j), threaded over pairs),
  `back_transform_half_density` (two reshaped GEMMs, nao⁴·n_occ instead of
  nao⁴·n_occ²) and `add_symmetrized` (the p↔q / r↔s / bra↔ket image sums,
  threaded over p). D_β's αβ term is B_αβ read bra↔ket, so one B_αβ serves
  both spins. Run 1.83 → **0.55 s**. Elementwise against the old loops
  (relative to the tensor's largest element): ≤ 6.1e-16 on the STO-3G
  cation, triplet, RI-UMP2 and RMP2 inputs, 1.9e-15 on 2 H2O⁺ / 6-31G\*.
  Peak memory grows by one nao⁴ buffer during the D_σ build.
- **O3 (Imat as a GEMM) dropped:** 1.3 % after O1.
- **Gates.** All 19 MP2-gradient and double-hybrid regression cases and the
  32 DH/MP2/gradient ctests pass (the three Eigen `conjugate_gradient_*`
  Not Run are the known registration issue). No new ctest: no C++ harness
  runs an SCF, so both dense/old-loop comparisons were temporary probes.

### G4 — MP2/UMP2 gradients onto the shared loop (2026-10-27)

- The 2e loop moved out of `gradient.cpp` into
  `src/gradient/two_electron_gradient.h` (`two_electron_gradient`, with the
  loops in `detail`). Pure move; SCF gradients identical.
- **RMP2.** A profile of the 2 H2O / 6-31G\* RMP2 gradient put every busy
  sample in `build_rmp2_gradient_intermediates`' own loops, none in derivative
  integrals. The cause was an AO pair density over μνλσ × ijab, O(N⁸), stored
  in `Gamma_pair_ao`, a field nothing read. Deleted, along with the equally
  unread `two_e_{non,}separable_gradient` fields. The per-atom derivative
  sweep and the `vhf1` matrices are replaced by one `two_electron_gradient`
  call after `dm1p` exists, Γ = 2 dm2buf (dense only) + the separable term.
  No explicit symmetrization was needed: symmetrized over r ↔ s both parts
  are already 8-fold symmetric (T̃ = 4t − 2t is symmetric under
  (i,a) ↔ (j,b)). Under `PLANCK_DEBUG_RMP2_TERMS` one extra sweep splits off
  the separable part for the print. 2 H2O / 6-31G\* RMP2 gradient run
  358 → 0.65 s.
- **UMP2.** Its sweep rebuilt the separable densities D_σ[p,v,r,s] inside the
  q loop, which they do not depend on: O(nao⁵ nocc²). D_α and D_β are now
  built once (threaded over p), Imat_σ is a separate nao⁵ contraction
  threaded over q (or `build_ri_imat` under RI), and the 2e term is one
  `two_electron_gradient` call with Γ = D_α + D_β (dense only) + the per-spin
  separable term. `build_ao_pair_table` (G1) and `Gamma_pair_ao` are gone.
  2 H2O⁺ / 6-31G\* UMP2 gradient run 136 → 3.5 s.
- Gradients identical in every printed component on dense RMP2 and UMP2
  (water cation, triplet water), RI-RMP2 and RI-UMP2.
- **Gates.** Two more `os,auto` comparators:
  `water_rmp2_gradient_engine_os_vs_auto` (core) and
  `water_radical_cation_ump2_gradient_engine_os_vs_auto`. After a full rebuild
  (including `planck-dft`, which links both `gradient.cpp` and
  `mp2_gradient.cpp`), the 35 Planck ctest gates for derivative kernels,
  engines, Fock accumulation, gradients and double hybrids pass, and the
  extended suite is 143/147 (the four known rccsdt cases). The ctest run also
  reports three Eigen `conjugate_gradient_*` tests as Not Run: the known Eigen
  test registration issue in Open Work, not a regression.

### G3 — shell-quartet derivative block kernels (2026-10-27)

**Kernel accuracy.** `planck-eri-derivative-kernels` checks every component
and all 12 derivatives against per-component OS on water in 6-31G, 6-31G\*,
cc-pVDZ and cc-pVTZ (D included, which OS computes explicitly):

| | 6-31G / 6-31G\* | cc-pVDZ | cc-pVTZ |
|---|---|---|---|
| HGP block | 2.6e-15 | 5.9e-15 | 1.8e-13 |
| Rys2D block | 2.8e-15 | 4.9e-15 | 1.0e-13 |
| per-component OS vs HGP | | | 2.7e-13 |

(relative to the sweep's largest derivative). The scoped 1e-13 bar was below
the floor: on cc-pVTZ (f raised to g) the two per-component engines differ by
more than that, and both block kernels sit inside the spread. The gate is 5e-13.
Mutations (2β → β, lowered term dropped, D without C) each fail on every basis.

**Crossover, measured end to end** (whole run, four threads, best of two):

| | energy crossover rule | HGP only | Rys2D everywhere |
|---|---|---|---|
| 4 H2O / 6-31G\* | **0.57 s** | 0.58 s | 0.72 s |
| 4 H2O / cc-pVDZ | **1.46 s** | 1.63 s | 2.02 s |
| 1 H2O / cc-pVTZ | **0.62 s** | 0.68 s | 0.69 s |

The energy build's rule also wins for derivatives, so it stays.

**Against G2**, whole run, four threads: 4 H2O / 6-31G\* 4.75 → 1.21 s,
1 H2O / cc-pVTZ 3.58 → 0.67 s (per-component OS: 10.43 s, 12.23 s). Gradients
identical in every printed component to OS and to G2.

**Gates.** Six `*_engine_os_vs_auto` cases (`tests/engine_gradient_compare.py
--engines os,auto`): RHF STO-3G and Cartesian 6-31G\*, UHF, ROHF (C1 CH3),
B3LYP, HSE06/cc-pVTZ (block Coulomb, AO-loop long range). Re-introducing
"D without C" fails them end to end. The PySCF-gated spherical 6-31G\*
RHF/ROHF gradient, geomopt and freq cases run Auto, so they cover the block
path as well.

**Coverage gap found on the way.** `run_regressions.py` passed only the input
path to `.py` cases, and every comparator defaulted `--build-dir` to `./build`.
So under `--build-dir build-full` the engine comparators (gradient, geomopt,
SCF energy, MPI smoke) had been running whatever stale binary sat in
`./build`. The runner now passes `--build-dir` and a case's optional `args`.

### G2 — one threaded triangular loop (2026-10-27)

UHF and ROHF dropped their loop copies. The loop visits `q ≥ p` with Γ
folded as above, threaded over 64 fixed bins. The unused
`assume_pair_exchange_symmetry = false` fallback (a full `nb²` ordered pair
list) is deleted; UKS ran its long-range exchange sweep three times, now once.

4 H2O / 6-31G\* RHF gradient, `engine auto`:

| | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|
| G1 | | | 30.5 s | |
| G2 | 12.8 s | 6.45 s | 3.42 s | 2.43 s |

One thread beats G1 at four by 2.4×: G1 never used more than one core, and
the triangle halves the work. Identical output at 1/2/4/8 threads.

Follow-on: RKS and UKS ran a second full Coulomb-only sweep on every gradient
to fill `WavefunctionGradientBreakdown`, which only the
`PLANCK_DFT_GRADIENT_DEBUG` report reads. `compute_{rks,uks}_gradient` take
`with_breakdown` (default false). 4 H2O / 6-31G\* B3LYP gradient run:
7.94 → 5.36 s.

### The DIIS fix G2 exposed (2026-10-27)

G2's new summation order moved an H2 geometry-optimization step by 1e-16, and
`h2_optfreq_smoke` started failing. The cause was in the SCF, not the
gradient. RHF, UHF, RKS and UKS stored the eigenpairs of the
**DIIS-extrapolated** Fock at convergence. For H2/STO-3G every DIIS error
vector is ~1e-16, the DIIS weights are arbitrary, and the extrapolated matrix
carries other iterations' eigenvalues with the right eigenvectors. The
gradient's `W = Σ εᵢ Cᵢ Cᵢᵀ` then had the wrong εᵢ: the 1e/Pulay term was off
by 1.6e-3 Ha/Bohr and the line search failed. It was hidden whenever the
error came out exactly zero. Every path now re-diagonalizes the true Fock once
at convergence when DIIS extrapolated, as PySCF does (teaching guide §7).

### G1 — dead work and routing (2026-10-27)

- `compute_eri_deriv_dispatch` resolves Auto to HGP (precedent:
  `full_symmetry_engine`). 4 H2O / 6-31G\* RHF gradient, four threads:
  56.4 → 29.7 s (1.90×, against an estimated 1.5×).
- The second RMP2 derivative sweep fed only the `PLANCK_DEBUG_RMP2_TERMS`
  print; it now runs only when that flag is set. 1 H2O / 6-31G\* RMP2:
  6.78 → 4.27 s.
- The MP2 sweeps built their `nao²` AO `ShellPair`s once instead of two per
  innermost iteration (superseded by G4). With all of G1: 6.78 → 3.17 s.

Printed gradients identical.

## Original measurement (2026-10-25, before G1)

- The gradient sat at ~99 % CPU at four threads: nothing in `src/gradient/`
  or `mp2_gradient.cpp` carried a pragma.
- 4 H2O / 6-31G\*, one thread: `engine os` 71.4 s, `engine hgp` 47.0 s.
- Profile (8 H2O / 6-31G, one thread, 15 s): `_compute_eri_deriv_elem`
  98.5 % inclusive of `compute_rhf_gradient`; self time led by `_eri_vrr`
  36 %, HRR 16 %, `a0c0` build 13 %, Boys 5 %,
  `EriScratch::resize_for_quartet` 3 %, memmove/memset 7 %, emulated TLS 3 %.
  Nuclear-derivative 1e terms ~5 %.

## Not in this scope

- The stored `nb⁴` tensor (`_compute_2e_auto`, used by MP2/CC/CASSCF/TDDFT and
  conventional SCF) still uses the shell-group hoist, not the block kernels.
  Measure its share of a real MP2/CC run before scoping it.
- Analytic Hessians.

## Invariants carried over

From `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`: fixed work partition for
determinism (4), price on the production loop (5), restructured code is not
bitwise under g++ (7), screening never reaches the Schwarz table (8). And from
the norm-factor gotcha: `Nc` is already in `_coefficients`; the derivative
raise acts on the bare angular part, so the component norm of the
**unraised** function applies (the RI `normC` bug is the same trap).
