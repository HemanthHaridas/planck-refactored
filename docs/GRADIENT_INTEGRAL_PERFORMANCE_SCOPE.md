# Why is an analytic gradient 20–120× the SCF it follows?

Scope for in-flight work (it expires when the work lands; rewrite it then as
the answer). Follows `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, which made the
memory-direct Fock build fast and is closed.

## Where it stands (2026-10-27)

**SCF gradients are done. MP2 gradients are now the problem.** G1–G3 have
landed: the 2e gradient runs by shell quartet on two derivative block kernels,
threaded, and an SCF gradient now costs about as much as the SCF itself.
The MP2/UMP2 gradients still run their own dense, unscreened, single-threaded
`nao⁴` sweeps, and against a now-cheap SCF they are 40× to 2000× it.

Same systems as the original measurement (M3, `build-full`, direct SCF,
`engine auto`, SAD, no symmetry, four threads; "now" is best of three, whole
run including the SCF):

| system | SCF now | gradient run 2026-10-25 | gradient run now | now ÷ SCF |
|---|---|---|---|---|
| 4 H2O / 6-31G\* (nb 76) | 0.39 s | 53.9 s | 0.56 s | 1.4× |
| 8 H2O / 6-31G (nb 104) | 0.55 s | 68.0 s | 0.89 s | 1.6× |
| 1 H2O / cc-pVTZ (nb 65 Cart.) | 0.54 s | 114.9 s | 0.62 s | 1.1× |
| 1 H2O / 6-31G\* RMP2 (conventional) | 0.08 s | 7.30 s | 3.13 s | 39× |
| 2 H2O / 6-31G\* RMP2 (conventional) | 0.17 s | > 5 min | 358 s | ≈ 2100× |

The 2026-10-25 runs had a concurrent compile on the machine (±20 %); the SCF
itself also got faster in between, so compare the gradient columns, not the
ratios across dates.

**Next: G4** (MP2/UMP2 onto the shared loop). Then G5 (1e derivatives, a
larger share now that the 2e term is cheap) and G6 (RI three-centre, measure
first). Two G3 items are also not built: block-level density screening and
screened-kernel (range-separated) block derivatives.

## How the 2e gradient runs now

All in `src/gradient/gradient.cpp` unless noted.

- **One loop for every SCF reference.** RHF, RKS, UHF, ROHF and UKS call
  `accumulate_shell_pair_eri_gradient` with their own Γ.
- **Permutations only change Γ.** In every index swap of a quartet, each
  centre's derivative still goes to that centre's own atom, so
  `pair_permuted_gamma` sums Γ over the (ij)/(kl) swaps and the bra–ket swap,
  and one derivative evaluation serves them all. Exact for any Γ. A zero Γ
  skips the derivative entirely.
- **`engine auto`, Coulomb kernel: shell quartets.**
  `accumulate_block_eri_gradient` walks shell-group pairs Q ≥ P and visits
  exactly the AO quartets the AO loop would (i ≤ j within a group; each
  unordered AO pair of pairs once in a diagonal block). Each block's 12
  derivatives per component come from one primitive pass of
  `auto_deriv_kernel()` (`src/integrals/auto_deriv_block_eri.h`):
  - `RysDerivBlockKernel` (`rys_deriv_block_eri.h`) for one-primitive blocks
    at L ≥ 4 and three-quartet blocks at L ≥ 7 (the energy build's
    crossover, re-measured for derivatives, below). Per-axis
    differentiation of the 2D Rys tables, libcint's route.
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

Still as before:

- 1e derivatives (`_compute_1e_deriv_A`, `_compute_nuclear_deriv_{A,C}_elem`,
  `os.cpp`): per AO pair, serial, the block pasted four times in
  `gradient.cpp`; the C-derivative is re-evaluated per (atom, q).
- `src/post_hf/mp2_gradient.cpp`: the RMP2 sweep (`build_rmp2_gradient_intermediates`)
  and the UMP2 sweep (`build_ump2_gradient_intermediates`) are `nao⁴` loops
  over `atom_aos[atom] × nao³`, unscreened and serial, calling the
  per-component derivative and using only `dI[0..2]` (centre A). The UMP2
  AO pair-density back-transform is fused inside the derivative loop.
- `src/post_hf/ri/ri_eri.cpp`: `compute_3c_eri`, `compute_2c_eri`,
  `compute_3c_deriv_elem` per component, serial, unscreened.

## Remaining work, in order

### G4 — MP2/UMP2 gradients onto the shared loop

Express the dense pair-density term and the separable `vhf1` term as one Γ
through the shell-quartet loop instead of the hand-written `nao⁴` sweeps.
Symmetrize Γ first (`dm2buf_full` carries one bra-derivative permutation),
then let `pair_permuted_gamma` handle the rest. Pull the UMP2 AO pair-density
back-transform out of the derivative loop. This gives MP2 gradients
threading, the block kernels and Schwarz screening at once.

Gate: `water_rmp2_gradient_engine_os_vs_hgp`,
`water_radical_cation_ump2_gradient_smoke`, the RMP2 FD/PySCF cases, the RI-MP2
gradient gates (the dense HF-reference term is shared), plus an `os,auto`
comparator on an RMP2 gradient.

### G5 — 1e derivative integrals

All three Cartesian components per pair in one primitive pass, threaded over
atoms; one shared 1e block instead of four copies. Measure its share first:
it was ~5 % of the old gradient and is now a visible fraction of a ~0.2 s
2e term.

### G6 — RI three-centre integrals (measure first)

`compute_3c_eri` / `compute_3c_deriv_elem` thread over aux shells (disjoint
columns, bitwise). Price their share on an RI-MP2 gradient and RI-JK SCF
before building. D3 (RI pair factors recomputed per iteration,
`docs/ERI_FOCK_BLAS_SCOPE.md`) is the larger RI-JK lever and stays separate.

### Open from G3

- **Block-level density screening**, `Q_ab Q_cd · max|Γ_block| < tol`, the
  Häser–Ahlrichs screen the Fock build uses. The loop screens on Schwarz and
  on a zero component Γ only. The Schwarz table itself stays unscreened
  (invariant 8). Not yet needed at these sizes.
- **Screened (range-separated) kernels in the block derivative kernels** (a
  C5 analogue). HSE06's long-range term still takes the AO loop.

## Record of what landed

### G1 — dead work and routing (2026-10-27)

- `compute_eri_deriv_dispatch` resolves Auto to HGP (precedent:
  `full_symmetry_engine`). 4 H2O / 6-31G\* RHF gradient, four threads:
  56.4 → 29.7 s (1.90×, against an estimated 1.5×).
- The second RMP2 derivative sweep fed only the `PLANCK_DEBUG_RMP2_TERMS`
  print; it now runs only when that flag is set. 1 H2O / 6-31G\* RMP2:
  6.78 → 4.27 s.
- The MP2 sweeps build their `nao²` AO `ShellPair`s once
  (`build_ao_pair_table`) instead of two per innermost iteration. With all of
  G1: 6.78 → 3.17 s.

Printed gradients identical.

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
