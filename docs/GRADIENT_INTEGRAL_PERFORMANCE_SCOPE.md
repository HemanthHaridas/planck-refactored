# Why is an analytic gradient 20–120× the SCF it follows?

Scope for in-flight work (it expires when the work lands; rewrite it then as
the answer). Follows `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, which made the
memory-direct Fock build fast and is closed. None of that work reaches the
derivative integrals.

## Short answer

The gradient's two-electron term still runs the oldest possible shape: one
AO component quartet at a time, each of its 12 derivative components built
from up to 24 independent contracted-ERI evaluations (raise and lower on each
centre, every one a fresh Boys + VRR + HRR per primitive quartet), over the
full **square** of AO pairs, on **one thread**, with Schwarz as the only
screen. `engine auto` routes it to per-component OS, the slowest engine.
Derivative ERIs are 98 % of the gradient (`sample`, 8 H2O / 6-31G).

The RMP2/UMP2 gradients are worse: dense `nao⁴` loops with no screening, two
`ShellPair` constructions (primitive pairs included) in the innermost loop,
and, for RMP2, a second full derivative sweep that only feeds a debug print.

## Measured (M3, `build-full`, 2026-10-25)

Direct SCF, `engine auto`, SAD, no symmetry, `OMP_NUM_THREADS=4` unless noted.
A concurrent compile and concurrent runs were on the machine, so read ±20 %.

| system | SCF single point | gradient | ratio |
|---|---|---|---|
| 4 H2O / 6-31G\* (nb 76) | 2.46 s | 53.9 s | 22× |
| 8 H2O / 6-31G (nb 104) | 1.51 s | 68.0 s | 45× |
| 1 H2O / cc-pVTZ (nb 65 Cart.) | 0.93 s | 114.9 s | 124× |
| 1 H2O / 6-31G\* RMP2 (conventional) | 0.58 s | 7.30 s | 13× |
| 2 H2O / 6-31G\* RMP2 (conventional) | 0.37 s | > 5 min | > 800× |

- The gradient sits at ~99 % CPU at four threads; worker threads spend the
  whole RMP2 gradient in the OpenMP barrier. Nothing in `src/gradient/` or
  `src/post_hf/mp2_gradient.cpp` carries a pragma.
- 4 H2O / 6-31G\*, one thread: `engine os` 71.4 s, `engine hgp` 47.0 s.
  Per-component HGP is ~1.5× OS; routing alone is not the lever.
- Profile (8 H2O / 6-31G, one thread, 15 s): `_compute_eri_deriv_elem` 98.5 %
  inclusive of `compute_rhf_gradient`; self time led by `_eri_vrr` 36 %, HRR
  16 %, `a0c0` build 13 %, Boys 5 %, `EriScratch::resize_for_quartet` 3 %,
  memmove/memset 7 %, emulated TLS 3 %. Nuclear-derivative 1e terms ~5 %.

## Where the logic lives

- `src/gradient/gradient.cpp`
  - `accumulate_shell_pair_eri_gradient` — the one 2e loop (RHF, RKS, UHF,
    ROHF, UKS); since G2 a threaded `q ≥ p` triangle, Schwarz only.
  - The 1e block (`_compute_1e_deriv_A`, `_compute_nuclear_deriv_{A,C}_elem`)
    is pasted four times (~l. 401, 704, 898, 1067); the C-derivative is
    re-evaluated per (atom, q), so each pair's primitives run 3× per atom.
  - `compute_eri_deriv_dispatch` — Auto → OS.
- `src/integrals/{os,hgp}.cpp` `_compute_eri_deriv_elem` — 12 raised
  (2ζ-weighted) + up to 12 lowered contracted ERIs per component; no
  translational invariance (D is computed, not `-(A+B+C)`).
- `src/post_hf/mp2_gradient.cpp` — RMP2 sweeps at ~l. 559 and ~l. 715 (the
  second feeds only `maybe_print_rmp2_term_matrix`, active only under
  `PLANCK_DEBUG_RMP2_TERMS=1`); UMP2 at ~l. 959, with the AO pair-density
  back-transform fused inside the derivative loop. Only `dI[0..2]` (centre A)
  is consumed; all 12 are computed.
- `src/post_hf/ri/ri_eri.cpp` — `compute_3c_eri`, `compute_2c_eri`,
  `compute_3c_deriv_elem`: per component, serial, unscreened; no pragma in
  `src/post_hf/ri/`.

## Plan, in order

Each step lands alone with its gate. Speedups are estimates until measured;
price them on the production loop (ERI scope invariant 5).

### G1 — Delete dead work (cheap, result bitwise)

- Gate the second RMP2 sweep behind `PLANCK_DEBUG_RMP2_TERMS` (or fold its
  four sums into the first sweep when the flag is on). ~2× on RMP2 gradients.
- Build the `nao²` AO `ShellPair`s once, outside the MP2 sweeps.
- Resolve Auto to HGP in `compute_eri_deriv_dispatch` (the F2 precedent,
  `full_symmetry_engine`). ~1.5×. Not bitwise; gated by the existing
  `*_engine_os_vs_hgp` gradient comparators.

Gate: every gradient / geomopt / freq regression, the RMP2/UMP2 FD and PySCF
cases; RMP2 output unchanged to the printed digit.

**Landed 2026-10-27** (all three, the second as "skip unless the flag is
on"). Measured at four threads, M3, quiet machine:

| change | case | before | after |
|---|---|---|---|
| Auto → HGP | 4 H2O / 6-31G\* RHF | 56.4 s (`os`) | 29.7 s, 1.90× |
| skip debug sweep | 1 H2O / 6-31G\* RMP2, `os` | 6.78 s (flag on) | 4.27 s, 1.59× |
| all of G1 | same RMP2 case | 6.78 s | 3.17 s (`auto`), 2.14× |

Auto → HGP gave 1.9×, not the estimated 1.5×. Every printed gradient
component was identical across all three pairs. Extended suite 135/139 (the
four known rccsdt cases). The pair-table hoist was not timed on its own.

### G2 — One threaded, symmetric 2e loop

1. Fold the UHF and ROHF loops onto `accumulate_shell_pair_eri_gradient`
   (their Γ goes in `gamma_fn`). One mechanism before it is threaded.
2. Traverse the pair triangle `q ≥ p` and fold the bra–ket swap into
   `accumulate_eri_gradient_permutations` (Γ is bra–ket symmetric for every
   SCF caller; assert it for any new one). ~2×.
3. Thread over bra pairs with per-bin `natoms×3` partials, bins a fixed
   function of the bra index, summed in fixed bin order, dynamic schedule
   (ERI scope invariant 4: fix the partition, not just the sum order). ~3.5× at
   four threads. Thread-count invariant, not bitwise against serial.

Gate: regressions above; gradients identical at `OMP_NUM_THREADS` 1/2/4/8.

**Landed 2026-10-27**, all three steps, plus two simplifications found on the
way:

- In every index-swap permutation each centre's derivative goes to that
  centre's own atom, so the permutations differ only in Γ. The loop sums Γ
  over the (ij)/(kl) swaps and, for q > p, the bra-ket swap, then applies one
  dI. That is exact for any Γ, so no bra-ket symmetry assertion is needed.
  The zero-Γ test now runs before the derivative integral, not after.
- The `assume_pair_exchange_symmetry = false` fallback (a full `nb²` ordered
  pair list) had no callers and is deleted.
- UKS ran the long-range exchange sweep three times, once per breakdown
  matrix; it now runs once.

4 H2O / 6-31G\* RHF gradient, `engine auto`, M3:

| | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|
| G1 | | | 30.5 s | |
| G2 | 12.8 s | 6.45 s | 3.42 s | 2.43 s |

8.9× over G1 at four threads (16× over the pre-G1 56.4 s). The single-thread
run beats G1 at four threads by 2.4×: G1 never used more than one core, and
the triangle halves the work. All 36 printed components equal G1's; outputs
identical at 1/2/4/8 threads.

**Follow-on landed 2026-10-27:** RKS and UKS ran a second full Coulomb-only
sweep on every gradient to fill `WavefunctionGradientBreakdown`, which only the
`PLANCK_DFT_GRADIENT_DEBUG` report reads. `compute_{rks,uks}_gradient` now take
`with_breakdown` (default false; the DFT driver passes the debug flag), and the
split sweep runs only then. 4 H2O / 6-31G\* B3LYP gradient run, four threads,
SCF included: 7.94 → 5.36 s. Gradient identical in all 36 components; the
debug report is byte-identical when requested.

### G3 — A shell-quartet derivative block kernel (the lever)

Per shell quartet, one primitive pass: VRR to `L_total + 1`, contracted into
four accumulators (weights 2α, 2β, 2γ and 1), HRR once per accumulator; read
raise/lower terms for centres A, B, C out of the blocks and take
`D = −(A + B + C)`. Replaces 12–24 full evaluations per AO component with one
pass per shell quartet: an estimated 10× (s-rich) to 100×+ (d/f), before G2.

- Sibling of `hgp_block_eri.h` behind the same `prepare`/`value`/`covers`
  contract, Coulomb under Auto first; explicit engines and screened kernels
  keep the per-component path (as the Fock build did before C5).
- **Rescoped 2026-10-26: two derivative kernels, split like the Fock build.**
  Since K2.1c the Fock build chains the closed forms, the 2D Rys block
  kernel (`rys_block_eri.h`) for one-primitive blocks at L ≥ 4 and
  three-quartet blocks at L ≥ 7, and the HGP block for the rest
  (`auto_block_eri.h`; crossover in `docs/ERI_KERNEL_VS_LIBCINT.md` §4). The
  derivative pass should follow the same split, for two reasons:
  - **The 2D Rys form differentiates per axis.** A derivative with respect
    to centre A on axis q changes only $I_q$:
    $\partial_{A_q} I_q(a, b \mid c, d) = 2\alpha I_q(a{+}1, b \mid c, d) - a I_q(a{-}1, b \mid c, d)$,
    built from tables one angular step larger, while $I_x I_y I_z$ for the
    other two axes are reused. This is libcint's route, and it keeps the
    per-primitive structure Rys2D already has (the exponent α is known per
    primitive quartet). It is also the only route that gives Rys a
    derivative kernel, which the shell-pair Phase D lacks.
  - **HGP keeps contracted blocks.** The four weighted accumulators above
    (2α, 2β, 2γ, 1) contract before the HRR, exactly why HGP wins those
    blocks in the energy build.
  Build HGP's derivative kernel first (it covers every class), then the Rys2D
  derivative kernel for the classes where the energy-build crossover says
  Rys wins, and re-measure the crossover for derivatives rather than
  assuming it transfers.
- The loop becomes shell-level (the shellpair-granularity Phase D). Add
  block-level density screening, `Q_ab Q_cd · max|Γ_block| < tol`; the Schwarz
  table itself stays unscreened (ERI scope invariant 8).
- Gate: a new unit test against per-component OS `_compute_eri_deriv_elem`,
  every derivative component, water 6-31G / 6-31G\* / cc-pVDZ / cc-pVTZ, ≤ 1e-13,
  mutation-verified (drop a weight, drop the lowered term, break translational
  invariance). Then FD and PySCF gradient regressions, and a new Auto-vs-OS
  gradient comparator (the existing ones compare two explicit engines and
  would not see the block path).

**Progress 2026-10-27: both kernels written and gated, not yet wired.**
`HgpDerivBlockKernel` (`hgp_deriv_block_eri.h`; the energy kernel's HRR moved
into the shared `hgp_block::hrr`) and `RysDerivBlockKernel`
(`rys_deriv_block_eri.h`, per-axis differentiation of the 2D tables). Gate in
`planck-eri-derivative-kernels`, every component and all 12 derivatives against
per-component OS: ≤ 6e-15 relative through cc-pVDZ. On cc-pVTZ the scoped
1e-13 bar is below the floor: per-component OS and HGP themselves differ by
5.4e-13 (2.7e-13 relative), with Rys2D at 1.0e-13 and the HGP block at 1.8e-13,
so the gate is 5e-13 relative. Mutations (2β → β, lowered term dropped,
D without C) each fail on every basis.

**Wired 2026-10-27.** Under `engine auto` with the Coulomb kernel,
`accumulate_block_eri_gradient` walks shell-group pairs Q ≥ P and keeps exactly
the AO quartets of the AO loop (i ≤ j within a group; each unordered AO pair of
pairs once in a diagonal block), with the same Γ folding and the same 64 fixed
bins, one kernel per thread. Blocks the chain declines fall back to the
per-component derivative. Explicit engines and screened kernels keep the AO
loop, so the cross-engine comparators still compare engines.

The chain is `auto_deriv_kernel()` (`auto_deriv_block_eri.h`): Rys2D, then
HGP, with `ChainBlockKernel` gaining a `deriv()` forwarder. Crossover measured
end to end (whole run, four threads, best of two), three variants:

| | energy crossover rule | HGP only | Rys2D everywhere |
|---|---|---|---|
| 4 H2O / 6-31G\* | **0.57 s** | 0.58 s | 0.72 s |
| 4 H2O / cc-pVDZ | **1.46 s** | 1.63 s | 2.02 s |
| 1 H2O / cc-pVTZ | **0.62 s** | 0.68 s | 0.69 s |

The energy build's rule (Rys2D at K = 1 for L ≥ 4, K ≤ 3 for L ≥ 7) also wins
for derivatives, so it stays.

Against G2 (`engine auto`, whole run, four threads): 4 H2O / 6-31G\* 4.75 →
1.21 s, 1 H2O / cc-pVTZ 3.58 → 0.67 s (per-component OS: 10.43 s and 12.23 s).
Gradients identical in every printed component to OS and to G2.

New gates: six `*_engine_os_vs_auto` cases (`tests/engine_gradient_compare.py
--engines os,auto`), covering RHF STO-3G and Cartesian 6-31G\*, UHF, ROHF (C1
CH3), B3LYP, and HSE06/cc-pVTZ (block Coulomb, AO-loop long range).
Re-introducing "D without C" fails them end to end.

**Coverage gap found and fixed on the way.** `run_regressions.py` passed only
the input path to `.py` cases, and every comparator defaults `--build-dir` to
`./build`. So under `--build-dir build-full` the engine comparators
(gradient, geomopt, SCF energy, MPI smoke) silently ran whatever stale binary
sat in `./build`. The runner now passes `--build-dir` and a case's optional
`args` list to `.py` cases.

### G4 — MP2/UMP2 gradients onto the same loop

Express the dense pair-density term and the separable `vhf1` term as one Γ
through G2/G3's loop instead of three hand-written `nao⁴` sweeps. Symmetrize Γ
first (`dm2buf_full` carries one bra-derivative permutation). Pull the UMP2
AO pair-density back-transform out of the derivative loop. This also gives
MP2 gradients Schwarz screening for the first time.

Gate: `water_rmp2_gradient_engine_os_vs_hgp`,
`water_radical_cation_ump2_gradient_smoke`, the RMP2 FD/PySCF cases, RI-MP2
gradient gates (the dense HF-reference term is shared).

### G5 — 1e derivative integrals (measure after G3)

All three Cartesian components per pair in one primitive pass, threaded over
atoms; one shared 1e block instead of four copies. ~5 % of today's gradient;
it becomes visible once G2–G3 land.

### G6 — RI three-centre integrals (measure first)

`compute_3c_eri` / `compute_3c_deriv_elem` thread over aux shells (disjoint
columns, bitwise). Price their share on an RI-MP2 gradient and RI-JK SCF
before building. D3 (RI pair factors recomputed per iteration,
`docs/ERI_FOCK_BLAS_SCOPE.md`) is the larger RI-JK lever and stays separate.

## Not in this scope

- The stored `nb⁴` tensor (`_compute_2e_auto`, used by MP2/CC/CASSCF/TDDFT and
  conventional SCF) still uses the shell-group hoist, not the D1 block
  kernels. Measure its share of a real MP2/CC run before scoping it.
- Analytic Hessians; screened-kernel (range-separated) derivatives in the
  block kernel (a C5 analogue, after G3).

## Invariants carried over

From `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`: fixed work partition for
determinism (4), price on the production loop (5), restructured code is not
bitwise under g++ (7), screening never reaches the Schwarz table (8). And from
the norm-factor gotcha: `Nc` is already in `_coefficients`; the derivative
raise acts on the bare angular part, so the component norm of the
**unraised** function applies (the RI `normC` bug is the same trap).
