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
  - `accumulate_shell_pair_eri_gradient` — the shared 2e loop (RHF, RKS, UKS
    via `compute_two_electron_kernel_gradient`); square pair×pair, Schwarz only.
  - `compute_uhf_gradient` (~l. 763) and `compute_rohf_gradient` (~l. 957) —
    **their own copies** of the same loop.
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
