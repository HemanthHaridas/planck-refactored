# What still separates Planck's direct Fock build from PySCF's on cc-pVXZ?

Research scope, rescoped 2026-10-25 after K0 and K1. It expires when the
investigation answers its question; rewrite it then as the answer. It
follows `docs/SCF_VS_PYSCF.md` and `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`
(closed), whose measurements are not repeated here.

## Where it stands

The original question was why a Planck shell quartet cost 11–16× libcint's.
Most of that was not the kernels:

- **A benchmark artifact.** PySCF's `RHF.get_jk` stores the ERI tensor
  whenever it fits, whatever `direct_scf` says, so the 11–16× compared
  Planck's direct SCF against PySCF's conventional SCF. Forced direct, the
  gap was 1.5–3.7× on cc-pVXZ and none on Pople bases (K0).
- **A Boys-function defect**, found by checking every integral against
  libcint before timing it, fixed in core v1.2.1 (K0;
  `docs/SCF_VS_PYSCF.md` §5).
- **Redundant primitives in the basis files**, now dropped at read time:
  2.1–2.3× on cc-pVDZ Fock builds and 1.5–1.8× on cc-pVTZ (K1, below).

**Now**, direct against direct on the hcore density (Planck iteration-1
`fock_s`, PySCF forced direct, warm `get_veff`, `direct_scf_tol` 1e-10, best
of 3). "Segmented" is PySCF's own basis with its general contractions split
into separate shells, the representation Planck evaluates (same functions,
energies within 1e-13):

| case | threads | Planck / PySCF native | Planck / PySCF segmented | segmented / native |
|---|---|---|---|---|
| 4 H2O 6-31G\* | 1 / 4 | 0.92× / 0.48× | 0.92× / 0.48× | 1.00× |
| 12 H2O 6-31G\* | 1 / 4 | 0.85× / 0.89× | 0.85× / 0.88× | 1.00× |
| 4 H2O cc-pVDZ | 1 / 4 | 1.21× / 0.79× | 0.74× / 0.47× | 1.64× / 1.69× |
| 8 H2O cc-pVDZ | 1 / 4 | 1.20× / 1.14× | 0.75× / 0.71× | 1.60× |
| 1 H2O cc-pVTZ | 1 / 4 | 2.75× / 1.02× | 1.95× / 0.73× | 1.41× / 1.39× |
| 2 H2O cc-pVTZ | 1 / 4 | 2.21× / 1.75× | 1.67× / 1.53× | 1.32× / 1.14× |
| 3 H2O cc-pVTZ | 1 / 4 | 2.21× / 2.25× | 1.72× / 1.72× | 1.29× / 1.30× |

(1 H2O cc-pVTZ at four threads is 25 ms and too small to time reliably.)

Two gaps remain, and they separate cleanly:

- **cc-pVTZ: high-L kernel cost (K2).** In the same representation Planck is
  still 1.5–1.9× slower, and the per-class sweep below puts all of it at
  total L ≥ 4.
- **cc-pVDZ: general contraction (G).** In the same representation Planck is
  faster than PySCF (0.47–0.75×). What PySCF has left is general
  contraction, worth 1.6–1.7× to libcint. It is worth much less to Planck
  (G below).

## Measured, and not to be re-run

| claim | status | source |
|---|---|---|
| VRR compute-bound, IPC 5.7 of ~6–7 | measured | ERI_PERFORMANCE_SCOPE |
| no cache or alignment problem; working set ≤ 11 KB | measured | ERI_CACHE_ANALYSIS |
| strided-pointer VRR, shell-granular ShellPair | disproven | ERI_PERFORMANCE_SCOPE |
| contract-once hoisting (old OS kernel) | slower everywhere | ERI_PERFORMANCE_SCOPE |
| lanes (four primitive quartets per pass) | landed, 1.05–1.38× | ERI_SHELL_QUARTET |
| closed forms to total L ≤ 3; HGP block kernel above | landed | ERI_SHELL_QUARTET |
| batched Boys calls (R2) | landed | ERI_SHELL_QUARTET |
| electron-transfer recurrence (D1c) | 1.04–1.06×, declined | ERI_SHELL_QUARTET |
| primitive screening beyond 1e-24 pairs (P2) | declined: G must hold 1e-10 | ERI_SHELL_QUARTET |
| PySCF's density prescreen explains the work gap (old H1) | **refuted**: it removes 5–8 %, like Planck's | K1 |
| redundant primitives in cc-pVXZ GBS files | **fixed**, see K1 | K1 |

### K0: per-class kernel time against libcint

`planck-eri-class-share --libcint nwater basis` (`PLANCK_LIBCINT` pointing
at PySCF's bundled `libcint.6.dylib`) loads Planck's basis into libcint
6.1.3 and times both codes on the same canonical Schwarz-surviving quartets,
1 thread, libcint with its optimizer and a preallocated cache. Values agree
to 3.4e-14 / 2.6e-14 / 7.2e-13 (relative 3e-15) after the Boys fix.

After K1 (sums of the per-L rows):

| fixture | Planck | libcint | ratio | before K1 |
|---|---|---|---|---|
| 4 H2O 6-31G\* | 0.053 s | 0.072 s | 0.73× | 0.73× |
| 4 H2O cc-pVDZ | 0.159 s | 0.246 s | 0.65× | 0.53× |
| 2 H2O cc-pVTZ | 0.487 s | 0.287 s | **1.70×** | 1.15× |

The 1.70× matches the Fock-level 1.53–1.95× against segmented PySCF. By
total L on 2 H2O cc-pVTZ:

| total L | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9–12 |
|---|---|---|---|---|---|---|---|---|---|---|
| Planck / libcint | 0.43× | 0.46× | 0.37× | 0.96× | 1.45× | 2.34× | 2.82× | 3.81× | 4.52× | 3.7–5.3× |
| Planck s | 0.004 | 0.010 | 0.018 | 0.047 | 0.072 | 0.088 | 0.090 | 0.073 | 0.048 | 0.037 |

L ≥ 4 is 0.41 s of Planck's 0.49 s against libcint's 0.16 s; L ≤ 3 is
already faster. The worst classes are all single-primitive:
`(1122|1,1,1,1)` 3.0 µs against 0.65 µs, `(2221|1,1,1,1)` 6.0 against 1.08,
`(2322|1,1,1,1)` 21.1 against 3.4. Lanes cannot help a K = 1 block (it runs
`vrr_lanes<1>`). The kernel sweep is 80–82 % of the production build on
cc-pVXZ and 52 % on 6-31G\*.

### K1: the work gap was in the basis file

PySCF's direct build (`init_direct_scf`: `CVHFnrs8_prescreen`, `q_cond` and
`dm_cond` from its `_VHFOpt`) applies the same Häser–Ahlrichs rule Planck
does. With PySCF's real arrays on the hcore density at 1e-10:

| fixture | canonical | Schwarz survivors | + density rule |
|---|---|---|---|
| 4 H2O cc-pVDZ | 490,545 | 243,247 | 224,470 (0.92) |
| 4 H2O 6-31G\* | 336,610 | 147,669 | 136,881 (0.93) |
| 2 H2O cc-pVTZ | 408,156 | 354,881 | 338,731 (0.95) |

The difference was the files. PySCF's cc-pVXZ uses the optimized general
contraction (Hashimoto, Hirao, Tatewaki 1995): an outer primitive that is
also its own uncontracted shell of the same L is left out of the
contraction. The BSE GBS files keep it: 28 primitives per water in cc-pVDZ
against 24 (42 against 34 in cc-pVTZ), same span. PySCF forced direct on
Planck's files ran 1.66–2.06× slower than on its own.

`drop_redundant_primitives` (`src/basis/gaussian.cpp`) now removes them for
every orbital basis. Total Fock time over the SCF, stock files before and
after:

| case | 1 thread | 4 threads | energies |
|---|---|---|---|
| 4 H2O cc-pVDZ | 2.09× | 2.05× | Δ 3e-10 |
| 8 H2O cc-pVDZ | 2.34× | 2.29× | Δ 1.2e-9, one iteration fewer |
| 1 H2O cc-pVTZ | 1.47× | 1.58× | identical |
| 2 H2O cc-pVTZ | 1.75× | 1.79× | Δ 1e-10 |

The differences are within the bench's `tol_energy` of 1e-9; the stock files
now reproduce hand-converted files exactly. Extended suite 135/139, the four
known rccsdt routing cases. MO coefficients and Mulliken/Löwdin/Mayer
analyses change, since the contracted functions do; no committed regression
case moved. Pople sets are unchanged; RI auxiliary bases are not transformed.

K1 also invalidated the test B2.2 was declined on (`docs/SCF_VS_PYSCF.md`
§4): it ran with PySCF in-core, where `get_veff` cannot see the contraction
scheme. Forced direct, libcint does pay for segmented shells (the table at
the top).

## K2 — High-L single-primitive kernels (the main lever)

**Question:** why does a K = 1 quartet at total L ≥ 4 cost Planck 2.3–5.5×
what it costs libcint, and which kernel should own those classes?

Ceiling: if Planck matched libcint at L ≥ 4, the 2 H2O cc-pVTZ sweep would
fall from 0.49 to about 0.24 s, under libcint's 0.29 s, and the build from
1.7× to about 1.0× of segmented PySCF. Larger bases (cc-pVQZ, aug-) move more
of the work into these classes.

**Hypotheses:**

- **K2-H1, algorithm.** For one primitive quartet, HGP builds the whole
  \([e0|f0]^{(m)}\) table for every auxiliary index m, then runs two HRR
  passes. Rys builds three 2D tables over \(\lceil (L+1)/2 \rceil\) roots and
  forms each Cartesian component as a sum of products. With no contraction
  to amortize over, Rys may simply be fewer operations at high L. libcint
  is a Rys code.
- **K2-H2, fixed cost per block.** `prepare`, scratch sizing, the HRR
  passes and the component readout are paid once per block whatever K is.
  A K = 1 block has nothing to amortize them over (1.1–2.3 µs for a
  one-primitive block in the ERI record).
- **K2-H3, the per-component Rys engine already wins.** Planck has a Rys
  engine; Auto sends these classes to the HGP block kernel because every
  class was a win over the per-component paths. A Rys *block* kernel was
  never built.

**K2.0 result (2026-10-25): the algorithm, not the overhead.**
`planck-eri-class-share --k2 nwater basis` splits the HGP block kernel into
Boys / VRR+contract / HRR / readout by timing progressively fuller sweeps of
the same quartets, and times the existing Rys hoisted block and libcint on
them. 2 H2O cc-pVTZ, 1 thread, µs per quartet, the fifteen costliest K = 1
classes at total L ≥ 4:

| class | Boys | VRR | HRR | read | HGP | Rys hoist | libcint | HGP / lib |
|---|---|---|---|---|---|---|---|---|
| (2222) | 0.02 | 4.60 | 4.45 | 1.15 | 10.22 | 305.7 | 2.05 | 4.98× |
| (1222) | 0.02 | 2.44 | 2.43 | 0.75 | 5.63 | 117.0 | 1.11 | 5.06× |
| (1212) | 0.02 | 1.35 | 1.48 | 0.34 | 3.19 | 49.4 | 0.65 | 4.95× |
| (1122) | 0.02 | 1.12 | 1.20 | 0.25 | 2.58 | 41.2 | 0.67 | 3.85× |
| (1112) | 0.02 | 0.65 | 0.70 | 0.17 | 1.53 | 18.2 | 0.33 | 4.66× |
| (2322) | 0.02 | 7.81 | 9.02 | 1.35 | 18.20 | 760.3 | 3.55 | 5.13× |

Over these classes: Boys 0 %, VRR 44 %, HRR 44 %, readout 12 %; HGP 0.051 s,
libcint 0.010 s. The ratio is 3.6–5.7× in every class.

- **K2-H2 (fixed cost) is not enough.** The VRR *alone* exceeds libcint's
  whole quartet in every class (4.60 against 2.05 µs on `(dd|dd)`), so no
  cut to HRR, readout or `prepare` can close the gap. HRR is real work
  here, not overhead: with K = 1 there is no contraction to amortize it
  over, and it runs on the full Cartesian product.
- **K2-H1 (algorithm) holds.** HGP's VRR builds \([e0|f0]^{(m)}\) over every
  \(e, f, m\): 3675 entries for `(dd|dd)`, 6370 for `(fd|dd)`. A Rys code
  builds three 2D tables per root, does its HRR in 2D, and assembles each
  component as a sum over \(\lceil (L+1)/2 \rceil\) roots of \(I_x I_y I_z\).
- **K2-H3 is half right.** No Rys kernel Planck has is the answer. The
  existing hoisted Rys block (`_contracted_eri_block_hoisted_views`, a 6D
  box per quartet) is 50–210× libcint. What wins is the 2D formulation.
- **The hoisted Rys block agrees with HGP**, to ≤ 2.7e-14 of each class's
  largest value on 1 and 2 H2O cc-pVTZ. A first version of the check divided
  by each block's own maximum and reported a 73 % "defect" in `(1222)`: the
  block was one that vanishes by symmetry but passes Schwarz screening, where
  HGP returned 3.7e-16 and Rys 0. A ratio against a block's maximum is
  meaningless when the maximum is roundoff; the check now uses the class
  maximum.

**Steps now:**

- **K2.1, a 2D Rys block kernel.** libcint's formulation: per primitive
  quartet, Rys roots and weights; per root, the 2D recurrence for
  \(I_x(e, f)\), \(I_y\), \(I_z\) up to \((L_{AB}, L_{CD})\); HRR on the 2D
  tables to \((a, b, c, d)\) per axis; then each component as
  \(\sum_r I_x I_y I_z\). Reuse Planck's root finder (`rys.cpp`). A
  `RysBlockKernel` under the block-kernel contract (`block_eri.h`), chained
  ahead of `HgpBlockKernel` for the classes it wins, decided by `covers()`.
  Prototype it in the harness first, against libcint and the HGP block, on
  the K2.0 classes: build it into production only at ≥ 2× over HGP there.
- **K2.2, which K it should own.** Rys does its assembly per primitive
  quartet; HGP contracts before its HRR. Time both for K = 1, 2, 3, 4 at
  each total L and set the crossover. On cc-pVTZ the high-L polarization
  shells are uncontracted, so K = 1 carries the gap.
- **K2.3, measure in production** on cc-pVTZ and on cc-pVDZ, where L ≥ 4 is
  a smaller share (32 % of the sweep, at 1.3–5.5× libcint), and on 6-31G\*,
  where a change must not cost.
`docs/ERI_PERFORMANCE_SCOPE.md` invariant 1 applies: a profile share or an
operation count is not evidence of removability. A candidate needs the
K2.0 microbenchmark first.

## G — General contraction (the small lever)

**Question:** is sharing primitive work across the contracted functions of
one shell (cc-pVDZ O s: 8 primitives, 2 contracted functions) worth
building in Planck?

Probably not much. The family probe (`planck-eri-class-share --family`),
rerun after K1, 1 thread:

| fixture | closed-form families (L ≤ 3) | HGP families (L ≥ 4) | whole sweep |
|---|---|---|---|
| 4 H2O cc-pVDZ | 1.28× | 0.73× | 1.00× |
| 2 H2O cc-pVTZ | 1.37× | 0.78× | 0.84× |
| 4 H2O 6-31G\* | 1.04× | 0.69× | 0.86× |

PySCF gains 1.6× because libcint's primitive work is expensive. Planck's is
cheap: in the same representation its kernels are already 0.65× libcint on
cc-pVDZ. Lanes already take the HGP half. Using families only for the closed
forms gives about 1.1× of a cc-pVDZ sweep. Visiting blocks in family order
costs ±1 % on cc and +3.5–7.4 % on Pople (`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`),
so it would have to be a traversal chosen by basis.

- **G0.** Rerun the family probe with families restricted to closed-form
  classes, on 4 and 8 H2O cc-pVDZ, at 1 and 4 threads. Build only if the
  whole-build gain is ≥ 1.10× and Pople is unchanged. Do after K2: K2
  changes the L ≥ 4 half that G would leave alone.

## Smaller items

- **End-to-end table.** `tests/benchmarks/pyscf_bench.py` (forced direct)
  for the SCF-level numbers in `docs/SCF_VS_PYSCF.md`, which are still the
  in-core ones.
- **Digestion on Pople bases.** The kernel sweep is only 52 % of the
  6-31G\* build, so the rest of the fused loop costs about as much as the
  kernels there. Planck already beats PySCF on Pople bases, so this is low
  priority.
- **RI auxiliary bases** could take the K1 transform. Measure the primitive
  counts in the shipped `-RIFIT` files first; JK-fit sets are usually
  uncontracted, so it may be zero.

## Gates for anything that lands

- Block sweep against per-component HGP ≤ 1e-14 on water 6-31G / 6-31G\* /
  cc-pVDZ / cc-pVTZ, mutation-verified.
- G identical at 1/2/4/8 threads (fixed partition, ERI record invariant 4).
- Core and extended regressions, and `tests/benchmarks/pyscf_bench.py` wall
  times.
- Any change to screening or to the basis: G ≤ 1e-10 against the reference
  on the same four bases, and SCF energies within 1e-9.

## Priority

K2 first: it is the only lever left on cc-pVTZ and above, where the gap is
largest. G after K2, and only if G0 clears its bar. The gradient work
(`docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md`, gaps of 20–120×) still
outranks all of this.
