# What separates Planck's direct Fock build from PySCF's on cc-pVXZ?

The question started as "why does a Planck shell quartet cost 11–16× what it
costs libcint?" Most of that turned out not to be kernel cost, and the part
that was has since been worked on. This document records what the gap was
made of, how each part was measured, and what is left. It follows
`docs/SCF_VS_PYSCF.md` and `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, whose
measurements are not repeated here. Status of the remaining work is in
`vault/Status/Open Work.md`.

## The answer

The gap had five parts:

| part | what it was | size | outcome |
|---|---|---|---|
| benchmark artifact | PySCF stored the ERI tensor and ran conventional SCF; Planck ran direct | most of the "11–16×" | harness fixed: both run direct |
| Boys-function defect | Planck's Boys core returned too few significant digits at high order | integral errors up to 7.2e-6 on cc-pVTZ | fixed (core v1.2.1) |
| redundant primitives | BSE cc-pVXZ files keep primitives the optimized general contraction drops | 1.5–2.3× of the Fock build | dropped at read time |
| high-L, single-primitive kernels | HGP's VRR does more work than a Rys code when no contraction amortizes it | 1.5–1.9× on cc-pVTZ, all at total L ≥ 4 | 2D Rys block kernel takes these blocks: 1.11–1.16× off the build |
| general contraction | libcint shares primitive work across a shell's contracted functions | 1.6–1.7× to PySCF on cc-pVDZ | worth ≈ 1.1× to Planck; not built |

What still separates the two codes:

- **On the blocks Rys2D owns, libcint is 2.3–2.6× faster.** Four obvious
  layout changes were measured and lost (below). What remains is how
  libcint lays out and walks its 2D tables.
- **Contracted high-L blocks stay on the HGP block kernel.** There the 2D
  Rys form loses, because it pays its tables and assembly per primitive
  quartet while HGP contracts before its HRR.
- **General contraction on cc-pVDZ** (part five above). Planck is faster
  than PySCF on cc-pVDZ in the same (segmented) representation, so this is
  PySCF's edge only against its own native basis.

## How it was measured

- **Fock-level:** Planck iteration-1 `fock_s` against PySCF forced direct,
  warm `get_veff`, `direct_scf_tol` 1e-10, best of 3, on the hcore density.
  PySCF is run both on its native basis and "segmented", meaning its
  general contractions split into separate shells. Segmented is the
  representation Planck evaluates: the same functions, energies within
  1e-13.
- **Kernel-level:** `planck-eri-class-share --libcint nwater basis`, with
  `PLANCK_LIBCINT` pointing at PySCF's bundled `libcint.6.dylib` (e.g. the
  one in `tests/pyscf/.venv`). It loads Planck's basis into libcint 6.1.3
  and times both codes on the same canonical Schwarz-surviving quartets,
  1 thread, libcint with its optimizer and a preallocated cache.
  `--k2` adds the high-L split, the Rys2D column, the accuracy gate and the
  K-by-L crossover table.
- **End to end:** `tests/benchmarks/pyscf_bench.py`, which forces PySCF
  direct and asserts it never stored the tensor.
- **Every integral is checked before it is timed.** That habit is what
  found the Boys defect.

## 1. The benchmark artifact

PySCF's `RHF.get_jk` stores the 8-fold ERI tensor whenever it fits under
`max_memory`, whatever `direct_scf` says. The original 11–16× compared
Planck's direct SCF against PySCF's conventional SCF. Forced direct, the
gap was 1.5–3.7× on cc-pVXZ and none on Pople bases.

## 2. The Boys-function defect

Comparing every integral against libcint before timing it showed errors of
up to 6.6e-8 on 6-31G\*, 4.0e-8 on cc-pVDZ and 7.2e-6 on cc-pVTZ (pure f
quartets), growing with L. Planck's Rys engine, which does not use the
Boys function, matched libcint to 2e-16, while OS, HGP and the block
kernel matched each other. That isolated the fault to the out-of-tree Boys
core, which returned high-order values with too few digits. After v1.2.1
the values agree to 3.4e-14 / 2.6e-14 / 7.2e-13 (relative 3e-15)
(`docs/SCF_VS_PYSCF.md` §5).

## 3. Redundant primitives in the basis files

**The density prescreen was not the work gap.** PySCF's direct build
(`init_direct_scf`: `CVHFnrs8_prescreen`, with `q_cond` and `dm_cond` from
its `_VHFOpt`) applies the same Häser–Ahlrichs rule Planck does. With
PySCF's real arrays on the hcore density at 1e-10:

| fixture | canonical | Schwarz survivors | + density rule |
|---|---|---|---|
| 4 H2O cc-pVDZ | 490,545 | 243,247 | 224,470 (0.92) |
| 4 H2O 6-31G\* | 336,610 | 147,669 | 136,881 (0.93) |
| 2 H2O cc-pVTZ | 408,156 | 354,881 | 338,731 (0.95) |

**The difference was the files.** PySCF's cc-pVXZ uses the optimized
general contraction (Hashimoto, Hirao and Tatewaki, 1995): an outer
primitive that is also its own uncontracted shell of the same L is left
out of the contraction. The BSE GBS files keep it. That is 28 primitives
per water in cc-pVDZ against 24, and 42 against 34 in cc-pVTZ, with the
same span. PySCF forced direct on Planck's files ran 1.66–2.06× slower
than on its own.

`drop_redundant_primitives` (`src/basis/gaussian.cpp`) now removes them
for every orbital basis. Total Fock time over the SCF, stock files before
and after:

| case | 1 thread | 4 threads | energies |
|---|---|---|---|
| 4 H2O cc-pVDZ | 2.09× | 2.05× | Δ 3e-10 |
| 8 H2O cc-pVDZ | 2.34× | 2.29× | Δ 1.2e-9, one iteration fewer |
| 1 H2O cc-pVTZ | 1.47× | 1.58× | identical |
| 2 H2O cc-pVTZ | 1.75× | 1.79× | Δ 1e-10 |

The stock files now reproduce hand-converted files exactly. MO
coefficients and Mulliken, Löwdin and Mayer analyses change, since the
contracted functions do; no committed regression case moved. Pople sets
are unchanged, and RI auxiliary bases are not transformed.

This also invalidated the test B2.2 was declined on (`docs/SCF_VS_PYSCF.md`
§4). That test ran with PySCF in-core, where `get_veff` cannot see the
contraction scheme. Forced direct, libcint does pay for segmented shells.

After these three fixes, direct against direct (before the Rys2D kernel):

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
This table has not been rerun since the Rys2D kernel landed.

## 4. High-L, single-primitive kernels

### Where the cc-pVTZ gap sat

Per-class kernel time against libcint, sums of the per-L rows:

| fixture | Planck | libcint | ratio |
|---|---|---|---|
| 4 H2O 6-31G\* | 0.053 s | 0.072 s | 0.73× |
| 4 H2O cc-pVDZ | 0.159 s | 0.246 s | 0.65× |
| 2 H2O cc-pVTZ | 0.487 s | 0.287 s | **1.70×** |

The 1.70× matches the Fock-level 1.53–1.95× against segmented PySCF. By
total L on 2 H2O cc-pVTZ:

| total L | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9–12 |
|---|---|---|---|---|---|---|---|---|---|---|
| Planck / libcint | 0.43× | 0.46× | 0.37× | 0.96× | 1.45× | 2.34× | 2.82× | 3.81× | 4.52× | 3.7–5.3× |
| Planck s | 0.004 | 0.010 | 0.018 | 0.047 | 0.072 | 0.088 | 0.090 | 0.073 | 0.048 | 0.037 |

L ≥ 4 is 0.41 s of Planck's 0.49 s against libcint's 0.16 s; L ≤ 3 was
already faster. The worst classes are all single-primitive (K = 1):
`(1122|1,1,1,1)` 3.0 µs against 0.65 µs, `(2221|1,1,1,1)` 6.0 against
1.08, `(2322|1,1,1,1)` 21.1 against 3.4. Lanes cannot help a K = 1 block,
which runs `vrr_lanes<1>`. The kernel sweep is 80–82 % of the production
build on cc-pVXZ and 52 % on 6-31G\*.

### Why HGP loses there: the algorithm, not overhead

Splitting the HGP block kernel into Boys / VRR+contract / HRR / readout by
timing progressively fuller sweeps (2 H2O cc-pVTZ, 1 thread, µs per
quartet, from the fifteen costliest K = 1 classes at L ≥ 4):

| class | Boys | VRR | HRR | read | HGP | hoisted Rys | libcint | HGP / lib |
|---|---|---|---|---|---|---|---|---|
| (2222) | 0.02 | 4.60 | 4.45 | 1.15 | 10.22 | 305.7 | 2.05 | 4.98× |
| (1222) | 0.02 | 2.44 | 2.43 | 0.75 | 5.63 | 117.0 | 1.11 | 5.06× |
| (1212) | 0.02 | 1.35 | 1.48 | 0.34 | 3.19 | 49.4 | 0.65 | 4.95× |
| (1122) | 0.02 | 1.12 | 1.20 | 0.25 | 2.58 | 41.2 | 0.67 | 3.85× |
| (1112) | 0.02 | 0.65 | 0.70 | 0.17 | 1.53 | 18.2 | 0.33 | 4.66× |
| (2322) | 0.02 | 7.81 | 9.02 | 1.35 | 18.20 | 760.3 | 3.55 | 5.13× |

Over these classes: Boys 0 %, VRR 44 %, HRR 44 %, readout 12 %. HGP takes
0.051 s against libcint's 0.010 s, 3.6–5.7× in every class.

- **Per-block overhead is not the cause.** The VRR *alone* exceeds
  libcint's whole quartet in every class (4.60 against 2.05 µs on
  `(dd|dd)`), so no cut to HRR, readout or `prepare` could close the gap.
  With K = 1 there is no contraction to amortize the HRR over, and it runs
  on the full Cartesian product.
- **The algorithm is the cause.** HGP's VRR builds \([e0|f0]^{(m)}\) over
  every \(e, f, m\): 3675 entries for `(dd|dd)`, 6370 for `(fd|dd)`. A Rys
  code builds three small tables per root, does its HRR per axis, and
  assembles each component as a sum over \(\lceil (L+1)/2 \rceil\) roots of
  \(I_x I_y I_z\).
- **Planck's existing Rys code was not the answer.** The hoisted Rys block
  (`_contracted_eri_block_hoisted_views`) sums the roots into a 6D box
  *before* its HRR, so the HRR runs on the 6D product. It is 50–210×
  libcint. It agrees with HGP to ≤ 2.7e-14 of each class's largest value.
  A first check divided by each block's own maximum and reported a 73 %
  "defect" in `(1222)`. That block vanishes by symmetry but passes Schwarz
  screening, so HGP returned 3.7e-16 and Rys 0. A ratio against a block's
  own maximum is meaningless when that maximum is roundoff; every
  comparison since uses the class or K-group maximum.

### Prerequisite: Rys roots and weights

The old `rys_roots_weights` (moments, a Stieltjes recurrence, Eigen
Golub–Welsch on a heap `MatrixXd`) cost:

| roots | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| µs | 0.03 | 0.46 | 0.78 | 1.08 | 1.42 | 1.81 | 2.37 | 3.04 |

The 4-root call alone was 1.7× libcint's whole `(pp|dd)` quartet, so no
kernel built on it could win. It is now three pieces (`rys_roots.cpp`):

- **Reference generator:** discretized Stieltjes (Gautschi) on a 60-panel,
  20-point Gauss–Legendre grid in t, with Neumaier-compensated sums.
  Roots come from the Jacobi matrix and weights from the Christoffel
  function. About 30 µs at n = 11; used only to build tables and in the
  test (`rys_roots_weights_reference`).
- **Asymptote** for T ≥ 35 + 7n: generalized Gauss–Laguerre (α = −½),
  with roots \(u_i/T\) and weights \(\lambda_i / (2\sqrt T)\), where it
  meets the generator's ~2e-15 floor.
- **Fits** below that: Chebyshev degree 13 on unit T segments, built per n
  on first use (0.01–0.04 s, 1.3 MB for all n), with one Clenshaw pass for
  all 2n roots and weights.

Three things the obvious plan (fit the old solver's output) ran into:

1. **The old solver was wrong** at T ∈ [1, ~5] for n ≥ 4, because of an
   upward Boys recursion (fixed in `a1cf7403`).
2. **Even fixed, its nodes could not seed fits.** The moment problem is
   ill-conditioned. The rule reproduced moments (≤ 5e-11) while its roots
   and weights were off by up to 1e-7 (n = 8) and 8e-3 (n = 11).
3. **`long double` is `double` on Apple silicon**, so nothing had extra
   precision. Naive summation over 1200 points left ~1e-14 of noise
   (~√N ε) that fits reproduced at any degree; compensated sums fixed it.

Accuracy (`planck-rys-roots`): ≤ 6.3e-15 for n ≤ 5 and 4.2e-14 at n = 11
against the generator. Moments are accurate to 4e-14 to 1.0e-13, against
4.8e-11 for the moment solver. Each call costs 32–43 ns for every n (21×
faster at n = 3, 72× at n = 8). Per-component Rys is 2.1–3.2× faster per
basis; with `engine rys`, ethylene cc-pVDZ is 2.22× faster end to end and
2 H2O cc-pVTZ goes from 241 to 69 s.

### The 2D Rys block kernel

`RysBlockKernel` (`src/integrals/rys_block_eri.h`) is libcint's
formulation, under the `block_eri.h` contract, for total L ≤ 20 (the
11-root limit). For each primitive quartet and root \(r\), on each axis
\(q\):

1. the 1D VRR \(I_q[e][f]\) for \(e \le l_A + l_B\), \(f \le l_C + l_D\);
2. the bra HRR in 1D: \(I_q(a, b{+}1 \mid f) = I_q(a{+}1, b \mid f) + (A_q - B_q)\, I_q(a, b \mid f)\);
3. the ket HRR, the same with \(C_q - D_q\), giving \(I_q(a, b \mid c, d)\).

Then every Cartesian component is \(\sum_r w_r I_x I_y I_z\). The
primitive-pair geometry and the 1D VRR are shared with the per-component
path through `src/integrals/rys_detail.h`. For `(dd|dd)`, the table work
is about 1.6k operations and the assembly about 19k, against HGP's
3675-entry table plus a 6D HRR.

**Accuracy.** Relative to the largest value per K group (K = the number of
primitive quartets), over every class it accepts:

| basis (1 H2O; cc-pVTZ 2 H2O) | K groups | Rys2D − HGP block | libcint − Rys2D |
|---|---|---|---|
| 6-31G | 1–1296 | 5.1e-15 | 3.1e-14 |
| 6-31G\* | 1–1296 | 5.0e-15 | 2.8e-14 |
| cc-pVDZ | 1–4096 | 5.7e-15 | 8.1e-14 |
| cc-pVTZ | 1–4096 | 2.7e-14 | 5.2e-14 |

Against per-component HGP (`planck-hgp-engine-smoke`) it agrees to
≤ 6.4e-15 on the same four bases. Four mutations fail the check:
- one root too few: 5.1e-2;
- a dropped `norm_ratio`: 3.8e-1;
- a negated A − B: 2.5;
- a negated C − D: 2.5.

**The HRR sign is invisible on 1 H2O 6-31G\*.** Hydrogen has no p shell
there, so every canonical pair with L_B > 0 is either on one centre or
has B = H s, and the transfer always multiplies zero. Sign mutations must
be checked on cc-pVDZ or larger.

**Speed, and the one change that mattered.** On the fifteen costliest
K = 1, L ≥ 4 classes of 2 H2O cc-pVTZ, three runs each:

| version | Rys2D s | HGP / Rys2D | kept |
|---|---|---|---|
| first correct version | 0.029–0.031 | 1.64–1.69× | — |
| weight folded into the z table | 0.031 | 1.59–1.62× | no |
| **component offsets cached per class** | 0.0247–0.0263 | 2.01–2.05× | **yes** |
| + root count as a template parameter | 0.0256–0.0262 | 1.93–1.96× | no |
| + VRR/HRR over all roots at once, root innermost | 0.026–0.0275 | 1.88–1.90× | no |
| + roots outer, components streamed | 0.0364 | 1.37–1.39× | no |

The first guess did nothing, so the time was split with early returns
inside `prepare` (µs per quartet, `(dd|dd)`):
- setup and fill: 0.10;
- **component offsets: 1.0**, recomputed from the shells' component views
  on every block;
- tables and assembly: 3.9;
- the harness reading every component: ~1.3, which HGP pays too.

The offsets depend only on the class \((l_A, l_B, l_C, l_D)\), because a
shell's Cartesian components come in a fixed order per L
(`gaussian.cpp`). With three to five roots, a loop over the roots is too
short for unrolling or vectorizing to pay. A first version of the split
read `getenv` inside a loop condition and timed itself at 10×; a
measurement switch must be read once.

Final: 0.0255 s against HGP's 0.0511 s, **2.00× faster**. Every class is
faster (1.70–2.39×). libcint is 0.0106 s, still 2.3–2.6× faster than
Rys2D in every class.

**Where it wins, by contraction depth.** HGP block / Rys2D time (> 1 means
Rys2D is faster), 2 H2O cc-pVTZ, both reading every component:

| K \ L | 0 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.69 | 0.82 | 1.14 | 1.29 | 1.73 | 1.90 | 2.29 | 2.16 | 2.38 | 2.50 | 2.56 | 2.17 |
| 3 | 0.34 | 0.39 | 0.52 | 0.55 | 0.76 | 0.86 | 1.16 | 1.13 | 1.32 | 1.50 | – | – |
| 8 | 0.23 | 0.26 | 0.35 | 0.37 | 0.52 | 0.62 | 0.84 | 0.87 | 1.09 | – | – | – |
| 9 | 0.27 | 0.29 | 0.38 | 0.40 | 0.56 | 0.63 | 0.89 | 0.84 | – | – | – | – |
| ≥ 24 | ≤ 0.29 | ≤ 0.22 | ≤ 0.29 | ≤ 0.31 | ≤ 0.55 | ≤ 0.51 | 0.69 | – | – | – | – | – |

4 H2O cc-pVDZ and 4 H2O 6-31G\* agree to within a few percent wherever
they have classes. HGP contracts its primitive tables before the HRR, so
a deeper contraction costs it one more VRR pass and no more HRR. The Rys
form runs its tables and assembly per primitive quartet. So the two
kernels split the work.

### In production

`src/integrals/auto_block_eri.h` defines Auto's block-kernel chain once,
for the engine, its tests and the benchmark:
- the closed forms for L ≤ 3;
- then `RysBlockKernel` in `crossover` mode, which accepts K = 1 at L ≥ 4
  and K ≤ 3 at L ≥ 7 and declines everything else;
- then the HGP block kernel, which takes the rest up to L = 12.

In crossover mode the Rys kernel's `covers()` is false, because it can no
longer promise every block of an L. The chain still covers L ≤ 12 through
HGP, which the dense-digestion predicate needs.

RHF direct, `engine auto`, Cartesian, best-of-three per-iteration medians,
before and after:

| case | 1 thread | 4 threads | wall, 1 thread |
|---|---|---|---|
| 1 H2O cc-pVTZ | 1.12× | 1.11× | 0.75 → 0.76 s |
| 2 H2O cc-pVTZ | 1.16× | 1.14× | 6.99 → 6.09 s |
| 3 H2O cc-pVTZ | 1.16× | 1.15× | 21.26 → 18.05 s |
| 4 H2O cc-pVDZ | 1.02× | 1.01× | 2.40 → 2.40 s |
| 8 H2O cc-pVDZ | 1.02× | 1.04× | 10.69 → 10.54 s |
| 4 H2O 6-31G\* | 1.01× | 0.99× | 0.83 → 0.86 s |
| 12 H2O 6-31G\* | 1.01× | 1.01× | 9.16 → 9.06 s |

Total energies agree to every printed digit, and iteration counts are
unchanged.
- **The build gains about 1.15×, not 2×,** because Rys2D halves only the
  blocks it takes; contracted blocks stay on HGP.
- **cc-pVDZ gains little:** total L ≥ 4 is about a third of its kernel
  sweep, and only part of that is single-primitive.
- **6-31G\* is unchanged.**

The Fock matrix is bitwise identical at 1, 2, 4 and 8 threads, including
on cc-pVTZ (`planck-fock-accumulate`).

The fast root finder also moved Auto's per-component dispatch table
(`kAutoEngine`), which still governs Auto's stored-tensor and two-phase
builds, screened kernels and L > 12 blocks. Refit from the calibration
benchmark (median of 9): Rys now wins 14 high-L buckets (was 2), HGP 67,
and OS none (was 13). The teaching guide §11 has the full cost surface.

## 5. General contraction

PySCF on its native basis is 1.6–1.7× faster than on the segmented one
for cc-pVDZ, because libcint shares primitive work across the contracted
functions of one shell (O s in cc-pVDZ: 8 primitives, 2 contracted
functions). The same idea is worth much less to Planck, because Planck's
primitive work is cheap: in the segmented representation its kernels are
already 0.65× libcint on cc-pVDZ. The family probe
(`planck-eri-class-share --family`), after the redundant-primitive fix, 1
thread:

| fixture | closed-form families (L ≤ 3) | HGP families (L ≥ 4) | whole sweep |
|---|---|---|---|
| 4 H2O cc-pVDZ | 1.28× | 0.73× | 1.00× |
| 2 H2O cc-pVTZ | 1.37× | 0.78× | 0.84× |
| 4 H2O 6-31G\* | 1.04× | 0.69× | 0.86× |

Lanes already take the HGP half. Using families only for the closed forms
would give about 1.1× of a cc-pVDZ sweep. Visiting blocks in family order
costs ±1 % on cc bases and +3.5–7.4 % on Pople bases
(`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`), so it would have to be a
traversal chosen by basis.

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
| PySCF's density prescreen explains the work gap | **refuted**: it removes 5–8 %, like Planck's | §3 |
| HGP's K = 1 high-L cost is per-block overhead | **refuted**: its VRR alone exceeds libcint's quartet | §4 |
| an existing Rys kernel serves high L | **refuted**: the hoisted 6D block is 50–210× libcint | §4 |
| Rys2D layout variants: folded weight, templated root count, all-roots tables, roots outer | slower or no gain | §4 |
| Rys2D for contracted blocks | slower than HGP at K ≥ 8 except one cell | §4 |
