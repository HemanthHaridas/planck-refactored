# What still separates Planck's direct Fock build from PySCF's on cc-pVXZ?

Research scope. It expires when the investigation answers its question;
rewrite it then as the answer. It follows `docs/SCF_VS_PYSCF.md` and
`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md` (closed), whose measurements are not
repeated here. Last updated 2026-10-26, after K2.1c.

## Current result

**The 2D Rys block kernel is in production (K2.1c): Auto's Fock build is
1.11–1.16× faster per iteration on cc-pVTZ, with identical energies.**
`src/integrals/auto_block_eri.h` defines the Auto chain once for the engine,
its tests and the benchmark: closed forms (L ≤ 3), then `RysBlockKernel` in
`crossover` mode (K = 1 at L ≥ 4, K ≤ 3 at L ≥ 7), then the HGP block
(4 ≤ L ≤ 12). Before and after on `pyscf_bench.py`'s inputs (RHF, direct,
`engine auto`, Cartesian, best of three per-iteration medians; after is the
production binary from K2.1c):

| case | 1 thread | 4 threads | wall, 1 thread |
|---|---|---|---|
| 1 H2O cc-pVTZ | 1.12× | 1.11× | 0.75 → 0.76 s |
| 2 H2O cc-pVTZ | 1.16× | 1.14× | 6.99 → 6.09 s |
| 3 H2O cc-pVTZ | 1.16× | 1.15× | 21.26 → 18.05 s |
| 4 H2O cc-pVDZ | 1.02× | 1.01× | 2.40 → 2.40 s |
| 8 H2O cc-pVDZ | 1.02× | 1.04× | 10.69 → 10.54 s |
| 4 H2O 6-31G\* | 1.01× | 0.99× | 0.83 → 0.86 s |
| 12 H2O 6-31G\* | 1.01× | 1.01× | 9.16 → 9.06 s |

Total energies agree to every printed digit and iteration counts are
unchanged. cc-pVDZ gains little: total L ≥ 4 is only about a third of its
kernel sweep (K0), and only part of that is single-primitive. 6-31G\* is
neutral, as required. Gates:
- `planck-hgp-engine-smoke` now sweeps Rys2D beside the HGP block against
  per-component HGP: ≤ 6.4e-15 on water 6-31G through cc-pVTZ (4.6 M
  component quartets on cc-pVTZ).
- `planck-fock-accumulate`'s dense-digestion check runs the production
  chain on all four bases. Its thread-count check (G bitwise at
  1/2/4/8 threads) now includes cc-pVTZ, where many blocks reach Rys2D.
- Core 80/80; extended 135/139 (the four known rccsdt routing cases).

The kernel itself: `RysBlockKernel` (`src/integrals/rys_block_eri.h`) is
libcint's formulation. For each primitive quartet and Rys root, it runs a 1D
VRR and a per-axis 1D HRR to \(I_q(a,b,c,d)\), then forms every component as
\(\sum_r w_r I_x I_y I_z\). On the fifteen costliest single-primitive
(K = 1) classes at total L ≥ 4 of 2 H2O cc-pVTZ, one thread:

| kernel | time | per class | against libcint |
|---|---|---|---|
| HGP block | 0.0511 s | — | 4.0–5.5× slower |
| **Rys2D** | **0.0255 s** | **1.70–2.39× faster than HGP**, none slower | 2.3–2.6× slower |
| libcint 6.1.3 | 0.0106 s | — | — |

- **Accuracy.** It agrees with the HGP block to ≤ 2.7e-14 and with libcint
  to ≤ 8.1e-14, relative to the largest value per K group. That covers
  every class it accepts on water 6-31G, 6-31G\*, cc-pVDZ and cc-pVTZ,
  K = 1 to 4096. Four mutations fail the gate (K2.1b below).
- **Where it wins** (b3). Rys2D wins at K = 1 for L ≥ 3 (1.14–2.56×) and at
  K = 3 for L ≥ 7 (1.10–1.50×). HGP wins everywhere else, down to 0.17×,
  because it contracts before its HRR, while Rys2D works per primitive
  quartet. The closed forms already own L ≤ 3, so the production rule is
  K = 1 at L ≥ 4 and K ≤ 3 at L ≥ 7.
- **Why the build gains 1.15× and not 2×.** Rys2D halves only the blocks it
  takes. On 2 H2O cc-pVTZ the kernel sweep is 80–82 % of the build and
  L ≥ 4 is 0.41 s of its 0.49 s, but contracted blocks stay on HGP.

**Next:**
- **Refit the Auto table** (`kAutoEngine`, Smaller items). It has been
  stale since K2.1a and is cheap.
- **Rescope G3 of `docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md`** (the
  shell-quartet derivative block kernel) on Rys2D. Differentiating the 2D
  per-axis tables is libcint's route to derivative integrals, and it gives
  Rys a derivative kernel, which Phase D lacks. The gradient gap (20–120×)
  outranks everything else here.
- Route the conventional Auto build's 6D hoist
  (`_contracted_eri_block_hoisted_views`, 50–210× libcint) through Rys2D,
  and delete the hoist.
- If the remaining 2.3–2.6× to libcint matters, compare against libcint's
  `g`-tensor layout (`g0`/`g2d` strides in `cint2e.c`). The obvious layout
  changes are already measured and lost (b2).

## Where the gap stands

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
  2.1–2.3× on cc-pVDZ Fock builds and 1.5–1.8× on cc-pVTZ (K1).

Direct against direct on the hcore density (Planck iteration-1 `fock_s`,
PySCF forced direct, warm `get_veff`, `direct_scf_tol` 1e-10, best of 3),
before K2.1c. "Segmented" is PySCF's own basis with its general contractions
split into separate shells, the representation Planck evaluates (same
functions, energies within 1e-13):

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

- **cc-pVTZ: high-L kernel cost (K2).** In the same representation Planck was
  1.5–1.9× slower, all of it at total L ≥ 4 (K0). K2.1c took 1.11–1.16×
  off the build; the table above has not been rerun against PySCF since.
- **cc-pVDZ: general contraction (G).** In the same representation Planck is
  faster than PySCF (0.47–0.75×). What PySCF has left is general
  contraction, worth 1.6–1.7× to libcint and much less to Planck (G below).

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
| redundant primitives in cc-pVXZ GBS files | **fixed** | K1 |
| HGP's K = 1 high-L cost is per-block overhead (K2-H2) | **refuted**: its VRR alone exceeds libcint's quartet | K2.0 |
| any existing Rys kernel serves high L | **refuted**: hoisted 6D block is 50–210× libcint | K2.0 |
| Rys2D layout variants: folded weight, templated root count, all-roots tables, roots-outer | all slower or no gain | K2.1b b2 |

## K0: per-class kernel time against libcint

`planck-eri-class-share --libcint nwater basis` (`PLANCK_LIBCINT` pointing
at PySCF's bundled `libcint.6.dylib`, e.g. the one in `tests/pyscf/.venv`)
loads Planck's basis into libcint 6.1.3 and times both codes on the same
canonical Schwarz-surviving quartets, 1 thread, libcint with its optimizer
and a preallocated cache. Values agree to 3.4e-14 / 2.6e-14 / 7.2e-13
(relative 3e-15) after the Boys fix.

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

## K1: the work gap was in the basis file

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
scheme. Forced direct, libcint does pay for segmented shells (the table
above).

## K2: high-L single-primitive kernels

**Question:** why does a K = 1 quartet at total L ≥ 4 cost Planck 2.3–5.5×
what it costs libcint, and which kernel should own those classes?

Ceiling: if Planck matched libcint at L ≥ 4, the 2 H2O cc-pVTZ sweep would
fall from 0.49 to about 0.24 s, under libcint's 0.29 s, and the build from
1.7× to about 1.0× of segmented PySCF. Larger bases (cc-pVQZ, aug-) move more
of the work into these classes.

### K2.0: the algorithm, not the overhead (2026-10-25)

`planck-eri-class-share --k2 nwater basis` splits the HGP block kernel into
Boys / VRR+contract / HRR / readout by timing progressively fuller sweeps of
the same quartets. 2 H2O cc-pVTZ, 1 thread, µs per quartet, from the
fifteen costliest K = 1 classes at total L ≥ 4 (the column "Rys hoist" is
the old 6D hoisted block, since replaced in the harness by Rys2D):

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

- **Fixed cost per block (K2-H2) is not enough.** The VRR *alone* exceeds
  libcint's whole quartet in every class (4.60 against 2.05 µs on
  `(dd|dd)`), so no cut to HRR, readout or `prepare` can close the gap.
  With K = 1 there is no contraction to amortize the HRR over, and it runs
  on the full Cartesian product.
- **The algorithm (K2-H1) is the cause.** HGP's VRR builds
  \([e0|f0]^{(m)}\) over every \(e, f, m\): 3675 entries for `(dd|dd)`,
  6370 for `(fd|dd)`. A Rys code builds three 2D tables per root, does its
  HRR in 2D, and assembles each component as a sum over
  \(\lceil (L+1)/2 \rceil\) roots of \(I_x I_y I_z\).
- **No existing Rys kernel was the answer (K2-H3).** The hoisted block
  (`_contracted_eri_block_hoisted_views`) sums the roots into a 6D box
  before its HRR and is 50–210× libcint. It agrees with HGP to ≤ 2.7e-14
  of each class's largest value. A first check divided by each block's own
  maximum and reported a 73 % "defect" in `(1222)`. The block vanishes by
  symmetry but passes Schwarz screening, so HGP returned 3.7e-16 and Rys 0.
  Every comparison since uses the class (or K group) maximum.

### K2.1a: fast Rys roots and weights (done 2026-10-25)

The old `rys_roots_weights` (moments, Stieltjes, Eigen Golub–Welsch on a
heap `MatrixXd`) cost 0.46–3.04 µs per call for 2–8 roots. The 4-root call
alone was 1.7× libcint's whole `(pp|dd)` quartet, so no kernel on it could
win. It is now three pieces (`rys_roots.cpp`):

- **Reference generator:** discretized Stieltjes (Gautschi) on a 60-panel,
  20-point Gauss–Legendre grid in t, with Neumaier-compensated sums; roots
  from the Jacobi matrix, weights from the Christoffel function. ~30 µs at
  n = 11, used only to build tables and in the test
  (`rys_roots_weights_reference`).
- **Asymptote** for T ≥ 35 + 7n: generalized Gauss–Laguerre (α = −½),
  roots \(u_i/T\), weights \(\lambda_i / (2\sqrt T)\), where it meets the
  generator's ~2e-15 floor.
- **Fits** below it: Chebyshev degree 13 on unit T segments, built per n on
  first use (0.01–0.04 s, 1.3 MB for all n), one Clenshaw pass for all 2n
  roots and weights.

The scoped plan (fit the old Golub–Welsch output) did not survive
measurement, in three ways worth keeping:
1. **The old solver was wrong** at T ∈ [1, ~5] for n ≥ 4 (upward Boys
   recursion); fixed separately in `a1cf7403`.
2. **Even fixed, its nodes could not seed fits.** The moment problem is
   ill-conditioned: the rule reproduced moments (≤ 5e-11) while its roots
   and weights were off by up to 1e-7 (n = 8) and 8e-3 (n = 11).
3. **`long double` is `double` on Apple silicon**, so nothing had extra
   precision. Naive summation over 1200 points left ~1e-14 of noise (~√N ε)
   that the fits reproduced at any degree; compensated sums fixed it.

Accuracy (`planck-rys-roots`): ≤ 6.3e-15 for n ≤ 5, 4.2e-14 at n = 11
against the generator; moments 4e-14 to 1.0e-13 (the moment solver:
4.8e-11). The test catches an off-by-one segment, degree 8, a cutoff of
25 + 5n and dropping the compensated sum. `planck-rys-box-invariance`
passes.

Performance: 32–43 ns per call for every n (21× at n = 3, 72× at n = 8).
Per-component Rys 2.1–3.2× per basis; OS/HGP control 0.94–1.02×. End to
end with `engine rys`: ethylene cc-pVDZ 2.22×, 2 H2O cc-pVTZ 241 → 69 s.
`engine auto` is unchanged (1.00×), as expected, since Auto's fused build
does not use per-component Rys.

### K2.1b: the 2D Rys block kernel (done 2026-10-26)

`RysBlockKernel`, header-only under the `block_eri.h` contract, total L ≤ 20
(the 11-root limit). Back-of-envelope before building: `(dd|dd)` is
≈ 1.6k operations of table work and ≈ 19k of assembly, so 1–2 µs, against
HGP's 10.2 µs and libcint's 2.05 µs.

- **b0, shared pieces** (`3e497504`). `VRR_DIM`, `screened_kernel_data`,
  `RysPrimGeom`, `_rys_eri_prep` and `_rys_vrr_1d` moved verbatim from
  `rys.cpp` into `src/integrals/rys_detail.h`. Outputs byte-identical to
  the previous build apart from timing columns (water RMP2 and ethylene
  cc-pVDZ with `engine rys`/`auto`, the H2O⁺ 6-31G\* UHF case, 4 H2O
  cc-pVDZ direct). `planck-compute-2e`, `planck-hgp-engine-smoke`,
  `planck-rys-roots` and `planck-rys-box-invariance` pass.
- **b1, correct and untuned** (`da39b154`). `--k2` times Rys2D in place
  of the hoisted block and gates it on every class it covers, against
  the HGP block and libcint. Results are relative to the largest value
  per K group, where K is the number of primitive quartets:

  | basis (1 H2O; cc-pVTZ 2 H2O) | K groups | Rys2D − HGP | libcint − Rys2D |
  |---|---|---|---|
  | 6-31G | 1–1296 | 5.1e-15 | 3.1e-14 |
  | 6-31G\* | 1–1296 | 5.0e-15 | 2.8e-14 |
  | cc-pVDZ | 1–4096 | 5.7e-15 | 8.1e-14 |
  | cc-pVTZ | 1–4096 | 2.7e-14 | 5.2e-14 |

  Mutations: one root too few fails at 5.1e-2, a dropped A `norm_ratio`
  at 3.8e-1, and a negated A − B or C − D at 2.5. **The HRR sign is
  invisible on 1 H2O 6-31G\***: hydrogen has no p shell there, so every
  canonical pair with L_B > 0 is on one centre or has B = H s, and the
  transfer always multiplies zero. Check sign mutations on cc-pVDZ or
  larger. Untuned it was 1.63× faster than HGP.
- **b2, tuning: one change mattered.** The first guess did nothing, so
  the time was split before anything else. For `(dd|dd)`, with early
  returns inside `prepare` (µs per quartet):
  - setup and fill 0.10;
  - **component offsets 1.0**, from walking `views[s][c]->_cartesian` on
    every block;
  - primitive loop (tables and assembly) 3.9;
  - the harness's read of every component ~1.3, which HGP pays too.

  A first probe read `getenv` in a loop condition and timed itself at
  10×; switches must be read once. The 15 classes, three runs each:

  | change | Rys2D s | HGP / Rys2D | kept |
  |---|---|---|---|
  | b1 | 0.029–0.031 | 1.64–1.69× | — |
  | weight folded into the z table | 0.031 | 1.59–1.62× | no |
  | **offsets cached per class** | 0.0247–0.0263 | 2.01–2.05× | **yes** |
  | + root count as a template parameter | 0.0256–0.0262 | 1.93–1.96× | no |
  | + VRR/HRR over all roots at once, root innermost | 0.026–0.0275 | 1.88–1.90× | no |
  | + roots outer, components streamed | 0.0364 | 1.37–1.39× | no |

  The offsets depend only on the class, because a shell's Cartesian
  components come in a fixed order per L (`gaussian.cpp`). Root-innermost
  storage was already b1's layout, and `prepare` now allocates nothing in
  steady state. With 3–5 roots, an inner loop over the roots is too short
  for unrolling or vectorizing to pay. The bar (≥ 2×, no class slower)
  is met.
- **b3, crossover by K and total L.** HGP block / Rys2D time, both reading
  every component, best of three per class, 2 H2O cc-pVTZ, 1 thread;
  > 1 means Rys2D is faster:

  | K \ L | 0 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 0.69 | 0.82 | 1.14 | 1.29 | 1.73 | 1.90 | 2.29 | 2.16 | 2.38 | 2.50 | 2.56 | 2.17 |
  | 3 | 0.34 | 0.39 | 0.52 | 0.55 | 0.76 | 0.86 | 1.16 | 1.13 | 1.32 | 1.50 | – | – |
  | 8 | 0.23 | 0.26 | 0.35 | 0.37 | 0.52 | 0.62 | 0.84 | 0.87 | 1.09 | – | – | – |
  | 9 | 0.27 | 0.29 | 0.38 | 0.40 | 0.56 | 0.63 | 0.89 | 0.84 | – | – | – | – |
  | ≥ 24 | ≤ 0.29 | ≤ 0.22 | ≤ 0.29 | ≤ 0.31 | ≤ 0.55 | ≤ 0.51 | 0.69 | – | – | – | – | – |

  4 H2O cc-pVDZ and 4 H2O 6-31G\* agree to within a few percent wherever
  they have classes (L ≤ 8). The K = 8, L = 9 cell (1.09×) is not worth
  a rule.

`docs/ERI_PERFORMANCE_SCOPE.md` invariant 1 applies to anything further: a
profile share or an operation count is not evidence of removability, and a
candidate needs the `--k2` microbenchmark first.

## G: general contraction (the small lever)

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
  whole-build gain is ≥ 1.10× and Pople is unchanged. Do after K2.1c, which
  changes the L ≥ 4 half that G would leave alone.

## Smaller items

- **Refit the Auto table.** After K2.1a, Rys is the fastest per-component
  engine in 15 of 238 buckets, and `kAutoEngine` (`rys.cpp`) still sends it
  none of them. Rerun `planck-auto-dispatch-benchmark` and
  `scripts/fit_auto_dispatch.py` after K2.1c. This only affects the paths
  that use the per-bucket table, not the fused Fock build.
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

The gradient work (`docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md`, gaps of
20–120×) outranks everything left here, and its G3 should now be rescoped
on Rys2D. Within this scope: the Auto-table refit (cheap), then the libcint
layout comparison only if the remaining cc-pVTZ gap matters, then G0.
