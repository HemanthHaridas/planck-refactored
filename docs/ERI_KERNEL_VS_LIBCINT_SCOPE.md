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

### K2.1 — A 2D Rys block kernel (scoped 2026-10-25)

**Question:** can a Rys block kernel built libcint's way reach libcint's
cost on the K = 1, total L ≥ 4 classes, and what has to exist first?

**The formulation.** Per primitive quartet: the geometry (as
`_rys_eri_prep`), n = ⌊L/2⌋ + 1 Rys roots and weights for T; per root, the
1D recurrence `_rys_vrr_1d` on each axis up to \((L_{AB}, L_{CD})\); a 2D
HRR per axis to \(I_q(a_q, b_q, c_q, d_q)\) for \(a_q \le L_A\), etc.; then
every Cartesian component as \(\sum_r w_r I_x I_y I_z\). Contracted
primitives accumulate into the component block. The per-axis tables are
tiny (`(dd|dd)`: 3 × 81 entries per root), and the cost is the assembly:
components × roots × 3 operations, 1296 × 5 × 3 ≈ 19k for `(dd|dd)` against
HGP's 3675-entry VRR table plus a 6D HRR. libcint does exactly this and
spends 2.05 µs on `(dd|dd)`.

**Measured: the root finder rules out reusing what exists.**
`Rys::rys_roots_weights` (`rys_roots.cpp`) builds moments in `long double`
(with `erfl`), runs a Stieltjes recurrence, then solves the Jacobi matrix
with Eigen's `SelfAdjointEigenSolver` on a heap-allocated `MatrixXd`
(Golub–Welsch). Per call, T uniform in [0, 40], g++-15 -O3:

| roots | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| µs | 0.03 | 0.46 | 0.78 | 1.08 | 1.42 | 1.81 | 2.37 | 3.04 |

Total L 4–5 needs 3 roots, 6–7 needs 4, 8–9 needs 5. The 4-root call alone
(1.08 µs) is 1.7× libcint's whole `(pp|dd)` quartet (0.65 µs). No kernel on
this finder can win. It also prices every per-component Rys evaluation in
production today (explicit `engine rys`, the Auto table's Rys buckets).

**Steps:**

- **K2.1a, fast roots and weights — DONE (2026-10-25).**
  `rys_roots_weights` (`rys_roots.cpp`) is now three pieces:
  - **Reference generator:** discretized Stieltjes (Gautschi). The Rys
    measure is replaced by a 60-panel, 20-point Gauss–Legendre grid in t;
    the recurrence is built on that discrete measure with Neumaier-
    compensated sums; roots come from the Jacobi matrix and weights from the
    Christoffel function. ~30 µs at n = 11, used only to build tables and in
    the test. Exposed as `rys_roots_weights_reference`.
  - **Asymptote** for T ≥ 35 + 7n: generalized Gauss–Laguerre (α = −½),
    roots \(u_i/T\), weights \(\lambda_i / (2\sqrt T)\). The cutoff is where
    it meets the generator's ~2e-15 floor, read off a measured table per n.
  - **Fits** below it: Chebyshev degree 13 on unit T segments, built per n
    on first use (0.01–0.04 s, 21–270 KB, 1.3 MB for all n), evaluated by
    one Clenshaw pass over all 2n roots and weights.

  The scoped plan (fit the old Golub–Welsch output) did not survive
  measurement, in three ways worth keeping:
  1. **The old solver was wrong** at T ∈ [1, ~5] for n ≥ 4 (upward Boys
     recursion); fixed separately in `a1cf7403`.
  2. **Even fixed, its nodes could not seed fits.** Built from moments, the
     rule reproduces moments (≤ 5e-11) but its individual roots and weights
     were off by up to 1e-7 (n = 8) and 8e-3 (n = 11): the moment problem is
     ill-conditioned. Fitting them would interpolate between samples whose
     errors are not consistent across T.
  3. **`long double` is `double` on Apple silicon** (53-bit mantissa), so no
     part of any of this had extra precision. The first discretized
     Stieltjes still carried ~1e-14 of noise, which the fits reproduced
     whatever their degree. The cause was naive summation over 1200 points
     (~√N ε). Compensated sums brought the generator to ≤ 1.4e-15 for
     n ≤ 5 and 5e-15 at n = 9.

  **Accuracy** (`planck-rys-roots`), in max |Δr| and max |Δw|/F₀ against
  the generator: ≤ 6.3e-15 for n ≤ 5, 4.2e-14 at n = 11, the same at degree
  13, 16 and 19 (the generator's rounding, not the fits). Moments against
  quadrature: 4e-14 (n = 1) to 1.0e-13 (n = 11), against 4.8e-11 for the
  moment solver. The test catches an off-by-one segment (9.7e-6), degree 8
  (6.6e-13), a cutoff of 25 + 5n (1.7e-12) and dropping the compensated sum.
  `planck-rys-box-invariance` (max-box vs per-component, 1e-13) failed at
  1.5e-13 with the first, noisier generator and passes now.

  **Performance:**
  - Root finder, per call, T ∈ [0, 40]: 32–43 ns for every n, against
    0.46–3.04 µs (21× at n = 3, 44× at n = 5, 72× at n = 8).
  - Per-component Rys (`planck-auto-dispatch-benchmark`, 1 thread, the old
    `rys_roots.cpp` relinked into the same build for the "before"): 2.1–3.2×
    per basis (geometric mean over buckets), up to 5.7× in a bucket. By
    total L: unchanged at L ≤ 1 (one root, closed form), 2.8–4.5× at
    L = 2–8, 1.4–1.9× at L ≥ 11 where the recurrences dominate. OS and HGP,
    which do not use the roots, measured 0.94–1.02× (the control).
  - Rys is now the fastest per-component engine in 15 of 238 buckets (none
    before), all high L (He cc-pV5Z, one cc-pVTZ), by 1.0–1.5×. The Auto
    table (`kAutoEngine`) predates this and should be refit.
  - End to end, 4 threads, best of 3, energies identical: ethylene
    cc-pVDZ with `engine rys` 1.55 → 0.70 s (2.22×); water RMP2/STO-3G
    `engine rys` 0.28 → 0.31 s (startup and one-time table build on a case
    needing 1–2 roots); 2 H2O cc-pVTZ with `engine rys` 241.2 → 69.4 s
    (3.48×); the control, 4 H2O cc-pVDZ with `engine auto`, 0.83 → 0.83 s
    (1.00×). Even at 69 s, explicit Rys on cc-pVTZ is far slower than Auto,
    because it still runs one component at a time: that is what K2.1b is
    for.
  - **The default engine gains nothing**, and is not expected to: Auto's
    fused Fock build runs the closed forms and the HGP block kernel, and
    sends only the (7, 8) / (8, 8) buckets to Rys. K2.1a's payoff is
    `engine rys` and, above all, K2.1b, which could not beat libcint on
    the old finder.
- **K2.1b, prototype `RysBlockKernel` in the harness** (scoped 2026-10-26).
  Header-only, under the `block_eri.h` contract, beside `hgp_block_eri.h`.
  It is not a block version of today's per-component Rys path. That path
  (`_rys_eri_build_sum` → `_rys_eri_hrr_to_eri`) sums
  \(w_r I_x I_y I_z\) into a 6D box *before* its HRR, so the HRR runs on
  the 6D product; that is the shape that makes the hoisted block 50–210×
  libcint. The 2D form runs the HRR per axis, per root, on tables of at most
  \((L_A+1)(L_B+1)(L_C+1)(L_D+1)\) entries, and assembles last.

  Back-of-envelope figure (an expectation, not evidence; invariant 1):
  `(dd|dd)` K = 1 is n = 5 roots, 3 axes × 5 roots × (25-entry VRR + 81-entry
  HRR) ≈ 1.6k operations of table work, plus 1296 components × 5 roots × 3
  ≈ 19k in assembly. At 4–8 flops per cycle that is 1–2 µs, against HGP
  10.2 and libcint 2.05. The assembly dominates, so the layout of the
  assembly is what decides the result.

  - **b0, extract (pure move) — DONE (2026-10-26).** `VRR_DIM`,
    `ScreenedKernelData`/`screened_kernel_data`, `RysPrimGeom`,
    `_rys_eri_prep` and `_rys_vrr_1d` moved verbatim from `rys.cpp` into
    `src/integrals/rys_detail.h` (namespace `RysQuad::detail`). Gates:
    `planck-compute-2e`, `planck-hgp-engine-smoke`, `planck-rys-roots` and
    `planck-rys-box-invariance` pass. Outputs are byte-identical to a HEAD
    build, apart from timing columns, on water RMP2 (`engine rys`/`auto`),
    ethylene cc-pVDZ (`rys`/`auto`), the H2O⁺ 6-31G\* UHF case and 4 H2O
    cc-pVDZ direct.
  - **b1, correct, untuned.** For each primitive quartet: prep, then
    `rys_roots_weights(n, T)` with n = ⌊L/2⌋ + 1. For each root and axis,
    run `_rys_vrr_1d` to \((L_A+L_B, L_C+L_D)\), then a 1D HRR in each
    pair to \(I_q(a,b,c,d)\). Accumulate \(\sum_r w_r I_x I_y I_z\) into
    the component block. Apply the norm handling exactly as
    `HgpBlockKernel` does (`coeff_product` carries the first component's
    norm, outputs × `norm_ratio`). Add it as a column in `--k2`.
    - Gate: against the HGP block, ≤ 1e-13 of the **class** maximum (never
      a per-block maximum; see K2.0) on water 6-31G / 6-31G\* / cc-pVDZ /
      cc-pVTZ, every class it accepts, K = 1..9. Also against libcint via
      K0. Mutation-check it: a wrong root count (n − 1), a dropped
      `norm_ratio`, and swapped A/B in the HRR must each fail.
  - **b2, tune for K = 1, L ≥ 4.** Each change is measured on its own in
    `--k2`:
    1. Fold \(w_r\) and the prefactor into one axis table per root, so the
       assembly is two multiplies and an add.
    2. Store the tables root-innermost (`I[q][idx][r]`), so each component
       is a length-n dot product over three contiguous rows.
    3. Compute each component's `(ix, iy, iz)` row indices once per class
       \((L_A, L_B, L_C, L_D)\), in a lazily built table bounded by
       `MAX_L`, and not again per quartet.
    4. Use fixed-size stack tables (n ≤ 11, per-axis L ≤ 2·`MAX_L`) and no
       heap allocation in `prepare`.
    - If it still misses the bar, try roots-outer with the components
      streamed inner. That layout gathers instead of taking a dot product,
      which is why it is second.
  - **Bar to land:** ≥ 2× over the HGP block on the K = 1, L ≥ 4 classes of
    2 H2O cc-pVTZ (summed time, and no class slower than HGP).
    **Stop condition:** under 1.3× after b2 → decline and record why.
    Between 1.3× and 2×, compare against libcint's `g`-tensor layout
    (`g0`/`g2d` strides in `cint2e.c`) before doing more work.
  - **b3, the K table, from the same run.** Rys block against HGP block
    for K = 1..9 at each total L. This is K2.2's input, so K2.2 only has to
    pick the crossover.
- **K2.1c, production.** Chain it in Auto: closed forms (L ≤ 3), then
  `RysBlockKernel` for the L and K it wins, then `HgpBlockKernel`. K is per
  block, so `RysBlockKernel::prepare` declines blocks above its K limit and
  the chain hands them to HGP (`covers()` stays per total L, as the dense
  digestion predicate needs). Coulomb only, as the other block kernels.
  - Follow-on, not part of K2.1: once `RysBlockKernel` lands, rescope G3 of
    `docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md` (the shell-quartet
    derivative block kernel) on top of it. Differentiating the 2D per-axis
    tables is libcint's route to derivative integrals, and it would give
    Rys a derivative kernel, which Phase D lacks.
  - Follow-on, not part of K2.1: once a 2D Rys block exists, the
    conventional Auto build's 6D hoist (`_contracted_eri_block_hoisted_views`,
    50–210× libcint) should go through it, and the hoist be deleted.

**Expected:** if K2.1b reaches libcint on the K = 1 classes, the 2 H2O
cc-pVTZ kernel sweep falls from 0.49 to about 0.24 s and the full build to
about parity with PySCF-segmented (the K2 ceiling above). K2.1a alone does
nothing for the HGP-served classes; its value is as the prerequisite and
for the per-component Rys path.

**Steps after K2.1:**

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
- **Refit the Auto table after K2.1a.** Rys is now the fastest
  per-component engine in 15 of 238 buckets, and `kAutoEngine` (`rys.cpp`)
  still sends it none of them. Rerun `planck-auto-dispatch-benchmark` and
  `scripts/fit_auto_dispatch.py`. This only affects the paths that use the
  per-bucket table, not the fused Fock build. It is cheap and independent of
  K2.1b, but if K2.1c lands first, refit after it.
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
