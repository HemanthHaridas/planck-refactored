# Shell-Quartet ERI Kernels for the Fused Fock Build — Scope

**Question:** How much faster can the memory-direct (fused) Fock build get by
evaluating whole shell quartets at once, in closed form at low angular
momentum and with a block kernel above it, and what else in the build
stands in the way?

**Status: in flight.** Landed: C0, C1, C2–C3.5, H1, L1, L2, L3a–c, P1, S1,
D1, F1, F2, F4, T1, 2b, K1 (`5fd08496`). In the working tree, gated, not
committed: K1.4 (dense digestion under S1, with K1.4b). Measured and
declined: D1c (electron-transfer recurrence), 5b (a class-batched
scheduler). Measured, not landed: D2a/B0 (shared-exponent families), B1
(family-quartet loop order, parked as a git stash until B2). Open, in order:
B2, the fixed accumulator partition, C5, P2, total L > 12, F3; D3 is
independent, K2 conditional, C4b deprioritised. When those land or are
abandoned, rewrite this as an answer.

## Answer so far

| fixture (one thread, `engine auto`) | Fock build now | before this work | speedup |
|---|---|---|---|
| 8 H2O / 6-31G (nb = 104) | **0.113 s** | 1.50 s | **~13×** |
| 4 H2O / 6-31G\* (nb = 76) | **0.070 s** | ~1.12 s† | **~16×** |
| 4 H2O / cc-pVDZ | **0.410 s** | 2.573 s§ | **6.3×**§ |
| 1 H2O / cc-pVTZ | **0.076 s** | 1.631 s§ | **21×**§ |
| naphthalene / 6-31G, D2h (nb = 106) | **0.206 s**‡ | 1.454 s | **7.1×** |

`planck-eri-class-share`, Apple M3, `build-full`, `OMP_NUM_THREADS=1`, best
of five (the `--k1` production arm, 2026-10-22). "Before" is the build before
C1, after the Boys library fix (C0), which gave another 1.37× / 1.13× on its
own. † The product of the per-step ratios; never timed before C1. § Against
the D0 build (after P1, before S1 and D1). ‡ With `sym_ops`, on a
group-averaged density. Explicit engines get only the engine-independent
steps; on 8 H2O / 6-31G and 4 H2O / 6-31G\* Auto is now ~7× and ~11× faster
than explicit `engine hgp`. Fock-build numbers, not SCF wall time. The ~200×
libcint gap in `docs/ERI_PERFORMANCE_SCOPE.md` predates this work and has
not been re-measured.

Per step, each a ratio over the build with all previous steps:

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c | D1 | 2b | K1 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | 1.49× | 1.05× | 1.14× | **~13×** |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | 4.90× | 1.23× | 1.17× | **~16×** |

On the symmetric fixture: L3a 1.70×, L3b 1.23×, L3c+P1 1.16×, S1 1.33×,
D1 ~1.5×, 2b 1.07×, K1.4 1.32× (1.454 → 0.855 → 0.702 → 0.597 → 0.454 →
0.295 → 0.274 → 0.206 s).

**In short:**
- **Evaluate shell quartets, not components.** Every engine returned one
  Cartesian component per call and reran the whole primitive loop for each,
  27–78× per shell quartet wherever HGP ran. Removing that repetition is
  most of the speedup: closed forms at total L ≤ 3, an HGP block kernel at
  4–12.
- **Layout decides whether that wins.** The block kernel is the same idea as
  an earlier contract-once hoist that measured 1.14–1.62× *slower*; the
  difference is indexing Cartesian functions by total L instead of a
  per-axis cube.
- **Inside the kernels, what is left is per-primitive-quartet overhead.** A
  primitive quartet writes only 180–300 ket elements into a 5–7.5 KB table,
  at ~6 cycles per element. A cheaper recurrence (D1c) or a better layout
  does not help; several primitive quartets per pass (lanes, 2b, 1.05–1.38×)
  and fewer primitive passes (families, ~1.7× on cc kernels, open) do.
- **Outside the kernels, the cost was digesting one component at a time.**
  Every value was read out, filtered and scattered into G alone. Digesting
  each block densely (K1) gave 1.11–1.22×, and under the skeleton scheme
  (K1.4) 1.32×. This, not lane fill, is why batching blocks (5b) had nothing
  to buy.

## How the build works now

`Integrals::fused_fock_build` (`src/integrals/fused_fock.h`) chooses one of
two traversals **per build**. OpenMP `schedule(static)` and the MPI stripe
run over bra group pairs in both; partials are keyed by
`omp_get_thread_num()` and summed in thread order.

**Dense (K1, K1.4).** Taken when a block kernel is given (Auto), the kernel
is Coulomb, every shell is at most f (total L ≤ 12), every density is
symmetric to 1e-10, and there is either no symmetry or skeleton mode. Per
bra pair p, kets q ≥ p:

1. Block Schwarz. Under S1, each op's image of the block is (min, max) of
   two entries of a per-build table `op_pair[o][p]`; an image that sorts
   lower skips the block, and permuting ops that map it onto itself go to
   `block_ops`.
2. `prepare` the block, then every component above the component Schwarz
   bound: v′ = deg · weight · value, deg = (A=B?1:2)(C=D?1:2)(p=q?1:2).
   Under S1, weight is the S1 weight of the component's canonical form:
   identity-map ops (the identity, a planar molecule's mirror) need only a
   sign test, and only `block_ops` need canonicalization.
3. Six writes per component (two Coulomb, four exchange), from a body
   instantiated per active write group, then (G + Gᵀ)/2 after the merge
   (and the group average under S1).

**Per-component.** Everything else: explicit engines, screened kernels,
L > 12, non-symmetric densities, and replication-mode symmetry.

1. **Block filters.** Ket group pairs start at `first_pair_of[bra.a]` (L1);
   block Schwarz; with symmetry, blocks an operation sends below the bra's
   first shell are skipped (L3c).
2. **Component filters.** Canonical order, component Schwarz, and with
   symmetry the orbit test.
3. **Values.** The block kernel `prepare`s at the first surviving component;
   a declined block uses per-component `eri_elem`.
4. **Accumulation.** Each value goes into G over its distinct 8-fold
   permutations (`distinct_eri_orbit`, L2), once per orbit element in
   replication mode, once weighted in skeleton mode.

Block kernels (`src/integrals/block_eri.h`): `NoBlockKernel` for explicit
engines; for Auto, `ClosedFormBlockKernel{3}` (`low_l_eri.h`) chained with
`HgpBlockKernel{4, 12}` (`hgp_block_eri.h`, 4 lanes in blocks with K ≥ 3),
one copy per thread. Both are Coulomb only.

## How it is measured

`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a
ctest). With no arguments it drives `fused_fock_build` on 1, 4 and 8 waters
(6-31G, 6-31G\*), one and four waters in cc-pVDZ, one water in cc-pVTZ and
naphthalene / 6-31G (axis-aligned D2h, group-averaged density), printing
per-class HGP times, the closed-form builds at L ≤ 1, 2, 3, the block kernel
one total-L class at a time, the symmetric build with and without
`sym_ops`, and an FNV-1a **fingerprint of G** per build. Modes, where
`nwater` 0 is naphthalene:

- `--profile N [nwater basis]`: N production builds, for `sample`.
- `--family nwater basis [within]`: the D2a family probe.
- `--census nwater basis`: the 5b S0 lane-fill census.
- `--k1 nwater basis`: production with `dense_digest` off (today's loop)
  against on, plus the census's unmasked RHF dense arm.

Rules learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling; one binary
  alternated 1.0 / 1.22 s). Compare interleaved binaries or one binary with
  a switch (`dense_digest` is one). A baseline binary for an uncommitted
  change is built by compiling the affected objects from the old headers and
  relinking with the target's own `link.txt`.
- **A fingerprint, not a timestamp, says which code ran**, and a failed
  build silently reruns the old binary. Check the exit status and the new
  symbols (`nm`).
- **Fingerprints depend on the fixture list**: the test density comes from
  `Eigen::MatrixXd::Random`, whose state carries across fixtures, so new
  fixtures go at the end.
- **Attribute before predicting.** Inlined helpers hide in their caller's
  self time. Temporary builds route phases through named `noinline`
  wrappers (`template <int Tag> phase(f)`; anonymous lambdas get numbered
  names) and add counters. They are never committed.
- **zsh does not word-split an unquoted variable.** `for f in "8 6-31g" ...;
  ./bench --k1 $f` passes one argument, misses the mode and runs the full
  default sweep for minutes with no matching output. Use `${=f}`.

## Where the time goes

Self time of the production build after D1 (before 2b and K1), one thread,
~6 000 `sample` frames per fixture. K1 removed most of the "fused loop
body" and "G writes" rows on the symmetric densities it covers, and K1.4
the `skeleton_orbit_weight` row; the kernel rows stand:

| frame | 8 H2O / 6-31G | naphthalene D2h | 4 H2O / 6-31G\* | 4 H2O / cc-pVDZ | 1 H2O / cc-pVTZ |
|---|---|---|---|---|---|
| HGP block kernel: VRR ket | 4.0 % | 8.8 % | **24.4 %** | **20.9 %** | **40.7 %** |
| HGP block kernel: VRR bra | 0.4 % | 1.0 % | 3.0 % | 2.6 % | 2.7 % |
| HGP block kernel: contract | 0.3 % | 0.8 % | 3.1 % | 2.5 % | 3.6 % |
| HGP block kernel: HRR, setup, readout | 0.5 % | 0.8 % | 7.1 % | 4.0 % | 11.3 % |
| closed-form kernel | **30.0 %** | **28.4 %** | 15.8 % | **24.8 %** | 7.8 % |
| closed-form readout | 10.3 % | 4.6 % | 5.2 % | 4.2 % | 1.2 % |
| Boys (both kernels) | **26.4 %** | **22.5 %** | 13.1 % | **22.6 %** | 5.3 % |
| fused loop body | 13.9 % | 12.7 % | 8.2 % | 4.7 % | 4.8 % |
| G writes | 13.6 % | 6.2 % | 13.9 % | 9.7 % | 14.7 % |
| `skeleton_orbit_weight` (S1) | — | **13.2 %** | — | — | — |

The HGP block kernel's cost, measured by primitive quartet and by block
(temporary timers and counters):

| | 4 H2O / 6-31G\* | 4 H2O / cc-pVDZ | 1 H2O / cc-pVTZ |
|---|---|---|---|
| share of the build | 38 % | 31 % | 60 % |
| primitive quartets per build | 83 k | 432 k | 121 k |
| ket elements per primitive quartet | 203 | 183 | 305 |
| VRR table per primitive quartet | 5.1 KB | 4.5 KB | 7.5 KB |
| per primitive quartet, blocks with K ≥ 9 | 290–430 ns | 290–410 ns | 320–650 ns |
| a block with K = 1 | 1.2 µs | 1.1 µs | 2.3 µs |
| kernel time in K = 1 blocks | 10 % | 8 % | 23 % |
| HRR and readout (`finish()`) | 16 % | 10 % | 15 % |

K is the number of primitive quartets in a block after P1. The ket VRR runs
at ~3 GFLOP/s: it is the fixed cost of a small recurrence per primitive
quartet, not arithmetic and not memory. The closed forms spend 49–65 % of
their time in blocks with K > 64, at 7–13 ns per primitive quartet, about
half of it in Boys.

**Shared exponents** (D0, counted after Schwarz and P1). The GBS reader
stores a general contraction as separate shells over the same exponents
(cc-pVDZ O 1s and 2s: two 9-primitive shells, same exponents), and Pople SP
shells share exponents across L. Merging them cuts primitive passes by:

| fixture | merge within L | merge across L |
|---|---|---|
| 8 H2O / 6-31G | 1.00× | 2.23× |
| 4 H2O / 6-31G\* | 1.01× | 2.13× |
| naphthalene / 6-31G | 1.00× | 3.73× |
| 4 H2O / 6-311G | — | 2.18× |
| 1 H2O / cc-pVDZ | 4.82× | 4.82× |
| 4 H2O / cc-pVDZ | 5.11× | 5.11× |
| 1 H2O / cc-pVTZ | 4.96× | 4.96× |

## Symmetry

**The fused signed-operation path.** Naphthalene / 6-31G (D2h, 8
operations), one Fock build, measured after D1 (before 2b and K1.4; the
symmetric build is now 0.206 s):

| | no symmetry | symmetry | ratio |
|---|---|---|---|
| canonical quartets visited | 13.37 M | 5.57 M | 2.4× |
| orbit representatives | — | 3.35 M | 4.0× |
| … of which forbidden | — | 1.43 M | |
| integral values computed | 13.37 M | 1.92 M | **7.0×** |
| closed-form / HGP block calls | 1.87 M / 22.9 k | 0.48 M / 6.9 k | 3.9× / 3.3× |
| Fock accumulations | 104.7 M | 15.0 M | **7.0×** |
| build time | 1.011 s | 0.295 s | **3.4×** |

- **A planar molecule caps the orbit at 4**: the mirror through the plane
  fixes every AO, so its factor of 2 reappears as forbidden (zero)
  integrals; values computed fall 7.0×, close to the group order.
- **What held the build at 3.4×** was per-block and per-quartet work that
  falls less than 7×: a block is evaluated if any component survives, and
  every surviving canonical quartet paid the orbit test (13 %). K1.4 moved
  most of the orbit test to the block.
- **A speedup can shrink a ratio**: symmetry paid 5.2× after S1 and 3.4×
  after D1, because D1 sped up the plain build more (it evaluates 3.9× more
  blocks).

**The full point-group path** (SAO-blocked HF, direct SCF). A separate loop
(`full_symmetry_build_skeleton` in `src/scf/scf.cpp` →
`{os,hgp,rys}_symm.cpp`) fills a dense n_b⁴ skeleton per Cartesian
component, contracted every iteration; Auto resolves to HGP there (F2). It
is persisted when n_b ≤ `threshold` (default 100); above that the SCF loops
take the fused path with the geometry's D2h-subgroup operations (F1), and
so K1.4. RHF, `scf_mode direct`, one thread:

| molecule | n_b | full group, persisted | fused, no symmetry | before F1 (rebuilt every iteration) | F1 |
|---|---|---|---|---|---|
| H2O / cc-pVDZ (C2v) | 25 | 0.047 s + 0.0004 s/iter | 0.0075 s/iter | | |
| NH3 / 6-31G\* (C3v) | 21 | 0.017 s + 0.0003 s/iter | 0.0024 s/iter | | |
| CH4 / cc-pVDZ (Td) | 35 | 0.054 s + 0.0016 s/iter | 0.018 s/iter | | |
| benzene / 6-31G\* (D6h) | 102 | 3.0 s (`threshold 200`, 866 MB) | 13.7 s | 17.7 s | **4.08 s** |
| naphthalene / 6-31G (D2h) | 106 | 7.3 s (`threshold 200`) | 21.3 s | 66.6 s | **6.35 s** |

Below n_b = 100 the persisted skeleton is 6–19× cheaper per iteration than
a fused build. Benzene on the F1 path uses 8 of its 24 operations and is
within 1.4× of the persisted full-group path without its n_b⁴ memory (F1
numbers predate K1).

## What landed

| step | commit | what it removed or added | gain | gate |
|---|---|---|---|---|
| C0 | Boys v1.1.0 | 17.7 % of the build in libm calls inside one branch of the out-of-tree Boys core | 1.37× (8 H2O), 1.13× (4 H2O\*) | two iteration counts re-pinned at the noise floor (`water_rhf_hgp_sto3g` 11 → 10, `ethylene_rhf_ccpvdz_auto` 16 → 15) |
| C1 | `c40a016c` | a `ShellPair` rebuilt per component quartet | 1.16–1.26× | bitwise |
| C2–C3.5 | `3c028544`, `4bbb2724`, `f0f7ceae` | closed-form shell kernels through total L ≤ 3 | 1.22–1.28×, 1.61×, 1.87× on 8 H2O | per-quartet sweep against HGP, 6.1e-16 |
| H1 | `46dadc6a` | MMAX+1 scalar `boys` calls → one `boys_vec` (HGP and RI); VRR zero-fill dropped | 1.10–1.12× × 1.04–1.06× | bitwise |
| L2 | `659883ab` | the quadratic search in `distinct_eri_orbit` (17.3 % of the 8 H2O build) | 1.20× / 1.09× / 1.11× | exhaustive oracle, 1296 tuples |
| L1 | `de72ca2c` | ket blocks whose first shell sorts below the bra's | 1.06× | bitwise |
| L3a | `f88991fe` | the full orbit built for non-representatives | 1.70× (D2h) | exhaustive, AO-permuting synthetic group |
| L3b | `a1f99d7a` | per-quartet orbit allocation; loop body compiled per symmetry mode | 1.23× (D2h) | bitwise |
| P1 | `15c532e4` | primitive pairs with \|c_a c_b N_a N_b (π/ζ)^{3/2} K_ab\| < 1e-24 (19 % of quartets on 8 H2O) | 1.15× / 1.10× / 1.04× | G byte-identical |
| L3c | `15c532e4` | blocks an operation sends below the bra's first shell (60 % of D2h blocks) | 1.07× (D2h) | `check_real_symmetry_ops` |
| S1 | `7ec71009` | per-orbit-element replication of every G write | 1.33× (D2h) | `check_skeleton_symmetry` |
| fix | `fd51d29d` | H1's Boys buffer overflow on (ff\|ff) in HGP's hoisted block path | — | f/g/h spherical gates |
| D1 | `52698142` | the per-component primitive loop at total L 4–12 | 1.5× sp, 4.3–4.9× d, 12.9× f | 20 fingerprints; `check_hgp_block_kernel` |
| F4 | `d0224110` | `guess sad` aborting full point-group SCF on C3v/D6h | correctness | `nh3_c3v_fullsym_631gd_sad_direct` |
| F2 | `17cf5adf` | `engine auto` falling through to per-component OS on the full point-group path | 1.07–1.12× per rebuilt skeleton | energies unchanged |
| F1 | `92838e03` | the n_b⁴ skeleton rebuilt every iteration above `threshold` | 4.3× benzene, 10.5× naphthalene (whole SCF) | energies vs the persisted path; 10 full-symmetry inputs at `threshold 1` |
| T1 | `92838e03` | `thread_local` block-kernel scratch | neutral | 28 fingerprints identical |
| 2b | `bbe705fc` | one primitive quartet per VRR pass: four at once (lanes) in blocks with K ≥ 3 | 1.38× cc-pVTZ, 1.21–1.23× cc-pVDZ / 6-31G\*, 1.05–1.07× sp | block sweep ≤ 1.1e-14 on four bases, mutation-tested; max\|ΔG\| ≤ 1.7e-14 |
| K1 | `5fd08496` | per-component digestion (canonical filter, orbit dedup, scatter) on symmetric densities, Auto, Coulomb, L ≤ 12 | 1.11–1.22× (1 thread), 1.09–1.17× (4) | `check_dense_digest`, 68 checks ≤ 8.9e-14, counting kernel; 5 mutations; MPI 1/2/4 |
| K1.4 | — | the per-component skeleton weight and per-block group-quartet canonicalization under S1 | 1.32× (1 thread), 1.33× (4), D2h naphthalene | 32 checks under real C2v ≤ 1e-13·\|G\|; 8 mutations |

Suites after K1.4: smoke 43/45, extended 135/139; the failures are the known
rccsdt routing cases. Details worth keeping:

- **Closed forms (C2–C3.5).** `low_l_shell_eri` treats a quartet as up to
  three *raises* (center, axis) in center order. Per primitive quartet,
  with $`\theta_m = c\,K_{AB}K_{CD}\,2\sqrt{\rho/\pi}\,F_m(T)`$, $d$ the
  raised center's $P-A$, $P-B$, $Q-C$ or $Q-D$, and $w$ its $W-P$ or $W-Q$:

  ```math
  \begin{aligned}
  A_i(m) &= d_{1,i}\,\theta_m + w_{1,i}\,\theta_{m+1} \\
  B_{ij}(m) &= d_{2,j}\,A_i(m) + w_{2,j}\,A_i(m+1) + [i=j]\,\kappa_{12}(\theta;\,m) \\
  C_{ijk} &= d_{3,k}\,B_{ij}(0) + w_{3,k}\,B_{ij}(1) + [k=i]\,\kappa_{13}(A^{(2)}_j) + [k=j]\,\kappa_{23}(A_i)
  \end{aligned}
  ```

  $\kappa_{yx}$ couples a new raise to an earlier one it lowers: on the bra
  $\bigl(X(m) - \tfrac{\rho}{\zeta}X(m+1)\bigr)/2\zeta$, on the ket the same
  with $\eta$, across $X(m+1)/2(\zeta+\eta)$. At most 27 outputs, in
  registers; per-AO `norm_ratio` at readout. Gate: every total-L ≤ 3
  component quartet of water in 6-31G, 6-31G\*, cc-pVDZ against HGP, every
  reachable raise layout required, mutation-verified.
- **S1.** When every density has the molecule's symmetry, each
  non-forbidden representative is accumulated once, weighted by its orbit
  size, and at the end (`quartet_orbit.h`)

  ```math
  G_{\mu\nu} = \frac{1}{|G|}\sum_R s_R(\mu)\,s_R(\nu)\,G^{\mathrm{skel}}_{R(\mu)R(\nu)}.
  ```

  Any
  non-symmetric density falls back to replication. Symmetric DFT gains;
  direct HF on linear molecules does not (no SAO blocking for infinite
  groups, so iterates break the symmetry).
- **D1.** `prepare(BlockEriRequest)` / `value(ca, cb, cc, cd)`;
  `ChainBlockKernel` tries two kernels. The HGP block kernel runs the
  Obara–Saika VRR per primitive quartet into an **L-major** table
  [e0|f0]^(m), contracts the m = 0 slice, then HRR once per shell quartet
  (`finish()`, split out of `prepare` and callable on its own; bitwise).
  Every total-L class 4–12 won or tied, so Auto takes them all. Gate:
  4.6 M component quartets in 256 classes up to (ff|ff) against
  per-component HGP, ≤ 8.9e-15, mutation-verified.
- **F1.** The axis-flip operations `update_integral_symmetry` builds form
  the largest D2h subgroup of the point group, and an SAO density is
  symmetric under any subgroup, which is all S1 needs. So an unpersisted
  full-group run falls through to the fused build; the per-iteration
  rebuild wrappers were deleted.
- **F4.** The raw SAD density is group-invariant; the idempotent projection
  splits a degenerate set at the n_occ cutoff. With full symmetry the guess
  is group-averaged, `P ← (1/|G|) Σ_R O_R P O_Rᵀ`.
- **2b.** `hgp_block::vrr_lanes<B>` runs the block kernel's VRR for B
  primitive quartets at once, lane-minor (`[e][f][m][lane]`), every
  innermost loop of compile-time length B. It takes primitive-pair pointers
  only; the caller contracts with its own weights, so a family pass can
  reuse it. `HgpBlockKernel` uses B = 4 in blocks with K ≥ 3, padding lanes
  at weight 0; smaller blocks keep the scalar loop. Against the scalar
  kernel, one thread: cc-pVTZ 0.127 → 0.092 s, 6-31G\* 0.100 → 0.081 s,
  cc-pVDZ 0.548 → 0.453 s, naphthalene 0.294 → 0.274 s, 8 H2O / 6-31G
  0.136 → 0.129 s. B = 2 sits between, B = 8 is slower than 4, and a K = 3
  block is cheaper padded to 4 than run scalar. Gates: the block sweep
  ≤ 1.1e-14 on four bases; dropping the laned W−Q term, weighting padding
  lanes or contracting lane 0 only each fail by 0.13–5.4; max|ΔG| against
  per-component HGP ≤ 1.7e-14; the 28 fingerprints (explicit-engine and
  closed-form builds) unchanged.
- **K1.** `fused_fock_build(..., dense_digest = true)`. The choice is per
  build because the two traversals visit different block sets (p ≤ q
  against the AO-index canonical filter, which needs kets with q < p);
  mixing them in one build double-counts or drops quartets. Writes per mode,
  with v′ = deg · value: Coulomb `G_ab, G_cd += ½ P v′`; exchange `G_ac,
  G_bd, G_ad, G_bc` with −⅛ (RHF Combined), +¼ (raw K), −¼ per spin (UHF
  Combined); then (G + Gᵀ)/2. Exact only for a symmetric density (the
  symmetrization deletes K of the antisymmetric part), hence the density
  test: the J/K entry points serve response code. The component Schwarz
  mask keeps screening identical to today (unmasked, G moves by the dropped
  2.8–6.2e-10). A generic two-channel body gave 1.09–1.18×; one body per
  active write group took back all but 0–4 points. Four threads drop to
  1.09× on cc-pVTZ: p ≤ q gives early bra pairs more kets (B1's units fix
  that, with B2). Gates: `check_dense_digest`, flag on against off, every
  `FusedTerm` mode, RHF and UHF, water in four bases, tol 1e-10 and 1e-6;
  a counting kernel proves the dense path ran; a non-symmetric density
  must fall back bitwise; mutations on each degeneracy factor, the exchange
  coefficient, the symmetrization and the mask. MPI: three Auto direct-SCF
  cases (`*_auto_direct_mpi_smoke`) match serial at 2 and 4 ranks (HF
  identical, B3LYP 1.4e-14 Eh), and explicit `engine hgp` to every printed
  digit (B3LYP 1.1e-13 Eh).
- **K1.4.** Skeleton-mode builds go dense with a per-component S1 weight.
  Ops that send the block's canonical shell quartet lower skip the block;
  higher ones are irrelevant to it; only ops fixing it are applied per
  component. Two steps after the first version (0.255 s, 1.08×): ops whose
  AO map is the identity get a sign test instead of canonicalization
  (0.247 s), and the per-block test reads a per-build pair-image table
  instead of canonicalizing four shell indices for every op on every block
  pair, ~1.9 M × 8 on naphthalene (0.206 s, 1.32×). Gates: dense against
  per-component under water's real C2v (op count asserted), 32 checks ≤
  1e-13·|G|; 8 mutations. The op-splits-a-group fallback
  (`skeleton_orbit_weight`, unchanged) has no fixture.

## Measured, not landed

### D1c — the electron-transfer recurrence (declined)

Head-Gordon and Pople's ETR at m = 0 after a one-centre VRR out to |e| ≤ M:

```math
[e|f{+}1_j] = \bigl(\tfrac{\zeta}{\eta}PA_j + QC_j\bigr)[e|f]
+ \tfrac{f_j}{2\eta}[e|f{-}1_j] + \tfrac{e_j}{2\eta}[e{-}1_j|f]
- \tfrac{\zeta}{\eta}[e{+}1_j|f]
```

(the textbook $-(\beta X_{AB}+\delta X_{CD})/\eta$ written with $P-A$, $Q-C$).

- **Precision.** A fixed bra→ket direction was off by 2.8e-9 on cc-pVTZ;
  transferring per primitive quartet toward the larger exponent sum gave
  2.4e-13 (27× the VRR), G up to 9.4e-13. The VRR for ratios in (0.25, 1]
  restores the VRR's error exactly.
- **Speed.** 1.18× / 1.06× / 1.09× (cc-pVTZ / 4 H2O cc-pVDZ / 4 H2O
  6-31G\*), 1.06× / 1.04× / 1.04× at VRR precision, against a predicted
  1.1–2.3× less recurrence work: the ket phase fell 2.0–2.1×, the bra VRR to
  |e| ≤ M grew 3.3–3.5× (cc-pVTZ: ket 55.6 → 26.1 ms, bra 4.0 → 14.0 ms).
- **A row layout** (V as [f][m][e], every term unit-stride through
  precomputed runs) was bitwise on 28 fingerprints and time-neutral: the
  cost is per primitive quartet, not arithmetic. Both reverted.

### 5b — a class-batched scheduler (declined after S0)

Collect the blocks that pass the block filters, group them by class, and
evaluate classes in batches across blocks (the shape of ORCA's SHARK).
`--census`, weighted by primitive quartets:

| fixture | HGP-class quartets already in within-block lanes | padding within-block | pooled per bra pair |
|---|---|---|---|
| 1 H2O / cc-pVTZ | 93.9 % | 13.4 % | 2.7 % |
| 4 H2O / cc-pVDZ | 96.5 % | 4.6 % | 0.8 % |
| 4 H2O / 6-31G\* | 95.8 % | 19.4 % | 2.5 % |
| 8 H2O / 6-31G | 98.8 % | 12.4 % | 0.5 % |
| naphthalene / 6-31G | 99.5 % | 9.4 % | 0.1 % |

Closed-form classes: 99.5–99.8 % already in lanes. Pooling would buy at most
~1.15× on the lane work; enumerating with a survival check cost 2–12 % of
the build. cc-pVTZ's K = 1 blocks are 23 % of HGP kernel time but ~6 % of
its primitive quartets: their cost is per block, which batching does not
touch while each block is digested component by component. The K1 census
measured that contract directly, and K1 landed on it.

### D2a and B0 — shared-exponent families

`--family`: per atom, a shell joins the largest shell whose exponent set
contains its own (cc general contractions within L, Pople SP across L). A
family quartet runs one primitive pass over the union of its members'
primitive pairs, keyed by the two exponents, and members split by **their
own** total L:

- **L ≤ 3:** T, the coefficient-free prefactor and Boys once per primitive
  quartet; the raises once per distinct (L_A, L_B, L_C, L_D) layout (linear
  in θ_m); each member adds w_k·w_l times them. B0 made this
  production-quality: raises templated on their count, no per-quartet
  allocation, bra members looped first with zero-weight rows skipped.
- **L ≥ 4:** one HGP VRR pass at the high members' maximum L (L-major
  indices and m = 0 values do not depend on L_A+L_B or M), then each
  member's own `finish()`.

Kernel time against the production kernels on the same member blocks (no
Schwarz or canonical filtering on either side), values within 7.2e-16:

| fixture | L ≤ 3, D2a → B0 (passes fewer) | L ≥ 4 (passes fewer) | kernel total (B0) |
|---|---|---|---|
| 1 H2O / cc-pVTZ | 1.69× → **1.54–1.78×** (5.9×) | 1.70× (2.6×) | **1.74×** |
| 4 H2O / cc-pVDZ | 1.66× → **1.54×** (5.9×) | 1.69× (2.4×) | — |
| naphthalene / 6-31G | 1.16× → **1.22–1.34×** (3.6×) | 1.05× | — |
| 4 H2O / 6-31G\* | 0.95× → **1.11–1.13×** (2.2×) | 1.21× (1.3×) | 1.18× |
| 4 H2O / 6-31G | 0.93× → **1.04–1.15×** (2.3×) | 1.49× | — |

- **Pople merges across L**: it wins on every Pople fixture with B0. With
  nothing merged the pass still runs at 0.74–0.81× of production, so ~20 %
  per-quartet overhead remains.
- **The L ≥ 4 numbers predate 2b** and are against the scalar kernel; the
  family pass must call `vrr_lanes` over the union primitive quartets to
  keep both gains.
- **Iteration order decided B0**: walking each layout's member list cost
  1.25× (cc-pVDZ) and 1.65× (cc-pVTZ) against bra-members-first with
  zero-weight rows skipped (cc's small members contribute at one exponent
  of nine). Removing allocation was worth 10–20 % on Pople; scalar Boys and
  weight folding were within noise.
- cc kernels are 73–80 % of those builds: an estimated 1.4–1.5× of the
  Fock build.

### B1 — the fused loop in family-quartet order (parked)

`fused_fock_build` takes the ket iteration as a parameter of its bra body.
With a block kernel (Auto) and `family_order` (default true) it runs one
unit per family quartet (P, Q), P ≤ Q: bra x ∈ P with kets y ∈ Q where
y.a ≥ x.a, and for P < Q also bra y ∈ Q with kets x ∈ P where x.a ≥ y.a.
That is exactly today's set of ordered block pairs (ket.a ≥ bra.a), each
through the same per-block body; explicit engines keep today's order.
Families come from `fused_families` once per build; the MPI stripe moved to
the unit index; the orbit buffer is per thread.

- **Exactly once:** accumulation counts identical on five fixtures with and
  without symmetry; max|ΔG| ≤ 4.7e-14 against today's order.
- **Cost of the order alone** (best of five, interleaved):

  | fixture | 1 thread | 4 threads |
  |---|---|---|
  | 4 H2O / cc-pVDZ | −1.6 % | −22 % |
  | 1 H2O / cc-pVTZ | −1.7 % | +2.6 % |
  | 4 H2O / 6-31G\* | +4.6 % | −8.5 % |
  | naphthalene / 6-31G | +5.6 % | +0.7 % |
  | 4 H2O / 6-31G | +7.6 % | +1.3 % |

  One unit per family *pair* was +76 % at 4 threads on cc-pVTZ (13
  families, 91 uneven units under `schedule(static)`).
- **Gates:** `planck-fock-accumulate`, `planck-hgp-engine-smoke`, smoke
  43/45 (known rccsdt), and MPI: `planck-mpi` (`-DBUILD_MPI=ON`,
  `build-full`, Homebrew Open MPI) at 1/2/4 ranks against the serial
  binaries, Auto direct SCF: RHF 4 H2O / cc-pVDZ, RHF naphthalene / 6-31G
  (D2h skeleton) and UHF H2O⁺ / 6-31G\* identical to every printed digit;
  B3LYP H2O / 6-31G\* within 9e-14 Eh. Wall 9.3 → 6.1 → 5.1 s and 6.3 → 3.7
  → 2.9 s on the larger cases.
- **Fate: parked as a git stash** ("B1: fused loop in family-quartet
  order", with the benchmark's `--b1` mode). It visits ordered block pairs,
  while K1 visits unordered ones, so it must be re-cut for K1 (within a unit,
  member pairs p ≤ q) when it returns with B2. Its four-thread balance is
  what K1's cc-pVTZ drop at four threads needs.

## Findings

**The floor was the primitive loop, not the recurrence.** The (ss|ss) closed
form has no recurrence, yet per component it was only 1.66–1.75× faster
than HGP: every call reran the primitive loop (1296 primitive quartets for
an O 1s in 6-31G\*, repeated 81× for (pp|pp)). Once per shell quartet: 4.9×,
and 10–16× at L = 4.

**Below the block kernel, the floor is the primitive quartet.** A few hundred
elements of work per primitive quartet is overhead-bound, so the lever is
the number of primitive passes and how many run together, not their
arithmetic (D1c, the row layout).

**The contract between generation and digestion caps what batching can
buy.** While every integral leaves its kernel through `value()` and is
digested alone, batching blocks changes nothing outside the kernels; making
the per-block digestion dense did (K1).

**A share of time is not a share of work.** cc-pVTZ's K = 1 blocks are 23 %
of HGP kernel time and ~6 % of its primitive quartets. Lanes across blocks
were scoped on the first number and refuted by the second (5b).

**An idea that measured slower may have been its first implementation.** A
contract-once hoist measured 1.14–1.62× slower; D1's block kernel is the
same idea at 1.5–13× faster, with L-major indexing.

**A profile share is a lower bound on what removing the work is worth, and
can mislead the other way.** The `ShellPair` rebuild was 8.4 % of samples
and worth 1.16–1.26×; L3c cut 59 % of the symmetric loop's iterations for
1.07×; T1 removed a 2.5–4.7 % frame for nothing measurable.

**Classify by the work, not the container.** D2a first classified family
quartets by the family's maximum L, which sent Pople's (ss|ss) members
through the HGP VRR (0.51–0.53×). Split by member L.

**Granularity decides load balance.** B1's units of one family pair left
four threads 76 % slower on cc-pVTZ; one family quartet per unit balanced
better than today's loop.

**Precision can decide an algorithm.** The ETR's −ζ/η subtraction cost
4–6 digits until the transfer direction was chosen per primitive quartet,
and still 27× the VRR's error after.

**The cheap fix often sits next to the hot frame.** HGP made one Boys call
per order through the loader's indirect call; OS had already switched to
`boys_vec`.

**A buffer sized for one call site can be wrong for another.** H1's
`boys_vec` buffer was sized for one component (MMAX ≤ 24); HGP's hoisted
block path passes per-axis maxima, so (ff|ff) reached MMAX = 36: SIGBUS on
any conventional HGP build with f shells, hidden for two days among known
failures. **A red gate that is "already failing" still needs its cause
checked.**

**Code layout can cost more than the change.** Hoisting L3b's orbit buffer
slowed molecules *without* symmetry by 5 % through register allocation;
compiling the loop body once per symmetry mode removed the coupling.

**A speedup can shrink a ratio.** Symmetry paid 5.2× after S1 and 3.4× after
D1, with both builds faster. Quote ratios with their date.

**Screening must never reach the Schwarz table.** The first P2 experiment
gave max|ΔG| = 3e-6 at 1e-14 because `schwarz_table_hgp` builds Q with the
same kernel, so Schwarz stopped being an upper bound.

**A cost-only test cannot be value-gated.** S1's forbidden-quartet skip
survives inversion: the final symmetrization cancels a forbidden quartet
exactly.

**Gates must be able to fail.** Every kernel gate was mutation-tested. The
flag-off fingerprint caught C3 routing (ss|ss) with the flag off;
layout-coverage caught water/6-31G\* lacking (d,p) layouts; sign-only
symmetry fixtures make every quartet its own representative.

**Print a census's mapping before believing its total.** D0's first count
required a representative strictly larger than the shell it absorbs, so
cc-pVDZ oxygen's equal-size 1s and 2s never merged: 1.47–1.83× instead of
4.8–5.1×. K1's first max|ΔG| of 6e-10 looked like a weight error and was the
dropped component Schwarz.

**Write the prediction down first.** It caught a misread class share, C4a's
3–4× low predictions, 5b's premise, and B1's Pople cost (+5–8 % against
±3 %).

**Specialize the inner loop before judging a digestion change.** K1's first
body served every term mode through one loop with per-component branches
and fell 2–8 points short of the census; one instantiation per write group
closed the gap.

**A copied fixture can hide the thing under test.** `update_integral_symmetry`
matches basis functions to shells by pointer; a copied `Calculator` kept
only the identity, and the first K1.4 "C2v" gate passed with |G| = 1. Assert
the fixture's defining property (here, the op count).

**When a fix does not move the profile share, the hot call site is
elsewhere.** K1.4's first fix removed the per-component canonicalization;
time improved 3 % but `canonicalize_orbit_quartet`'s share did not move. The
hot calls were in the per-block test, and a table took the build from
1.11× to 1.32×.

## Open, in order

| step | what | bitwise? |
|---|---|---|
| B2 | families on top of K1, with B1's loop order re-cut for p ≤ q | no; lands alone |
| partition | fixed accumulator partition | no; lands alone |
| C5 | screened kernels in both block kernels (then dense too) | screened kernels only |
| P2 | quartet-level primitive screening | no (~1e-12); lands alone |
| L > 12 | block kernel above total L = 12 | — |
| F3 | full point-group skeleton from the block kernels | yes, per engine |
| D3 | RI-JK pair factors (independent) | per the BLAS doc |
| K2 | batches across ket blocks (conditional) | no |

**B2 — families on top of K1.** The family kernel produces each member
quartet as a dense block for K1 to digest; the L ≥ 4 pass calls `vrr_lanes`
over the union primitive quartets. B1's family-quartet units return with it
(from the stash), re-cut so each unit visits member pairs p ≤ q; that also
fixes K1's four-thread imbalance on cc-pVTZ.

- *Benefit (measured on kernels):* L ≤ 3 1.54–1.78× on cc, 1.04–1.34× on
  Pople (B0); L ≥ 4 ~1.7× on cc against the scalar kernel. Estimated build:
  cc 1.4–1.5×, Pople 1.0–1.3×, on top of K1.
- *Risks:* the L ≥ 4 gain shrinks against the laned kernel (lanes and
  families compete for the same primitive work); Pople's ~20 % per-quartet
  overhead at 1:1; scratch per family quartet (cc-pVTZ: up to 100 member
  quartets, each up to n_comp⁴ for (ff|ff)), to bound before relying on it;
  the validation surface (every family layout, member-subset selection,
  union primitive pairs under P1).
- *Gates:* a family sweep in `planck-hgp-engine-smoke` (every member value
  of every family quartet against per-component HGP on water 6-31G,
  6-31G\*, cc-pVDZ, cc-pVTZ, mutation-tested on the coefficient
  contraction, member-subset selection and layout sharing); max|ΔG| ≤
  1e-12 against the K1 build; `check_dense_digest` with and without
symmetry; MPI 1/2/4; both suites.

**The fixed accumulator partition.** The fused loop's partials are keyed by
thread index, so a run is reproducible at a given thread count but not
across counts (CO2 / 6-31G\* direct differs between 1 and 4 threads in the
DIIS residual digits and the last kcal/mol digit; all 10 Hartree digits
identical). Key the partials by a fixed function of the unit index and merge
in fixed order, the FCI sigma-build rule; memory n_bins × nb² (64 × 2 MB at
nb = 500). Gate: G bitwise identical at 1/2/4/8 threads. Predates this work.

**C5 — screened kernels.** Both block kernels decline screened kernels, so
range-separated hybrids take per-component HGP. They need `hgp_vrr`'s
scaling: T × `boys_scale`, prefactor × `prefactor_scale`, W−P/W−Q ×
`screen.rho / rho`, and the straddling coupling × `boys_scale` (the
`inv_2_delta` gotcha). Gate: a screened sweep against OS on a mixed
bra/ket fixture for LongRange and ShortRange, plus the
`water_*_engine_os_vs_hgp` comparators.

**P2 — quartet-level primitive screening (a policy decision).** After P1,
~20 % of primitive quartets on water clusters have a prefactor bound between
1e-24 and 1e-12. Skipping below ε, Schwarz held exact (before D1):

| fixture | ε = 1e-14 | ε = 1e-12 | max\|ΔG\| at 1e-14 / 1e-12 |
|---|---|---|---|
| 8 H2O / 6-31G | 1.13× | 1.37× | 4.0e-12 / 3.3e-10 |
| 4 H2O / 6-31G\* | 1.07× | 1.18× | 4.9e-12 / 3.7e-10 |
| naphthalene D2h | 1.15× | 1.22× | 4.8e-12 / 4.8e-10 |

Proposed ε = 1e-4 · `tol_eri`, through `BlockEriRequest::prim_eps`;
`schwarz_table_*` must pass ε = 0. Re-measure after B2.

**L > 12.** Blocks above total L = 12 (g shells and up) take per-component
HGP. Extend when a fixture needs it; the VRR table is 3.7 MB for (gg|gg).

**F3.** After F1 the full-group skeleton runs only where it is persisted,
so this shortens a one-time build (0.017–0.054 s on the small regression
molecules, 1.3–3.4 s near n_b = 100). Worth it only if `threshold` is raised
routinely.

**D3 (independent).** RI-JK recomputes its density-independent pair factors
twice per iteration, ~61 % of the RI Fock build (B0 in
`docs/ERI_FOCK_BLAS_SCOPE.md`).

**K2 (conditional).** Batches of same-class ket blocks for a fixed bra unit
on top of K1: lanes across kets, bra-side work hoisted, the HRR as array
work, digestion as small matrix products. Only if a census after K1 and B2
shows per-block fixed costs (setup, HRR at 10–16 % of the HGP kernel) still
worth the largest restructure here; the exactly-once argument then has to
hold at batch level.

**C4b (deprioritised).** A four-raise closed form matched HGP to 1.3e-15 on
every total-L = 4 layout and measured 10–16× over per-component HGP (C4a),
but its competitor is now the block kernel, whose L = 4 work is 4–9 % of sp
builds. Revisit only if L = 4 becomes prominent again.

## Ruled out

- **A per-block readout table for the closed forms**: readout ≤ 3.6 % (A0).
- **Forcing `accumulate_term` / `distinct_eri_orbit` inline**: stubs < 1 %.
- **Avoiding `boys(0, T)` in the (ss|ss) branch**: mostly the Boys
  evaluation itself.
- **A generator for an L = 4 closed form**: the hand-written level was
  correct first time (C4a).
- **The ETR (D1c), a unit-stride row layout for the VRR, and a
  class-batched scheduler on today's digestion (5b)**: see above.
- **Vectorizing the closed forms' primitive loop** while Boys, about half
  their cost, is an out-of-tree core that evaluates one T per call.
- **Loosening S1's density tolerance** so linear HF takes the skeleton
  path: it would change the SCF trajectory.

## Out of scope

- **Gradients**: derivative ERIs are a separate kernel family (Open Work
  Phase D); P1 covers them.
- **Spherical**: the direct spherical path back-projects to Cartesian and
  uses the same fused loop.
- **SAO blocking for linear point groups**: would let direct linear HF take
  S1's path; no tested workflow needs it.
