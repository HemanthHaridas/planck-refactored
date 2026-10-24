# Shell-Quartet ERI Kernels for the Fused Fock Build

**Question:** How much faster can the memory-direct (fused) Fock build get by
evaluating whole shell quartets at once (closed forms at low angular momentum,
a block kernel above), and what else in the build stands in the way?

**Status (2026-10-24).** The kernel work is done: a Fock build is 7–23×
faster at one thread than before it started, and the remaining kernel ideas
were measured and declined, and G no longer depends on the thread count.
Open: R1b (conditional), C5, F3; D3, K2 and C4b sit outside the main line. The step names (C1, D1, K1, …) are kept because commits, tests
and code comments cite them.

## Answer

| fixture (one thread, `engine auto`) | before this work | now | speedup | now, four threads |
|---|---|---|---|---|
| 8 H2O / 6-31G (nb = 104) | 1.50 s | 0.105 s | **14×** | 0.0284 s |
| 4 H2O / 6-31G\* (nb = 76) | ~1.12 s† | 0.0625 s | **~18×** | 0.0169 s |
| 4 H2O / cc-pVDZ | 2.573 s§ | 0.358 s | **7.2×**§ | 0.0959 s |
| 1 H2O / cc-pVTZ | 1.631 s§ | 0.0698 s | **23×**§ | 0.0193 s |
| naphthalene / 6-31G, D2h (nb = 106) | 1.454 s | 0.188 s‡ | **7.7×** | 0.0548 s‡ |

`planck-eri-class-share --k1` production arm, Apple M3, `build-full`
(g++-15 `-O3`), Boys core v1.2.0, best of three (seven on 8 H2O and
cc-pVDZ) interleaved, 2026-10-24. "Before" is the build before C1, after the
Boys fix C0 (itself 1.37× / 1.13×). † The product of per-step ratios; never
timed before C1. § Against the D0 build (after P1, before S1 and D1). ‡ With
`sym_ops`, on a group-averaged density. Explicit engines get only the
engine-independent steps; on 8 H2O / 6-31G and 4 H2O / 6-31G\* Auto is ~7×
and ~11× faster than explicit `engine hgp`. Fock-build numbers, not SCF wall
time. The ~200× libcint gap in `docs/ERI_PERFORMANCE_SCOPE.md` predates this
work and has not been re-measured.

Per step, each a ratio over the build with all previous steps (one thread):

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c | D1 | 2b | K1 | R2 | scratch | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | 1.49× | 1.05× | 1.14× | 1.05× | ~1.0× | **14×** |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | 4.90× | 1.23× | 1.17× | 1.04× | 1.03× | **~18×** |

On the symmetric fixture: L3a 1.70×, L3b 1.23×, L3c+P1 1.16×, S1 1.33×,
D1 ~1.5×, 2b 1.07×, K1.4 1.32×, R2 1.11× (1.454 → 0.855 → 0.702 → 0.597 →
0.454 → 0.295 → 0.274 → 0.206 → 0.187 s). At four threads W1's round-robin
schedule gives 1→4-thread scaling of 3.4–3.7×.

**In short:**
- **Evaluate shell quartets, not components.** Every engine returned one
  Cartesian component per call and reran the whole primitive loop for each,
  27–78× per shell quartet wherever HGP ran. Removing that repetition is most
  of the speedup: closed forms at total L ≤ 3, an HGP block kernel at 4–12.
- **Layout decides whether that wins.** The block kernel is the same idea as
  an earlier contract-once hoist that measured 1.14–1.62× slower; the
  difference is indexing Cartesian functions by total L instead of a per-axis
  cube.
- **Inside the kernels, what is left is per-primitive-quartet overhead.** A
  cheaper recurrence (D1c) or a better layout does not help; several primitive
  quartets per pass (lanes, 2b) and cheaper per-quartet work (R2) do. Sharing
  per-quartet work across shells that share exponents pays only when both
  sides share and the visiting order is cheap, which held on cc bases alone;
  no form of it reached production.
- **Outside the kernels, the cost was digesting one component at a time.**
  Digesting each block densely (K1) gave 1.11–1.22×, under the skeleton scheme
  (K1.4) 1.32×.
- **Then the threads.** The dense traversal is a triangle; a contiguous
  schedule capped four threads at ~2.3×. Round-robin (W1) gave 1.10–1.97× at
  four threads for one line. Keying the partial Fock matrices by bin
  instead of thread then made G independent of the thread count and let
  the bins be scheduled dynamically, which recovered the load imbalance
  under symmetry (1.06× at four threads on naphthalene).

## How the build works now

`Integrals::fused_fock_build` (`src/integrals/fused_fock.h`) chooses one of
two traversals **per build**. Both run over bra units: the MPI stripe takes
bra mod size, and OpenMP hands out **bins** (`schedule(dynamic, 1)`), bin b
holding bras b, b + n_bins, … in increasing order (round-robin, as W1 found
a triangle needs). Partials are keyed by bin, not thread, and summed per
element in bin order, so G is identical at any thread count. n_bins is a
function of nb only: 64, fewer once 64 nb² passes 256 MB per output, at
least 8. Kernel scratch stays per thread; it carries no values across
blocks.

**Dense (K1, K1.4).** Taken when the block kernel `covers()` every total L the
basis can produce (today: Auto, Coulomb, shells up to f, total L ≤ 12), every
density is symmetric to 1e-10, and there is either no symmetry or skeleton
mode. Per bra pair p, kets q ≥ p:

1. Block Schwarz. Under S1, each op's image of the block is (min, max) of two
   entries of a per-build table `op_pair[o][p]`; an image that sorts lower
   skips the block, and permuting ops that map it onto itself go to
   `block_ops`.
2. `prepare` the block lazily, at its first surviving component, then every
   component above the component Schwarz bound:
   v′ = deg · weight · value, deg = (A=B?1:2)(C=D?1:2)(p=q?1:2). Under S1,
   weight is the S1 weight of the component's canonical form
   (`skeleton_orbit_weight` with what the block already decided: identity-map
   ops need only a sign test, only `block_ops` need canonicalization).
3. Six writes per component (two Coulomb, four exchange), from a body
   instantiated per active write group, then (G + Gᵀ)/2 after the merge (and
   the group average under S1).

**Per-component.** Everything else: explicit engines, screened kernels,
total L > 12, non-symmetric densities, replication-mode symmetry. Block
filters (L1, block Schwarz, L3c), component filters (canonical order,
component Schwarz, the orbit test), the block kernel prepared at the first
surviving component (a declined block uses per-component `eri_elem`), and
each value accumulated over its distinct 8-fold permutations
(`distinct_eri_orbit`, L2).

**Block kernels** (`src/integrals/block_eri.h`): `prepare(request)`,
`value(ca, cb, cc, cd)`, `covers(total_L, kernel)`. `NoBlockKernel` for
explicit engines; for Auto, `ClosedFormBlockKernel{3}` (`low_l_eri.h`)
chained with `HgpBlockKernel{4, 12}` (`hgp_block_eri.h`, 4 lanes in blocks
with K ≥ 3, `vrr_lanes<1>` below), one copy per thread. Both are Coulomb
only. The closed forms compute T for every primitive quartet of a block and
make one batched Boys call; the HGP kernel makes one per lane group (R2).
Scratch is a member (T1) and grow-only (`grow_to`): a buffer keeps its
largest size, so a block neither calls `resize` nor re-zeroes a regrown tail.

## How it is measured

`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a ctest).
With no arguments it drives `fused_fock_build` on 1, 4 and 8 waters (6-31G,
6-31G\*), one and four waters in cc-pVDZ, one water in cc-pVTZ and
naphthalene / 6-31G (axis-aligned D2h, group-averaged density), printing
per-class times, the closed-form builds at L ≤ 1, 2, 3, the block kernel one
total-L class at a time, the symmetric build, and an FNV-1a **fingerprint of
G** per build (28 in all). Modes, where `nwater` 0 is naphthalene:

- `--profile N [nwater basis]`: N production builds, for `sample`.
- `--k1 nwater basis`: today's per-component loop against dense, plus the
  production dense time (the headline numbers). Its timed dense arm prepares
  every block up front and reads every component unmasked, unlike
  production; use the production time for decisions (R1).
- `--family nwater basis [within] [bra]`: the shared-exponent family probe
  (`bra`: each bra member its own family, R3). `--census nwater basis`: the
  5b lane-fill census.

Rules learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling). Compare
  interleaved binaries, best of three or more, or one binary with a switch;
  a 2–3 % difference at best of three can be noise (the scratch step's 8 H2O
  "0.974×" was 1.01× at best of seven).
- **Build an A/B baseline as a whole target, never by mixing objects.** When
  R2 added a member to `ClosedFormBlockKernel`, a binary mixing old and new
  objects violated the one-definition rule and crashed in `--k1`. Build the
  target, copy the binary out, then change the source.
- **Swapping a header within the same second as the last build can leave
  objects stale**: `make` compares timestamps to the second here. Touch the
  header, or wait, before rebuilding (a mixed binary aborted silently).
- **A fingerprint, not a timestamp, says which code ran**, and a failed build
  silently reruns the old binary. Check the exit status and the new symbols.
- **Fingerprints compare whole builds.** The test density comes from
  `Eigen::MatrixXd::Random`, whose state carries across fixtures (new
  fixtures go at the end), and any change to the compiled file can move bits
  on paths it did not touch. To decide whether a kernel change is bitwise,
  compile the old and new kernel into **one** test and compare their outputs
  call by call on real shell pairs.
- **Restructured code is not bitwise under g++'s defaults.** g++ compiles C++
  with `-ffp-contract=fast`, so whether `a + b·c` becomes a fused
  multiply-add depends on the surrounding code (R2). The gate for a
  restructure is tolerance plus the suites; "bitwise" is a claim to test.
- **Attribute before predicting.** Inlined helpers hide in their caller's self
  time; temporary builds route phases through named `noinline` wrappers and
  add counters, never committed. Read a `sample` call tree for who calls a
  hot leaf (the scratch step's `memset` was inside `vector::resize`).
- **Concurrent runs of a freshly linked binary have taken ~126 s each** to
  start (cause not found; the Boys blob is cached and a lone start takes
  0.07 s). Run benchmark binaries one at a time.
- **zsh does not word-split an unquoted variable.** Use `${=f}` in loops over
  argument strings, or write the loop in bash.

## Where the time goes

`sample` self time over 10 s of `--profile` production builds, bucketed by
symbol, 2026-10-24 (after R2, before grow-only scratch). One thread; four
threads in brackets where they differ by more than a point:

| phase | 8 H2O 6-31G | 4 H2O 6-31G\* | 4 H2O cc-pVDZ | 1 H2O cc-pVTZ | naphthalene D2h |
|---|---|---|---|---|---|
| closed-form kernel (`low_l_shell_eri`) | 39 % | 25 % | 39 % | 13 % | 40 % (35 %) |
| Boys (core evaluation + entry) | 22 % | 14 % | 23 % | 7 % | 22 % (20 %) |
| closed-form readout (`value`) | 13 % | 9 % | 6 % | 2 % | 7 % |
| digestion loop (`dense_body`) | 20 % | 19 % | 11 % | 21 % (20 %) | 20 % (17 %) |
| HGP VRR + finish + prepare | 2 % | 29 % (28 %) | 19 % (18 %) | 52 % (50 %) | 5 % |
| scratch zeroing + `vector::resize` | 3.5 % | 4.7 % | 2.8 % | 4.0 % | 2.4 % |
| skeleton weight | — | — | — | — | 3 % |
| idle at four threads | 1 % | 5 % | 3 % | 6 % | 11 % |

- **The closed forms and Boys are the build** on sp, Pople-d and cc-pVDZ
  bases (38–62 %); Boys is nearly all the core's evaluation, so batching
  removed call overhead, not work.
- **HGP is the build on cc-pVTZ (52 %) and a third of 6-31G\***, already
  laned; no open item targets it except K2.
- **Scratch zeroing has since gone** to ≤ 0.1 % (grow-only scratch).
- **The four-thread idle was the barrier closing the parallel loop**
  (`gomp_team_end`), not serial work: on naphthalene three threads waited
  10.3 / 17.3 / 16.1 % and the fourth 0.6 %. Under S1 the work per bra unit
  is uneven and a static schedule cannot adapt; dynamic bins (partition)
  recover most of it.

The HGP block kernel's cost by primitive quartet and by block, measured after
D1 (temporary timers and counters):

| | 4 H2O / 6-31G\* | 4 H2O / cc-pVDZ | 1 H2O / cc-pVTZ |
|---|---|---|---|
| primitive quartets per build | 83 k | 432 k | 121 k |
| ket elements per primitive quartet | 203 | 183 | 305 |
| VRR table per primitive quartet | 5.1 KB | 4.5 KB | 7.5 KB |
| per primitive quartet, blocks with K ≥ 9 | 290–430 ns | 290–410 ns | 320–650 ns |
| a block with K = 1 | 1.2 µs | 1.1 µs | 2.3 µs |
| kernel time in K = 1 blocks | 10 % | 8 % | 23 % |
| HRR and readout (`finish()`) | 16 % | 10 % | 15 % |

K is the number of primitive quartets in a block after P1. The ket VRR runs at
~3 GFLOP/s: the fixed cost of a small recurrence per primitive quartet, not
arithmetic and not memory. The closed forms spend 49–65 % of their time in
blocks with K > 64, at 7–13 ns per primitive quartet.

**Shared exponents** (D0, after Schwarz and P1). The GBS reader stores a
general contraction as separate shells over the same exponents, and Pople SP
shells share exponents across L. Merging them would cut primitive passes by:

| fixture | merge within L | merge across L |
|---|---|---|
| 8 H2O / 6-31G | 1.00× | 2.23× |
| 4 H2O / 6-31G\* | 1.01× | 2.13× |
| naphthalene / 6-31G | 1.00× | 3.73× |
| 1 H2O / cc-pVDZ | 4.82× | 4.82× |
| 4 H2O / cc-pVDZ | 5.11× | 5.11× |
| 1 H2O / cc-pVTZ | 4.96× | 4.96× |

## Symmetry

**The fused signed-operation path.** Naphthalene / 6-31G (D2h, 8 operations),
one Fock build, after D1 (before 2b and K1.4):

| | no symmetry | symmetry | ratio |
|---|---|---|---|
| canonical quartets visited | 13.37 M | 5.57 M | 2.4× |
| orbit representatives | — | 3.35 M (1.43 M forbidden) | 4.0× |
| integral values computed | 13.37 M | 1.92 M | **7.0×** |
| Fock accumulations | 104.7 M | 15.0 M | **7.0×** |
| build time | 1.011 s | 0.295 s | **3.4×** |

- **A planar molecule caps the orbit at 4**: the mirror through the plane
  fixes every AO, so its factor of 2 reappears as forbidden (zero) integrals.
- **What held the build at 3.4×** was per-block and per-quartet work that
  falls less than 7×; K1.4 moved most of the orbit test to the block.
- **A speedup can shrink a ratio**: symmetry paid 5.2× after S1 and 3.4× after
  D1, because D1 sped up the plain build more.

**The full point-group path** (SAO-blocked HF, direct SCF). A separate loop
(`full_symmetry_build_skeleton` in `src/scf/scf.cpp` →
`{os,hgp,rys}_symm.cpp`) fills a dense n_b⁴ skeleton per Cartesian component;
Auto resolves to HGP there (F2). It is persisted when n_b ≤ `threshold`
(default 100); above that the SCF takes the fused path with the geometry's
D2h-subgroup operations (F1), and so K1.4. RHF, `scf_mode direct`, one thread:

| molecule | n_b | full group, persisted | fused, no symmetry | before F1 | F1 |
|---|---|---|---|---|---|
| H2O / cc-pVDZ (C2v) | 25 | 0.047 s + 0.0004 s/iter | 0.0075 s/iter | | |
| NH3 / 6-31G\* (C3v) | 21 | 0.017 s + 0.0003 s/iter | 0.0024 s/iter | | |
| CH4 / cc-pVDZ (Td) | 35 | 0.054 s + 0.0016 s/iter | 0.018 s/iter | | |
| benzene / 6-31G\* (D6h) | 102 | 3.0 s (`threshold 200`) | 13.7 s | 17.7 s | **4.08 s** |
| naphthalene / 6-31G (D2h) | 106 | 7.3 s (`threshold 200`) | 21.3 s | 66.6 s | **6.35 s** |

Below n_b = 100 the persisted skeleton is 6–19× cheaper per iteration than a
fused build (F1 numbers predate K1).

## What landed

| step | commit | what it removed or added | gain | gate |
|---|---|---|---|---|
| C0 | Boys v1.1.0 | libm calls inside one branch of the Boys core (17.7 % of the build) | 1.37× (8 H2O), 1.13× (4 H2O\*) | two iteration counts re-pinned at the noise floor |
| C1 | `c40a016c` | a `ShellPair` rebuilt per component quartet | 1.16–1.26× | bitwise |
| C2–C3.5 | `3c028544`, `4bbb2724`, `f0f7ceae` | closed-form shell kernels through total L ≤ 3 | 1.22–1.28×, 1.61×, 1.87× on 8 H2O | per-quartet sweep against HGP, 6.1e-16 |
| H1 | `46dadc6a` | MMAX+1 scalar Boys calls → one vector call (HGP, RI); VRR zero-fill | 1.10–1.12× × 1.04–1.06× | bitwise |
| L2 | `659883ab` | the quadratic search in `distinct_eri_orbit` | 1.20× / 1.09× / 1.11× | exhaustive oracle, 1296 tuples |
| L1 | `de72ca2c` | ket blocks whose first shell sorts below the bra's | 1.06× | bitwise |
| L3a | `f88991fe` | the full orbit built for non-representatives | 1.70× (D2h) | exhaustive, synthetic group |
| L3b | `a1f99d7a` | per-quartet orbit allocation; body per symmetry mode | 1.23× (D2h) | bitwise |
| P1 | `15c532e4` | primitive pairs with \|c_a c_b N_a N_b (π/ζ)^{3/2} K_ab\| < 1e-24 | 1.15× / 1.10× / 1.04× | G byte-identical |
| L3c | `15c532e4` | blocks an operation sends below the bra's first shell | 1.07× (D2h) | `check_real_symmetry_ops` |
| S1 | `7ec71009` | per-orbit-element replication of every G write | 1.33× (D2h) | `check_skeleton_symmetry` |
| fix | `fd51d29d` | H1's Boys buffer overflow on (ff\|ff) | — | f/g/h spherical gates |
| D1 | `52698142` | the per-component primitive loop at total L 4–12 | 1.5× sp, 4.3–4.9× d, 12.9× f | 20 fingerprints; `check_hgp_block_kernel` |
| F2 | `17cf5adf` | `engine auto` falling through to per-component OS on the full point-group path | 1.07–1.12× per skeleton | energies unchanged |
| F1 | `92838e03` | the n_b⁴ skeleton rebuilt every iteration above `threshold` | 4.3× benzene, 10.5× naphthalene (whole SCF) | energies vs the persisted path |
| T1 | `92838e03` | `thread_local` block-kernel scratch | neutral | 28 fingerprints identical |
| 2b | `bbe705fc` | one primitive quartet per VRR pass → four (lanes) in blocks with K ≥ 3 | 1.38× cc-pVTZ, 1.21–1.23× cc-pVDZ / 6-31G\*, 1.05–1.07× sp | block sweep ≤ 1.1e-14, mutation-tested |
| K1 | `5fd08496` | per-component digestion on symmetric densities | 1.11–1.22× | `check_dense_digest`; 5 mutations; MPI 1/2/4 |
| K1.4 | `f2849647` | per-component skeleton weight and per-block canonicalization under S1 | 1.32× (D2h) | 32 checks under real C2v; 8 mutations |
| F4 | `d0224110`, `0c8a9fc7` | `guess sad` aborting full point-group SCF on C3v/D6h | correctness | `nh3_c3v_fullsym_631gd_sad_direct` |
| W1 | `ffc44caf` | contiguous-range thread schedule on a triangular loop | 1.10–1.97× at four threads | `planck-fock-accumulate` at 1/4/8 threads |
| core v1.2.0 | `planck-boys-core` `aa44c44` | single-x evaluation overhead; adds a batched entry | ~1.01–1.02× | bitwise against v1.1.0 through the C ABI |
| R2 | `102f1d01` | one Boys call per primitive quartet / lane → one batched call per block / lane group | 1.01–1.11× (1 thread), 1.03–1.11× (4) | tolerance, ≤ 1.1e-16 (g++ FMA contraction) |
| loader | `13c60826` | the untested `boys_vec` fallback: core v1.2.0 is required, a v1.1.0 license fails at load | — | `planck-boys-vec`; v1.1.0 license checked by hand |
| scratch | `0e3e074d` | per-block `vector::resize` and re-zeroed regrown tails | 1.03–1.04× on 6-31G\* / cc-pVTZ, ±1 % elsewhere | bitwise, 28 fingerprints |
| partition | `ea353a83` | partials keyed by thread (G depended on the thread count) and a static schedule under uneven S1 work | 1.06× naphthalene, 1.02× cc-pVTZ at four threads; ±1–2 % elsewhere | 28 fingerprints identical at 1/2/4/8 threads |

B2.1 (`66669cd5`, family-quartet units in the dense traversal, off by
default) landed and was deleted in `1e047aa2`; see Families.

Suites: smoke 43/45 and extended 135/139 after W1 and R2 (the known rccsdt
routing cases); smoke 43/45 after the loader, scratch and partition commits, extended
not re-run. Details worth keeping:

- **Closed forms (C2–C3.5).** `low_l_shell_eri` treats a quartet as up to
  three *raises* (center, axis) in center order. Per primitive quartet, with
  $`\theta_m = c\,K_{AB}K_{CD}\,2\sqrt{\rho/\pi}\,F_m(T)`$, $d$ the raised
  center's $P-A$, $P-B$, $Q-C$ or $Q-D$, and $w$ its $W-P$ or $W-Q$:

  ```math
  \begin{aligned}
  A_i(m) &= d_{1,i}\,\theta_m + w_{1,i}\,\theta_{m+1} \\
  B_{ij}(m) &= d_{2,j}\,A_i(m) + w_{2,j}\,A_i(m+1) + [i=j]\,\kappa_{12}(\theta;\,m) \\
  C_{ijk} &= d_{3,k}\,B_{ij}(0) + w_{3,k}\,B_{ij}(1) + [k=i]\,\kappa_{13}(A^{(2)}_j) + [k=j]\,\kappa_{23}(A_i)
  \end{aligned}
  ```

  $\kappa_{yx}$ couples a new raise to an earlier one it lowers: on the bra
  $\bigl(X(m) - \tfrac{\rho}{\zeta}X(m+1)\bigr)/2\zeta$, on the ket the same
  with $\eta$, across $X(m+1)/2(\zeta+\eta)$. At most 27 outputs; per-AO
  `norm_ratio` at readout. Gate: every total-L ≤ 3 component quartet of water
  in 6-31G, 6-31G\*, cc-pVDZ against HGP, every reachable raise layout
  required, mutation-verified.
- **S1.** When every density has the molecule's symmetry, each non-forbidden
  representative is accumulated once, weighted by its orbit size, and at the
  end (`quartet_orbit.h`)

  ```math
  G_{\mu\nu} = \frac{1}{|G|}\sum_R s_R(\mu)\,s_R(\nu)\,G^{\mathrm{skel}}_{R(\mu)R(\nu)}.
  ```

  A non-symmetric density falls back to replication. Direct HF on linear
  molecules does not gain (no SAO blocking for infinite groups).
- **D1.** The HGP block kernel runs the Obara–Saika VRR per primitive quartet
  into an **L-major** table [e0|f0]^(m), contracts the m = 0 slice, then HRR
  once per shell quartet (`finish()`). Every total-L class 4–12 won or tied.
  Gate: 4.6 M component quartets in 256 classes up to (ff|ff) against
  per-component HGP, ≤ 8.9e-15, mutation-verified.
- **F1.** The axis-flip operations form the largest D2h subgroup of the point
  group, and an SAO density is symmetric under any subgroup, which is all S1
  needs; an unpersisted full-group run falls through to the fused build.
- **F4.** The raw SAD density is group-invariant; the guess built from it was
  not, because making it idempotent (top n natural orbitals) is ill-posed
  when a degenerate set straddles the cutoff. The guess is now the raw
  density scaled to the electron count, not idempotent. Full account,
  including why fixing the projection's metric alone made things worse:
  `docs/PLANCK_TEACHING_GUIDE.md` §6, "Why the SAD guess is not made
  idempotent".
- **2b.** `hgp_block::vrr_lanes<B>` runs the VRR for B primitive quartets at
  once, lane-minor (`[e][f][m][lane]`). `HgpBlockKernel` uses B = 4 in blocks
  with K ≥ 3, padding lanes at weight 0; smaller blocks run `vrr_lanes<1>`,
  so the recurrence exists once. B = 2 sits between, B = 8 is slower than 4,
  and a K = 3 block is cheaper padded to 4 than run alone. Mutations
  (dropping the laned W−Q term, weighting padding lanes, contracting lane 0
  only) each fail the block sweep by 0.13–5.4.
- **K1.** `fused_fock_build(..., dense_digest = true)`. Per build, because
  the two traversals visit different block sets; mixing them double-counts
  or drops quartets. Writes with v′ = deg · value: Coulomb `G_ab, G_cd += ½ P
  v′`; exchange `G_ac, G_bd, G_ad, G_bc` with −⅛ (RHF Combined), +¼ (raw K),
  −¼ per spin (UHF Combined); then (G + Gᵀ)/2. Exact only for a symmetric
  density, hence the density test. The component Schwarz mask keeps screening
  identical (unmasked, G moves by the dropped 2.8–6.2e-10). One body per
  active write group took back the 2–8 points a generic body lost. MPI: Auto
  direct SCF matches serial at 2 and 4 ranks.
- **K1.4.** Skeleton-mode builds go dense with a per-component S1 weight; ops
  that send the block's canonical shell quartet lower skip the block, only
  ops fixing it are applied per component, identity-map ops get a sign test,
  and the per-block test reads a per-build pair-image table (0.273 → 0.206 s
  on naphthalene). Gate: dense against per-component under water's real C2v
  (op count asserted), relative to max|G|. A mutation re-run found the gate
  NaN-blind (`std::max(worst, NaN)` returns `worst`) and unable to tell
  whether dense ran when block counts tie; both closed (`max_output_diff`
  returns +inf on non-finite; the counting kernel records blocks whose bra
  pair sorts after the ket pair, which only the per-component loop
  prepares). 7 of 9 mutations caught; the 2 survivors drop the sign test,
  which only saves work.
- **W1.** The dense traversal is a triangle (bra p visits every ket q ≥ p);
  `schedule(static)` gave the first of four threads 1 − (3/4)² = 44 % of it, a
  2.29× ceiling, measured 2.32–2.33× on 6-31G\*, cc-pVDZ, cc-pVTZ and 1.72× on
  naphthalene. `schedule(static, 1)`:

  | fixture, 4 threads | static | static, 1 | gain | 1 → 4 now |
  |---|---|---|---|---|
  | 8 H2O / 6-31G | 0.0353 s | 0.0320 s | 1.10× | 3.6× |
  | 4 H2O / 6-31G\* | 0.0291 s | 0.0190 s | 1.53× | 3.6× |
  | 4 H2O / cc-pVDZ | 0.1722 s | 0.1090 s | 1.58× | 3.7× |
  | 1 H2O / cc-pVTZ | 0.0320 s | 0.0209 s | 1.53× | 3.6× |
  | naphthalene / 6-31G | 0.1229 s | 0.0623 s | 1.97× | 3.4× |

  The per-component traversal shares the loop and gains too.
- **R2.** Core v1.2.0 adds `planck_boys_batch(x, count, nmax, out)` with
  `out[i·(nmax+1) + n] = F_n(x[i])`, bitwise equal to the vector entry for
  every x (verified through the C ABI against v1.1.0 on 90 M single-x values,
  the vector entry with its NaN tail, the batched entry at five widths, Apple
  clang and g++-15; per x the batched entry is 1.53–2.26× faster). Not bitwise
  in Planck, by the compiler: across 67 032 real quartet calls T and every
  Boys value are byte-identical and 583 outputs differ by ≤ 1.1e-16, all in
  the (ss|ss) branch `out[0] += base·F[0]`, which g++ fuses differently in the
  two-pass loop (with `-ffp-contract=off` the rounding moves rather than
  vanishes). The loader then made the batched entry required (`13c60826`).
- **Scratch.** The 2.4–4.7 % of zero/resize was almost all inside
  `vector::resize`: out of line, and a buffer shrunk by a small block
  re-zeroes its tail when the next larger one regrows it. `grow_to` keeps the
  largest size, which gets the effect of sizing once per build without a
  setup step; `acc` still zeroes its live extent. The removed share turned
  into time only where HGP runs.
- **Partition.** Before, G at one thread was not G at four (CO2 / 6-31G\*
  direct differed in the DIIS residual digits and the last kcal/mol digit;
  all 10 Hartree digits identical). Keying partials by bin fixes which
  terms each partial sums and in what order, whatever thread runs it; this
  is the FCI sigma-build rule (what must be fixed is the partition of work
  into accumulators, not just the order they are summed). Not bitwise
  against the thread-keyed build, since one thread now sums 64 partials.
  Bin count, five reps against the build before, four threads:

  | fixture | 16 bins | 32 bins | 64 bins | 128 bins |
  |---|---|---|---|---|
  | 8 H2O / 6-31G | 0.950× | 0.969× | 0.982–0.990× | 0.972× |
  | 1 H2O / cc-pVTZ | 0.985× | 1.010× | 1.021× | 1.037× |
  | naphthalene / 6-31G | 0.987× | 1.026× | 1.058–1.064× | 1.075× |

  At one thread 16–64 bins are within ±1.5 %; 128 costs 1.4–2.2 %, so 64.

## Measured, not landed

**D1c — the electron-transfer recurrence (declined).** Head-Gordon and
Pople's ETR at m = 0 after a one-centre VRR out to |e| ≤ M:

```math
[e|f{+}1_j] = \bigl(\tfrac{\zeta}{\eta}PA_j + QC_j\bigr)[e|f]
+ \tfrac{f_j}{2\eta}[e|f{-}1_j] + \tfrac{e_j}{2\eta}[e{-}1_j|f]
- \tfrac{\zeta}{\eta}[e{+}1_j|f]
```

A fixed bra→ket direction was off by 2.8e-9 on cc-pVTZ; transferring per
primitive quartet toward the larger exponent sum gave 2.4e-13 (27× the VRR).
At that precision it was 1.04–1.06× against a predicted 1.1–2.3×: the ket
phase fell 2×, the bra VRR to |e| ≤ M grew 3.3–3.5×. A row layout was bitwise
and time-neutral: the cost is per primitive quartet.

**5b — a class-batched scheduler (declined).** Group surviving blocks by
class and evaluate classes in batches across blocks. `--census`:
93.9–99.5 % of HGP-class primitive quartets were already in within-block
lanes (closed forms 99.5–99.8 %); pooling would buy at most ~1.15× on the
lane work, and enumerating with a survival check cost 2–12 % of the build.
cc-pVTZ's K = 1 blocks are 23 % of HGP kernel time but ~6 % of its primitive
quartets: their cost is per block.

**Families: D2a, B0, B1, B2, R3 (none in production).** A family is a set of
shells on one atom whose exponent sets nest (cc general contractions within
L, Pople SP across L). A family quartet runs one primitive pass over the
union of its members' primitive pairs; members split by their own total L
(L ≤ 3: T, prefactor and Boys once per primitive quartet, raises once per
layout; L ≥ 4: one VRR pass at the high members' maximum L, then each
member's `finish()`).

- **Kernel gains (B0, both sides sharing, against today's kernels on the
  same member blocks, values within 7.2e-16).** L ≤ 3: 1.17–1.78× (cc-pVTZ
  1.78×, cc-pVDZ 1.73× then, 1.58× after R2; naphthalene 1.37×). L ≥ 4: a
  scalar family pass is 0.92–1.03× on cc bases and 0.48–0.81× where little
  merges; lanes and families compete for the same primitive work, so lanes
  took the HGP half (B2.3 dropped).
- **Sharing on both sides needs the blocks visited in family order, and that
  order costs** (B1, B2.1): +3.5 % / +3.6 % / +7.4 % at one thread on 8 H2O,
  6-31G\* and naphthalene, ±1 % on cc bases, −4.7 % to +8.8 % at four
  threads. Its earlier four-thread "gains" (−12 to −14 %) were the load
  imbalance W1 fixed. A first cut was +50 % / +84 % on 8 H2O until a
  unit-level Schwarz bound removed ~1.3 M fully screened units.
- **Sharing on the ket side only keeps today's order but loses most of the
  merge** (R3). `--family nwater basis bra`, closed-form rows, one thread:

  | fixture | two-sided | ket side only | passes fewer, two-sided → ket only | closed forms + Boys | ket only, build (Amdahl bound) |
  |---|---|---|---|---|---|
  | 8 H2O / 6-31G | 1.22× | 1.06× | 2.29× → 1.51× | 61 % | ~1.03× |
  | 4 H2O / 6-31G\* | 1.04× | 0.90× | 2.17× → 1.45× | 38 % | ~0.96× |
  | 4 H2O / cc-pVDZ | 1.58× | 1.13× | 5.91× → 2.31× | 61 % | ~1.07× |
  | 1 H2O / cc-pVTZ | 1.56× | 1.08× | 5.88× → 2.06× | 20 % | ~1.01× |
  | naphthalene / 6-31G | 1.20× | 0.95× | 3.61× → 1.79× | 62 % | ~0.98× |

  Under the 1.10× gate set on cc-pVDZ and naphthalene, and the bound ignores
  the production loop's screening and grouping overhead.
- **What would pay**: two-sided sharing in family order on cc bases only
  (≤ ~1.29× of a cc-pVDZ build); on Pople bases the order's cost leaves
  ~1.04–1.08×. That means a traversal chosen by basis, which was not taken
  up (B2.2 parked). The family-order traversal itself (B2.1, `family_units`)
  was deleted (`1e047aa2`); B1 survives as a git stash.

**R1 — the closed-form readout as a flat array (declined).**
`ClosedFormBlockKernel::value` re-derives each component's raise slot on every
call. R1 moved that into `prepare`, filling a flat norm-folded `out_[27]` so
`value()` is one load. Bitwise (28 fingerprints matched). Predicted
1.03–1.12×; production build against R2, best of three interleaved:

| fixture | 1 thread | 4 threads | `prepare` noinline, 1 thread |
|---|---|---|---|
| 8 H2O / 6-31G | 0.953× | 0.922× | 0.981× |
| 4 H2O / 6-31G\* | 0.960× | 0.941× | — |
| 1 H2O / cc-pVDZ | 0.957× | 1.000× | — |
| 1 H2O / cc-pVTZ | 0.976× | 0.971× | 0.990× |
| naphthalene / 6-31G | 0.933× | 0.942× | 0.941× |

Two causes, each about half. **Inlining:** production prepares lazily inside
the component loop, so the larger `prepare` is inlined into the hot
digestion nest of every mode; `noinline` recovers 2–3 %. **Eager work:**
`prepare` fills every component, production reads only those passing the
Schwarz mask and, under S1, with nonzero weight (85 % read on 8 H2O; the
symmetric build loses most, its read fraction not counted). The `--k1` timed
arm showed a net gain (+12 ms prepare, −16 ms digestion) because it prepares
outside the loop and reads everything.

**S2 — threading the serial symmetric phases (premise refuted).**
`density_has_symmetry` and `symmetrize_over_ops` are 0.04 % of samples; the
four-thread idle they were blamed for is load imbalance at the loop's closing
barrier (see Where the time goes).

**P2 — quartet-level primitive screening (declined: G must stay accurate to
1e-10).** After P1, ~20 % of primitive quartets on water clusters have a
prefactor bound between 1e-24 and 1e-12. Skipping below ε, Schwarz held exact
(before D1):

| fixture | ε = 1e-14 | ε = 1e-12 | max\|ΔG\| at 1e-14 / 1e-12 |
|---|---|---|---|
| 8 H2O / 6-31G | 1.13× | 1.37× | 4.0e-12 / 3.3e-10 |
| 4 H2O / 6-31G\* | 1.07× | 1.18× | 4.9e-12 / 3.7e-10 |
| naphthalene D2h | 1.15× | 1.22× | 4.8e-12 / 4.8e-10 |

**Total L > 12 (declined: g-shell bases are not routine work).** Those blocks
take per-component HGP; the VRR table would be 3.7 MB for (gg|gg).

## Findings

**The floor was the primitive loop, not the recurrence.** The (ss|ss) closed
form has no recurrence, yet per component it was only 1.66–1.75× faster than
HGP: every call reran the primitive loop. Once per shell quartet: 4.9×, and
10–16× at L = 4.

**Below the block kernel, the floor is the primitive quartet.** A few hundred
elements of work per primitive quartet is overhead-bound; the lever is the
number of primitive passes and how many run together, not their arithmetic.

**The contract between generation and digestion caps what batching can buy.**
While every integral leaves its kernel through `value()` and is digested
alone, batching blocks changes nothing outside the kernels; dense per-block
digestion did (K1).

**Sharing work across shells needs both sides and a cheap order.** Two-sided
family passes gain 1.2–1.8× on the closed forms, but the family order costs
up to 7 % on Pople bases, and fixing the bra to keep today's order discards
most of the merge (R3).

**Moving work earlier is not removing it.** R1 moved the readout decode into
`prepare` and lost 2–8 %: the bigger function inlined into the hot loop, and
it computed values the loop never reads.

**A four-thread gain can be load imbalance in disguise.** B2.1's family units
measured 12–14 % faster at four threads; on a balanced schedule the gain
vanished and only the one-thread cost remained. Check the scaling of the
baseline before crediting a parallel gain to the change.

**A triangular loop needs a cyclic schedule.** Contiguous chunks of a
triangle hand the first thread 1 − (1 − 1/T)² of the work; at four threads
that capped the build at 2.29×, and one line (W1) recovered up to 1.97×.

**Idle time is not serial work until the waiting thread says so.** S2 blamed
the four-thread idle on two serial functions; a per-thread split of the
sample put every waiting sample at the loop's closing barrier.

**Determinism and dynamic scheduling are compatible** once accumulators are
keyed by a fixed partition of the work rather than by thread: the partition
item made G thread-count invariant and dynamically balanced in one change.

**A share of time is not a share of work, and an out-of-tree share is not a
gain.** cc-pVTZ's K = 1 blocks are 23 % of HGP kernel time and ~6 % of its
primitive quartets (5b). Boys profiled at 9–32 % of the build, and a 1.3–2.3×
faster Boys evaluation plus batching realized 1.03–1.13×. Scratch zeroing was
3.5 % of 8 H2O and removing it was worth nothing measurable there.

**An idea that measured slower may have been its first implementation.** A
contract-once hoist measured 1.14–1.62× slower; D1's block kernel is the same
idea at 1.5–13× faster, with L-major indexing.

**Price a change on the production loop.** The `--k1` timed arm prepares every
block up front and reads every component unmasked; R1 looked like a net gain
there and was 2–8 % slower in production.

**A profile share is a lower bound on what removing work is worth, and can
mislead the other way.** The `ShellPair` rebuild was 8.4 % of samples and
worth 1.16–1.26×; L3c cut 59 % of the symmetric loop's iterations for 1.07×;
T1 removed a 2.5–4.7 % frame for nothing measurable.

**When a fix does not move the profile share, the hot call site is
elsewhere.** K1.4's first fix removed the per-component canonicalization;
time improved 3 % but the function's share did not move. The hot calls were
in the per-block test, and a table took the build from 1.11× to 1.32×.

**Classify by the work, not the container.** D2a first classified family
quartets by the family's maximum L, which sent Pople's (ss|ss) members
through the HGP VRR (0.51–0.53×).

**Granularity decides load balance.** One family pair per unit left four
threads 76 % slower on cc-pVTZ.

**Precision can decide an algorithm.** The ETR's −ζ/η subtraction cost 4–6
digits until the transfer direction was chosen per primitive quartet.

**A buffer sized for one call site can be wrong for another.** H1's Boys
buffer was sized for one component; HGP's hoisted block path reached
MMAX = 36 on (ff|ff): SIGBUS, hidden for two days among known failures. **A
red gate that is "already failing" still needs its cause checked.**

**A gate that never runs is worse than none.** `planck-boys-vec` was built but
never registered in ctest and never loaded the Boys library, so it segfaulted
unnoticed since the license gate. R2 found it by needing it.

**Code layout can cost more than the change.** Hoisting L3b's orbit buffer
slowed molecules without symmetry by 5 % through register allocation; R1's
larger `prepare` cost 2–3 % the same way.

**Screening must never reach the Schwarz table.** The first P2 experiment gave
max|ΔG| = 3e-6 because `schwarz_table_hgp` builds Q with the same kernel.

**A cost-only test cannot be value-gated.** S1's forbidden-quartet skip, and
K1.4's sign test, survive inversion: the final symmetrization cancels a
forbidden quartet exactly.

**Gates must be able to fail, including on NaN.** Every kernel gate was
mutation-tested; K1.4's gate passed a NaN G because `std::max(x, NaN)` returns
x, and passed a build that never went dense because block counts tied.
Compare with `!(worst <= tol)` and assert the path that ran directly.

**Print a census's mapping before believing its total.** D0's first count
never merged cc-pVDZ oxygen's equal-size 1s and 2s (1.47–1.83× instead of
4.8–5.1×). K1's first max|ΔG| of 6e-10 looked like a weight error and was the
dropped component Schwarz.

**Write the prediction down first.** It caught a misread class share, C4a's
3–4× low predictions, 5b's premise, B1's Pople cost, R2's shortfall, R1's
loss and R3's shortfall.

**A copied fixture can hide the thing under test.** A copied `Calculator`
kept only the identity operation, and the first K1.4 "C2v" gate passed with
|G| = 1. Assert the fixture's defining property.

## Open

**R1b — the readout from a per-build table (conditional).** R1's lesson
applied: the decode depends only on each AO's Cartesian exponents, so
precompute per AO its axis code and norm ratio once per build and keep
`value()` lazy as a few table loads. Bitwise. Ceiling 2–13 % (largest on
8 H2O); land only at ≥ 1.03× on 8 H2O, timed on the production loop.

**C5 — screened kernels.** Both block kernels decline screened kernels, so
range-separated hybrids take per-component HGP. They need `hgp_vrr`'s
scaling: T × `boys_scale`, prefactor × `prefactor_scale`, W−P/W−Q ×
`screen.rho / rho`, and the straddling coupling × `boys_scale` (the
`inv_2_delta` gotcha). Gate: a screened sweep against OS on a mixed bra/ket
fixture for LongRange and ShortRange, plus the `water_*_engine_os_vs_hgp`
comparators.

**F3.** After F1 the full-group skeleton runs only where it is persisted, so
feeding it from the block kernels shortens a one-time build (0.017–0.054 s on
small molecules, 1.3–3.4 s near n_b = 100). Worth it only if `threshold` is
raised routinely.

**Outside the main line.** **D3:** RI-JK recomputes its density-independent
pair factors twice per iteration, ~61 % of the RI Fock build
(`docs/ERI_FOCK_BLAS_SCOPE.md`). **K2 (conditional):** batches of same-class
ket blocks for a fixed bra unit (lanes across kets, bra-side work hoisted, the
HRR as array work, digestion as small matrix products); the only lever left
on HGP, and only if a census shows per-block fixed costs (setup, HRR at
10–16 % of the HGP kernel) worth the largest restructure here. **C4b
(deprioritised):** a four-raise closed form matched HGP to 1.3e-15 on every
total-L = 4 layout and measured 10–16× over per-component HGP (C4a), but its
competitor is now the block kernel, whose L = 4 work is 4–9 % of sp builds.

## Ruled out

- **Forcing `accumulate_term` / `distinct_eri_orbit` inline**: stubs < 1 %.
- **Avoiding `boys(0, T)` in the (ss|ss) branch**: the cost was the Boys
  evaluation itself (addressed in the core, R2).
- **A generator for an L = 4 closed form**: the hand-written level was correct
  first time (C4a).
- **The ETR (D1c), a unit-stride row layout for the VRR, and a class-batched
  scheduler (5b).**
- **Family sharing in any form tried** (B1, B2.1, B2.3, R3), and the
  basis-dependent traversal that would pay (B2.2, parked).
- **Precomputing the closed-form readout in `prepare` (R1).**
- **Threading the serial symmetric phases (S2)**: not where the idle is.
- **Primitive screening (P2)**: breaks the 1e-10 accuracy requirement where
  it pays.
- **Block kernels above total L = 12**: not routine work.
- **Loosening S1's density tolerance** so linear HF takes the skeleton path:
  it would change the SCF trajectory.
- **Demanding bitwise equality from a code restructure under g++'s default
  flags**: see R2.

## Out of scope

- **Gradients**: derivative ERIs are a separate kernel family (Open Work
  Phase D); P1 covers them.
- **Spherical**: the direct spherical path back-projects to Cartesian and uses
  the same fused loop.
- **SAO blocking for linear point groups**: would let direct linear HF take
  S1's path; no tested workflow needs it.
