# Shell-Quartet ERI Kernels for the Fused Fock Build — Scope

**Question:** How much faster can the memory-direct (fused) Fock build get by
evaluating whole shell quartets at once (closed forms at low angular momentum,
a block kernel above), and what else in the build stands in the way?

**Status: in flight.** Rewritten 2026-10-23 as current state; earlier
revisions carried each step's plan, and their measurements are kept below.
When the open steps land or are abandoned, rewrite this as an answer.

- **Landed:** C0, C1, C2–C3.5, H1, L1–L3c, P1, S1, D1, F1, F2, F4, T1, 2b,
  K1, K1.4, B2.1 (default off), W1, and core v1.2.0 (the Boys release R2
  builds on).
- **Written, not committed:** R2's Planck callers (batched Boys calls).
- **Open, in order:** R1 (closed-form readout), R2 (land, then re-profile),
  R3 (ket-side family sharing), the fixed accumulator partition, C5
  (screened kernels), P2 (primitive screening, a policy call), total L > 12,
  F3. D3 is independent, K2 conditional, C4b deprioritised.
- **Declined or superseded:** D1c, 5b, B1, B2.2 (by R3), B2.3.

## Answer so far

| fixture (one thread, `engine auto`) | before this work | now (HEAD) | with R2 (uncommitted) | speedup |
|---|---|---|---|---|
| 8 H2O / 6-31G (nb = 104) | 1.50 s | 0.1135 s | **0.1083 s** | **~14×** |
| 4 H2O / 6-31G\* (nb = 76) | ~1.12 s† | 0.0679 s | **0.0654 s** | **~17×** |
| 4 H2O / cc-pVDZ | 2.573 s§ | 0.4004 s | **0.3813 s** | **6.7×**§ |
| 1 H2O / cc-pVTZ | 1.631 s§ | 0.0746 s | **0.0739 s** | **22×**§ |
| naphthalene / 6-31G, D2h (nb = 106) | 1.454 s | 0.2072 s‡ | **0.1873 s**‡ | **7.8×** |

At four threads (W1's round-robin schedule), HEAD → R2: 0.0309 → 0.0283 s,
0.0186 → 0.0177 s, 0.1071 → 0.0987 s, 0.0208 → 0.0202 s, 0.0606 → 0.0547 s
(1→4-thread scaling 3.4–3.7×).

`planck-eri-class-share --k1` production arm, Apple M3, `build-full`
(g++-15 `-O3`), best of three interleaved, Boys core v1.2.0, 2026-10-23.
"Before" is the build before C1, after the Boys fix C0 (itself 1.37× / 1.13×).
† The product of per-step ratios; never timed before C1. § Against the D0
build (after P1, before S1 and D1). ‡ With `sym_ops`, on a group-averaged
density. Explicit engines get only the engine-independent steps; on
8 H2O / 6-31G and 4 H2O / 6-31G\* Auto is ~7× and ~11× faster than explicit
`engine hgp`. Fock-build numbers, not SCF wall time. The ~200× libcint gap in
`docs/ERI_PERFORMANCE_SCOPE.md` predates this work and has not been
re-measured.

Per step, each a ratio over the build with all previous steps (one thread):

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c | D1 | 2b | K1 | R2 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | 1.49× | 1.05× | 1.14× | 1.05× | **~14×** |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | 4.90× | 1.23× | 1.17× | 1.04× | **~17×** |

On the symmetric fixture: L3a 1.70×, L3b 1.23×, L3c+P1 1.16×, S1 1.33×,
D1 ~1.5×, 2b 1.07×, K1.4 1.32×, R2 1.11× (1.454 → 0.855 → 0.702 → 0.597 →
0.454 → 0.295 → 0.274 → 0.206 → 0.187 s).

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
  quartets per pass (lanes, 2b) and cheaper or shared per-quartet work (R2,
  R3) do.
- **Outside the kernels, the cost was digesting one component at a time.**
  Digesting each block densely (K1) gave 1.11–1.22×, under the skeleton scheme
  (K1.4) 1.32×.
- **Then the threads.** The dense traversal is a triangle; a contiguous
  schedule capped four threads at ~2.3×. Round-robin (W1) gave 1.10–1.97× at
  four threads for one line.

## How the build works now

`Integrals::fused_fock_build` (`src/integrals/fused_fock.h`) chooses one of
two traversals **per build**. OpenMP `schedule(static, 1)` (round-robin, W1)
and the MPI stripe run over bra units in both; partials are keyed by
`omp_get_thread_num()` and summed in thread order, so a run is reproducible
at a given thread count (not yet across counts: the fixed partition).

**Dense (K1, K1.4).** Taken when the block kernel `covers()` every total L the
basis can produce (today: Auto, Coulomb, shells up to f, total L ≤ 12), every
density is symmetric to 1e-10, and there is either no symmetry or skeleton
mode. Per bra pair p, kets q ≥ p (or per family-quartet unit with
`family_units`, B2.1, off by default):

1. Block Schwarz. Under S1, each op's image of the block is (min, max) of two
   entries of a per-build table `op_pair[o][p]`; an image that sorts lower
   skips the block, and permuting ops that map it onto itself go to
   `block_ops`.
2. `prepare` the block, then every component above the component Schwarz
   bound: v′ = deg · weight · value, deg = (A=B?1:2)(C=D?1:2)(p=q?1:2). Under
   S1, weight is the S1 weight of the component's canonical form
   (`skeleton_orbit_weight` with what the block already decided: identity-map
   ops need only a sign test, only `block_ops` need canonicalization).
3. Six writes per component (two Coulomb, four exchange), from a body
   instantiated per active write group, then (G + Gᵀ)/2 after the merge (and
   the group average under S1).

**Per-component.** Everything else: explicit engines, screened kernels,
L > 12, non-symmetric densities, replication-mode symmetry. Block filters
(L1, block Schwarz, L3c), component filters (canonical order, component
Schwarz, the orbit test), the block kernel prepared at the first surviving
component (a declined block uses per-component `eri_elem`), and each value
accumulated over its distinct 8-fold permutations (`distinct_eri_orbit`, L2).

**Block kernels** (`src/integrals/block_eri.h`): `prepare(request)`,
`value(ca, cb, cc, cd)`, `covers(total_L, kernel)`. `NoBlockKernel` for
explicit engines; for Auto, `ClosedFormBlockKernel{3}` (`low_l_eri.h`)
chained with `HgpBlockKernel{4, 12}` (`hgp_block_eri.h`, 4 lanes in blocks
with K ≥ 3, `vrr_lanes<1>` below), one copy per thread, scratch as members
(T1). Both are Coulomb only. With R2, the closed forms compute T for every
primitive quartet of a block and make one batched Boys call; the HGP kernel
makes one per lane group.

## How it is measured

`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a ctest).
With no arguments it drives `fused_fock_build` on 1, 4 and 8 waters (6-31G,
6-31G\*), one and four waters in cc-pVDZ, one water in cc-pVTZ and
naphthalene / 6-31G (axis-aligned D2h, group-averaged density), printing
per-class times, the closed-form builds at L ≤ 1, 2, 3, the block kernel one
total-L class at a time, the symmetric build, and an FNV-1a **fingerprint of
G** per build. Modes, where `nwater` 0 is naphthalene:

- `--profile N [nwater basis]`: N production builds, for `sample`.
- `--k1 nwater basis`: today's per-component loop against dense, plus the
  production dense time (the headline numbers).
- `--b21 nwater basis`: family-quartet units against bra units.
- `--family nwater basis [within]`, `--census nwater basis`: the D2a family
  probe and the 5b lane-fill census.

Rules learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling). Compare
  interleaved binaries, best of three or five, or one binary with a switch.
- **Build an A/B baseline as a whole target, never by mixing objects.**
  Compiling one object from old headers and relinking with the target's
  `link.txt` is safe only if no inline class it uses changed. When R2 added a
  member to `ClosedFormBlockKernel`, the mixed binary violated the
  one-definition rule and crashed in `--k1`. Stash, build the target, copy
  the binary out, pop, rebuild.
- **After `git stash pop`, touch the restored files.** A restored header can
  get the same second as an object compiled just before the pop; `make` then
  treats the object as current and links stale code (here: a crash rather
  than a wrong number, by luck).
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
  multiply-add depends on the surrounding code. Moving arithmetic into a
  different loop shape can change last bits with identical inputs (R2). The
  gate for a restructure is tolerance plus the suites; "bitwise" is a claim
  to test, not to assume.
- **Attribute before predicting.** Inlined helpers hide in their caller's self
  time; temporary builds route phases through named `noinline` wrappers and
  add counters, never committed.
- **zsh does not word-split an unquoted variable.** Use `${=f}` in loops over
  argument strings, or write the loop in bash.

## Where the time goes

**Re-profile after W1 (2026-10-23), before R2.** `sample` self time over 10 s
of `--profile` production builds, bucketed by symbol, one and four threads
(the shares barely move between them):

| phase | 8 H2O 6-31G | 4 H2O 6-31G\* | 4 H2O cc-pVDZ | 1 H2O cc-pVTZ | naphthalene D2h |
|---|---|---|---|---|---|
| closed-form kernel (`low_l_shell_eri`) | 34–35 % | 23–25 % | 35 % | 13 % | 33–38 % |
| Boys (out-of-tree core) | 32 % | 18–19 % | 30–31 % | 9–10 % | 28–31 % |
| closed-form readout (`value`) | 12 % | 9 % | 6 % | 2 % | 6–7 % |
| digestion loop (`dense_body`) | 18 % | 17 % | 9 % | 19–20 % | 16–17 % |
| HGP VRR (`vrr_lanes`) | 2 % | 17–18 % | 11–12 % | 33–36 % | 3–4 % |
| HGP HRR / finish | 0 % | 7–8 % | 4 % | 12–13 % | 1 % |
| HGP prepare (incl. `vector::resize`) | 0 % | 2–3 % | 2 % | 3–4 % | 0 % |
| skeleton weight | — | — | — | — | 2–3 % |
| idle at four threads | 3 % | 4 % | 2 % | 6 % | 10 % |

- **The closed forms and Boys were the build** on sp, Pople-d and cc-pVDZ
  bases (41–66 %).
- **The readout is avoidable overhead** (R1): `ClosedFormBlockKernel::value`
  re-decodes each component's raise slot on every call, out of line through a
  weak-symbol stub; HGP's readout is one indexed load.
- **HGP matters on cc-pVTZ (49 %) and 6-31G\* (25 %)**, already laned.
- **Idle after W1 is small**, except naphthalene (10 %), where
  `density_has_symmetry` and `symmetrize_over_ops` run serially.
- **A Boys share did not convert into its expected gain.** The faster
  single-x core (v1.2.0) moved HEAD's builds by ~1–2 % and the batched calls
  by 1.01–1.11×, about 1.03–1.13× together against a 1.10–1.19× prediction
  from the 9–32 % share.

**Re-profile with R2 (2026-10-23, one thread, uncommitted callers):**

| phase | 8 H2O 6-31G | 4 H2O 6-31G\* | 4 H2O cc-pVDZ | 1 H2O cc-pVTZ | naphthalene D2h |
|---|---|---|---|---|---|
| closed-form kernel | 39 % | 25 % | 40 % | 13 % | 39 % |
| Boys (core evaluation + entry) | **22 %** | **13 %** | **22 %** | **7 %** | **24 %** |
| closed-form readout | 13 % | 9 % | 7 % | 2 % | 7 % |
| digestion loop | 20 % | 19 % | 10 % | 21 % | 19 % |
| HGP VRR + HRR + prepare | 3 % | 30 % | 19 % | 53 % | 5 % |
| scratch zeroing + `vector::resize` | 3 % | 5 % | 4 % | 5 % | 2 % |

Boys fell from 18–32 % to 13–24 % (7 % on cc-pVTZ), nearly all of it now the
core's evaluation itself: batching removed the call overhead, not the work.
The closed forms are now the largest phase on sp, cc-pVDZ and naphthalene,
and readout + digestion (R1's target) is 17–33 %. The per-block
`vector::resize` of the new scratch is visible (~1–2 %); size it once per
build if it stays.

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
shells share exponents across L. Merging them cuts primitive passes by:

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
| B2.1 | `66669cd5` | family-quartet units in the dense traversal (`family_units`, off) | — (a cost; see below) | exactly-once counting gate; 2 mutations |
| W1 | `ffc44caf` | contiguous-range thread schedule on a triangular loop | 1.10–1.97× at four threads | `planck-fock-accumulate` at 1/4/8 threads |
| core v1.2.0 | `planck-boys-core` `aa44c44` | single-x evaluation overhead; adds a batched entry | ~1.01–1.02× on HEAD's builds | bitwise against v1.1.0 through the C ABI |

Suites after W1 and after R2's callers: smoke 43/45, extended 135/139 (the
known rccsdt routing cases). Details worth keeping:

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
  (op count asserted), relative to max|G|. A mutation re-run (2026-10-23)
  found the gate NaN-blind (`std::max(worst, NaN)` returns `worst`) and unable
  to tell whether dense ran when block counts tie; both closed
  (`max_output_diff` returns +inf on non-finite; the counting kernel records
  blocks whose bra pair sorts after the ket pair, which only the
  per-component loop prepares). 7 of 9 mutations caught; the 2 survivors drop
  the sign test, which only saves work.
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

## Measured, not landed

### D1c — the electron-transfer recurrence (declined)

Head-Gordon and Pople's ETR at m = 0 after a one-centre VRR out to |e| ≤ M:

```math
[e|f{+}1_j] = \bigl(\tfrac{\zeta}{\eta}PA_j + QC_j\bigr)[e|f]
+ \tfrac{f_j}{2\eta}[e|f{-}1_j] + \tfrac{e_j}{2\eta}[e{-}1_j|f]
- \tfrac{\zeta}{\eta}[e{+}1_j|f]
```

- **Precision.** A fixed bra→ket direction was off by 2.8e-9 on cc-pVTZ;
  transferring per primitive quartet toward the larger exponent sum gave
  2.4e-13 (27× the VRR).
- **Speed.** 1.04–1.06× at VRR precision against a predicted 1.1–2.3×: the
  ket phase fell 2×, the bra VRR to |e| ≤ M grew 3.3–3.5×. A row layout was
  bitwise and time-neutral: the cost is per primitive quartet.

### 5b — a class-batched scheduler (declined after S0)

Group surviving blocks by class and evaluate classes in batches across blocks.
`--census`: 93.9–99.5 % of HGP-class primitive quartets were already in
within-block lanes (closed forms 99.5–99.8 %); pooling would buy at most
~1.15× on the lane work, and enumerating with a survival check cost 2–12 % of
the build. cc-pVTZ's K = 1 blocks are 23 % of HGP kernel time but ~6 % of its
primitive quartets: their cost is per block, which batching does not touch
while each block is digested component by component (K1 landed on that).

### Families: D2a, B0, B1, B2 (B2.1 landed off; B2.2 superseded by R3; B2.3 dropped)

A family is a set of shells on one atom whose exponent sets nest (cc general
contractions within L, Pople SP across L). A family quartet runs one primitive
pass over the union of its members' primitive pairs; members split by their
own total L (L ≤ 3: T, prefactor and Boys once per primitive quartet, raises
once per layout; L ≥ 4: one VRR pass at the high members' maximum L, then each
member's `finish()`).

- **Kernel gains (B0, against today's kernels on the same member blocks,
  values within 7.2e-16).** After lanes and K1: L ≤ 3 1.17–1.78× (cc-pVTZ
  1.78×, cc-pVDZ 1.73×, naphthalene 1.37×); L ≥ 4 a scalar family pass is
  0.92–1.03× on cc bases and 0.48–0.81× where little merges. Lanes and
  families compete for the same primitive work, so lanes took the HGP half.
- **The loop order costs (B1, B2.1).** Visiting blocks in family-quartet
  units is exactly-once and MPI-safe, but on the balanced (W1) schedule it is
  only a cost: +3.5 % / +3.6 % / +7.4 % at one thread on 8 H2O, 6-31G\* and
  naphthalene (cc bases ±1 %), −4.7 % to +8.8 % at four threads. Its earlier
  four-thread "gains" (−12 to −14 % on cc-pVDZ and 6-31G\*) were the load
  imbalance W1 fixed. A first B2.1 cut was +50 % / +84 % on 8 H2O until a
  unit-level Schwarz bound removed ~1.3 M fully screened units.
- **Fate.** B2.1 is in the tree behind `family_units` (off); delete it if
  neither R3 nor a later B2.2 needs the order. B2.2 (closed-form family
  kernel in family order) is superseded by R3, which shares the same work
  without the order. B2.3 (HGP families) is dropped. B1 remains a git stash
  ("B1: fused loop in family-quartet order") for its history only.

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

**A four-thread gain can be load imbalance in disguise.** B2.1's family units
measured 12–14 % faster at four threads; on a balanced schedule the gain
vanished and only the one-thread cost remained. Check the scaling of the
baseline before crediting a parallel gain to the change.

**A triangular loop needs a cyclic schedule.** Contiguous chunks of a
triangle hand the first thread 1 − (1 − 1/T)² of the work; at four threads
that capped the build at 2.29×, and one line (W1) recovered up to 1.97×.

**A share of time is not a share of work, and an out-of-tree share is not a
gain.** cc-pVTZ's K = 1 blocks are 23 % of HGP kernel time and ~6 % of its
primitive quartets (5b). Boys profiled at 9–32 % of the build, and a 1.3–2.3×
faster Boys evaluation plus batching realized 1.03–1.13×.

**An idea that measured slower may have been its first implementation.** A
contract-once hoist measured 1.14–1.62× slower; D1's block kernel is the same
idea at 1.5–13× faster, with L-major indexing.

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

**A gate that never runs is worse than none.** `planck-boys-vec` (the
`boys_vec`-vs-`boys` bitwise gate) was built but never registered in ctest
and never loaded the Boys library, so it segfaulted unnoticed since the
license gate. R2 found it by needing it.

**Code layout can cost more than the change.** Hoisting L3b's orbit buffer
slowed molecules without symmetry by 5 % through register allocation.

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
3–4× low predictions, 5b's premise, B1's Pople cost, and R2's shortfall.

**A copied fixture can hide the thing under test.** A copied `Calculator`
kept only the identity operation, and the first K1.4 "C2v" gate passed with
|G| = 1. Assert the fixture's defining property.

## Open, in order

| step | what | targets | bitwise? |
|---|---|---|---|
| R1 | closed-form readout as a flat, norm-folded array filled in `prepare` | readout 2–12 %, part of digestion | yes |
| R2 | land the batched Boys callers; re-profile; run the v1.1.0 fallback once | Boys 9–32 % (1.01–1.11× measured) | no (rounding, ≤ 1.1e-16) |
| R3 | family sharing of T / Boys / prefactors across a bra unit's ket blocks, no family order | closed forms + Boys on cc / naphthalene | no |
| partition | fixed accumulator partition, cyclic bins | determinism across thread counts | no |
| C5 | screened kernels in both block kernels (then dense) | range-separated hybrids | screened only |
| P2 | quartet-level primitive screening (policy) | ~20 % of primitive quartets | no (~1e-12 to 1e-10) |
| L > 12, F3, D3, K2, C4b | see below | — | — |

**R1 — the closed-form readout as a flat array (in-tree, not started).**
`ClosedFormBlockKernel::value` re-derives each component's raise slot on every
call (a loop over the four component views reading their Cartesian exponents,
then the slot index, then four norm-ratio loads), out of line through a
weak-symbol stub: 2–12 % of the build, plus part of the 9–20 % digestion loop
that calls it. `prepare` fills a flat per-component array, norm ratios folded
in, indexed `((ca·n_B + cb)·n_C + cc)·n_D + cd` as HGP's is; `value()`
becomes that load, and the dense loop can read the array directly. Steps:
fill the array in `prepare` (same multiplications done earlier; gate with the
single-file old-vs-new comparison, not fingerprints), read it from the dense
loop if the call still shows, then the suites and a re-profile. Prediction:
1.03–1.12×, largest on 8 H2O / 6-31G. (A0 measured the readout at ≤ 3.6 %
before K1 and ruled this out; K1 made the rest of the loop cheaper, and the
readout is now 2–12 %.)

**R2 — the batched Boys entry.** Every Boys call evaluated one x through the
out-of-tree core: the closed forms once per primitive quartet (orders 0–3),
the HGP block kernel once per lane. Two halves, the core's internals scoped
outside this repo:

1. **Core v1.2.0 (released).** A faster single-x path and a batched export,
   `planck_boys_batch(x, count, nmax, out)` with
   `out[i·(nmax+1) + n] = F_n(x[i])`, bitwise equal to the vector entry for
   every x. Verified through the C ABI against v1.1.0 (`tools/verify_abi.cpp`
   in the core repo): 90 M single-x values over the table range, its row
   edges, random x and the asymptotic range; the vector entry including its
   NaN tail; the batched entry at five widths; Apple clang and g++-15. Per x:
   vector entry 1.03–2.27×, batched entry 1.53–2.26×. Through Planck's
   loader, the released blob's batched, vector and scalar entries agree byte
   for byte on 3.3 M points per order.
2. **Planck callers (written, not committed).** The loader resolves
   `planck_boys_batch` optionally; `Lookup::boys_batch` uses it, or loops
   `boys_vec` for a v1.1.0 library (license), with the same values. The
   closed forms compute T for every primitive quartet of the block, make one
   batched call, then accumulate (scratch as a kernel member); the HGP
   kernel makes one batched call per lane group. `planck-boys-vec` now loads
   the library, runs under ctest and pins `boys_batch` to `boys_vec`.
   Measured, HEAD → R2 on the same v1.2.0 library, best of three
   interleaved: 1.01–1.11× at one thread, 1.03–1.11× at four (largest on
   naphthalene and 8 H2O, smallest on cc-pVTZ, whose work is HGP).
3. **Not bitwise, by the compiler.** Compiling HEAD's and the new
   `low_l_shell_eri` into one test on the smoke fixtures' 67 032 real quartet
   calls: under the production flags, T and every Boys value are
   byte-identical in every call, and 583 outputs differ, all in the (ss|ss)
   branch `out[0] += base·F[0]`, by at most 1.1e-16 (g++'s
   `-ffp-contract=fast` fuses that multiply-add differently in the two loop
   shapes). With `-ffp-contract=off` the rounding moves rather than
   vanishes (43 261 outputs differ, ≤ 2.2e-16). Gates: tolerance (closed
   forms within 1.3e-15 of HGP, `planck-hgp-engine-smoke`,
   `planck-compute-2e`, `planck-fock-accumulate`), smoke 43/45, extended
   135/139.
4. **Open:** commit; run the v1.1.0 fallback once (swap in a v1.1.0 license,
   run `planck-boys-vec`); size the scratch once per build (its per-block
   `resize` is ~1–2 %). Re-profiled with R2 (see "Where the time goes"):
   Boys is 13–24 % (7 % on cc-pVTZ), nearly all evaluation, so further
   Boys gains need either fewer evaluations (R3's sharing, P2) or a cheaper
   evaluation in the core, not more batching.

**R3 — family sharing within today's bra units (not started).** For a fixed
bra block, the ket blocks of one shared-exponent family share their primitive
pairs, hence their T values, Boys values and prefactors. Group each bra
unit's kets by family and evaluate that shared work once per family, inside
K1's traversal: no family order, so no Pople cost, but only the ket side is
shared. A family's T values form one batch for R2. First step: a bra-fixed
mode in the `--family` probe against W1 + R2 production; land only at
≥ 1.10× on cc-pVDZ or naphthalene, else record and stop. B2.2 in family order
returns only if R3 falls short and the order can be repaid.

**The fixed accumulator partition.** Partials are keyed by thread index, so
a run is reproducible at a given thread count but not across counts
(CO2 / 6-31G\* direct differs between 1 and 4 threads in the DIIS residual
digits and the last kcal/mol digit; all 10 Hartree digits identical). Key the
partials by unit index mod 64, each bin's units in increasing order, bins
merged in fixed order; a thread owns whole bins, keeping W1's balance (the
FCI sigma-build rule). Memory 64 × nb² (64 × 2 MB at nb = 500). Gate: G
bitwise identical at 1/2/4/8 threads.

**C5 — screened kernels.** Both block kernels decline screened kernels, so
range-separated hybrids take per-component HGP. They need `hgp_vrr`'s
scaling: T × `boys_scale`, prefactor × `prefactor_scale`, W−P/W−Q ×
`screen.rho / rho`, and the straddling coupling × `boys_scale` (the
`inv_2_delta` gotcha). Gate: a screened sweep against OS on a mixed bra/ket
fixture for LongRange and ShortRange, plus the `water_*_engine_os_vs_hgp`
comparators.

**P2 — quartet-level primitive screening (a policy decision).** After P1,
~20 % of primitive quartets on water clusters have a prefactor bound between
1e-24 and 1e-12. Skipping below ε, Schwarz held exact (before D1):

| fixture | ε = 1e-14 | ε = 1e-12 | max\|ΔG\| at 1e-14 / 1e-12 |
|---|---|---|---|
| 8 H2O / 6-31G | 1.13× | 1.37× | 4.0e-12 / 3.3e-10 |
| 4 H2O / 6-31G\* | 1.07× | 1.18× | 4.9e-12 / 3.7e-10 |
| naphthalene D2h | 1.15× | 1.22× | 4.8e-12 / 4.8e-10 |

Proposed ε = 1e-4 · `tol_eri` through `BlockEriRequest::prim_eps`;
`schwarz_table_*` must pass ε = 0. Re-measure on today's kernels before
deciding.

**L > 12.** Blocks above total L = 12 (g shells and up) take per-component
HGP. Extend when a fixture needs it; the VRR table is 3.7 MB for (gg|gg).

**F3.** After F1 the full-group skeleton runs only where it is persisted, so
this shortens a one-time build (0.017–0.054 s on small molecules, 1.3–3.4 s
near n_b = 100). Worth it only if `threshold` is raised routinely.

**D3 (independent).** RI-JK recomputes its density-independent pair factors
twice per iteration, ~61 % of the RI Fock build (`docs/ERI_FOCK_BLAS_SCOPE.md`).

**K2 (conditional).** Batches of same-class ket blocks for a fixed bra unit:
lanes across kets, bra-side work hoisted, the HRR as array work, digestion as
small matrix products. Only if a census after R1–R3 shows per-block fixed
costs (setup, HRR at 10–16 % of the HGP kernel) still worth the largest
restructure here.

**C4b (deprioritised).** A four-raise closed form matched HGP to 1.3e-15 on
every total-L = 4 layout and measured 10–16× over per-component HGP (C4a),
but its competitor is now the block kernel, whose L = 4 work is 4–9 % of sp
builds.

## Ruled out

- **Forcing `accumulate_term` / `distinct_eri_orbit` inline**: stubs < 1 %.
- **Avoiding `boys(0, T)` in the (ss|ss) branch**: the cost was the Boys
  evaluation itself (now addressed in the core, R2).
- **A generator for an L = 4 closed form**: the hand-written level was correct
  first time (C4a).
- **The ETR (D1c), a unit-stride row layout for the VRR, and a class-batched
  scheduler on per-component digestion (5b).**
- **The family loop order as a lever (B1, B2.1)** and **HGP families against
  laned production (B2.3)**: see Families.
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
