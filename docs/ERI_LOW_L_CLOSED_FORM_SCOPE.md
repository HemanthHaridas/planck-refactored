# Shell-Quartet ERI Kernels for the Fused Fock Build — Scope

**Question:** How much faster can the memory-direct (fused) Fock build get by
evaluating whole shell quartets at once, in closed form at low angular
momentum and with a block kernel above it, and what else in the build
stands in the way?

**Status: in flight.** Landed: C0, C1, C2–C3.5, H1, L1, L2, L3a–c, P1, S1,
D1, F1, F2, F4, T1. Measured and declined: D1c (electron-transfer
recurrence), 5b (a class-batched scheduler). Measured, not landed: lanes
across primitive quartets (2b, prototype kept in the tree) and D2a
(shared-exponent families). Open, in order: D2b's B0 probe, 2b, D2b, the fixed
accumulator partition, a per-block skeleton test, C5, P2, total L > 12, F3;
D3 is independent and C4b deprioritised. When those land or are abandoned,
rewrite this as an answer.

## Answer so far

| fixture (one thread, `engine auto`) | Fock build now | before this work | speedup | vs explicit `engine hgp` now |
|---|---|---|---|---|
| 8 H2O / 6-31G (nb = 104) | **0.137 s** | 1.50 s | **~11×** | ~6.0× |
| 4 H2O / 6-31G\* (nb = 76) | **0.104 s** | ~1.12 s† | **~11×** | ~7.6× |
| 1 H2O / cc-pVDZ (nb = 25) | **0.007 s** | ~0.058 s† | **~8×** | ~8× |
| naphthalene / 6-31G, D2h (nb = 106) | **0.295 s**‡ | 1.454 s | **4.9×** | — |
| 4 H2O / cc-pVDZ | **0.572 s** | 2.573 s§ | **4.5×**§ | — |
| 1 H2O / cc-pVTZ | **0.128 s** | 1.631 s§ | **12.7×**§ | — |

`planck-eri-class-share`, Apple M3, `build-full`, `OMP_NUM_THREADS=1`, one
warmed run per build. "Before" is the build before C1, after the Boys
library fix (C0), which gave another 1.37× / 1.13× on its own.
† The product of the per-step ratios; never timed before C1. ‡ With
`sym_ops`, on a group-averaged density (S1 needs a symmetric one). § Against
the D0 build (after P1, before S1 and D1). The last column is what Auto buys
over an explicit engine, which gets only the engine-independent steps.
Fock-build numbers, not SCF wall time. The ~200× libcint gap in
`docs/ERI_PERFORMANCE_SCOPE.md` predates this work and has not been
re-measured.

Per step, each a ratio over the build with all previous steps:

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c | D1 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | 1.49× | **~11×** |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | 4.90× | **~11×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | 1.43× | — | — | 1.00× | 4.28× | **~8×** |

On the symmetric fixture: L3a 1.70×, L3b 1.23×, L3c+P1 1.16×, S1 1.33×,
D1 ~1.5× (1.454 → 0.855 → 0.702 → 0.597 → 0.454 → 0.295 s).

**In short:**
- **Evaluate shell quartets, not components.** Every engine returned one
  Cartesian component per call and reran the whole primitive loop for each,
  27–78× per shell quartet wherever HGP ran. Removing that repetition is
  nearly all of the speedup: closed forms at total L ≤ 3, an HGP block
  kernel at 4–12.
- **Layout decides whether that wins.** The block kernel is the same idea as
  an earlier contract-once hoist that measured 1.14–1.62× *slower*; the
  difference is indexing Cartesian functions by total L instead of a
  per-axis cube.
- **What is left is per-primitive-quartet overhead.** A primitive quartet
  writes only 180–300 ket elements into a 5–7.5 KB table, at ~6 cycles per
  element. A cheaper recurrence (D1c) or a better layout does not help;
  fewer primitive passes (families, ~1.7× on cc kernels) and several
  primitive quartets per pass (lanes, 1.13–1.32× of the build) do.

## How the build works now

`Integrals::fused_fock_build` (`src/integrals/fused_fock.h`): OpenMP
`schedule(static)` and the MPI stripe both run over bra group pairs. For
each bra:

1. **Block filters.** Ket group pairs start at `first_pair_of[bra.a]` (L1);
   block Schwarz (`qmax[bra]·qmax[ket]`); with symmetry, blocks an operation
   sends below the bra's first shell are skipped (L3c).
2. **Component filters.** Canonical order, component Schwarz, and with
   symmetry the orbit test: replication keeps the representative; skeleton
   mode (S1) also takes its orbit size and drops forbidden quartets.
3. **Values.** At the first surviving component the block kernel
   `prepare`s the whole shell quartet; later components read
   `value(ca, cb, cc, cd)`. A declined block uses per-component `eri_elem`.
4. **Accumulation.** Each value goes into G over its distinct 8-fold
   permutations (`distinct_eri_orbit`, L2), into per-thread partials keyed
   by `omp_get_thread_num()`; in replication mode once per orbit element,
   in skeleton mode once, weighted, with G symmetrized at the end.

Block kernels (`src/integrals/block_eri.h`): `NoBlockKernel` for explicit
engines; for Auto, `ClosedFormBlockKernel{3}` (`low_l_eri.h`) chained with
`HgpBlockKernel{4, 12}` (`hgp_block_eri.h`), one copy per thread. Both are
Coulomb only; screened kernels fall through to `eri_elem`.

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

Rules learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling; one binary
  alternated 1.0 / 1.22 s). Compare interleaved binaries or one binary with
  a switch. A baseline binary for an uncommitted change is built by
  compiling the affected objects from the old headers and relinking with
  the target's own `link.txt`.
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

## Where the time goes

Self time of the production build after D1, one thread, ~6 000 `sample`
frames per fixture:

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
operations), one Fock build:

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
- **What holds the build at 3.4×** is per-block and per-quartet work that
  falls less than 7×: a block is evaluated if any component survives, and
  every surviving canonical quartet pays the orbit test (13 %).
- **The ratio was 5.2× after S1 and is 3.4× after D1**: D1 sped up the
  plain build more, because it evaluates 3.9× more blocks.

**The full point-group path** (SAO-blocked HF, direct SCF). A separate loop
(`full_symmetry_build_skeleton` in `src/scf/scf.cpp` →
`{os,hgp,rys}_symm.cpp`) fills a dense n_b⁴ skeleton per Cartesian
component, contracted every iteration; Auto resolves to HGP there (F2). It
is persisted when n_b ≤ `threshold` (default 100); above that the SCF loops
take the fused path with the geometry's D2h-subgroup operations (F1).
RHF, `scf_mode direct`, one thread:

| molecule | n_b | full group, persisted | fused, no symmetry | before F1 (rebuilt every iteration) | F1 |
|---|---|---|---|---|---|
| H2O / cc-pVDZ (C2v) | 25 | 0.047 s + 0.0004 s/iter | 0.0075 s/iter | | |
| NH3 / 6-31G\* (C3v) | 21 | 0.017 s + 0.0003 s/iter | 0.0024 s/iter | | |
| CH4 / cc-pVDZ (Td) | 35 | 0.054 s + 0.0016 s/iter | 0.018 s/iter | | |
| benzene / 6-31G\* (D6h) | 102 | 3.0 s (`threshold 200`, 866 MB) | 13.7 s | 17.7 s | **4.08 s** |
| naphthalene / 6-31G (D2h) | 106 | 7.3 s (`threshold 200`) | 21.3 s | 66.6 s | **6.35 s** |

Below n_b = 100 the persisted skeleton is 6–19× cheaper per iteration than
a fused build. Benzene on the F1 path uses 8 of its 24 operations and is
within 1.4× of the persisted full-group path without its n_b⁴ memory.

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

Suites after D1: smoke 43/45, extended 132/136; the failures are the known
rccsdt routing cases. Details worth keeping:

- **Closed forms (C2–C3.5).** `low_l_shell_eri` treats a quartet as up to
  three *raises* (center, axis) in center order. Per primitive quartet,
  with $\theta_m = c\,K_{AB}K_{CD}\,2\sqrt{\rho/\pi}\,F_m(T)$, $d$ the
  raised center's $P-A$, $P-B$, $Q-C$ or $Q-D$, and $w$ its $W-P$ or $W-Q$:

  $$
  \begin{aligned}
  A_i(m) &= d_{1,i}\,\theta_m + w_{1,i}\,\theta_{m+1} \\
  B_{ij}(m) &= d_{2,j}\,A_i(m) + w_{2,j}\,A_i(m+1) + [i=j]\,\kappa_{12}(\theta;\,m) \\
  C_{ijk} &= d_{3,k}\,B_{ij}(0) + w_{3,k}\,B_{ij}(1) + [k=i]\,\kappa_{13}(A^{(2)}_j) + [k=j]\,\kappa_{23}(A_i)
  \end{aligned}
  $$

  $\kappa_{yx}$ couples a new raise to an earlier one it lowers: on the bra
  $\bigl(X(m) - \tfrac{\rho}{\zeta}X(m+1)\bigr)/2\zeta$, on the ket the same
  with $\eta$, across $X(m+1)/2(\zeta+\eta)$. At most 27 outputs, in
  registers; per-AO `norm_ratio` at readout. Gate: every total-L ≤ 3
  component quartet of water in 6-31G, 6-31G\*, cc-pVDZ against HGP, every
  reachable raise layout required, mutation-verified.
- **S1.** When every density has the molecule's symmetry, each
  non-forbidden representative is accumulated once, weighted by its orbit
  size, and $G_{\mu\nu} = \frac{1}{|G|}\sum_R s_R(\mu)s_R(\nu)\,
  G^{\mathrm{skel}}_{R(\mu)R(\nu)}$ at the end (`quartet_orbit.h`). Any
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

## Measured, not landed

### D1c — the electron-transfer recurrence (declined)

Head-Gordon and Pople's ETR at m = 0 after a one-centre VRR out to
|e| ≤ M,
$[e|f{+}1_j] = \bigl(\tfrac{\zeta}{\eta}PA_j + QC_j\bigr)[e|f]
+ \tfrac{f_j}{2\eta}[e|f{-}1_j] + \tfrac{e_j}{2\eta}[e{-}1_j|f]
- \tfrac{\zeta}{\eta}[e{+}1_j|f]$
(the textbook $-(\beta X_{AB}+\delta X_{CD})/\eta$ written with $P-A$, $Q-C$).

- **Precision.** A fixed bra→ket direction was off by 2.8e-9 on cc-pVTZ.
  Transferring per primitive quartet toward the larger exponent sum gave
  2.4e-13 (27× the VRR); G moved by up to 9.4e-13. VRR for ratios in
  (0.25, 1] restores the VRR's error exactly.
- **Speed.** 1.18× / 1.06× / 1.09× (cc-pVTZ / 4 H2O cc-pVDZ / 4 H2O 6-31G\*);
  1.06× / 1.04× / 1.04× at VRR precision. An operation-count model
  predicted 1.1–2.3× less recurrence work. The ket phase fell 2.0–2.1×, but
  the bra VRR to |e| ≤ M grew 3.3–3.5× (cc-pVTZ: ket 55.6 → 26.1 ms, bra
  4.0 → 14.0 ms, contraction 4.5 → 7.3 ms per build).
- **Row layout, also reverted:** V as [f][m][e] with every term unit-stride
  through precomputed contiguous runs was bitwise on all 28 fingerprints
  and time-neutral on four fixtures. That and the counters above showed
  the cost is per primitive quartet, not arithmetic.

### 2b — lanes across primitive quartets (prototype kept, not landed)

The HGP primitive loop over B primitive quartets at once, V as
`[e][f][m][lane]`, per-lane coefficients, innermost loops of compile-time
length B, padding lanes repeating lane 0 at weight 0, lanes summed before
accumulating. In the working tree behind `PLANCK_HGP_LANES` (off by
default). Sweep error ≤ 1.1e-14, not bitwise. Interleaved, best of two:

| fixture | HGP kernel, scalar → B = 4 | build, scalar → B = 4 | build, B = 4 with scalar for K < 4 |
|---|---|---|---|
| 1 H2O / cc-pVTZ | 77 → 52 ms (1.49×) | 0.128 → 0.103 s (1.25×) | **0.097 s (1.32×)** |
| 4 H2O / cc-pVDZ | 170 → 87 ms (1.96×) | 0.556 → 0.472 s (1.18×) | **0.471 s (1.19×)** |
| 4 H2O / 6-31G\* | 39 → 25 ms (1.56×) | 0.103 → 0.089 s (1.16×) | 0.091 s (1.13×) |
| 8 H2O / 6-31G | 6.8 → 3.1 ms (2.2×) | 0.145 → 0.142 s (1.02×) | — |

B = 2 sits between; B = 8 is slower than 4. The rule to land is lanes at
K ≥ 3 (K = 3 blocks are better padded than run scalar). Revisit after
D2b, since families change K.

### D2a — shared-exponent families

`--family`: per atom, a shell joins the largest shell whose exponent set
contains its own. A family quartet runs one primitive pass over the union of
its members' primitive pairs (keyed by the two exponents), and member
quartets split by **their own** total L:

- **L ≤ 3:** T, the coefficient-free prefactor and Boys once per primitive
  quartet; the raises once per distinct (L_A, L_B, L_C, L_D) layout (the
  outputs are linear in θ_m); each member adds w_k·w_l times them.
- **L ≥ 4:** one HGP VRR pass at the high members' maximum L (L-major
  indices and m = 0 values do not depend on L_A+L_B or M), then each
  member's own `finish()`.

Kernel time against the production kernels on the same member blocks (no
Schwarz or canonical filtering on either side), all values within 7.2e-16:

| fixture | L ≤ 3 (passes fewer) | L ≥ 4 (passes fewer) | kernel total |
|---|---|---|---|
| 1 H2O / cc-pVTZ | 1.69× (5.9×) | 1.70× (2.6×) | **1.70×** |
| 4 H2O / cc-pVDZ | 1.66× (5.9×) | 1.69× (2.4×) | **1.67×** |
| naphthalene / 6-31G | 1.16× (3.6×) | 1.05× (1.0×) | 1.15× |
| 4 H2O / 6-31G\* | 0.95× (2.2×) | 1.21× (1.3×) | 1.08× |
| 4 H2O / 6-31G | 0.93× (2.3×) | 1.49× (1.0×) | 0.97× |

- **cc bases: ~1.7× on kernels that are 73–80 % of the build**, an
  estimated 1.4–1.5× of the Fock build.
- **Pople is understated.** With nothing merged (`within` on Pople) the
  probe's closed-form family pass runs at 0.59–0.68× of production: it
  carries 1.5–1.7× overhead of its own (layout lookup, raises then a
  weighted add, `boys_vec` where production's (ss|ss) uses `boys(0, T)`).
- **Guess was 1.3–1.6× on 6-31G, 1.5–2× on cc-pVDZ.** cc landed in range.

### 5b — a class-batched scheduler (declined after S0)

The question: collect the blocks that pass the block filters, group them by
class, and evaluate classes in batches across blocks (the shape of ORCA's
SHARK). `--census` replays the fused loop's enumeration, weighted by
primitive quartets:

| fixture | HGP-class quartets already in within-block lanes | padding within-block | padding pooled per bra pair |
|---|---|---|---|
| 1 H2O / cc-pVTZ | 93.9 % | 13.4 % | 2.7 % |
| 4 H2O / cc-pVDZ | 96.5 % | 4.6 % | 0.8 % |
| 4 H2O / 6-31G\* | 95.8 % | 19.4 % | 2.5 % |
| 8 H2O / 6-31G | 98.8 % | 12.4 % | 0.5 % |
| naphthalene / 6-31G | 99.5 % | 9.4 % | 0.1 % |

Closed-form classes: 99.5–99.8 % already in within-block lanes. Enumerating
with a survival check cost 2–12 % of the build, and 1.5–14.6 % of the
primitive quartets in blocks passing the block filters belong to blocks with
no surviving component. Pooling across blocks would buy at most ~1.15× on
the lane work (≤ ~8 % of the cc-pVTZ build), and cc-pVTZ's K = 1 blocks
(23 % of kernel time, ~6 % of its primitive quartets) cost per block, which
pooling does not touch. The family win needs only member quartets evaluated
together; the fixed accumulator partition is independent. Both stay open on
their own.

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
4.8–5.1×.

**Write the prediction down first.** It caught a misread class share, C4a's
3–4× low predictions, and 5b's premise.

## Open, in order

| step | what | bitwise? |
|---|---|---|
| D2b B0 | production-quality closed-form family pass, in the probe | nothing lands |
| 2b | lanes at K ≥ 3 in the HGP block kernel, as a reusable laned core | no; lands alone |
| D2b B1–B3 | families in the fused loop | no (summation order); lands alone |
| partition | fixed accumulator partition | no; lands alone |
| skeleton test | per-block skeleton test | yes |
| C5 | screened kernels in both block kernels | screened kernels only |
| P2 | quartet-level primitive screening | no (~1e-12); lands alone |
| L > 12 | block kernel above total L = 12 | — |
| F3 | full point-group skeleton from the block kernels | yes, per engine |
| D3 | RI-JK pair factors (independent) | per the BLAS doc |

**D2b — families in the fused loop (scoped, 2026-10-21).** D2a's family
passes, with `fused_fock_build` (Auto only) visiting family pairs instead of
shell pairs, so a family quartet's members share one primitive pass.

- **Exactly once.** Families partition the shells, so every shell pair
  (canonical, a ≤ b by first AO) belongs to exactly one unordered family
  pair. The loop runs over family pairs P ≤ Q, bra members from P, ket
  members from Q:
  - P < Q: every member component quartet with j ≥ i and l ≥ k is produced
    by no other (P′, Q′), so the bra–ket test `(k,l) ≥ (i,j)` is dropped;
    the tuple is put in canonical order (bra and ket swapped when
    (k,l) < (i,j)) before the orbit test and accumulation.
  - P = Q: today's canonical filter, unchanged.

  The member blocks visited, and the component quartets accumulated, are
  the same set as today, in a different order.
- **Filters keep their granularity.** Block Schwarz and L3c apply per
  member block; a family quartet whose live members are all dead is
  skipped; component Schwarz and the orbit test stay per component. The
  family pass skips the accumulation (not the primitive work) of dead
  members.
- **Families** are built once per Fock build from the shells: per atom, a
  shell joins the largest shell whose exponent set contains its own. A
  family primitive pair is kept if any member keeps it (P1's cutoff on the
  largest member coefficient product).
- **Kernels.** A family block kernel: `prepare(family quartet)`, then
  `value(member_ab, member_cd, ca, cb, cc, cd)`. Members with total L ≤ 3
  take a closed-form family pass (T, prefactor and Boys once per primitive
  quartet, raises once per layout, weighted per member); members with
  L ≥ 4 an HGP VRR pass at their maximum L plus each member's `finish()`.
  Explicit engines keep the pure per-component path.
- **MPI and threads.** The stripe and `schedule(static)` move to family
  bra pairs: fewer, larger units. Per-thread partials as today (the fixed
  partition is a separate step).

Steps:

- **B0 — production-quality closed-form family pass, in the probe.** The
  D2a probe's pass carries 1.5–1.7× overhead at 1:1 on Pople. Template on
  the raise count, precompute layouts per family-quartet class, use
  `boys(0, T)` when every member is (ss|ss). Measure at 1:1 against
  production (target ≥ 1.0×) and with across-L merging on 6-31G,
  6-31G\*, naphthalene. Decides the family policy for Pople: across L,
  or within L only (no Pople merges).
- **B1 — the family-pair loop, members evaluated one by one** with today's
  block kernels. Isolates the reordering: gate on the count of accumulated
  component quartets (identical to today's, per fixture, with and without
  symmetry) and max|ΔG| ≤ 1e-13; also measures what coarser OpenMP units
  cost.
- **B2 — family passes in the loop.** Gates: a family sweep in
  `planck-hgp-engine-smoke` (every member value of every family quartet
  against per-component HGP on water 6-31G, 6-31G\*, cc-pVDZ, cc-pVTZ,
  mutation-tested on the coefficient contraction, the member-subset
  selection and the layout sharing); re-baselined fingerprints; max|ΔG| ≤
  1e-12 against today's build; `check_skeleton_symmetry`,
  `check_real_symmetry_ops`; MPI 1/2/4 ranks; smoke and extended suites.
- **B3 — measure the builds**, then land B1+B2 together, alone.

Predictions (written first): cc-pVDZ / cc-pVTZ builds 1.4–1.5× (kernels
1.7× at 73–80 % of the build); Pople decided by B0, 1.0–1.3×; B1 alone
within ±3 % of today.

**2b — lanes. Lands after B0 and before B1, alone.** It is measured and
ready, and not bitwise, so it re-baselines the fingerprints on its own. D2b
must then be measured against the laned kernel: D2a's 1.7× at L ≥ 4 was
against the scalar one, and lanes and families compete for the same HGP
primitive work. Land it with the laned primitive loop taking a list of
primitive quartets and per-quartet weights into accumulators, so that D2b's
HGP family pass reuses the same core (the union of a family quartet's
primitive quartets, several member accumulators) and the lane width is
written once. B0 is closed-form only and does not depend on it.

**The fixed accumulator partition.** The fused loop's partials are keyed by
thread index, so a run is reproducible at a given thread count but not
across counts (CO2 / 6-31G\* direct differs between 1 and 4 threads in the
DIIS residual digits and the last kcal/mol digit; all 10 Hartree digits
identical). Key the partials by a fixed function of the bra index instead
and merge in fixed order, the FCI sigma-build rule; memory n_bins × nb²
(64 × 2 MB at nb = 500). Gate: G bitwise identical at 1/2/4/8 threads.
Predates this work.

**A per-block skeleton test.** `skeleton_orbit_weight` is 13 % of the
symmetric build, dominated by representatives, which apply every operation.
When every operation maps the block's shells onto shells (the L3c group
map), only the operations fixing the block's shell quartet can stabilize a
component, and each maps a component to a fixed component with a fixed
sign: set the test up once per block. Measure before landing.

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
`schwarz_table_*` must pass ε = 0. Re-measure after D2b.

**L > 12.** Blocks above total L = 12 (g shells and up) take per-component
HGP. Extend when a fixture needs it; the VRR table is 3.7 MB for (gg|gg).

**F3.** After F1 the full-group skeleton runs only where it is persisted,
so this shortens a one-time build (0.017–0.054 s on the small regression
molecules, 1.3–3.4 s near n_b = 100). Worth it only if `threshold` is raised
routinely.

**D3 (independent).** RI-JK recomputes its density-independent pair factors
twice per iteration, ~61 % of the RI Fock build (B0 in
`docs/ERI_FOCK_BLAS_SCOPE.md`).

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
- **The ETR (D1c) and a unit-stride row layout for the VRR**: see above.
- **A class-batched scheduler (5b)**: see above.
- **Vectorizing the closed forms' primitive loop** while Boys, about half
  their cost, is an out-of-tree core that evaluates one T per call.
- **Loosening S1's density tolerance** so linear HF takes the skeleton
  path: it would change the SCF trajectory.

## Out of scope

- **A local accumulation buffer per block**: changes summation order under
  the threading rules.
- **Gradients**: derivative ERIs are a separate kernel family (Open Work
  Phase D); P1 covers them.
- **Spherical**: the direct spherical path back-projects to Cartesian and
  uses the same fused loop.
- **SAO blocking for linear point groups**: would let direct linear HF take
  S1's path; no tested workflow needs it.
