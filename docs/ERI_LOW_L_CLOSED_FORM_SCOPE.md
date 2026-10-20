# Shell-Quartet ERI Kernels for the Fused Fock Build — Scope

**Question:** How much faster can the memory-direct (fused) Fock build get by
evaluating whole shell quartets at once, in closed form at low angular
momentum and with a block kernel above it, and what else in the build
stands in the way?

**Status: in flight.** Landed: C0, C1, C2–C3.5, H1, L1, L2, L3a–c, P1, S1,
D1, F4. Measured: A0, D0, C4a, a post-D1 profile, and the full point-group
path. Open, in order: the full point-group follow-ups F1–F3 (its default
above nb = 100 is slower than no symmetry), block-kernel scratch out of
thread-local storage, D1c (the electron-transfer recurrence), a per-block
skeleton test, C5, D2a, D2b, P2, total L > 12; D3 is an independent track
and C4b is deprioritised. When
those land or are abandoned, rewrite this as an answer.

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
† The product of the per-step ratios; that fixture was never timed before
C1. ‡ With `sym_ops`, on a group-averaged density (the S1 skeleton path
needs a symmetric one). § Against the D0 production build (after P1, before
S1 and D1). The last column is what Auto buys over an explicit engine,
which gets only the engine-independent steps. These are Fock-build
numbers, not SCF wall time. The ~200× libcint gap in
`docs/ERI_PERFORMANCE_SCOPE.md` predates this work and has not been
re-measured.

Per step, each a ratio over the build with all previous steps (a row
multiplies):

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c | D1 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | 1.49× | **~11×** |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | 4.90× | **~11×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | 1.43× | — | — | 1.00× | 4.28× | **~8×** |

On the symmetric fixture: L3a 1.70×, L3b 1.23×, L3c+P1 1.16×, S1 1.33×,
D1 ~1.5× (1.454 → 0.855 → 0.702 → 0.597 → 0.454 → 0.295 s, the last two on
a symmetric density).

**In short:**
- **Evaluate shell quartets, not components.** Every engine returned one
  Cartesian component per call and reran the whole primitive loop for each,
  27–78× per shell quartet wherever HGP ran. Removing that repetition is
  where nearly all of the speedup comes from: closed forms at total L ≤ 3,
  an HGP block kernel at 4–12.
- **Layout decides whether that wins.** The block kernel is the same idea as
  an earlier contract-once hoist that measured 1.14–1.62× *slower*. The
  difference is indexing Cartesian functions by total L (a triangle)
  instead of a per-axis cube.
- **Once the kernels were fast, the work around them became the cost:** the
  loop, symmetry bookkeeping, primitive quartets that contribute nothing,
  and now the block kernel's own ket recurrence.

## How the build works now

`Integrals::fused_fock_build` (`src/integrals/fused_fock.h`) visits shell
blocks (bra shell pair, ket shell pair) and, inside each, the component
quartets. In order:

1. **Block filters.** Block-level Schwarz (`qmax[bra]·qmax[ket]`); ket
   blocks whose first shell sorts below the bra's are skipped (L1); with
   symmetry, blocks an operation sends below the bra's first shell are
   skipped (L3c).
2. **Component filters.** Canonical order, component Schwarz, and with
   symmetry the orbit test: replication mode keeps the orbit
   representative; skeleton mode (S1) also takes its orbit size and drops
   symmetry-forbidden quartets.
3. **Values.** At the first surviving component of a block, the block
   kernel is asked to `prepare` the whole shell quartet; every later
   component reads `value(ca, cb, cc, cd)`. A declined block uses the
   engine's per-component `eri_elem`.
4. **Accumulation.** Each value goes into G over its distinct 8-fold
   permutations (`distinct_eri_orbit`, L2); in replication mode, once per
   orbit element; in skeleton mode, once, weighted by the orbit size, with
   G symmetrized over the operations at the end.

Block kernels (`src/integrals/block_eri.h`): `NoBlockKernel` for explicit
engines (pure per-component); for Auto, `ClosedFormBlockKernel{3}`
(`low_l_eri.h`) chained with `HgpBlockKernel{4, 12}` (`hgp_block_eri.h`).
Both are Coulomb only; screened kernels fall through to `eri_elem`.

## How it is measured

`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a
ctest) drives `fused_fock_build` on 1, 4 and 8 waters (6-31G, 6-31G\*), one
and four waters in cc-pVDZ, one water in cc-pVTZ, and naphthalene / 6-31G
(axis-aligned, so `update_integral_symmetry` finds all 8 D2h operations;
its density is group-averaged so the skeleton path runs). Per fixture it
prints the HGP time by (L_AB, L_CD) class, the closed-form builds at
L ≤ 1, 2, 3 with max|ΔG| against HGP and a G fingerprint each, the HGP
block kernel one total-L class at a time, the symmetric production build
with and without `sym_ops`, and an FNV-1a **fingerprint of G** from the
flag-off build. `--profile N [nwater basis]` repeats one production build
(Auto's block kernels) for `sample`; nwater 0 is naphthalene.

Rules learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling; one binary
  alternated 1.0 / 1.22 s). Compare interleaved binaries or one binary with
  a switch. A single run is weak evidence.
- **A fingerprint, not a timestamp, says which code ran**, and a failed
  build silently reruns the old binary. Check the build's exit status and
  the new symbols (`nm`) before trusting an equality.
- **Fingerprints depend on the fixture list**: the test density comes from
  `Eigen::MatrixXd::Random`, whose state carries across fixtures, so new
  fixtures go at the end.
- **Attribute before predicting.** Inlined helpers hide in their caller's
  self time. Temporary builds mark helpers `noinline`, route kernel phases
  through named `noinline` wrappers (anonymous lambdas get numbered names
  that cannot be told apart), and add counters. They are never committed.

## Where the time goes now

Self time of the production build after D1 (2026-10-20), one thread, about
6 000 `sample` frames per fixture. The attribution build ran within ~3 % of
production (0.142 vs 0.137 s on 8 H2O / 6-31G).

| frame | 8 H2O / 6-31G | naphthalene D2h | 4 H2O / 6-31G\* | 4 H2O / cc-pVDZ | 1 H2O / cc-pVTZ |
|---|---|---|---|---|---|
| HGP block kernel: **VRR ket** | 4.0 % | 8.8 % | **24.4 %** | **20.9 %** | **40.7 %** |
| HGP block kernel: VRR bra | 0.4 % | 1.0 % | 3.0 % | 2.6 % | 2.7 % |
| HGP block kernel: contract | 0.3 % | 0.8 % | 3.1 % | 2.5 % | 3.6 % |
| HGP block kernel: HRR, setup, readout | 0.5 % | 0.8 % | 7.1 % | 4.0 % | 11.3 % |
| closed-form kernel | **30.0 %** | **28.4 %** | 15.8 % | **24.8 %** | 7.8 % |
| closed-form readout | 10.3 % | 4.6 % | 5.2 % | 4.2 % | 1.2 % |
| Boys (both kernels) | **26.4 %** | **22.5 %** | 13.1 % | **22.6 %** | 5.3 % |
| fused loop body | 13.9 % | 12.7 % | 8.2 % | 4.7 % | 4.8 % |
| G writes (`accumulate_term` + `distinct_eri_orbit`) | 13.6 % | 6.2 % | 13.9 % | 9.7 % | 14.7 % |
| `skeleton_orbit_weight` (S1) | — | **13.2 %** | — | — | — |
| emulated thread-local access | 0.5 % | 0.8 % | 3.6 % | 2.5 % | 4.7 % |

What it says:
- **d and f bases: the block kernel's ket VRR** (21–41 %). It recurses every
  bra function |e| ≤ L_A+L_B at every ket level, carrying the Boys-order
  index m throughout. D1c replaces it.
- **sp bases: the closed forms and Boys** (~60–67 %), as before D1. The block
  kernel's L = 4 work is only 4–9 % there, which is why C4b is
  deprioritised.
- **The symmetric build: `skeleton_orbit_weight`** (13 %), dominated by
  representatives, which must test every operation (step 3).
- **Emulated thread-local access** (2.5–4.7 % on d and f): g++ on macOS
  implements `thread_local` through `__emutls_get_address`, and the block
  kernel reaches four scratch vectors that way per `prepare` (step 1).
- **G writes** (6–15 %) are the Fock contraction itself; only symmetry
  shrinks them.

Before D1 the picture was different: per-component HGP was 32 %, 77 % and
26 % of 8 H2O / 6-31G, 4 H2O / 6-31G\* and naphthalene (A0), and by class
(D0) HGP at L = 4 was 12–45 % of the build and at L ≥ 5 up to 87 %
(cc-pVTZ).

**Shared exponents** (D0, counted on what the build executes after
Schwarz and P1, before D1). The GBS reader stores a general contraction as
separate shells over the same exponents (cc-pVDZ O 1s and 2s: two
9-primitive shells, same nine exponents), and Pople SP shells share
exponents across L. Merging such shells would cut primitive passes by:

| fixture | merge within L | merge across L |
|---|---|---|
| 8 H2O / 6-31G | 1.00× | **2.23×** |
| 4 H2O / 6-31G\* | 1.01× | **2.13×** |
| naphthalene / 6-31G | 1.00× | **3.73×** |
| 4 H2O / 6-311G | — | **2.18×** |
| 1 H2O / cc-pVDZ | **4.82×** | 4.82× |
| 4 H2O / cc-pVDZ | **5.11×** | 5.11× |
| 1 H2O / cc-pVTZ | **4.96×** | 4.96× |

A shell merges into the largest shell on the same atom whose exponent set
contains its own. cc sets share only within L, Pople sets only across L.
Primitive passes are not time: a merged pass does more arithmetic, so D2a
must measure the real gain.

## Symmetry today

On naphthalene / 6-31G (D2h, 8 operations), one Fock build, production
kernels, counted with temporary counters:

| | no symmetry | symmetry | ratio |
|---|---|---|---|
| canonical quartets visited | 13.37 M | 5.57 M | 2.4× |
| orbit representatives | — | 3.35 M | **4.0×** |
| … of which symmetry-forbidden | — | 1.43 M | |
| integral values computed | 13.37 M | 1.92 M | **7.0×** |
| closed-form block calls | 1.87 M | 0.48 M | 3.9× |
| HGP block-kernel calls | 22.9 k | 6.9 k | 3.3× |
| Fock accumulations (distinct permutation tuples) | 104.7 M | 15.0 M | **7.0×** |
| build time | 1.011 s | 0.295 s | **3.4×** |

- **A planar molecule caps the orbit at 4.** The mirror through the plane
  fixes every AO (it only flips the sign of out-of-plane p), so it is in
  every stabilizer, and orbits have at most 8/2 = 4 members. The plane's
  factor of 2 reappears as **zero integrals**: an operation mapping a
  quartet to itself with sign −1 forces it to vanish. Values computed fall
  7.0×, close to the group order.
- **Accumulations now fall as far as values.** Before S1 every nonzero value
  was replicated across its orbit and each element's 8-fold permutations, so
  writes fell only 1.75× (59.7 M). The skeleton accumulates each
  representative once.
- **What holds the build at 3.4×** is per-block and per-quartet work that
  falls less than 7×. Block-kernel calls fall 3.3–3.9×: a block is evaluated
  if any of its components survives, and being forbidden depends on each
  component's axis labels. And every canonical quartet that survives L3c
  still pays the orbit test (13 % of the symmetric build).
- **The ratio was 5.2× after S1 and is 3.4× after D1**, because D1 sped up the
  plain build (2.37 → 1.01 s) more than the symmetric one (0.454 → 0.295 s):
  the plain build evaluated 3.9× more blocks for the block kernel to speed up.

## The full point-group path does not get any of this

Symmetric HF with SAO blocking and direct SCF does not use the fused loop.
`full_symmetry_build_skeleton` (`src/scf/scf.cpp`) calls
`{os,hgp,rys}_symm.cpp` → `Symmetry::build_skeleton_eri`, a separate loop
that calls the engine per Cartesian component and fills a dense
n_b⁴ skeleton, contracted against the density every iteration. Of this
work only P1 (in the `ShellPair` constructor) reaches it, and H1 only with
`engine hgp`. The dispatch there checks for Rys and HGP and otherwise uses
OS, so **`engine auto` builds the skeleton with per-component OS**. The
skeleton is persisted across iterations only when n_b ≤ `threshold`
(default 100); above that, every iteration rebuilds it. Symmetric DFT and
HF with PCM or a linear point group use the fused signed-operation path
instead.

Measured 2026-10-20: RHF, `scf_mode direct`, one thread, Auto engine,
energies identical on both paths (temporary timer, never committed):

| molecule | n_b | full symmetry | fused, no symmetry |
|---|---|---|---|
| H2O / cc-pVDZ (C2v) | 25 | 0.047 s once + 0.0004 s/iter | 0.0075 s/iter |
| NH3 / 6-31G\* (C3v) | 21 | 0.017 s once + 0.0003 s/iter | 0.0024 s/iter |
| CH4 / cc-pVDZ (Td) | 35 | 0.054 s once + 0.0016 s/iter | 0.018 s/iter |
| benzene / 6-31G\* (D6h), default | 102 | **1.47 s/iter** (17.7 s) | 0.84 s/iter (11.8 s) |
| benzene, `threshold 200` | 102 | 1.34 s once + 0.126 s/iter (3.0 s) | |
| naphthalene / 6-31G (D2h), default | 106 | **3.50 s/iter** (66.6 s) | 1.00 s/iter (19.0 s) |
| naphthalene, `threshold 200` | 106 | 3.43 s once + 0.163 s/iter (7.3 s) | |

- **Below n_b = 100 it works well**: the skeleton build is small and the
  per-iteration contraction is 6–19× cheaper than a fused build.
- **Above n_b = 100 the default is slower than no symmetry** (1.75× on
  benzene, 3.5× on naphthalene): each iteration rebuilds an n_b⁴ tensor
  (~1 GB on naphthalene) with per-component OS.
- **Persisted, it is fast per iteration** (0.13–0.16 s, against ~0.30 s for
  the fused symmetric build on the same naphthalene Fock build, from the
  probe), but it holds the n_b⁴ skeleton, and its one-time build costs
  1.3–3.4 s, as much as 1.6–3.4 plain fused iterations.
- **For an Abelian group the fused path (S1) would do better than either**:
  naphthalene at ~0.3 s/iter is ~6 s for 19 iterations with no n_b⁴ memory.
  That is estimated from the probe, not measured in an HF run.
- **Side finding (fixed by F4):** with `guess sad`, NH3 (C3v) and benzene
  (D6h) aborted with "initial density is not symmetry-adapted" (8.0e-5 and
  2.6e-3).

## What landed

| step | commit | what it removed or added | gain | gate |
|---|---|---|---|---|
| C0 | Boys v1.1.0 | 17.7 % of the build in libm calls inside one branch of the out-of-tree Boys core | 1.37× (8 H2O), 1.13× (4 H2O\*) | two regression iteration counts re-pinned at the noise floor (`water_rhf_hgp_sto3g` 11 → 10, `ethylene_rhf_ccpvdz_auto` 16 → 15) |
| C1 | `c40a016c` | a `ShellPair` rebuilt per component quartet; now looked up in an `nb × nb` table of the caller's pairs | 1.16–1.26× | bitwise |
| C2–C3.5 | `3c028544`, `4bbb2724`, `f0f7ceae` | closed-form shell kernels through total L ≤ 3 | 1.22–1.28×, 1.61×, 1.87× on 8 H2O | per-quartet sweep against HGP, 6.1e-16 |
| H1 | `46dadc6a` | MMAX+1 scalar `boys` calls → one `boys_vec` (HGP and RI); m = 0 slice accumulated straight from the VRR buffer; VRR zero-fill dropped | 1.10–1.12× × 1.04–1.06× | bitwise |
| L2 | `659883ab` | the quadratic search in `distinct_eri_orbit` (17.3 % of the 8 H2O build) | 1.20× / 1.09× / 1.11× | exhaustive oracle over 1296 tuples, order included |
| L1 | `de72ca2c` | ket blocks whose first shell sorts below the bra's (49 % of blocks surviving Schwarz) | 1.06× | bitwise |
| L3a | `f88991fe` | building the full orbit for non-representatives (~7 in 8 under D2h) | 1.70× (D2h) | exhaustive comparison on an AO-permuting synthetic group |
| L3b | `a1f99d7a` | per-quartet orbit allocation; loop body compiled per symmetry mode | 1.23× (D2h) | bitwise |
| P1 | `15c532e4` | primitive pairs with \|c_a c_b N_a N_b (π/ζ)^{3/2} K_ab\| < 1e-24 (19 % of executed quartets on 8 H2O) | 1.15× / 1.10× / 1.04× | G byte-identical on all fixtures |
| L3c | `15c532e4` | whole blocks an operation sends below the bra's first shell (60 % of D2h blocks) | 1.07× (D2h) | `check_real_symmetry_ops` |
| S1 | `7ec71009` | per-orbit-element replication of every G write | 1.33× (D2h) | `check_skeleton_symmetry` |
| fix | `fd51d29d` | H1's Boys buffer overflow on (ff\|ff) in HGP's hoisted block path | — | the f/g/h spherical gates |
| D1 | — | the per-component primitive loop at total L 4–12 | 1.5× sp, 4.3–4.9× d, 12.9× f | 20 fingerprints; `check_hgp_block_kernel` |
| F4 | — | `guess sad` aborting full point-group SCF on C3v/D6h | — (correctness) | `nh3_c3v_fullsym_631gd_sad_direct` |

Suites: smoke 43/45 and extended 132/136 after D1; the failures are the
known rccsdt routing cases.

### F4 — a symmetric SAD guess for the full point-group path

The raw SAD density is group-invariant
(shell-wise spherical averages); the break was the idempotent projection,
which keeps the top n_occ natural orbitals and splits a degenerate set
(e, t) at that cutoff. When full symmetry is active the projected guess
(RHF, and each spin of UHF/ROHF) is averaged over the group with the
contravariant law, `P ← (1/|G|) Σ_R O_R P O_Rᵀ` (`symmetrize_density`,
`src/scf/scf.cpp`). It preserves Tr(PS) and gives up idempotence, which
a guess does not need. NH3 / 6-31G\* (Cartesian and spherical) and
benzene / 6-31G\* now converge from SAD to the hcore energy (benzene to
1e-11 Eh, 13 against 12 iterations). Gated by
`nh3_c3v_fullsym_631gd_sad_direct`, which fails with the averaging
removed.

### The closed-form kernels (C2–C3.5)

`low_l_shell_eri` (`src/integrals/low_l_eri.h`) treats a quartet's angular
momentum as up to three *raises* (center, axis), applied in center order.
Per primitive quartet, let $\theta_m = c\,K_{AB}K_{CD}\,2\sqrt{\rho/\pi}\,F_m(T)$
from one `boys_vec` call, $d$ be $P-A$, $P-B$, $Q-C$ or $Q-D$ for the raised
center, and $w$ be $W-P$ (bra) or $W-Q$ (ket). Then

$$
\begin{aligned}
A_i(m) &= d_{1,i}\,\theta_m + w_{1,i}\,\theta_{m+1} \\
B_{ij}(m) &= d_{2,j}\,A_i(m) + w_{2,j}\,A_i(m+1) + [i=j]\,\kappa_{12}(\theta;\,m) \\
C_{ijk} &= d_{3,k}\,B_{ij}(0) + w_{3,k}\,B_{ij}(1) + [k=i]\,\kappa_{13}(A^{(2)}_j) + [k=j]\,\kappa_{23}(A_i)
\end{aligned}
$$

$A^{(2)}_j$ is the second raise alone. $\kappa_{yx}(X;\,m)$ couples a new
raise $x$ to an earlier raise $y$ it lowers: both on the bra,
$\bigl(X(m) - \tfrac{\rho}{\zeta}X(m+1)\bigr)/2\zeta$; both on the ket, the
same with $\eta$; one on each side, $X(m+1)/2(\zeta+\eta)$. At most 27
outputs, all in registers.

- **Feeding and readout.** Each shell's first-component pair; outputs scaled
  by a per-AO `norm_ratio` ($d_{xx}$ carries $1/\sqrt3$, exactly 1 for s and
  p); slots read from the Cartesian exponents in raise order.
- **Gate.** `planck-hgp-engine-smoke` sweeps every total-L ≤ 3 component
  quartet of water in 6-31G, 6-31G\* and cc-pVDZ (7840 / 15736 / 43456)
  against HGP: max relative 6.1e-16, every reachable raise layout required
  (15 on sp, 31 on cc-pVDZ), mutation-verified on every coupling term, the
  W−P sign, the readout and the norm ratio. max|ΔG| ≤ 4.4e-15 against HGP;
  not bitwise, by design.

### S1 — skeleton accumulation on the signed-operation path

When every density the build contracts has the molecule's symmetry, each
non-forbidden representative is accumulated **once**, weighted by its orbit
size, into a skeleton $G^{\mathrm{skel}}$, and at the end (Dupuis–King, as a
gather because the operations are signed permutations)

$$
G_{\mu\nu} = \frac{1}{\lvert G\rvert}\sum_{R} s_R(\mu)\,s_R(\nu)\;G^{\mathrm{skel}}_{R(\mu)\,R(\nu)} .
$$

`skeleton_orbit_weight` decides representative, stabilizer size and
forbidden status in one pass over the operations (orbit size =
|ops|/|stab|, which needs the operations to form a group;
`update_integral_symmetry` builds the identity plus every valid axis flip,
which is closed). `density_has_symmetry` checks every non-null density per
call (absolute tolerance 1e-10); `symmetrize_over_ops` is the final gather.
All three are in `src/integrals/quartet_orbit.h`. Any non-symmetric density
(PCM, symmetry-broken UHF, random test densities) falls back to
replication, unchanged.

- **Result.** 0.604 → 0.454 s on the same symmetric density (1.33×,
  predicted 1.2–1.3×). max|ΔG| 7e-14 against the no-symmetry build;
  non-symmetric fingerprints bitwise unchanged.
- **Gate.** `check_skeleton_symmetry` in `planck-fock-accumulate`:
  group-averaged densities built from dense $O_R$ matrices (independent of
  the production gather), water C2v plus a synthetic order-2 flip group,
  every engine, RHF, UHF and J/K, ≤ 1.6e-14. The older random-density gates
  still pass against replication; a skeleton build would miss them by O(1),
  so they prove the fallback fires. Mutations: dropping the weight fails 13
  checks; skipping the symmetrization fails 7.
- **Who gains.** Symmetric DFT: its direct J/K builds always pass the
  operations, and SAO blocking keeps the density symmetric. **Direct HF on
  linear molecules gains nothing**: SAO blocking is not implemented for
  infinite groups, so CO2's iterates break the symmetry (deviation 0.6 at
  iteration 2, 3e-10 at convergence) and every build after the first falls
  back. No regression case runs a symmetric linear molecule through direct
  HF, and no symmetric open-shell case in the suite changes energy with
  symmetry off.

### D1 — the block interface and the HGP block kernel

**The interface** (`block_eri.h`): `prepare(BlockEriRequest)` returns false
for a class the kernel does not handle; `value(ca, cb, cc, cd)` reads one
component; `ChainBlockKernel` tries two kernels in order. The request
carries the first-component pairs, component views, L values, the per-AO
norm ratio, the kernel and ω (for C5) and `prim_eps = 0` (for P2). Moving
the closed forms behind it was **bitwise**: 20 fingerprints (flag-off and
closed forms at L ≤ 1, 2, 3 on five fixtures, including the symmetric one)
unchanged.

**The HGP block kernel** (`hgp_block_eri.h`). Per primitive quartet, the
Obara–Saika VRR in the W form builds $[e0|f0]^{(m)}$ for |e| ≤ L_A+L_B,
|f| ≤ L_C+L_D, with Cartesian functions indexed **L-major** (35 functions
for |e| ≤ 4, not a 125-cell per-axis cube). The m = 0 slice is contracted
over primitives for |e| ≥ L_A, |f| ≥ L_C only. HRR runs once per shell
quartet: on the bra for every ket function, then on the ket for each final
(a, b). Readout is a table lookup times the norm ratio.

- **Gate.** `check_hgp_block_kernel` in `planck-hgp-engine-smoke` sweeps
  every shell quartet of water in 6-31G, 6-31G\*, cc-pVDZ and cc-pVTZ
  against per-component HGP: 4.6 M component quartets in 256 classes up to
  (ff|ff), worst relative difference 8.9e-15, in a few seconds. Dropping
  any of the three VRR coupling terms, flipping either HRR sign, or dropping
  the norm ratio each fails; the HRR and norm mutations fail only cc-pVDZ,
  where an HRR with A ≠ B and L_B > 0 first occurs.
- **Dispatch.** The probe builds G with the block kernel on one total-L
  class at a time, on top of the closed forms. Every class from 4 to 12 won
  or tied, so Auto takes them all:

  | fixture | closed forms only | + block kernel, L 4–12 | speedup | max\|ΔG\| vs HGP |
  |---|---|---|---|---|
  | 8 H2O / 6-31G | 0.204 s | 0.137 s | 1.49× | 3.8e-15 |
  | naphthalene D2h | 0.478 s | 0.301 s | 1.59× | 3.1e-15 |
  | 4 H2O / 6-31G\* | 0.509 s | 0.104 s | 4.90× | 3.6e-15 |
  | 1 H2O / cc-pVDZ | 0.029 s | 0.007 s | 4.28× | 4.9e-15 |
  | 4 H2O / cc-pVDZ | 2.658 s | 0.572 s | 4.65× | 5.3e-15 |
  | 1 H2O / cc-pVTZ | 1.660 s | 0.128 s | 12.94× | 1.8e-14 |

  The per-class savings add up to the all-class figure. Above L = 12 and for
  screened kernels, blocks take per-component HGP. The VRR table is
  Ne·Nf·(M+1) doubles: 0.1 MB for (dd|dd), 0.7 MB for (ff|ff).

## Findings

**The floor was the primitive loop, not the recurrence.** The (ss|ss) closed
form has no recurrence left, yet per component it was only 1.66–1.75×
faster than HGP: every call reran the full primitive loop (an O 1s in
6-31G\* has 6 primitives, so 1296 primitive quartets, repeated 3× for
(ps|ss), 9× for (pp|ss), 81× for (pp|pp)). At shell granularity it ran once:
4.9×, and 10–16× at L = 4.

**An idea that measured slower may have been its first implementation.**
A contract-once-per-shell-quartet hoist measured 1.14–1.62× slower
(`docs/ERI_PERFORMANCE_SCOPE.md`), and D1's block kernel is the same idea at
1.5–13× faster. The hoist sized its VRR as a per-axis cube (cells no
component reaches, Boys orders no recurrence reads), snapshot-copied the
accumulator, and re-gathered a sub-box per component. L-major indexing
removed all three.

**A profile share is a lower bound on what removing the work is worth.**
The `ShellPair` rebuild was 8.4 % of samples and worth 1.16–1.26×, because
its allocator and cache traffic were charged elsewhere. The converse also
happened: L3c cut 59 % of the symmetric loop's iterations for 1.07×, because
the iterations it removed were the cheap ones.

**The cheap fix often sits next to the hot frame.** HGP made one Boys call
per order through the loader's indirect call (6.8 % self in the wrapper).
OS had already switched to `boys_vec`; HGP and RI never had.

**A buffer sized for one call site can be wrong for another.** H1's
`boys_vec` wrote into `F[4*MAX_L + 1]`, sized for one component (MMAX ≤ 24).
HGP's hoisted block path (`MaxBoxLayout`, used by the conventional tensor
build) passes per-*axis* maxima, so (ff|ff) reached MMAX = 36 and smashed
the stack: SIGBUS on any conventional HGP build with f shells or higher.
(dd|dd) peaks at exactly 24, so no d-basis gate saw it, and the f/g/h
spherical gates that did crash sat among known failures for two days. **A
red gate that is "already failing" still needs its cause checked.**

**Code layout can cost more than the change.** Hoisting L3b's orbit buffer
slowed molecules *without* symmetry by 5 %: the symmetry branches sat inside
the hot loop, so moving them shifted register allocation. Compiling the loop
body once per symmetry mode removed the coupling.

**A speedup can shrink a ratio.** Symmetry paid 5.2× after S1 and 3.4× after
D1, with both builds faster. A ratio between two paths moves whenever a
change helps one more than the other; quote ratios with their date.

**Screening must never reach the Schwarz table.** The first P2 experiment
gave max|ΔG| = 3e-6 at a 1e-14 threshold (4.6e-10 even at 1e-20), because
`schwarz_table_hgp` builds Q with the same kernel: skipping inside it
lowered Q below the true (ij|ij), so Schwarz stopped being an upper bound.
With Q built unscreened the error is a well-behaved ~400× the threshold.

**A cost-only test cannot be value-gated.** S1's forbidden-quartet skip
survives inversion because the final symmetrization cancels a forbidden
quartet exactly (the operation fixing it with sign −1 maps its contribution
to its own negative). It only saves work.

**Gates must be able to fail.** Every kernel gate was mutation-tested. Gates
caught what review missed: the flag-off fingerprint caught C3 sending
(ss|ss) to the closed form with the flag off; the layout-coverage
requirement caught water/6-31G\* being unable to produce (d,p)-ordered
layouts; sign-only symmetry fixtures make every quartet its own
representative, so L3a refuses them and L3c got a gate with real
shell-moving operations.

**Print a census's mapping before believing its total.** D0's first
general-contraction count required a representative to be *strictly
larger* than the shell it absorbs, so cc-pVDZ oxygen's equal-size 1s and 2s
never merged: 1.47–1.83× instead of 4.8–5.1×, which nearly retired D2.

**Write the prediction down first.** One table row misread the class data
(cc-pVDZ L = 3 as 50 % instead of 31 %); C4a's predictions were low by
3–4×. Both were caught only because the number was written down first.

## Open, in order

### Full point-group follow-ups (F1–F3), ahead of everything below

- **F1 — stop rebuilding the skeleton every iteration above the persist
  threshold.** For Abelian groups (D2h and its subgroups) route the build
  to the fused signed-operation path, which already runs S1's skeleton
  scheme with no n_b⁴ memory; for non-Abelian groups (C3v, Td, D6h) the
  dense-operation path needs a memory-direct skeleton build (accumulate
  into G instead of materializing n_b⁴), already listed in Open Work. Gate:
  energies against symmetry off on the five molecules above, and the
  full-symmetry regression ladder.
- **F2 — fix the Auto fall-through.** `full_symmetry_build_skeleton` sends
  `engine auto` to OS. Route it to HGP (fastest on this path in the
  teaching guide's §12) or to the per-bucket table. One line; values move
  at ~1e-15.
- **F3 — feed `build_skeleton_eri` from the block kernels.** The skeleton
  loop calls the engine per component; give it the `BlockEriRequest`
  interface so the closed forms and the HGP block kernel evaluate whole
  shell quartets. Cuts the one-time (persisted) build several-fold.

### The fused-loop steps

```
 cheap, bitwise-gated                  kernels                         shared exponents (value-changing)         last
 1 TLS scratch ──▶ 2 D1c ETR ──▶ 3 skeleton test ──▶ 4 C5 screened ──┬──▶ 5 D2a prototype ──▶ 6 D2b families ──┐
                                                                      │                                        │
                                                                      └────────────────────▶ 7 P2 ε screening ──┴──▶ 8 L>12
 independent:  D3 RI-JK pair factors (BLAS doc B0).   deprioritised:  C4b.
```

| step | needs | bitwise? |
|---|---|---|
| 1 TLS scratch | — | yes |
| 2 D1c | 1 (same code) | no, ~1e-15 vs the current block kernel; flag-off fingerprints unchanged |
| 3 skeleton test | — | yes |
| 4 C5 | 2: screened scaling added once to both block kernels | screened kernels only |
| 5 D2a | 2: a family kernel builds on the block kernel's recurrences | nothing lands |
| 6 D2b | D2a | no (summation order) |
| 7 P2 | after D2b, so its bound is written once per family and re-measured | no (~1e-12) |
| 8 L > 12 | a fixture that has it | — |
| D3 | — (different code) | per the BLAS doc |

Steps 1–4 keep the flag-off fingerprints; D2b and P2 each re-baseline
everything once and land alone, so a fingerprint change has one cause.

### 1 · Block-kernel scratch out of thread-local storage

`HgpBlockKernel::prepare` reaches four `thread_local` vectors per call;
g++'s emulated TLS on macOS makes each access a call to
`__emutls_get_address` (2.5–4.7 % of d and f builds). The kernel object is
already copied once per bra iteration, so the vectors can be members.
Bitwise; gate on the fingerprints and the block sweep.

### 2 · D1c — the electron-transfer recurrence in the HGP block kernel

The ket VRR is 21–41 % of d and f builds. Head-Gordon and Pople's
electron-transfer recurrence,

$$
[e|f{+}1_i] = -\frac{\beta (A-B)_i + \delta (C-D)_i}{\eta}\,[e|f]
 + \frac{f_i}{2\eta}[e|f{-}1_i] + \frac{e_i}{2\eta}[e{-}1_i|f]
 - \frac{\zeta}{\eta}[e{+}1_i|f],
$$

runs at m = 0 only, per primitive quartet, after a bra-only VRR built out
to |e| ≤ L_A+L_B+L_C+L_D. The table shrinks from Ne·Nf·(M+1) to
n(L_tot)·(M+1) plus an m-free (e, f) table, and the ket work loses its m
loop. Prototype first: time the ket phase both ways per class on the five
profile fixtures, and check the ETR's −ζ/η subtraction against
per-component HGP on the full sweep, including cc-pVTZ. Gate as D1, with
mutation tests on each ETR term.

### 3 · A per-block skeleton test

`skeleton_orbit_weight` is 13 % of the symmetric build. It already returns
at the first smaller image, so the ~40 % of calls that are not
representatives are cheap; the representatives must apply every operation
to count the stabilizer and look for a sign-flipping fixer. When every
operation maps each of the block's shells onto one shell (the L3c group
map), only the operations that fix the block's shell quartet can be in a
component's stabilizer, and under those each component maps to a fixed
component of the same block with a fixed sign. So the test can be set up
once per block. Bitwise; measure before landing.

### 4 · C5 — screened kernels, across both block kernels

Both block kernels decline screened kernels, so range-separated hybrids
still take per-component HGP everywhere. They need `hgp_vrr`'s screened
scaling: T × `boys_scale`, prefactor × `prefactor_scale`, W−P/W−Q ×
`screen.rho / rho`, and the straddling coupling × `boys_scale` (the
`inv_2_delta` gotcha). Added once, through the request's kernel and ω.
Gate: a screened per-quartet sweep against OS on a mixed bra/ket-AM fixture
for LongRange and ShortRange, plus the `water_*_engine_os_vs_hgp`
comparators.

### 5 · D2a — shared-exponent family prototype (nothing lands)

A **family** is the set of segmented shells on one atom whose exponent sets
are contained in one exponent set: Pople SP shells (6-31G oxygen's valence s
and p over the same three exponents, 16 segmented blocks per family
quartet, each running its own primitive loop) and general contractions
(cc-pVDZ O 1s/2s). A family quartet can run one primitive pass: compute the
coefficient-free intermediates once, then add each member block's
coefficient product times the intermediates of its own L. Families exist
only inside the fused loop; `Shell`, `ShellPair`, normalization, 1e
integrals, gradients and the spherical transform are untouched, and
per-member Schwarz, canonical filtering and accumulation stay as they are.
The prototype is a probe mode, validated against HGP and timed against the
sum of its member blocks on the current path (closed forms and the block
kernel). Guess: 1.3–1.6× on 6-31G builds, 1.5–2× on cc-pVDZ.

### 6 · D2b — shared-exponent families in the fused loop

Land what D2a admits, as block kernels: families built once per Fock build;
a family primitive pair is kept if *any* member needs it (P1's cutoff on the
largest member coefficient product). Gates: per-component validation
against HGP on every reachable family layout, mutation-tested on the
coefficient contraction and subset selection; G and energies within ~1e-12
of the current path. Lands alone.

### 7 · P2 — quartet-level primitive screening (a policy decision)

After P1, another ~20 % of primitive quartets on water clusters have a
prefactor bound between 1e-24 and 1e-12. Skipping them below ε, with the
Schwarz table held exact (measured before D1):

| fixture | ε = 1e-14 | ε = 1e-12 | max\|ΔG\| at 1e-14 / 1e-12 |
|---|---|---|---|
| 8 H2O / 6-31G | 0.194 → 0.171 s, 1.13× | 1.37× | 4.0e-12 / 3.3e-10 |
| 4 H2O / 6-31G\* | 0.504 → 0.471 s, 1.07× | 1.18× | 4.9e-12 / 3.7e-10 |
| naphthalene D2h | 0.598 → 0.520 s, 1.15× | 1.22× | 4.8e-12 / 4.8e-10 |

ΔG is ~300–500 × ε. Proposed rule: ε = 1e-4 · `tol_eri` (1e-14 at the
default). ε arrives through `BlockEriRequest::prim_eps` (both block kernels
decline a nonzero ε today); `schwarz_table_*` must pass ε = 0 (gate: Q built
with screening on equals the exact Q). Re-measure after D2b. Lands alone;
energies move at ~1e-12 and convergence-floor iteration counts may move.

### 8 · L > 12

The block kernel is dispatched up to total L = 12, the highest class any
fixture had (cc-pVTZ (ff|ff)). Beyond that (g and higher shells) blocks take
per-component HGP. Extend the range when a fixture needs it: the VRR table
grows as Ne·Nf·(M+1), 3.7 MB for (gg|gg), so measure memory as well as time
(D1c shrinks it).

### Independent · D3 — RI-JK pair factors (B0 in `docs/ERI_FOCK_BLAS_SCOPE.md`)

Exact four-centre integrals stay $K^4$ per shell quartet whatever the
kernels do; large cc-pVTZ/QZ runs are made fast in practice by density
fitting. Planck's RI-JK recomputes its density-independent pair factors
twice per iteration, ~61 % of the RI Fock build. Scoped by the BLAS doc;
touches none of the code above.

### Deprioritised · C4b — a four-raise closed form

C4a prototyped the closed form's fourth level (~70 lines, outputs in
registers):

$$
D_{ijkl} = d_{4,l}\,C_{ijk}(0) + w_{4,l}\,C_{ijk}(1)
 + [l=i]\,\kappa_{14}\bigl(V^{23}_{jk}\bigr)
 + [l=j]\,\kappa_{24}\bigl(V^{13}_{ik}\bigr)
 + [l=k]\,\kappa_{34}\bigl(B_{ij}\bigr),
$$

with $V^{13}_{ik}(m) = d_{3,k}A_i(m) + w_{3,k}A_i(m+1) + [i=k]\,\kappa_{13}(\theta;m)$
and $V^{23}$ built from $A^{(2)}$. It matched HGP to 1.3e-15 on every
total-L = 4 layout and measured 10–16× over **per-component** HGP:

| layout | class | 8 H2O / 6-31G | 4 H2O / 6-31G\* | 1 H2O / cc-pVDZ | 4 H2O / cc-pVDZ |
|---|---|---|---|---|---|
| ABCD | (pp\|pp) | 15.9× | 14.3× | 10.4× | 18.6× |
| AACC | (ds\|ds) | — | 7.0× | 6.1× | 8.2× |
| AABB / CCDD | (dd\|ss) / (ss\|dd) | — | 5.4× / 5.4× | 3.9× / 3.9× | 4.3× / 5.0× |
| **all L = 4** | | **15.9×** | **10.4×** | **10.5×** | **12.6×** |

Its competitor is now the HGP block kernel, whose L = 4 work is 4–9 % of sp
builds after D1, so even a free L = 4 kernel would save under 10 %. Revisit
only if D1c leaves L = 4 prominent.

## Ruled out

- **A per-block readout table for the closed forms.** The readout was
  ≤ 3.6 % (A0).
- **Forcing `accumulate_term` / `distinct_eri_orbit` inline.** Their dyld
  stubs cost < 1 %.
- **Avoiding the scalar `boys(0, T)` in the (ss|ss) branch.** Mostly the Boys
  evaluation itself; only the indirect call is avoidable.
- **A generator for an L = 4 closed form.** The hand-written level was
  correct first time (C4a).
- **Faster per-primitive work** (vectorizing the primitive loop, a cheaper
  Boys path) for now: bounded at ~1.5–2×, not specific to deep contraction,
  and easier once D1c and D2b fix the kernel shapes.
- **Loosening S1's density tolerance** so linear HF takes the skeleton path:
  it would build G from the group average of a non-symmetric iterate and
  change the SCF trajectory.

## Out of scope

- **A local accumulation buffer per block.** Changes summation order and
  touches the threading determinism rules.
- **Gradients.** Derivative ERIs are a separate kernel family (Open Work
  Phase D); P1 already covers them.
- **Spherical.** The direct spherical path back-projects to Cartesian and
  goes through the same fused loop.
- **SAO blocking for linear point groups** (in their D2h/C2v subgroup). It
  would let direct linear HF take S1's path, but no tested workflow needs it.
- **The fused direct loop is not thread-count invariant.** Its accumulators
  are per-thread partials keyed by thread index, so a run is reproducible at
  a given thread count but not across counts: CO2 / 6-31G\* direct differs
  between 1 and 4 threads in the DIIS residual digits and the last kcal/mol
  digit (all 10 Hartree digits identical). Same defect as the FCI sigma
  build: the *partition* of work into accumulators must be fixed, not just
  the summation order. It predates this work.
