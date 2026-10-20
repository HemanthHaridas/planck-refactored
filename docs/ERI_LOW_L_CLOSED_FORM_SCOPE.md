# Closed-Form Low-Angular-Momentum ERI Kernels — Scope

**Question:** How much faster can the direct (memory-direct, fused) Fock
build get by evaluating low-angular-momentum shell quartets in closed form,
and what else in the build stands in the way?

**Status: in flight.** Landed: C0, C1, C2–C3.5, H1, L1, L2, L3a–c, P1, S1.
Measured: A0, D0, C4a. Open, in dependency order: D1, C4b, C5, D2a, D2b,
P2, then the L ≥ 5 decision, with D3 as an independent track. When those
land or are abandoned, rewrite this as an answer.

## Answer so far

| fixture (one thread, `engine auto`) | Fock build now | before this work | speedup | vs explicit `engine hgp` now |
|---|---|---|---|---|
| 8 H2O / 6-31G (nb = 104) | **0.187 s** | 1.50 s | **8.0×** | 4.4× |
| 4 H2O / 6-31G\* (nb = 76) | **0.497 s** | ~1.12 s† | **~2.3×** | 1.5× |
| 1 H2O / cc-pVDZ (nb = 25) | **0.028 s** | ~0.058 s† | **~2.1×** | 2.0× |
| naphthalene / 6-31G, D2h (nb = 106) | **0.454 s**‡ | 1.454 s | **3.2×** | ~2.4× (not re-measured after S1) |

Best of three runs of `planck-eri-class-share`, Apple M3, `build-full`,
`OMP_NUM_THREADS=1`. "Before" is the build before C1, after the Boys-library
fix (C0), which gave another 1.37× / 1.13× on its own. † The product of the
per-step ratios; that fixture was never timed before C1. ‡ After S1, on a
group-averaged density (the skeleton path needs a symmetric one). The last
column is what Auto buys over an explicit engine, which gets only the
engine-independent steps. Symmetry now pays **5.2×** on naphthalene (2.36 s
without `sym_ops`), against 1.9× before the L3 series. These are Fock-build
numbers, not SCF wall time. The ~200× libcint gap in
`docs/ERI_PERFORMANCE_SCOPE.md` predates this work and has not been
re-measured.

**In short:**
- **Closed forms help only at shell granularity.** Per component they are
  ~1.7× faster than HGP. Returning every component of a shell quartet from
  one primitive pass is ~5× faster at total L ≤ 3 and 10–16× at L = 4,
  because the repetition removed grows with the number of components.
- **Once the kernels were fast, the work around them became the cost:** the
  loop, symmetry bookkeeping, and primitive quartets that contribute
  nothing.
- **The basis decides the payoff.** Total L ≤ 3 is almost all of a pure sp
  build but only part of a d build. HGP is still ~¾ of a 6-31G\* build and
  ~99 % of a cc-pVTZ one. That is what the open steps target.

Per step, each a ratio over the build with all previous steps (a row
multiplies):

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c | total |
|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | **8.0×** |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | **~2.3×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | 1.43× | — | — | 1.00× | **~2.1×** |

On the symmetric fixture: L3a 1.70×, L3b 1.23×, L3c+P1 1.16×, S1 1.33×
(1.454 → 0.855 → 0.702 → 0.597 → 0.454 s, the last on a symmetric density).

## How it is measured

`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a
ctest) drives the production `Integrals::fused_fock_build` on 1, 4 and
8 waters (6-31G\*, 6-31G), one water in cc-pVDZ, and naphthalene / 6-31G
(axis-aligned, so `update_integral_symmetry` finds all 8 D2h sign flips; its
density is group-averaged so the skeleton path runs). Per fixture it prints
the HGP time by (L_AB, L_CD) class, the closed-form build at
`closed_form_max_l` 1–3 with max|ΔG| against HGP, the symmetric build with
and without `sym_ops`, and an FNV-1a **fingerprint of G** from the flag-off
build. `--profile N [nwater basis]` repeats one build for `sample`.

Rules learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling; one binary
  alternated 1.0 / 1.22 s). Compare best-of runs of interleaved binaries, or
  one binary with a switch. A single run is not evidence.
- **A fingerprint, not a timestamp, says which code ran.** Twice a binary
  was older than its source. Check `nm` or the fingerprint.
- **Fingerprints depend on the fixture list**: the test density comes from
  `Eigen::MatrixXd::Random`, whose state carries across fixtures.
- **Attribute before predicting.** Inlined helpers hide in their caller's
  self time; temporary `noinline` builds and counters split it first. They
  are never committed.

## Where the time goes now

**By class** (D0, after P1 and L3c): the share of the production build still
in HGP.

| fixture | production build | HGP at L = 4 | HGP at L ≥ 5 |
|---|---|---|---|
| 8 H2O / 6-31G | 0.222 s | 30 % | 0 |
| naphthalene D2h | 0.651 s | 34 % | 0 |
| 4 H2O / 6-31G\* | 0.504 s | 34 % | 50 % |
| 1 H2O / cc-pVDZ | 0.032 s | 39 % | 34 % |
| 4 H2O / cc-pVDZ | 2.573 s | 45 % | 36 % |
| 1 H2O / cc-pVTZ | 1.631 s | 12 % | **87 %** |

**By frame** (A0, before P1, L3c and S1): self time, the loop's helpers
split out by a `noinline` build.

| frame | 8 H2O / 6-31G | 4 H2O / 6-31G\* | naphthalene D2h |
|---|---|---|---|
| HGP (`hgp_vrr`, contract, HRR, scratch) | 31.7 % | **76.8 %** | 26.0 % |
| Boys (both kernels) | 25.2 % | 11.0 % | 17.0 % |
| closed-form kernel | 22.4 % | 3.9 % | 14.7 % |
| loop body (filters, traversal) | 8.5 % | 2.1 % | **19.7 %** |
| `is_orbit_representative` | — | — | 9.9 % |
| `accumulate_term` (the `G` writes) | 5.8 % | 2.2 % | 7.7 % |
| low-L slot/norm readout | 3.6 % | 0.6 % | 1.2 % |
| `distinct_eri_orbit` | 1.8 % | < 1 % | 1.7 % |

So: on sp bases, HGP at L = 4, (pp|pp), is 30–34 % (C4b). On cc-pVDZ, L = 4
and L ≥ 5 are about equal (C4b and D1). On cc-pVTZ almost everything is HGP
at L ≥ 5 (D1, then L ≥ 5 kernels). On every basis family 2–5× of the
primitive passes repeat work over a shared exponent set (D2).

**Primitive repetition** (D0, counted on what the build executes after
Schwarz and P1). Per-component calls rerun HGP's whole primitive loop for
each component; the GBS reader stores a general contraction as separate
shells over the same exponents (cc-pVDZ O 1s and 2s: two 9-primitive shells,
same nine exponents), and Pople SP shells share exponents across L.

| fixture | executed / per shell quartet | HGP blocks only | merge within L | merge across L |
|---|---|---|---|---|
| 8 H2O / 6-31G | 1.11× | 49× | 1.00× | **2.23×** |
| 4 H2O / 6-31G\* | 2.84× | 52× | 1.01× | **2.13×** |
| naphthalene / 6-31G | 1.23× | 27× | 1.00× | **3.73×** |
| 4 H2O / 6-311G | 1.10× | 46× | — | **2.18×** |
| 1 H2O / cc-pVDZ | 1.79× | 48× | **4.82×** | 4.82× |
| 4 H2O / cc-pVDZ | 1.99× | 56× | **5.11×** | 5.11× |
| 1 H2O / cc-pVTZ | **9.25×** | **78×** | **4.96×** | 4.96× |

A group merges into the largest shell on the same atom whose exponent set
contains its own. cc sets share only within L, Pople sets only across L.
Primitive passes are not time: a merged or block pass does more arithmetic,
so D1 and D2a must measure the real gain.

## What landed

| step | commit | what it removed or added | gain | gate |
|---|---|---|---|---|
| C0 | Boys v1.1.0 | 17.7 % of the build in libm calls inside one branch of the out-of-tree Boys core | 1.37× (8 H2O), 1.13× (4 H2O\*) | two regression iteration counts re-pinned at the noise floor (`water_rhf_hgp_sto3g` 11 → 10, `ethylene_rhf_ccpvdz_auto` 16 → 15) |
| C1 | `c40a016c` | a `ShellPair` rebuilt per component quartet; now looked up in an `nb × nb` table of the caller's pairs | 1.16–1.26× | bitwise |
| C2–C3.5 | `3c028544`, `4bbb2724`, `f0f7ceae` | closed-form shell kernels through total L ≤ 3 (below) | 1.22–1.28×, 1.61×, 1.87× on 8 H2O | per-quartet sweep against HGP, 6.1e-16 |
| H1 | `46dadc6a` | MMAX+1 scalar `boys` calls → one `boys_vec` (HGP and RI); m = 0 slice accumulated straight from the VRR buffer; VRR zero-fill dropped | 1.10–1.12× × 1.04–1.06× | bitwise |
| L2 | `659883ab` | the quadratic search in `distinct_eri_orbit` (17.3 % of the 8 H2O build) | 1.20× / 1.09× / 1.11× | exhaustive oracle over 1296 tuples, order included |
| L1 | `de72ca2c` | ket blocks whose `.a` sorts below the bra's (49 % of blocks surviving Schwarz) | 1.06× | bitwise |
| L3a | `f88991fe` | building the full orbit for non-representatives (~7 in 8 under D2h) | 1.70× (D2h) | exhaustive comparison on an AO-permuting synthetic group |
| L3b | `a1f99d7a` | per-quartet orbit allocation; loop body compiled per symmetry mode | 1.23× (D2h) | bitwise |
| P1 | `15c532e4` | primitive pairs with \|c_a c_b N_a N_b (π/ζ)^{3/2} K_ab\| < 1e-24 (19 % of executed quartets on 8 H2O) | 1.15× / 1.10× / 1.04× | G byte-identical on all fixtures |
| L3c | `15c532e4` | whole blocks an op sends below the bra's first group (60 % of D2h blocks) | 1.07× (D2h) | `check_real_symmetry_ops` |
| S1 | `7ec71009` | per-orbit-element replication of every `G` write (below) | 1.33× (D2h) | `check_skeleton_symmetry` |

Suites after each step: core green, smoke 43/45 (the two known rccsdt
routing failures).

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

- **Wiring.** Called once per block, on the first surviving component
  quartet, behind `closed_form_max_l`. Only Auto sets it (to 3), so explicit
  engines stay pure and the cross-engine comparators keep meaning. Coulomb
  only.
- **Feeding and readout.** Each shell's first-component pair; outputs scaled
  by a per-AO `norm_ratio` ($d_{xx}$ carries $1/\sqrt3$, exactly 1 for s and
  p); slots read from the Cartesian exponents in raise order.
- **Gate.** `planck-hgp-engine-smoke` sweeps every total-L ≤ 3 component
  quartet of water in 6-31G, 6-31G\* and cc-pVDZ (7840 / 15736 / 43456)
  against HGP: max relative 6.1e-16, every reachable raise layout required
  (15 on sp, 31 on cc-pVDZ), mutation-verified on every coupling term, the
  W−P sign, the readout and the norm ratio. max|ΔG| ≤ 4.4e-15 against HGP;
  not bitwise, by design.

### S1 — skeleton accumulation on the signed-op path

**Why symmetry paid only 4×** (naphthalene / 6-31G, D2h, per build,
temporary counters, before S1):

| | no symmetry | symmetry | ratio |
|---|---|---|---|
| canonical quartets surviving Schwarz | 13.37 M | 13.37 M | — |
| orbit representatives | — | 3.35 M | **4.0×** |
| … of which symmetry-forbidden (zero) | — | 1.43 M | |
| integral values computed | 13.37 M | 1.92 M | **7.0×** |
| closed-form kernel calls (per block) | 1.87 M | 0.48 M | 3.9× |
| G writes | 104.7 M | 59.7 M | **1.75×** |

- **A planar molecule caps the orbit at 4.** The mirror through the plane
  fixes every AO (it only flips the sign of out-of-plane p), so it is in
  every stabilizer and orbits have at most 8/2 = 4 members. That plane's
  factor of 2 shows up as **zero integrals** instead: an op mapping a
  quartet to itself with sign −1 forces it to vanish.
- **Kernel calls are per block**, and being forbidden is per component, so
  they fell only 3.9×.
- **G writes barely fell**: replication rewrote every nonzero integral
  across its orbit and each element's 8-fold permutations.

**The change.** When every density the build contracts has the molecule's
symmetry, each non-forbidden representative is accumulated **once**,
weighted by its orbit size, into a skeleton $G^{\mathrm{skel}}$, and at the
end (Dupuis–King, as a gather because the ops are signed permutations)

$$
G_{\mu\nu} = \frac{1}{\lvert G\rvert}\sum_{R} s_R(\mu)\,s_R(\nu)\;G^{\mathrm{skel}}_{R(\mu)\,R(\nu)} .
$$

`skeleton_orbit_weight` decides representative, stabilizer size and
forbidden status in one pass over the ops (orbit size = |ops|/|stab|, which
needs the ops to form a group; `update_integral_symmetry` builds identity
plus every valid axis flip, which is closed). `density_has_symmetry` checks
every non-null density per call (absolute tolerance 1e-10);
`symmetrize_over_ops` is the final gather. All three are in
`src/integrals/quartet_orbit.h`. The bra body is compiled in three modes
(none, replication, skeleton). Any non-symmetric density — PCM,
symmetry-broken UHF, random test densities — falls back to replication,
unchanged.

- **Result.** 0.604 → 0.454 s on the same symmetric density (1.33×,
  predicted 1.2–1.3×). max|ΔG| 7e-14 against the no-symmetry build;
  non-symmetric fingerprints bitwise unchanged.
- **Gate.** `check_skeleton_symmetry` in `planck-fock-accumulate`:
  group-averaged densities built from dense $O_R$ matrices (independent of
  the production gather), water C2v plus a synthetic order-2 flip group,
  every engine, RHF, UHF and J/K, ≤ 1.6e-14. The flip group is a true ERI
  symmetry only in STO-3G, so its J/K reference is replication, not the
  no-symmetry build. The older random-density gates still pass against
  replication; a skeleton build would miss them by O(1), so they prove the
  fallback fires.
- **Mutations.** Dropping the weight fails 13 checks; skipping the
  symmetrization fails 7. Inverting the forbidden test survives, and must
  (see Findings).
- **End to end.** CO2 / 6-31G\* direct and water / STO-3G HSE06 match
  symmetry-off in every printed Eh digit.
- **Who gains.** Symmetric DFT: its direct J/K builds always pass the ops,
  and SAO blocking keeps the density symmetric (water took the skeleton path
  on all 16 builds). Symmetric HF reaches this loop only with PCM (which
  falls back) or a linear point group. **Direct HF on linear molecules gains
  nothing**: SAO blocking is not implemented for infinite groups, so CO2's
  iterates break the symmetry (deviation 0.6 at iteration 2, 3e-10 at
  convergence) and every build after the first falls back. Loosening the
  tolerance would build G from the group average of a non-symmetric iterate
  and change the SCF path. No regression case runs a symmetric linear
  molecule through direct HF, and no symmetric open-shell case in the suite
  changes energy with symmetry off, so blocking linear groups in their D2h
  subgroup would change no tested answer.

## Findings

**The floor is the primitive loop, not the recurrence.** The (ss|ss) closed
form has no recurrence left, yet per component it was only 1.66–1.75×
faster than HGP: every call reruns the full primitive loop (an O 1s in
6-31G\* has 6 primitives, so 1296 primitive quartets, repeated 3× for
(ps|ss), 9× for (pp|ss), 81× for (pp|pp)). At shell granularity it ran once:
4.9×, and 10–16× at L = 4.

**This is not the hoist that failed.** `docs/ERI_PERFORMANCE_SCOPE.md`
records a contract-once-per-shell-quartet hoist as 1.14–1.62× *slower*: it
kept the general VRR/HRR with snapshot copies and per-component gathers. A
closed-form kernel has no buffer, gather or HRR; its outputs live in
registers.

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
build) passes per-*axis* maxima over a shell pair, so (ff|ff) reached
MMAX = 36 and smashed the stack: SIGBUS on any conventional HGP build with f
shells or higher. (dd|dd) peaks at exactly 24, so no d-basis gate saw it,
and the f/g/h spherical gates that did crash sat among known failures for
two days. Bisected inside the commit, fixed 2026-10-20 by sizing `F` to
`6*(VRR_DIM - 1) + 1` with an assert. **A red gate that is "already
failing" still needs its cause checked.**

**Code layout can cost more than the change.** Hoisting L3b's orbit buffer
slowed molecules *without* symmetry by 5 %: the symmetry branches sat inside
the hot loop, so moving them shifted register allocation. Compiling the loop
body once per symmetry mode removed the coupling.

**Screening must never reach the Schwarz table.** The first P2 experiment
gave max|ΔG| = 3e-6 at a 1e-14 threshold (4.6e-10 even at 1e-20), because
`schwarz_table_hgp` builds Q with the same kernel: skipping inside it
lowered Q below the true (ij|ij), so Schwarz stopped being an upper bound.
With Q built unscreened the error is a well-behaved ~400× the threshold.

**A cost-only test cannot be value-gated.** S1's forbidden-quartet skip
survives inversion because the final symmetrization cancels a forbidden
quartet exactly (the op fixing it with sign −1 maps its contribution to its
own negative). It only saves work.

**Gates must be able to fail.** Every kernel gate was mutation-tested. Gates
caught what review missed: the flag-off fingerprint caught C3 sending
(ss|ss) to the closed form with the flag off; the layout-coverage
requirement caught water/6-31G\* being unable to produce (d,p)-ordered
layouts; sign-only symmetry fixtures make every quartet its own
representative, so L3a refuses them and L3c got a gate with real
shell-moving ops.

**Print a census's mapping before believing its total.** D0's first
general-contraction count required a representative to be *strictly
larger* than the shell it absorbs, so cc-pVDZ oxygen's equal-size 1s and 2s
never merged: 1.47–1.83× instead of 4.8–5.1×, which nearly retired D2.

**Write the prediction down first.** One table row misread the class data
(cc-pVDZ L = 3 as 50 % instead of 31 %); C4a's predictions were low by
3–4×. Both were caught only because the number was written down first.

## C4a — the four-raise prototype (measured, not landed)

Four raises give $3^4 = 81$ ordered outputs: exact for (pp|pp), redundant on
d layouts ((ds|ds) and (dd|ss) need 36, (ps|pd) 54). Removing the fourth
raise couples it to each earlier raise on the same axis:

$$
D_{ijkl} = d_{4,l}\,C_{ijk}(0) + w_{4,l}\,C_{ijk}(1)
 + [l=i]\,\kappa_{14}\bigl(V^{23}_{jk}\bigr)
 + [l=j]\,\kappa_{24}\bigl(V^{13}_{ik}\bigr)
 + [l=k]\,\kappa_{34}\bigl(B_{ij}\bigr),
\qquad
V^{13}_{ik}(m) = d_{3,k}\,A_i(m) + w_{3,k}\,A_i(m+1) + [i=k]\,\kappa_{13}(\theta;\,m)
$$

with $V^{23}$ built from $A^{(2)}$; auxiliary orders $\theta$ 0–4, $A$ 0–3,
$A^{(2)}$ and $B$ 0–2, $V^{13}$, $V^{23}$, $C$ 0–1. About 70 lines. A
temporary probe validated every total-L = 4 block component by component
against HGP (max|error| ≤ 1.3e-15; dropping the $[l=k]$ coupling gives 1.0)
and timed it against HGP on the same work:

| layout | class | 8 H2O / 6-31G | 4 H2O / 6-31G\* | 1 H2O / cc-pVDZ | 4 H2O / cc-pVDZ |
|---|---|---|---|---|---|
| ABCD | (pp\|pp) | 15.9× | 14.3× | 10.4× | 18.6× |
| AACC | (ds\|ds) | — | 7.0× | 6.1× | 8.2× |
| AABB / CCDD | (dd\|ss) / (ss\|dd) | — | 5.4× / 5.4× | 3.9× / 3.9× | 4.3× / 5.0× |
| mixed d+p+p | e.g. (dp\|ps) | — | 10.5–13.8× | 9.2–19.0× | 8.7–16.7× |
| **all L = 4** | | **15.9×** | **10.4×** | **10.5×** | **12.6×** |

Predicted beforehand: (pp|pp) ≥ 4×, (ds|ds) 2–3×. So C4b covers every L = 4
layout, needs no generator (one becomes worth it only at L = 5, 243
outputs), and a five-raise chain is a fair L ≥ 5 candidate.

## Open, in dependency order

```
 interface           bitwise-gated kernels          shared exponents (value-changing)         last
 1 D1 block ──┬──▶ 2 C4b L=4 kernel ──▶ 3 C5 screened ──┬──▶ 4 D2a prototype ──▶ 5 D2b families ──┐
   interface  │                                          │                                        │
              └──────────────────────────────────────────┴────────────────────▶ 6 P2 ε screening ──┴──▶ 7 L≥5
 independent:  D3 RI-JK pair factors (BLAS doc B0)
```

| step | needs | bitwise? |
|---|---|---|
| 1 D1 | D0, C4a | yes (per-component fallback), ~1e-15 where a block kernel dispatches |
| 2 C4b | D1: lands as an implementation of its interface | ~1e-15 vs HGP; flag-off fingerprints unchanged |
| 3 C5 | C4b: screened scaling added once to the whole L ≤ 4 family | screened kernels only |
| 4 D2a | C4b: an SP family quartet spans (ss\|ss)…(pp\|pp) | nothing lands |
| 5 D2b | D1, C4b, D2a | no (summation order) |
| 6 P2 | D1 (carries ε); after D2b, so its bound is written once per family and re-measured | no (~1e-12) |
| 7 L ≥ 5 | C4b, D2b, P2: needs a profile with all of it in place | — |
| D3 | — (different code) | per the BLAS doc |

- **D1 does not need Phase E** (retiring the per-AO `ShellPair` adapter):
  the closed-form path already feeds a shell kernel with first-component
  pairs and a per-AO `norm_ratio`.
- **Fingerprints stay strong as long as possible**: steps 1–3 keep the
  flag-off fingerprints; D2b and P2 each re-baseline everything once and
  land alone, so a fingerprint change has one cause.
- **The thread-count drift** (Out of scope) is not ordered here. If it is
  fixed, fix it before D1, which rewrites the loop it lives in.

### 1 · D1 — a shell-quartet block interface for every engine

The engine interface becomes a block call: two first-component shell pairs
and the shell layouts in, every component of the shell quartet out, one
primitive loop. Readout and `norm_ratio` stay in the loop.
- **The interface** carries `prim_eps = 0` (P2) and the screened-kernel
  scaling (C5) from day one, inert by default, so no later step changes the
  signature.
- **The closed forms (L ≤ 3) move onto it**, replacing the
  `closed_form_max_l` side path. Bitwise.
- **A per-component adapter** wraps today's `eri_elem` for every engine and
  class without a block kernel. Bitwise.
- **An HGP block kernel**: VRR once per primitive quartet for the whole
  shell quartet, HRR outside, intermediates in fixed-size stack storage
  (sized for per-axis maxima — see the H1 finding). Dispatched per class
  only where measured to win. D0 found HGP repeating its primitive loop
  27–78× per shell quartet; the failed hoist's snapshot copies and gathers
  are the mechanism to avoid.
- **Explicit engines stay pure** (adapter only) unless Auto is selected.

Gates: flag-off and closed-form fingerprints unchanged; a per-quartet sweep
of the HGP block kernel against per-component HGP on every reachable layout
(water in 6-31G, 6-31G\*, cc-pVDZ, cc-pVTZ), mutation-tested; core, smoke,
and the fixed-thread-count identity cases.

### 2 · C4b — the four-raise kernel, on D1

C4a's prototype becomes the fourth level of `low_l_shell_eri`, by hand,
outputs in registers, as a D1 block kernel for every L = 4 layout:

| fixture | L = 4 share | class speedup (C4a) | predicted | ceiling |
|---|---|---|---|---|
| 8 H2O / 6-31G | 30 % | 15.9× | 1.39× | 1.43× |
| naphthalene D2h | 34 % | ~15× | 1.47× | 1.52× |
| 4 H2O / 6-31G\* | 34 % | 10.4× | 1.44× | 1.52× |
| 1 H2O / cc-pVDZ | 39 % | 10.5× | 1.55× | 1.64× |
| 4 H2O / cc-pVDZ | 45 % | 12.6× | 1.71× | 1.82× |
| 1 H2O / cc-pVTZ | 12 % | — | ~1.12× | 1.14× |

The L = 4 shares predate S1; re-measure the naphthalene row. Gates as for
C3.5: the C4a validation made permanent in `planck-hgp-engine-smoke` with
every reachable four-raise layout required, mutation tests on each coupling,
sign and norm, unchanged flag-off fingerprints, max|ΔG| ≤ ~5e-15.

### 3 · C5 — screened kernels, across the closed-form family

The closed forms need `hgp_vrr`'s screened scaling: T × `boys_scale`,
prefactor × `prefactor_scale`, W−P/W−Q × `screen.rho / rho`, and the
straddling coupling × `boys_scale` (the `inv_2_delta` gotcha). Added once,
to every level, through D1's parameter. Gate: a screened per-quartet sweep
against OS on a mixed bra/ket-AM fixture for LongRange and ShortRange, plus
the `water_*_engine_os_vs_hgp` comparators. Pays only for range-separated
hybrids; until it passes, screened kernels stay on the adapter.

### 4 · D2a — shared-exponent family prototype (nothing lands)

A **family** is the set of segmented shells on one atom whose exponent sets
are contained in one exponent set: Pople SP shells (6-31G oxygen's valence s
and p over the same three exponents — 16 segmented blocks per family
quartet, each running its own primitive loop) and general contractions
(cc-pVDZ O 1s/2s). A family quartet can run one primitive pass: compute the
coefficient-free intermediates once, then add each member block's
coefficient product times the intermediates of its own L (an SP family
needs every subset of the four raises, not only the chain's).

Families exist only inside the fused loop: `Shell`, `ShellPair`, `_index`,
normalization, 1e integrals, gradients and the spherical transform are
untouched, and per-member Schwarz, canonical filtering and accumulation stay
as they are.

The prototype is a probe mode with an SP family kernel (all 16 subsets) and
a within-L kernel (one s pass contracted into each member), validated
against HGP and timed against the sum of its member blocks on the current
path. Guess, from D0 and A0: 1.3–1.6× on 6-31G builds, 1.5–2× on cc-pVDZ.
D2a decides whether D2b goes ahead and for which families.

### 5 · D2b — shared-exponent families in the fused loop

Land what D2a admits, as D1 block kernels: families built once per Fock
build; family primitive pairs keep a pair if *any* member needs it (P1's
cutoff on the largest member coefficient product); kernels dispatched per
family-quartet class where D2a measured a win. Gates: per-component
validation against HGP on every reachable family layout, mutation-tested on
the coefficient contraction and subset selection; G and energies within
~1e-12 of the current path. Lands alone.

### 6 · P2 — quartet-level primitive screening (a policy decision)

After P1, another ~20 % of primitive quartets on water clusters have a
prefactor bound between 1e-24 and 1e-12. Skipping them below ε, with the
Schwarz table held exact:

| fixture | ε = 1e-14 | ε = 1e-12 | max\|ΔG\| at 1e-14 / 1e-12 |
|---|---|---|---|
| 8 H2O / 6-31G | 0.194 → 0.171 s, 1.13× | 1.37× | 4.0e-12 / 3.3e-10 |
| 4 H2O / 6-31G\* | 0.504 → 0.471 s, 1.07× | 1.18× | 4.9e-12 / 3.7e-10 |
| naphthalene D2h | 0.598 → 0.520 s, 1.15× | 1.22× | 4.8e-12 / 4.8e-10 |

ΔG is ~300–500 × ε. Proposed rule: ε = 1e-4 · `tol_eri` (1e-14 at the
default). Re-measure after D2b, which removes many of the same passes. ε
arrives through D1's `prim_eps`; `schwarz_table_*` must pass ε = 0 (gate:
Q built with screening on equals the exact Q). Lands alone; energies move at
~1e-12 and convergence-floor iteration counts may move.

### 7 · L ≥ 5 — a decision, not a step

After C4b, d bases are still ~35–50 % HGP at L ≥ 5 and cc-pVTZ ~87 %.
Candidates, each needing a prototype first: a five-raise chain (243 ordered
outputs); D1's HGP block kernel if it wins at high L; per-class generated
VRR over distinct Cartesian components with no runtime-sized scratch. Each
written against D2b's family interface, chosen from a fresh profile.

### Independent · D3 — RI-JK pair factors (B0 in `docs/ERI_FOCK_BLAS_SCOPE.md`)

Exact four-centre integrals stay $K^4$ per shell quartet whatever D1 and D2b
do; large cc-pVTZ/QZ runs are made fast in practice by density fitting.
Planck's RI-JK recomputes its density-independent pair factors twice per
iteration, ~61 % of the RI Fock build. Scoped by the BLAS doc; touches none
of the code above, and for big deep-contracted jobs is probably the cheapest
large win.

## Ruled out

- **L4, a per-block readout table.** The readout is ≤ 3.6 % (A0).
- **Forcing `accumulate_term` / `distinct_eri_orbit` inline.** Their dyld
  stubs cost < 1 %.
- **Avoiding the scalar `boys(0, T)` in the (ss|ss) branch.** 6.6 % on
  8 H2O, but mostly the Boys evaluation itself; only the indirect call is
  avoidable.
- **A generator for the L = 4 kernel.** The hand-written level was correct
  first time (C4a).
- **Faster per-primitive work** (vectorizing the primitive loop, a cheaper
  Boys path) for now: bounded at ~1.5–2×, not specific to deep contraction,
  and easier once D1 and D2b fix the kernel shapes.
- **Loosening S1's density tolerance** so linear HF takes the skeleton path:
  it would change the SCF trajectory on non-symmetric iterates.

## Out of scope

- **A local accumulation buffer per block.** Changes summation order, touches
  the threading determinism rules, and the `G` writes were ≤ 8 % of the
  build.
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
  the summation order. Predates this work
  (`water_rhf_spherical_direct_631gd`'s drift is the same thing).
