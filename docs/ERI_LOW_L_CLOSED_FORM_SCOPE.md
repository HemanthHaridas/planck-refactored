# Closed-Form Low-Angular-Momentum ERI Kernels — Scope

**The direct Fock build is now 8.0× faster on sp bases, 2.3× on d bases,
and 2.4× on symmetric molecules than before this work, with G bitwise
unchanged by every step after the closed forms themselves.**

| fixture (one thread, `engine auto`) | Fock build now | before this work | speedup | vs explicit `engine hgp` now |
|---|---|---|---|---|
| 8 H2O / 6-31G (nb = 104) | **0.187 s** | 1.50 s | **8.0×** | 4.4× |
| 4 H2O / 6-31G\* (nb = 76) | **0.497 s** | ~1.12 s† | **~2.3×** | 1.5× |
| 1 H2O / cc-pVDZ (nb = 25) | **0.028 s** | ~0.058 s† | **~2.1×** | 2.0× |
| naphthalene / 6-31G, D2h (nb = 106) | **0.454 s**§ | 1.454 s‡ | **3.2×** | ~2.4× (not re-measured after S1) |

Best of three runs of `planck-eri-class-share` on `15c532e4`, Apple M3,
`build-full`, `OMP_NUM_THREADS=1`. "Before" is the build before C1, after
the Boys-library fix (C0), which gave another 1.37× / 1.13× on its own.
† The product of the measured per-step ratios below; that fixture was never
timed before C1. ‡ The symmetric build before L3a. The last column is what
the Auto engine buys over an explicit engine, which gets only the
engine-independent steps. § After S1, on a group-averaged density (the
skeleton path needs a symmetric one); the earlier rows are on `15c532e4`.

**Symmetry now pays 5.2×** on naphthalene (2.36 s without `sym_ops`),
against 1.9× before the L3 series. 4× was the orbit cap for this planar
molecule; S1's skeleton accumulation took it past that (see S1).

**Next:** C4b alone is predicted at 1.4–1.7× on sp and cc-pVDZ builds (C4a
measured its kernel at 10–16× over HGP on its class). On cc-pVTZ, where HGP
at L ≥ 5 is 87 % of the build, D1 is the lever.

Status: **in flight.** Landed: C0, C1, C2–C3.5, H1, L1, L2, L3a, L3b, P1,
L3c, S1. Measured: A0, D0 (corrected), C4a. Open, in dependency order: D1,
C4b, C5, D2a, D2b, P2, then the L ≥ 5 decision, with D3 as an independent
track. When those
land or are abandoned, rewrite this as an answer.

**Question:** How much faster can the direct (memory-direct, fused) Fock
build get by evaluating low-angular-momentum shell quartets in closed form,
and what else in the build stands in the way?

## How it got there

Each step is a ratio over the build *with all previous steps*, so a row
multiplies:

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c | total |
|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | **8.0×** |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | **~2.3×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | 1.43× | — | — | 1.00× | **~2.1×** |

On the symmetric fixture the L3 series did most of the work: L3a 1.70×,
L3b 1.23×, then L3c and P1 together 1.16× (1.454 → 0.855 → 0.702 → 0.597 s).

| kind of step | steps | what it removed |
|---|---|---|
| closed-form kernels | C2, C3, C3.5 | the primitive loop repeated per Cartesian component, for total L ≤ 3 |
| work around the kernels | C1, H1, L1, L2 | `ShellPair` rebuilds, per-order Boys calls, empty blocks, the quadratic orbit dedup |
| symmetry bookkeeping | L3a, L3b, L3c | full orbits built for quartets that are not representatives, per-quartet allocation, rejected blocks |
| dead primitive work | P1 | primitive pairs whose prefactor has underflowed (19 % of executed quartets on water clusters) |

## Short answer

Closed forms help only at **shell** granularity. A closed form that returns
one Cartesian component per call is about 1.7× faster than HGP on its class.
One that returns every component of a shell quartet from a single primitive
pass is about 5× faster at total L ≤ 3, and 10–16× at L = 4, because the
repetition it removes grows with the number of components. Once the kernels
were fast, the work around them became the cost: the loop, symmetry
bookkeeping, and primitive quartets that contribute nothing.

The basis decides the payoff. The closed forms cover total L ≤ 3, which is
almost all of a pure sp build but only part of one with d shells. HGP is
still ~¾ of a 6-31G\* build and ~99 % of a cc-pVTZ one, and that is what the
open steps target. These are Fock-build numbers, not SCF wall time. The
~200× libcint gap in `docs/ERI_PERFORMANCE_SCOPE.md` was measured before
this work and has not been re-measured; at 2–8× this is a large step, not a
fix.

## How it is measured

`planck-eri-class-share` (`tests/eri_class_share_benchmark.cpp`, not a
ctest) drives the production `Integrals::fused_fock_build` on five fixtures:
1, 4 and 8 waters (6-31G\* and 6-31G), one water in cc-pVDZ, and
naphthalene / 6-31G (nb = 106, axis-aligned, so `update_integral_symmetry`
finds all 8 D2h sign flips). For each fixture it prints:
- the HGP time by (L_AB, L_CD) class and the coverage projections;
- the closed-form build at `closed_form_max_l` 1, 2 and 3, with max|ΔG|
  against HGP;
- for naphthalene, the build with and without `sym_ops`;
- an FNV-1a **fingerprint of G** from the flag-off build.

`--profile N [nwater basis]` repeats the production-shaped build for
`sample` (nwater 0 is naphthalene). Apple M3, `build-full` (g++-15 `-O3`),
`OMP_NUM_THREADS=1`.

Rules this work learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling): one binary
  alternated between 1.0 s and 1.22 s. Compare best-of runs of interleaved
  binaries, or one binary with a switch off and on. A single run is not
  evidence.
- **A fingerprint, not a timestamp, says which code ran.** Twice a binary
  was older than its source (a restore within the same second as the link; a
  build in flight while files changed). Check `nm` or the fingerprint.
- **Fingerprints depend on the fixture list.** The test density comes from
  `Eigen::MatrixXd::Random`, whose state carries across fixtures, so adding
  or removing a fixture changes every fingerprint after it. Compare
  fingerprints only between runs with the same list.
- **Attribute before predicting.** Inlined helpers hide inside their
  caller's self time. Profiling builds that mark them `noinline`, with
  temporary counters, split the time before anything changes. They are
  never committed.

## Where the time goes now

**By class** (D0, after P1 and L3c): the share of the production build that
is still HGP. The L = 4 time is the L = 4 classes' share of the HGP-only
build times the HGP-only build time. Single runs; shares within a run are
consistent.

| fixture | production build | HGP at L = 4 | HGP at L ≥ 5 |
|---|---|---|---|
| 8 H2O / 6-31G | 0.222 s | 30 % | 0 |
| naphthalene D2h | 0.651 s | 34 % | 0 |
| 4 H2O / 6-31G\* | 0.504 s | 34 % | 50 % |
| 1 H2O / cc-pVDZ | 0.032 s | 39 % | 34 % |
| 4 H2O / cc-pVDZ | 2.573 s | 45 % | 36 % |
| 1 H2O / cc-pVTZ | 1.631 s | 12 % | **87 %** |

**By frame** (A0, on `a1f99d7a`, before P1 and L3c): self time as a share of
the build, with the loop's helpers split out by a `noinline` build.

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

What it says:
- **sp bases:** HGP at L = 4, (pp|pp), is 30–34 % of the build; the rest is
  spread over Boys, the closed-form kernel and the loop. C4b takes the HGP
  part.
- **cc-pVDZ:** L = 4 and L ≥ 5 are about equal. C4b and D1 both matter.
- **cc-pVTZ:** almost everything is HGP at L ≥ 5. Only D1 and the L ≥ 5
  kernels move it.
- **Every basis family:** 2–5× of the primitive passes repeat work over a
  shared exponent set (D0). D2a and D2b target that.
- **Symmetric builds:** the loop and the representative check were the
  largest item before L3c. The `G` writes, which symmetry barely reduces,
  are S1's target.

## Findings

**The floor is the primitive loop, not the recurrence.** The (ss|ss) closed
form has no recurrence left to remove, yet a per-component version was only
1.66–1.75× faster than HGP. Every call runs the full primitive loop: one Boys
call and one prefactor product per primitive quartet. An O 1s in 6-31G\* has
6 primitives, so (1s 1s|1s 1s) is 1296 primitive quartets, and the
per-component interface repeats that 3× for (ps|ss), 9× for (pp|ss), 81×
for (pp|pp). The shell-level prototype ran it once: 4.9×. C4a confirmed the
scaling: at L = 4, where blocks have up to 81 components, the shell-level
kernel is 10–16× faster than HGP.

**This is not the hoist that failed.** `docs/ERI_PERFORMANCE_SCOPE.md`
records a contract-once-per-shell-quartet hoist as 1.14–1.62× *slower*: it
kept the general VRR/HRR with its buffers and gathers. A closed-form kernel
has no buffer, gather or HRR, and its outputs live in registers.

**A profile share is a lower bound on what removing the work is worth.**
The `ShellPair` rebuild was 8.4 % of samples and worth 1.16–1.26×, because
its allocator and cache traffic were charged to other frames. The converse
also happened: L3c cut 59 % of the symmetric loop's iterations for 1.07×,
because the iterations it removed were the cheap ones.

**The cheap fix often sits next to the hot frame.** HGP made one Boys call
per order through the loader's indirect call (6.8 % *self* in the wrapper).
OS had already switched to one `boys_vec` call; HGP and RI never had.

**Code layout can cost more than the change.** Hoisting L3b's orbit buffer
slowed molecules *without* symmetry by 5 %. Nothing there touched the
buffer, but the symmetry branches sat inside the hot loop, so moving them
shifted its register allocation. Compiling the loop body once per symmetry
mode removed the coupling.

**Screening must never reach the Schwarz table.** The first P2 experiment
gave max|ΔG| = 3e-6 at a 1e-14 threshold, the same at 1e-12, and 4.6e-10
even at 1e-20. No bound allows that. The cause was the Q table, which
`schwarz_table_hgp` builds with the same HGP kernel. Skipping inside the
kernel lowered Q below the true (ij|ij), so Schwarz stopped being an upper
bound and real quartets were dropped. With Q built unscreened, the error is
a well-behaved ~400× the threshold.

**Gates must be able to fail.** Every kernel gate was mutation-tested by
breaking each coupling term, sign and norm in turn. Gates caught three real
problems that review missed:
- *the flag-off fingerprint*, which caught C3 sending (ss|ss) blocks to the
  closed form with the flag off (`0 ≤ max_l` for max_l = 0);
- *the layout-coverage requirement*, which caught water/6-31G\* being unable
  to produce the (d,p)-ordered layouts, so water/cc-pVDZ joined the sweep;
- *the L3a fixture check*, which refuses sign-only operations, because they
  make every quartet its own representative. `check_symmetry_ops` had the
  same blind spot for L3c, so L3c got a gate with real shell-moving ops.

**Print a census's mapping before believing its total.** D0's first
general-contraction count required a representative shell to be *strictly
larger* than the shell it absorbs. Two equal-size shells over the same
exponents, such as cc-pVDZ oxygen's 9-primitive 1s and 2s, never merged.
The count came out 1.47–1.83× instead of 4.8–5.1×, and that wrong number
nearly retired D2. It was caught only when a variant of the census printed
its group-to-representative map. A total is not evidence that the rule
behind it does what it says.

**Write the prediction down first, and expect to be wrong.** One table row
misread the class data (cc-pVDZ L = 3 as 50 % instead of 31 %); the
measurement exposed it. C4a's predictions (≥ 4× for (pp|pp), 2–3× for
(ds|ds)) were low by 3–4×. Both errors were caught only because the number
was written down before the run.

## Landed

### C0 — class-share probe, and the Boys-core fix (2026-10-17)

The probe above. Its first profile found 17.7 % of the build in libm calls
inside one branch of the out-of-tree Boys core. That was fixed there and
shipped in Boys library v1.1.0: ~1.13× on 4 H2O / 6-31G\*, ~1.37× on 8 H2O /
6-31G (1.21× / 1.64× when linked directly; the loader's indirect call cannot
inline). `water_rhf_hgp_sto3g` (11 → 10 SCF iterations) and
`ethylene_rhf_ccpvdz_auto` (16 → 15) moved at the convergence noise floor,
with energies unchanged, and were re-pinned.

### C1 — reuse the caller's ShellPairs (`c40a016c`)

`fused_fock_build` rebuilt a `ShellPair` (heap, `pow`, `exp` per primitive
pair) for every surviving component quartet, although every caller passes
the same pairs. It now looks them up in an `nb × nb` pointer table.
Bitwise; 1.16–1.26×; every engine. On the way, `planck-hgp-engine-smoke`
and `planck-compute-2e` were found segfaulting under ctest (no
`require_boys_library()` call) and fixed.

### C2, C3, C3.5 — closed-form shell kernels through total L ≤ 3 (`3c028544`, `4bbb2724`, `f0f7ceae`)

`low_l_shell_eri` (`src/integrals/low_l_eri.h`) treats a quartet's angular
momentum as up to three *raises* (center, axis), applied in center order.
Per primitive quartet, let $\theta_m = c\,K_{AB}K_{CD}\,2\sqrt{\rho/\pi}\,F_m(T)$
from one `boys_vec` call. Let $d$ be $P-A$, $P-B$, $Q-C$ or $Q-D$ for the
raised center, and $w$ be $W-P$ (bra) or $W-Q$ (ket). Then

$$
\begin{aligned}
A_i(m) &= d_{1,i}\,\theta_m + w_{1,i}\,\theta_{m+1} \\
B_{ij}(m) &= d_{2,j}\,A_i(m) + w_{2,j}\,A_i(m+1) + [i=j]\,\kappa_{12}(\theta;\,m) \\
C_{ijk} &= d_{3,k}\,B_{ij}(0) + w_{3,k}\,B_{ij}(1) + [k=i]\,\kappa_{13}(A^{(2)}_j) + [k=j]\,\kappa_{23}(A_i)
\end{aligned}
$$

$A^{(2)}_j$ is the second raise alone. $\kappa_{yx}(X;\,m)$ is the
Obara–Saika coupling of a new raise $x$ to an earlier raise $y$ that it
lowers:
- both raises on the bra: $\bigl(X(m) - \tfrac{\rho}{\zeta}X(m+1)\bigr)/2\zeta$;
- both on the ket: the same with $\eta$;
- one on each side: $X(m+1)/2(\zeta+\eta)$.

At most 27 outputs, all in registers.

- **Wiring.** Called once per block, on the first component quartet that
  survives the filters, behind `closed_form_max_l`. Only Auto sets it (to
  3), so explicit `engine os/hgp/rys` stay pure and the cross-engine
  comparators stay meaningful. Coulomb only.
- **Feeding and readout.** The kernel gets each shell's first-component
  pair. Outputs are scaled by a per-AO `norm_ratio` ($d_{xx}$ carries
  $1/\sqrt3$, $d_{xy}$ 1; exactly 1.0 for s and p). Slots are read from the
  Cartesian exponents in raise order, so one rule covers every layout.
- **Speed, each over the previous step:** C2 1.22–1.28× on 8 H2O / 6-31G
  (predicted 1.23×), C3 1.61× (predicted 1.42×), C3.5 1.87× (predicted
  1.6×), which is 4.4× over HGP on its own class.
- **Gates.** `planck-hgp-engine-smoke` runs every component quartet with
  total L ≤ 3 of water in 6-31G, 6-31G\* and cc-pVDZ (7840 / 15736 / 43456
  quartets) through the kernel against HGP: max relative difference
  6.1e-16. Every reachable raise layout is required (15 on sp, 31 on
  cc-pVDZ). Mutation-verified on every coupling term, the W−P sign, the
  readout and the norm ratio. max|ΔG| against HGP ≤ 4.4e-15. Not bitwise
  against HGP, by design.
- **Suites** after each step: core 79/79, smoke 43/45 (the two known rccsdt
  routing failures).

### H1 — hotspot round on the HGP kernel (`46dadc6a`)

All bitwise. One `boys_vec` call instead of MMAX+1 `boys` calls in
`hgp_vrr` and both RI seeds (1.10–1.12×). The m=0 slice is accumulated
straight from the VRR buffer, and the VRR zero-fill, whose zeros were never
read, is gone (1.04–1.06×).

**It shipped a stack overflow, fixed 2026-10-20.** The `boys_vec` call
writes into `double F[4*MAX_L + 1]`, sized for one component's total
angular momentum (MMAX ≤ 24). The hoisted block path (`MaxBoxLayout`)
passes the per-*axis* maxima over a shell pair instead, so (ff|ff) reaches
MMAX = 36. That overran the buffer, and the corrupted stack showed up as
SIGBUS on any HGP conventional build with f shells or higher. The old
per-order `boys` loop never touched a stack buffer. (dd|dd) peaks at exactly
24, which is why no d-basis gate saw it.

The bug went unnoticed because the f/g/h spherical gates crashed alongside
already-known failures, not as a new red. Bisected inside the commit
(reverting `boys_vec` alone fixes it; restoring the zero-fill alone does
not). Fixed by sizing `F` to `6*(VRR_DIM - 1) + 1` (MMAX ≤ 12·MAX_L) plus an
assert. `water_rhf_spherical_ccpvtz_fshell`, `ne_rhf_spherical_ccpvqz_gshell`
and `ne_rhf_spherical_ccpv5z_hshell` pass again. The RI and OS seeds use
real totals and fit their buffers.

### L2 — closed-form orbit dedup (`659883ab`)

`distinct_eri_orbit` searched each of the 8 permuted tuples against every
one already kept: 17.3 % of the 8 H2O build, 3.7× the `G` writes it fed.
Which tuples repeat follows from three equalities. With
$X = (\{k,l\} = \{i,j\})$ as multisets, tuple *n* of the fixed order (ij|kl),
(ji|kl), (ij|lk), (ji|lk), (kl|ij), (lk|ij), (kl|ji), (lk|ji) is kept iff:

| n | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| keep iff | always | i≠j | k≠l | i≠j ∧ k≠l | ¬X | ¬X ∧ k≠l | ¬X ∧ i≠j | ¬X ∧ i≠j ∧ k≠l |

It emits the same tuples in the same order, so the result is bitwise.
`planck-fock-accumulate` keeps the old search as an oracle over all 1296
tuples in 0..5, order included (mutation-verified). 1.20× on 8 H2O / 6-31G,
1.09× on 6-31G\*, 1.11× on D2h naphthalene.

### L1 — skip blocks that cannot hold a canonical quartet (`de72ca2c`)

Groups are AO-contiguous and `group_pairs` is ordered by `.a`, so a ket
block with `.a` below the bra's has k < i for every component. That was 49 %
of the blocks surviving Schwarz, and about half of all component
iterations. The ket loop now starts at `first_pair_of[bra.a]`. Bitwise.
1.06× on 8 H2O / 6-31G (predicted 1.05–1.07×), ~1.01–1.02× elsewhere.

### L3a — reject non-representatives before building the orbit (`f88991fe`)

The symmetric loop built the full orbit (allocate, apply every op, dedup,
sort) for every canonical quartet, then used it only if the quartet was the
representative, about 1 in 8 under D2h. `build_quartet_orbit` was 35.7 % of
that build and per-quartet heap traffic another ~25–30 %, so symmetry paid
only 1.9×. `is_orbit_representative` (`src/integrals/quartet_orbit.h`)
returns at the first operation whose canonicalized image is smaller, with
no allocation.
- **Gate:** an exhaustive comparison against `build_quartet_orbit` on a
  synthetic nb = 6 group whose ops permute AOs (231 quartets, 85
  representatives; the test refuses fixtures with none or all).
  Mutation-verified.
- **Speed:** 1.70× on D2h naphthalene (1.454 → 0.855 s); no-symmetry
  fixtures unchanged.

### L3b — reuse the orbit buffer; compile the loop per symmetry mode (`a1f99d7a`)

`fill_quartet_orbit` fills a reused per-bra buffer instead of allocating
(~13 % of the post-L3a symmetric build). The bra-loop body is a generic
lambda instantiated with `std::true_type` / `std::false_type`, so the
non-symmetric loop carries no symmetry code (see the code-layout finding).
Bitwise. 1.23× on D2h naphthalene (0.862 → 0.702 s, predicted ~1.15×);
no-symmetry fixtures unchanged.

### P1 — drop negligible primitive pairs (`15c532e4`)

A census of the primitive quartets the kernels actually execute, against
the prefactor bound $\lvert c_{ab}K_{ab}\,c_{cd}K_{cd}\rvert\,2\sqrt{\rho/\pi}$
(the (ss|ss) value at $F_0 = 1$), counted once per block for the closed form
and once per component for HGP:

| fixture | closed form < 1e-18 | HGP < 1e-18 | closed form < 1e-12 | HGP < 1e-12 |
|---|---|---|---|---|
| 8 H2O / 6-31G | 23.4 % | 18.7 % | 39.5 % | 29.3 % |
| naphthalene / 6-31G | 36.9 % | 19.8 % | 52.4 % | 33.9 % |
| 4 H2O / 6-31G\* | 21.5 % | 6.6 % | 38.1 % | 18.7 % |
| 1 H2O / cc-pVDZ | 5.0 % | 0.1 % | 8.1 % | 0.4 % |

The below-1e-18 band is a **pair** property: tight primitives on distant
atoms, where $e^{-\alpha\beta R^2/\zeta}$ has underflowed. One water has almost
none. The `ShellPair` constructor (`src/base/types.h`) now drops a primitive
pair whose $\lvert c_a c_b N_a N_b\,(\pi/\zeta)^{3/2} K_{ab}\rvert$ is below
1e-24, about $e^{-55}$ for a unit pair. It sits in one place and covers every
engine, the 1e integrals and the gradients. Every consumer iterates
`primitive_pairs.size()`, and the only `front()` user already guards
`empty()`.

- **Exactness:** G is **byte-identical** on all five fixtures, at
  `closed_form_max_l` 0 and 3. The cutoff removes 19 % of closed-form and
  18 % of HGP primitive quartets on 8 H2O. Raising it to 1e-16 removes only
  ~2 % more, so 1e-24 sits at the knee with margin.
- **Speed** (best of three interleaved runs):

  | fixture | production (Auto) | HGP only |
  |---|---|---|
  | 8 H2O / 6-31G | 0.222 → 0.193 s, **1.15×** | 1.18× |
  | naphthalene D2h | 0.714 → 0.647 s, **1.10×** | 1.17× |
  | 4 H2O / 6-31G\* | 0.526 → 0.507 s, 1.04× | 1.08× |
  | 1 H2O / cc-pVDZ | unchanged | unchanged |

- **The cutoff is absolute**, not tied to `tol_eri`. At 1e-24 even
  derivative integrals (×2α ≤ ~3e4 in cc-pVTZ) stay below 1e-19.
- **Suites:** core 79/79, smoke 43/45 (the two known rccsdt routing
  failures).

### L3c — reject whole blocks on the symmetry path (`15c532e4`)

A quartet's canonical form starts with its smallest AO. If any operation
sends one of a block's four shell groups below the bra's first group A,
every component quartet of the block maps to a smaller one, and the block
holds no representative. Setup builds `op_group[o][g]` from each op's
`ao_map` and checks that every op maps a group onto one group (otherwise the
rejection is off). The ket loop then skips such blocks with one `min` per
op.

- **Effect on D2h naphthalene:** 60 % of the blocks surviving Schwarz are
  rejected. Component iterations drop 14.4 M → 6.0 M and representative
  checks 13.4 M → 5.6 M. The representatives found (3,349,882) and the values
  computed (1,716,665 closed-form + 202,903 HGP) are unchanged.
- **Bitwise:** G is byte-identical at max_l 0 and 3; all five fingerprints
  are unchanged.
- **Speed:** 1.07× (0.725 → 0.680 s, interleaved, switched in one binary).
  With P1: 0.699 → 0.605 s.
- **Gate.** `check_real_symmetry_ops` in `planck-fock-accumulate` builds
  water's real C2v ops with `update_integral_symmetry` (the mirror and C2 swap
  the H shells) and compares fused against two-phase on every engine, RHF
  and UHF, in STO-3G and 6-31G\*: max|ΔG| ≤ 1e-14. Mutation-verified with
  `< A` → `!= A`, which is harmless under sign flips but rejects blocks that
  do hold representatives once shells move. The new gate fails on every
  engine (max|ΔG| 0.14–0.46), while all 16 old sign-flip checks still pass.
- **Production path.** For HF, the fused loop gets `sym_ops` only when SAO
  blocking is off (PCM, or linear point groups); otherwise full symmetry
  takes over. The DFT direct J/K builds always pass them. CO2 / 6-31G\*
  direct (D∞h, 8 ops) converges to the symmetry-off energy in all 10 printed
  digits.
- **Suites:** core 79/79, smoke 43/45 (the known rccsdt failures).

## Measurements that set the plan

Each of these used temporary instrumentation (counters, `noinline` splits or
a probe mode) that was never committed. Only the numbers land.

### A0 — the loop's self time, split (2026-10-20)

The frame table under "Where the time goes now" and the P1 census. It
decided three things. L4, a per-block readout table, is dropped: the readout
is at most 3.6 %. The symmetric item became L3c. Primitive screening moved in
from out of scope, as P1 and P2.

### D0 — how much primitive work is repeated, after screening (2026-10-20)

Deep contraction multiplies the primitive loop by two repetitions that the
architecture, not the arithmetic, causes:
- **Per-component calls.** The engine callable
  `eri_elem(spAB, spCD, lAx…lDz)` returns one component, and HGP's
  `_contracted_eri_elem` reruns the whole primitive loop (Boys, VRR,
  contraction) for each one, with no caching. The closed forms removed this
  for total L ≤ 3 only.
- **Segmented storage.** The GBS reader stores a general contraction as
  separate shells. Oxygen's 1s and 2s in cc-pVDZ are two 9-primitive shells
  over the **same nine exponents**, and the third s shell is the last of
  them.

An unscreened count over all shell quartets of one water put these at
4.9× / 4.8× (cc-pVDZ) and 14.3× / 4.9× (cc-pVTZ). D0 counted what the
production build actually executes, after Schwarz and P1. Per evaluated
block it recorded the primitive quartets executed, the unique count for the
segmented shell quartet, and the count once shells sharing an exponent set
are merged. A group maps to the largest shell on the same atom whose
exponent set contains its own, ties broken by index. Two variants of the
merge: within one L only (general contraction, D2 as first scoped), and
across L (adding Pople SP shells, where s and p share every exponent).

| fixture | executed / per shell quartet | HGP blocks only | merge within L | merge across L |
|---|---|---|---|---|
| 8 H2O / 6-31G | 1.11× | 49× | 1.00× | **2.23×** |
| 4 H2O / 6-31G\* | 2.84× | 52× | 1.01× | **2.13×** |
| naphthalene / 6-31G | 1.23× | 27× | 1.00× | **3.73×** |
| 4 H2O / 6-311G | 1.10× | 46× | — | **2.18×** |
| 1 H2O / cc-pVDZ | 1.79× | 48× | **4.82×** | 4.82× |
| 4 H2O / cc-pVDZ | 1.99× | 56× | **5.11×** | 5.11× |
| 1 H2O / cc-pVTZ | **9.25×** | **78×** | **4.96×** | 4.96× |

The "merge" columns are the ratio of per-shell-quartet primitive passes to
per-merged-quartet passes. The first version of this table reported
1.47–1.83× for cc bases: its rule never merged two equal-size shells (see
Findings). The cc sets have no cross-L sharing, and the Pople sets have no
within-L sharing; each family of bases has exactly one kind.

With the class table under "Where the time goes now", that says:
- **D1 is the lever for d and f bases.** Wherever HGP runs it repeats its
  primitive loop 27–78× per unique shell quartet. On cc-pVTZ that is
  9.4 M of 10.4 M executed primitive quartets. The ratio overstates the gain,
  because a block kernel's VRR covers every component and so costs more per
  primitive quartet; D1 must measure that per class. Even 3× net on the HGP
  part would be ~2.4× on the cc-pVTZ build.
- **C4b is the lever for sp bases and cc-pVDZ** (30–45 % of the build), and
  worth only 1.1× on cc-pVTZ.
- **Shared exponents are worth 2–5× in primitive passes on every basis
  family tested.** On cc bases it is the general contraction (4.8–5.1×,
  close to the unscreened 4.8×); on Pople bases it is the SP shells
  (2.1–3.7×). Both are the same mechanism, and D2 is rescoped to cover both
  (D2a, D2b). Primitive passes are not time: a merged pass does more
  arithmetic, so D2a measures the real gain.

### C4a — the four-raise prototype (2026-10-20)

Four raises give $3^4 = 81$ *ordered* outputs. That is exact for (pp|pp),
the only L = 4 class on sp bases. On d layouts it is redundant: (ds|ds) and
(dd|ss) need 36 distinct components, (ps|pd) 54, so up to 2.25× of the
arithmetic is wasted. Predicted beforehand: (pp|pp) ≥ 4×, (ds|ds) 2–3×.

The prototype extends the three-raise chain by hand, with every
intermediate in fixed-size local arrays. Removing the fourth raise couples
it to each earlier raise on the same axis, which needs the three two-raise
subsets:

$$
D_{ijkl} = d_{4,l}\,C_{ijk}(0) + w_{4,l}\,C_{ijk}(1)
 + [l=i]\,\kappa_{14}\bigl(V^{23}_{jk}\bigr)
 + [l=j]\,\kappa_{24}\bigl(V^{13}_{ik}\bigr)
 + [l=k]\,\kappa_{34}\bigl(B_{ij}\bigr)
$$

$$
V^{13}_{ik}(m) = d_{3,k}\,A_i(m) + w_{3,k}\,A_i(m+1) + [i=k]\,\kappa_{13}(\theta;\,m)
$$

$V^{23}_{jk}$ is the same, built from the second raise $A^{(2)}_j$. The
auxiliary orders needed are $\theta$ 0–4, $A$ 0–3, $A^{(2)}$ and $B$ 0–2, and
$V^{13}$, $V^{23}$, $C$ 0–1. It is about 70 lines.

A temporary probe mode enumerated every total-L = 4 shell block surviving
Schwarz. It validated each one component by component against HGP, fed
exactly as the fused loop feeds the closed forms (first-component pairs,
per-AO `norm_ratio`, raise-order readout). Then it timed the kernel (once
per block) against HGP (once per component passing the canonical and
Schwarz filters), best of five.

- **Correctness.** max|error| ≤ 1.3e-15 against integrals up to ~1, on every
  layout of every fixture. Dropping the $[l=k]$ coupling gives max|error|
  1.0.
- **Speed, kernel against HGP on the same work:**

  | layout (raised centers) | class | 8 H2O / 6-31G | 4 H2O / 6-31G\* | 1 H2O / cc-pVDZ | 4 H2O / cc-pVDZ |
  |---|---|---|---|---|---|
  | ABCD | (pp\|pp) | 15.9× | 14.3× | 10.4× | 18.6× |
  | AACC | (ds\|ds) | — | 7.0× | 6.1× | 8.2× |
  | AABB / CCDD | (dd\|ss) / (ss\|dd) | — | 5.4× / 5.4× | 3.9× / 3.9× | 4.3× / 5.0× |
  | mixed d+p+p layouts | e.g. (dp\|ps) | — | 10.5–13.8× | 9.2–19.0× | 8.7–16.7× |
  | **all L = 4** | | **15.9×** | **10.4×** | **10.5×** | **12.6×** |

**Decision: C4b covers every L = 4 layout.** Both predictions were 3–4×
too low. Even the most redundant layouts, (dd|ss) and (ss|dd) with 45 of
81 outputs wasted, are 3.9–5.4× faster. The gain beats C3.5's 4.4× because
the removed repetition grows with the component count: 81 components per
(pp|pp) block, against at most 27 at L = 3.

Two consequences:
- **C4b does not need a generator.** The hand-written level was correct on
  its first run and is mutation-sensitive. A generator becomes worth its
  cost only if the chain goes to L = 5 (243 ordered outputs).
- **A five-raise chain is a fair L ≥ 5 candidate.** Redundancy cost far less
  than feared at L = 4.

## Open, in dependency order

```
 symmetric loop        interface         bitwise-gated kernels          shared exponents (value-changing)         last
 1 S1 skeleton ──▶ 2 D1 block ──┬──▶ 3 C4b L=4 kernel ──▶ 4 C5 screened ──┬──▶ 5 D2a prototype ──▶ 6 D2b families ──┐
      + symmetrize      interface    │                                         │                                        │
                                     └─────────────────────────────────────────┴────────────────────▶ 7 P2 ε screening ──┴──▶ 8 L≥5
 independent:  D3 RI-JK pair factors (BLAS doc B0)
```

| step | needs | why it needs it | bitwise? |
|---|---|---|---|
| 1 S1 (done) | — (strictly) | touches only the accumulation side of the symmetric branch; doing it before D1 means D1 rewrites a simpler loop (no orbit buffer, no replication) | no, ~1e-15; re-baselines only the symmetric fixture's fingerprint |
| 2 D1 | S1; D0 and C4a (done) | D0 says which classes a block kernel must win on; C4a says the interface carries ordered-raise kernels for every L = 4 layout | yes (per-component fallback), or ~1e-15 where a block kernel is dispatched |
| 3 C4b | D1 | lands as an implementation of D1's interface, not as another special case in the fused loop | ~1e-15 vs HGP; flag-off fingerprints unchanged |
| 4 C5 | C4b | screened scaling goes into the closed-form family once, covering L ≤ 4, instead of patching the L ≤ 3 kernel and redoing it | screened kernels only |
| 5 D2a | C4b | a prototype; an SP family quartet spans (ss\|ss) through (pp\|pp), so it needs the four-raise chain | nothing lands |
| 6 D2b | D1, C4b, D2a | family kernels are D1 block kernels; D2a decides which families to merge | no (summation order) |
| 7 P2 | D1 (and the policy decision) | D1's interface carries ε, which removes P2's plumbing blocker; after D2b so its bound is written once, per family, and its value re-measured after D2b removes many of the same passes | no (~1e-12) |
| 8 L ≥ 5 | C4b, D2b, P2 | needs a profile with everything above in place; new kernels are written once against the family interface instead of retrofitted | — |
| D3 | — | different code (RI-JK) | per the BLAS doc |

Three things about this ordering:
- **D1 does not need Phase E** (retiring the per-AO `ShellPair` adapter).
  The closed-form path already feeds a shell-level kernel with
  first-component pairs and a per-AO `norm_ratio` (the C2 wiring), and an
  HGP block kernel can be fed the same way. Phase E is cleanup that can
  follow D1.
- **Fingerprints stay strong as long as possible.** Steps 2–4 keep the
  flag-off G fingerprints unchanged. S1 re-baselines only the symmetric
  fixture, and only in builds that take the skeleton path. Steps 6 and 7
  re-baseline everything once each. Each value-changing step lands alone, so
  a fingerprint change has one cause.
- **The thread-count drift** (Out of scope) is not ordered here. Every
  "thread-identity" gate below means identical output at a fixed thread
  count. If that defect is fixed, fix it before D1, because D1 rewrites the
  loop it lives in.

### 1 · S1 — skeleton accumulation and final symmetrization on the signed-op path — DONE

**Result (2026-10-20).** On D2h naphthalene / 6-31G the build is
**1.33× faster** on the same symmetric density (0.604 → 0.454 s, against a
prediction of 1.2–1.3×). Symmetry now pays **5.2×** (2.36 s without
`sym_ops`), up from 4.0×. Against the no-symmetry build, max|ΔG| is 7e-14.
Non-symmetric fingerprints are bitwise identical to the previous commit.

How it is built:
- `skeleton_orbit_weight` gives representative, stabilizer and forbidden
  status in one pass over the ops.
- `density_has_symmetry` is the per-call check, with an absolute tolerance
  of 1e-10. It runs on every non-null density.
- `symmetrize_over_ops` is the final gather.

All three live in `quartet_orbit.h`. The bra body is instantiated in three
modes (none, replication, skeleton). The orbit size is |ops|/|stabilizer|,
which needs the ops to form a group. They do: `update_integral_symmetry`
builds the identity plus every valid axis flip.

What the gates found:
- **Skeleton gate** (`check_skeleton_symmetry` in `planck-fock-accumulate`):
  group-averaged densities, built from dense O_R matrices independently of
  the production gather. Water C2v plus a synthetic order-2 flip group,
  every engine, RHF, UHF, and J/K. Passes at ≤ 1.6e-14. The flip group is a
  true ERI symmetry only in STO-3G, so for J/K it is checked against
  replication, not against the no-symmetry build.
- **Fallback:** the existing random-density gates still pass against
  replication. The skeleton would differ from replication by O(1) on those
  densities, so their passing proves the fallback fires.
- **Mutations:** dropping the weight fails 13 checks; skipping the
  symmetrization fails 7. **Inverting the forbidden test survives, and must.**
  The final symmetrization cancels a forbidden quartet's contribution exactly
  (the op that fixes it with sign −1 maps it to its own negative), so that
  test only saves cost and no value gate can see it.
- **End to end:** CO2 / 6-31G\* direct and water / STO-3G HSE06, each with
  symmetry on, match symmetry off in every printed Eh digit. Water DFT takes
  the skeleton path on all 16 builds, because SAO blocking keeps the density
  symmetric.
- **Direct HF CO2 takes it only on iteration 1.** SAO is off for this linear
  case, so the iterates break the symmetry (deviation 0.6 at iteration 2,
  falling to 3e-10 at convergence). The build falls back, as designed, so
  linear HF gains nothing. A looser tolerance would build G from the group
  average of a non-symmetric iterate and change the SCF path, so the
  tolerance was not loosened.

The scoping notes follow.


**Why symmetry pays 4×, not 8×** (naphthalene / 6-31G, D2h, per Fock build,
temporary counters):

| | no symmetry | symmetry | ratio |
|---|---|---|---|
| canonical quartets surviving Schwarz | 13.37 M | 13.37 M | — |
| orbit representatives | — | 3.35 M | **4.0×** |
| … of which symmetry-forbidden (zero) | — | 1.43 M | |
| integral values computed | 13.37 M | 1.92 M | **7.0×** |
| closed-form kernel calls (per block) | 1.87 M | 0.48 M | 3.9× |
| G writes (accumulation) | 104.7 M | 59.7 M | **1.75×** |
| build time (best of three, no counters) | 2.37 s | 0.60 s | 4.0× |

Three things, in order of size:
- **A planar molecule caps the orbit at 4.** By the orbit–stabilizer
  theorem an orbit has $\lvert G\rvert / \lvert\mathrm{Stab}\rvert$ members. The
  mirror through the molecular plane maps every AO to itself and only flips
  the sign of the out-of-plane p functions, so it is in every stabilizer and
  orbits have at most $8/2 = 4$ members: canonical/representatives is
  3.99×. That plane's factor of 2 shows up instead as **zero integrals**. If
  an operation maps a quartet to itself with sign −1, the integral equals
  its own negative. 1.43 M of the 3.35 M representatives are skipped on that
  ground, so integral evaluations really are down 7.0×.
- **Closed-form calls are per block.** A block is evaluated if any of its
  components is needed. Being forbidden is a per-component property (the
  axis parity), so almost no block is entirely zero, and kernel calls fall
  only 3.9×.
- **G writes barely fall.** Each representative's value is replicated
  across its orbit and then across each element's distinct 8-fold
  permutations, so every nonzero integral still pays its full set of writes.
  Only forbidden quartets save any: 1.75×. The orbit fill and the
  per-element sign loop come on top.

**The change.** Accumulate each non-forbidden representative **once**, with
its 8-fold permutational contributions weighted by its orbit size, into a
skeleton $G^{\mathrm{skel}}$. At the end of the build, symmetrize:

$$
G_{\mu\nu} = \frac{1}{\lvert G\rvert}\sum_{R} s_R(\mu)\,s_R(\nu)\;G^{\mathrm{skel}}_{R(\mu)\,R(\nu)}
$$

That costs $n_b^2\,\lvert G\rvert$, which is trivial. It is the Dupuis–King
skeleton scheme, which Planck's full-symmetry HF path already uses with
dense operation matrices. Here the operations are signed AO permutations, so
the symmetrization is a gather.

- **One pass over the operations, no allocation.** For each canonical
  quartet, apply each op once. That answers three questions together: is
  there a smaller image (not a representative)? how many ops map the quartet
  to itself (the stabilizer, giving the orbit size)? does one map it to
  itself with sign −1 (forbidden)? This replaces both
  `is_orbit_representative` and `fill_quartet_orbit` on this path.
- **Validity: the density must be symmetric.** The skeleton scheme needs
  $O_R P O_R^{\mathsf T} = P$ for every operation. Today's replication works
  for *any* P, because it reproduces the integrals exactly. PCM runs (SAO is
  disabled there because the reaction field breaks symmetry) and
  symmetry-broken UHF do not satisfy this. So the build checks the
  deviation per call ($n_b^2\,\lvert G\rvert$) for every density it contracts
  (P, or Pa and Pb), and falls back to today's replication above a
  tolerance. The fallback is the current code, unchanged.
- **Who gains.** HF reaches this path only with PCM or linear point groups;
  other symmetric HF runs use the full-symmetry path, which is already
  skeleton-based. The DFT driver passes the same signed ops into its direct
  J/K builds, so symmetric DFT is the main beneficiary, together with linear
  molecules.

Predicted on D2h naphthalene: G writes 59.7 M → ~15 M (the 1.92 M computed
quartets × their distinct permutations); the orbit fill, the sign loop and
most of `accumulate_term`'s 7.7 % go. Roughly 1.2–1.3× on the symmetric
build, to be measured. Kernel calls are unchanged: block granularity is D1's
business, not S1's.

Gates:
- **Skeleton against replication**, on symmetrized densities (P averaged
  over the group), for water C2v and naphthalene D2h, RHF and UHF, every
  engine, max|ΔG| ≤ ~1e-14. The existing `check_symmetry_ops` /
  `check_real_symmetry_ops` gates use random densities, so they only
  exercise the fallback. Keep them, and assert that they did take it.
- **The fallback must fire** on a deliberately asymmetric density, so that
  the check cannot silently pass a PCM-like case.
- **Mutations:** drop the stabilizer weight; skip the symmetrization; invert
  the forbidden sign test. Each must fail the skeleton gate.
- **End to end:** CO2 / 6-31G\* direct (D∞h) and a symmetric DFT case must
  match their symmetry-off energies in all printed digits, plus core and
  smoke.

### 2 · D1 — a shell-quartet block interface for every engine

The fused loop's engine interface becomes a block call. It takes the two
first-component shell pairs and the shell layouts, runs the primitive loop
once, and returns every component of the shell quartet. Readout and
`norm_ratio` scaling stay in the loop, as they are for the closed forms
today. The step has five parts:
- **The interface**, with two parameters that later steps need and that
  default to inert: `prim_eps = 0` (P2) and the screened-kernel scaling
  (C5). Adding them now means no later step changes the signature.
- **The existing closed forms (L ≤ 3) moved onto it**, replacing the
  `closed_form_max_l` side path. Bitwise against today: same kernel, same
  feed, same order.
- **A per-component adapter** wrapping today's `eri_elem`, for every engine
  and class without a block kernel. Bitwise.
- **An HGP shell-quartet block kernel**: VRR once per primitive quartet for
  the whole shell quartet, HRR outside, intermediates in fixed-size stack
  storage. It is dispatched per class only where its measurement shows a
  win; D0 found HGP repeating its primitive loop 27–78× per shell quartet,
  so the headroom is large. The earlier contract-once hoist was 1.14–1.62×
  *slower* (1.23× on cc-pVDZ) because of snapshot copies and per-component
  gathers (`docs/ERI_PERFORMANCE_SCOPE.md`); that is the mechanism to avoid,
  not the idea.
- **Explicit engines stay pure** (adapter only) unless the user selects
  Auto, so the cross-engine comparators keep meaning something.

Gates: flag-off fingerprints unchanged; with the block kernel off, the
closed-form fingerprints unchanged too; a per-quartet sweep of the HGP block
kernel against per-component HGP on every reachable layout (water in 6-31G,
6-31G\*, cc-pVDZ, cc-pVTZ), mutation-tested; core, smoke and the
fixed-thread-count identity cases.

### 3 · C4b — the four-raise kernel, on D1

C4a's prototype becomes the fourth level of `low_l_shell_eri`, written by
hand. Outputs stay in registers, with **no** buffers or gathers. It lands as
a D1 block kernel for every L = 4 layout. Predicted from D0's L = 4 shares
and C4a's measured class speedups:

| fixture | L = 4 share | class speedup (C4a) | predicted | ceiling (class free) |
|---|---|---|---|---|
| 8 H2O / 6-31G | 30 % | 15.9× | 1.39× | 1.43× |
| naphthalene D2h | 34 % | ~15× (pp\|pp) | 1.47× | 1.52× |
| 4 H2O / 6-31G\* | 34 % | 10.4× | 1.44× | 1.52× |
| 1 H2O / cc-pVDZ | 39 % | 10.5× | 1.55× | 1.64× |
| 4 H2O / cc-pVDZ | 45 % | 12.6× | 1.71× | 1.82× |
| 1 H2O / cc-pVTZ | 12 % | — | ~1.12× | 1.14× |

The class speedups are so large that every prediction sits close to its
ceiling.

Gates as for C3.5: C4a's per-component validation, made permanent in
`planck-hgp-engine-smoke`, with every reachable four-raise layout required
and counted; mutation tests on each coupling term, sign and norm; unchanged
flag-off fingerprints; max|ΔG| ≤ ~5e-15 on all fixtures; core, smoke and the
identity cases.

### 4 · C5 — screened kernels, across the closed-form family

The closed forms need `hgp_vrr`'s screened scaling: T × `boys_scale`,
prefactor × `prefactor_scale`, W−P/W−Q × `screen.rho / rho`, and the
straddling coupling × `boys_scale` (the `inv_2_delta` term behind
`vault/Gotchas/HGP Screened inv_2_delta`). After C4b the closed-form family
covers L ≤ 4, so the scaling is added once, to every level together,
through D1's screened-scaling parameter. Gate: a screened per-quartet sweep
against OS on a mixed bra/ket-AM fixture, for LongRange and ShortRange,
plus the existing `water_*_engine_os_vs_hgp` comparators. It pays only for
range-separated hybrids. Until it passes, screened kernels stay on the
adapter.

### 5 · D2a — shared-exponent family prototype (nothing lands)

**The idea.** Many segmented shells on one atom share an exponent set and
differ only in their contraction coefficients, and sometimes in L:
- **Pople SP shells.** 6-31G oxygen's valence is an s and a p shell over the
  same three exponents, plus an s and a p over one more. The s and p blocks
  of a quartet of such shells, (ss|ss), (ps|ss), …, (pp|pp), are 16
  segmented blocks today, each running its own primitive loop over the same
  primitives.
- **General contraction.** cc-pVDZ oxygen's 1s and 2s are two 9-primitive
  s shells over the same exponents, with a third s shell on the last of
  them.

A **family** is the set of segmented shells on one atom whose exponent sets
are contained in one exponent set. A family quartet can be evaluated with
one primitive pass. Per primitive quartet, compute the coefficient-free
intermediates once: $\theta_m$ without the coefficient product, and the
raise-chain subsets up to the highest L in each family. Then, for every
member block of the family quartet, add its coefficient product times the
intermediates of its own L. For an SP family quartet, the member blocks need
every subset of the four raises, not only the chain's: (ps|ss) is the first
raise alone, (ss|sp) the fourth alone, and so on.

**Keep the shells as they are.** Families exist only inside the fused loop.
The `Shell` and `ShellPair` data model, `_index`, normalization, the 1e
integrals, the gradients and the spherical transform are untouched. The
first scope of D2 rewrote all of those; this version does not need to,
because the saving is in the Fock loop's primitive passes, not in how shells
are stored. Per-member-block Schwarz, canonical filtering and accumulation
stay exactly as they are; the kernel only shares the primitive pass.

**The prototype.** A probe mode, like C4a, with two kernels:
- an **SP family kernel**: all 16 raise subsets in one pass;
- a **within-L family kernel**: one s-only pass contracted into each
  member's coefficients.

Validate every component against HGP, then time a family quartet against
the sum of its member blocks on the current path (closed form plus C4b's
four-raise kernel where needed). Fixtures: 8 H2O / 6-31G, D2h naphthalene,
4 H2O / 6-311G, cc-pVDZ water, 4 H2O / cc-pVDZ.

**Prediction, from D0 and the A0 frame split.** Boys and the per-primitive
geometry are roughly half of a post-C4b closed-form pass, and the family
merge removes 2–5× of the passes. But a merged pass computes every subset
and contracts into several members, so it costs more than any one member's
pass. Guess: 1.3–1.6× on the 6-31G builds and 1.5–2× on cc-pVDZ. D2a
decides whether D2b goes ahead, and for which families: SP, within-L, or
both.

### 6 · D2b — shared-exponent families in the fused loop

Land what D2a admits, as D1 block kernels:
- **Families**, built once per Fock build from the segmented groups: same
  atom, exponent set contained in a representative's, any L.
- **Family primitive pairs.** P1 drops a primitive pair per segmented
  `ShellPair`; a family pair must keep a primitive pair if *any* member
  needs it, so its cutoff uses the largest member coefficient product.
- **The family kernels** from D2a, dispatched per family-quartet class where
  D2a measured a win; other blocks stay on the current path.

Gates: per-component validation against HGP on every reachable family
layout (Pople and cc fixtures), mutation-tested on the coefficient
contraction and the subset selection; G and energies unchanged to ~1e-12
against the current path (summation order changes); core and smoke. Land it
alone, as a value-changing step.

### 7 · P2 — quartet-level primitive screening (a policy decision)

After P1, another ~20 % of primitive quartets on water clusters have a
prefactor bound between 1e-24 and 1e-12. Skipping them in the closed-form
and HGP primitive loops when the bound is below ε, with the Schwarz table
held exact, measured:

| fixture | ε = 1e-14 | ε = 1e-12 | max\|ΔG\| at 1e-14 / 1e-12 |
|---|---|---|---|
| 8 H2O / 6-31G | 0.194 → 0.171 s, 1.13× | 1.37× | 4.0e-12 / 3.3e-10 |
| 4 H2O / 6-31G\* | 0.504 → 0.471 s, 1.07× | 1.18× | 4.9e-12 / 3.7e-10 |
| naphthalene D2h | 0.598 → 0.520 s, 1.15× | 1.22× | 4.8e-12 / 4.8e-10 |

ΔG is ~300–500 × ε on every fixture. The proposed rule is
ε = 1e-4 · `tol_eri` (1e-14 at the default), which keeps the primitive error
about 100× below what Schwarz screening at `tol_eri` already admits.

- **Re-measure first.** D2b removes many of the same primitive passes, so
  the gains above will shrink; measure again after D2b before deciding.
- **What D1 removes:** the plumbing blocker. ε arrives through the block
  interface's `prim_eps`, and the adapter passes it to the per-component
  engines.
- **What remains:** the policy decision (a non-bitwise default) and
  Schwarz safety. `schwarz_table_*` must pass ε = 0 (see Findings); the gate
  is that Q built with screening on equals the exact Q.
- **Landing:** alone, so the fingerprint re-baseline has one cause.
  Regression energies move at ~1e-12, and iteration counts at the
  convergence floor may move, as they did for the Boys fix.

### 8 · L ≥ 5 — a decision, not a step

After C4b, d bases are still ~35–50 % HGP at L ≥ 5, and cc-pVTZ ~87 %
(before D2b and P2 shift the shares). Candidates, each needing a C0-style
prototype first:
- a five-raise chain (243 ordered outputs), a fair candidate since C4a
  showed redundancy costs little at L = 4;
- D1's HGP block kernel, if step 2 found it wins at high L;
- per-class generated VRR over *distinct* Cartesian components, unrolled,
  with no runtime-sized scratch.

Each is written against D2b's family interface. Choose from a fresh
profile with steps 1–7 in place.

### Independent · D3 — RI-JK pair factors (B0 in `docs/ERI_FOCK_BLAS_SCOPE.md`)

Exact four-centre integrals stay $K^4$ per shell quartet whatever D1 and
D2b do. Large cc-pVTZ/QZ runs are made fast in practice by density fitting.
Planck's RI-JK recomputes its density-independent pair factors twice per
iteration, ~61 % of the RI Fock build. Owned and scoped by the BLAS doc as
B0. It touches none of the code above, so it can run at any time, and for
big deep-contracted jobs it is probably the cheapest large win.

Not on this list: faster per-primitive work (vectorizing the primitive loop,
a cheaper Boys call path). It is bounded at maybe 1.5–2×, is not specific to
deep contraction, and would be easier to do once D1 and D2b fix the kernel
shapes.

## Ruled out

- **L4, a per-block readout table.** The readout is ≤ 3.6 % of the build
  (A0).
- **Forcing `accumulate_term` / `distinct_eri_orbit` inline.** They are weak
  symbols called through dyld stubs, but the stubs cost < 1 %.
- **The scalar `boys(0, T)` in the (ss|ss) branch.** It is 6.6 % on 8 H2O,
  but most of that is the Boys evaluation itself. Only the indirect-call
  overhead is avoidable, worth a few percent at most.
- **A generator for the L = 4 kernel.** C4a's hand-written level was correct
  first time (C4a).

## Out of scope

- **A local accumulation buffer per block.** It changes summation order (not
  bitwise), touches the threading determinism rules, and the `G` writes are
  ≤ 8 % of the build.
- **Gradients.** Derivative ERIs are a separate kernel family (Open Work
  Phase D). P1 already covers them.
- **Spherical.** The direct spherical path back-projects to Cartesian and
  goes through the same fused loop.
- **The fused direct loop is not thread-count invariant.** Its accumulators
  are per-thread partials keyed by thread index and summed in fixed order.
  That makes a run reproducible at a given thread count but not across
  thread counts: CO2 / 6-31G\* direct, with or without symmetry, differs
  between 1 and 4 threads in the DIIS residual digits and the last digit of
  the kcal/mol energy (all 10 Hartree digits identical). It is the same
  defect the FCI sigma-build work recorded: the *partition* of work into
  accumulators must be fixed, not just the summation order. It predates this
  work (`water_rhf_spherical_direct_631gd`'s drift is the same thing) and is
  not chased here.
