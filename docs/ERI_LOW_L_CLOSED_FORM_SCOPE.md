# Closed-Form Low-Angular-Momentum ERI Kernels — Scope

Status: **in flight.** Landed: the class-share probe (C0), the Boys-core fix
it found, ShellPair reuse (C1), closed-form shell kernels through total
L ≤ 3 (C2, C3, C3.5), a hotspot round on the HGP kernel (H1), the
closed-form orbit dedup (L2), skipping non-canonical blocks (L1), and the
early non-representative exit on the symmetry path (L3a). Open: L3b, L4,
the C4 decision, and C5 (screened kernels). When those land or are
abandoned, rewrite this as an answer.

**Question:** How much faster can the direct (memory-direct, fused) Fock
build get by evaluating low-angular-momentum shell quartets in closed form,
and what else in the build stands in the way?

## Short answer

Closed forms help only at **shell** granularity. A closed form that returns
one Cartesian component per call is about 1.7× faster than HGP on its class.
One that returns every component of a shell quartet from a single primitive
pass is about 5× faster. Once the kernels were fast, the loop around them
became the next cost, and fixing that is the L series.

Each step is a ratio over the build *with all previous steps* (one thread,
production configuration), so a row multiplies:

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | total |
|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | **~6.8×** (1.50 → 0.22 s) |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | **~2.2×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | 1.43× | — | — | — |

Symmetric molecules gain on top of this. On D2h naphthalene / 6-31G, L3a
made the symmetric build 1.70× faster, so integral symmetry now pays 3.1×
(it paid 1.9× before, against an ideal of ~8×).

The basis decides the payoff. The closed forms cover total L ≤ 3, which is
almost all of a pure sp build but only part of one with d shells. On 6-31G\*,
HGP at L ≥ 4 is still ~59 % of the build. These are Fock-build numbers, not
SCF wall time, and they apply to the default `engine auto` only; explicit
engines get just the engine-independent steps (C1, H1, L2, L1, L3a). **Against the
~200× libcint gap this is a large step, not a fix.**

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

Rules that this work learned the hard way:
- **Timings are bimodal under load** (P/E-core scheduling): one binary
  alternated between 1.0 s and 1.22 s. Compare best-of runs of interleaved
  binaries, or one binary with a flag off and on. A single run is not
  evidence.
- **A fingerprint, not a timestamp, says which code ran.** Twice, a binary
  turned out older than the source (a restore within the same second as the
  link; a build in flight while files changed). Check `nm` or the
  fingerprint.
- **Attribute before predicting.** Inlined helpers hide inside their
  caller's self time. Profiling builds that mark them `noinline` (never
  committed) split the time before anything is changed. Every step below
  was predicted from such a split, and every prediction was checked.

## Where the time goes now

After C3.5, with the kernels and accumulation helpers separated (self time
as a share of the build):

| frame | 8 H2O / 6-31G | 4 H2O / 6-31G\* |
|---|---|---|
| `hgp_vrr` (HGP, total L ≥ 4) | 23.2 % | 59.3 % |
| closed-form kernel | 17.8 % | 3.2 % |
| `distinct_eri_orbit` (removed by L2) | 17.3 % | 7.9 % |
| loop traversal and filters (about half removed by L1) | 13.9 % | 4.1 % |
| Boys | ~19 % | ~10 % |
| `accumulate_term` (the `G` writes) | 4.7 % | 1.8 % |

Loop counters (L0, per build, temporary instrumentation):

| | 8 H2O / 6-31G | 4 H2O / 6-31G\* | 1 H2O / cc-pVDZ |
|---|---|---|---|
| ket-loop iterations | 6.91 M | 672 k | 6.1 k |
| … surviving block Schwarz | 987 k | 293 k | 6.1 k |
| … that cannot hold a canonical quartet (skipped by L1) | **486 k (49 %)** | 142 k (49 %) | 2.7 k (45 %) |
| component iterations | 4.36 M | 3.86 M | 114 k |
| … passing the canonical filter | 2.13 M (49 %) | 1.84 M (48 %) | 53 k (46 %) |
| accepted quartets (closed form) | 2.00 M (95 %) | 1.64 M (46 %) | 53 k (41 %) |

On the symmetric fixture (naphthalene, D2h), before L3a, the picture was
worse. With `sym_ops` the build was only **1.90× faster** than without,
although symmetry cuts the ERIs computed about 8×. Inclusive shares of that
build: `build_quartet_orbit` 35.7 %, per-quartet heap traffic ~25–30 %
(`free`, `operator new`, and `mach_absolute_time` called from inside macOS's
`free`), accumulation 16.2 %, ERI work ~25 %. L3a removed most of the first
two; the symmetric build has not been re-profiled since.

## Findings

**The floor is the primitive loop, not the recurrence.** The (ss|ss) closed
form has no recurrence left to remove, yet a per-component version was only
1.66–1.75× faster than HGP. Every call runs the full primitive loop, one
Boys call and one prefactor product per primitive quartet. An O 1s in
6-31G\* has 6 primitives, so (1s 1s|1s 1s) is 1296 primitive quartets, and
the per-component interface repeats that 3× for (ps|ss), 9× for (pp|ss),
81× for (pp|pp). The shell-level prototype ran it once: 4.9×.

**This is not the hoist that failed.** `docs/ERI_PERFORMANCE_SCOPE.md`
records a contract-once-per-shell-quartet hoist as 1.14–1.62× *slower*: it
kept the general VRR/HRR with its buffers and gathers. A closed-form kernel
has no buffer, gather or HRR, and its outputs live in registers.

**A profile share is a lower bound on what removing the work is worth.**
The `ShellPair` rebuild was 8.4 % of samples and worth 1.16–1.26×, because
its allocator and cache traffic were charged to other frames.

**The cheap fix often sits next to the hot frame.** HGP made one Boys call
per order through the loader's indirect call (6.8 % *self* in the wrapper).
OS had already switched to one `boys_vec` call; HGP and RI never had.

**Gates must be able to fail.** Every kernel gate was mutation-tested by
breaking each coupling term, sign and norm in turn. Two real bugs were
caught by gates rather than by review:
- *The flag-off fingerprint* caught C3's first version sending (ss|ss)
  blocks to the closed form with the flag off (`0 ≤ max_l` for max_l = 0),
  which would have silently changed every explicit engine.
- *The layout-coverage requirement* caught a fixture gap: water/6-31G\*
  cannot produce the (d,p)-ordered layouts, so water/cc-pVDZ joined the
  sweep.

**Write the prediction down first.** One table row misread the class data
(cc-pVDZ L = 3 as 50 % instead of 31 %); the measurement exposed it. A
counter inconsistency similarly exposed a misplaced instrumentation line in
L0 before any number was used.

## Landed

### C0 and the Boys-core fix (2026-10-17)

The class-share probe, above. Its first profile found 17.7 % of the build
in libm calls inside one branch of the out-of-tree Boys core. That was
fixed there and shipped in Boys library v1.1.0: ~1.13× on 4 H2O / 6-31G\*,
~1.37× on 8 H2O / 6-31G (1.21× / 1.64× linked directly; the loader's
indirect call cannot inline). `water_rhf_hgp_sto3g` (11 → 10 SCF
iterations) and `ethylene_rhf_ccpvdz_auto` (16 → 15) moved at the
convergence noise floor, energies unchanged, and were re-pinned.

### C1 — reuse the caller's ShellPairs (`c40a016c`)

`fused_fock_build` rebuilt a `ShellPair` (heap, `pow`, `exp` per primitive
pair) for every surviving component quartet, although every caller passes
the same pairs in `shell_pairs`. It now looks them up in an `nb × nb`
pointer table. Bitwise; 1.16–1.26×; helps every engine. On the way,
`planck-hgp-engine-smoke` and `planck-compute-2e` were found segfaulting
under ctest (they never called `require_boys_library()`) and fixed.

### C2, C3, C3.5 — closed-form shell kernels through total L ≤ 3 (`3c028544`, `4bbb2724`, `f0f7ceae`)

`low_l_shell_eri` (`src/integrals/low_l_eri.h`) treats a quartet's angular
momentum as up to three *raises* (center, axis), applied in center order.
Per primitive quartet, with θ_m = c K_AB K_CD 2√(ρ/π) F_m(T) from one
`boys_vec` call, d = PA/PB/QC/QD and w = W−P (bra) or W−Q (ket):

    A_i(m)  = d1_i θm + w1_i θ(m+1)
    B_ij(m) = d2_j A_i(m) + w2_j A_i(m+1) + [i=j] cpl(1,2; θ)
    C_ijk   = d3_k B_ij(0) + w3_k B_ij(1) + [k=i] cpl(1,3; A2_j) + [k=j] cpl(2,3; A_i)

A2_j is the second raise alone. cpl(y,x; X) is the Obara–Saika coupling:
(X(m) − (ρ/ζ)X(m+1))/2ζ with both raises on the bra, the same with η on
the ket, X(m+1)/2(ζ+η) when they straddle. At most 27 outputs, all in
registers.

- **Wiring.** `fused_fock_build` calls the kernel once per block, on the
  first component quartet that survives the filters, behind
  `int closed_form_max_l`. Only the Auto engine sets it (to 3), so explicit
  `engine os/hgp/rys` stay pure and the cross-engine comparators stay
  meaningful. Coulomb only. Filters, Schwarz, symmetry orbit and
  accumulation run per component, unchanged.
- **Feeding and readout.** The kernel gets the first-component pair of each
  shell. Outputs are scaled by a per-AO `norm_ratio` (d xx carries 1/√3,
  d xy 1; exactly 1.0 for s and p). Slots are read from each component's
  Cartesian exponents in raise order, so one rule covers every layout.
- **Speed, each over the previous step** (same binary, flag levels):
  C2 1.22–1.28× on 8 H2O / 6-31G (predicted 1.23×); C3 1.61× (predicted
  1.42×); C3.5 1.87× (predicted 1.6×). Beating the model at three raises
  answers C4's generalization question in its favour.
- **Gates.** `planck-hgp-engine-smoke` runs every component quartet with
  total L ≤ 3 of water in 6-31G, 6-31G\* and cc-pVDZ (7840 / 15736 / 43456
  quartets) through the kernel, fed exactly as the loop feeds it, against
  HGP: max relative difference 6.1e-16. Every reachable raise layout is
  required (15 on the sp fixtures, 31 on cc-pVDZ). Mutation-verified on
  every coupling term, the W−P sign, the axis readout and the norm ratio.
  max|ΔG| against HGP ≤ 4.4e-15; the flag-off fingerprints never moved.
- **Not bitwise against HGP**, by design (different arithmetic). Generalizing
  the kernel moved the L ≤ 1 results by about an ulp relative to C2.
- **Suites** after each step: core 79/79, `planck-compute-2e` OK, smoke
  43/45 (the two known rccsdt routing failures), output identical at 1 and
  4 threads on full-symmetry RHF/UHF direct, NH3/6-31G\* and
  ethylene/cc-pVDZ.

### H1 — hotspot round on the HGP kernel (`46dadc6a`)

All bitwise. One `boys_vec` call instead of MMAX+1 `boys` calls in
`hgp_vrr` and both RI seeds (1.10–1.12×). Accumulate the m=0 slice straight
from the VRR buffer instead of copying it out first, and drop the VRR
zero-fill, whose zeros were never read (1.04–1.06×).

### L2 — closed-form orbit dedup (`659883ab`)

`distinct_eri_orbit` searched each of the 8 permuted tuples against every
one already kept: 17.3 % of the 8 H2O build, 3.7× the `G` writes it fed.
Which tuples repeat follows from three equalities. With
`X = ({k,l} == {i,j})` as multisets, tuple *n* of the fixed order (ij|kl),
(ji|kl), (ij|lk), (ji|lk), (kl|ij), (lk|ij), (kl|ji), (lk|ji) is kept iff:

| n | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| keep iff | always | i≠j | k≠l | i≠j ∧ k≠l | ¬X | ¬X ∧ k≠l | ¬X ∧ i≠j | ¬X ∧ i≠j ∧ k≠l |

That holds for any (i,j,k,l), and it emits the same tuples in the same order,
so the result is bitwise. `planck-fock-accumulate` keeps the old search as
an oracle over all 1296 tuples in 0..5, order included (mutation-verified).
Best of three interleaved runs: 8 H2O / 6-31G 1.20× (predicted 1.15–1.2×),
4 H2O / 6-31G\* 1.09× (predicted ~1.07×), D2h naphthalene 1.11×. Core 79/79.

### L1 — skip blocks that cannot hold a canonical quartet

The ket loop ran over all shell-group pairs, with canonical order (k ≥ i,
then l ≥ j) enforced only per component. Groups are AO-contiguous and
`group_pairs` is ordered by `.a`, so every ket block with
`group_pairs[ket].a < group_pairs[bra].a` has k < i for every component and
contributes nothing: ~49 % of the blocks surviving Schwarz, and about half
of all component iterations. The ket loop now starts at
`first_pair_of[group_pairs[bra].a]`, a table built once per call. The
skipped blocks contributed nothing and the rest are visited in the same
order, so the result is bitwise; the fingerprints are identical on all five
fixtures, including naphthalene with `sym_ops`. Best of three interleaved
runs:

| fixture | before | L1 | gain | predicted |
|---|---|---|---|---|
| 8 H2O / 6-31G | 0.232 s | 0.219 s | **1.06×** | 1.05–1.07× |
| 4 H2O / 6-31G\* | 0.532 s | 0.526 s | ~1.01× | ~1.02× |
| naphthalene / 6-31G, D2h | 1.499 s | 1.473 s | ~1.02× | — |

The d-basis gain is within noise; it lands anyway, as a bitwise change
that makes the loop do less. `planck-fock-accumulate` OK, core 79/79, smoke
43/45 (the known rccsdt failures).

### L3a — reject non-representatives early on the symmetry path

With integral symmetry on, the fused loop receives `sym_ops`, at most 8
D2h-subgroup signed AO permutations. It used to call `build_quartet_orbit`
for every canonical component quartet: allocate a vector, apply every
operation, deduplicate, sort. It then computed the ERI only if the quartet
was the orbit's representative (the lexicographic minimum), about 1 in 8
under D2h. Everything else was thrown away.

`is_orbit_representative` (`src/integrals/quartet_orbit.h`) answers the
representative question directly. A quartet is not the representative as
soon as one operation maps it, after canonicalization, to a
lexicographically smaller quartet. The check returns at the first such
image, with no allocation, dedup or sort. The loop calls it first and builds
the full orbit only for representatives. The accumulated quartets are
unchanged: a symmetry-forbidden non-representative is skipped either way,
and a forbidden representative is still caught by the full build.

- **Gate.** `planck-fock-accumulate` compares the pre-check with
  `build_quartet_orbit`'s own answer on every canonical quartet of a
  synthetic nb = 6 group whose operations permute AOs as well as flipping
  signs. Sign-only operations would make every quartet its own
  representative, so the check would test nothing. That is 231 quartets, 85
  representatives; the test fails if the fixture has none, or only
  representatives. Mutation-verified: dropping the canonicalization fails at
  (0,1,3,3). All the symmetric fused-vs-two-phase checks pass, and the G
  fingerprints are identical on all five probe fixtures, naphthalene with
  `sym_ops` included.
- **Speed** (best of three interleaved runs): D2h naphthalene / 6-31G
  1.454 → 0.855 s, **1.70×**. Fixtures without symmetry are unchanged
  (8 H2O / 6-31G 0.220 → 0.217 s).
- **Suites.** Core 79/79. Smoke 43/45 (the known rccsdt routing failures).
  Output is identical at 1 and 4 threads for the C2v symmetry reference
  case, triplet water UHF, full-symmetry direct water and ethylene/cc-pVDZ.

## Open

### L3b — no heap for representatives

Representatives still get their orbit from `build_quartet_orbit`, which
allocates a `std::vector` per call. The orbit has at most `sym_ops.size()`
≤ 8 elements, so it fits a fixed-capacity stack array, built in the same
order and sorted the same way. The vector-returning function stays for
other callers. After L3a only representatives (about 1 in 8 canonical
quartets) pay the allocation, so the remaining gain is smaller than the
original ~25–30 % heap share. Re-profile the symmetric build to size it
before doing the work. Gate: the naphthalene fingerprint with `sym_ops`.

### L4 — per-block low-L readout table

Each component recomputes its slot and four norm-ratio products, though both
depend only on the component indices within the block. No visible cost of its
own in the L0 attribution; only if a later profile shows it.

### C4 — total L = 4 (d bases)

After C3.5, HGP's remaining work is total L = 4 (all of it on sp bases) and
L ≥ 5 on d bases. C3.5 showed that three raises still beat the ~5× model.
So L = 4 can be the same chain with four raises (81 outputs, one more
level), code-generated, since four nested levels by hand invite the
coupling bugs the mutation tests exist for. It **must not** reintroduce the
failed hoist's buffers or gathers. Take a fresh profile first: on sp bases
the loop is the larger item (L series), on d bases HGP at L ≥ 4.

### C5 — screened kernels

The closed forms need the same screened scaling as `hgp_vrr`: T ×
`boys_scale`, prefactor × `prefactor_scale`, W−P/W−Q × `screen.rho / rho`.
The straddling coupling is exactly the `inv_2_delta` term behind the HGP
screened-kernel bug (`vault/Gotchas/HGP Screened inv_2_delta`), so it also
takes × `boys_scale`. Until a screened sweep against OS passes on a mixed
bra/ket-AM fixture, screened kernels stay on HGP.

## Out of scope

- **A local accumulation buffer per block.** It would change summation order
  (not bitwise) and touch the threading determinism rules, and the `G`
  writes are only 4.7 % of the build.
- **Gradients.** Derivative ERIs are a separate kernel family (Open Work
  Phase D).
- **Primitive screening.** Skipping primitive quartets with negligible
  `K_ab K_cd` attacks the primitive-loop floor directly and is independent of
  closed forms. Deep contractions are the target: cc-pVDZ (ss|ss) ran about
  5 µs per quartet before C2.
- **Spherical.** The direct spherical path back-projects to Cartesian and
  goes through the same fused loop.
- **`water_rhf_spherical_direct_631gd` thread drift.** Its last residual
  digits differ between 1 and 4 threads (energy identical to all 10 digits).
  It predates this work and sits outside the fused loop. Not chased.
