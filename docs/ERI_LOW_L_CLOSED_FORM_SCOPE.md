# Closed-Form Low-Angular-Momentum ERI Kernels — Scope

Status: **in flight.** Landed: C0, C1, C2–C3.5, H1, L1, L2, L3a, L3b, the
A0 split, P1 (negligible primitive pairs) and L3c (block-level orbit
rejection). Open, in order: P2 (a policy decision), C4a, C4b, the L ≥ 5
decision, C5, and the deep-contraction architecture steps D0–D3. When those
land or are abandoned, rewrite this as an answer.

**Question:** How much faster can the direct (memory-direct, fused) Fock
build get by evaluating low-angular-momentum shell quartets in closed form,
and what else in the build stands in the way?

## Short answer

Closed forms help only at **shell** granularity. A closed form that returns
one Cartesian component per call is about 1.7× faster than HGP on its class.
One that returns every component of a shell quartet from a single primitive
pass is about 5× faster. Once the kernels were fast, the work around them
became the cost: the loop (L series), symmetry bookkeeping, and primitive
quartets that contribute nothing (P series).

Each step is a ratio over the build *with all previous steps* (one thread,
production `engine auto`), so a row multiplies:

| fixture | C1 | C2 | H1 | C3 | C3.5 | L2 | L1 | P1+L3c\* | total |
|---|---|---|---|---|---|---|---|---|---|
| 8 H2O / 6-31G | 1.26× | ~1.25× | 1.16× | 1.61× | 1.87× | 1.20× | 1.06× | 1.14× | **~7.9×** (1.50 → 0.19 s) |
| 4 H2O / 6-31G\* | 1.16× | ~1.04× | 1.16× | 1.13× | 1.26× | 1.09× | ~1.01× | 1.03× | **~2.3×** |
| 1 H2O / cc-pVDZ | 1.07× | 1.10× | — | 1.23× | 1.43× | — | — | 1.00× | — |

\* landed together; measured as one step.

Symmetric molecules gain more. On D2h naphthalene / 6-31G, L3a, L3b and L3c
took the symmetric build from 1.454 to 0.605 s (2.4×, with P1). Integral
symmetry now pays 3.9× (2.366 s without `sym_ops`), against 1.9× before and
an ideal of ~8×.

The basis decides the payoff. The closed forms cover total L ≤ 3, which is
almost all of a pure sp build but only part of one with d shells: on
6-31G\*, HGP is still ~¾ of the build. These are Fock-build numbers, not SCF
wall time. Explicit engines get only the engine-independent steps (C1, H1,
L1, L2, L3, P1). **Against the ~200× libcint gap this is a large step, not a
fix.**

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
- **Attribute before predicting.** Inlined helpers hide inside their
  caller's self time. Profiling builds that mark them `noinline`, with
  temporary counters, split the time before anything changes. They are
  never committed.

## Where the time goes now

Measured on `a1f99d7a` (after L3b, before P1/L3c), with the loop's helpers
split out by the A0 `noinline` build. Self time as a share of the build:

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

Counters per build:

| | 8 H2O / 6-31G | 4 H2O / 6-31G\* | naphthalene D2h | 1 H2O / cc-pVDZ |
|---|---|---|---|---|
| ket blocks / surviving Schwarz | 3.52 M / 501 k | 347 k / 150 k | 2.49 M / 1.91 M | 3.4 k / 3.4 k |
| component iterations | 2.23 M | 2.04 M | 14.4 M | 68 k |
| representative checks / passing | — | — | 13.4 M / 3.35 M | — |
| closed-form blocks / values | 494 k / 1.91 M | 133 k / 749 k | 478 k / 1.72 M | 2.7 k / 22 k |
| HGP values | 93 k | 889 k | 203 k | 31 k |

Primitive quartets actually executed, against the prefactor bound
|base| = |c_ab K_ab c_cd K_cd| · 2√(ρ/π), the (ss|ss) value at F₀ = 1.
Closed-form quartets are counted once per block, HGP quartets once per
component:

| fixture | closed form < 1e-18 | HGP < 1e-18 | closed form < 1e-12 | HGP < 1e-12 |
|---|---|---|---|---|
| 8 H2O / 6-31G | 23.4 % | 18.7 % | 39.5 % | 29.3 % |
| naphthalene / 6-31G | 36.9 % | 19.8 % | 52.4 % | 33.9 % |
| 4 H2O / 6-31G\* | 21.5 % | 6.6 % | 38.1 % | 18.7 % |
| 1 H2O / cc-pVDZ | 5.0 % | 0.1 % | 8.1 % | 0.4 % |

HGP time left at total L = 4 (class share of the HGP-only build, against
the post-L3b build):

| fixture | L = 4 HGP time | share of build | L ≥ 5 share after C4 |
|---|---|---|---|
| 8 H2O / 6-31G | 0.080 s, (pp\|pp) only | 37 % | 0 |
| 4 H2O / 6-31G\* | 0.180 s | 35 % | ~68 % |
| 1 H2O / cc-pVDZ | 0.012 s | 44 % | ~61 % |
| naphthalene D2h | 0.219 s, (pp\|pp) only | 31 % | 0 |

What it says:
- **sp bases:** the cost is spread over HGP at L = 4, Boys, the kernel and
  the loop. C4 removes the HGP part.
- **d bases:** HGP dominates, at L = 4 and above.
- **Symmetric builds:** the loop and the representative check (30 %) were
  the largest item. L3c takes part of it.
- **Extended molecules:** a fifth to a third of primitive quartets do
  nothing. P1 removes the pair-level part; P2 is the rest.

## Findings

**The floor is the primitive loop, not the recurrence.** The (ss|ss) closed
form has no recurrence left to remove, yet a per-component version was only
1.66–1.75× faster than HGP. Every call runs the full primitive loop: one Boys
call and one prefactor product per primitive quartet. An O 1s in 6-31G\* has
6 primitives, so (1s 1s|1s 1s) is 1296 primitive quartets, and the
per-component interface repeats that 3× for (ps|ss), 9× for (pp|ss), 81×
for (pp|pp). The shell-level prototype ran it once: 4.9×.

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

**Write the prediction down first.** One table row misread the class data
(cc-pVDZ L = 3 as 50 % instead of 31 %); the measurement exposed it. A
counter inconsistency exposed a misplaced instrumentation line before any
number was used.

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
Per primitive quartet, with θ_m = c K_AB K_CD 2√(ρ/π) F_m(T) from one
`boys_vec` call, d = PA/PB/QC/QD and w = W−P (bra) or W−Q (ket):

    A_i(m)  = d1_i θm + w1_i θ(m+1)
    B_ij(m) = d2_j A_i(m) + w2_j A_i(m+1) + [i=j] cpl(1,2; θ)
    C_ijk   = d3_k B_ij(0) + w3_k B_ij(1) + [k=i] cpl(1,3; A2_j) + [k=j] cpl(2,3; A_i)

A2_j is the second raise alone. cpl(y,x; X) is the Obara–Saika coupling:
(X(m) − (ρ/ζ)X(m+1))/2ζ with both raises on the bra, the same with η on
the ket, X(m+1)/2(ζ+η) when they straddle. At most 27 outputs, all in
registers.

- **Wiring.** Called once per block, on the first component quartet that
  survives the filters, behind `closed_form_max_l`. Only Auto sets it (to
  3), so explicit `engine os/hgp/rys` stay pure and the cross-engine
  comparators stay meaningful. Coulomb only.
- **Feeding and readout.** The kernel gets each shell's first-component
  pair. Outputs are scaled by a per-AO `norm_ratio` (d xx 1/√3, d xy 1;
  exactly 1.0 for s and p). Slots are read from the Cartesian exponents in
  raise order, so one rule covers every layout.
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
  routing failures), output identical at 1 and 4 threads on the
  full-symmetry direct cases.

### H1 — hotspot round on the HGP kernel (`46dadc6a`)

All bitwise. One `boys_vec` call instead of MMAX+1 `boys` calls in
`hgp_vrr` and both RI seeds (1.10–1.12×). The m=0 slice is accumulated
straight from the VRR buffer, and the VRR zero-fill, whose zeros were never
read, is gone (1.04–1.06×).

### L2 — closed-form orbit dedup (`659883ab`)

`distinct_eri_orbit` searched each of the 8 permuted tuples against every
one already kept: 17.3 % of the 8 H2O build, 3.7× the `G` writes it fed.
Which tuples repeat follows from three equalities. With
`X = ({k,l} == {i,j})` as multisets, tuple *n* of the fixed order (ij|kl),
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

### A0 — split the loop's self time (2026-10-20)

The profiling build and census behind "Where the time goes now". It
decided three things. L4 (a per-block readout table) is dropped, since the
readout is ≤ 3.6 %. The symmetric item is L3c. Primitive screening moved in
from out of scope, as P1 and P2. Nothing from A0 itself is committed.

### P1 — drop negligible primitive pairs

The below-1e-18 band in the census is a **pair** property: tight
primitives on distant atoms, where exp(−αβR²/ζ) has underflowed. One water
has almost none. The `ShellPair` constructor (`src/base/types.h`) now drops
a primitive pair with |c_a c_b N_a N_b n_A n_B · prefactor| < 1e-24, about
e^−55 for a unit pair. It sits in one place and covers every engine, the 1e
integrals and the gradients. Every consumer iterates
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

### L3c — reject whole blocks on the symmetry path

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
  and UHF, in STO-3G and 6-31G\*: max|ΔG| ≤ 1e-14. The test now links
  `symmetry/integral_symmetry.cpp`. Mutation-verified with `< A` → `!= A`,
  which is harmless under sign flips but rejects blocks that do hold
  representatives once shells move. The new gate fails on every engine
  (max|ΔG| 0.14–0.46), while all 16 old sign-flip checks still pass. That
  shows the new gate is the only one that sees this class of bug.
- **Production path.** For HF, the fused loop gets `sym_ops` only when SAO
  blocking is off (PCM, or linear point groups); otherwise full symmetry
  takes over. CO2 / 6-31G\* direct (D∞h, 8 ops) converges to the
  symmetry-off energy in all 10 printed digits. The symmetric Fock build is
  0.059 s per iteration against 0.335 s without symmetry.
- **Suites:** core 79/79, smoke 43/45 (the known rccsdt failures).

## Open, in order

### P2 — quartet-level primitive screening (a policy decision)

After P1, another ~20 % of primitive quartets on water clusters have |base|
between 1e-24 and 1e-12. Skipping them in the closed-form and HGP primitive
loops when |base| < ε, with the Schwarz table held exact, measured:

| fixture | ε = 1e-14 | ε = 1e-12 | max\|ΔG\| at 1e-14 / 1e-12 |
|---|---|---|---|
| 8 H2O / 6-31G | 0.194 → 0.171 s, 1.13× | 1.37× | 4.0e-12 / 3.3e-10 |
| 4 H2O / 6-31G\* | 0.504 → 0.471 s, 1.07× | 1.18× | 4.9e-12 / 3.7e-10 |
| naphthalene D2h | 0.598 → 0.520 s, 1.15× | 1.22× | 4.8e-12 / 4.8e-10 |

ΔG is ~300–500 × ε on every fixture. The proposed rule is ε = 1e-4 ·
`tol_eri` (1e-14 at the default), which keeps the primitive error about
100× below what Schwarz screening at `tol_eri` already admits. Not landed,
because:
- **It is not bitwise.** Regression energies move at ~1e-12, and iteration
  counts at the convergence floor may move, as they did for the Boys fix.
- **ε must reach the kernels.** `low_l_shell_eri` can take it from the fused
  loop, but the OS/HGP/Rys `eri_elem` callable is a shared engine signature.
- **Schwarz safety.** `schwarz_table_*` must pass ε = 0 (see Findings). The
  gate: Q built with screening on equals the exact Q.

If a non-bitwise default is accepted, do the closed-form kernel first (one
parameter, most of the sp gain), then HGP behind the same parameter.

### C4a — four-raise prototype in the probe (nothing lands)

Four raises give 3⁴ = 81 *ordered* outputs. That is exact for (pp|pp), the
only L = 4 class on sp bases. d layouts are redundant: (ds|ds) and (dd|ss)
need 36 distinct components, (ps|pd) 54, so up to 2.25× of the arithmetic
is wasted. Prototype (pp|pp) and (ds|ds) with the kernel's shell-level
feed, timed per class against HGP. Predicted: (pp|pp) ≥ 4×, (ds|ds) 2–3×.
- Both ≥ 2×: one generated four-raise chain for every L = 4 layout.
- d layouts < 2×: C4 covers (pp|pp) only, and d at L = 4 waits for the
  L ≥ 5 design.

### C4b — the generated four-raise kernel

Generate the fourth level rather than hand-write it. Outputs stay in
registers, with **no** buffers or gathers (the failed hoist). Wiring is
unchanged: `closed_form_max_l = 4` from Auto, explicit engines pure.
Predicted at 4–5× on the class, over L3b. Re-derive from a fresh probe once
P1 and L3c land, since both shift the shares:

| fixture | predicted | ceiling (class free) |
|---|---|---|
| 8 H2O / 6-31G | 1.38–1.42× | 1.59× |
| naphthalene D2h | 1.31–1.33× | 1.45× |
| 4 H2O / 6-31G\* | ~1.3× (less if C4a limits d) | 1.53× |
| 1 H2O / cc-pVDZ | 1.5× | 1.77× |

Gates as for C3.5: a per-quartet sweep against HGP on water in 6-31G,
6-31G\* and cc-pVDZ, with every reachable four-raise layout required;
mutation tests on each coupling term, sign and norm; unchanged flag-off
fingerprints; max|ΔG| ≤ ~5e-15; core, smoke and the thread-identity cases.
The generator must also reproduce the hand-written three-raise outputs for
every L ≤ 3 layout before its new level is trusted.

### After C4 — L ≥ 5 (a decision, not a step)

d bases will still be ~60–70 % HGP at L ≥ 5. Five raises is 243 ordered
outputs, more than the registers hold, and the redundancy grows with every
d shell, so the raise chain ends at L = 4. Candidates, each needing a
C0-style prototype first:
- per-class generated VRR over *distinct* Cartesian components, unrolled,
  with no runtime-sized scratch;
- P2 extended to HGP, if accepted. The census found almost nothing to screen
  on compact d-basis molecules (cc-pVDZ water: 0.4 % of HGP quartets below
  1e-12), so it pays on extended systems, not on deep contractions as such.

Choose from a fresh profile after C4b. D1 below is the general form of
the first candidate.

### C5 — screened kernels

The closed forms need `hgp_vrr`'s screened scaling: T × `boys_scale`,
prefactor × `prefactor_scale`, W−P/W−Q × `screen.rho / rho`, and the
straddling coupling × `boys_scale` (the `inv_2_delta` term behind
`vault/Gotchas/HGP Screened inv_2_delta`). Until a screened sweep against
OS passes on a mixed bra/ket-AM fixture, screened kernels stay on HGP. It
pays only for range-separated hybrids, so it ranks after C4.

### D — deep-contracted bases: architecture steps

Deep contraction (cc-pVXZ, ANO) multiplies the primitive loop by two
repetitions that the architecture, not the arithmetic, causes. Counted for
water over all shell quartets, with no screening (so ratios, not timings):

| primitive quartets executed | cc-pVDZ | cc-pVTZ |
|---|---|---|
| per Cartesian component (HGP today) | 1.6 M | 16.4 M |
| once per segmented shell quartet (the closed-form kernel) | 330 k | 1.15 M |
| once per general-contraction shell quartet | 69 k | 236 k |
| repetition from per-component calls | 4.9× | 14.3× |
| repetition from segmented storage | 4.8× | 4.9× |

- **Per-component calls.** The engine callable
  `eri_elem(spAB, spCD, lAx…lDz)` returns one component, and HGP's
  `_contracted_eri_elem` reruns the whole primitive loop (Boys, VRR,
  contraction) for each one, with no caching. The closed-form kernel
  removed this for total L ≤ 3 only.
- **Segmented storage.** The GBS reader stores a general contraction as
  separate shells. Oxygen's 1s and 2s in cc-pVDZ are two 9-primitive shells
  over the **same nine exponents**, and the third s shell is the last of
  them. Every primitive integral over those exponents is computed once per
  contracted function.

Steps, in order:

#### D0 — measure the two repetitions after screening (nothing lands)

An A0-style probe on water / cc-pVDZ and cc-pVTZ. Count executed primitive
quartets against unique ones (per shell quartet, and per general-contraction
quartet) after Schwarz and P1. That gives what D1 and D2 are each worth
before either is designed. The unscreened counts above are an upper bound.

#### D1 — a shell-quartet block interface for every engine

The fused loop's engine interface becomes a block call that returns all
components of a shell quartet, with the primitive loop run once.
Prerequisite: retire the per-AO `ShellPair` adapter (Open Work, shell-pair
granularity Phase E), so the loop holds shell-level pairs. C4b and the
L ≥ 5 kernels then become implementations of this one interface, class by
class. Constraint from history: the earlier contract-once hoist was
1.14–1.62× *slower* (1.23× on cc-pVDZ), because of snapshot copies and
per-component gathers (`docs/ERI_PERFORMANCE_SCOPE.md`). A block kernel must
keep intermediates in registers or fixed-size stack storage, as the closed
forms do, and must be measured per class against the per-component path
before it is dispatched. Gates: the per-quartet sweep against the
per-component engine on every reachable layout, flag-off fingerprints, and
the thread-identity cases.

#### D2 — general contraction in `Shell` and `ShellPair`

- **`Shell`:** holds one exponent set and an n_prim × n_contracted
  coefficient matrix. The reader merges shells with the same L on the same
  atom whose exponents form a subset: O s becomes 9 × 3 in cc-pVDZ, 10 × 4 in
  cc-pVTZ.
- **`ShellPair`:** built once per exponent-set pair.
- **Kernels:** compute the K⁴ primitive block once, then contract it in
  quarter steps (K⁴ → n·K³ → …). Contracting each primitive straight into all
  n⁴ outputs would cost 256 multiply-adds per primitive for cc-pVTZ O s
  quartets, about what the Boys call saves.

This is the most invasive step. `ContractedView._index`, normalization (Nc is
folded per contraction), the 1e integrals, the gradients, the spherical
transform and every engine assume one contraction per shell. It needs D1's
block interface, and it pays only for generally contracted bases (none of
the Pople sets). Gate: G and energies unchanged against the segmented path
on cc-pVDZ / cc-pVTZ inputs (not bitwise; summation order changes), plus the
PySCF-anchored spherical gates.

#### D3 — RI-JK pair factors (B0 in `docs/ERI_FOCK_BLAS_SCOPE.md`)

Exact four-centre integrals stay K⁴ per shell quartet whatever D1 and D2 do.
Large cc-pVTZ/QZ runs are made fast in practice by density fitting.
Planck's RI-JK recomputes its density-independent pair factors twice per
iteration, ~61 % of the RI Fock build. Owned and scoped by the BLAS doc as
B0; listed here because, for big deep-contracted jobs, it is probably the
cheapest large win. Independent of D0–D2.

Not on this list: primitive screening (P2) pays on extended systems, not
compact ones (0.4 % of HGP quartets below 1e-12 on one cc-pVDZ water).
Faster per-primitive work (vectorizing the primitive loop, a cheaper Boys
call path) is bounded at maybe 1.5–2× and is not specific to deep
contraction.

## Ruled out

- **L4, a per-block readout table.** The readout is ≤ 3.6 % of the build
  (A0).
- **Forcing `accumulate_term` / `distinct_eri_orbit` inline.** They are weak
  symbols called through dyld stubs, but the stubs cost < 1 %.
- **The scalar `boys(0, T)` in the (ss|ss) branch.** It is 6.6 % on 8 H2O,
  but most of that is the Boys evaluation itself. Only the indirect-call
  overhead is avoidable, worth a few percent at most.

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
  the kcal/mol energy (all 10 Hartree digits identical). This is the same
  defect the FCI sigma-build work recorded: the *partition* of work into
  accumulators must be fixed, not just the summation order. It predates this
  work (`water_rhf_spherical_direct_631gd`'s drift is the same thing; it
  does go through the fused loop) and is not chased here. The symmetric
  identity checks earlier in this doc held because full-symmetry runs bypass
  the fused loop.
