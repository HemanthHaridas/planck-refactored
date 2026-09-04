# FCIQMC Serial Performance

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Why was FCIQMC slow, and what is left after the serial fixes?**

## Short answer

Opened as "thread FCIQMC" and answered mostly without threading it. Three serial
changes took N2/STO-3G from 71.63 s to 13.74 s — 5.21x — every one of them bitwise
identical, and threading is still unbuilt. The scope's own ordering rule is what
produced that: fix the allocation before threading, because threading an
allocation-bound loop parallelizes `malloc` contention. Following it exposed two
further serial defects that threading would have hidden rather than fixed.
FCIQMC has zero `#pragma omp` to this day.

## Where the logic lives

- `src/post_hf/ci/fciqmc.cpp` — `propagate_stochastic` (the spawn loop to thread)
- `src/post_hf/ci/fciqmc.h` — `WalkerPopulation` (the walker map)
- `src/post_hf/fciqmc_driver.cpp` — `make_ops` (the memoized diagonal)
- `RandomSource::derive` — RNG shard derivation
- `h2_fciqmc_threads1` / `h2_fciqmc_threads4` — the invariance gate
- `docs/FCIQMC_RESEARCH_SCOPE.md` §6 — the decided threading policy
- `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` — the precedent, with its traps

## What invariants matter

### 1. A profile share is a lower bound on what removing that work is worth, not an estimate of it

Three for three: T1 measured 1.76x against an Amdahl cap of 1.42x, T4 measured
2.61x against 1.83x, and the FCI sigma build's identical allocation fix returned
4.8x against ~2.1x. Removing work also removes the cache pressure and bookkeeping
attributed to other frames, which a profile share alone does not capture.

Design rule:

- Treat a measured profile share as a floor on the payoff of removing that work,
  not a ceiling — do not dismiss a change because the share looks small.

### 2. Judge a change by the profile share it removes, not only by the clock

The L1-norm binning bought only 1.13x on the clock and would look like a dud —
but it took that work from 17.0 % to 0.9 % of the profile, and it was 17 % of an
already-reduced total.

Design rule:

- When a change looks marginal on wall-clock alone, check what fraction of the
  profile it removed before writing it off.

### 3. A model of a workload is not a measurement of it

The churn model for the T4 diagonal memo predicted hundreds of MB and demanded an
eviction policy; measurement showed the real occupied set is nearly static — 1820
entries and 85 KB.

Design rule:

- Measure the actual workload (a temporary env-gated probe, a microbenchmark)
  before designing around a theoretical worst case.

### 4. Profile parsing is a real source of wrong numbers

`sample` emits a call tree with `+ ! : |` prefixes and inclusive counts. A naive
`^\s*(\d+)` regex matched almost nothing and reported `slater_condon_element` at
0.2 % when it was the dominant frame at 53.1 %. Computing an inclusive share by
depth-matching sibling nodes then gave 3909.9 %, and taking the single largest
root gave 100.0 % because the sample window had caught only one phase.

Design rule:

- Build real parent links from the depth stack, exclude nodes nested under
  another node of the same function, and sample the whole run — do not trust a
  quick regex over `sample` output.

### 5. A build in flight is not pinned to the working tree

Stashing a change while its build was still running made `make` compile a file
no longer present in the tree and report `MAKE_EXIT=0` — a meaningless green
that cost a 25-minute rebuild.

Design rule:

- Kill any in-flight build before stashing or otherwise mutating the working
  tree it is compiling from.

### 6. Threading the spawn loop requires binning by parent, never by output or thread

The design for T2 (not yet built) is decided and gated ahead of the code: partition
the parents by `hash(parent) % kBins`, each thread accumulating into its own bin,
merged in fixed bin order. Three properties of the loop force this shape:

1. `next` is the only mutable shared state — every write goes through
   `WalkerPopulation::add`, which is `_walkers[det] += w` on an `unordered_map`.
2. The initiator rule reads `population`, not `next` — a pure read on a `const&`,
   so it needs no protection.
3. `RandomSource &rng` is a single mutable object shared by every parent. This is
   the sharpest hazard: threading the loop as written is a data race on generator
   state, not merely a determinism question. Each bin must take its own stream
   from `RandomSource::derive(bin_index)` — verified to be a pure function of the
   seed and index, independent of how many shards are taken. Deriving per
   *thread* instead of per *bin* would reintroduce thread-count dependence.

One structural advantage over the FCI sigma build: that build binned by index
range, so a determinant could migrate between bins when the bin count changed.
Binning by `hash(parent) % kBins` maps a parent to the same bin regardless of bin
count, so invariance survives changing `kBins`.

Design rule:

- Do not use `omp atomic` on the walker map, or any completion-order reduction —
  that is the DFT-grid jitter defect and would silently break the invariance the
  gate asserts.
- Do not tie `kBins` to thread count, and do not use `schedule(dynamic)` in a way
  that changes which bin gets which parent.
- Do not accept "energies agree to 1e-10" as a substitute for bitwise equality.
  Every other parallel path in this codebase is bitwise invariant and the gate
  is `atol = 0.0`; a tolerance would hide exactly the reduction-order defect the
  design exists to prevent.
- Do not thread the estimators — `projected_energy` is 1.9 % of runtime and not
  worth the determinism risk.
- The T4 memoized-diagonal map is not thread-safe as written. It must be
  prefilled before the parallel region, or made per-bin — never a shared
  mutable map — though it cannot change any resulting value either way.

## What was measured

Three landed serial changes, each verified bitwise identical on `n2_fciqmc_sto3g`
and `h2_fciqmc_sto3g` (every printed line, not only the energies):

| step | change | N2, 1 thread | share removed |
|---|---|---|---|
| T1 | fixed-capacity orbital lists | 71.63 -> 40.66 s (**1.76x**) | malloc 29.5 % -> 1.4 % |
| T4 | memoize the diagonal | 40.66 -> 15.57 s (**2.61x**) | `slater_condon` 53.1 % -> 6.6 % |
| — | bin the L1 norm | 15.57 -> 13.74 s (**1.13x**) | `ordered_l1_norm` 17.0 % -> 0.9 % |

**T1 — the spawn path allocated five heap vectors per attempt.**
`occupied`/`virtuals` returned `std::vector<int>` by value, four per call, plus a
vector for the excitation-class list. At ~14 000 occupied determinants x 30 000
iterations that is on the order of 1e9 allocations for values that never exceed
31 entries. Replaced with a fixed-capacity `OrbitalList` (`std::array<int,32>` +
count) and a `std::array<...,5>` for the classes. The capacity is a bound, not a
guess: `build_all_mo_ci_setup` rejects `n_act > kMaxPackedSpatialOrbitals` =
`(64-1)/2` = 31 before either FCI or FCIQMC runs. `enumerate_connections`
correctly keeps its `std::vector` return — verified rather than assumed:
`propagate_stochastic` calls `draw_excitation` and never `enumerate_connections`,
whose size is genuinely variable and which runs ~30 000 times against the spawn
path's ~1e9.

**T4 — the diagonal was recomputed 37 745 times per determinant.** `H_ii` is a
pure function of the determinant (`h_eff` and `ga` are built once and never
mutated), but the spawn loop asked for it once per parent per iteration. The
`n_diff == 0` branch is O(n_act^2) against O(n_act) for a single excitation, so
it was ~86 % of `slater_condon_element`'s work by operation count. Memoized in
the `ops.diagonal` lambda: an `unordered_map<DetKey,double>` held by
`shared_ptr`, so no signature changes and no new state threaded through the
propagators. Both assumptions were measured before building: a temporary
env-gated probe counted 68 696 226 calls over 1820 distinct determinants —
37 745x reuse, 99.9974 % hit rate, 85 KB table. A microbenchmark on production
shapes measured the memo 75x faster at `n_act = 10` and 226x at 20, the gap
widening because recompute is O(n_act^2) against an O(1) probe.

**The L1 norm sorted the whole population, 50 000 times.** `ordered_l1_norm`
built a vector of the entire population and `std::sort`ed it by determinant key
purely to fix summation order. The sort was never needed for its ordering: the
summands are `|w|`, all non-negative, so the only order-dependence is
floating-point reassociation, and what the contract requires is a canonical
*partition* of the terms, not a sorted one. Binning on
`DetKeyHash{}(det) % 64` into fixed accumulators summed in bin order gives that
in O(n) — a determinant lands in the same bin whatever order it was inserted in.
`kNormBins` is a fixed constant deliberately: two populations with identical
contents must produce identical bins, so the partition may not depend on
capacity, load factor, or insertion history.

## What is left: T2, threading the spawn loop

Ceiling **2.81x at 4 threads** on the current serial baseline
(`propagate_stochastic` is 85.8 % of `run_fciqmc` inclusive; 1.74x / 2.81x / 4.05x
at 2/4/8).

The ceiling has moved twice, both times downward in absolute terms: 3.75x when
sized against the post-T1 profile, 2.19x after T4 removed work from inside the
loop T2 threads, then back up to 2.81x once the L1-norm sort left the serial
tail. Against a 71.63 s baseline 3.75x was ~19 s saved; against 13.74 s, 2.81x is
~8.8 s. Re-measure before building, not from this number — every landed step has
changed it.

Verify, in this order, once T2 is built:

1. CPU > 100 % at 4 threads — one `ps` call, catches an inert pragma before any
   timing is read.
2. Bitwise identical at `OMP_NUM_THREADS` = 1/2/4/8. Extend the existing gate to
   8 and add an N2 pair — a 4-determinant space may not exercise the partition
   at all.
3. All FCIQMC and FCI regression cases green.
4. Speed, against a freshly measured ceiling.

**T3 — the merge, only if it shows up.** The fixed-order merge is serial. At
`kBins = 64` and a few thousand occupied determinants it should be negligible;
the sigma build's equivalent measured 0.1 %. Profile after T2 and act only if it
is material — do not pre-optimize it.

## Validation strategy that should remain in place

- `n2_fciqmc_sto3g` and `h2_fciqmc_sto3g` bitwise-identical regression checks
  across every serial change landed
- `h2_fciqmc_threads1` / `h2_fciqmc_threads4` at `atol = 0.0`, gating the T2
  design before the threading code exists
- Re-profiling with `sample` (parsed correctly, per invariant 4) before trusting
  any new ceiling estimate

## Remaining architecture concern

Nothing currently needs T2. The method is validated but unused —
`FCIQMC_RESEARCH_SCOPE.md` Q1 remains unanswered in practice, and the largest
active space anywhere in the tree is `nactorb 6`. The three serial changes stand
on their own; T2 is worth building when a target exists, and the ceiling should
be re-measured then rather than taken from this document.
