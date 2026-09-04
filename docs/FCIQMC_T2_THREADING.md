# FCIQMC Spawn-Loop Threading

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Can the FCIQMC spawn loop be threaded without breaking bitwise thread-count invariance, and what speedup does that buy?**

## Short answer

The FCIQMC spawn loop (`propagate_stochastic`, `src/post_hf/ci/fciqmc.cpp`) is
threaded, bitwise thread-count invariant (`OMP_NUM_THREADS` = 1/2/4/8,
`atol = 0.0`), and gives a real but modest 1.57x at 4 threads — well short of
the ~4x Amdahl ceiling the serial structure allows. The gap is fully diagnosed:
at 4 threads, the fixed-bin-order merge alone costs more per call (48.2us) than
the entire threaded region (78.5us), and together with the still-serial
RNG-shard construction (25.9us) and bin-clear pass (15.3us), serial-outside-
the-pragma work now exceeds the parallel region — structurally capping the
achievable speedup regardless of thread balance. Both remaining levers were
investigated and neither is worth taking: merging via reuse was tried and
reverted (net slower, plus a reordering hazard caught only by bisection), and
the RNG-shard construction cost is dominated by unavoidable seeding work, not
container overhead (~4% available, judged not worth a code change).

The serial work that preceded this (5.21x, no threads) is answered in
`docs/FCIQMC_SERIAL_PERFORMANCE.md`.

## Where the logic lives

- `src/post_hf/ci/fciqmc.cpp` — `propagate_stochastic`, the loop that was threaded
- `src/post_hf/ci/fciqmc.h` — `WalkerPopulation`; `RandomSource::derive` at line 225
- `src/post_hf/fciqmc_driver.cpp` — `make_ops`, the T4 diagonal memo
- `tests/regression_cases.json` — `h2_fciqmc_threads1/4`, the invariance gate
- `docs/FCIQMC_RESEARCH_SCOPE.md` §6 — the decided threading policy
- `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` — the precedent, with its own traps

## What invariants matter

### 1. Binning must be by parent, not by output

Every parallel path in Planck is bitwise thread-count invariant, by design and
by gate (`FCIQMC_RESEARCH_SCOPE.md` §6, decided explicitly for FCIQMC rather
than discovered as a constraint after the fact). FCIQMC keeps that property via
partition-by-parent: each parent determinant hashes to one of `kBins = 64`
fixed bins (`hash(parent) % kBins`, never tied to thread count), each bin
accumulates into its own `WalkerPopulation`, `#pragma omp parallel for
schedule(static)` runs the bins, and the per-bin results merge back in fixed
bin order (0..kBins-1) regardless of which thread computed which bin or in
what order they finished.

Binning is by parent, not child: two different parents in two different bins
can spawn onto the same child determinant, and that cross-bin annihilation is
resolved once, in the fixed-order merge — never during the parallel region,
which would be the unsynchronized-map-write hazard this design exists to
avoid.

Three properties make this sound, each checked in the source rather than
assumed:

1. Every write during the parallel region targets only that thread's own
   `next_bins[bin]` — no shared mutable state, no atomic, no completion-order
   dependence.
2. The initiator rule reads `population` (the incoming, not partially-built,
   population) — a read-only lookup on a `const&`, safe to share across
   threads.
3. Both remaining sources of shared mutable state were removed before the
   pragma went in: the RNG (each parent draws from a per-bin `RandomSource`
   derived from one fresh 64-bit value per call) and the T4 diagonal memo
   (prefilled serially before the call, so the parallel region only reads it).

Design rule:

- No `omp atomic` on the walker map, no completion-order reduction. That is
  the DFT-grid jitter defect this codebase specifically avoids elsewhere.
- `kBins` must never be tied to thread count. The FCI sigma build paid for
  this twice: `schedule(dynamic)` gives an accumulator a different subset of
  terms per run, and keying by `omp_get_thread_num()` makes contents depend
  on the thread count itself.
- The invariance gate is `atol = 0.0`. A tolerance would hide exactly the
  reduction-order defect the whole design exists to prevent.

### 2. A reordering step's correctness standard is self-reproducibility plus an independent reference, never bitwise identity to the previous ordering

IEEE double addition is not associative, so binning the accumulation (S2)
cannot be bitwise identical to the unbinned serial form (S1) — and testing
that expectation is what caught it, not a code defect. S1 sums every parent's
contributions in one interleaved pass; S2 sums per-bin, then concatenates
bin-by-bin. On N2 the two orderings agree for 52 iterations (no multi-parent
overlap yet at that walker count) and diverge at iteration 53, only in the
projected-energy numerator — the signature of reassociation of an identical
term set, not a missing or misrouted write (the population totals stayed
bit-identical at that same step).

Design rule:

- For any reordering change, gate on bitwise identity to itself (proving the
  new order is fixed) plus agreement with an independent exact reference
  (proving the reordering hides no real defect) — never bitwise identity to
  the previous ordering.

### 3. `.clear()` does not reset a container to its fresh-construction state

A standalone test of `unordered_map::clear()`-and-reinsert against the same
key sequence found: the first reuse cycle's iteration order differs from both
fresh construction and a second reuse cycle, but the second and third reuse
cycles agree with each other — a reused map does not reproduce fresh-
construction's bucket layout, but does settle into a stable, repeatable layout
after one warm-up call. `.clear()` empties entries, not the bucket array, and
a reused map retains its peak bucket count. A later call whose actual entry
count is smaller than an earlier call's peak therefore iterates in a different
order than a freshly-constructed map holding the same entries would.

Design rule:

- Do not assume `.clear()` resets a container to its fresh-construction
  state, especially for `unordered_map`. Verify iteration-order stability
  standalone before relying on it for cross-thread-count invariance — the
  correct bar is self-reproducibility at fixed seed plus agreement with the
  independent reference (invariant 2), not bitwise match to fresh
  construction.

### 4. Container reuse proven safe for internal state does not transfer to a return value without checking NRVO

`next` (the fixed-bin-order merge target) is `propagate_stochastic`'s return
value, unlike the purely-internal `next_bins`/`bin_parents`. Making it a
function-local `static` to reuse across calls (the same pattern that worked
for the internal containers) disables NRVO on `return next;` — a static
cannot be moved-from, forcing a full copy-construct into the caller's
receiving `pop` on every call. This was found only by bisecting the real
binary after a standalone map-reuse microbenchmark reported it as safe: the
microbenchmark measured the reuse mechanism in isolation and missed the
separate, larger cost of losing NRVO.

Design rule:

- A container-reuse pattern proven correct for purely internal state does not
  automatically transfer to a return value — check NRVO eligibility
  separately, with a benchmark built narrowly around exactly that question,
  not reused from a benchmark that tested a different mechanism.

### 5. Re-measure the Amdahl ceiling immediately before adding threads

The ceiling at 4 threads moved three times as serial steps landed: 2.86x
(post-T4, before S1-S3) to 3.97x (post S1-S3, before S4) to 2.81x-then-2.86x-
then-3.97x again depending on measurement point. Every prior estimate for this
step was wrong within a day of being written down.

Design rule:

- Never quote an earlier ceiling estimate — re-measure immediately before
  building the next threading step, because every serial change upstream of
  it moves the number.

## What was built and measured

Landing order:

| step | what | result |
|---|---|---|
| S1 | per-bin RNG streams, still serial | one real bug caught (below); verified reproducible + FCI-agreeing |
| S2 | bin the accumulation, still serial | bitwise identity to S1 is not achievable (IEEE non-associativity) |
| S3 | prefill the T4 memo before the call | pure mechanical move, bitwise identical to S2 |
| S4 | the `#pragma omp` | correct and invariant; measured 1.51x at 4 threads against a 3.97x ceiling |
| S4.5/R1 | reuse `bin_parents` (no reordering risk) | bitwise identical to S4; ~5% faster |
| S4.5/R2 | reuse `next_bins` (reordering-sensitive) | reproducible + FCI-agreeing; 1.57x at 4 threads |
| S4.5/R3 | invariance re-verified on R2's own baseline | bitwise identical at 1/2/4/8 threads |

**S1's bug: fixed-seed reproducibility passed on a run that was silently
frozen.** The first per-bin-RNG implementation derived each bin's stream via
`rng.derive(b)`, a `const` method that reads `rng`'s state without advancing
it. Because `rng` lives in the driver and is passed by reference into every
one of 50,000 `propagate_stochastic` calls, every call rebuilt the identical
64 streams — the walker population never left its initial state (reference
weight exactly `10000.00`, zero variance, shift equal to the seed diagonal to
`1e-13`). The reproducibility gate passed, because the same broken trajectory
replayed twice is still "reproducible" by that check alone. It took reading
the population's own diagnostics, not the gate, to see it was frozen. Fixed by
drawing one raw 64-bit value from `rng` per call (`RandomSource::raw64()`,
which genuinely advances the caller's engine) and deriving the 64 bin streams
from that single draw — a pure function of `(call draw, bin index)` within one
call, independent of thread count, while differing across calls.

**S4: correct, 1.51x, and why not more.** At 4 threads S4 measured ~185% CPU
(not ~400%) and 56% of thread-samples in `__psynch_cvwait` — idle at the join
barrier. The first hypothesis (bin-to-thread load imbalance, from a 3.2x
spread in summed `|weight|` per bin) was wrong: direct per-thread
`draw_excitation` counts were nearly identical across threads, so the
imbalance measurement was a red herring built on the wrong proxy (weight does
not predict per-parent CPU cost when `n_spawn_attempts` is a per-run
constant). A greedy bin-to-thread packing scheme was scoped against this false
lead and never built — recorded so the next person does not retake that turn.

The real cause, isolated with a standalone microbenchmark: `propagate_
stochastic` averaged ~249 us/call at 4 threads. A synthetic benchmark
reproducing the exact per-call allocation pattern — 64 fresh `unordered_map`s
(`next_bins`) + 64 fresh `vector`s (`bin_parents`) at the measured occupancy —
measured 123 us/call, roughly half the total, for pure container construction
that happens before the parallel region starts, regardless of whether it
exists. Repartitioning work across threads cannot touch cost that is paid
once per call before any thread forks.

**R1/R2/R3: reuse instead of reconstruct.** Fix: persist `next_bins` and
`bin_parents` as function-local `static`s in `propagate_stochastic` (there is
exactly one call site, driven by a single thread across the whole
50,000-iteration loop, so a function-local static is the entire mechanism —
no reentrancy or recursion to guard) and `.clear()` them each call instead of
reconstructing.

- `bin_parents` (R1) is write-once grouped input — `push_back` order doesn't
  affect which parent lands in which bin, nothing sums or overwrites within
  it — so reuse cannot change any output value. Gate: bitwise identical to
  S4. Verified, on N2 and H2 at 1 and 4 threads (the only line that ever
  differs across the pre/post binaries is `Wall Time`). Measured delta:
  18.05s -> 17.19s (1 thread), 11.52s -> 10.92s (4 threads), ~5% either way —
  consistent with the 123us -> 111us/call microbenchmark prediction.
- `next_bins` (R2) is reordering-sensitive (invariant 3), so R2 used the S1
  standard: self-reproducibility at fixed seed (passed — two full runs of N2
  and H2, byte-identical) plus agreement with exact FCI (passed — H2 shift
  0.22 sigma / projected 0.33 sigma, N2 shift 0.93 sigma / projected 1.31
  sigma, all inside the 5 sigma gate). N2 (14,400 determinants, routine
  multi-parent overlap) diverges from the R1-only numbers starting at the
  Shift/Projected energy lines, exactly as expected; H2 (4 determinants <
  `kBins`, so at most one parent per bin — the same non-overlap condition S2
  found) is bitwise unchanged, also as expected. No committed regression
  reference needed updating: every FCIQMC gate (`h2_fciqmc_sto3g`,
  `h2_fciqmc_threads1/4`, `n2_fciqmc_sto3g`) asserts `metric_within_sigma`
  against the exact FCI energy or `metric_close_case` against a paired live
  run — none pins a raw RNG-trajectory-specific number.
- R3 re-verified invariance on R2's own baseline (not S4's): bitwise identical
  at `OMP_NUM_THREADS` = 1/2/4/8 on both N2 and H2. All four FCIQMC gates and
  the four non-QMC FCI gates sharing `build_all_mo_ci_setup` pass.

**The result, and why it undersells the fix:**

| | 1 thread | 4 threads | speedup |
|---|---|---|---|
| pre-R1 (S4) | 18.05 s | 11.52 s | 1.51x |
| post R1+R2 | 16.6 s | 10.6 s | 1.57x |

R1+R2 gave a real, measured gain — not the zero a broken reuse mechanism could
have produced — but the barrier-wait share did not move: re-profiled at 56.4%
`__psynch_cvwait`, statistically the same as S4's original 56%. `next_bins`/
`bin_parents` were half the isolated 123us/call container cost; the other half
of the per-call serial-before-parallel-region cost is untouched by R1/R2. The
re-profile's top-of-stack breakdown names where it sits: `unordered_map::
operator[]` (the T4 memo lookup), the memoized-diagonal lambda plus
`slater_condon_element` beneath it, and RNG generation via
`mersenne_twister_engine::_M_gen_rand`.

**Why the barrier-wait share didn't move: the merge, not the containers.** A
temporary env-gated probe (`PLANCK_FCIQMC_PHASE_PROBE`, following the same
inert-unless-set, deleted-once-it-answers-the-question discipline as the S3
miss-probe) instrumented all four remaining serial phases plus the parallel
region directly inside the real binary, on the real N2 gate, rather than
inferring from disjoint microbenchmarks:

| phase | 1 thread (us/call) | 4 threads (us/call) | share @4t |
|---|---|---|---|
| `next_bins` clear (64 maps) | 15.2 | 15.3 | 8.8% |
| `bin_rngs` construction | 25.9 | 25.9 | 14.9% |
| `bin_parents` partition | 5.9 | 5.9 | 3.4% |
| parallel region (the pragma) | 197.6 | 78.5 | 45.2% |
| merge (fixed bin-order sum) | 41.5 | 48.2 | 27.7% |

Confirmed reproducible on a repeat run (48.0/48.1/48.2 us across three
5000-call windows, not sample noise). Two findings: the parallel region itself
works as designed (2.52x at 4 threads, 197.6us -> 78.5us, consistent with real
parallel work bounded by fork/join overhead); and the merge is the largest
untouched serial cost, never targeted by R1/R2 — at 4 threads, total
serial-outside-the-pragma time (clear + rng + partition + merge = 15.3 + 25.9
+ 5.9 + 48.2 = 95.3 us/call) now exceeds the parallel region itself (78.5
us/call), which is structurally why the barrier-wait share cannot fall below
~50% regardless of thread balance inside the pragma.

A secondary, unexplained observation, recorded rather than chased: merge time
rose from 41.5us (1 thread) to 48.2us (4 threads) — the same serial work
(walking 64 populated bins in fixed order) taking 16% longer purely as a
function of thread count. The likely mechanism is post-join cache locality:
each bin's `unordered_map` was just written by a possibly-different core, so
the single merging thread now walks memory that is cold or NUMA-remote
relative to before threading existed. Not verified further — flagged as the
first thing to check if a future pass targets the merge.

**Two fix attempts on the remaining phases, one reverted, one not built.**

*The merge: tried, found to be a net loss on two independent grounds.* R1/R2's
reuse pattern was applied to `next` on the reasoning that, since the sequence
of `next.add(det, w)` calls is fixed by the outer (bin) and inner (each bin's
own, already-fixed-by-R2) loop order regardless of `next`'s own bucket layout,
reuse could only change where entries physically live, never the order `+=`
is applied to a given key — i.e. reordering-safe, R1's category, not R2's.
That reasoning was wrong (invariant 3 applied here too, missed on first
inspection), caught only by bisecting the real binary: an equilibration=0 A/B
on N2 was bitwise identical through 70 total `propagate_stochastic` calls and
diverged by 80, because `next`'s bucket count grows in discrete steps
(libstdc++: 2, 5, 11, 23, 47, 97, ...) as the walker population ramps up, and
a reused map retains its peak bucket count after `.clear()`. Re-tested against
the R2 standard once correctly categorized and all three checks passed (N2
projected energy landed at 2.70 sigma against exact FCI, inside the 5 sigma
gate but a different, equally valid trajectory from before). But correctness
was never the reason to revert — speed was, on the separate NRVO mechanism
(invariant 4): a standalone benchmark isolating fresh-local-with-NRVO vs
static-with-forced-copy (identical contents) measured 65.3us/call vs
97.3us/call. End to end on N2 this made `propagate_stochastic` measurably
slower: 16.6s/10.6s (1/4 threads) before the attempt, 17.9s/11.8s after,
reproduced across repeat runs. Reverted — `next` stays a fresh local.

*`bin_rngs`: measured, not worth the diff.* The vector holding the 64 per-bin
`RandomSource` streams is purely internal (never returned, index-accessed
like `bin_parents`), so persisting it as a static carries neither of the
merge's two hazards. A standalone microbenchmark measured the saving before
writing any real code: 26.8us/call (fresh vector) vs 25.8us/call (reused,
overwritten by index) — about 4%. The dominant cost is not the vector's own
allocation but constructing and seeding 64 `std::mt19937_64` engines from
scratch every call, real and unavoidable given S1's correctness requirement
that each call's 64 bin streams be fresh and independent. Judged not worth a
code change for a measured ~4%. Left as documented, unfixed cost.

## Validation strategy that should remain in place

- `h2_fciqmc_threads1/4`, `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g` at
  `OMP_NUM_THREADS` = 1/2/4/8, bitwise identical (`atol = 0.0`) where the
  design requires it
- Self-reproducibility at fixed seed plus agreement with exact FCI
  (`metric_within_sigma`) as the standard for any reordering change, per
  invariant 2
- Re-profiling with a real in-binary phase probe (not disjoint
  microbenchmarks) before trusting a diagnosis of where remaining time goes

## Remaining architecture concern

**Ceiling instability and diminishing returns.** The absolute payoff scales
with how much FCIQMC is actually run, not with walker count (4x the walkers
left the threadable share flat, since both the parallel and serial parts scale
with occupied population equally) but modestly with system size (15x the
determinants moved the ceiling 2.86x -> 3.07x). On the committed N2 gate the
saving is ~2 seconds; on a 102s HF/6-31G run it was ~40s. This loops back to
`FCIQMC_RESEARCH_SCOPE.md` Q1: nothing in the tree currently runs FCIQMC at a
size where the ceiling gap matters. The threading is correct and available;
whether to chase the remaining gap (the merge, or `bin_rngs`) is a decision to
make when a real target exists, not before. Parallelizing the merge itself
would reintroduce the exact completion-order hazard this whole design was
built to avoid, so any future work there is a smaller, careful target, not
simply "thread more of it."

**S5, the invariance gate's blind spot — not built.** `h2_fciqmc_threads1/4`
runs on 4 determinants — smaller than `kBins`, so at most one parent ever
lands in each bin and the gate can never exercise a merge-order defect. What's
missing: an N2-sized `threads1`/`threads4` pair at `atol = 0.0`, extended to 8
threads, verified non-vacuous by perturbing the merge order (reversing the bin
loop) and confirming the N2 gate goes red. A gate that cannot fail is not a
gate — this is the one piece of the original scope that still needs doing.
