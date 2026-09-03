# FCIQMC T2: threading the spawn loop

**Answer.** The FCIQMC spawn loop (`propagate_stochastic`,
`src/post_hf/ci/fciqmc.cpp`) is threaded, bitwise thread-count invariant
(`OMP_NUM_THREADS` = 1/2/4/8, `atol = 0.0`), and gives a real but modest
**1.57x at 4 threads** — well short of the ~4x Amdahl ceiling the serial
structure allows. The gap is fully diagnosed: at 4 threads, the fixed-bin-
order **merge** alone costs more per call (48.2us) than the entire threaded
region (78.5us), and together with the still-serial RNG-shard construction
(25.9us) and bin-clear pass (15.3us), serial-outside-the-pragma work now
*exceeds* the parallel region — structurally capping the achievable speedup
regardless of thread balance. **Both remaining levers were investigated and
neither is worth taking**: merging via reuse was tried and reverted (net
*slower*, plus a reordering hazard caught only by bisection), and the RNG-
shard construction cost is dominated by unavoidable seeding work, not
container overhead (~4% available, judged not worth a code change). See
"Why the barrier-wait share didn't move" and "Two fix attempts, one reverted"
below.

The serial work that preceded this (5.21x, no threads) is answered in
`FCIQMC_SERIAL_PERFORMANCE.md`.

## The design

Every parallel path in Planck is bitwise thread-count invariant, by design and
by gate (`FCIQMC_RESEARCH_SCOPE.md` §6, decided explicitly for FCIQMC rather
than discovered as a constraint after the fact). FCIQMC keeps that property
via **partition-by-parent**: each parent determinant hashes to one of `kBins =
64` fixed bins (`hash(parent) % kBins`, never tied to thread count), each bin
accumulates into its own `WalkerPopulation`, `#pragma omp parallel for
schedule(static)` runs the bins, and the per-bin results merge back in fixed
bin order (0..kBins-1) regardless of which thread computed which bin or in
what order they finished.

Three properties make this sound, each checked in the source rather than
assumed:

1. Every write during the parallel region targets only that thread's own
   `next_bins[bin]` — no shared mutable state, no atomic, no completion-order
   dependence.
2. The initiator rule reads `population` (the *incoming*, not partially-built,
   population) — a read-only lookup on a `const&`, safe to share across
   threads.
3. Both remaining sources of shared mutable state were removed before the
   pragma went in: the RNG (each parent draws from a per-bin `RandomSource`
   derived from one fresh 64-bit value per call) and the T4 diagonal memo
   (prefilled serially before the call, so the parallel region only reads it).

Binning is by **parent**, not child: two different parents in two different
bins can spawn onto the same child determinant, and that cross-bin
annihilation is resolved once, in the fixed-order merge — never during the
parallel region, which would be the unsynchronized-map-write hazard this
design exists to avoid.

## What shipped, in landing order

| step | what | result |
|---|---|---|
| S1 | per-bin RNG streams, still serial | one real bug caught (see below); verified reproducible + FCI-agreeing |
| S2 | bin the accumulation, still serial | bitwise identity to S1 is not achievable (IEEE non-associativity) — see below |
| S3 | prefill the T4 memo before the call | pure mechanical move, bitwise identical to S2 |
| S4 | the `#pragma omp` | correct and invariant; measured 1.51x at 4 threads against a 3.97x ceiling |
| S4.5/R1 | reuse `bin_parents` (no reordering risk) | bitwise identical to S4; ~5% faster |
| S4.5/R2 | reuse `next_bins` (reordering-sensitive) | reproducible + FCI-agreeing; 1.57x at 4 threads |
| S4.5/R3 | invariance re-verified on R2's own baseline | bitwise identical at 1/2/4/8 threads |

### Two bugs, each caught by a check that almost didn't catch them

**S1's bug: fixed-seed reproducibility passed on a run that was silently
frozen.** The first per-bin-RNG implementation derived each bin's stream via
`rng.derive(b)`, a `const` method that reads `rng`'s state without advancing
it. Because `rng` lives in the driver and is passed by reference into every
one of 50,000 `propagate_stochastic` calls, every call rebuilt the identical
64 streams — the walker population never left its initial state (reference
weight exactly `10000.00`, zero variance, shift equal to the seed diagonal to
`1e-13`). The reproducibility gate passed, because the same broken trajectory
replayed twice is still "reproducible" by that check alone. It took reading
the population's own diagnostics, not the gate, to see it was frozen. Fixed
by drawing one raw 64-bit value from `rng` per call (`RandomSource::raw64()`,
which genuinely advances the caller's engine) and deriving the 64 bin streams
from that single draw — a pure function of `(call draw, bin index)` within one
call, independent of thread count, while differing across calls.

**S2's instruction was itself wrong, and testing it caught that rather than a
code defect.** The scope said to verify S2 (binning the accumulation) was
bitwise identical to S1. It cannot be, because IEEE double addition is not
associative: S1 sums every parent's contributions in one interleaved pass; S2
sums per-bin, then concatenates bin-by-bin. On N2 the two orderings agree for
52 iterations (no multi-parent overlap yet at that walker count) and diverge
at iteration 53, only in the projected-energy numerator — the signature of
reassociation of an identical term set, not a missing or misrouted write (the
population totals stayed bit-identical at that same step). The correct
standard for a reordering step, used from here on: bitwise identity to
**itself** (proving the new order is fixed) plus agreement with an
independent exact reference (proving the reordering hides no real defect) —
never bitwise identity to the previous ordering.

## S4: correct, 1.51x, and why not more

At 4 threads S4 measured ~185% CPU (not ~400%) and 56% of thread-samples in
`__psynch_cvwait` — idle at the join barrier. The first hypothesis (bin-to-
thread load imbalance, from a 3.2x spread in summed `|weight|` per bin) was
wrong: direct per-thread `draw_excitation` counts were nearly identical
across threads, so the imbalance measurement was a red herring built on the
wrong proxy (weight does not predict per-parent CPU cost when
`n_spawn_attempts` is a per-run constant). A greedy bin-to-thread packing
scheme was scoped against this false lead and never built — recorded so the
next person does not retake that turn.

The real cause, isolated with a standalone microbenchmark: `propagate_
stochastic` averaged ~249 us/call at 4 threads. A synthetic benchmark
reproducing the exact per-call allocation pattern — 64 fresh `unordered_map`s
(`next_bins`) + 64 fresh `vector`s (`bin_parents`) at the measured occupancy —
measured 123 us/call, roughly half the total, for pure container
construction that happens **before** the parallel region starts, regardless
of whether it exists. Repartitioning work across threads cannot touch cost
that is paid once per call before any thread forks.

## R1/R2/R3: reuse instead of reconstruct

Fix: persist `next_bins` and `bin_parents` as function-local `static`s in
`propagate_stochastic` (there is exactly one call site, driven by a single
thread across the whole 50,000-iteration loop, so a function-local static is
the entire mechanism — no reentrancy or recursion to guard) and `.clear()`
them each call instead of reconstructing.

The two containers needed different verification standards, discovered before
either was built rather than assumed:

- **`bin_parents` (R1) is write-once grouped input** — `push_back` order
  doesn't affect which parent lands in which bin, nothing sums or overwrites
  within it — so reuse cannot change any output value. Gate: bitwise
  identical to S4. **Verified**, on N2 and H2 at 1 and 4 threads (the only
  line that ever differs across the pre/post binaries is `Wall Time`).
  Measured delta: 18.05s → 17.19s (1 thread), 11.52s → 10.92s (4 threads),
  ~5% either way — consistent with the 123us → 111us/call microbenchmark
  prediction.

- **`next_bins` (R2) is reordering-sensitive.** A standalone test of
  `unordered_map::clear()`-and-reinsert against the same key sequence found:
  the first reuse cycle's iteration order differs from both fresh
  construction *and* a second reuse cycle, but the second and third reuse
  cycles agree with each other — a reused map does not reproduce fresh-
  construction's bucket layout, but does settle into a stable, repeatable
  layout after one warm-up call. That stability (not bitwise match to fresh
  construction) is what cross-thread-count invariance actually needs, so R2
  used the S1 standard: self-reproducibility at fixed seed (**passed** — two
  full runs of N2 and H2, byte-identical) plus agreement with exact FCI
  (**passed** — H2 shift 0.22σ / projected 0.33σ, N2 shift 0.93σ / projected
  1.31σ, all inside the 5σ gate). N2 (14,400 determinants, routine
  multi-parent overlap) diverges from the R1-only numbers starting at the
  Shift/Projected energy lines, exactly as expected; H2 (4 determinants <
  `kBins`, so at most one parent per bin — the same non-overlap condition S2
  found) is bitwise unchanged, also as expected.

  No committed regression reference needed updating: every FCIQMC gate
  (`h2_fciqmc_sto3g`, `h2_fciqmc_threads1/4`, `n2_fciqmc_sto3g`) asserts
  `metric_within_sigma` against the exact FCI energy or `metric_close_case`
  against a paired live run — none pins a raw RNG-trajectory-specific number.

- **R3 re-verified invariance on R2's own baseline** (not S4's): bitwise
  identical at `OMP_NUM_THREADS` = 1/2/4/8 on both N2 and H2. All four FCIQMC
  gates and the four non-QMC FCI gates sharing `build_all_mo_ci_setup` pass.

## The result, and why it undersells the fix

| | 1 thread | 4 threads | speedup |
|---|---|---|---|
| pre-R1 (S4) | 18.05 s | 11.52 s | 1.51x |
| post R1+R2 | 16.6 s | 10.6 s | 1.57x |

R1+R2 gave a real, measured gain — not the zero a broken reuse mechanism
could have produced — but **the barrier-wait share did not move**: re-
profiled at 56.4% `__psynch_cvwait`, statistically the same as S4's original
56%. `next_bins`/`bin_parents` were half the isolated 123us/call container
cost; the other half of the per-call serial-before-parallel-region cost is
untouched by R1/R2. The re-profile's top-of-stack breakdown names where it
sits: `unordered_map::operator[]` (the T4 memo lookup), the memoized-diagonal
lambda plus `slater_condon_element` beneath it, and RNG generation via
`mersenne_twister_engine::_M_gen_rand`. Two concrete candidates for a further
pass — `bin_rngs` construction (S1's `kBins` fresh `RandomSource`s every
call) and the `ops.diagonal` prefill pass (S3's full walk over `population`
immediately before the call) — are named but neither built nor measured here.

## Why the barrier-wait share didn't move: the merge, not the containers

R1/R2 halved the isolated container-construction cost the S4.5 microbenchmark
measured (123us -> ~111us), but the real binary's `__psynch_cvwait` share sat
at 56% both before and after. Isolated microbenchmarks of the two remaining
serial-before-parallel candidates named at the time — `bin_rngs` construction
(64 fresh `mt19937_64` seedings, S1) and the T4 memo lookup pass (S3's
prefill) — measured 27us/call and 10us/call respectively on matched synthetic
inputs, which don't add up to a number that explains a persistently-flat 56%.

**The missing piece was never measured because nobody had timed the merge.**
A temporary env-gated probe (`PLANCK_FCIQMC_PHASE_PROBE`, following the same
inert-unless-set, deleted-once-it-answers-the-question discipline as the S3
miss-probe) instrumented all four remaining serial phases plus the parallel
region directly inside the real binary, on the real N2 gate, rather than
inferring from disjoint microbenchmarks:

| phase | 1 thread (us/call) | 4 threads (us/call) | share @4t |
|---|---|---|---|
| `next_bins` clear (64 maps) | 15.2 | 15.3 | 8.8% |
| `bin_rngs` construction | 25.9 | 25.9 | 14.9% |
| `bin_parents` partition | 5.9 | 5.9 | 3.4% |
| **parallel region (the pragma)** | 197.6 | 78.5 | 45.2% |
| **merge (fixed bin-order sum)** | 41.5 | 48.2 | **27.7%** |

Two findings, confirmed reproducible on a repeat run (48.0/48.1/48.2 us
across three 5000-call windows, not sample noise):

1. **The parallel region itself works as designed** — 2.52x at 4 threads
   (197.6us -> 78.5us), consistent with real parallel work bounded by
   fork/join overhead, not a hidden imbalance.
2. **The merge is the largest untouched serial cost, and R1/R2 never
   targeted it.** At 4 threads, total serial-outside-the-pragma time
   (clear + rng + partition + merge = 15.3 + 25.9 + 5.9 + 48.2 = **95.3
   us/call**) now *exceeds* the parallel region itself (78.5 us/call). That
   is structurally why the barrier-wait share cannot fall below ~50%
   regardless of how well-balanced the 4 threads are inside the pragma: more
   than half the call's wall time is spent before any thread forks or after
   they all join, and none of it was addressed by S4.5.

**A secondary, unexplained observation, recorded rather than chased:** merge
time rose from 41.5us (1 thread) to 48.2us (4 threads) — the same serial work
(walking 64 populated bins in fixed order) taking 16% longer purely as a
function of thread count. The likely mechanism is post-join cache locality:
each bin's `unordered_map` was just written by a possibly-different core, so
the single merging thread now walks memory that is cold or NUMA-remote
relative to before threading existed. Not verified further — flagged as the
first thing to check if a future pass targets the merge.

**What this means for a future step, if one is ever taken:** `bin_rngs`
(27us) and the merge (48us) together are larger than what R1/R2 removed, and
are both fixed per-call costs unrelated to occupied population size at this
scale (`clear`, `rng`, and `partition` are all flat between 1 and 4 threads,
confirming they're genuinely thread-count-independent serial work, not an
artifact of the probe). Parallelizing the merge itself would reintroduce the
exact completion-order hazard this whole design was built to avoid (the merge
must stay in fixed bin order for cross-thread-count invariance), so any
future work here is a smaller, careful target — not simply "thread more of
it." Left unbuilt, consistent with `FCIQMC_RESEARCH_SCOPE.md` Q1: nothing in
the tree runs FCIQMC at a size where this gap currently matters.

## Two fix attempts, one reverted

The two remaining phases (`bin_rngs`, the merge) were investigated as
candidates for the same reuse pattern R1/R2 used. One was tried, measured
worse than doing nothing, and reverted; the other was measured and judged not
worth the diff before any code was written.

### The merge: tried, found to be a net loss on TWO independent grounds

R1/R2's reuse pattern — persist a container as a function-local `static`,
`.clear()` at the top of each call instead of reconstructing — was applied to
`next` (the fixed-bin-order merge target) on the reasoning that, since the
sequence of `next.add(det, w)` calls is fixed by the outer (bin) and inner
(each bin's own, already-fixed-by-R2) loop order regardless of `next`'s own
bucket layout, reuse could only change where entries physically live, never
the order `+=` is applied to a given key — i.e. reordering-safe, R1's
category, not R2's.

**That reasoning was wrong, and a standalone microbenchmark using a static
(non-growing) key set failed to catch it — it took bisecting the real binary
to find the actual mechanism.** An equilibration=0 A/B on N2, run at
successively longer sampling-step counts, was bitwise identical through 70
total `propagate_stochastic` calls and diverged by 80. The cause: `next`'s
bucket count grows in discrete steps (libstdc++: 2, 5, 11, 23, 47, 97, ...)
as the walker population ramps up from near-empty toward its target size, and
a **reused** map retains its peak bucket count after `.clear()` — `.clear()`
empties the entries, not the bucket array. A later call whose actual entry
count is smaller than an earlier call's peak therefore iterates the reused
map in a *different* order than a freshly-constructed map holding the exact
same entries would — reassociating the sum for any determinant written more
than once per call, which happens routinely (the death term and a spawn can
share a target; two different bins can spawn onto the same child). This is
exactly R2's own mechanism, misapplied to a case that looked different on
first inspection. Re-tested against the R2 standard once correctly
categorized — self-reproducibility at fixed seed, agreement with exact FCI,
invariance across `OMP_NUM_THREADS` = 1/2/4/8 — and all three passed (N2
projected energy landed at 2.70σ against exact FCI, inside the 5σ gate but a
different, equally valid trajectory from before, exactly as R1/S1's own
precedent established: a legitimate reassociation is not required to
reproduce the old trajectory's specific sigma).

**But correctness was never the reason to revert it — speed was, and on a
second, independent mechanism the isolated map-reuse microbenchmark also
missed.** `next` is this function's **return value**; `next_bins` and
`bin_parents` are purely internal working state, never returned. Making a
return value a function-local `static` disables NRVO on `return next;` — a
static cannot be moved-from, so the compiler is forced to copy-construct the
full map into the caller's receiving `pop` on every call, instead of eliding
that copy. A standalone benchmark isolating exactly this difference (same map
contents and growth history, only fresh-local-with-NRVO vs
static-with-forced-copy) measured 65.3us/call against 97.3us/call — the
forced copy costs *more* than the map-reuse itself saves. End to end on the
real N2 gate this made `propagate_stochastic` measurably *slower*: 16.6s/1
thread and 10.6s/4 threads before the attempt, 17.9s/11.8s after, reproduced
across repeat runs, not noise.

**Reverted.** `next` stays a fresh local. Two lessons, both costing a wrong
answer first: (1) a container-reuse pattern proven correct for purely
internal state (`next_bins`, `bin_parents`) does not transfer to a return
value without checking NRVO eligibility separately — the two costs (bucket
reassociation, forced-copy overhead) are independent failure modes, and
either alone was sufficient reason not to do this; (2) an isolated
microbenchmark of the *specific mechanism under suspicion* (map-reuse
timing) can look favorable while missing a *different* mechanism entirely
(the return-value copy) that the real binary's structure exposes — the
standalone benchmark that finally caught the copy cost had to be built
narrowly around exactly the fresh-local-vs-static-return question, not
reused from the map-reuse benchmark that came before it.

### `bin_rngs`: measured, not worth the diff

The vector holding the 64 per-bin `RandomSource` streams is purely internal
(never returned, and index-accessed like `bin_parents`, not iterated in a way
that could reassociate anything), so persisting it as a static and
overwriting by index instead of `reserve()` + `push_back()` each call carries
neither of the merge's two hazards. A standalone microbenchmark measured the
saving anyway before writing any real code: 26.8us/call (fresh vector) vs
25.8us/call (reused, overwritten by index) — about 4%. The dominant cost is
not the vector's own allocation (one `reserve()` call, not 64 separate
allocations) but constructing and seeding 64 `std::mt19937_64` engines from
scratch every call, which is real, unavoidable work given the correctness
requirement that each call's 64 bin streams be fresh and independent — S1's
whole design. Judged not worth a code change, a new comment block, and gate
re-verification for a measured ~4%. Left as documented, unfixed cost.

## Ceiling instability, and what it means going forward

The Amdahl ceiling at 4 threads moved **three times** as serial steps landed:
2.86x (post-T4, before S1-S3) → 3.97x (post S1-S3, before S4) → 2.81x-then-
2.86x-then-3.97x again depending on measurement point. The lesson carried
forward from this: **re-measure the ceiling immediately before adding
threads, never quote an earlier number.** Every prior estimate for this step
was wrong within a day of being written down.

The absolute payoff scales with how much FCIQMC is actually run, not with
walker count (4x the walkers left the threadable share flat, since both the
parallel and serial parts scale with occupied population equally) but
modestly with system size (15x the determinants moved the ceiling 2.86x →
3.07x). On the committed N2 gate the saving is ~2 seconds; on a 102s HF/6-31G
run it was ~40s. This loops back to `FCIQMC_RESEARCH_SCOPE.md` Q1: **nothing
in the tree currently runs FCIQMC at a size where the ceiling gap matters.**
The threading is correct and available; whether to chase the remaining gap
is a decision to make when a real target exists, not before.

## What must not be done here (binding constraints, verified not assumed)

- **No `omp atomic` on the walker map, no completion-order reduction.** That
  is the DFT-grid jitter defect this codebase specifically avoids elsewhere.
- **`kBins` must never be tied to thread count.** The FCI sigma build paid
  for this twice: `schedule(dynamic)` gives an accumulator a different
  *subset* of terms per run, and keying by `omp_get_thread_num()` makes
  contents depend on the thread count itself.
- **The invariance gate is `atol = 0.0`.** A tolerance would hide exactly the
  reduction-order defect the whole design exists to prevent.
- **Do not assume `.clear()` resets a container to its fresh-construction
  state.** It does not, for `unordered_map` (bucket count, and therefore
  iteration order, can persist across a clear) — verified standalone rather
  than trusted from the standard's silence on bucket retention.

## Open: S5, the invariance gate's blind spot

`h2_fciqmc_threads1/4` runs on 4 determinants — smaller than `kBins`, so at
most one parent ever lands in each bin and the gate can never exercise a
merge-order defect. **Not built**: an N2-sized `threads1`/`threads4` pair at
`atol = 0.0`, extended to 8 threads, verified non-vacuous by perturbing the
merge order (reversing the bin loop) and confirming the N2 gate goes red. A
gate that cannot fail is not a gate — this is the one piece of the original
scope that still needs doing.

## Key locations

| what | where |
|---|---|
| the loop | `propagate_stochastic`, `src/post_hf/ci/fciqmc.cpp` |
| RNG shard derivation | `RandomSource::derive`, `fciqmc.h:225` |
| the T4 memo | `make_ops`, `src/post_hf/fciqmc_driver.cpp` |
| the walker map | `WalkerPopulation`, `fciqmc.h` |
| the invariance gate | `h2_fciqmc_threads1/4`, `tests/regression_cases.json` |
| the precedent and its traps | `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
