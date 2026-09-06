# Scope: Can FCIQMC Be Rewritten to Be Genuinely Parallel?

**Scope for an investigation. Not started.** This is a question, not a plan.
The incremental threading of the spawn loop is done and measured
(`docs/FCIQMC_T2_THREADING.md`): 1.57× at 4 threads against a ~3× ceiling,
with the gap fully diagnosed as per-call serial scaffolding that now
exceeds the parallel region. This scope asks the different question the
T2 work deliberately did not: **is the 1.57× a property of FCIQMC, or a
property of how this FCIQMC is shaped — and would a rewrite of the data
structures and the call granularity get genuine (near-linear, or at least
solidly super-2×) parallelism?**

The output of this investigation is a decision with evidence: either "yes,
here is the shape that works and what it costs to build", or "no, and here
is the structural reason, measured not asserted". It is explicitly allowed
to conclude that the current 1.57× is the right stopping point.

## Why this is worth asking now

`docs/FCIQMC_T2_THREADING.md`'s "Remaining architecture concern" says the
remaining gap is the fixed-order merge plus RNG-shard construction, and
that "any future work there is a smaller, careful target, not simply
'thread more of it'." That is the correct conclusion *for the current
shape*. But every one of the T2 findings is downstream of two design
choices that were made early and never revisited:

1. **The walker population is a single `std::unordered_map<DetKey,
   double>`** (`WalkerPopulation`, `fciqmc.h`). Every determinism trap in
   the T2 doc — `.clear()` not resetting bucket layout, IEEE
   non-associativity across bin-merge orders, NRVO loss on a `static`
   return value — is a consequence of accumulating into a hash map whose
   iteration order is unspecified and unstable.
2. **`propagate_stochastic` is called ~50,000 times from a serial outer
   step loop** (`fciqmc_driver.cpp`), and each call rebuilds all 64 bins,
   64 `mt19937_64` engines, the parent partition, and a fresh return map
   from scratch. The parallel region is ~78 µs/call; the serial
   scaffolding around it is ~95 µs/call. The outer loop is inherently
   sequential (step `n+1` needs step `n`'s population and shift), so this
   per-call overhead is paid 50,000 times and cannot be amortized by the
   outer loop.

Neither choice was wrong for getting FCIQMC *working and validated*
(`docs/FCIQMC_RESEARCH_SCOPE.md` — the method reproduces exact FCI on
N2/STO-3G). The question is whether they are now the ceiling.

## The three hypotheses to test

Each is independently checkable and independently sufficient to change the
conclusion. Test them in this order — H1 is cheapest and, if it fails,
kills the rewrite before any code is written.

### H1 — the per-step work is too small to parallelize, regardless of shape

**Claim:** at the walker counts FCIQMC actually runs (N2 gate: ~0.69
walkers/determinant, ~10,000 walkers total), one `propagate_stochastic`
call does so little arithmetic that fork/join overhead dominates any
partitioning, and no rewrite of the data structure changes that. The
1.57× is then near the real ceiling and the answer is "no".

**How to test, before touching any code:**

- Instrument the real N2 gate (env-gated probe, same
  `PLANCK_FCIQMC_PHASE_PROBE` discipline as T2 — inert unless set, deleted
  once it answers) to record, per `propagate_stochastic` call: total
  parents, total spawn attempts, total `slater_condon_element` /
  `off_diagonal` evaluations, and wall-time of the pure per-parent work
  (the loop body only, excluding all scaffolding).
- Compute the arithmetic intensity: useful FLOPs (or evaluations) per call
  against the ~78 µs the parallel region takes now. Compare against the
  FCI sigma build, which threaded a structurally identical scatter to
  3.54× at 4 threads (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`) — that path's
  per-call work was large enough. Is FCIQMC's per-call work within, say,
  2× of that, or 20× smaller?
- Cross-check by running the N2 gate at 10× and 100× `fciqmc_walkers`
  (deliberately past saturation, so the answer is not physically
  meaningful but the *timing* is): does the threadable share grow with
  walker count? The T2 doc says "4× the walkers left the threadable share
  flat" — confirm or refute that on a wider sweep, because if the share is
  genuinely flat in walker count then the per-step work is bounded by
  *determinant count*, not population, and a rewrite cannot help at fixed
  system size.

**If H1 holds** — per-step work is 10×+ too small and flat in walker count
— stop here. Record it as "FCIQMC's per-step granularity is below the
threading threshold at reachable system sizes; the 1.57× stands; revisit
only if a target system with a much larger active space appears (see
`FCIQMC_RESEARCH_SCOPE.md` Q1)." Do not build the rewrite.

---

#### H1 result (measured 2026-09-06): the per-parent work is real, but the *region* has a granularity ceiling of ~2.3–2.6× at 4 threads and regresses at 8 — so H2 alone cannot reach near-linear

Probe (`PLANCK_FCIQMC_H1_PROBE`, added, measured, reverted — `grep`
confirms clean) instrumenting `propagate_stochastic` on the real N2/STO-3G
gate config: per-call parent count, spawn attempts, off-diagonal
evaluations, wall-time of the pragma region in isolation, and wall-time of
the whole call.

**The per-parent cost is a fixed property of the system, flat in walker
count.** Across an 80× walker sweep (2,000 → 160,000 walkers on N2/STO-3G,
`ndet` = 14,400):

| walkers | parents/call | ns/parent | region µs/call | whole µs/call | region share |
|---|---|---|---|---|---|
| 2,000 | 629 | 152 | 96 | 150 | 64 % |
| 10,000 (gate) | 908 | 147 | 133 | 200 | 66 % |
| 40,000 | 1,196 | 147 | 176 | 256 | 69 % |
| 160,000 | 1,372 | 148 | 203 | 292 | 70 % |

`ns/parent` is **146–152 ns across the whole range** — one `draw_excitation`
+ one `slater_condon_element` (off-diagonal branch) + one memoized diagonal
lookup + two `unordered_map` inserts. `parents/call` grows only 2.2× for
80× walkers because N2's 14,400-determinant space saturates. The T2 doc's
"4× the walkers left the threadable share flat" is confirmed and extended:
the *share* drifts 64 %→70 %, and `ns/parent` is dead flat.

**The region does not thread well even isolated from the serial
scaffolding.** Measuring the pragma region alone at 1/2/4/8 threads (gate
config, 908 parents):

| threads | region µs | region speedup | whole µs | whole speedup |
|---|---|---|---|---|
| 1 | 134.8 | 1.00× | 203.5 | 1.00× |
| 2 | 89.8 | 1.50× | 162.2 | 1.25× |
| 4 | 59.1 | **2.28×** | 133.0 | **1.53×** |
| 8 | 71.4 | 1.89× (regresses) | 145.7 | 1.40× (regresses) |

At 160,000 walkers (1,372 parents, more work per bin) the region reaches
**2.60× at 4 threads** and still regresses at 8 (2.27×). The 8-thread
regression reproduces exactly on repeat runs (70.7 µs twice at 10k).

**The cause is bin granularity, not scaffolding.** 908 parents ÷ 64 fixed
bins ≈ 14 parents/bin, ≈ 2.1 µs of arithmetic per bin, against fork/join
plus each `next_bins[bin]` being an `unordered_map` the region touches.
That is why 8 threads (≈ 8 bins each, ≈ 17 µs total) cannot cover the
thread-spawn cost. The FCI sigma build threaded to 3.54× because its
per-call work is `O(ndet)` full excitation enumerations (~600 connections
per determinant) — roughly two orders of magnitude more arithmetic per
outer-loop unit than FCIQMC's one sampled draw.

**What this means for the rewrite:**

- **H1 does not fully hold** — the per-parent work is not "10×+ too small";
  the region *does* speed up, to ~2.3× at 4 threads. But it **bounds H2
  hard**: even a perfect H2 (zero serial scaffolding) lands the whole call
  at the region's own ceiling, ~2.3–2.6× at 4 threads, never near-linear,
  and 8 threads is off the table at reachable system sizes.
- The remaining ~0.7–1.0× between the current 1.57× and the region's 2.3×
  is what H2's scaffolding removal can actually recover. That is real
  (~35–45 % faster) but it is a bounded, one-time gain, not a
  scaling-with-cores gain.
- **The bin count is the lever H1 exposes that the scope did not name.**
  `kBins = 64` was chosen for merge-order determinism, not throughput.
  Fewer, larger bins (e.g. `kBins = 16` or `= n_threads`, keeping the
  fixed-partition-by-parent-hash property) would raise per-bin work and
  may push the 4-thread region past 2.6× — but `kBins` tied to thread
  count is exactly the invariance hazard `FCI_SIGMA_BUILD_PERFORMANCE.md`
  paid for twice, so this must stay a fixed count, just a smaller one, and
  be re-gated. This is a cheaper experiment than the full H2 rewrite and
  should be tried first (call it **H2.0**).

**Recommendation:** H2 is worth doing for the bounded ~1.5× it recovers on
top of the current 1.57× (≈ 2.3–2.5× total at 4 threads), *if and only if*
a target system appears that runs FCIQMC long enough to care — the same
Q1 gate as everything else in this area. Try H2.0 (smaller fixed `kBins`)
first as a one-line experiment. Near-linear parallelism of a single
trajectory is **not reachable** by any data-structure rewrite at reachable
walker counts; that leaves H3 (replicas) as the only genuine
scaling-with-cores axis, and H3 only tightens the error bar.

### H2 — the serial-per-call scaffolding is the ceiling, and it is eliminable by hoisting state out of the call

**Claim:** the per-parent work *does* parallelize fine (the T2 doc's own
measurement: 197.6 µs → 78.5 µs, 2.52× inside the pragma), but 95 µs/call
of serial setup/teardown drags the total to 1.57×. That setup is
rebuilt-from-scratch state — 64 bins, 64 RNG engines, the partition, the
merge target — and *all of it can be persistent across the 50,000 calls*
because there is exactly one call site driven by one thread. A rewrite
that hoists this state into a reusable `SpawnWorkspace` owned by the
driver, threaded through by reference, removes the per-call construction
cost entirely.

**Why the T2 work did not already do this:** it tried, partially (R1/R2
made `next_bins` and `bin_parents` function-local `static`s) and got
~5%, then hit two walls — `.clear()` does not restore bucket layout
(invariant 3), and making the *return value* `static` loses NRVO
(invariant 4). Both walls are consequences of keeping `unordered_map` as
the accumulator and keeping the value-returning signature. **H2 says: fix
both by changing the data structure, not by reusing the map.** Specifically:

- **Replace the per-bin `unordered_map` accumulator with a structure whose
  iteration order is defined and stable across reuse.** Candidates to
  evaluate: a sorted `std::vector<std::pair<DetKey, Weight>>` merged with
  `std::inplace_merge` (order is total and reuse-stable by construction);
  a flat open-addressing hash table with a fixed capacity and a defined
  probe order that `.clear()` genuinely resets (just memset the control
  bytes); or a two-level structure (fixed bin array of small sorted
  vectors). The requirement is: *summing the same set of `(det, weight)`
  additions produces a bitwise-identical result on every call, regardless
  of how many times the structure has been reused* — which is exactly what
  `unordered_map` cannot give and is the root of the T2 reordering traps.
- **Make `propagate_stochastic` write into a caller-owned output rather
  than return one** — `void propagate_stochastic(..., SpawnWorkspace& ws,
  WalkerPopulation& out)` — so there is no NRVO question at all.
- **Hoist the 64 `mt19937_64` engines into the workspace**, re-seeded
  per call from one `rng.raw64()` draw (the S1 correctness requirement:
  each call's bin streams fresh and thread-count-independent). Measure
  whether re-seeding 64 existing engines in place is materially cheaper
  than constructing 64 — the T2 doc measured ~4% for the vector-reuse part
  alone but explicitly did *not* separate "reuse the vector" from "the
  seeding is unavoidable"; H2 needs that separated. If `mt19937_64`
  re-seed is genuinely as expensive as construct, evaluate a cheaper
  per-bin stream (a counter-based RNG — Philox/Threefry-style — is O(1) to
  "seed" because seeding is just setting a key, and is the standard choice
  for exactly this "N independent streams, re-derived every step" pattern;
  `RandomSource` is currently mt19937 and there is no counter-based RNG in
  the tree).

**How to test:** build the `SpawnWorkspace` with a sorted-vector
accumulator first (simplest reuse-stable option, and `std::inplace_merge`
of two sorted ranges is a well-understood deterministic operation). Wire
it in behind the same `atol = 0.0` invariance gate the T2 work uses
(`h2_fciqmc_threads1/4` plus a new N2-sized pair — S5 from the T2 doc,
still unbuilt, and a prerequisite here since H2 changes the merge). Re-run
the phase probe: does serial-outside-the-pragma drop from 95 µs/call to
near zero? Re-measure the ceiling and the achieved speedup on the N2 gate.

**If H2 holds** — serial scaffolding drops to near zero and the achieved
speedup jumps toward the (re-measured) ceiling — this is the rewrite, and
it is bounded: one new `SpawnWorkspace` type, one changed function
signature, one accumulator data-structure swap, all behind the existing
invariance discipline. Estimate the ceiling this exposes (the T2 doc's
own numbers suggest the parallel region alone is ~2.5× at 4 threads, so
removing the serial drag should land the *total* near there — but
re-measure, per T2 invariant 5, every prior ceiling estimate in this area
was wrong within a day).

### H3 — the outer step loop itself is not as sequential as it looks

**Claim (weakest, test last):** the 50,000-iteration outer loop over
imaginary-time steps is treated as strictly sequential, but FCIQMC
literature runs *multiple independent replicas* (different RNG seeds, same
Hamiltonian) both to get error bars and as a parallelization axis — each
replica is an embarrassingly parallel full trajectory. Planck already has
the reproducibility machinery (`RandomSource`, fixed-seed contract) and
the statistical machinery (`blocked_standard_error`, `metric_within_sigma`)
to combine replicas. A rewrite that runs R replicas across R thread groups,
each replica internally serial (or internally threaded per H2), is genuine
near-linear parallelism in R with zero shared mutable state and zero
determinism hazard — the replicas never touch.

**Why this might not be worth it:** it does not speed up a *single*
trajectory, so it only helps if the goal is a tighter error bar in fixed
wall-time (which it does, cleanly) rather than a converged answer sooner.
And it multiplies memory by R (R walker populations). For the committed N2
gate — seconds of runtime, already inside its error bar — it buys nothing.
It matters only at a real target size where one trajectory is minutes-to-
hours and the error bar is the deliverable.

**How to test:** this one is a design sketch plus a memory estimate, not a
measurement — there is nothing to measure until it is built. Sketch the
driver change (an outer replica loop, `#pragma omp parallel` over
replicas, each with its own `RandomSource(seed + replica_index)`,
`ShiftController`, and `WalkerPopulation`; combine the per-replica shift
and projected-energy series at the end via the existing blocking
analysis). Estimate: for R = 4 replicas on the N2 gate, memory goes 4×
(~tens of MB, negligible) and wall-time is unchanged while the error bar
tightens by ~2×. State whether that trade is worth a driver rewrite *given
no current target needs it* — the honest answer is probably "document it
as the right move when a target appears, do not build it now", the same
conclusion `FCIQMC_RESEARCH_SCOPE.md` Q1 reaches for the method as a whole.

## What a rewrite must not break

- **Bitwise thread-count invariance at `atol = 0.0`**
  (`FCIQMC_RESEARCH_SCOPE.md` §6, a decided policy, not a discovered
  constraint). Any new accumulator must produce identical results across
  `OMP_NUM_THREADS` = 1/2/4/8. This is *easier* to guarantee with a
  reuse-stable structure than with the current `unordered_map`, which is
  the point — but it must be gated, and the gate must be non-vacuous
  (S5: an N2-sized `threads1/threads4` pair that goes red when the merge
  order is perturbed).
- **Fixed-seed reproducibility** — `RandomSource`'s contract. A rewrite
  that changes the RNG (H2's counter-based option) changes every
  trajectory; that is allowed (it is a reordering-class change, gated by
  self-reproducibility + `metric_within_sigma` against exact FCI, never by
  matching old numbers — T2 invariant 2), but it must be a deliberate,
  recorded decision, and the reproducibility gate must still pass on the
  new RNG.
- **The shared integral setup.** `build_all_mo_ci_setup` is shared with
  `run_fci` (a move, not a copy — `FCIQMC_RESEARCH_SCOPE.md`). A rewrite
  touches `propagate_stochastic` and the driver loop, not the Hamiltonian
  construction — the four non-QMC FCI gates that share that setup must stay
  green.
- **`slater_condon_element` is public** and consumed by CASSCF/RASSCF
  response code. Do not change its signature or semantics as a side effect.

## Status

- **H1 — DONE (2026-09-06).** Result inline above: the per-parent work is
  ~148 ns and flat in walker count; the pragma region threads to only
  ~2.3× at 4 threads (2.6× at 160k walkers) and regresses at 8, a bin-
  granularity ceiling. H1 does not fully kill the rewrite but bounds it:
  near-linear single-trajectory parallelism is unreachable, and H2's
  realistic prize is ~1.5× on top of the current 1.57× (≈ 2.3–2.5× total
  at 4 threads). A new cheaper experiment, **H2.0 (smaller fixed `kBins`)**,
  falls out of H1 and should precede the full H2 rewrite.
- **H2.0 / H2 — not started.** Gated on a real target appearing
  (`FCIQMC_RESEARCH_SCOPE.md` Q1) — the bounded ~1.5× is not worth the
  rewrite until FCIQMC runs somewhere long enough to care.
- **H3 — not started.** Design sketch + memory estimate only; the honest
  expected conclusion is "document as the right move when a target
  appears".

## Deliverable

A short answer doc (`FCIQMC_PARALLELISM.md`, house shape) recording:

- H1's measured per-parent cost and region-vs-thread-count curve — done,
  the numbers are in the H1 result section above and move into the answer
  doc verbatim.
- If H2.0/H2 are ever built: the `kBins` sweep, then (if that is not
  enough) H2's `SpawnWorkspace` rewrite, the accumulator structure chosen
  and why, the before/after phase-probe breakdown, and the re-measured
  achieved speedup vs the H1 region ceiling.
- H3's replica-parallel sketch and memory estimate, with an explicit
  build/don't-build recommendation tied to whether a target system exists.
- The retired hypotheses, with their measurements, so the next person does
  not retake a turn (the T2 doc already has two of these — the bin-to-
  thread packing red herring, and the `next`-as-static reversion).

H1's finding is substantive enough that this scope file stays (as an
answer-in-progress) rather than collapsing to one paragraph in the T2
doc. It converts to `FCIQMC_PARALLELISM.md` when H2/H3 are resolved or
explicitly declined.

## Key code locations

| what | where |
|---|---|
| the spawn loop, its 64-bin partition and fixed-order merge | `propagate_stochastic`, `src/post_hf/ci/fciqmc.cpp` |
| the walker map (the accumulator to potentially replace) | `WalkerPopulation`, `src/post_hf/ci/fciqmc.h` |
| the serial outer step loop, the T4 diagonal prefill | `src/post_hf/fciqmc_driver.cpp`, the `for (step ...)` loop |
| the RNG and its reproducibility contract | `RandomSource`, `src/post_hf/ci/fciqmc.h` |
| the invariance gate (needs the S5 N2-sized extension) | `h2_fciqmc_threads1` / `h2_fciqmc_threads4`, `tests/regression_cases.json` |
| the decided determinism policy | `docs/FCIQMC_RESEARCH_SCOPE.md` §6 |
| the incremental threading already done, and its diagnosed ceiling | `docs/FCIQMC_T2_THREADING.md` |
| the serial fixes that preceded threading | `docs/FCIQMC_SERIAL_PERFORMANCE.md` |
| the structurally identical scatter that threaded to 3.54× | `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` |

---

Status will live in `vault/Status/Open Work.md` once the investigation
starts. Parent context: `docs/FCIQMC_RESEARCH_SCOPE.md`.
