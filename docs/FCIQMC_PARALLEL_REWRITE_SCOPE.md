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

#### H1 result (measured 2026-09-06): the answer is fixture-dependent — on N2 the region has a granularity ceiling of ~2.3×, on HF/6-31G it threads to 4.5× at 8 threads and climbs. The N2 ceiling was a saturation artifact, not a property of FCIQMC.

Probe (`PLANCK_FCIQMC_H1_PROBE`, added, measured, reverted — `grep`
confirms clean) instrumenting `propagate_stochastic` on two real fixtures:
the N2/STO-3G gate and HF/6-31G (`tests/inputs/exploratory/fciqmc/
validation/hf_base.hfinp` — `ndet` = 213,444, 14.8× N2, and it does NOT
saturate until 213k walkers where N2 saturates around 44k).

**The per-parent cost is a fixed property of the system, flat in walker
count — confirmed on both.** N2 `ns/parent` is **146–152** across a 2k→160k
walker sweep; HF `ns/parent` is **~202 at 1 thread** (~58 at 4 threads,
i.e. 202/3.44) and equally flat across 10k→100k walkers. It is one
`draw_excitation` + one `slater_condon_element` (off-diagonal) + one
memoized diagonal lookup + two `unordered_map` inserts; HF's is slightly
higher because it has 6 virtuals/spin vs N2's 3.

**But `parents/call` — and therefore per-bin work — depends entirely on
whether the determinant space is saturated:**

| fixture | ndet | walkers | parents/call | parents/bin |
|---|---|---|---|---|
| N2/STO-3G | 14,400 | 10,000 (gate) | 908 | ~14 |
| N2/STO-3G | 14,400 | 160,000 | 1,372 | ~21 |
| HF/6-31G | 213,444 | 10,000 | 15,867 | ~248 |
| HF/6-31G | 213,444 | 50,000 (base) | 27,505 | ~430 |

N2's `parents/call` grows only 2.2× for 80× walkers because its space
saturates — at that point every determinant is occupied and adding walkers
just deepens weights. HF's space is 15× larger and stays unsaturated, so
`parents/call` is 30× N2's at the same walker count.

**The region-threading result splits cleanly on that:**

*N2/STO-3G* (908 parents, ~14/bin) — **granularity-ceilinged**:

| threads | region µs | region speedup | whole µs | whole speedup |
|---|---|---|---|---|
| 1 | 134.8 | 1.00× | 203.5 | 1.00× |
| 2 | 89.8 | 1.50× | 162.2 | 1.25× |
| 4 | 59.1 | **2.28×** | 133.0 | 1.53× |
| 8 | 71.4 | 1.89× (regresses) | 145.7 | 1.40× (regresses) |

*HF/6-31G* (27,505 parents, ~430/bin) — **scales like the FCI sigma build**:

| threads | region µs | region speedup | whole µs | whole speedup |
|---|---|---|---|---|
| 1 | 5569 | 1.00× | 7039 | 1.00× |
| 2 | 3129 | 1.78× | 4652 | 1.51× |
| 4 | 1617 | **3.44×** | 3139 | **2.24×** |
| 8 | 1245 | **4.47×** (still climbing) | 2849 | **2.47×** |

**The cause is bin granularity, and it is a fixture property.** N2's 908
parents ÷ 64 bins ≈ 14/bin ≈ 2.1 µs arithmetic per bin — fork/join plus
each `next_bins[bin]` being an `unordered_map` the region touches
dominates, and 8 threads (≈ 8 bins each) cannot cover the thread-spawn
cost. HF's ~430 parents/bin ≈ 87 µs of arithmetic per bin — comfortably
above the fork/join floor, so the region threads to 3.44×/4.47× with no
regression, the same regime the FCI sigma build (~600 excitation
enumerations per determinant, `O(ndet)` per call) sits in.

**What this means for the rewrite:**

- **H1 is refuted as stated.** The per-parent work is not too small in
  general — on any fixture whose determinant space is not saturated
  (which is the interesting case — `FCIQMC_RESEARCH_SCOPE.md` Q1 is about
  large active spaces), the region threads to 3.4×+ at 4 threads and keeps
  scaling. The N2 gate is a *saturated* fixture chosen deliberately so
  sampling is a real sample of a small space; that same choice makes it a
  bad throughput fixture. **Use HF/6-31G (or larger) for all future
  parallel-FCIQMC measurement.**
- **H2 becomes worth doing.** On HF the region ceiling is ≥ 4.5× at 8
  threads, while the whole call is stuck at 2.24×/2.47× because the serial
  scaffolding is ~1.5 ms/call (30× N2's absolute cost — HF partitions and
  merges 30× more parents) and does not thread. That ~1.5 ms is exactly
  the `SpawnWorkspace` target: hoist the 64-bin construction, the 64 RNG
  engines, the partition, and the merge into reusable driver-owned state,
  swap `unordered_map` for a reuse-stable accumulator, and the whole call
  should track the region toward 3.5–4×+ at 4 threads on an unsaturated
  fixture.
- **H2.0 (smaller fixed `kBins`) is now specifically an N2-class fix, not
  general.** It would help a saturated fixture where bins are starved; on
  HF the bins are already large enough. If any real target is
  saturation-limited it is worth trying, but the primary lever is H2's
  scaffolding removal on the unsaturated case.
- **Near-linear is plausible on a large unsaturated fixture** — HF's
  region already does 4.47×/8 threads and had not plateaued. Whether the
  *whole call* reaches near-linear depends on H2 killing the serial
  scaffolding; that is now a measurable question with a fixture that can
  answer it, not a foregone "no".

**Recommendation:** H1 does not kill the rewrite — it relocates it. The
work is: (1) adopt HF/6-31G as the parallel fixture, (2) build H2's
`SpawnWorkspace`, measuring against the HF region ceiling (≥ 4.5×), (3)
keep H2.0 in reserve for a saturation-limited target. Still gated on a
real target appearing (`FCIQMC_RESEARCH_SCOPE.md` Q1) — nothing in the
tree runs FCIQMC long enough for even the current 2.24×/HF to matter —
but the ceiling is now known to be high, not low.

### H2 — the serial-per-call scaffolding is the ceiling, and it is eliminable by hoisting state into a reusable workspace

**Claim:** the per-parent work parallelizes well on an unsaturated fixture
(H1: HF/6-31G region 3.44×/4t, 4.47×/8t, still climbing), but the whole
call is stuck at 2.24×/4t because ~1.5 ms/call of serial setup/teardown —
64-bin construction, 64 `mt19937_64` engines, the parent partition, the
fixed-order merge — is rebuilt every one of ~50,000 calls and does not
thread. There is exactly one call site, driven by one thread across the
whole outer loop, so *all of that state can be persistent*. Hoist it into
a `SpawnWorkspace` owned by the driver and passed by reference, swap the
per-bin `unordered_map` for a reuse-stable accumulator, and the whole call
should track the region toward 3.5–4×+ at 4 threads.

**Why the T2 work did not already do this:** R1/R2 tried the cheap version
(function-local `static`s) and hit two walls — `unordered_map::clear()`
does not restore bucket layout (T2 invariant 3), and making the
value-returning result `static` loses NRVO (T2 invariant 4). Both are
consequences of keeping `unordered_map` and the value-returning signature.
H2 fixes the data structure and the signature, not the reuse mechanism.

**Ordered so each step is independently verifiable, and a wrong step is
caught before the next.** Steps H2.1–H2.3 are prerequisites that land and
gate on their own; H2.4–H2.6 are the rewrite proper; H2.7 is the
measurement that decides whether it was worth it.

| step | adds | verifies against | if it fails |
|---|---|---|---|
| **H2.1** | the S5 gate: an **N2-sized** `threads1`/`threads4` pair at `atol = 0.0`, made non-vacuous by a temporary reversed-bin-order mutation that must turn it red | the current tree (must pass as-is — S5 is testing infrastructure, not a code change) | S5 cannot be made to go red → the gate is vacuous; fix the gate before any H2 code, or every later "bitwise identical" claim is worthless |
| **H2.2** | a standalone unit test (`fciqmc_accumulator`) for whatever reuse-stable accumulator H2.4 will use, checking: same `(det, weight)` multiset in any insertion order → **bitwise-identical** iterated sum; and **bitwise-identical after N reuse cycles** (the property `unordered_map` fails, T2 invariant 3) | an independent `std::map`-based reference sum on random `(det, weight)` fixtures with deliberate duplicate keys | the accumulator is not actually reuse-stable → pick a different structure (the three candidates below) before wiring it in |
| **H2.3** | measure, in isolation, the cost of **re-seeding 64 `mt19937_64` engines in place** vs constructing 64 fresh — the number T2 flagged as unseparated | a microbenchmark of exactly that one difference, identical seed sequence | re-seed is ≈ as expensive as construct → H2.5 needs a counter-based RNG (Philox/Threefry), a bigger change; decide here, not mid-rewrite |
| **H2.4** | `SpawnWorkspace` type + `void propagate_stochastic(..., SpawnWorkspace& ws, WalkerPopulation& out)` — the persistent bins (using H2.2's accumulator), partition buffer, and merge target live in `ws`; the output is caller-owned so there is no NRVO question | **bitwise identical to the pre-H2 tree at `OMP_NUM_THREADS` = 1** on `h2_fciqmc_sto3g` and `n2_fciqmc_sto3g` (serial, so no reordering — a swap of the accumulator that changes a serial result is a bug, not reassociation) | serial result changes → the accumulator swap reordered something it should not have; localize with H2.2's reference before proceeding |
| **H2.5** | move the 64 RNG engines into `ws`, re-seeded per call from one `rng.raw64()` draw (the S1 contract: fresh, thread-count-independent bin streams). If H2.3 said re-seed is too costly, this step is instead "swap `RandomSource`'s engine for a counter-based one" — a deliberate, separately-recorded RNG change | self-reproducibility at fixed seed **plus** `metric_within_sigma` against exact FCI (T2 invariant 2 — never bitwise-vs-the-old-numbers, since changing the RNG or its call pattern is a reordering-class change) | reproducibility fails, or the FCI-agreement sigma blows past the gate → the bin-stream derivation is wrong (the S1 "frozen trajectory" trap: a `const derive()` that does not advance); check the population diagnostics, not just the gate |
| **H2.6** | re-enable threading on H2.4's structure (`#pragma omp parallel for schedule(static)` over the persistent bins) and re-verify invariance | **bitwise identical across `OMP_NUM_THREADS` = 1/2/4/8** on `h2_fciqmc_threads1/4`, the new S5 N2-sized pair, `n2_fciqmc_sto3g`, and the four non-QMC FCI gates sharing `build_all_mo_ci_setup` | any thread count disagrees → the accumulator or the merge is not partition-deterministic after all; H2.2's reuse-stability test missed the threaded-write case, extend it |
| **H2.7** | re-run the H1 probe on **HF/6-31G** (the unsaturated fixture — not N2): per-call parent count, region µs, whole-call µs at 1/2/4/8 threads, before/after the rewrite. Report serial-scaffolding µs/call (should drop from ~1.5 ms toward near zero) and whole-call speedup vs the region ceiling | the H1 numbers already recorded (`region 3.44×/4t, 4.47×/8t`; `whole 2.24×/4t`) | whole-call speedup does *not* move toward the region ceiling → the ~1.5 ms was not actually the bottleneck; re-profile with an in-binary phase probe (T2's `PLANCK_FCIQMC_PHASE_PROBE` pattern) before concluding |

**The accumulator (H2.2/H2.4) — three candidates, evaluate in this order:**

1. **Sorted `std::vector<std::pair<DetKey, Weight>>` + `std::inplace_merge`.**
   Order is total and reuse-stable by construction; `.clear()` genuinely
   resets a vector. The spawn appends unsorted, then one sort + a dedup
   pass that sums equal keys. Simplest to reason about and to test; the
   likely first choice.
2. **Flat open-addressing hash table, fixed capacity, defined probe order,
   `.clear()` = memset the control bytes.** Faster inserts than sort, and
   `.clear()` is genuinely O(capacity) with no bucket-layout memory — but
   it is a new data structure to get right, and the "defined probe order"
   has to be actually defined (linear probing from `hash % cap`, no
   Robin-Hood reordering).
3. **Two-level: fixed bin array of small sorted vectors.** Only if 1 and 2
   both measure poorly.

**Non-negotiable across all of H2:**

- The invariance gate is `atol = 0.0` and must be non-vacuous (H2.1). A
  tolerance would hide the reduction-order defect the whole design exists
  to prevent.
- No `omp atomic` on the accumulator, no completion-order merge. Fixed bin
  order, partition by `hash(parent) % kBins` with `kBins` a fixed count
  (H2 does *not* change `kBins` — that is H2.0's job, and H1 showed it is
  N2-class-only).
- Any RNG change (H2.5's fallback) is recorded as a deliberate decision
  with its own before/after `metric_within_sigma`, per
  `FCIQMC_RESEARCH_SCOPE.md` §6 and T2 invariant 2 — not slipped in.

**If H2 holds** (H2.7 shows the whole call tracking toward the region
ceiling on HF): the rewrite is bounded — one new `SpawnWorkspace` type,
one changed signature, one accumulator swap, one RNG hoist, all behind the
existing invariance discipline. **If H2.7 shows the serial scaffolding was
not the bottleneck**, the remaining cost is inside the merge itself (T2
already found merge time *rising* with thread count, unexplained) and the
next step is a merge-specific investigation, not more of H2.

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
  (S5: an N2-sized `threads1/threads4` pair — N2 is fine *here*, since the
  invariance gate needs multi-parent bins, not throughput, and N2's 908
  parents ÷ 64 bins already gives that — that goes red when the merge
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

- **H1 — DONE (2026-09-06), and its own first conclusion was fixture-
  bound and wrong.** Measured on N2/STO-3G first: region threads to only
  2.3× at 4 threads and regresses at 8 — read as a general granularity
  ceiling. Then measured on HF/6-31G (`ndet` = 213k, unsaturated): region
  threads to **3.44× at 4 threads and 4.47× at 8, still climbing**. The N2
  ceiling was a *saturation artifact* — N2's 14,400-det space is full at
  gate walker counts, giving ~14 parents/bin; HF gives ~430/bin and
  threads like the FCI sigma build. The per-parent work (~148–202 ns,
  flat in walker count) is real on both. **H1 is refuted as stated: the
  work is not too small on any non-saturated fixture, which is exactly the
  Q1 large-active-space case.**
- **H2 — SCOPED into seven verifiable steps (H2.1–H2.7), not started.**
  On HF the whole call is stuck at 2.24×/4 threads against a region
  ceiling of ≥ 4.5×, because the serial scaffolding is ~1.5 ms/call (30×
  N2's, since HF partitions/merges 30× more parents) and does not thread.
  The ladder: **H2.1** build a non-vacuous N2-sized invariance gate (S5),
  **H2.2** unit-test a reuse-stable accumulator, **H2.3** measure the RNG
  re-seed cost in isolation (decides whether H2.5 needs a counter-based
  RNG), **H2.4** the `SpawnWorkspace` type + out-param signature +
  accumulator swap (gated bitwise-serial), **H2.5** hoist the RNG engines
  (gated by reproducibility + FCI-sigma), **H2.6** re-thread and re-verify
  invariance at 1/2/4/8, **H2.7** re-measure on HF against the region
  ceiling. Each step's own verification gates the next.
- **H2.0 (smaller fixed `kBins`) — reserve, N2-class only.** Helps a
  saturation-starved fixture; on HF the bins are already large enough.
  H2 does **not** touch `kBins`.
- **H3 — not started.** Design sketch + memory estimate only.
- **Fixture decision: use HF/6-31G (or larger) for all future parallel-
  FCIQMC measurement.** N2/STO-3G stays the *correctness* gate (small
  space, real sampling) but is a misleading *throughput* fixture.

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
