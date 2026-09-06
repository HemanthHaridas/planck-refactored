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
| **H2.1 — DONE** (see below) | the S5 gate: an **N2-sized** `threads1`/`threads4` pair at `atol = 0.0`. **The `threads1` case PINS both energies to fixed values** (not just `metric_present`), because a merge-order defect is thread-count-invariant and a `threads4 == threads1` comparison alone cannot see it. | the current tree (passes as-is); non-vacuity verified with two mutation classes | — |
| **H2.2 — DONE** (see below) | `SpawnAccumulator` (`src/post_hf/ci/spawn_accumulator.h`, header-only) — candidate 1, a flat `vector<pair<DetKey,Weight>>` that `finalize()` sorts by a **total order on `(alpha, beta, bit-pattern-of-weight)`** then folds equal-key runs left-to-right, so the sum is a pure function of the multiset. Gated by `planck-fciqmc-accumulator` against an independent `std::map`-based reference. | insertion-order invariance (5 shuffles × 20 seeds), reuse stability (10 grow-then-shrink cycles), `finalize()` idempotence, and a non-vacuity check that the canonical fold actually differs from an insertion-order fold | — |
| **H2.3 — DONE** (see below) | isolated microbenchmark of the 64 per-bin RNG streams three ways: (A) current fresh-vector + 64× `derive()`, (B) persistent vector + 64× `mt19937_64::seed()` in place, (C) persistent + a counter-based stream (`reseed` = set a key) | 200k iterations each, identical seed sequence | **RESULT: re-seed IS ≈ as expensive as construct** (A 42 µs, B 39 µs — the ~2 µs gap is only the vector alloc). **`mt19937_64::seed()` is ~600 ns each × 64 = ~38 µs/call, and reusing the engine object cannot avoid it.** C is ~74 ns. **H2.5 needs a counter-based RNG** — this was NOT a foregone conclusion, T2 called the state-fill "unavoidable" having only tried B |
| **H2.4 — DONE** (see below) | `SpawnWorkspace` type (`spawn_accumulator.h`) + `void propagate_stochastic(..., SpawnWorkspace& ws, WalkerPopulation& out)` with the value-returning form kept as a thin convenience overload for the ~20 test call sites. Persistent bins (`SpawnAccumulator`), partition buffer, merge target in `ws`, built once by the driver. Output caller-owned → no NRVO question. RNG stays per-call (that is H2.5). | self-reproducibility at fixed seed **+** thread-count invariance at `atol = 0.0` (1/2/4/8) **+** `metric_within_sigma` vs exact FCI — **NOT bitwise-vs-the-pre-H2 numbers**: swapping `unordered_map` (bucket-order iteration) for `SpawnAccumulator` (canonical-sorted-order fold) is a legitimate reassociation, exactly the R2 category, so the S5 pin re-pins to the new value | reproducibility fails, or invariance breaks, or the FCI-sigma blows the gate → localize with `planck-fciqmc-accumulator`'s reference before proceeding |
| **H2.5 — DONE** (see below) | `RandomSource`'s engine swapped `std::mt19937_64` → **xoshiro256\*\*** (256-bit state from a SplitMix64 fill, O(1) reseed). The 64 per-bin streams now live in `ws.bin_rngs` and are re-keyed per call by `ws.rekey_streams(rng.raw64())` — same `derive()` recipe, no 64 constructions. Public surface unchanged (`uniform`/`uniform_int`/`stochastic_round`/`raw64`/`derive`/`seed`). | self-reproducibility at fixed seed **+** thread-count invariance `atol=0.0` (1/2/4/8) **+** `metric_within_sigma` vs exact FCI — never bitwise-vs-old, the RNG swap changes every trajectory | reproducibility fails, or the FCI sigma blows the gate → the bin-stream derivation is wrong (the S1 "frozen trajectory" trap: a `const derive()` that does not advance); check the population diagnostics, not just the gate |
| **H2.6** | re-enable threading on H2.4's structure (`#pragma omp parallel for schedule(static)` over the persistent bins) and re-verify invariance | **bitwise identical across `OMP_NUM_THREADS` = 1/2/4/8** on `h2_fciqmc_threads1/4`, the new S5 N2-sized pair, `n2_fciqmc_sto3g`, and the four non-QMC FCI gates sharing `build_all_mo_ci_setup` | any thread count disagrees → the accumulator or the merge is not partition-deterministic after all; H2.2's reuse-stability test missed the threaded-write case, extend it |
| **H2.7** | re-run the H1 probe on **HF/6-31G** (the unsaturated fixture — not N2): per-call parent count, region µs, whole-call µs at 1/2/4/8 threads, before/after the rewrite. Report serial-scaffolding µs/call (should drop from ~1.5 ms toward near zero) and whole-call speedup vs the region ceiling | the H1 numbers already recorded (`region 3.44×/4t, 4.47×/8t`; `whole 2.24×/4t`) | whole-call speedup does *not* move toward the region ceiling → the ~1.5 ms was not actually the bottleneck; re-profile with an in-binary phase probe (T2's `PLANCK_FCIQMC_PHASE_PROBE` pattern) before concluding |

#### H2.1 result (2026-09-06): the S5 gate needs a PINNED `threads1`, not just a `threads1`/`threads4` comparison

Landed as `n2_fciqmc_s5_threads1` / `n2_fciqmc_s5_threads4`
(`tests/inputs/regression/post_hf/n2_fciqmc_s5_short.hfinp` — N2/STO-3G,
300 equil + 200 sampling, ~0.06 s/run, ~579 occupied determinants → ~9
parents/bin, so cross-bin annihilation is genuinely exercised, unlike the
`h2_fciqmc_threads1/4` case where 4 determinants < `kBins` = 64 means ≤ 1
parent/bin). Both `extended`-tagged. Fully deterministic — bit-identical
across 3 repeats at 1 thread and across `OMP_NUM_THREADS` = 1/2/4/8 on the
clean tree:

```
fciqmc_shift_energy      -109.2733562973
fciqmc_projected_energy  -107.5441088131
```

`n2_fciqmc_s5_threads1` `metric_close`s both at `atol = 0.0` to those
values; `n2_fciqmc_s5_threads4` `metric_close_case`s both against
`threads1` at `atol = 0.0`.

**Non-vacuity verified with two mutation classes, and the finding is that
the `threads1` pin is load-bearing on its own:**

| mutation | effect | `threads1` pin | `threads4 == threads1` |
|---|---|---|---|
| **reverse the bin-merge order** (`next_bins.rbegin()..rend()`) | shift `-109.2733562973` → `-109.5354023681`; **thread-count-invariant** (T1 and T4 both move to the new value) | **RED** ✓ | green — cannot see it |
| **`local_bin` per bin + `#pragma omp critical` completion-order merge** (the DFT-grid-jitter defect) | value changes AND T1 (`-108.7298896908`) ≠ T4 (`-109.5866265932`) | **RED** ✓ | **RED** ✓ |

So a merge-order or accumulator-reassociation change that preserves
thread-count invariance — which is exactly the failure mode H2.4's
accumulator swap risks — is invisible to a `threads1`/`threads4`
comparison and is caught **only** by pinning `threads1` to a known-good
value. The existing `h2_fciqmc_threads1` case (`metric_present` only) does
not have this property; the S5 pair is stricter on purpose.

No production code changed for H2.1 — one new input file, two new JSON
cases. All FCIQMC gates plus smoke (35/35) pass.

#### H2.2 result (2026-09-06): candidate 1 (sorted vector) built and gated; the within-key fold order is the load-bearing detail

`SpawnAccumulator` (`src/post_hf/ci/spawn_accumulator.h`, header-only,
Eigen-free-under-test):

- **`add(det, w)`** just appends `(det, w)` to a `std::vector` — order-
  immaterial, cheap.
- **`finalize()`** `std::sort`s the whole vector by a **total order on
  `(alpha, beta, std::bit_cast<uint64_t>(weight))`**, then folds
  consecutive equal-key runs left-to-right. The critical detail, found
  while writing the test: sorting by key alone is **not** enough — a
  `std::sort` is not stable, so a ≥ 3-long same-key run would fold in an
  arbitrary order and the sum would still depend on the insertion order
  (IEEE `+` is commutative but not associative). Adding the weight-bit
  tiebreak makes the run's fold order `((w_a + w_b) + w_c)` with
  `w_a ≤ w_b ≤ w_c` by bit pattern — a pure function of the multiset.
- **`reset()`** is `vector::clear()` (keeps capacity, carries no bucket-
  layout memory) — the genuine reset `unordered_map` cannot do.

Gated by `planck-fciqmc-accumulator` (`tests/fciqmc_accumulator.cpp`)
against an **independent** `std::map<DetKey, vector<double>>` reference
that groups by key and folds each key's per-key vector in the same
ascending-bit order — the same canonical sum arrived at a different way
(per-key grouping vs. one global sort-and-scan). Checks:

| test | what |
|---|---|
| insertion-order invariance | 20 structured multisets (40 keys, runs of 1–6, magnitudes 1 → 1e-9, exact-cancellation pairs), each shuffled 5 ways → all 5 finalized sums bitwise-identical to the reference |
| reuse stability | 10 cycles of `reset()` → fill with a *larger* multiset → `reset()` → refill with the small one → require bitwise-identical to cycle 0 (the exact shape that made a reused `unordered_map` iterate differently) |
| `finalize()` idempotent | second call does not change the bytes |
| non-vacuity | assert the canonical fold *does* differ from an insertion-order fold on ≥ 1 of 50 seeds — so the invariance test is not silently untestable |
| hand case | `add(1.0); add(1e-16); add(-1.0)` folds as `(1e-16 + 1.0) + (-1.0) == 0.0`, the 1e-16 lost — confirms the ascending-bit fold order concretely |

**Mutation-verified both ways** on a throwaway edit: dropping the
weight-bit tiebreak → 101 failures (insertion-order + reference); making
`reset()` not clear `entries_` → 10 failures (small refill carries the big
multiset). Reverted.

`inplace_merge` (the scope's stated mechanism for candidate 1) was not
used — a single `std::sort` + one dedup scan is simpler and the spawn
never produces two pre-sorted halves to merge. Candidates 2 and 3 stay in
reserve if H2.7 shows the sort cost matters (the sort is over ~430
entries/bin on HF, `O(n log n)` with n ≈ 430, against the ~87 µs of
arithmetic per bin — unlikely to dominate, but H2.7 measures it).

No production code changed for H2.2 either — `SpawnAccumulator` has no
caller until H2.4. All FCIQMC unit tests + smoke (35/35) pass.

#### H2.3 result (2026-09-06): reusing the mt19937 engines does NOT help — H2.5 needs a counter-based RNG

The current per-call RNG setup (`propagate_stochastic`, `fciqmc.cpp`):
`RandomSource call_source(rng.raw64())`, then a fresh
`std::vector<RandomSource>` filled with 64× `call_source.derive(b)` —
each `derive()` **constructs a fresh `RandomSource`**, which constructs a
fresh `std::mt19937_64` (a 312×64-bit state fill).

Standalone microbenchmark (`h23_rng_bench.cpp`, scratchpad — not a kept
gate, like the H1 probe), mirroring `RandomSource`/`derive()` exactly,
200k iterations, three ways:

| way | per call | vs A |
|---|---|---|
| **A** current: fresh vector + 64× `derive()` (64 mt19937 constructions) | **~42 µs** | 1.00× |
| **B** persistent `vector<RandomSource>(64)` + 64× `mt19937_64::seed()` in place (no vector alloc) | **~39 µs** | 0.93× |
| **C** persistent + a counter-based stream (`reseed` = set a 64-bit key) | **~74 ns** | 0.002× |

Breakdown:

- **A − B ≈ 2 µs** — the vector allocation only. This is the ~4% R1/R2
  measured (26.8 → 25.8 µs on N2) and correctly judged not worth a diff.
- **B − C ≈ 38 µs** — the `mt19937_64` state fill. Isolated separately:
  `std::mt19937_64::seed()` costs **~600 ns each** on libstdc++ (a 312-word
  LCG fill), so 64 of them = ~38 µs, against a *draw* at ~1.4 ns.

**T2 called this cost "real and unavoidable given S1's correctness
requirement" — that was wrong.** T2 only tried option B (reuse the engine
object), which still has to `seed()` it. S1's requirement is that each
call's 64 bin streams be fresh, independent, and thread-count-independent
— a counter-based generator satisfies that with an O(1) "seed" (set the
key), which is option C. So **H2.5 is not "hoist the mt19937 engines into
the `SpawnWorkspace`" — it is "replace `RandomSource`'s `mt19937_64` with a
counter-based engine"**, a real (but small and self-contained) RNG change,
gated by reproducibility + `metric_within_sigma` against exact FCI per
T2 invariant 2.

**Which counter-based engine, for H2.5 to decide:** SplitMix64 (already in
this codebase, in `DetKeyHash` and `derive()`) is the trivial choice but
is a 64-bit-state sequential generator — adequate for MC, marginal for
millions of draws per stream. xoshiro256** or a Philox-4×64 counter mode
are the defensible choices; xoshiro256** is ~10 lines, no `<random>`
dependency, and its "seed" is a 256-bit state set. H2.5 picks one and
records the choice.

**Context for the payoff:** on HF/6-31G the whole call is ~7 ms at 1
thread, so ~38 µs is ~0.5% there — but at 4 threads the parallel region
drops to ~1.6 ms and the ~1.5 ms of serial scaffolding is the ceiling
(H1), so removing ~38 µs is a real ~2.5% of *that*, and it removes a fixed
per-call cost that does not shrink with threads. Worth doing as part of
H2.5, not on its own.

#### H2.4 result (2026-09-06): the workspace + accumulator swap is in; the "bitwise-vs-pre-H2" gate in the ladder table was wrong

Landed:

- **`SpawnWorkspace`** (`src/post_hf/ci/spawn_accumulator.h`) — holds
  `vector<SpawnAccumulator> bins` + `vector<vector<pair<DetKey,Weight>>>
  parents`, sized to `kBins` on first use, `reset_for_call()` clears
  without freeing. The RNG is deliberately *not* here yet (H2.5).
- **`void propagate_stochastic(..., SpawnWorkspace& ws, WalkerPopulation&
  out)`** — the production form. `out.clear()` first, no return value, so
  the M1 NRVO reversion is done properly (by signature, not a local
  `static`). The old value-returning signature is kept as a thin overload
  (`SpawnWorkspace ws; ... ; return out;`) so none of the ~20
  `fciqmc_walkers.cpp` call sites change.
- The per-bin accumulator is now `SpawnAccumulator` instead of
  `WalkerPopulation`'s `unordered_map`; each bin `finalize()`s at the end
  of its loop; the merge iterates bins in fixed order and each bin in its
  canonical `(alpha, beta, weight-bits)` order.
- **The driver** (`fciqmc_driver.cpp`) builds one `SpawnWorkspace` before
  the step loop and passes it to every call.

**The ladder table's H2.4 gate ("bitwise identical to the pre-H2 tree at
`OMP_NUM_THREADS` = 1") was wrong, for the same reason R2 was gated the
way it was.** Swapping `unordered_map` (iterate in bucket order) for
`SpawnAccumulator` (fold each key's run in canonical order, iterate keys
in sorted order) reassociates a fixed multiset's sum — a legitimate
floating-point reordering, not a defect. The N2 S5 shift moved
`-109.2733562973` → `-108.7582220854`. The **correct** gate, applied:

| check | result |
|---|---|
| self-reproducibility (fixed seed, 5 runs) | bit-identical `-108.7582220854` every run |
| thread-count invariance `atol = 0.0` at 1/2/4/8 | bit-identical at all four |
| FCI agreement — `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g` (`metric_within_sigma` vs exact FCI) | both PASS |
| `h2_fciqmc_threads1/4`, `n2_fciqmc_s5_threads4` | PASS |
| S5 pin `n2_fciqmc_s5_threads1` | re-pinned to `-108.7582220854` / `-107.6009067556` |
| S5 non-vacuity re-checked on the *new* code path | reversing the merge order still fails the pin while T1==T4 holds ✓ |
| `planck-fciqmc-walkers` (56 s), `planck-fciqmc-accumulator`, smoke (35/35) | all pass |

**A build-hygiene trap cost a wrong number once:** a `cmake --build` right
after a `git`-style revert produced a binary giving `-108.9523080219`; a
forced clean recompile of `fciqmc.cpp` gave the reproducible
`-108.7582220854`. "A build in flight is not pinned to the working tree" —
`touch` the sources and rebuild before trusting a post-revert measurement.

No timing measured here — H2.4 is a correctness step. The scaffolding-cost
payoff is H2.7's job, after H2.5 (the RNG) and H2.6 (re-threading).

#### H2.5 result (2026-09-06): `RandomSource` is now xoshiro256\*\*, streams hoisted into `ws`

`RandomSource`'s engine (`src/post_hf/ci/fciqmc.h`) is **xoshiro256\*\***
(Blackman & Vigna) instead of `std::mt19937_64`:

- state is `uint64_t[4]`, filled from a SplitMix64 stream (the canonical
  xoshiro seeding recipe) — `reseed(seed)` is 4 multiplies, O(1), against
  mt19937's ~600 ns 312-word fill.
- `uniform()` is `(next() >> 11) * 2^-53` — the same 53-bit mantissa the
  old `std::generate_canonical<double, 53>` produced.
- the public surface is byte-for-byte the same
  (`uniform`/`uniform_int`/`stochastic_round`/`raw64`/`derive`/`seed`), so
  every call site — `draw_excitation`, `stochastic_round`, the driver's
  `RandomSource rng(opt.seed)` — is unchanged.

The 64 per-bin streams moved into `SpawnWorkspace::bin_rngs` and are
re-keyed each call by `rekey_streams(rng.raw64())` — the identical
`RandomSource call_source(seed); ... call_source.derive(b)` recipe, in
place, no 64 constructions.

**Gated (never bitwise-vs-old — the RNG swap changes every trajectory):**

| check | result |
|---|---|
| self-reproducibility (fixed seed, 5 runs) | bit-identical `-108.5036712075` |
| thread-count invariance `atol=0.0` at 1/2/4/8 | bit-identical |
| FCI agreement — `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g` (`metric_within_sigma`) | PASS |
| `h2_fciqmc_threads1/4`, `n2_fciqmc_s5_threads4` | PASS |
| S5 pin `n2_fciqmc_s5_threads1` re-pinned to `-108.5036712075` / `-107.5859595426` | PASS |
| S5 non-vacuity re-checked on the new engine (reverse merge order) | fails the pin while T1==T4 holds ✓ |
| `planck-fciqmc-walkers` (all RNG-repro / statistical / `p_gen` / blocking tests) | PASS |
| smoke (35/35), extended (see below) | PASS |

**Incidental speedup, not the point but recorded:** removing ~38 µs/call
of mt19937 seeding over ~50,000 calls — `n2_fciqmc_sto3g` **11.2 s →
9.2 s**, `planck-fciqmc-walkers` **56 s → 24 s** (it re-keys streams far
more often per unit work than the real driver). H2.7 measures the actual
scaffolding-removal payoff on HF; this just confirms the H2.3 arithmetic.

**The accumulator — remaining candidates, only if H2.7 shows the sort matters:**

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
- **H2 — SCOPED into seven verifiable steps (H2.1–H2.7). H2.1 + H2.2 + H2.3
  + H2.4 + H2.5 DONE; H2.6–H2.7 not started.**
  On HF the whole call is stuck at 2.24×/4 threads against a region
  ceiling of ≥ 4.5×, because the serial scaffolding is ~1.5 ms/call (30×
  N2's, since HF partitions/merges 30× more parents) and does not thread.
  The ladder: **H2.1 (DONE)** the S5 invariance gate —
  `n2_fciqmc_s5_threads1/4`, N2-sized (~9 parents/bin), with `threads1`
  **pinning both energies** because a merge-order defect is
  thread-count-invariant and a `threads1`/`threads4` comparison alone
  cannot see it. **H2.2 (DONE)** `SpawnAccumulator`
  (`src/post_hf/ci/spawn_accumulator.h`) — a sorted `vector<pair<DetKey,
  Weight>>` whose `finalize()` sorts by `(alpha, beta, weight-bits)` and
  folds equal-key runs left-to-right, so the sum is a pure function of the
  multiset. Gated by `planck-fciqmc-accumulator`, mutation-verified. The
  weight-bit tiebreak is load-bearing: `std::sort` is not stable, so a
  ≥ 3-long same-key run would otherwise still fold in insertion order.
  **H2.3 (DONE)** the 64 per-bin RNG cost, isolated three ways: fresh
  construct 42 µs, mt19937 reuse+`seed()` 39 µs (the ~2 µs gap is only the
  vector alloc — R1/R2's 4%), counter-based 74 ns. **`mt19937_64::seed()`
  is ~600 ns × 64 ≈ 38 µs and reusing the engine cannot avoid it — T2's
  "unavoidable" was wrong, having only tried reuse.** So **H2.5 = replace
  `RandomSource`'s mt19937 with a counter-based engine** (xoshiro256** /
  Philox), not just hoist it. **H2.4 (DONE)** `SpawnWorkspace` +
  `void propagate_stochastic(..., ws, out)` (value-returning form kept as a
  thin overload for the ~20 test call sites); per-bin accumulator swapped
  `unordered_map` → `SpawnAccumulator`; driver builds one workspace before
  the step loop. **The ladder's "bitwise-vs-pre-H2" gate was wrong** — the
  accumulator swap reassociates the serial sum (R2 category), so gated on
  self-reproducibility + thread-count invariance (1/2/4/8, `atol=0.0`) +
  FCI agreement instead; S5 re-pinned. **H2.5 (DONE)** `RandomSource`'s
  engine swapped `mt19937_64` → **xoshiro256\*\*** (O(1) reseed from a
  SplitMix64 fill), 64 streams hoisted into `ws.bin_rngs`, re-keyed in
  place. Same public surface, same 53-bit `uniform()`. Gated on
  reproducibility + invariance + FCI-sigma; S5 re-pinned again. Incidental:
  `n2_fciqmc_sto3g` 11.2 s → 9.2 s, `planck-fciqmc-walkers` 56 s → 24 s.
  **H2.6** re-thread and re-verify invariance at 1/2/4/8, **H2.7**
  re-measure on HF against the region ceiling. Each step's own
  verification gates the next.
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
