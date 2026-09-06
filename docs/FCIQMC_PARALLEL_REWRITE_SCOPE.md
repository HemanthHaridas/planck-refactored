# Scope: Can FCIQMC Be Rewritten to Be Genuinely Parallel?

**Investigation DONE. H1 answered (fixture-dependent; the N2 ceiling was a
saturation artifact), and H2's `SpawnWorkspace` / `SpawnAccumulator` /
xoshiro rewrite LANDED (H2.1–H2.7).** Bottom line: on the interesting
(unsaturated) fixture HF/6-31G the whole-call speedup went **2.24×→2.42×
at 4 threads and 2.47×→2.87× at 8 threads**, the RNG hoist landed its full
~38 µs/call.

**H2.8 tried to shard the "remaining serial merge" — built it, verified
correct, measured 3–8 % SLOWER on HF/6-31G at every thread count, reverted,
then reprofiled properly with `sample`. The H2.7 phase-probe diagnosis was
wrong: the serial merge is 2.0 % of self-time, not 37 %.** What the probe
bracketed was the per-bin `SpawnAccumulator::finalize()` sort (5.5 %) —
which is already inside the `#pragma omp` region and already threads — plus
accumulator-vector allocation. The real hot path is **`draw_excitation` at
40.8 %** (in the region) and a ~10–15 % serial tail of small per-step
driver passes (`ordered_l1_norm`, `compress`, `signed_population`, the
diagonal prefill), none of them the merge. Full profile table in H2.8.3.

A 4-thread reprofile (H2.8.4) confirms `draw_excitation` stays ~40 % of
*useful* work at 4 threads (it threads fine) while **~54 % of the machine
sits idle at the barrier** on the serial per-step driver tail. Two
targets: `draw_excitation` serial efficiency (1-thread wall) and
collapsing the serial per-step map passes (4-thread wall).

**H2.8.4-a LANDED the first (2026-09-06):** `draw_excitation` no longer
builds four 128-byte `OrbitalList` per call — it works off four
`std::popcount`s and a bounded `nth_set_bit` select. Bit-identical
(pure-arithmetic refactor, S5 1/2/4/8 unchanged, `p_gen` oracle passes,
mutation-verified 19 failures). Measured **HF/6-31G 1-thread, `verbosity
normal` (production): 78.1 s → 60.5 s (−22 %)** (−20 % on the older
`verbose` fixture, which pays an extra `signed_population` pass); N2 gate
12.86 s → 11.56 s; `draw_excitation` self-time **40.8 % → 27.1 %**.
4-thread ~unchanged (barrier idle dominates).

**H2.8.4-b SCOPED (2026-09-06), not started:** the serial per-step tail is
**4–5 full traversals of the ~26k-entry walker hash map per step**
(diagonal prefill, partition, merge, `compress`, `ordered_l1_norm`;
+`signed_population` at `verbose`). Plan: fold the four
`for (det,w) : pop)` passes into ONE — compute `hash(det)` once, do all
per-entry work, hand `propagate_stochastic` a pre-partitioned
`bin_parents` + the L1 norm + the compressed set; then thread that single
pass with the same fixed-64-bin discipline the spawn region uses. Target:
54 % idle → ~25–30 %. Section H2.8.4-b below. Step 2 (the `verbose`
measurement correction) is already done.

H3 (replica parallelism) remains a design sketch. All of it stays gated on
a real FCIQMC workload appearing (`FCIQMC_RESEARCH_SCOPE.md` Q1) — nothing
in the tree runs FCIQMC at a size where it matters yet — but H2.8.4-a is
on the branch and ready.

Original framing follows.

---

The incremental threading of the spawn loop was done and measured
(`docs/FCIQMC_T2_THREADING.md`): 1.57× at 4 threads against a ~3× ceiling,
with the gap diagnosed as per-call serial scaffolding that exceeds the
parallel region. This scope asked the different question the T2 work
deliberately did not: **is the 1.57× a property of FCIQMC, or a property
of how this FCIQMC is shaped — and would a rewrite of the data structures
and the call granularity get genuine (near-linear, or at least solidly
super-2×) parallelism?**

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
| **H2.6 — DONE, NO CODE CHANGE** (see below) | verification only — the `#pragma omp parallel for schedule(static)` survived H2.4/H2.5 intact, so "re-enable" was a no-op; the threading was already correct on the new `SpawnWorkspace`/`SpawnAccumulator`/xoshiro structure. | **bitwise identical across `OMP_NUM_THREADS` = 1/2/4/8** on `h2_fciqmc_threads1/4`, the S5 N2 pair, `n2_fciqmc_sto3g` (50k steps, deepest multi-parent bins), and the four non-QMC FCI gates sharing `build_all_mo_ci_setup` — all pass; and `schedule(dynamic)` was checked too and *stays* invariant (unlike the FCI sigma build) because here the accumulator partition does not depend on the schedule | — |
| **H2.7 — DONE** (see below) | re-ran the H1 probe on HF/6-31G, 50k walkers, at 1/2/4/8 threads, with a per-phase split | H1's numbers | **PARTIAL WIN.** whole-call 2.24×→**2.42×**/4t, 2.47×→**2.87×**/8t. The RNG hoist landed exactly as H2.3 predicted (rekey 38 µs → **0.14 µs**). But serial scaffolding is still ~1270 µs/call, and it is now **~1100 µs of merge** — untouched by H2.4–H2.6 and not threaded. The merge is the last lever. |

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

#### H2.6 result (2026-09-06): no code change — the `#pragma omp` was already correct on the new structure

The `#pragma omp parallel for schedule(static)` over the 64 bins was never
removed through H2.4 or H2.5, so "re-enable threading" was a no-op. Every
invariance check at 1/2/4/8 through H2.4 and H2.5 already passed; H2.6 is
the dedicated, thorough re-verification the ladder called for:

| gate | 1/2/4/8 result |
|---|---|
| `n2_fciqmc_s5_short` (S5 pair) | bitwise-identical |
| `h2_fciqmc_sto3g` | bitwise-identical |
| `n2_fciqmc_sto3g` (50k steps, ~900 parents/call, deepest multi-parent bins — the strongest invariance evidence) | bitwise-identical |
| `h2_fci_sto3g`, `water_fci_sto3g`, `o2_fci_rohf_sto3g`, `be_fci_spherical_631gd` (non-QMC, share `build_all_mo_ci_setup`) | bitwise-identical, and equal to their committed references |

**H2.2's serial-only accumulator test is sufficient** — the parallel-for
unit is one bin, `schedule(static)` gives each thread a disjoint
contiguous range, and every write inside the region targets only
`next_bins[bin]`/`bin_rngs[bin]`. No `SpawnAccumulator` is written by two
threads, so there is no threaded-write case for the test to have missed.

**`schedule(dynamic)` was checked and *stays* invariant** — bitwise-
identical at 1/4/8 with `dynamic`. This is the opposite of the FCI sigma
build, where `dynamic` broke invariance because *there* the accumulator
partition depended on the schedule; here it does not (bin = fixed
`hash(parent) % kBins`, merge = fixed bin order), so which thread computes
which bin is irrelevant. `static` is kept anyway (locality, and the T2
doc pins it).

#### H2.7 result (2026-09-06): partial win — the RNG hoist landed, but the merge is now the whole serial ceiling

Re-ran the H1 probe (`PLANCK_FCIQMC_H1_PROBE`, re-added, measured,
reverted) on HF/6-31G, 50k walkers, with a per-phase split, at 1/2/4/8
threads. Post-H2 rewrite vs the H1 baseline:

| threads | region µs (H1 → now) | whole µs (H1 → now) | whole speedup (H1 → now) |
|---|---|---|---|
| 1 | 5569 → 5867 | 7039 → 7139 | 1.00× |
| 2 | 3129 → 3062 | 4652 → 4311 | 1.51× → 1.66× |
| 4 | 1617 → 1659 | 3139 → **2957** | 2.24× → **2.42×** |
| 8 | 1245 → 1198 | 2849 → **2492** | 2.47× → **2.87×** |

Per-phase, at 4 threads (µs/call):

| phase | µs | scales with threads? |
|---|---|---|
| **rekey (RNG)** | **0.14** | — was ~38 µs pre-H2.5; **H2.3's arithmetic confirmed in the real binary** |
| partition | ~170 | no |
| **merge** (walk 64 finalized accumulators, `out.add` ~27,500 entries) | **~1100** | **no** |
| region (the `#pragma omp` block) | 1659 | yes (3.54×/4t) |
| serial-outside-region total | **~1270** | no |

**What the rewrite bought:** the RNG hoist landed its full ~38 µs/call
(rekey is now 0.14 µs — 270× cheaper), and removing the 64-bin
construction moved the whole-call speedup 2.24→2.42×/4t and 2.47→2.87×/8t.
The `n2_fciqmc_sto3g` gate dropped 11.2 s → 8.6 s along the way.

**What it did NOT buy:** the whole-call speedup is still ~1.5× short of
the region's 3.54×/4t, and the phase split shows why — **~1100 µs/call of
that ~1270 µs serial scaffolding is the merge**, which H2.4–H2.6 never
touched. H1's "~1.5 ms serial drag" was construction + RNG + partition +
merge; the rewrite removed the first two, leaving the merge as almost the
entire remainder, and it does not thread (it must stay in fixed bin order
for invariance).

**Verdict:** H2.7's stated failure criterion — "whole-call speedup does
*not* move toward the region ceiling" — was NOT triggered: it *did* move
(2.24→2.42, 2.47→2.87). But the move is bounded by the merge, which is now
the sole remaining serial cost. **The merge is the last lever, and it is
its own investigation, not more of H2** — parallelizing it reintroduces
the completion-order hazard the whole binning design exists to avoid, so
any approach has to keep the fixed bin order (e.g. a parallel prefix-sum
over per-bin sizes into one flat output array, then a threaded scatter by
precomputed offset — the scatter is disjoint by construction, the ordering
is fixed by the offsets). That is a smaller, careful target for when a
real FCIQMC workload exists (`FCIQMC_RESEARCH_SCOPE.md` Q1), the same gate
as everything else here.

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

**H2 held partway** (H2.7 result above): the rewrite is bounded and
landed — one new `SpawnWorkspace` type, one changed signature, one
accumulator swap, one RNG engine swap — and moved the whole call
2.24→2.42×/4t, 2.47→2.87×/8t. But **the serial scaffolding was only
half the bottleneck**: H2.4–H2.6 removed the 64-bin construction and the
~38 µs RNG seeding, but ~1100 µs/call of merge remains and does not
thread. That merge is scoped and its collision rate measured as **H2.8**
below (0.73 % on HF/6-31G → the merge is 99.27 % concatenation →
candidate 1: 64 output shards keyed `hash(child) % 64`, threaded fill,
serial fixed-order concat). Not built — gated on a real workload
(`FCIQMC_RESEARCH_SCOPE.md` Q1) the same as everything else.

### H2.8 — the fixed-order merge (dead-end) → the real hot path is `draw_excitation`; H2.8.4-a landed it 40.8 %→27.1 %, HF 1t −22 % (production); H2.8.4-b scopes the 4-thread serial tail

Scoped, built and reverted the merge shard (H2.8.1–H2.8.3), reprofiled
properly (the merge was never the bottleneck — H2.7's phase probe was
wrong, full record below), then scoped and **landed** the real 1-thread
lever: `draw_excitation` serial efficiency (H2.8.4-a — bit-identical,
`draw_excitation` self-time 40.8 %→27.1 %, HF/6-31G 1-thread wall −22 %
at production verbosity). The 4-thread wall — a ~54 % barrier idle on the
serial per-step driver tail (4–5 walks of the walker hash map per step) —
is scoped as **H2.8.4-b** below, not started. Still gated on a real Q1
workload before any of it leaves the branch.

**The one lever H2.7 left.** After H2.4–H2.6 the serial cost outside the
`#pragma omp` region on HF/6-31G (50k walkers, 4 threads) is ~1270 µs/call,
of which **~1100 µs is the merge** — `propagate_stochastic`, `fciqmc.cpp:636`:

```cpp
for (auto &bin : next_bins)             // 64 accumulators, fixed order
    for (const auto &[det, w] : bin)    // ~430 finalized entries/bin on HF
        out.add(det, w);                // fciqmc.h:107 — _walkers[det] += w
```

~27,500 `unordered_map<DetKey,Weight>::operator[] += ` per call, serial,
~40 ns each. It is 37 % of the 2957 µs whole call and 87 % of the
non-region serial cost; the region already threads to 3.54×/4t, so the
merge is exactly what caps the whole call at 2.42×.

#### Why it is not "just add a pragma"

The merge **must** stay a pure function of the 64 finalized multisets,
independent of thread count and completion order — that is the whole
reason `SpawnAccumulator` exists and why every bin is keyed by
`hash(parent) % kBins` and merged `0..63`. `#pragma omp` + `out.add`
straight away reintroduces:

- a data race on `_walkers` (concurrent `operator[]` + rehash), and
- if fixed with `omp critical` / `omp atomic`, **completion-order
  accumulation** — the DFT-grid-jitter defect this codebase specifically
  refuses (`docs/DFT_XC_REDUCTION_DETERMINISM`), and the exact mutation
  H2.1's S5 gate is built to catch.

So the target is a **fixed-order parallel scatter**: the output order is
fixed by precomputed offsets, not by which thread finishes first.

#### The shape (H2.7's sketch, made concrete)

`out` is `WalkerPopulation` = `unordered_map`. A cross-bin determinant
(same child spawned from parents in different bins) appears in ≥ 2
finalized accumulators, so the merge is genuinely a **reduction**, not a
concatenation — a bare scatter-by-offset does not suffice on its own.

Two candidate structures, in ladder order:

1. **Keep `out` an `unordered_map`, thread only the per-bin `finalize()`
   fan-in.** The bins are *already* deduplicated internally; only
   cross-bin keys collide. If cross-bin collisions are rare on HF (needs
   measuring — a probe counting `out.add` calls that hit an existing key
   vs. a fresh one), then: parallel-insert the 64 bins into 64 disjoint
   `unordered_map` shards keyed by `hash(child) % 64` (child, not parent —
   a child lands in one output shard regardless of which bin spawned it),
   each shard built by one thread, then a serial `0..63` shard walk into
   `out`. The serial tail is then only the shard-count concatenation
   (no per-key `+=`, the shards already reduced), and the fixed `0..63`
   shard order keeps it deterministic. **This is the least new code** — it
   reuses the binning discipline already in the file, one more `%`.

2. **Flat sorted output.** `inplace_merge` the 64 already-sorted
   accumulator entry vectors pairwise in a fixed tree (each is sorted by
   `(alpha,beta,weight-bits)` post-`finalize()`), fold equal-key runs
   left-to-right — same canonical fold `SpawnAccumulator::finalize` uses.
   The pairwise merges parallelize (disjoint pairs), the fold is one
   serial O(n) scan. `out` becomes a sorted `vector<pair<DetKey,Weight>>`
   instead of a map — which **also** removes the hash cost from `compress`,
   `ordered_l1_norm`, `weight_at`, and next step's partition, but is a
   larger change (every `WalkerPopulation` consumer touched) and needs its
   own before/after on those call sites.

Ladder says: **measure cross-bin collision rate first** (one probe, same
`PLANCK_FCIQMC_*_PROBE` inert-then-reverted discipline). If collisions are
< a few %, candidate 1. If not, candidate 2, priced against the whole
`WalkerPopulation`-becomes-a-vector change, not just the merge.

#### H2.8.1 result (2026-09-06): collision rate measured — **0.73 %**, candidate 1 it is

Probe (`PLANCK_FCIQMC_MERGE_PROBE`, added, measured, reverted — `git diff`
clean), HF/6-31G short run (400 equil + 200 sampling), 4 threads: per
merge, `entries` (summed `bin.size()` over all 64 finalized accumulators)
vs `distinct` (`out.size()` after the merge). Steady across all 200
sampling steps:

```
entries ~26,000   distinct ~25,800   collisions ~190   =>  0.73 %
```

So **99.27 % of the merge is a pure concatenation** — 64 already-deduplicated,
already-sorted lists appended into `out`. Only ~190 entries/call actually
`+=` onto an existing key.

Candidate 1 is the build. Concretely:

- `SpawnWorkspace` gains `std::vector<SpawnAccumulator> out_shards` (64,
  keyed `hash(child) % 64`).
- **After** the per-bin spawn region, a second `#pragma omp parallel for`
  over the 64 output shards: shard `s` walks all 64 finalized `next_bins`
  and pulls the entries whose `hash(child) % 64 == s` into `out_shards[s]`,
  then `out_shards[s].finalize()`. Writes are disjoint by construction (a
  child maps to exactly one shard), no lock. The 0.73 % cross-bin
  collisions land in the same shard (same child hash), so `finalize()`'s
  canonical fold resolves them — deterministically, same `(alpha, beta,
  weight-bits)` order the per-bin accumulators already use.
- Serial tail: `for (s = 0..63) for (entry : out_shards[s]) out.add(...)`.
  Still `unordered_map::operator[]`, but now **one insert per distinct
  child, never a collision** (shards are pre-reduced), and the 64-way
  fixed shard order keeps it thread-count-independent.

Cost: the shard-walk is `64 × entries` comparisons (`hash % 64` per entry
per shard = 64× redundant scan) — ~1.7M `%` ops/call, but threaded 64-way
and pure arithmetic. Cheaper alternative if that scan measures badly:
one serial pass tagging each entry's shard + a counting-sort into shard
ranges, then parallel `finalize()` per range. Start with the simple
64×-scan; it is the least code and the `%` is ~1 ns.

Two `#pragma omp` regions now (spawn, then shard-merge) with a serial
partition-tag between if the simple scan is too slow — the outer step loop
is unchanged, one more workspace vector.

#### Verify

Same gate as every H2 step, non-negotiable:

- **`n2_fciqmc_s5_threads1`** re-pinned to whatever value the reassociation
  lands on (candidate 2 folds in a *different* order than candidate 1's
  shard walk, so both re-pin), + **`n2_fciqmc_s5_threads4` == threads1 at
  `atol = 0.0`**.
- **S5 non-vacuity re-checked on the new merge**: reversing the
  bin/shard/merge-tree order must fail the pin while T1==T4 holds — the
  H2.1 mutation, run against the new code path.
- bitwise thread-count invariance 1/2/4/8 on `n2_fciqmc_sto3g` (50k
  steps, deepest bins).
- `metric_within_sigma` vs exact FCI on `h2_fciqmc_sto3g`,
  `n2_fciqmc_sto3g`.
- the 4 non-QMC FCI gates sharing `build_all_mo_ci_setup` unchanged.

#### Fixture

**HF/6-31G** (`tests/inputs/exploratory/fciqmc/validation/hf_base.hfinp`,
`ndet` = 213k, unsaturated) for the phase-probe before/after. ~~N2 is
saturated and its merge is ~35 µs, too small to measure a change
against.~~ **H2.8.2 refuted this** — over a 50k-step run N2/STO-3G's merge
is 15,000+ entries, HF-scale, and the sharded path regresses it ~1.5×.
N2/STO-3G is the correctness gate because its *determinant space* is small
enough for real sampling, **not** because its merge is cheap.

#### Gate

**Do not build until a real FCIQMC workload exists**
(`FCIQMC_RESEARCH_SCOPE.md` Q1). The current tree runs FCIQMC for seconds;
2.42× vs a hypothetical 3.5× on HF buys nothing yet. This section is the
plan for the day a target (Cr2 CAS(12,18) or similar) makes one
trajectory take minutes-to-hours. Same gate as H3.

#### Expected outcome

Candidate 1, if collisions are rare: merge ~1100 µs → ~150–250 µs
(serial shard concat only), whole call ~2957 µs → ~2100 µs, whole-call
speedup 2.42× → ~3.1–3.3×/4t against the 3.54× region ceiling. Candidate
2 could go slightly further (removes hash from downstream consumers too)
but the honest estimate needs the downstream call-site measurements it
has not had.

#### H2.8.2 result (2026-09-06): BUILT, correct and deterministic — and SLOWER ON HF/6-31G TOO. Genuine dead-end, reverted.

Candidate 1 was implemented in full:

- `SpawnWorkspace` gained `std::vector<SpawnAccumulator> out_shards`,
  sized and `reset()`-ed alongside the other per-call state.
- The serial `out.add`-per-entry merge was replaced by: a serial
  `O(entries)` scatter (one pass over the 64 finalized bins, appending
  each entry to `out_shards[hash(child) % kBins]` — **not** the
  64×-redundant scan a first draft used; that alone regressed N2 ~2×), a
  `#pragma omp parallel for` calling `finalize()` on the 64 shards, then a
  serial fixed-order `0..63` concat into `out`.

**Correctness held on every gate:** S5 invariance 1/2/4/8 bit-identical,
self-reproducibility bit-identical, S5 non-vacuity (reverse the shard
concat → shift `-109.16…` → `-108.08…` with T1 still == T4, the H2.1
property, on the new path ✓), `n2_fciqmc_sto3g` thread-count invariant,
FCI agreement within 5σ, 4 non-QMC FCI gates unchanged.

**Performance: SLOWER on both fixtures.** Measured `build-full`, this
machine, 50k walkers, identical inputs:

| fixture | threads | baseline | candidate 1 | Δ |
|---|---|---|---|---|
| N2/STO-3G, 50k steps | 1 | 8.6 s | 12.9 s | **+50 %** |
| N2/STO-3G, 50k steps | 4 | 7.6 s | 7.75 s | +2 % |
| **HF/6-31G, 2k+3k steps** | **1** | **53.3 s** | **55.3 s** | **+3.8 %** |
| **HF/6-31G, 2k+3k steps** | **4** | **26.2 s** | **28.2 s** | **+7.8 %** |
| **HF/6-31G, 2k+3k steps** | **8** | **23.2 s** | **24.7 s** | **+6.4 %** |

The scope projected HF's merge `~1100 µs → ~200 µs` and whole-call
`2.42× → ~3.1×`. **Neither happened.** HF got 3–8 % *slower* at every
thread count.

**So H2.8 is a genuine dead-end**, not a "works on HF, gated on Q1" —
confirmed slower on the fixture it was designed for. Candidate 2 (flat
sorted `WalkerPopulation` via pairwise `inplace_merge`) is not worth
attempting: same `O(n log n)`-replacing-`O(n)` wall plus a whole-suite
`WalkerPopulation`-becomes-a-vector change. Reverted; `git diff` is
docs-only.

#### H2.8.3 — reprofiled with `sample` (leaf/self-time) + `xctrace CPU Counters`, and the H2.7 "~1100 µs merge" diagnosis was wrong

Profiled the **baseline** (post-H2.7, no shards) on HF/6-31G, 1 thread,
40 s of `sample` at 1 ms after equilibration. Self-time (top-of-stack),
34,081 samples:

| category | samples | % | where |
|---|---|---|---|
| **`draw_excitation`** | 13,916 | **40.8 %** | spawn RNG + class/index arithmetic — **inside the parallel region** |
| `unordered_map::operator[]` | 3,841 | **11.3 %** | hash + bucket-walk + node-link; split across `out.add` (serial merge), `compress`, `signed_population`, diag memo |
| `slater_condon_element` | 3,798 | 11.1 % | `H_ij` off-diagonal — in the region |
| spawn-loop body (`._omp_fn.0`) | 2,868 | 8.4 % | per-attempt weight math, initiator check — in the region |
| allocator + `memset` | 2,491 | 7.3 % | `SpawnAccumulator` vector growth, `next_pop`/`signed_population` |
| diagonal memo lookup | 2,291 | 6.7 % | the T4 `unordered_map::find` — in the region |
| **`SpawnAccumulator::finalize()` sort** (`__introsort_loop<KeyWeightLess>`) | 1,869 | **5.5 %** | the **per-bin** sort — **inside the parallel region** |
| **serial merge + partition** (`propagate_stochastic` outer) | 686 | **2.0 %** | the `out.add(det,w)` concat loop + the parent-partition pass |
| `compress()` | 619 | 1.8 % | post-step map prune |
| `ordered_l1_norm()` | 608 | 1.8 % | per-step L1 norm |
| `excite_one` / parity / popcount | 848 | 2.5 % | in the region |

**What this overturns:**

- **The serial merge is 2.0 % of self-time, not 37 %.** H2.7's "~1100 µs,
  37 % of the whole call" came from a phase probe timing *wall-clock
  bracketing the merge region*. That bracket was dominated by
  `SpawnAccumulator::finalize()` — the **per-bin sort, 5.5 %, which is
  already inside the `#pragma omp` region and already threads** — plus the
  allocator churn of the accumulator vectors. The actual serial `out.add`
  concat is 686 samples. **There was never 1100 µs of serial merge to
  cut.** Candidate 1 replaced a 2 % serial loop with a 2 % serial scatter
  + *another* copy of the 5.5 % sort (now 64 shard-sorts on top of the 64
  bin-sorts) + a second fork/join — exactly why it measured slower.
- **`unordered_map` cost is real (11.3 %) but spread**, and only ~2 % of it
  is the merge. Cutting it means fewer maps, not a different merge:
  `signed_population` (only consumed by the Verbose `<N_I>/<N_0>` dump) and
  the separate `compress` pass are the removable ones.
- **`rehash` is 0.2 %.** The map is warm across steps (reused capacity), so
  the earlier "cache-miss-bound rehash" guess in H2.8.2 was also wrong —
  the `operator[]` cost is steady-state hash+probe, not growth.

**The `xctrace "CPU Counters"` run** (CPU-Bottlenecks mode, not raw
cache-miss events — the CLI template does not expose per-symbol L1D/L2
miss attribution) showed the hot path is **not** front-end / I-cache
bound: the `draw_excitation` + `slater_condon` inner loop is
compute-and-small-working-set bound (the `h_eff` matrix and `ga` vector
for HF/6-31G are ~1 KB + ~2 KB, L1-resident), consistent with 40.8 % in a
branchy integer-arithmetic function that does no large striding.

**Where the real time is: `draw_excitation` at 40.8 %, in the parallel
region.** The whole-call speedup ceiling (2.42×/4t) is set by everything
*outside* that region — and the profile says that "outside" is **not the
merge**; it is the per-step driver work that runs once per iteration on
one thread: `ordered_l1_norm` (1.8 %), `compress` (1.8 %), the
`signed_population` update, `projected_energy`, the `ShiftController`, and
the diagonal-prefill pass — each small, none the "1.1 ms" the phase probe
implied, summing to the ~10–15 % serial tail that caps Amdahl at ~2.5×.

**Retired hypotheses, for the next person:**

1. *"The merge is ~1100 µs / 37 % of the call and is the bottleneck."*
   Measured false — the serial merge is **2.0 %** of self-time. The phase
   probe bracketed the already-parallel per-bin `finalize()` sort (5.5 %)
   and accumulator allocation, not serial work.
2. *"The merge cost is cache-miss / rehash bound."* Rehash is 0.2 %; the
   map is warm. `operator[]` self-time is steady-state hash+probe, spread
   across four call sites, only ~2 % of it in the merge.
3. *"N2 is saturated so its merge is negligible; it is only the
   correctness fixture."* N2's merge is HF-scale by entry count (15k+),
   but — per (1) — the merge is not where the time is on *either* fixture.
   What makes N2 the correctness fixture is its determinant space being
   small enough for real sampling.

**If a real Q1 workload ever makes FCIQMC wall-time matter**, the levers
in profile order are: (a) `draw_excitation` — 40.8 %, already threaded, so
this is a *serial-efficiency* target (fewer branches, precompute the
per-parent class sizes once instead of per-attempt); (b) collapse the
three separate per-step `unordered_map` passes (`compress`,
`signed_population`, the diagonal prefill) into one walk of `pop`;
(c) gate `signed_population` off entirely unless `verbosity verbose`
(it already is — confirm the guard covers the accumulation, not just the
print). The merge is not on this list.

#### H2.8.4 — 4-thread reprofile + `draw_excitation` serial-efficiency scope (2026-09-06)

**4-thread `sample`** (HF/6-31G, `hf_prof` input, 30 s, ~100k samples
across 4 cores):

| category | 4t raw % | 4t % of *useful* (idle removed) | 1t % |
|---|---|---|---|
| **omp barrier / idle** | **46.4 %** | — | — |
| `draw_excitation` | 21.8 % | **40.7 %** | 40.8 % |
| `unordered_map::operator[]` | 6.6 % | 12.3 % | 11.3 % |
| `slater_condon` | 5.9 % | 11.0 % | 11.1 % |
| spawn loop body | 4.5 % | 8.5 % | 8.4 % |
| diagonal memo | 3.7 % | 6.9 % | 6.7 % |
| `finalize()` sort | 3.1 % | 5.8 % | 5.5 % |
| serial merge/partition | 1.0 % | 1.9 % | 2.0 % |

**`draw_excitation` does NOT shrink as a fraction of useful work at 4
threads** — it is 40.7 % either way, because it is *inside* the parallel
region and threads cleanly. What appears at 4 threads is the **46 % barrier
idle**: three worker threads waiting while one runs the serial per-step
driver tail (partition, `ordered_l1_norm`, `compress`, `signed_population`,
`ShiftController`, `projected_energy`, diagonal prefill). That idle *is*
the Amdahl tail the 1-thread profile's "~10–15 % serial" predicted, now
paid across 4 cores.

**So there are two independent targets, and the profile says which binds
at which thread count:**

- **1 thread:** `draw_excitation` (40.8 %) is the wall. Serial-efficiency
  work pays directly.
- **4+ threads:** the barrier idle (46 %) is the wall. The per-step serial
  driver tail is what to cut (lever (b)/(c) above); `draw_excitation`
  efficiency then only helps the ~22 % of machine-time still doing spawn
  work.

Both are worth doing for a real Q1 workload. This section scopes (a) —
`draw_excitation` serial efficiency — because it is self-contained, has a
unit gate already (`planck-fciqmc-walkers`), and is the 1-thread wall.

##### Why `draw_excitation` costs 40.8 %: the disassembly

`objdump` of `draw_excitation` (`build-full`, arm64) shows the function
opens with **`sub sp, sp, #0x2a0`** (672-byte frame) and then, *before any
random draw*:

1. **Four `OrbitalList` constructions** — `occupied(alpha)`,
   `occupied(beta)`, `virtuals(alpha)`, `virtuals(beta)`. Each
   `OrbitalList` is `std::array<int,32>` + a count, value-initialized
   (`items{}`), so each construction emits **4× `stp q31,q31`** = 128
   bytes zeroed — **512 bytes of stack zeroing per call**.
2. **Four bit-scan loops** of `n_act` iterations (11 for HF/6-31G, up to
   31): `lsl` a 1 by `p`, `tst` against the determinant word, conditional
   `str` of `p` and a count bump. **~44 iterations of branchy scalar code
   per call.**

Only *after* all that does the function compute the five class sizes, pick
a class (`rng.uniform_int(n_live)`), pick an index (`rng.uniform_int(class_size)`),
and do **one or two** `occ_x[k/vx]` / `unrank_pair` lookups for the chosen
class. The `k/va`, `k%va` divides and `unrank_pair`'s `while` loop are
real but downstream and single-class — small next to the four unconditional
full builds.

**The waste, precisely:** the picker needs only the four *counts*
(`na,nb,va,vb`) to choose a class and its `p_gen`, then needs to map one
integer `k` to specific orbital indices *within one spin channel*. It
builds all four full index arrays, every call, every spawn attempt, to use
at most two entries of one of them.

##### The plan (H2.8.4-a), ladder order

**Step 1 — stop building what isn't used. Counts, not lists.**
`na/nb/va/vb` are just `popcount`:

```cpp
const int na = std::popcount(parent.alpha & act_mask);   // act_mask = low_bit_mask(n_act)
const int va = n_act - na;
// ...beta likewise
```

That replaces the two `occupied` builds with two `popcount` instructions
and drops the two `virtuals` builds entirely (they were only ever used for
their `.size()` plus indexed access). `act_mask` is loop-invariant across
the whole run — compute once in `propagate_stochastic` (or pass `n_act`
and mask inline; a `popcount` of an already-masked word is one instr).
Expected: removes the 512-byte zeroing and 2 of the 4 scan loops
outright.

**Step 2 — index into the determinant directly, no materialized list.**
The chosen class needs "the `i`-th occupied alpha orbital" and "the `a`-th
virtual alpha orbital". That is a *select* on the bitmask, not an array
lookup:

```cpp
// i-th set bit of x (0-indexed). arm64 has no PDEP; this is the portable form.
// (as landed: `if (i == 0) return b; --i;` inside the hit branch, and the
// loop bound is `kCIStringBits`, not a literal 64.)
inline int nth_set_bit(CIString x, int i) {
    for (int b = 0; b < 64; ++b) {
        if (x & (CIString(1) << b)) { if (i == 0) return b; --i; }
    }
    return -1;   // caller guarantees i < popcount(x)
}
```

Called **at most twice** per spawn (once per orbital of a double), on the
one spin channel the class uses — versus the current **four** full scans.
For a single excitation it is one `nth_set_bit(occ)` + one
`nth_set_bit(vir)` where `vir = ~parent.alpha & act_mask`. The loop is
bounded by `n_act ≤ 31` and hits its target at the k-th set bit, so it is
~half the work of a full scan on average, done twice not four times.

(If `x86_64` matters later, `nth_set_bit` has a 2-instruction `PDEP`+`TZCNT`
form — but arm64 is the dev target and the portable loop is fine at
`n_act ≤ 31`.)

**Step 3 — hoist the class-size computation out of the attempt loop.**
`propagate_stochastic`'s inner loop calls `draw_excitation(det, ...)`
`n_spawn_attempts` times for the *same* `det`. `na/nb/va/vb` and the five
class sizes and `n_live` and the per-class `p_gen` prefactor are identical
across those attempts. With `n_spawn_attempts = 1` (both fixtures) this is
a no-op; but the interface should let the caller pass a small precomputed
`ParentClassInfo` so a run with `n_spawn_attempts > 1` computes it once.
Shape:

```cpp
struct ParentClassInfo {           // ~6 ints, fits a register pair or two
    int na, nb, va, vb;
    std::array<std::pair<Klass,int>, 5> live;
    int n_live;
};
ParentClassInfo classify(const DetKey&, int n_act);          // the popcounts + class sizes
Excitation draw_excitation(const DetKey&, int n_act, const ParentClassInfo&, RandomSource&);
```

Keep the current 3-arg `draw_excitation` as a one-line wrapper
(`classify` then the 4-arg form) so the ~20 test call sites and
`draw_excitation_in_space` are untouched.

##### H2.8.4-a result (2026-09-06): BUILT (steps 1+2), bit-identical, ~20 % faster on HF/6-31G at 1 thread

Steps 1 and 2 landed together in `draw_excitation` (`fciqmc.cpp`); step 3
(the `n_spawn_attempts > 1` hoist) was **skipped** — YAGNI, both fixtures
run `n_spawn_attempts = 1`, and the interface churn buys nothing until a
workload needs it.

- The four `OrbitalList` builds became four masks + four `std::popcount`:
  `occ_a = parent.alpha & act_mask`, `vir_a = ~parent.alpha & act_mask`,
  `act_mask = low_bit_mask(n_act)`; `na = popcount(occ_a)`, etc.
- Every `occ_x[idx]` / `vir_x[idx]` in the class switch became
  `nth_set_bit(mask, idx)` — a bounded (`b < kCIStringBits`) scan that
  exits at the idx-th set bit, called at most twice per spawn on the one
  spin channel the drawn class uses.
- `OrbitalList` / `occupied` / `virtuals` are kept — `enumerate_connections`
  (the oracle, and the projected-energy sum) still uses them; both are
  cold.

**Bit-identical, as required for a pure-arithmetic refactor:**

| check | result |
|---|---|
| `planck-fciqmc-walkers` (F2 `p_gen` oracle: frequency **and** support, open-shell cases) | all pass |
| `planck-fciqmc-accumulator` | all pass |
| S5 `n2_fciqmc_s5_short` 1/2/4/8 | bit-identical to baseline pin `-108.5036712075` / `-107.5859595426` |
| `h2_fciqmc_sto3g` 1t vs 4t | bit-identical `-1.1375594170` / `-1.1373925711` |
| `n2_fciqmc_sto3g` | bit-identical `-107.6433197414` / `-107.6448944489`, FCI 0.27σ |
| 4 non-QMC FCI gates (`enumerate_connections` untouched) | unchanged, equal to committed refs |

**Mutation-verified non-vacuous:** perturbing `nth_set_bit` (`i == 0` →
`i == 1`) produces **19 failures** in `planck-fciqmc-walkers` (F3.3, F4.1,
F4.2, F4.4, F4.5 — the `p_gen` corruption cascades through spawning). The
gate is not silently passing.

**Measured (`build-full`, this machine, 1 thread):**

| fixture | baseline | H2.8.4-a | Δ |
|---|---|---|---|
| HF/6-31G `hf_prof`, `verbosity verbose` (the original fixture) | 81.6 s (80.68 / 81.77 / 82.43) | **65.4 s** | **−20 %** |
| HF/6-31G `hf_prof`, **`verbosity normal` (production)** | **78.1 s** | **60.5 s** | **−22 %** |
| N2/STO-3G gate (`verbosity normal`) | 12.86 s | **11.56 s** | **−10 %** |

The `verbose` fixture pays a 6th full `pop` walk per step
(`signed_population[det] += w`, the `<N_I>/<N_0>` dump accumulator); a
production run at `verbosity normal` skips it. The N2 gate already runs
`normal`, so its number was always clean. The honest production HF figure
is **−22 %**.

**`draw_excitation` self-time (HF/6-31G, 1 thread, `sample`): 40.8 % →
27.1 %** — a 13.7 pp drop. `nth_set_bit` inlined into `draw_excitation`
(no separate frame), so its cost is inside that 27.1 %. Not the ~15–20 %
target — the residual is real arithmetic that stays: the five class-size
products (`na*(na-1)/2 * va*(va-1)/2` for the doubles), two `uniform_int`
calls, `unrank_pair`'s `while` loop, and the `k/va` / `k%va` divides.
Everything else in the profile scaled up proportionally (same absolute
time, smaller pie).

**4-thread:** N2 7.75 s → 7.90 s, HF unchanged within noise — as
predicted, the 4-thread wall is the ~54 % barrier idle (the serial
per-step driver tail), not `draw_excitation`. That is lever (b), scoped
as **H2.8.4-b** below.

##### Gate (unchanged)

This is landed on the branch but **stays gated on a real Q1 workload**
before it goes to `devel` alongside the rest of the FCIQMC parallelism
work — nothing in the tree runs FCIQMC at 1 thread long enough for a
20 % improvement to matter yet. It is low-risk (bit-identical, existing
`p_gen` gate, mutation-verified) so it is ready when a target appears.

#### H2.8.4-b — the serial per-step driver tail (scoped 2026-09-06, not started)

**The 4-thread wall.** At 4 threads on HF/6-31G, `sample` shows **54 %
idle** — three worker threads parked at the barrier while one thread runs
the serial per-step driver work. `draw_excitation` efficiency (H2.8.4-a)
does not touch this; it *widens* the relative gap, because a smaller
parallel region against an unchanged serial tail is a worse Amdahl ratio.

##### What runs serially, per step

The outer `for (step ...)` loop in `fciqmc_driver.cpp` does, in order,
between the `#pragma omp` regions of `propagate_stochastic`:

| # | pass | code | walks | cost shape |
|---|---|---|---|---|
| 1 | **diagonal prefill** | `for (det,w) : pop) ops.diagonal(det)` (`:288`) | full `pop` + a `diag_cache` hash lookup per det | ~26k hash probes on `pop` + ~26k on `diag_cache` |
| 2 | **partition** | inside `propagate_stochastic` head — `for (det,w) : population) bin_parents[hash % kBins].push_back(...)` | full `pop` | ~26k hash iterations + `hash % 64` + `push_back` |
| 3 | **merge** | `propagate_stochastic` tail — `for (bin) for (e : bin) out.add(e)` | 64 accumulators → `out` (`unordered_map`) | ~26k `unordered_map::operator[] +=`, memory-latency-bound (this is the "~1100 µs" H2.7 mismeasured as the *whole* serial cost) |
| 4 | **`compress(1e-12)`** | `pop.compress` (`fciqmc.cpp:16`) | full `_walkers`, conditional `erase` | ~26k iterations + erases |
| 5 | **`ordered_l1_norm(pop)`** | (`fciqmc.cpp` `ordered_l1_norm`) | full `pop`, bin `|w|` into 64 | ~26k hash iterations + `hash % 64` |
| 6 | **`signed_population[det] += w`** | driver `:411`, **`verbosity verbose` only, already guarded** | full `pop` | ~26k `unordered_map::operator[] +=` — absent at `verbosity normal` (the production path and the N2 gate); ~3.5–5 s/run of the HF `verbose` fixture, see the corrected table above |
| 7 | `projected_energy` | `enumerate_connections(reference)` | ~600 connections, **not** `pop` | negligible (0.1 %) |
| 8 | `ctl.update`, bounds checks, sample `push_back` | scalar | — | negligible |

So the serial tail is **four to five full traversals of the ~26k-entry
walker hash map per step** at `verbosity normal` (five to six at
`verbose`), three of them (1, 3, 4) paying `unordered_map` node-chasing /
`operator[]` latency, two (2, 5) at least iterating it. The 4-thread
profile's 14 % of *useful* samples in `unordered_map::operator[]` + 8 % in
the diagonal memo is the visible tip; the rest hides in memory-latency
stalls that `sample` attributes elsewhere.

##### The plan, ladder order

**Step 1 — fold passes 1, 2, 4, 5 into ONE walk of `pop`.** They are all
`for (det,w) : pop) { ... }` with independent per-entry work:

- pass 1 wants `ops.diagonal(det)` resident → call it
- pass 2 wants `det` in `bin_parents[hash(det) % kBins]` → push it
- pass 4 wants small-`|w|` entries dropped → can't erase-while-iterating
  the map being read, but *can* record which keys to drop (or build the
  next `pop` filtered)
- pass 5 wants `Σ|w|` binned → add `|w|` to `l1_bins[hash % kBins]`

One pass computes the `hash(det)` **once** (currently 3–4× per det across
the passes), does all four per-entry updates, and hands
`propagate_stochastic` a *pre-partitioned* `bin_parents` plus the L1 norm
plus the compressed set — so passes 2, 4, 5 disappear from
`propagate_stochastic` and the driver entirely. The interface shift:
`propagate_stochastic` takes `bin_parents` as an input (already filled)
instead of building it, and the driver's `compress` + `ordered_l1_norm`
calls move into the fused pre-pass.

This is a **reassociation of the L1-norm sum** (same 64-bin fixed
partition, but the bin a det lands in is now computed in the pre-pass, not
in `ordered_l1_norm` — identical function of `det`, so identical result)
and a **reassociation of nothing else** (prefill and partition are pure
mechanical moves; compress just changes *when* the filter is applied).
Expected `atol = 0.0` bitwise on S5 for prefill+partition; `ordered_l1_norm`
must be checked — its existing "independent of insertion order" unit test
already spans 18 orders of magnitude, and the bin function is unchanged,
so it should hold, but verify.

**Step 2 — DONE (2026-09-06): the `verbosity verbose` measurement
artifact, confirmed and corrected.** Pass 6 (`signed_population[det] += w`)
is already `want_coefficient_ratios`-guarded on both the accumulation
(`:411`) and the consumption (`:573`) — no code change needed. But
`hf_base.hfinp` sets `verbosity verbose`, so H2.8.4-a's 65.4 s **included**
pass 6. Re-measured at `verbosity normal`: baseline **78.1 s**, H2.8.4-a
**60.5 s** — the production improvement is **−22 %**, and the table above
now carries both rows. Pass 6 costs ~3.5 s in the baseline / ~5 s in
H2.8.4-a (relatively larger there, since the rest shrank). Nothing to
build; the fix was to stop quoting a `verbose`-inflated number.

**Step 3 — thread the fused pre-pass, if step 1 is not enough.** After
step 1 the serial tail is one `pop` walk + the merge. That walk is a
scatter into `bin_parents` (parent's bin) + `l1_bins` (child... no,
parent's `|w|`) — both fixed 64-bin partitions, both the exact shape the
spawn region already threads deterministically. So it can take the *same*
`#pragma omp` treatment: partition `pop`'s buckets across threads, each
thread fills its slice of `bin_parents` / `l1_bins`, fixed-order combine.
The merge (pass 3) is the residual and is its own problem (H2.8.1–3
showed sharding it does not pay — leave it serial, or revisit only with
a real large-`ndet` workload where 26k → 260k changes the calculus).

##### Verify

- **S5 `n2_fciqmc_s5_short` 1/2/4/8 at `atol = 0.0`** — steps 1 and 3 are
  mechanical moves + one sum reassociation (`ordered_l1_norm`'s bin fill).
  Prefill/partition must stay bitwise; `ordered_l1_norm` re-pin only if
  the bin-fill order genuinely changed (it should not — same
  `hash % kBins`).
- **S5 non-vacuity** — reversing the fused pre-pass's bin order (or the
  `l1_bins` combine order) must break the pin while T1==T4, same as the
  H2.1 property.
- **`planck-fciqmc-walkers`** + **`ordered_l1_norm` insertion-order unit
  test** unchanged.
- **FCI agreement** `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g` within 5σ.
- **Measure**: 4-thread HF/6-31G `hf_prof` wall before/after (the target
  metric — 1-thread will move less), and re-run `sample` at 4 threads —
  target is the 54 % idle dropping toward ~25–30 % (Amdahl with the
  serial tail cut from ~5 walks to ~1 walk + merge).

##### Gate

Same as everything in H2.8: **do not build until a real Q1 workload
exists.** The 4-thread wall only bites when one trajectory takes
minutes-to-hours, which nothing in the tree does. Step 2 (confirm the
`verbosity verbose` measurement artifact and re-quote H2.8.4-a's number)
is the one piece worth doing now — it is a measurement correction, not a
code change.

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
- **H2 — DONE (H2.1–H2.7). Landed: partial win — whole-call speedup
  2.24→2.42×/4t, 2.47→2.87×/8t on HF. The RNG hoist landed its full
  ~38 µs/call; the ~1100 µs/call merge is the sole remaining serial cost
  and is a separate, gated investigation.**
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
  **H2.6 (DONE, no code change)** — the `#pragma omp` survived H2.4/H2.5
  intact, so "re-enable" was a no-op; thorough re-verification at 1/2/4/8
  on the S5 pair, `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g` (50k steps), and the
  4 non-QMC FCI gates all bitwise-identical, and `schedule(dynamic)` stays
  invariant too (the accumulator partition does not depend on the
  schedule, unlike the FCI sigma build). **H2.7 (DONE)** re-ran the H1
  probe on HF/6-31G with a per-phase split: rekey 38 µs → **0.14 µs**
  (H2.3's arithmetic confirmed live), 64-bin construction gone, whole-call
  2.24→2.42×/4t and 2.47→2.87×/8t. But ~1100 µs/call of **merge** remains
  — untouched by H2.4–H2.6, not threaded (must stay fixed bin order for
  invariance). H1's "~1.5 ms serial drag" was construction + RNG +
  partition + merge; the rewrite removed the first two. The merge is a
  fixed-order parallel-scatter problem (prefix-sum bin sizes, threaded
  scatter by offset), a separate careful target gated on a real workload —
  not more of H2.
- **H2.0 (smaller fixed `kBins`) — reserve, N2-class only.** Helps a
  saturation-starved fixture; on HF the bins are already large enough.
  H2 does **not** touch `kBins`.
- **H2.8 (the fixed-order merge) — candidate 1 BUILT, verified correct,
  REVERTED, then REPROFILED (2026-09-06). The merge was never the
  bottleneck — H2.7's phase-probe diagnosis was wrong.** Candidate 1 (64
  `hash(child) % 64` output shards, serial scatter, parallel `finalize()`,
  serial concat) passed every correctness gate but measured 3–8 % *slower*
  on HF/6-31G at every thread count. A proper `sample` reprofile of the
  baseline (self-time, 34k samples, HF/6-31G 1t) shows why: **the serial
  merge is 2.0 %**, not H2.7's "37 %". The phase probe had bracketed the
  per-bin `SpawnAccumulator::finalize()` sort (5.5 %) — *already inside the
  `#pragma omp` region, already threaded* — plus accumulator allocation.
  Real hot path: **`draw_excitation` 40.8 %** (in the region), `H_ij`
  11.1 %, `unordered_map::operator[]` 11.3 % (spread across 4 call sites,
  ~2 % of it the merge), allocator 7.3 %. `rehash` is 0.2 % — the map is
  warm, so the H2.8.2 "cache-miss/rehash bound" guess was also wrong. The
  ~2.5× Amdahl ceiling is a ~10–15 % serial tail of *small* per-step driver
  passes (`ordered_l1_norm`, `compress`, `signed_population`, diagonal
  prefill), not the merge. Full table + three retired hypotheses + the real
  lever list in H2.8.3. Reverted; `git diff` docs-only.
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
