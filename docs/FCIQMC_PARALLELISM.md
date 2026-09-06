# FCIQMC Parallelism

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Can FCIQMC be made to scale past the 1.57x the incremental spawn-loop
threading gave (`docs/FCIQMC_T2_THREADING.md`), and what does the profile
say the real ceiling is?**

Parent context: `docs/FCIQMC_RESEARCH_SCOPE.md`. The incremental threading
that preceded this: `docs/FCIQMC_T2_THREADING.md`. The serial fixes before
that: `docs/FCIQMC_SERIAL_PERFORMANCE.md`.

## Short answer

The 1.57x was a property of *this* FCIQMC's shape, not of the method. Three
things were tried:

1. **A `SpawnWorkspace` / `SpawnAccumulator` / xoshiro256** rewrite** of the
   per-call scaffolding (H2). Landed, bitwise thread-count invariant. Moved
   the whole-call speedup from **2.24x -> 2.42x at 4 threads, 2.47x -> 2.87x
   at 8** on HF/6-31G. The RNG hoist landed its full ~38 us/call.

2. **Sharding the fixed-order merge** (H2.8). Built, verified correct,
   **measured 3-8 % *slower* on HF/6-31G at every thread count**, reverted.
   A proper `sample` reprofile then showed the merge is **2.0 % of
   self-time**, not the 37 % an H2.7 phase probe had implied — the probe
   had bracketed an *already-threaded* per-bin sort.

3. **The real 1-thread hot path, `draw_excitation`** (H2.8.4-a). Rewritten
   off four `std::array<int,32>` builds per call to four `std::popcount`s +
   a bounded `nth_set_bit` select. Bit-identical. `draw_excitation`
   self-time **40.8 % -> 27.1 %**; HF/6-31G 1-thread wall **-22 %** at
   production verbosity.

4. **Folding the serial per-step driver tail** (H2.8.4-b). Merged the
   diagonal-prefill + partition passes, and `compress` + `ordered_l1_norm`,
   into two `pop` walks instead of four. Bit-identical. ~5-7 % wall. Modest
   because the **merge is untouched** and it does not shard for a win.

Net measured on HF/6-31G (`verbosity normal`, `build-full`, this machine):

| | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| pre-H2 (T2 threading only) | — | 2.24x | 2.47x |
| after H2 rewrite | — | 2.42x | 2.87x |
| after H2.8.4-a (`draw_excitation`) | -22 % wall | — | — |
| after H2.8.4-b (fold) | -5 % more | -6 % more | — |

**The residual ceiling is the fixed-order merge** — `unordered_map::
operator[]` latency on ~26k entries per step, serial, and H2.8 established
it cannot be cheaply parallelized. And **H3** (replica parallelism, the
one axis that would give genuine near-linear scaling) is a design sketch,
not built.

**All of it is gated on a real FCIQMC workload appearing
(`FCIQMC_RESEARCH_SCOPE.md` Q1).** Nothing in the tree runs FCIQMC at a
size or duration where any of this matters, so the landed pieces sit on
the `fciqmc-parallel-rewrite-scope` branch, bit-identical and low-risk,
ready when a target exists.

## Where the logic lives

- `src/post_hf/ci/fciqmc.cpp` — `propagate_stochastic` (the spawn loop, its
  64-bin partition, the `#pragma omp parallel for` region, the fixed-order
  merge); `draw_excitation` and `nth_set_bit`; `WalkerPopulation::compress`
  / `compress_with_l1_norm`; `ordered_l1_norm`
- `src/post_hf/ci/fciqmc.h` — `WalkerPopulation` (the `unordered_map` the
  merge writes into); `RandomSource` (xoshiro256**, and its fixed-seed
  contract); `kFciqmcBins`
- `src/post_hf/ci/spawn_accumulator.h` — `SpawnAccumulator` (the
  reuse-stable per-bin accumulator), `SpawnWorkspace` (the hoisted
  per-call scaffolding, `parents_prefilled`)
- `src/post_hf/fciqmc_driver.cpp` — the serial outer `for (step ...)` loop;
  `make_ops` and the T4 diagonal memo
- `tests/regression_cases.json` — `h2_fciqmc_threads1/4` and
  `n2_fciqmc_s5_threads1/4` (the S5 invariance gates); `h2_fciqmc_sto3g`,
  `n2_fciqmc_sto3g` (FCI-agreement gates)
- `tests/fciqmc_walkers.cpp`, `tests/fciqmc_accumulator.cpp` — the unit
  suites (`planck-fciqmc-walkers`, `planck-fciqmc-accumulator`)
- `docs/FCIQMC_RESEARCH_SCOPE.md` §6 — the decided determinism policy
- `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` — the structurally identical
  scatter that threaded to 3.54x, and its determinism traps

## Invariants a rewrite must not break

- **Bitwise thread-count invariance at `atol = 0.0`** across
  `OMP_NUM_THREADS` = 1/2/4/8. A decided policy (`FCIQMC_RESEARCH_SCOPE.md`
  §6), not a discovered constraint. Kept via **partition-by-parent**: each
  parent hashes to one of `kFciqmcBins = 64` fixed bins
  (`DetKeyHash{}(det) % kFciqmcBins`, never tied to thread count), each bin
  accumulates independently, and bins merge back in fixed 0..63 order —
  never by completion order (the DFT-grid-jitter defect).
- **The gate must be non-vacuous.** A merge-order or partition
  reassociation is *thread-count-invariant*, so a `threads1 == threads4`
  comparison alone cannot see it. `n2_fciqmc_s5_threads1` therefore
  **pins both energies** to fixed values; `threads4` compares against
  `threads1`. Verified: reversing the bin-merge order, or binning by
  `det.alpha` instead of the full hash, shifts the pin while T1 == T4.
- **Fixed-seed reproducibility** — `RandomSource`'s contract. An RNG change
  (H2.5 swapped mt19937_64 -> xoshiro256**) alters every trajectory; that
  is a reordering-class change, gated by self-reproducibility +
  `metric_within_sigma` against exact FCI, never by matching old numbers.
- **The shared integral setup.** `build_all_mo_ci_setup` is a *move* out of
  `run_fci`, not a copy. A parallelism change touches `propagate_stochastic`
  and the driver loop, not the Hamiltonian — the four non-QMC FCI gates
  that share that setup must stay green.
- **`slater_condon_element` is public**, also consumed by CASSCF/RASSCF
  response code — do not change its signature as a side effect.

## H1 — is the per-step work too small to parallelize?

No, on any non-saturated fixture — and that is the interesting case.

Measured (`PLANCK_FCIQMC_H1_PROBE`, added, measured, reverted) on two real
fixtures: the N2/STO-3G gate and HF/6-31G
(`tests/inputs/exploratory/fciqmc/validation/hf_base.hfinp`, `ndet` =
213,444).

**Per-parent cost is a fixed property of the system, flat in walker
count:** N2 `ns/parent` is **146-152** across a 2k->160k walker sweep; HF
is **~202 at 1 thread** (~58 at 4). It is one `draw_excitation` + one
`slater_condon_element` + one memoized diagonal + two `unordered_map`
inserts.

**But `parents/call` — and so whether the region threads — depends on
whether the determinant space is saturated:**

| fixture | ndet | walkers | parents/call | parents/bin |
|---|---|---|---|---|
| N2/STO-3G | 14,400 | 10,000 (gate) | 908 | ~14 |
| N2/STO-3G | 14,400 | 160,000 | 1,372 | ~21 |
| HF/6-31G | 213,444 | 10,000 | 15,867 | ~248 |
| HF/6-31G | 213,444 | 50,000 (base) | 27,505 | ~430 |

N2's `parents/call` grows only 2.2x for 80x walkers because its space
saturates — every determinant is occupied and adding walkers just deepens
weights. HF stays unsaturated.

**The region-threading result splits cleanly on that:**

*N2/STO-3G* (~14 parents/bin) — **granularity-ceilinged**:

| threads | region speedup | whole-call speedup |
|---|---|---|
| 4 | 2.28x | 1.53x |
| 8 | 1.89x (regresses) | 1.40x (regresses) |

*HF/6-31G* (~430 parents/bin) — **scales like the FCI sigma build**:

| threads | region speedup | whole-call speedup |
|---|---|---|
| 4 | 3.44x | 2.24x |
| 8 | 4.47x (still climbing) | 2.47x |

N2's ~14 parents/bin is ~2.1 us of arithmetic per bin — fork/join plus
each `next_bins[bin]` touch dominates, and 8 threads cannot cover the
thread-spawn cost. HF's ~87 us/bin is comfortably above the fork/join
floor. **The N2 gate is a *saturated* fixture, chosen deliberately so
sampling is a real sample of a small space — and that same choice makes it
a bad throughput fixture. Use HF/6-31G (or larger) for all parallel-FCIQMC
measurement.**

## H2 — the `SpawnWorkspace` rewrite

The per-parent work threads fine on an unsaturated fixture (H1). The whole
call was stuck at 2.24x/4t because ~1.5 ms/call of serial setup — 64 bin
constructions, 64 `mt19937_64` engine seedings, the parent partition, the
merge — was rebuilt every ~50,000 calls and did not thread. There is one
call site, driven by one thread across the whole outer loop, so all of
that state is now persistent, in a caller-owned `SpawnWorkspace`.

Three pieces:

- **`SpawnAccumulator`** (`spawn_accumulator.h`) — a `vector<pair<DetKey,
  Weight>>` whose `finalize()` sorts by a total order on `(alpha, beta,
  bit-pattern of weight)` then folds equal-key runs left-to-right, so the
  sum is a pure function of the multiset. `reset()` is `vector::clear()` —
  a genuine reset, unlike `unordered_map::clear()` which keeps the peak
  bucket count. **The weight-bit tiebreak is load-bearing:** `std::sort` is
  not stable, so a >=3-long same-key run would otherwise still fold in
  insertion order. Gated by `planck-fciqmc-accumulator` against an
  independent `std::map` reference; mutation-verified.

- **`SpawnWorkspace`** — the 64 accumulators, 64 parent buckets, and 64 RNG
  streams, built once by the driver, `reset_for_call()` clearing without
  freeing. `propagate_stochastic` gained a `void ...(ws, out)` form; the
  value-returning form is kept as a thin overload so the ~20 test call
  sites are untouched.

- **`RandomSource` engine swapped `mt19937_64` -> xoshiro256**.** H2.3
  measured the per-bin RNG cost three ways: fresh construct 42 us,
  mt19937 reuse + `seed()` 39 us (the ~2 us gap is only the vector alloc),
  counter-based 74 ns. **`mt19937_64::seed()` is ~600 ns each x 64 ~= 38
  us/call and reusing the engine object cannot avoid it** — the 312-word
  state fill is the whole cost. T2 called this "unavoidable" having only
  tried reuse. xoshiro256**'s "seed" is four SplitMix64 outputs into a
  256-bit state, O(1). Public surface unchanged, `uniform()` still 53-bit.
  The 64 streams live in `ws.bin_rngs`, re-keyed in place each call by the
  identical `derive()` recipe.

**Gated — not bitwise-vs-the-pre-H2 numbers.** The `unordered_map` ->
`SpawnAccumulator` swap reassociates a fixed multiset's sum, and the RNG
swap changes every trajectory. Both are legitimate; gated instead on
self-reproducibility (5 runs bit-identical) + thread-count invariance
(`atol = 0.0`, 1/2/4/8) + `metric_within_sigma` vs exact FCI. The S5 pin
was re-taken after each step.

**H2.7 result — partial win, and the merge misdiagnosed.** A per-phase
probe on HF/6-31G, 4 threads:

| phase | us/call | scales with threads? |
|---|---|---|
| rekey (RNG) | 0.14 (was ~38 pre-H2.5) | — |
| partition | ~170 | no |
| **merge** | **~1100** | **no** |
| region (`#pragma omp`) | 1659 | yes (3.54x/4t) |

Whole call 2.24 -> 2.42x/4t, 2.47 -> 2.87x/8t; `n2_fciqmc_sto3g` 11.2 s ->
8.6 s. The RNG hoist landed its full ~38 us/call. The `~1100 us merge`
figure was recorded as the sole remaining serial cost — **that was
wrong**, see H2.8.

## H2.8 — the fixed-order merge is not the bottleneck

**H2.8.1** measured the cross-bin collision rate on HF/6-31G: **0.73 %** —
the merge is 99.27 % concatenation of 64 already-deduplicated, already-
sorted lists. So candidate 1: 64 output shards keyed `hash(child) % 64`,
filled by a serial scatter, `finalize()`d in a threaded region, concat'd
in fixed order.

**H2.8.2 — built, correct, deterministic, and SLOWER.** Candidate 1 passed
every correctness gate — S5 invariance 1/2/4/8, S5 non-vacuity, FCI
agreement — but measured:

| fixture | threads | baseline | candidate 1 | Δ |
|---|---|---|---|---|
| N2/STO-3G, 50k steps | 1 | 8.6 s | 12.9 s | +50 % |
| HF/6-31G, 2k+3k steps | 1 | 53.3 s | 55.3 s | +3.8 % |
| HF/6-31G, 2k+3k steps | 4 | 26.2 s | 28.2 s | +7.8 % |
| HF/6-31G, 2k+3k steps | 8 | 23.2 s | 24.7 s | +6.4 % |

Slower at every thread count on the fixture it was designed for. Reverted.

**H2.8.3 — reprofiled with `sample`, and H2.7's diagnosis was wrong.**
Self-time (top-of-stack, 34,081 samples, baseline HF/6-31G, 1 thread):

| category | % | where |
|---|---|---|
| **`draw_excitation`** | **40.8 %** | spawn RNG + class/index arithmetic — in the region |
| `unordered_map::operator[]` | 11.3 % | 4 call sites; ~2 % of it the merge |
| `slater_condon_element` | 11.1 % | H_ij — in the region |
| spawn-loop body | 8.4 % | in the region |
| allocator + memset | 7.3 % | accumulator vectors |
| diagonal memo lookup | 6.7 % | in the region |
| **`SpawnAccumulator::finalize()` sort** | **5.5 %** | **already inside `#pragma omp`, already threads** |
| **serial merge + partition** | **2.0 %** | the `out.add` concat + parent partition |
| `compress()` | 1.8 % | |
| `ordered_l1_norm()` | 1.8 % | |

**The serial merge is 2.0 %, not 37 %.** H2.7's phase probe timed
wall-clock *bracketing the merge region*, and that bracket was dominated by
the per-bin `finalize()` sort — 5.5 %, already threaded — plus accumulator
allocation. There was never 1100 us of serial merge to cut. Candidate 1
replaced a 2 % serial loop with a 2 % serial scatter + *another* copy of
the 5.5 % sort (64 shard-sorts on top of the 64 bin-sorts) + a second
fork/join — exactly why it measured slower.

`rehash` is 0.2 % — the map is warm across steps — so the "cache-miss /
rehash bound" guess was also wrong; `operator[]` self-time is steady-state
hash+probe.

## H2.8.4-a — `draw_excitation`, the real 1-thread lever

The disassembly: `draw_excitation` opens with a 672-byte stack frame, then
**before any random draw** builds four `OrbitalList` (`std::array<int,32>`)
by value — **512 bytes of stack zeroing per call** — and runs four
`n_act`-bit scan loops, to then use only the four *counts* (to pick a
class) and one or two orbital-index lookups within the chosen spin channel.

Rewritten: four bitmasks over the active window + `std::popcount` for the
counts; a bounded `nth_set_bit(mask, i)` select — exits at the i-th set
bit, called at most twice per spawn on one channel — replaces every
`occ_x[idx]` / `vir_x[idx]`. `OrbitalList` / `occupied` / `virtuals` are
kept for the cold `enumerate_connections` oracle.

**Bit-identical** — a pure-arithmetic refactor: `uniform_int` is called in
the same order with the same arguments, and the orbital each `k` maps to is
the same orbital.

| check | result |
|---|---|
| S5 1/2/4/8 `atol = 0.0` | bit-identical to the pre-refactor pin |
| `planck-fciqmc-walkers` (F2 `p_gen` oracle: frequency **and** support, open-shell) | pass |
| 4 non-QMC FCI gates | unchanged |
| mutation: `nth_set_bit` `i == 0` -> `i == 1` | **19 failures** in `planck-fciqmc-walkers` |

**Measured (1 thread, `verbosity normal`):**

| fixture | before | after | Δ |
|---|---|---|---|
| HF/6-31G `hf_prof` | 78.1 s | 60.5 s | **-22 %** |
| N2/STO-3G gate | 12.86 s | 11.56 s | -10 % |
| `draw_excitation` self-time | 40.8 % | 27.1 % | -13.7 pp |

(The older `hf_base.hfinp` fixture is `verbosity verbose`, which pays a
6th full `pop` walk per step — `signed_population`, the `<N_I>/<N_0>` dump
accumulator, already guarded. On that fixture the figure is -20 %.)

The residual 27.1 % is real arithmetic that stays: the class-size products
(`na*(na-1)/2 * va*(va-1)/2` for the doubles), two `uniform_int` calls,
`unrank_pair`'s loop, the `k/va` / `k%va` divides.

## H2.8.4-b — folding the serial per-step driver tail

The 4-thread wall is ~54 % barrier idle: three workers park while one runs
the serial per-step driver work. Enumerated, that is **4-5 full traversals
of the ~26k-entry walker hash map per step** at `verbosity normal`:

| pass | what | latency-bound? |
|---|---|---|
| diagonal prefill | `for (det,w) : pop) ops.diagonal(det)` | yes |
| partition | `pop` -> 64 `bin_parents` buckets | iterate |
| **merge** | 64 accumulators -> `out` (`unordered_map::operator[] +=`) | **yes — this is the residual** |
| `compress(1e-12)` | `_walkers` walk + conditional `erase` | yes |
| `ordered_l1_norm` | `pop` walk, bin `|w|` into 64 | iterate |
| `signed_population` | `pop` walk — `verbosity verbose` only | (guarded) |

**Fold A** (prefill + partition): the driver's prefill loop now also fills
`SpawnWorkspace::parents` with the identical `DetKeyHash{}(det) %
kFciqmcBins` binning. A `parents_prefilled` flag makes `propagate_stochastic`
skip its own partition. `kBins` moved to the header as `kFciqmcBins` so the
two partitions provably match.

**Fold B** (compress + l1 norm): `WalkerPopulation::compress_with_l1_norm`
does both in one traversal. The norm is **byte-identical** to `compress()`
then `ordered_l1_norm()` — same survivors (erased iff `|w| <= threshold`,
and only survivors contribute either way), same fixed 64-bin partition (a
pure function of `det`, so erasing other entries mid-scan cannot move a
survivor's bin), same fixed 0..63 summation order. Gated by a new unit
assertion.

**Bit-identical**, S5 non-vacuity re-verified (binning fold A by `det.alpha`
shifts the pin while T1 == T4).

**Measured (`verbosity normal`):**

| fixture | H2.8.4-a | + folds | Δ |
|---|---|---|---|
| HF/6-31G `hf_prof` 1t | 64.3 s | 61.0 s | -5 % |
| HF/6-31G `hf_prof` 4t | 34.1 s | 32.1 s | -6 % |
| N2/STO-3G gate 4t | 7.75 s | 7.23 s | -7 % |

**Modest.** The fold removed ~1.5 of the ~5 serial walks, but the **merge
— the biggest one — is untouched**, and H2.8 proved it does not shard for
a win. Threading the fused pre-pass instead would chase the smaller half:
the 4-thread number barely moved from folding the non-merge passes, so
the pre-pass is not the dominant serial cost. That leaves the merge as
the residual ceiling, with no cheap parallelization, and H3 as the only
axis that would give genuine near-linear scaling.

## Retired hypotheses

So the next person does not retake a turn. (The T2 doc,
`docs/FCIQMC_T2_THREADING.md`, has two more — the bin-to-thread packing red
herring and the `next`-as-static reversion.)

1. **"The 1.57x is a property of FCIQMC's per-step granularity."** Refuted
   — it is a property of a *saturated* fixture. On HF/6-31G the region
   threads to 3.44x/4t, 4.47x/8t and keeps climbing.

2. **"The ~1100 us merge is `O(n log n)`-parallelizable serial work and is
   the bottleneck."** Measured false. The serial merge is **2.0 %** of
   self-time. The phase probe that produced "37 %" had bracketed the
   already-threaded per-bin `finalize()` sort (5.5 %) and accumulator
   allocation. Sharding the merge adds a second sort + a barrier and
   measured 3-8 % *slower* on HF.

3. **"The merge cost is cache-miss / rehash bound."** `rehash` is 0.2 %;
   the map is warm across steps. `operator[]` self-time is steady-state
   hash+probe, spread across four call sites, only ~2 % of it in the merge.

4. **"N2/STO-3G's merge is negligible because its space is saturated."**
   Its merge is HF-scale by entry count (15k+ over a 50k-step run). But —
   per (2) — the merge is not where the time is on *either* fixture. What
   makes N2 the correctness fixture is its *determinant space* being small
   enough for real sampling, not its merge being cheap.

5. **"Reusing the `mt19937_64` engine objects avoids the seeding cost."**
   Refuted — `seed()` is ~600 ns each (the 312-word state fill), and reuse
   still has to call it. The fix is a counter-based engine (xoshiro256**),
   O(1) reseed.

6. **"Threading the fused per-step pre-pass will close the 4-thread gap."**
   Declined on measurement — folding the non-merge passes moved the
   4-thread number ~6 %, so the pre-pass is not the dominant serial cost;
   the merge is, and it has no cheap parallelization.

## H3 — replica parallelism (not built)

The one axis that gives genuine near-linear scaling. FCIQMC literature runs
multiple independent replicas (different RNG seeds, same Hamiltonian) both
for error bars and as a parallelization axis — each replica is an
embarrassingly parallel full trajectory with zero shared mutable state.
Planck has the reproducibility machinery (`RandomSource`, fixed-seed
contract) and the statistical machinery (`blocked_standard_error`,
`metric_within_sigma`) to combine them.

**Why it is not built:** it does not speed up a *single* trajectory, only
tightens the error bar in fixed wall-time, and it multiplies memory by R
(R walker populations). For the committed gates — seconds of runtime,
already inside their error bars — it buys nothing. It matters only at a
real target size where one trajectory is minutes-to-hours and the error
bar is the deliverable. The recommendation is: **document it as the right
move when a target appears, do not build it now** — the same conclusion
`FCIQMC_RESEARCH_SCOPE.md` Q1 reaches for the method as a whole.

Sketch: an outer replica loop, `#pragma omp parallel` over replicas, each
with its own `RandomSource(seed + replica_index)`, `ShiftController`, and
`WalkerPopulation`; combine the per-replica shift and projected-energy
series at the end via the existing blocking analysis. For R = 4 on the N2
gate: memory 4x (~tens of MB, negligible), wall-time unchanged, error bar
~2x tighter.

## Reserve: smaller `kBins`

A smaller fixed `kBins` would help a *saturation-starved* fixture — more,
smaller bins mean less fork/join relative to the arithmetic — which is the
N2-class case, not HF. On HF the bins are already large enough (~430
parents/bin). Worth trying only if a real target turns out to be
saturation-limited. None of the landed work touches `kBins`.
