# FCIQMC Parallelism

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Can FCIQMC scale past the 1.57x the incremental spawn-loop threading gave, and what does the profile say the real ceiling is?**

## Short answer

The 1.57x (`docs/FCIQMC_T2_THREADING.md`) was a property of a *saturated*
fixture, not of the method. On an unsaturated fixture (HF/6-31G, `ndet` =
213 444) the parallel region threads to 3.44x at 4 threads and keeps
climbing. Four changes landed 2026-09-06, all bitwise thread-count
invariant, on the `fciqmc-parallel-rewrite-scope` branch:

1. **A `SpawnWorkspace` / `SpawnAccumulator` / xoshiro256** rewrite** of the
   per-call scaffolding — whole-call speedup 2.24x -> 2.42x/4t, 2.47x ->
   2.87x/8t on HF.
2. **Sharding the fixed-order merge** — built, correct, measured 3-8 %
   *slower* on HF, reverted. A `sample` reprofile showed the merge is 2.0 %
   of self-time, not the 37 % a phase probe had implied.
3. **`draw_excitation`** (40.8 % self-time, the real 1-thread hot path) —
   rewritten off four `std::array<int,32>` builds per call to
   `std::popcount` counts + a bounded `nth_set_bit` select. HF/6-31G
   1-thread wall -22 %; `draw_excitation` self-time -> 27.1 %.
4. **Folding the serial per-step driver tail** — 4 `pop` walks -> 2. ~5-7 %
   more wall.

| HF/6-31G, `verbosity normal`, `build-full` | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| pre-H2 (T2 threading only) | — | 2.24x | 2.47x |
| after the `SpawnWorkspace` rewrite | — | 2.42x | 2.87x |
| after `draw_excitation` (`hf_prof`, wall) | 78.1 s -> 60.5 s | — | — |
| after the fold (`hf_prof`, wall) | -> 61.0 s | 34.1 s -> 32.1 s | — |

The residual ceiling is the fixed-order merge — `unordered_map::operator[]`
latency on ~26k entries per step, serial, with no cheap parallelization.
The one axis that would give genuine near-linear scaling, **replica
parallelism (H3)**, is a design sketch, declined pending a real target.

All of this is gated on a real FCIQMC workload appearing
(`FCIQMC_RESEARCH_SCOPE.md` Q1) — nothing in the tree runs FCIQMC at a
size or duration where any of it matters.

## Where the logic lives

- `src/post_hf/ci/fciqmc.cpp` — `propagate_stochastic` (the spawn loop, its
  64-bin partition, the `#pragma omp parallel for schedule(static)` region,
  the fixed-order merge at the tail); `draw_excitation` and `nth_set_bit`;
  `WalkerPopulation::compress` / `compress_with_l1_norm`; `ordered_l1_norm`
- `src/post_hf/ci/fciqmc.h` — `WalkerPopulation` (the `unordered_map` the
  merge writes into); `RandomSource` (xoshiro256**, its fixed-seed
  contract, `derive`); `kFciqmcBins`
- `src/post_hf/ci/spawn_accumulator.h` — `SpawnAccumulator` (the
  reuse-stable per-bin accumulator, `finalize()`), `SpawnWorkspace` (the
  hoisted per-call scaffolding, `parents_prefilled`)
- `src/post_hf/fciqmc_driver.cpp` — the serial outer `for (step ...)` loop;
  `make_ops` and the T4 diagonal memo
- `tests/regression_cases.json` — `n2_fciqmc_s5_threads1/4` (the S5
  invariance gate; `threads1` pins both energies) and `h2_fciqmc_threads1/4`;
  `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g` (FCI-agreement gates)
- `tests/fciqmc_walkers.cpp`, `tests/fciqmc_accumulator.cpp` — the unit
  suites (`planck-fciqmc-walkers`, `planck-fciqmc-accumulator`)
- `tests/inputs/exploratory/fciqmc/validation/hf_base.hfinp` — the
  unsaturated timing fixture (HF/6-31G, `ndet` = 213 444), not a registered
  regression case
- `docs/FCIQMC_RESEARCH_SCOPE.md` §6 — the decided determinism policy
- `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` — the structurally identical
  scatter that threaded to 3.54x, and its determinism traps

## What invariants matter

### 1. A per-call ceiling that looks like a property of the method can be a property of one fixture

The incremental threading measured 1.57x on the N2/STO-3G gate and
concluded the per-step work was too fine-grained to thread further. That
conclusion did not survive a second fixture. Per-parent cost is a fixed
~148-202 ns and flat in walker count on both N2 and HF, but `parents/bin`
— and so whether the region threads — depends on whether the determinant
space is saturated:

| fixture | ndet | walkers | parents/call | parents/bin | region @4t | region @8t |
|---|---|---|---|---|---|---|
| N2/STO-3G (gate) | 14 400 | 10 000 | 908 | ~14 | 2.28x | 1.89x (regresses) |
| HF/6-31G | 213 444 | 50 000 | 27 505 | ~430 | 3.44x | 4.47x (climbing) |

N2's 14 400-determinant space is *full* at gate walker counts — every
determinant is occupied and more walkers only deepen weights — so bins get
~14 parents each, ~2 us of arithmetic against the fork/join cost. HF is
15x larger and unsaturated, so bins get ~430 parents (~87 us) and thread
like the FCI sigma build.

Design rule:

- Measure a parallel-scaling ceiling on a fixture whose problem size is
  representative of the target, not on the correctness fixture. N2/STO-3G
  is deliberately saturated so sampling is a real sample of a small space;
  that same choice makes it a misleading *throughput* fixture. Use
  HF/6-31G or larger for all parallel-FCIQMC measurement.

### 2. A phase probe that times wall-clock around a region attributes the whole region, including its parallel parts

An H2.7 per-phase probe reported the merge at ~1100 us/call and recorded
it as the sole remaining serial cost. A `sample` reprofile (self-time,
top-of-stack, 34 081 leaf samples, baseline HF/6-31G, 1 thread) contradicts
it:

| category | % | where |
|---|---|---|
| `draw_excitation` | 40.8 % | in the parallel region |
| `unordered_map::operator[]` | 11.3 % | 4 call sites; ~2 % of it the merge |
| `slater_condon_element` | 11.1 % | in the region |
| spawn-loop body | 8.4 % | in the region |
| allocator + memset | 7.3 % | accumulator vectors |
| diagonal memo lookup | 6.7 % | in the region |
| `SpawnAccumulator::finalize()` sort | 5.5 % | already inside `#pragma omp`, already threads |
| serial merge + partition | 2.0 % | the `out.add` concat + parent partition |
| `compress()` | 1.8 % | |
| `ordered_l1_norm()` | 1.8 % | |

The serial merge is 2.0 %, not 37 %. The probe's bracket around "the merge
phase" was dominated by the per-bin `finalize()` sort (5.5 %, which runs
*inside* the threaded region) plus accumulator-vector allocation.

Design rule:

- A phase probe measures wall-clock spent between two points, including
  work that threads. Cross-check a "serial bottleneck" claim with a
  leaf-sample self-time profile before restructuring around it. `rehash`
  is 0.2 % here — the walker map is warm across steps — so a
  "cache-miss / rehash bound" hypothesis was also wrong; `operator[]`
  self-time is steady-state hash+probe.

### 3. A merge that is 99 % concatenation is still not worth sharding if the shard needs a sort

The cross-bin collision rate on HF/6-31G is 0.73 % — the merge is 99.27 %
concatenation of 64 already-deduplicated, already-sorted lists. The obvious
fix (64 output shards keyed `hash(child) % 64`, filled by a serial
scatter, `finalize()`d in a threaded region, concat'd in fixed order) was
built and passes every correctness gate, but measured slower on both
fixtures:

| fixture | threads | baseline | sharded | Δ |
|---|---|---|---|---|
| N2/STO-3G, 50k steps | 1 | 8.6 s | 12.9 s | +50 % |
| HF/6-31G, 2k+3k steps | 1 | 53.3 s | 55.3 s | +3.8 % |
| HF/6-31G, 2k+3k steps | 4 | 26.2 s | 28.2 s | +7.8 % |
| HF/6-31G, 2k+3k steps | 8 | 23.2 s | 24.7 s | +6.4 % |

The sharded path replaces a 2.0 % serial `out.add` loop with a 2.0 %
serial scatter + *another* copy of the `finalize()` sort (64 shard-sorts
on top of the 64 bin-sorts, an `O(n log n)` term where there was an `O(n)`
one) + a second fork/join.

Design rule:

- A near-pure concatenation only shards for a win if the shards can be
  concatenated without a sort. If deduplicating the 0.73 % that collide
  requires re-sorting the shard, the sort dominates and the plain
  warm-hash merge is already close to optimal for this access pattern.

### 4. `draw_excitation` builds four full orbital lists to use at most two entries of one

`draw_excitation` opened with a 672-byte stack frame, then — *before any
random draw* — built four `OrbitalList` (`std::array<int,32>`) by value
(512 bytes of stack zeroing per call) and ran four `n_act`-bit scan loops,
to then use only the four *counts* (to pick an excitation class) and one
or two orbital-index lookups within the chosen spin channel. That was
40.8 % of self-time, and it threads fine — so it is a serial-efficiency
target, not a parallelism one.

Design rule:

- When a hot function computes a rich structure to consume a scalar
  summary of it, replace the structure with the summary. Here: four
  bitmasks over the active window + `std::popcount` for the counts, and a
  bounded `nth_set_bit(mask, i)` select (exits at the i-th set bit, called
  at most twice per spawn on one channel) for the lookups. Keep the full
  builder for the cold path (`enumerate_connections`, the brute-force
  oracle).

### 5. A pure-arithmetic refactor of a sampler must stay bitwise, and the gate that proves it must be non-vacuous

The `draw_excitation` rewrite and the driver-tail fold are both
reassociations of *nothing* — `uniform_int` is called in the same order
with the same arguments, and the orbital each drawn index maps to is the
same orbital — so S5 must stay bit-identical at `atol = 0.0`, unlike the
`SpawnAccumulator` and RNG changes which legitimately reassociate. But a
partition or merge-order defect is *thread-count-invariant*, so a
`threads1 == threads4` comparison cannot see it.

Design rule:

- `n2_fciqmc_s5_threads1` pins both energies to fixed values;
  `n2_fciqmc_s5_threads4` compares against `threads1`. Verify
  non-vacuity by mutating the partition — binning by `det.alpha` instead
  of the full hash, or reversing the bin-merge order — and confirming the
  pin moves while T1 == T4. The `nth_set_bit` refactor is separately
  mutation-verified: `i == 0` -> `i == 1` produces 19 failures in
  `planck-fciqmc-walkers` (the `p_gen` corruption cascades through
  spawning).

### 6. Folding serial passes only helps if the folded passes were the bottleneck

The serial per-step driver tail is 4-5 full traversals of the ~26k-entry
walker hash map (`verbosity normal`): diagonal prefill, partition, merge,
`compress`, `ordered_l1_norm` (+`signed_population` at `verbose`, already
guarded). Folding the *non-merge* passes pairwise — prefill+partition into
one walk, `compress`+`ordered_l1_norm` into one — bought ~5-7 % wall. The
4-thread number barely moved, because the merge (the biggest pass) is
untouched.

Design rule:

- Before threading a folded pre-pass, check whether folding it already
  moved the target metric. If it did not, the folded work was not the
  dominant serial cost and threading it chases the smaller half. Here the
  merge is the dominant half, and it has no cheap parallelization — which
  leaves H3 as the only route to near-linear.

## What was fixed

1. **`SpawnAccumulator` replaces the per-bin `unordered_map`.** A
   `vector<pair<DetKey, Weight>>` whose `finalize()` sorts by a total order
   on `(alpha, beta, bit-pattern of weight)` then folds equal-key runs
   left-to-right, so the sum is a pure function of the multiset. `reset()`
   is `vector::clear()` — a genuine reset, unlike `unordered_map::clear()`
   which keeps the peak bucket count. The weight-bit tiebreak is
   load-bearing: `std::sort` is not stable, so a >=3-long same-key run
   would otherwise fold in insertion order. Gated by
   `planck-fciqmc-accumulator` against an independent `std::map` reference;
   mutation-verified.

2. **`SpawnWorkspace` hoists the per-call scaffolding.** The 64
   accumulators, 64 parent buckets, and 64 RNG streams, built once by the
   driver instead of ~50 000 times. `propagate_stochastic` gained a
   `void ...(ws, out)` form; the value-returning form is kept as a thin
   overload so the ~20 test call sites are untouched.

3. **`RandomSource`'s engine swapped `mt19937_64` -> xoshiro256**.**
   `mt19937_64::seed()` is ~600 ns each (the 312-word state fill); at 64
   per-bin streams re-seeded every step that is ~38 us/call, and reusing
   the engine object cannot avoid it — measured three ways: fresh
   construct 42 us, reuse + `seed()` 39 us (the 2 us gap is only the
   vector alloc), counter-based 74 ns. xoshiro256**'s "seed" is four
   SplitMix64 outputs into a 256-bit state, O(1). Public surface and the
   53-bit `uniform()` unchanged; the 64 streams live in `ws.bin_rngs`,
   re-keyed in place each call by the identical `derive()` recipe. This
   alters every trajectory — a reordering-class change, gated on
   self-reproducibility + thread-count invariance + `metric_within_sigma`
   vs exact FCI, not on matching old numbers.

4. **`draw_excitation` off `popcount` + `nth_set_bit`** (invariant 4).
   Bit-identical. `draw_excitation` self-time 40.8 % -> 27.1 %; HF/6-31G
   1-thread wall 78.1 s -> 60.5 s (-22 %) at `verbosity normal`, N2 gate
   -10 %. The residual 27.1 % is arithmetic that stays: the class-size
   products (`na*(na-1)/2 * va*(va-1)/2`), two `uniform_int` calls,
   `unrank_pair`'s loop, the `k/va` / `k%va` divides.

5. **Two per-step `pop` walks folded into one each.**
   - Fold A: the driver's diagonal-prefill loop also fills
     `SpawnWorkspace::parents` (same `DetKeyHash{}(det) % kFciqmcBins`
     binning, same `w == 0.0` skip); a `parents_prefilled` flag makes
     `propagate_stochastic` skip its own partition. `kBins` moved to the
     header as `kFciqmcBins` so the two partitions provably match.
   - Fold B: `WalkerPopulation::compress_with_l1_norm` does the compress
     erase-scan and the ordered-L1-norm bin-and-sum in one traversal. The
     norm is byte-identical to `compress()` then `ordered_l1_norm()` —
     same survivors, same fixed 64-bin partition, same fixed 0..63
     summation order. Gated by a new unit assertion.

   Bit-identical, S5 non-vacuity re-verified. HF `hf_prof` -5 %/1t,
   -6 %/4t; N2 gate -7 %/4t.

**Reverted:** sharding the merge (invariant 3) — built, correct,
deterministic, measured slower on both fixtures. `git diff` for that
investigation is docs-only.

## Validation strategy that should remain in place

- **Bitwise thread-count invariance at `atol = 0.0`** across
  `OMP_NUM_THREADS` = 1/2/4/8, via the S5 gate with a *pinned* `threads1`
  (invariant 5). Kept structurally by partition-by-parent: each parent
  hashes to one of `kFciqmcBins = 64` fixed bins (never tied to thread
  count), each bin accumulates independently, bins merge in fixed 0..63
  order — never `omp atomic`, never completion-order (the DFT-grid jitter
  defect; every other parallel path in this codebase is bitwise
  thread-count-invariant by design).
- **`planck-fciqmc-walkers`** — the F2 `p_gen` oracle (frequency *and*
  support, open-shell cases) is what catches a `nth_set_bit` off-by-one or
  a wrong class size. **`planck-fciqmc-accumulator`** for the
  `SpawnAccumulator` fold order.
- **`metric_within_sigma` vs exact FCI** on `h2_fciqmc_sto3g` and
  `n2_fciqmc_sto3g` — the changes that reassociate (accumulator, RNG) are
  gated here, never on matching pre-change numbers.
- **The 4 non-QMC FCI gates** — `build_all_mo_ci_setup` is a move out of
  `run_fci`, and `slater_condon_element` is public and consumed by
  CASSCF/RASSCF response code, so those paths are downstream and must stay
  green.
- **HF/6-31G, not N2/STO-3G, for throughput measurement** (invariant 1).

## What was measured after the fixes

**Where the 1-thread time is now** (HF/6-31G, `sample`, post-`draw_excitation`):
`draw_excitation` 27.1 %, `unordered_map::operator[]` 15.2 %,
`slater_condon_element` 12.5 %, spawn-loop body 10.6 %, allocator 8.7 %,
diagonal memo 8.2 %, `finalize()` sort 7.3 %, serial merge/partition 2.5 %.
Everything scaled up proportionally from the pre-refactor profile — the pie
is smaller, not reshaped.

**Where the 4-thread time goes:** ~54 % barrier idle. Three workers park
while one runs the serial per-step driver tail. Folding the non-merge
passes (fix 5) moved the 4-thread wall ~6 %; the residual is the merge and
the fork/join around a region that, post-`draw_excitation`, is smaller
relative to the unchanged serial tail.

**Net HF/6-31G whole-call speedup:** 2.24x -> 2.42x at 4 threads, 2.47x ->
2.87x at 8 threads from the `SpawnWorkspace` rewrite; the region ceiling on
that fixture is >=4.5x at 8 threads. The gap between whole-call and region
is the serial tail, dominated by the merge.

## Remaining architecture concern

**The merge is the residual serial cost and has no cheap parallelization.**
It is `unordered_map::operator[]` latency on ~26k entries per step. Sharding
it was tried and reverted (invariant 3). A fixed-order parallel scatter
(prefix-sum the per-bin sizes, threaded scatter into one flat output array
by precomputed offset, ordering fixed by the offsets so no completion-order
hazard) is possible but is a careful target for a real large-`ndet`
workload, where 26k -> 260k might change whether the sort or the scatter
dominates.

**H3 — replica parallelism — is the only route to near-linear, and is not
built.** FCIQMC literature runs multiple independent replicas (different
RNG seeds, same Hamiltonian) both for error bars and as a parallelization
axis; each replica is an embarrassingly parallel full trajectory with zero
shared mutable state. Planck has the reproducibility machinery
(`RandomSource`, fixed-seed contract) and the statistical machinery
(`blocked_standard_error`, `metric_within_sigma`) to combine them. It is
not built because it does not speed up a *single* trajectory — it tightens
the error bar in fixed wall-time — and it multiplies memory by R (R walker
populations). For the committed gates, seconds of runtime already inside
their error bars, it buys nothing. Sketch: an outer replica loop,
`#pragma omp parallel` over replicas, each with its own
`RandomSource(seed + replica_index)`, `ShiftController`, and
`WalkerPopulation`; combine the per-replica series at the end via the
existing blocking analysis. The recommendation is to build it when a
target system makes one trajectory minutes-to-hours and the error bar the
deliverable — the same conclusion `FCIQMC_RESEARCH_SCOPE.md` Q1 reaches
for the method as a whole.

**A smaller fixed `kBins`** would help a *saturation-starved* fixture (more,
smaller bins, less fork/join relative to the arithmetic) — the N2-class
case, not HF. None of the landed work touches `kBins`; worth trying only if
a real target turns out to be saturation-limited.
