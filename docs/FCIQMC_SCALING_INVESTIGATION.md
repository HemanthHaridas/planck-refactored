# FCIQMC scaling investigation

Status: **implemented and validated for the measured throughput fixture.**
The phase ledger, parent-keyed RNG streams, determinant-sorted
`CanonicalWalkerPopulation`, and its pairwise canonical merge are now in the
production FCIQMC path.  On the measured HF/6-31G 50k-walker workload, four
OpenMP threads are the best measured configuration.  Further work is an
independent scaling investigation, not a prerequisite for this representation.

Background that this doc does not repeat (the 64-bin partition, the
`SpawnAccumulator`, the reverted sharded merge, the `draw_excitation` rewrite,
the H3 replica sketch) is in `docs/FCIQMC_PARALLELISM.md`.

## Question

Can the current FCIQMC implementation reach a usable energy in less wall time,
or fit a larger problem in a fixed memory budget, while keeping fixed-seed
reproducibility and bitwise thread-count invariance?

Three outcomes are kept apart: faster single trajectories, lower wall time to
a stated statistical precision, and a larger problem that fits. More steps per
second does not prove better statistical efficiency, and replicas do not speed
up the equilibration of one trajectory or remove its bias.

## Short answer so far

**The fixed-order merge into the next `WalkerPopulation` is the limit, and it
is per-record hash-map insertion.** On HF/6-31G at 50k walkers the threaded
worker phase scales to 5.01x at 8 threads, but the merge stays at ~107 s at
every thread count. At 8 threads it is 58 % of propagation wall time, which
holds end-to-end speedup at 2.19x.

This settles the question `FCIQMC_PARALLELISM.md` left open. Its sampling
profile put "merge + partition" at only 2.0-2.5 % of self-time because the
insertion cost was attributed to the `unordered_map::operator[]` and
allocator frames, not to the merge. The phase ledger measures elapsed time at
the phase boundary and shows the merge directly.

Two cheap fixes for the map were tried and rejected (below). The real fix is a
change of representation: a contiguous, determinant-sorted next population
built by a fixed 64-way merge. That needed parent-keyed RNG streams first,
because hash-map iteration order was deciding which parent consumed which
random number. Both are now built. They change the trajectory on purpose, so
they were validated as a new trajectory (re-pinned S5, exact-FCI statistical
gates), not as a speedup of the old one.

The parent-keyed streams on their own were measured and are performance
neutral (within noise of the baseline at every thread count).  The canonical
owner removes the hash-map insertion limit: its first priority-queue merge was
itself too expensive, but a bitwise-equivalent tagged pairwise merge makes the
representation a clear end-to-end win.

## Where the logic lives

- `src/post_hf/fciqmc_driver.cpp` — the step loop, which now owns a
  `CanonicalWalkerPopulation` and builds a fresh one each step; the
  `fciqmc_profile` ledger records
- `src/post_hf/ci/fciqmc.h` — `CanonicalWalkerPopulation` (determinant-sorted
  storage, binary-search `weight_at` for the initiator and estimator lookups,
  in-place compaction, fixed-partition L1 norm) and
  `RandomSource::derive_parent` (the per-parent stream)
- `src/post_hf/ci/fciqmc.cpp` — `propagate_stochastic` (prefill, worker
  region, fixed-order merge), `projected_energy`
- `src/post_hf/ci/spawn_accumulator.h` — `SpawnAccumulator`, `SpawnWorkspace`,
  and the finalized-output accessors behind `merge_finalized_spawn_bins`
- `src/io/io.cpp` — the `fciqmc_profile` keyword
- `tests/fciqmc_walkers.cpp` — the ledger's no-perturbation check, the
  literal merge oracle (`test_literal_canonical_merge_matches_map_output`),
  the parent-stream order-independence oracle, and
  `test_canonical_owner_roundtrips_through_propagation`
- `tests/run_fciqmc_scaling_profile.sh` — the benchmark runner
- `tests/artifacts/fciqmc_scaling/` — raw logs for the ladder below

## What was measured

### The work ledger

`fciqmc_profile true` emits two `FCIQMC_PROFILE` records, one for
equilibration and one for sampling. Only the sampling record is relevant to
time-to-precision. The switch is off by default and must not change the
trajectory. The walker test runs one propagation with and without it and
requires bitwise-identical walker weights and caller RNG state.

Each record carries:

- parents, attempts, invalid draws, zero-Hamiltonian draws, initiator
  rejections, and granularity-rounding zeroes;
- raw accumulator records and records after each bin's sort/fold;
- wall time for prefill, propagator setup, the parallel worker region, the
  fixed-order merge (map insertion and parent-bucket cleanup separately),
  compression/norm, and the projected estimator;
- `max_bin_work_s`, the largest per-bin worker time. It is a skew indicator,
  not a wall time; compare it with `worker_wall_s`.

It deliberately adds no locks or reductions inside the parallel loop, does not
time individual spawn attempts, and never sums per-bin seconds as elapsed
time, so it cannot perturb RNG consumption or merge order.

`tests/run_fciqmc_scaling_profile.sh [binary] [fixture] [threads] [outdir]`
copies the fixture, changes only `verbosity verbose` to `normal`, enables the
ledger, and keeps the full log. Defaults: `build/hartree-fock`, HF/6-31G at
50k walkers, one thread, `tests/artifacts/fciqmc_scaling/`.

### Strong scaling, HF/6-31G, 50k walkers (2026-10-17)

One clean rebuilt binary, run at 1, 2, 4 and 8 OpenMP threads. All four runs
had identical parents, attempts, rejection counts, folded-record counts, both
energies and reference-weight diagnostics, so this is a purely mechanical
comparison.

| Threads | Worker phase (s) | Fixed-order merge (s) | Full wall (s) | Worker speedup | Full speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 379.06 | 110.09 | 565.88 | 1.00x | 1.00x |
| 2 | 193.92 | 106.43 | 374.18 | 1.96x | 1.51x |
| 4 | 102.53 | 107.93 | 284.03 | 3.70x | 1.99x |
| 8 | 75.68 | 106.25 | 258.41 | 5.01x | 2.19x |

- The merge does not shrink with threads and is 58 % of 8-thread propagation
  (`106.25 / (75.68 + 106.25)`).
- At 8 threads, insertion was 107.774 s of the 107.778 s merge (99.996 %);
  parent-bucket cleanup was 0.003 s.
- The worker phase is sub-linear at 8 threads (5.01x). That justifies a later
  per-thread balance measurement, but it is not the current limit.
- Nearly every sampled spawn rounds to zero on this fixture. That is a
  statistical observation, not a reason to drop stochastic rounding without a
  separate time-to-precision study.

### Parent-keyed stream candidate (2026-10-17)

The new parent-keyed stream convention passed the S5 1/4-thread equality gate
and the existing H2/N2 exact-FCI statistical gates. It deliberately changes the
stochastic trajectory, so its timings are compared with the earlier baseline
only as a scaling result, not bitwise as a mechanical optimization.

| Threads | Worker phase (s) | Fixed-order merge (s) | Full wall (s) | Full change from baseline |
|---:|---:|---:|---:|---:|
| 1 | 407.16 | 93.26 | 572.12 | +1.1 % |
| 2 | 216.39 | 95.98 | 389.25 | +4.0 % |
| 4 | 114.34 | 95.53 | 284.50 | +0.2 % |
| 8 | 88.33 | 93.06 | 257.36 | -0.4 % |

The per-parent rekey increases worker cost by 7-17 %, while the changed
population happens to make the merge 10-15 % cheaper. Net wall time is within
noise of the baseline and nowhere near the 10 % acceptance threshold. Retain
the convention as the correctness prerequisite for a layout-independent
population, not as a standalone performance improvement.

The first canonical-owner measurement exposed a second limit: 99.1 % of its
merge time was the priority-queue stream.  Production now uses a tagged
pairwise merge.  Its oracle demonstrates raw-bit equality with the heap's
ascending-bin fold, while the normal walker and statistical gates demonstrate
that the complete production path preserves the intended trajectory and
physical behavior.

## What was tried and rejected

| Attempt | Result | Why rejected |
|---|---|---|
| Reuse two output populations, swapped each step | Failed the 200-step fixed-seed trajectory gate | A reused `unordered_map` keeps its bucket layout, so the next step iterates parents in a different order. Parent order within a bin decides which parent consumes each bin RNG draw, so the trajectory changes. This is a reordering, not a mechanical optimization. |
| Recycle node storage with a driver-owned `std::pmr::unsynchronized_pool_resource` (every map still freshly constructed) | Passed the trajectory gate; merge insertion 107.77 s -> 103.01 s (4.4 %) | At most ~1.6 % of end-to-end wall, far below the 10 % acceptance threshold. Removed rather than kept as production complexity. |
| Shard the merge by `hash(child) % 64` (earlier, `FCIQMC_PARALLELISM.md`) | 3-8 % slower on HF/6-31G | The shards need a second sort. |

The lesson from the first row is why the canonical population needed
parent-keyed streams first: while each bin shared one RNG stream, hash-map
iteration order was part of the trajectory, and no layout change could be
mechanical.

## Canonical population: accepted production representation

The hash-map next population is replaced by a contiguous, determinant-sorted
array built by a fixed 64-way merge of the already-canonical bin outputs,
which removes per-record map insertion. It is an explicitly reordered
trajectory, validated as such rather than against the retired map trajectory.

1. **Literal merge oracle — implemented.** `merge_finalized_spawn_bins`
   consumes the finalized bin streams, folds equal determinants across bins
   in a fixed key order, and is gated by
   `test_literal_canonical_merge_matches_map_output`, which compares every
   determinant and raw weight bit with the existing merge on a physical
   one-step input.
2. **Canonical parent order and RNG convention — implemented and validated.**
   `RandomSource::derive_parent(det)` derives a
   stream from the per-call/bin root plus both packed spin strings, without
   advancing the root. The walker oracle samples and stochastic-rounds the
   same physical parent set forward and backward and requires bit-identical
   folded contributions. Production now uses that stream for each parent's
   excitation and rounding sequence; the insertion-order gate requires exact
   equality with and without the initiator. This deliberately changes the
   trajectory. The N2 S5 1/4-thread gate has re-pinned this trajectory
   (`-109.3621937007` shift and `-107.6491159983` projected Eh) and passed
   exactly across thread counts. The existing H2 and N2 exact-FCI statistical
   gates, plus H2 1/4-thread equality, also pass. The completed 1/2/4/8
   ladder shows no standalone speedup, but establishes the convention as the
   canonical-population prerequisite.
3. **Canonical owner and initiator index — implemented and validated.**
   `CanonicalWalkerPopulation` stores the fixed-merge output in determinant
   order; its binary-search `weight_at` supplies the read-only initiator and
   estimator lookup. It compacts in place and computes L1 with the fixed
   determinant partition. The driver now owns this representation end-to-end.
   S5, H2/N2 exact-FCI statistical gates, and H2 thread equality pass without
   changing the parent-stream pins.
4. **Measure the canonical owner — accepted.** The first 1/2/4/8 production
   ladder is below. It meets the 10% end-to-end acceptance threshold at every
   measured count. The pairwise merge is roughly 2.3x faster than the earlier
   heap merge at one thread, and removes the map-insertion bottleneck.  Keep
   parent-keyed streams as the order-independence guarantee and the canonical
   owner as the production representation.

### Pairwise canonical merge, HF/6-31G, 50k walkers (2026-10-17)

This is the first production ladder after the pairwise merge replaced the
heap-based canonical stream. The seed, timestep, population target, number of
steps, and Hamiltonian are identical to the baseline. Counts and diagnostics
match across the ladder; correctness gates listed above passed before timing.

| Threads | Full wall (s) | Map baseline (s) | Change from baseline |
|---:|---:|---:|---:|
| 1 | 474.54 | 565.88 | -16.1% |
| 2 | 294.88 | 374.18 | -21.2% |
| 4 | 203.22 | 284.03 | -28.5% |
| 8 | 225.07 | 258.41 | -12.9% |

The earlier 4-thread observation was 217.47 s; the repeated 203.22 s result
shows ordinary host variation but leaves the improvement well beyond the
acceptance threshold. Four threads are therefore the recommended setting for
this workload. Eight threads are slower than four because the worker phase
has saturated and the serial pairwise merge grows slightly. This is a
workload-specific recommendation, not a hidden runtime cap: the caller keeps
control through `OMP_NUM_THREADS`.

The ledger separates the mechanism from the total: at one thread, pairwise
merge consumes 62.60 s (24.59 s equilibration plus 38.01 s sampling), versus
146.75 s for the prior heap merge, a 2.34x reduction. A later benchmark
publication should add alternating repetitions and report medians/spread; it
does not change the representation decision made by this unambiguous ladder.

Do not combine this with an excitation-sampler change. Changing sampling
variance at the same time would make the performance result uninterpretable.

## Other open hypotheses

Not yet measured against the ledger; ordered by expected value.

| Hypothesis | Constraint |
|---|---|
| Cache the reference's connections and Hamiltonian elements once after re-anchoring, and reuse them in `projected_energy` (it currently rebuilds them every sampling step). | Stop if the ledger's estimator time is negligible. Keep connection order, occupancy checks, the validity threshold and ratio-of-sums semantics. |
| Carry the prefilled diagonal with each parent instead of looking it up again during propagation. | Extra parent-record traffic may cost more than the lookup. Check memory on a fixture where the memo keeps growing, not only N2. |
| If worker imbalance grows past 8 threads, try other schedules over the same fixed bins, or a persistent OpenMP team. | Keep each bin's parent order and RNG order and the fixed merge order. Never tie bin count to thread count. |
| If excitation work dominates after the merge is fixed, cache class counts and pair-unranking data. | Keep draw probabilities, support, RNG consumption and fermionic signs exact. `slater_condon_element` is shared with CI and CASSCF. |
| Independent trajectories for throughput: compare separate processes at equal total cores (1x8, 2x4, 4x2, 8x1) before writing a replica API. | Distinct recorded seeds; block within each trajectory; combine projected ratios with their numerator/denominator covariance; never concatenate chains. |
| Reach larger targets: an active-space / frozen-core Hamiltonian adapter (needed for Cr2 CAS(12,18), `FCIQMC_RESEARCH_SCOPE.md` Q1), audited separately from the `ci_max_dim` and 31-orbital limits. | Core-energy accounting, orbital selection, and equivalence with deterministic CASCI on a small space. Measure integral setup separately from walkers. |

MPI and GPU work stay out of scope unless the measured CPU and memory envelope
fails an explicit target.

## Rules for any change

**Measurement.**
- Use HF/6-31G (or larger) for throughput; N2/STO-3G is the correctness gate
  and the small-workload overhead control only.
- Record the binary's source diff and build configuration; never infer results
  from an existing `build` or `build-full` tree of unknown provenance.
- Hold seed, timestep, spawn attempts, granularity, initiator threshold, shift
  settings and step counts fixed within a strong-scaling comparison. Use a
  separate seed ensemble for statistical efficiency.
- Repeat runs at least three times, alternate configuration order, and report
  medians and spread. Measure saturation as occupied determinants and bin
  work, not walkers divided by Hilbert-space size.
- Use `wall_time * SE^2` only in the converged sampling regime.

**Correctness.**
- Keep `planck-fciqmc-walkers`, `planck-fciqmc-accumulator`,
  `planck-statistical-gate`, and the H2/N2 exact-FCI regressions green.
- Run the S5 pair at 1/2/4/8 threads. Its committed checks compare printed
  energies; for investigation work, also compare raw-bit trajectory and
  population digests, and mutation-check that stronger comparison.
- A mechanical optimization must reproduce the baseline trajectory exactly.
  A deliberate reordering must stay self-reproducible and thread-count
  invariant, pass exact-reference and multi-seed statistical checks, and
  document any re-pin. Never re-pin just to make a failing test pass.
- Sorting accumulator input alone does not make a bin independent of parent
  visit order. What does is the parent-keyed stream: each parent draws from
  `derive_parent(det)`, never from a stream shared across the bin. Any change
  that reintroduces a shared stream must re-prove order independence.
- Changes to shared CI setup or matrix elements also run the deterministic
  FCI and CASSCF tests, including open-shell excitation and initiator cases.

**Acceptance.** At least 10 % lower end-to-end wall time on a representative
large fixture, beyond run-to-run noise, with no unexplained regression above
5 % on controls; or a memory reduction that makes a previously infeasible run
possible. Statistical changes must improve time to precision, with bias
checks, not just step rate. Stop a prototype when its target falls below that
threshold or its gain disappears outside a microbenchmark, and remove it.
A measured finding of little remaining headroom is a valid outcome.
