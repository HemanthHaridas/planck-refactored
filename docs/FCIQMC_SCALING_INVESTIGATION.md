# FCIQMC scaling investigation

Scoped 2026-10-11 against working-tree source at HEAD `6c7a02ac`.
Status: investigation proposed; no new performance measurements or solver changes.
The checkout contains unrelated local changes, so HEAD alone does not identify a
future benchmark binary. Record the relevant source diff and build configuration.

## Decision to answer

Can the current implementation reduce wall time to a scientifically usable
FCIQMC energy, or reach a larger problem within a fixed memory budget, while
preserving fixed-seed reproducibility and bitwise thread-count invariance?

Separate three outcomes: faster individual trajectories, lower wall time to a
specified statistical precision, and greater problem-size capacity. More steps
per second does not establish better statistical efficiency; multiple replicas
do not accelerate equilibration of one trajectory or remove systematic bias.

**Working assessment:** further gains are plausible. There are concrete repeated
work and allocation targets in the current source, but neither a dominant
bottleneck nor a speedup magnitude is established. Start with a reproducible
baseline and two small prototypes, not a distributed rewrite.

## What the current source establishes

| Area | Evidence | Investigation consequence |
|---|---|---|
| Spawn parallelism | `ci/fciqmc.cpp:propagate_stochastic` uses 64 fixed parent bins and `omp parallel for schedule(static)`; each bin has its own RNG and accumulator. | Threading already exists. At most 64 bin iterations can execute concurrently; measure bin work and barriers before changing scheduling. |
| Accumulation | `ci/spawn_accumulator.h` appends to vectors, sorts by determinant and weight bits, then folds; the cross-bin merge is serial in bin order. | Sorting already runs inside the parallel region. Separate its cost from the serial merge. |
| Output allocation | `fciqmc_driver.cpp:run_fciqmc` constructs `WalkerPopulation next_pop` inside every step and move-assigns it to `pop`. | Workspace vectors persist, but the output map does not. Measure node allocation, bucket allocation, and destruction before claiming a warm-map baseline. |
| Diagonal cache | `fciqmc_driver.cpp:make_ops` stores an unbounded map. The driver prefills it, and propagation looks up each parent's diagonal again. | Measure duplicate lookup cost and cumulative distinct determinants. Memory follows visited determinants, not just current occupancy. |
| Projected estimator | `ci/fciqmc.cpp:projected_energy` enumerates reference connections every valid sampling step and evaluates matrix elements for occupied connections. | The Hamiltonian and sampling reference are fixed after re-anchoring. Repeated connection construction and matrix elements are candidates for caching. |
| Serial tail | Prefill and partition are fused; compression and L1 norm are fused. Projected estimation and shift control remain outside propagation. Verbose mode also updates a cumulative coefficient map. | Profile the complete driver at normal verbosity, with a separate verbose diagnostic measurement. |
| Reachability | `fci.cpp:build_all_mo_ci_setup` uses all MOs and all electrons, applies `ci_max_dim`, limits packed spatial orbitals to 31, and builds dense MO integrals. | Sparse walkers alone do not remove setup and admission limits. No active-space FCIQMC driver is wired here. |

Paths above are relative to `src/post_hf/`.

The historical records need careful interpretation:

- `FCIQMC_PARALLELISM.md` records 2.42x at four threads and 2.87x at eight
  for the workspace rewrite on HF/6-31G. Later sampler and driver changes also
  landed. These are historical results, not a fresh measurement of this checkout.
- That document reports only 2–2.5% serial merge/partition self-time in one
  profile, but also calls the merge the dominant residual limit. Neither leaf
  CPU samples nor worker barrier percentages alone determine serial wall time.
  Reconcile the attribution before investing in another merge rewrite.
- The same record describes a warm output map; the current driver constructs
  a new one every step. The source takes precedence.
- Saturation must be measured as occupied determinants and bin work. Nominal
  walker weight divided by Hilbert-space size does not establish saturation.
- S5 is already present: `n2_fciqmc_s5_threads1/4`. Older notes calling it
  unbuilt, or FCIQMC entirely serial, are superseded.
- The Cr2 target input selects CASSCF CAS(12,12), not FCIQMC CAS(12,18).
  It cannot serve as a large FCIQMC benchmark without additional setup work.

## Stage 1: establish the scaling envelope

Deliver one repeatable benchmark runner, a machine/build manifest, raw results,
and a phase/memory report. Keep timing probes optional and verify their overhead
against an uninstrumented build. Do not infer current results from existing
`build` or `build-full` binaries without establishing their provenance.

### Workloads and controls

1. Use N2/STO-3G S5 for quick correctness and the full N2 gate for exact-FCI
   agreement. Include N2 timings as the small-workload overhead control.
2. Use the existing HF/6-31G validation inputs as the first throughput ladder:
   target weights 10k, 25k, 50k, and 100k; measure actual occupancy at each.
   Make benchmark copies at `verbosity normal` because `hf_base.hfinp` is
   verbose. Preserve its other parameters and record every override.
3. Screen one larger reachable, nondegenerate molecular fixture for sustained
   occupancy materially above HF, ideally roughly 10x. Check orbital count,
   electron count, dimension admission, SCF convergence, timestep stability,
   and memory before accepting it. If none is practical, report that coverage
   limit and use a synthetic population replay only for kernel measurements.
4. Vary threads over 1/2/4/8 and available physical cores, avoiding unintended
   oversubscription. Record CPU/core topology, affinity and OpenMP runtime,
   compiler/optimization flags, other numerical-library thread settings, and
   input/source hashes. On heterogeneous CPUs distinguish core types.

Start the full thread sweep at HF/50k, then expand population and problem size
where it distinguishes hypotheses. Use at least three repeated runs, alternate
configuration order, and report medians and dispersion. Separate setup,
equilibration, and stationary sampling. Short smoke runs cannot establish
equilibration or statistical precision; extend runs until blocking is useful.

Hold seed, timestep, spawn attempts, granularity, initiator threshold, shift
settings, and step counts fixed for each strong-scaling comparison. Use a
separate seed ensemble for statistical efficiency. Treat population scaling
and orbital/electron scaling as different experiments; do not fit one exponent
across chemically different systems.

### Required observations

- End-to-end and sampling wall time, time per step, speedup `T1/Tp`, parallel
  efficiency `T1/(p*Tp)`, and attempted/accepted spawns per second.
- Disjoint timings for diagonal prefill/partition, workspace reset/rekey,
  spawn/death, per-bin sort/fold, fixed-order merge, map destruction,
  compress/norm, projected estimation, population control, and diagnostics.
  Separate per-thread CPU work from elapsed critical-path time; reconcile
  phase totals with driver wall time and sampling profiles.
- Per-bin parent counts, attempts, accepted children, elapsed work, and output
  sizes; cross-bin collisions and barrier time. Weight alone is a poor work
  proxy: attempts are per occupied parent in this implementation.
- Live occupancy, cumulative diagonal-cache entries, cache misses, allocation
  counts/bytes, vector capacities, and peak RSS. At fixed nominal population,
  compare long-run memory against cumulative visitation.
- Both energy estimates, blocked errors, reference-weight health, population
  stability, and autocorrelation/blocking diagnostics. Compare time to a
  predeclared precision after checking timestep, population/initiator, and
  equilibration effects. Use `wall_time * SE^2` only in the converged sampling
  regime where that efficiency comparison is meaningful.

## Stage 2: bounded prototypes, ordered by evidence

Implement and measure one change at a time. These are hypotheses, not promised
improvements; select at most two initial prototypes after Stage 1.

| Priority | Hypothesis and experiment | Falsification / constraint |
|---|---|---|
| A | Cache the ordered reference connections and their Hamiltonian elements once after sampling-reference selection. Reuse them in projected estimation. | Stop if estimator cost is negligible. Preserve connection order, occupancy checks, validity threshold, and ratio-of-sums semantics; account for the connection table's memory. |
| A | Remove output-map churn using two caller-owned populations exchanged each step, then evaluate capacity reservation or a node pool if allocation remains significant. | Reusing buckets alone does not eliminate node allocation in `unordered_map::clear()`. Measure allocation and whole-run time. Map layout can change next-step parent order and RNG assignment, so this may require the reordering validation standard. |
| B | Carry the prefilled diagonal with each parent record to avoid the second hash lookup. If the cache grows with visitation, compare a bounded cache or live-population diagonal storage. | Additional parent-record traffic may exceed lookup savings. Compute misses safely outside shared parallel reads. Check memory and speed on a churn-heavy workload, not only N2. |
| B | If barriers or imbalance dominate, test scheduling over the same fixed bins and a persistent OpenMP team. | Preserve each bin's parent visitation/RNG order and fixed merge order. A persistent team removes repeated region entry, not serial dependencies. Do not scale bin count with thread count. |
| C | If the measured merge remains material at large occupancy, test a contiguous population or deterministic ownership layout. Preserve per-child contribution order using explicit bin/order keys. | A parallel scatter alone does not perform annihilation. Include sorting/deduplication, lookup needs, and conversion costs. The previous extra-sort sharded merge was 3–8% slower on HF; do not repeat it without a changed cost regime. |
| C | If excitation work dominates, cache invariant class counts/pair-unranking data; separately evaluate excitation-aware matrix-element calls. | Keep draw probabilities, support, RNG consumption, and fermionic signs correct. Shared Slater–Condon consumers need regression coverage if changed. |

For a changed sampler, increased spawn count, importance sampling, or a
semi-stochastic extension, require a separate algorithm experiment judged by
time to precision. Such changes can alter variance and bias and are not the
first implementation-scaling prototypes.

## Stage 3: throughput and capacity branches

**Independent trajectories:** first compare separate processes at equal total
core budgets (for example 1x8, 2x4, 4x2, 8x1) before adding a replica API.
Measure repeated setup/equilibration and memory costs. Give every run a distinct
recorded seed and private mutable state; avoid nested OpenMP oversubscription.
Block within trajectories and combine estimates with a defined uncertainty
calculation that preserves numerator/denominator covariance for projected
ratios. Do not concatenate chains as one stationary series or assume replicas
remove population/initiator bias. This tests throughput, not faster dynamics.

**Larger scientific targets:** audit the FCI-derived dimension admission limit
and input integer range separately from actual allocation limits. Scope an
active-space/frozen-core Hamiltonian adapter if Cr2 CAS is the chosen target;
include core-energy accounting, orbital selection, and small-space equivalence
to deterministic CASCI. Measure integral setup/storage separately from walkers.
Changing the 31-orbital representation is a further project, not a threading fix.

MPI distribution and GPU work remain follow-ons only if the measured CPU/memory
envelope fails an explicit target. Neither is required to answer this initial
investigation, and no claim of near-linear scaling should precede measurements.

## Correctness and acceptance

- Retain `planck-fciqmc-walkers`, `planck-fciqmc-accumulator`, and
  `planck-statistical-gate`, plus H2/N2 exact-FCI regression comparisons.
- Run S5 at 1/2/4/8 threads and repeated fixed seeds. Its committed checks pin
  or compare two *printed* energies at zero tolerance; strengthen investigation
  checks with raw-bit trajectory/population digests so sub-print-precision or
  transient differences cannot hide. Mutation-check the stronger comparison.
- Canonicalizing accumulator input does not make a stochastic bin independent
  of parent visitation order: its RNG draws are still consumed sequentially.
  Preserve that order for changes claimed to be trajectory-identical.
- Pure mechanical optimizations must retain the baseline trajectory. Deliberate
  reordering must retain self-reproducibility and thread-count invariance, pass
  exact-reference and multi-seed statistical checks, and document any pin change.
  Never repin merely to make a failed test pass.
- Changes to shared CI setup or matrix elements also run the affected deterministic
  FCI/CASSCF tests. Exercise open-shell excitation support and initiator behavior
  for changes touching those paths.

Proposed acceptance threshold: at least 10% lower end-to-end wall time on a
representative large fixture, beyond run-to-run noise, with no unexplained
regression over 5% on controls; alternatively, demonstrate a useful reduction
in peak memory that enables a previously infeasible run. Statistical changes
must improve time to precision with bias checks, not just step rate. Small,
simple gains may be recorded without justifying a larger redesign.

Stop a prototype when its target is below the useful cost threshold, total
runtime regresses, or gains disappear outside a microbenchmark. Re-profile after
each accepted change. The final deliverable is a measured recommendation:
which changes to land, which were rejected and why, and the remaining wall-time,
memory, and scientific-accuracy limits. A defensible finding of little remaining
headroom on the tested workloads is a successful investigation outcome.
