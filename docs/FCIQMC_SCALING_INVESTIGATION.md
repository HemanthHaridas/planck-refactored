# FCIQMC Scalability

## Purpose

This document records the production FCIQMC propagation design, the measured
performance envelope, and the evidence required before another optimization is
accepted. It is deliberately a current-state document: superseded experiments
are not retained as an implementation plan.

The scope is one fixed-seed FCIQMC trajectory. An accepted optimization must
not change its sampling law, support, `p_gen`, deterministic reduction order,
or fixed-seed trajectory. Replica throughput, MPI/GPU execution, and
active-space/frozen-core FCIQMC are separate capability projects.

## Production architecture

`CanonicalWalkerPopulation` stores the live wavefunction as a sorted vector of
`(DetKey, weight)` entries. This gives deterministic traversal, binary-search
lookup for initiator and estimator queries, and in-place compaction.

One propagation step is:

1. Partition parent determinants into 64 fixed bins by parent hash. The bin
   count does not depend on the OpenMP thread count.
2. Derive each parent's random stream from its determinant. Excitation and
   stochastic-rounding draws therefore do not depend on container layout or
   visitation order.
3. Accumulate each bin privately, then sort and fold each bin in canonical key
   order.
4. Merge the 64 finalized streams through a tagged, balanced pairwise tree.
   Equal keys retain ascending source-bin association at every tree node.
5. Store the sorted result in the next canonical population, compact it, then
   update population control and the projected estimator.

The production merge is
`merge_finalized_spawn_bins_pairwise_parallel` in
`src/post_hf/ci/spawn_accumulator.h`. Independent pair merges at a tree level
run in parallel, but each result is written to a predetermined slot; the tree
shape and floating-point association are unchanged. The sequential pairwise
tree is retained only as a test oracle.

## Correctness contract

The following properties are non-negotiable.

- Parent partitioning, not child partitioning, defines independent work.
- The 64-bin layout and ascending-bin equal-key reduction association are
  invariant across OpenMP thread counts.
- Parent-keyed RNG streams make generated excitations independent of traversal
  order.
- `p_gen` equals the probability for the exact generated excitation; support
  and observed frequency are tested separately.
- A mechanical optimization reproduces the fixed-seed trajectory bitwise. Any
  intentional statistical change needs a new, explicitly documented reference.

`planck-fciqmc-walkers` covers excitation support, `p_gen`, literal sampler
equivalence, fixed-seed reproducibility, canonical-population propagation, and
the parallel-tree raw-bit oracle. `planck-fciqmc-accumulator` covers local
ordered folding and merge behavior. N2 S5 runs at one and four threads are the
fixed-seed integration gate; H2/N2 exact-FCI cases remain statistical controls.

## Measured performance

The representative throughput fixture is
`tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp`: HF/6-31G,
50,000 walkers, 20,000 equilibration steps, and 30,000 sampling steps. Each
accepted figure uses the same seed, timestep, walker target, Hamiltonian, and
step counts. Timings are three-run alternating medians unless stated otherwise.

| configuration | 1 thread wall | 2 thread wall | 4 thread wall | 8 thread wall |
|---|---:|---:|---:|---:|
| map population baseline | 565.88 s | 374.18 s | 284.03 s | 258.41 s |
| canonical population + pairwise merge | 474.54 s | 294.88 s | 203.221 s | 225.07 s |
| E1 rank-select + M1 parallel tree | 403.748 s | — | **163.958 s** | — |

Four threads are the best measured configuration for this workload. This is a
recommendation, not a runtime limit; callers continue to select
`OMP_NUM_THREADS`.

At four threads, M1 changed the phase ledger as follows:

| phase | E1 median | M1 median | change |
|---|---:|---:|---:|
| sampling worker | 50.233810 s | 51.357996 s | +2.2% |
| sampling pairwise merge | 38.811332 s | 24.049039 s | -38.0% |
| full wall | 182.431 s | 163.958 s | -10.1% |

All three M1 runs have identical parent, attempt, raw-record, folded-record,
and final-energy values. The small worker increase is the cost of materializing
tree levels and is more than recovered by the parallel merge.

## Landed optimization decisions

| item | decision | evidence |
|---|---|---|
| Canonical population | landed | removes hash-layout traversal dependence |
| Parent-derived RNG | landed | deterministic trajectory across layout and threads |
| Tagged pairwise merge | landed | raw-bit equality with the heap/reference fold; faster than heap merge |
| E0 literal excitation oracle | landed, test-only | exact excitation, `p_gen`, phase, validity, and RNG-state equality |
| E1 rank selection | landed | 1.112x focused benchmark; 10.2% four-thread full-run wall reduction before M1 |
| M1 level-parallel pairwise tree | landed | sequential-tree raw-bit oracle; N2 S5 1/4-thread gate; 10.1% additional wall reduction |
| E2 table dispatch | rejected | exact supplied-input descriptor oracle, but 0.968x versus the switch |
| Reused map populations | rejected | bucket layout changes parent traversal |
| PMR node recycling | rejected | negligible end-to-end return |
| Child-hash sharded merge | rejected | second sorting stage loses performance |

E1 replaces a scan-to-the-selected-bit implementation of `nth_set_bit` with
repeated least-significant-set-bit clearing followed by `countr_zero`. It does
not change excitation classes, RNG calls, signs, or `p_gen` arithmetic order.

## Profiling and acceptance workflow

Use the phase ledger to establish the dominant phase, then use a CPU sample to
identify the responsible routine. A stack sample is attribution evidence, not
a performance result. Do not modify a candidate until both identify a concrete
mechanism.

```sh
tests/run_fciqmc_cpu_profile.sh \
  build/hartree-fock \
  tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp \
  1 30 /private/tmp/fciqmc-profile

tests/run_fciqmc_cpu_profile.sh \
  build/hartree-fock \
  tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp \
  4 30 /private/tmp/fciqmc-profile
```

An accepted change requires:

1. an isolated, deterministic oracle for the altered operation;
2. walker and accumulator invariant tests plus fixed-seed N2 S5 at one and
   four threads;
3. H2/N2 statistical regression controls when sampling behavior is involved;
4. three alternating representative fixture runs; and
5. at least 10% end-to-end improvement beyond noise, no unexplained control
   regression above 5%, or a documented memory reduction that enables a
   previously infeasible calculation.

## Post-M1 attribution and next decision

Fresh 30-second one- and four-thread CPU samples were collected after M1 on
the representative fixture. Their ledgers preserve the expected trajectory
counts. At four threads, sampling spends `49.528 s` in worker propagation and
`23.355 s` in pairwise merging; M1 therefore remains effective, but worker
propagation is again the dominant scalable cost.

In the one-thread sample, `draw_excitation` accounts for 7,135 of 16,064
propagation-stack samples (44.4%); at four threads it accounts for 3,544 of
8,320 (42.6%). `slater_condon_element` and `SpawnAccumulator::finalize()` /
sorting remain visible secondary costs. This authorizes an excitation-generator
attribution benchmark, but not a production dispatch rewrite: the sampled
frame still includes rank selection, random draws, orbital updates, parity,
and class selection.

### E2 — decomposed excitation-generator benchmark

Build a deterministic, test-only benchmark that measures the current literal
generator and a candidate dispatch mapping separately from orbital updates,
parity, and RNG generation. Both paths must consume the same supplied class
and rank values and produce exactly the same excitation descriptor and
`p_gen`. The benchmark must report a digest and per-component times over the
same closed-shell, open-shell, and edge active-space corpus used by E0.

Only if that benchmark attributes a material fraction to dispatch may a
branch-only production candidate be written. It must preserve class order, RNG
calls, signs, and `p_gen` operation order, then pass the normal exact and
full-trajectory gates.

**Closed as a non-candidate (2026-10-20):** `planck-fciqmc-walkers` constructs
plans with a supplied non-empty class and within-class rank. A literal `switch`
and a table-dispatch candidate receive the same plans; neither path draws RNG,
selects orbitals, evaluates parity, or updates a determinant. The invariant
requires raw descriptor and `p_gen` equality, then materializes both results
to confirm equal excitations. The opt-in benchmark is:

```sh
PLANCK_FCIQMC_DISPATCH_BENCHMARK=1 build/planck-fciqmc-walkers
```

`PLANCK_FCIQMC_DISPATCH_BENCHMARK_DRAWS` and
`PLANCK_FCIQMC_DISPATCH_BENCHMARK_ROUNDS` override the defaults. Five
alternating 10,000,000-draw rounds gave `0.033586 s` for the literal switch
and `0.034704 s` for table dispatch: **0.968x**, with a matching digest. The
indirect call loses, so production retains the switch and dispatch is closed.

### E3 — generator-component attribution

**Implemented, awaiting measurement:** the same supplied-input corpus now
times five test-only stages independently: mask/popcount and class-size setup,
rank decomposition, rank-to-orbital bit selection, parity/determinant
materialization, and the recomposed no-RNG pipeline. The recomposed pipeline
must produce the same final-excitation digest as the precomputed staged path.

```sh
PLANCK_FCIQMC_COMPONENT_BENCHMARK=1 build/planck-fciqmc-walkers
```

`PLANCK_FCIQMC_COMPONENT_BENCHMARK_DRAWS` and
`PLANCK_FCIQMC_COMPONENT_BENCHMARK_ROUNDS` override the defaults. This is an
attribution measurement, not a production candidate. It determines whether to
investigate a local generator change or to stop generator work and move to
sorting or the shared Hamiltonian.

The following table describes the decision after E3:

| observed limit | permitted next investigation | required isolation |
|---|---|---|
| setup, bit selection, or parity | local generator candidate | literal sampler oracle and full fixed-seed trajectory gate |
| orbital updates or parity | local generator work, not dispatch rewrite | literal excitation oracle |
| `SpawnAccumulator::finalize()` / sorting | local folding or allocation prototype | ordered-fold raw-bit oracle and memory accounting |
| `slater_condon_element` | separate shared-Hamiltonian project | CI, CASSCF, open-shell, and FCIQMC validation |
| residual pairwise merge | larger-population merge study | same tagged tree and explicit temporary-memory bound |

Descriptor caching is not authorized without a physical-population prototype
that reports retained bytes and lookup cost. The reference estimator is not a
candidate unless the ledger shows it is material.

## Out of scope

- Replica trajectories improve time-to-error-bar, not a single-chain step
  rate, and must be compared at equal total core counts.
- MPI and GPU work require an explicit CPU or memory target beyond this
  measured envelope.
- Active-space/frozen-core FCIQMC requires core-energy, orbital-selection, and
  small-CASCI equivalence validation; it is not a propagation micro-optimization.
