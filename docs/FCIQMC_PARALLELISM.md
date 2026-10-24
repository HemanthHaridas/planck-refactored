# FCIQMC Parallel Propagation

This document describes the production parallel propagation contract. The
performance investigation, measurements, and the next candidate are kept in
[FCIQMC_SCALING_INVESTIGATION.md](FCIQMC_SCALING_INVESTIGATION.md). The earlier
serial and first-threading experiments are historical context only; they are
not the current implementation.

## Purpose and status

`correlation fciqmc` propagates one stochastic CI trajectory in parallel while
preserving its fixed-seed trajectory bit-for-bit across supported OpenMP thread
counts. This is a correctness requirement, not a benchmark convenience.

The current production representation is landed and validated:

- parents are assigned to 64 fixed bins by determinant hash;
- each parent receives a deterministic parent-keyed random stream;
- a bin has a private accumulator and is the unit of OpenMP work;
- the next walker population is canonical: determinant-sorted contiguous
  storage, not a hash map;
- finalized bin streams are joined by a tagged pairwise merge that retains the
  ascending-bin floating-point association for every equal determinant.

On the representative HF/6-31G, 50k-walker fixture, the canonical pairwise
path reduced full wall time by 16--29% versus the former map population. The
subsequent E1 rank-select change is accepted at four threads: median wall time
is 182.431 s versus 203.221 s for the pairwise baseline, a further 10.2%
reduction. Four threads are the best measured setting for this fixture; this
is advice, not a runtime limit.

## Execution model

For one propagation step, let `D` be the current canonical population and
`B = 64` the fixed bin count.

1. Assign each parent determinant `d` to `hash(d) mod B`.
2. Derive its `RandomSource` from `d`, rather than consuming a shared
   bin-order stream. A parent's random decisions are therefore independent of
   container layout and visitation order.
3. Process bins in a static OpenMP loop. A bin appends its spawn/death changes
   into its own `SpawnAccumulator`, then sorts and folds them by determinant.
4. Merge the finalized streams in a fixed tagged pairwise tree. The tag is the
   original bin number, so contributions to equal determinants are summed in
   the same ascending-bin order as the retained heap-merge oracle.
5. Install the sorted result as `CanonicalWalkerPopulation`, compact exact
   zeroes, calculate the L1 population, update shift control, and sample the
   projected estimator.

The important distinction is that bins are a deterministic partition of
**parents**, not a thread-local or child-hash partition. Threads may execute
independent bins in a different physical order without changing a parent RNG
stream or a final addition order.

## Code map

| Component | Responsibility |
|---|---|
| `src/post_hf/ci/fciqmc.{h,cpp}` | Canonical population, parent RNG derivation, propagation, excitation generator |
| `src/post_hf/ci/spawn_accumulator.h` | Local sort/fold and tagged pairwise canonical merge |
| `src/post_hf/fciqmc_driver.cpp` | Molecular driver and profile output |
| `tests/fciqmc_walkers.cpp` | Raw trajectory, parent-order, canonical-owner, and sampler-oracle gates |
| `tests/fciqmc_accumulator.cpp` | Accumulator and merge-oracle gates |
| `tests/run_fciqmc_scaling_profile.sh` | Strong-scaling ledger runner |
| `tests/run_fciqmc_cpu_profile.sh` | macOS CPU sampling runner |

## Invariants

### Parent streams, not traversal streams

Changing an `unordered_map` bucket layout or OpenMP schedule must not decide
which parent consumes which random number. `RandomSource::derive_parent(d)` is
the boundary that enforces this. Any proposal that restores a shared RNG stream
needs a new proof of parent-order independence before it can be used.

### Canonical reduction order

Floating-point addition is not associative. The pairwise merger must retain
the previous canonical order for equal keys, not merely produce a close
number. Its tagged merge is checked against the heap implementation at raw-bit
level. The heap merge remains a test oracle; it is not the production path.

### Thread-count invariance

For a mechanical optimization, the walker population, shift, estimators, and
caller RNG state must be bitwise unchanged at one and four threads (and at all
supported thread counts in the scaling gate). Reproducibility across a fresh
repeat is necessary but insufficient: it cannot reveal a schedule-dependent
trajectory.

### Stochastic correctness remains separate

Threading cannot weaken sampler validation. The support, `p_gen`, frequency,
exact-FCI statistical, initiator, and population-control tests remain required
regardless of whether a candidate appears to preserve a fixed seed.

## Validation and measurement

Run the invariant executables after a propagation or sampler change:

```sh
build/planck-fciqmc-walkers
build/planck-fciqmc-accumulator
```

Then run the N2 S5 fixed-seed pair and the H2/N2 statistical regression cases.
For a throughput result, use the HF/6-31G 50k fixture with
`tests/run_fciqmc_scaling_profile.sh`; retain complete logs and compare phase
records, not only terminal wall time. A CPU `sample` identifies a candidate;
it does not establish a speedup. At least three alternating measurements are
required for a performance claim.

## What remains

E1 removed the selected-rank cost in `draw_excitation`. The post-E1 profile
now makes the serial tagged pairwise merge the four-thread limit: it costs
39.399 s in sampling while worker time is 52.275 s. The next scoped candidate
is therefore a level-parallel realization of the same pairwise tree, with the
sequential tree as its raw-bit oracle. Class-dispatch, descriptor-cache, and
shared-Hamiltonian changes remain deferred until direct evidence supports
them.

Replica throughput and active-space/frozen-core FCIQMC are capability tracks,
not substitutions for this deterministic single-trajectory scaling contract.
