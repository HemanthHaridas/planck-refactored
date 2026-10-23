# FCIQMC First Threading Pass — Historical Record

This document records the first OpenMP threading pass. It has been superseded
by the production parallel architecture in
[FCIQMC_PARALLELISM.md](FCIQMC_PARALLELISM.md). Do not use its timing numbers
or its map-based merge description to reason about current performance.

## What the first pass established

The initial implementation showed that FCIQMC could use OpenMP without losing
fixed-seed thread-count invariance. It established the principle that remains
in force today:

- partition work by parent determinant, not by child determinant or thread;
- make the partition count independent of OpenMP thread count;
- process the partitions with a static schedule; and
- reduce their results in a defined order.

That pass also exposed the limitation of leaving a map-backed, serial merge at
the end of the step. Its limited speedup was not an OpenMP failure; it was a
representation bottleneck.

## What replaced it

The current path strengthens the original design:

| First pass | Current production path |
|---|---|
| parent bins | 64 fixed parent bins |
| bin-local map-like accumulation | sorted/folded `SpawnAccumulator` output |
| traversal-sensitive random usage risk | `RandomSource::derive_parent(det)` |
| map-backed next population | sorted `CanonicalWalkerPopulation` |
| fixed serial reduction | tagged pairwise merge preserving equal-key bin order |

The tagged merge has a heap-merge oracle and all mechanical changes are gated
by raw trajectory equality, not merely matching printed energies. See
`FCIQMC_PARALLELISM.md` for these invariants and
`FCIQMC_SCALING_INVESTIGATION.md` for the measurement protocol and the latest
results.

## Validation rule that survives the rewrite

Threading is valid only when it retains the stochastic law and the committed
fixed-seed trajectory across supported thread counts. A faster calculation
whose parent visitation changes RNG consumption is a different simulation, not
a threading optimization. Any intentional trajectory change must be isolated,
documented, and separately validated statistically.
