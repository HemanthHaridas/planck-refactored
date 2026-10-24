# FCIQMC Serial Performance — Historical Record

This file preserves the conclusion of the original serial-performance work.
It is not a description of the current implementation: FCIQMC propagation is
now parallel, owns a canonical sorted population, and uses a pairwise merge.
Read [FCIQMC_PARALLELISM.md](FCIQMC_PARALLELISM.md) for the current execution
contract and [FCIQMC_SCALING_INVESTIGATION.md](FCIQMC_SCALING_INVESTIGATION.md)
for measured performance and remaining work.

## Retained conclusion

Before OpenMP propagation was introduced, three mechanical changes reduced the
old N2/STO-3G serial fixture from 71.63 s to 13.74 s (5.21x) while retaining a
bitwise trajectory. The lesson remains valid: remove allocation, repeated
bookkeeping, and avoidable generator work before attempting parallelism.

Those experiments established three continuing rules:

1. A profile share is a candidate-selection signal, not a speedup estimate.
   Removing a cost can also remove cache pressure and bookkeeping attributed to
   other frames.
2. A fixed-seed mechanical optimization must prove the complete trajectory,
   not only terminal energy. The walker population and random-stream state are
   part of the result.
3. A local microbenchmark does not justify production complexity. The current
   acceptance criterion remains a repeatable material reduction on the
   representative molecular fixture without a statistical or determinism
   regression.

## Current replacement

The serial `WalkerPopulation` map is no longer the next-step owner. Production
uses `CanonicalWalkerPopulation`, parent-keyed random streams, fixed parent
bins, and a tagged pairwise merge. The old map timing and its pre-OpenMP
scalability limits must therefore not be compared directly with current
measurements.

For a new serial or worker-local candidate, use the present validation ladder:

```sh
build/planck-fciqmc-walkers
build/planck-fciqmc-accumulator
```

Then run the fixed-seed and statistical regressions before claiming a
throughput result. The active candidate process is defined in the scaling
scope, rather than this historical note.
