# FCIQMC scaling investigation

**R2 passed the adoption gate on the Apple M3.** On water, a serial cache lookup took 16.27 ns per parent and distributing a parent to its work bin took 1.85 ns. The driver now skips the serial lookup for determinants occupied on consecutive steps. R0 qualified the HF-to-FCIQMC Hamiltonian handoff. R1 established the timing, memory and correctness baselines on an Apple M3.

## Why start with water

These are full trajectories at 50k target weight, four threads and seed 20250901. Each row is one diagnostic run with 20,000 equilibration and 30,000 sampling steps. Prefill includes the serial diagonal lookup and parent binning.

| R1 measurement | Water/6-31G | Hydrogen fluoride/6-31G |
|---|---:|---:|
| Mean occupied parents per sampling step | 178,914 | 48,485 |
| Sampling wall time | 557.82 s | 111.14 s |
| Prefill time | 188.54 s (33.8%) | 22.71 s (20.4%) |
| Prefill per parent visit | 35.13 ns | 15.61 ns |
| Merge per parent visit | 16.56 ns | 18.05 ns |
| Peak process RSS | 295.9 MiB | 46.6 MiB |

Water spent more than twice as long per parent in prefill as hydrogen fluoride. The water pilot's CPU sample also placed the diagonal callback second among application leaf frames. The full water trajectory's projected energy agreed with independent PySCF FCI to 0.38 reported standard errors. [R1 control measurements](../tests/benchmarks/fciqmc/controls_m3_r1.json)

## R2 lookup and binning measurements

The baseline driver prefilled the run's diagonal cache while binning every occupied parent. The worker then queried the same cache for its death or cloning contribution. On the 2,000-equilibration/2,000-sampling pilots, we split the serial traversal on every 32nd sampling step. The next step timed the usual fused traversal. A separate four-thread pass replayed the worker's cache hits in bin order and included its OpenMP launch and barrier.

| Pilot | Serial lookup | Binning | Worker lookup replay | Adjacent fused pass | Cache misses / sampled parent visits |
|---|---:|---:|---:|---:|---:|
| Water | 16.27 ns | 1.85 ns | 4.57 ns | 21.44 ns | 2,794 / 11,242,267 |
| Hydrogen fluoride | 12.33 ns | 1.87 ns | 4.09 ns | 15.32 ns | 503 / 2,919,863 |

Times are wall nanoseconds per parent visit. Splitting the loop changes locality, and replaying the worker lookup warms the cache. The adjacent fused pass measures the actual combined cost. Both probe runs reproduced the baseline Hamiltonian, result and work counts. Single pilot wall times with and without the probes were 45.80 and 51.67 s for water, 11.53 and 11.13 s for HF.

## Preparing newly occupied parents

The dominant serial cost was checking diagonals already in the cache. The driver walks the previous and current sorted canonical populations together. A surviving determinant already has a cached diagonal. A newly occupied determinant, including one that returns after an absence, goes through the existing serial lookup before workers start. All nonzero parents are binned in their original order with their current weight. Workers use their usual diagonal callback. [Driver prefill](../src/post_hf/fciqmc_driver.cpp) · [Parent preparation](../src/post_hf/ci/spawn_accumulator.h)

This uses the previous population as the record of which keys were prefilled. Its allocation is released before constructing the next population and merge buffers. The test exercises initial, surviving and returning determinants, including equal alpha strings with different beta strings. [Walker tests](../tests/fciqmc_walkers.cpp)

On the water pilot, sampling prefill fell from 8.28 s to 0.94 s and solver wall time from 51.67 s to 37.78 s. On HF, prefill fell from 1.37 s to 0.22 s and wall time from 11.13 s to 9.62 s. Both trajectories reproduced the Hamiltonian, result, stochastic work counters and final diagonal-cache size. The three unit/statistical gates and nine molecular regressions passed, including the N₂ thread-identity and spherical, open-shell and ECP cases.

## R2 acceptance decision

Adopt the change for canonical FCIQMC propagation. Each fixture ran for three full-length, fixed-seed pairs in A/B, B/A, A/B order. The columns with arrows give independent baseline and candidate medians. The three percentages use the wall times from each matched pair. Sampling prefill and worker times are components of the sampling phase.

| Fixture, threads | Wall (s) | Paired reductions | Sampling prefill (s) | Sampling worker (s) | Peak RSS (MiB) |
|---|---:|---|---:|---:|---:|
| Water, 4 | 789.63 → 584.47 | 25.5%, 22.8%, 30.9% | 169.91 → 12.72 | 249.61 → 237.06 | 329.6 → 327.6 |
| HF, 4 | 164.16 → 135.62 | 17.4%, 14.6%, 17.6% | 19.57 → 3.16 | 51.35 → 51.41 | 45.2 → 48.7 |
| Be, 2 | 10.06 → 9.49 | 6.0%, 4.5%, 5.7% | 0.38 → 0.12 | 3.22 → 3.12 | 15.1 → 15.2 |

Every water pair exceeded the 10% end-to-end threshold. The longest candidate run finished sooner than the shortest baseline run. HF and Be passed the 5% control-regression gate. Both builds produced identical Hamiltonian and result fingerprints, integer work counters, cache sizes and recorded vector capacities within each fixture. Process RSS varied across repetitions. The worker still queries the shared cache, and its time includes that lookup. [Individual runs and probe measurements](../tests/benchmarks/fciqmc/diagonal_prefill_m3_r2.json)

To repeat a pair, build the baseline at `7def132a` and the candidate with the same Release flags. Run each binary through `tests/run_fciqmc_scaling_profile.sh BINARY FIXTURE THREADS OUTPUT_DIR` with a separate output directory for each invocation. Set `OMP_DYNAMIC=false` and the Accelerate, OpenBLAS and MKL thread limits to one. The archive gives the input paths, hashes and thread counts. [Scaling runner](../tests/run_fciqmc_scaling_profile.sh)

## Thread selection after R2

Use 8 OpenMP threads for the HF/6-31G 50k fixture on this M3. The optimized solver's median wall time was 122.92 s, 3.23× faster than one thread and 11.7% shorter than the next fastest setting (4 threads). The fixed-seed sweep used three full trajectories at each setting in 1/2/4/8 order. All 12 reproduced Hamiltonian `f032498d77f973b5`, result `ea152a6472c1789b`, the work counters in both phases, and 105,050 final diagonal-cache entries.

| Threads | R1 median wall (s) | R2 median wall (s) | R2 wall range (s) | R2 speedup | Worker (s) | Merge (s) | Prefill (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 424.83 | 396.49 | 390.91–397.01 | 1.00× | 196.18 | 39.81 | 3.22 |
| 2 | 250.39 | 217.41 | 216.35–220.60 | 1.82× | 97.60 | 28.48 | 3.21 |
| 4 | 169.38 | 139.24 | 134.77–139.34 | 2.85× | 53.29 | 25.19 | 3.22 |
| 8 | 151.51 | 122.92 | 120.12–126.20 | 3.23× | 41.90 | 27.02 | 3.25 |

From four to eight threads, median sampling worker time fell by 11.39 s and merge time rose by 1.83 s. Prefill stayed near 3.2 s at every thread count. Worker, merge and prefill are median sampling times. Speedup uses the optimized one-thread run as its baseline. The R1 wall times came from the previous sweep on the same host. The paired R2 measurements above establish the effect of changing the implementation at four threads. [Optimized sweep](../tests/benchmarks/fciqmc/hf50k_m3_r2.json) · [R1 sweep](../tests/benchmarks/fciqmc/hf50k_m3_r1.json) · [Thread sweep runner](../tests/run_fciqmc_e1_profile_ladder.sh)

## Next steps

**Small populations at high thread counts.** Spherical Be was fastest at two threads. Going from two to four saved 0.73 s in the sampling worker and added 1.36 s to the merge. If this workload is the target, test a sequential merge for the last few tree levels as a separate R2 experiment. Preserve the pair order and check HF 50k as the larger control.

**A calculation exceeds all-MO capacity.** Start R3 once a specific molecule, active space and resource budget identify the admission limit. Scope a fixed-orbital adapter that supplies active one- and two-electron integrals, spin counts and the core plus nuclear energy offset. Verify all-MO equivalence with zero frozen orbitals and compare a small frozen-core case with deterministic CASCI at identical orbitals. [FCIQMC research scope](FCIQMC_RESEARCH_SCOPE.md)

The E2 dispatch experiment is closed. E4 remains test-only pending evidence that bit selection merits a new paired trial. The detailed R0 handoff results, R1 reproduction commands and earlier optimization decisions are preserved in the version of this document at commit `7def132a` (`git show 7def132a:docs/FCIQMC_SCALING_INVESTIGATION.md`).
