# FCIQMC scaling investigation

**R3 reached Cr₂ CAS(12,18) within the M3 pilot budget.** At 25,000 walkers, two of three seeds still switched projection reference. One projected estimate exceeded an independent variational upper bound by 0.690 Eh. The nested CAS(12,10) control measures how much the estimated energy varies across seeds after the exact projection comes within 1 mEh of the ground state. [Matched population runs](#r3-long-runs-at-25000-walkers) · [Late sampling results](#r3-sampling-length-and-seed-spread-in-cas1210)

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

## Small-population merge results

Use four threads for the cached spherical Be fixture on this M3. At eight threads, running the last few merge levels sequentially reduced merge time on Be. The production merge now executes levels with four or fewer pairs sequentially when there are at most 4,096 tagged entries. For the 64-bin tree these are the levels with four pairs, two pairs and one pair. Each level keeps its pair slots and addition order. The Be trajectories peaked at 1,812 tagged entries and HF 50k peaked at 50,522. These two fixtures establish the effect of the chosen cutoff over their measured populations.

| Fixture | Threads | Previous wall (s) | New wall (s) | Paired wall reduction | Sampling merge (s) |
|---|---:|---:|---:|---:|---:|
| Be | 1 | 11.21 | 11.16 | 0.5% | 0.88 → 0.88 |
| Be | 2 | 9.44 | 8.53 | 9.6% | 2.02 → 1.53 |
| Be | 4 | 9.80 | 8.08 | 17.5% | 3.22 → 2.16 |
| Be | 8 | 13.81 | 9.84 | 28.9% | 5.22 → 2.90 |
| HF 50k | 8 | 120.06 | 119.70 | 1.3% | 26.70 → 26.72 |

Each row reports three full-length pairs on the same build configuration and host, in A/B, B/A, A/B order. At eight threads on Be the three wall-time reductions were 28.9%, 23.9% and 29.0%. Median worker time stayed at 2.46 s while sampling merge time fell from 5.22 to 2.90 s. With profiling disabled, three more pairs gave a 27.9% median paired reduction and identical printed estimators and reference statistics. The HF 50k control passed the 5% regression limit. All 30 profiled runs reproduced the archived Hamiltonians, trajectories, work counters, cache sizes and vector capacities. [Measurements and run settings](../tests/benchmarks/fciqmc/small_tree_merge_m3_r2.json)

## R3 fixed-orbital Cr2 pilot

Cr₂ at 1.68 Å fits the 600 s and 1 GiB per-run pilot budget on the M3. The earlier all-MO path stopped after RHF because the Cartesian STO-3G basis has 38 orbitals and the packed CI representation admits 31. With 18 frozen doubly occupied orbitals and two excluded virtuals, only the 18 active orbitals enter the bitstrings. Six alpha and six beta electrons give 344,622,096 possible determinants. The pilot uses the contiguous RHF orbitals in their returned order. Their Cr 3d/4s and 4p character is quantified below. [Cr₂ input](../tests/inputs/exploratory/fciqmc/cr2_cas12_18_pilot.hfinp)

Both runs used four threads, 10,000 target walkers, timestep 0.0005, initiator threshold 3, seed 20250901 and 2,000 sampling steps. Only the equilibration length changed. The table gives process wall time and peak RSS including SCF, and total energies in Eh with the reported blocked standard errors.

| Equilibration steps | Wall (s) | Peak RSS (MiB) | Shift (Eh) | Projected (Eh) | Usable projected samples |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 8.56 | 84.2 | −2067.269 ± 0.158 | −2063.837 ± 0.026 | 1,994 / 2,000 |
| 10,000 | 48.65 | 109.8 | −2068.514 ± 0.139 | −2062.651 ± 0.053 | 803 / 2,000 |

Both trajectories re-anchored the projection after the seeded determinant lost weight. The estimator gap grew from 3.43 to 5.86 Eh. These runs establish resource feasibility. Neither provides a converged energy at this geometry. To reproduce them, run the scaling runner on the linked input and change only `fciqmc_equilibration` to 10000 for the second run. [Profiles, timings and reference diagnostics](../tests/benchmarks/fciqmc/cr2_cas12_18_m3_r3.json)

**Hamiltonian checks.** On the short N₂ control, the previous binary, the new default path and explicit CAS(14,10) all gave Hamiltonian `3d0ae86188394164` and trajectory `af01f83434290758`. LiH/STO-3G CAS(2,4) tests both a frozen core and an excluded virtual. PySCF CASCI on the Planck all-MO FCIDUMP gave −7.863681638749 Eh on exactly the same orbitals. The FCIQMC projected energy was −7.8636881653 ± 0.0000172 Eh and the shift was −7.8636733178 ± 0.00101 Eh. To regenerate the exact reference, add `fcidump lih.FCIDUMP` to the fixture's SCF block, read it with `mf = pyscf.tools.fcidump.to_scf("lih.FCIDUMP")`, then run `pyscf.mcscf.CASCI(mf, 4, 2).kernel(numpy.eye(6))`. The identity coefficients retain Planck's MO order. [LiH regression input](../tests/inputs/regression/post_hf/lih_fciqmc_cas24_sto3g.hfinp)

## R3 walker population and spawn attempts

Increasing the target population from 10,000 to 50,000 at one spawn attempt widened the shift–projected gap for each of three seeds. Each population used seeds 20250901, 20250902 and 20250903 with the pilot's four threads, 2,000 equilibration steps and 2,000 sampling steps. The original 10,000-walker seed reproduced its archived trajectory. Across seeds, the projection re-anchored in two of the three runs at each population.

The second sweep fixed the population at 10,000 and changed the number of spawn attempts per occupied determinant. The profile counts confirm that each phase made exactly the requested number of draws per parent visit. Four draws kept the seeded reference for all three seeds, made every projected sample usable and reduced the estimator gap in each matched run. Sixteen draws also kept the reference throughout sampling. With four draws, the shifts ranged from −2065.564 to −2065.528 Eh and the projected energies from −2064.555 to −2064.515 Eh. All 15 gaps exceeded the driver's warning threshold of five times the larger reported standard error.

| Target walkers | Draws per parent | Gap (Eh) | Usable samples / 2,000 | Mean reference weight | Wall (s) | Max RSS (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 1 | 2.00–3.73 | 1,899–2,000 | 15.5–25.1 | 7.5–8.4 | 84.3 |
| 25,000 | 1 | 2.29–4.17 | 1,236–2,000 | 22.9–40.1 | 15.7–17.6 | 124.9 |
| 50,000 | 1 | 2.47–4.43 | 1,236–2,000 | 32.2–56.9 | 30.3–33.3 | 209.5 |
| 10,000 | 4 | 1.00–1.01 | 2,000 | 90.5–133.5 | 14.6–15.0 | 76.1 |
| 10,000 | 16 | 0.89–0.99 | 2,000 | 177.3–183.7 | 39.2–40.4 | 73.7 |

Ranges span three independent seeds at each setting. The gap is the absolute difference between the shift and projected total energies in one run. The driver measures mean absolute reference weight over accepted projected samples. All runs fit the 600 s and 1 GiB budget. The 25,000-walker run at seed 20250903 briefly shared the host with the orbital checkpoint calculation. To reproduce a row, change only `fciqmc_walkers`, `fciqmc_spawn_attempts` and `fciqmc_seed` in the Cr₂ pilot input and use the scaling runner at four threads. [Per-seed estimators, uncertainties, reference determinants and work counters](../tests/benchmarks/fciqmc/cr2_population_spawn_m3_r3.json)

**Orbital character at 1.68 Å.** A checkpoint run reproduced Hamiltonian `3ffac35b4805c0ec`. In its AO overlap metric, the active MOs retained 11.9813 of the 12 nominal Cr 3d/4s valence directions by summed squared overlap (99.84%). The least well represented valence direction had squared overlap 0.9916. The active block also retained 5.9789 of the six 4p directions. The two excluded MOs carried 1.9841 of the two Cartesian d-shell trace directions. The projection orthonormalizes atom-centered shells in the AO metric, starting with the core (1s, 2s, 2p, 3s, 3p), followed by the valence (five traceless 3d functions and 4s per Cr), 4p and finally the two Cartesian traces. The archived weights sum squared overlaps with the frozen, active and excluded RHF columns. [Orbital projection weights](../tests/benchmarks/fciqmc/cr2_population_spawn_m3_r3.json)

## R3 equilibration at four spawn attempts

Extending equilibration from 2,000 to 10,000 steps narrowed the estimator gap on one seed and widened it on two. Each run used 10,000 target walkers, four attempts per occupied determinant, timestep 0.0005 and 2,000 sampling steps. The sampling window therefore moved from imaginary time 1–2 to 5–6 in inverse hartrees. The three short controls reproduced the archived Hamiltonian and trajectories with the current executable.

| Seed | Shift (Eh), short → long | Projected (Eh), short → long | Gap (Eh), short → long | Mean reference weight, short → long | Usable long samples / 2,000 | Long-run reference |
|---|---:|---:|---:|---:|---:|---|
| 20250901 | −2065.550 → −2065.342 | −2064.549 → −2064.660 | 1.002 → 0.682 | 133.52 → 11.99 | 2,000 | seeded |
| 20250902 | −2065.528 → −2065.537 | −2064.515 → −2064.241 | 1.012 → 1.296 | 123.17 → 6.47 | 1,292 | changed |
| 20250903 | −2065.564 → −2065.755 | −2064.555 → −2064.473 | 1.009 → 1.282 | 90.50 → 5.62 | 1,393 | changed |

All three short runs used the seeded reference on every sampling step. In the longer runs, the driver chose the most populated determinant after equilibration and kept it fixed during sampling. Two seeds switched determinants there. Mean reference weight fell for every seed, even when measured only on accepted samples. All three long-run gaps exceeded five times the larger reported standard error. The long runs took 64.4–68.7 s and peaked at 78.5 MiB. [Per-seed errors, reference determinants and run settings](../tests/benchmarks/fciqmc/cr2_equilibration_m3_r3.json)

## R3 long runs at 25,000 walkers

Increasing the target from 10,000 to 25,000 walkers narrowed the shift–projected gap for seed 20250902 and widened it for the other two. Each seed made the same decision about keeping or changing the seeded reference at equilibration. The replacement determinants themselves differed between populations. Only the target walker count changed within each pair. Both populations used 10,000 equilibration and 2,000 sampling steps, four draws per occupied determinant, timestep 0.0005 and initiator threshold 3.

| Seed | Shift (Eh), 10k → 25k | Projected (Eh), 10k → 25k | Gap (Eh), 10k → 25k | Usable samples / 2,000 | Mean accepted reference weight | 25k reference |
|---|---:|---:|---:|---:|---:|---|
| 20250901 | −2065.342 → −2065.509 | −2064.660 → −2064.693 | 0.682 → 0.816 | 2,000 → 2,000 | 12.0 → 13.3 | seeded |
| 20250902 | −2065.537 → −2065.688 | −2064.241 → −2064.788 | 1.296 → 0.899 | 1,292 → 2,000 | 6.5 → 5.9 | changed |
| 20250903 | −2065.755 → −2065.937 | −2064.473 → −2063.804 | 1.282 → 2.132 | 1,393 → 1,284 | 5.6 → 4.9 | changed |

All three 25k runs triggered the driver's estimator-gap warning. Wall time rose by 2.33–2.37× to 152–163 s and total parent visits rose by 2.31–2.36×. Peak RSS was 96.5–114.7 MiB. During sampling, workers accounted for 87–88% of wall time and the initiator rule rejected about 98% of spawn attempts. Both populations used the same solver binary and four-thread configuration. [Per-seed estimates, errors, work counts and settings](../tests/benchmarks/fciqmc/cr2_long_25k_m3_r3.json)

### Independent energy check

Freezing the same 18 core orbitals and keeping the first ten of the 18 active RHF orbitals gives a 44,100-determinant CAS(12,10). PySCF CASCI on the all-MO Planck FCIDUMP gave **−2064.494324 Eh**. The normalized CI vector reproduced that energy by direct expectation. Embedding this vector in CAS(12,18) assigns zero amplitude to determinants that occupy the eight additional orbitals, so its energy bounds the full-space ground-state energy from above. Seed 20250903's 25k projected estimate was **0.690 Eh above that bound**, with a reported standard error of 0.102 Eh. The bound gives no comparable accuracy certificate for estimates below it.

We also ran the stochastic solver directly in CAS(12,10) at the same 25k settings and seeds. Here the exact reference lets us measure signed error for each estimator. All three runs retained the seeded reference. Halving the timestep doubled the equilibration and sampling step counts to 20,000 and 4,000, and doubled the shift-update interval to ten steps. This preserves both the imaginary-time windows and the time between shift updates. All six controls kept every projected sample.

| Timestep | Shift minus CASCI (mEh), three seeds | Projected minus CASCI (mEh), three seeds | Usable samples per seed |
|---|---:|---:|---:|
| 0.0005 | −86.7 to −63.2 | +16.1 to +29.9 | 2,000 / 2,000 |
| 0.00025 | −75.9 to −65.3 | +24.0 to +35.2 | 4,000 / 4,000 |

To reproduce the deterministic control, export the all-MO FCIDUMP from the Cr₂ pilot. Read it with `pyscf.tools.fcidump.to_scf` and run `mcscf.CASCI(mf, 10, (6, 6)).kernel(numpy.eye(38))` in the input MO order. For the stochastic controls change only `nactorb` to 10 in the respective 25k input and use the timestep and step counts above. [FCIDUMP checksum and individual control results](../tests/benchmarks/fciqmc/cr2_long_25k_m3_r3.json)

## R3 initiator threshold and projection time in CAS(12,10)

At imaginary time 5–6, disabling the initiator rule moved the projected energy by only +0.17 to +0.37 mEh across matched seeds. It remained 16–30 mEh above the exact energy of −2064.494323848 Eh. The shift moved upward by 2.3–9.9 mEh and remained below CASCI. The threshold-off runs recorded zero initiator rejections in both phases. They used the same Hamiltonian, solver binary, 25,000 target walkers, timestep 0.0005 and four draws as the threshold-3 controls. Each table entry gives shift / projected energy as signed deviations from CASCI in mEh.

| Seed | Threshold 3, τ 5–6 | Threshold 0, τ 5–6 | Threshold 0, τ 55–56 |
|---|---:|---:|---:|
| 20250901 | −86.7 / +18.1 | −84.3 / +18.4 | +10.8 / +11.0 |
| 20250902 | −86.2 / +29.9 | −81.6 / +30.1 | +12.7 / +0.6 |
| 20250903 | −63.2 / +16.1 | −53.3 / +16.3 | +4.0 / +9.2 |

**The initial determinant is still relaxing at τ = 5.** Applying the continuous imaginary-time exponential of the 44,100-determinant Hamiltonian to the seeded determinant gave projected energies 27.345 and 22.999 mEh above CASCI at τ = 5 and 6. The mean growth energy over that window, computed from the change in the log of the exact state's L1 norm, was 59.239 mEh below CASCI. Thus even the deterministic projection and growth energy lie on opposite sides of the ground-state energy in the early window. The stochastic shift also depends on population feedback. The reference script checks each projected energy against the Rayleigh quotient of the state at half the projection time. [Finite-time reference calculation](../tests/pyscf/cr2_fciqmc_projection.py)

By τ = 55 and 56 the exact projected deviations were +0.875 and +0.830 mEh, and the mean growth deviation was −0.754 mEh. To sample that window, the three threshold-off runs increased equilibration from 10,000 to 110,000 steps and retained 2,000 sampling steps. All six threshold-off runs kept the seeded reference and every projected sample. In the early window disabling the rule reduced the mean accepted reference weight on all three seeds, from 2,415–2,547 to 2,352–2,491 walkers. In the later window it was 1,119–1,284 walkers. The later projected energies were closer to CASCI on all three seeds, although two remained more than five reported standard errors above it. The long runs took 84–88 s and all six runs fit the 600 s and 1 GiB per-run budget. A repeat of the deterministic check briefly shared the host with the second late run. [Per-seed energies, uncertainties, work counts and exact window references](../tests/benchmarks/fciqmc/cr2_cas12_10_initiator_m3_r3.json)

## R3 sampling length and seed spread in CAS(12,10)

The three original seeds repeated the same 110,000-step equilibration, then extended sampling from 2,000 to 10,000 steps (τ = 55–56 to 55–60). The short window is the first 2,000 steps of each matched long run. Only the sampling length changed. Each run used 25,000 walkers, four spawn attempts per occupied determinant, timestep 0.0005 and initiator threshold zero. The table gives signed errors against CASCI and the driver's blocked standard errors in mEh. Reference weights are mean absolute weights over accepted samples.

| Seed | Shift error, short → long | Projected error, short → long | Reference weight, short → long |
|---|---:|---:|---:|
| 20250901 | +10.76 ± 15.10 → −2.27 ± 7.92 | +11.01 ± 2.14 → +3.16 ± 2.69 | 1,204 → 1,171 |
| 20250902 | +12.67 ± 14.50 → −6.21 ± 7.38 | +0.64 ± 0.49 → +6.30 ± 2.27 | 1,284 → 1,252 |
| 20250903 | +3.99 ± 21.10 → +3.82 ± 8.05 | +9.18 ± 1.65 → −1.89 ± 4.41 | 1,119 → 1,093 |

For these three seeds, the mean projected error fell from +6.94 to +2.52 mEh, although the second seed moved farther from CASCI. Exact continuous propagation from the seeded determinant gave projected errors of +0.875 mEh at τ = 55 and +0.670 mEh at τ = 60. The mean exact L1 growth energy over that window was 0.680 mEh below CASCI.

Five fresh seeds (20250904–20250908) tested the same 10,000-step window. Each row below gives the equally weighted mean signed error, the sample standard deviation across seeds (SD), and the root mean square of the individual runs' reported blocked standard errors (RMS SE). All values are in mEh.

| Seeds and sampling length | Mean shift error | Shift SD / RMS SE | Mean projected error | Projected SD / RMS SE |
|---|---:|---:|---:|---:|
| Original three, 2,000 steps | +9.14 | 4.56 / 17.16 | +6.94 | 5.53 / 1.59 |
| Original three, 10,000 steps | −1.55 | 5.06 / 7.79 | +2.52 | 4.13 / 3.26 |
| Five fresh, 10,000 steps | +0.40 | 7.75 / 7.51 | −6.02 | 9.82 / 2.48 |
| All eight, 10,000 steps | −0.33 | 6.53 / 7.62 | −2.82 | 8.92 / 2.80 |

Four of the five fresh projected estimates missed CASCI by more than twice their reported SE, with signed errors ranging from −14.86 to +7.33 mEh. All five shifts fell within two reported SEs. The driver reports a ratio of summed numerator and denominator for the projected energy and computes its error bar by blocking the per-step ratios. On these seeds the reported projected errors understated the observed variation by about a factor of four. The five-seed comparison alone cannot distinguish slow fluctuations from a persistent energy bias.

All eight long runs retained the seeded reference and accepted every projected sample. Mean absolute reference weights ranged from 1,093 to 1,252 walkers. The runs took 90.7–95.3 s each and peaked at 57.6 MiB on the M3. Equilibration accounted for about 90% of total time. Sampling cost 7.9–8.7 s per 10,000 steps, leaving room to lengthen the window within the 600 s budget. To reproduce the runs, apply the settings above to the Cr₂ pilot input and run the scaling profiler at four threads with seeds 20250901–20250908. [Per-seed energies, reference weights, timings and exact controls](../tests/benchmarks/fciqmc/cr2_cas12_10_sampling_m3_r3.json)

## Next steps

Extend the five fresh CAS(12,10) seeds to 50,000 sampling steps (τ = 55–80) with the same 110,000-step equilibration, population, timestep and threshold. Compare the added 40,000 steps separately from the shared 10,000-step prefix. Check whether the projected spread across seeds falls and whether the reported single-run errors track it. CAS(12,18) still needs its own accuracy check.

The E2 dispatch experiment is closed. E4 remains test-only pending evidence that bit selection merits a new paired trial. The detailed R0 handoff results, R1 reproduction commands and earlier optimization decisions are preserved in the version of this document at commit `7def132a` (`git show 7def132a:docs/FCIQMC_SCALING_INVESTIGATION.md`).
