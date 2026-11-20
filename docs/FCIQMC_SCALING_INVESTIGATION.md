# FCIQMC scaling investigation

This document answers one question: can Planck's FCIQMC reach Cr₂ CAS(12,18), a space deterministic FCI cannot hold, and give an energy that can be trusted?

**Resources: yes. Energy: not yet established.** On the Apple M3, a 25,000-walker CAS(12,18) run fits the 600 s and 1 GiB per-run budget. The pre-fix projected energies exceeded a variational upper bound (−2064.896580 Eh) by 0.108–1.092 Eh. A November 17 code review found four defects in the propagator and estimators that could produce or hide that result, and all four are fixed (`ed9ea6a6`). On the CAS(12,10) exact control, the projected error bar now matches the spread across seeds, where it understated it about 4×, and the projected RMSE against CASCI halved. On the fixed code the CAS(12,18) projected energies still sit 0.165–0.231 Eh above the bound at τ = 5–6, now consistently across seeds. Extending imaginary time to τ = 55 at 25,000 walkers did not converge: the shift kept falling and the projection anchor wandered. In exact CASCI vectors the RHF determinant's weight collapses as orbitals are added (54%, 0.55% and 0.02% in 10, 12 and 13 orbitals). The evidence now points to too few walkers, together with a reference that carries almost none of the wavefunction.

[Next steps](#current-status-and-next-steps) · [Defects fixed](#defects-fixed-on-november-17) · [Validation](#validation-after-the-fixes) · [Performance](#performance-on-the-m3) · [Cr₂ target](#the-cr2-target) · [Pre-fix Cr₂ measurements](#pre-fix-cr2-measurements)

## Current status and next steps

1. **Done: the bound violation persists on the fixed code, so the defects did not cause it.** Seeds 20250901–20250903 ran with the pre-fix settings: 25,000 walkers, four spawn attempts, initiator threshold 3, timestep 0.0005, 10,000 equilibration and 2,000 sampling steps (τ = 5–6). The Hamiltonian ID (`3ffac35b4805c0ec`) matched the pre-fix runs.

   | Seed | Projected (Eh) | Above bound, fixed (Eh) | Above bound, pre-fix (Eh) | Mean reference weight, fixed / pre-fix | Shift (Eh) |
   |---|---:|---:|---:|---:|---:|
   | 20250901 | −2064.6660 ± 0.0149 | +0.231 | +0.203 | 57.9 / 13.3 | −2065.0686 ± 0.0231 |
   | 20250902 | −2064.6671 ± 0.0060 | +0.229 | +0.108 | 68.1 / 5.9 | −2064.9816 ± 0.0197 |
   | 20250903 | −2064.7319 ± 0.0081 | +0.165 | +1.092 | 42.5 / 4.9 | −2065.1978 ± 0.0243 |

   The seeds now agree within 0.07 Eh, where they spread over 0.98 Eh before. Every projected step was usable, no seed re-anchored and no parent was below one walker. All three projected errors were flagged unconverged, and the shift–projected gaps of 0.31–0.47 Eh triggered the driver's warning. The initiator rule still rejected 94.6–95.2% of sampling spawn attempts. Occupied parents averaged about 18,300 per sampling step. Runs took 67–88 s and peaked at 61–75 MiB. [Per-seed results](../tests/benchmarks/fciqmc/cr2_cas12_18_review_fixes_m3_r3.json)
2. **Separate the remaining causes on CAS(12,18), one variable at a time.**
   - a. Imaginary time: lengthen equilibration until the energy stops moving. In CAS(12,10), exact propagation was still 27 mEh above CASCI at τ = 5 and under 1 mEh at τ = 55.
   - b. Initiator bias: at a converged τ, compare thresholds 3, 1 and 0. Threshold 3 rejected about 98% of spawn attempts in the pre-fix runs.
   - c. Reference choice: compare the single-determinant projection with a fixed multideterminant trial on the same walker trajectories. This needs new code. Build the trial in Planck's determinant basis and check its Rayleigh quotient on the shared Hamiltonian. This step has moved ahead of a and b. The RHF determinant holds 0.55% of the exact CAS(12,12) wavefunction and 0.02% of CAS(12,13), where other closed-shell determinants hold 56% and 38% ([reference weight](#the-cr2-target)). The CAS(12,13) CASCI vector embeds exactly in CAS(12,18), so it is a ready trial and starting state.
   - **Step a result: at 25,000 walkers, longer imaginary time does not converge.** Seed 20250901 ran with the step-1 settings at four sampling windows:

     | τ | Projected minus bound (Eh) | Shift (Eh) | Mean reference weight | Projection anchor | Wall (s) |
     |---:|---:|---:|---:|---|---:|
     | 5 | +0.231 | −2065.069 | 57.9 | RHF `0x3f/0x3f` | 67 |
     | 15 | +0.637 | −2065.241 | 9.3 | `0x442d/0x043e` | 232 |
     | 30 | +0.250 | −2065.310 | 29.6 | `0x42f/0x42f` | 310 |
     | 55 | +1.200 | −2065.350 | 6.4 | `0xc1e/0x1127` | 418 |

     The shift fell at every window and ended 0.42 Eh below the 12-orbital SC-NEVPT2 estimate. The projection anchor wandered between determinants holding 9–30 walkers. At τ = 55, 746 of 2,000 steps had fewer than 3 walkers on the anchor. Only at τ = 30 did the anchor coincide with the dominant determinant of exact CAS(12,13). A falling shift and a population that does not settle on the dominant configurations both point to too few walkers. That is a problem with the walker distribution itself, which a better estimator cannot repair. [τ study](../tests/benchmarks/fciqmc/cr2_cas12_18_tau_m3_r3.json)
   - **Revised order.** First test sampling where the answer is known: exact CAS(12,13) (2.9 million determinants, −2064.746875 Eh, dominant determinant `0x42f/0x42f`) is the largest exact control available. Run FCIQMC there at 25k, 100k and 400k walkers, with τ ≈ 30 and initiator threshold 3. This shows whether the shift converges with walker count, how many walkers that takes, and whether the projected energy recovers once the population is right. Then repeat the walker scan in CAS(12,18), and add the multideterminant trial (c) only if the projected energy is still the weak point.

     The inputs are `tests/inputs/exploratory/fciqmc/cr2_cas12_13_walkers_{25k,100k,400k}.hfinp`: 60,000 equilibration steps (τ = 30), 4,000 sampling steps, four spawn attempts, timestep 0.0005, seed 20250901, profiling on. Run each with `BASIS_PATH=<repo>/basis-sets OMP_NUM_THREADS=<n> build/hartree-fock <input>`. Results do not depend on the thread count, so any machine and thread count is fine. From the CAS(12,18) timings, expect roughly 5 minutes at 25k on four M3 threads and roughly linear growth with walker count; that is too long for the 400k run on a laptop.
3. **Smaller items, independent of 1 and 2.**
   - The shift error bar is 2–6× larger than the spread across seeds on every fixture checked ([validation](#validation-after-the-fixes)). A probable cause is that population control makes the shift's long-time correlation negative, so the maximum over blocking levels overstates the error. Testing this needs a per-step series dump.
   - On CAS(12,10) both fixed estimators average about 4 mEh below CASCI, about two standard errors of the mean with five seeds. Ten more seeds over a 50,000-step window would show whether this is a population-control bias.
   - `fciqmc_sample_low_weight_parents` no longer has any effect and can be removed ([why](#validation-after-the-fixes)).
   - The N₂ regression's projected energy moved from 0.41σ to 1.28σ from exact. Both are within the gate; a few seeds would confirm it is noise.

The E2 dispatch experiment is closed. E4 remains test-only until there is evidence that bit selection merits a new paired trial. The detailed R0 handoff results, R1 reproduction commands and earlier optimization decisions are in the version of this document at commit `7def132a` (`git show 7def132a:docs/FCIQMC_SCALING_INVESTIGATION.md`).

## Defects fixed on November 17

A code review read the propagator, estimators and driver against the R3 Cr₂ measurements. It found four defects and one weak heuristic. Each item gives the defect, the evidence and the fix. [Propagator](../src/post_hf/ci/fciqmc.cpp) · [Population and blocking](../src/post_hf/ci/fciqmc.h) · [Driver](../src/post_hf/fciqmc_driver.cpp)

**1. Surviving parent weights were never rounded.** The death step writes `weight * (1 − dt (H_ii − S))` without rounding. Stochastic rounding applied only to spawned children, and the driver removed only weights at or below 1e-12. Real-weight FCIQMC rounds every weight below the occupation threshold to 0 or the threshold after annihilation. Without that step, a determinant spawned onto once kept a slowly shrinking fractional weight for thousands of steps, because the survival factor is close to 1 at dt = 0.0005. The pre-fix profiles show the consequences:
- CAS(12,12) held 94,048 occupied parents for a 25,000-walker target, which integer walkers cannot do.
- 93–96% of parent visits and 51–80% of the absolute weight were below one walker ([table](#spawn-work-by-parent-weight)).
- 99.988% of CAS(12,12) spawn attempts rounded to zero, since `dt·|H_ij|·|w| / p_gen` is small when |w| ≈ 0.01.
- The initiator rule treats any nonzero incoming weight as occupied, so stale sub-walker weights admitted spawns that an integer population would reject.

*Fix:* `CanonicalWalkerPopulation::compress_with_l1_norm` takes an occupation threshold and a rounding root. After each step, every weight with |w| below the spawn granularity becomes ±granularity with probability |w|/granularity and is removed otherwise. Each determinant draws from `root.derive_parent(det)`, using one raw draw from the run RNG per step as the root, so the result does not depend on thread count. At zero granularity nothing changes.

**2. The projected error bar described a different estimator.** The driver reports the ratio of sums `H_00 + Σ_t num_t / Σ_t c0_t`, but it computed the error by blocking the per-step ratios `num_t / c0_t`. The code comment claimed this overstated the uncertainty. On five fresh CAS(12,10) seeds at 10,000 steps, the projected spread across seeds was 9.82 mEh against an RMS reported error of 2.48 mEh ([pre-fix seed spread](#sampling-length-and-seed-spread-in-cas1210)).

*Fix:* the projected error is the blocked standard error of `y_t = (num_t − R c0_t) / mean(c0)`, with `R = Σ num / Σ c0`. This is the first-order (delta-method) error of the ratio, including the numerator–denominator covariance.

**3. Blocking took the maximum without checking for a plateau.** `blocked_standard_error` returns the largest standard error over the blocking levels. The header claimed this never underestimates. That holds only when the correlation time is much shorter than the series; the G2 gate tested AR(1) series of that kind. At small timesteps, a correlation time of order 1/(gap·dt) steps can be comparable to the sampling window. The blocking curve is then still rising at its last level, and the maximum is too small.

*Fix:* `blocking_analysis` returns the error and a `converged` flag, using the Lee et al. (PRE 83, 066706, 2011) criterion: some level with block length B satisfies `B³ > 2N (se_B / se_0)⁴`. `blocked_standard_error` returns the same value as before. When the criterion fails, the driver appends `[ERROR BAR UNCONVERGED]` to the energy line and prints a warning. Parsers that read `<energy> +/- <error>` are unaffected.

**4. The projected sums dropped weakly occupied steps.** A step entered neither sum unless |c₀| ≥ 3 walkers. That threshold had protected the per-step division of the older mean-of-ratios estimator. The ratio of sums divides once, so the threshold protected nothing. Instead it averaged only over steps where the reference was well occupied, and the numerator is correlated with c₀. In CAS(12,18), where the reference averaged 5–13 walkers, the extended seed 20250901 run kept only 3,483 of its last 8,000 steps.

*Fix:* every sampling step enters both sums. The 3-walker threshold now only decides whether the projected energy is reported at all: it is suppressed when fewer than a quarter of steps pass. The output line gives the number of weak steps.

**5. Re-anchoring used one step's largest weight.** At the end of equilibration the driver picked the determinant with the largest |w| on that single step. It switched whenever that determinant beat the seeded reference by any margin; the factor of 2 controlled only the warning. At 5–13 walkers per determinant this choice was mostly sampling noise. For the same seed the replacement determinant differed between populations, and runs that switched kept fewer usable projections (1,292 and 1,393 of 2,000).

*Fix:* the reference is chosen from signed weights summed over the last min(1,000, equilibration/4) equilibration steps. It moves only when the new determinant beats the seeded reference by more than 2×, the same margin as the drift warning.

**Two smaller fixes.** The driver comment saying the shift never touches the reference was corrected: the reference is also the starting population. A stale comment describing the population as a hash map was removed.

## Validation after the fixes

**Gates.** `planck-fciqmc-walkers`, `planck-fciqmc-accumulator` and `planck-statistical-gate` pass, as do all 13 FCIQMC regression cases. Two new unit tests cover the fixes. `test_occupation_rounding` checks that rounding is unbiased, produces only 0 or ±threshold, is reproducible from its root, and does nothing without a root. `test_blocking_reports_unconverged_plateau` checks that AR(1) with τ ≪ N converges and AR(1) with τ ≈ N does not. Run the cases with paired thread counts in one invocation, because each `_threads4` case compares against its `_threads1` partner. The short N₂ control printed identical estimators at 1, 2, 4 and 8 threads, and the raw-bit thread-identity case passed.

Every trajectory with nonzero granularity changed, which is every case at the default of 1.0. Archived result fingerprints from before `ed9ea6a6` no longer apply. `n2_fciqmc_s5_threads1` checks one exact trajectory and was re-pinned to −108.9703898222 Eh (shift) and −107.5753149721 Eh (projected).

**Regression energies against exact FCI.** These are single runs at the regression inputs as they stood when the fixes landed. The earlier N₂ deviation is from the [driver validation](FCIQMC_DRIVER_AND_VALIDATION.md).

| Case | Projected (Eh) | Deviation | Blocking |
|---|---:|---:|---|
| N₂/STO-3G | −107.6424563 ± 8.3e-3 | 1.28σ (was 0.41σ) | converged |
| H₂/STO-3G, 8,000 steps | −1.1376377 ± 1.8e-4 | 1.99σ | unconverged |
| LiH CAS(2,4), 30,000 steps | −7.8636844 ± 3.2e-5 | 0.09σ | unconverged |

**The H₂ and LiH unconverged flags are correct.** These spaces have only four and six determinants, so a smooth series could have tripped the plateau test without the error being wrong. A 20-seed study tested this: seeds 30000000–30000019, one thread, regression inputs otherwise unchanged. SD/RMS is the standard deviation of the 20 estimates divided by the RMS of their reported errors; a value of 1 means the error bars are honest. With 20 seeds it is uncertain by about ±16%.

| Case, sampling steps | Projected SD/RMS | Projected runs flagged | Within 2 SE of exact | Shift SD/RMS |
|---|---:|---:|---:|---:|
| H₂, 8,000 (old regression) | 1.31 | 19/20 | 16/20 | 0.35 |
| H₂, 32,000 | 1.20 | 10/20 | 19/20 | 0.25 |
| H₂, 128,000 (new regression) | 0.88 | 0/20 | 18/20 | 0.19 |
| LiH CAS(2,4), 30,000 (old regression) | 1.34 | 19/20 | 18/20 | 0.19 |
| LiH CAS(2,4), 120,000 | 1.12 | 1/20 | 19/20 | 0.16 |
| LiH CAS(2,4), 240,000 (new regression) | 0.93 | 0/20 | 20/20 | 0.12 |

At the old lengths the projected error understated the spread by about 30%, and the flag fired on nearly every run. At dt = 0.001 the slowest mode relaxes by about `1 − gap·dt` per step, so the correlation time is of order a thousand steps. Unlike N₂, these spaces have almost no fast sampling noise to dominate the blocking curve. The plateau criterion stays as it is. The two regression inputs now sample 128,000 and 240,000 steps, under a second of solver time each. At the new lengths, H₂ gives −1.1373682 ± 9.9e-5 Eh (0.95σ) and LiH −7.8636798 ± 1.0e-5 Eh (0.18σ), both unflagged.

The same study found the opposite error on the shift. Its blocked error is 3–6× larger than the spread across seeds, and it was never flagged. This is the safe direction, but it weakens the shift `metric_within_sigma` gates. To reproduce the study, run a regression input 20 times with `OMP_NUM_THREADS=1`, changing only `fciqmc_seed` and `fciqmc_steps`. Then compare the standard deviation of the printed estimates with the RMS of their printed errors.

**CAS(12,10) exact control.** The five fresh seeds 20250904–20250908 were repeated with the pre-fix settings: 25,000 walkers, four spawn attempts, initiator threshold 0, timestep 0.0005, 110,000 equilibration and 10,000 sampling steps, four threads. The Hamiltonian ID matched the baseline (`339dffb09993d608`). The baseline is the [pre-fix run on the same seeds](#sampling-length-and-seed-spread-in-cas1210). Values are signed errors against CASCI (−2064.494323848 Eh) in mEh.

| Five seeds, 10,000 steps | Baseline projected | Fixed projected | Baseline shift | Fixed shift |
|---|---:|---:|---:|---:|
| Mean signed error | −6.02 | −3.64 | +0.40 | −4.01 |
| SD across seeds / RMS reported SE | 9.82 / 2.48 | 4.15 / 3.87 | 7.75 / 7.51 | 4.20 / 7.56 |
| Within two reported SEs | 1/5 | 4/5 | 5/5 | 5/5 |
| RMSE | ≈10.6 | 5.20 | ≈6.9 | 5.49 |

The projected error bar now matches the spread across seeds: the ratio is 1.07, against 4.0 before. With five seeds this ratio is uncertain by about ±35%. Every projected error was flagged unconverged, as expected for a 10,000-step window. The projected spread itself halved. Both fixed means lie about two standard errors of the mean below CASCI (−3.64 ± 1.86 and −4.01 ± 1.88 mEh); five seeds cannot tell a population-control bias from chance.

No sampling parent visit was below one walker on any seed (0 of about 44 million per run). So `fciqmc_sample_low_weight_parents`, which acts only on such parents, now has no effect. Sampling parent visits fell from about 55.9 million per 10,000 steps to about 44 million. Mean reference weight was 961–1,147 walkers, and no seed re-anchored. Wall times (78–95 s) are not comparable with the baseline, because the host also ran the H₂ and LiH seed studies. [Per-seed results](../tests/benchmarks/fciqmc/cr2_cas12_10_review_fixes_m3_r3.json)

## Performance on the M3

These measurements predate the November 17 fixes. They concern the propagation machinery, which the fixes did not change. Occupation rounding reduces the number of occupied parents, so the absolute timings will differ on the current code.

### R1 baselines: why start with water

Full trajectories at 50,000 target weight, four threads and seed 20250901. Each row is one diagnostic run with 20,000 equilibration and 30,000 sampling steps. Prefill includes the serial diagonal lookup and parent binning.

| R1 measurement | Water/6-31G | Hydrogen fluoride/6-31G |
|---|---:|---:|
| Mean occupied parents per sampling step | 178,914 | 48,485 |
| Sampling wall time | 557.82 s | 111.14 s |
| Prefill time | 188.54 s (33.8%) | 22.71 s (20.4%) |
| Prefill per parent visit | 35.13 ns | 15.61 ns |
| Merge per parent visit | 16.56 ns | 18.05 ns |
| Peak process RSS | 295.9 MiB | 46.6 MiB |

Water spent more than twice as long per parent in prefill as hydrogen fluoride. The water pilot's CPU sample also placed the diagonal callback second among application leaf frames. The full water trajectory's projected energy agreed with independent PySCF FCI to 0.38 reported standard errors. [R1 control measurements](../tests/benchmarks/fciqmc/controls_m3_r1.json)

### R2: skipping the serial diagonal lookup for surviving parents

The baseline driver prefilled the diagonal cache while binning every occupied parent, and the worker then queried the same cache. On the 2,000-equilibration/2,000-sampling pilots, the serial traversal was split on every 32nd sampling step, and the next step timed the usual fused traversal. A separate four-thread pass replayed the worker's cache hits in bin order, including its OpenMP launch and barrier.

| Pilot | Serial lookup | Binning | Worker lookup replay | Adjacent fused pass | Cache misses / sampled parent visits |
|---|---:|---:|---:|---:|---:|
| Water | 16.27 ns | 1.85 ns | 4.57 ns | 21.44 ns | 2,794 / 11,242,267 |
| Hydrogen fluoride | 12.33 ns | 1.87 ns | 4.09 ns | 15.32 ns | 503 / 2,919,863 |

Times are wall nanoseconds per parent visit. Splitting the loop changes locality, and replaying the worker lookup warms the cache, so the adjacent fused pass measures the actual combined cost. Both probe runs reproduced the baseline Hamiltonian, result and work counts. Single pilot wall times with and without the probes were 45.80 and 51.67 s for water, 11.53 and 11.13 s for HF.

The dominant serial cost was checking diagonals that were already cached. The driver now walks the previous and current sorted populations together. A surviving determinant already has a cached diagonal. A newly occupied determinant, including one returning after an absence, goes through the serial lookup before the workers start. All nonzero parents are binned in their original order with their current weight. The previous population serves as the record of which keys were prefilled, and its allocation is released before the next population and merge buffers are built. The walker test exercises initial, surviving and returning determinants, including equal alpha strings with different beta strings. [Driver prefill](../src/post_hf/fciqmc_driver.cpp) · [Parent preparation](../src/post_hf/ci/spawn_accumulator.h) · [Walker tests](../tests/fciqmc_walkers.cpp)

On the pilots, sampling prefill fell from 8.28 to 0.94 s for water (wall 51.67 → 37.78 s) and from 1.37 to 0.22 s for HF (wall 11.13 → 9.62 s), with identical Hamiltonian, result, work counters and final cache size. The adoption gate used three full-length fixed-seed pairs per fixture, in A/B, B/A, A/B order:

| Fixture, threads | Wall (s) | Paired reductions | Sampling prefill (s) | Sampling worker (s) | Peak RSS (MiB) |
|---|---:|---|---:|---:|---:|
| Water, 4 | 789.63 → 584.47 | 25.5%, 22.8%, 30.9% | 169.91 → 12.72 | 249.61 → 237.06 | 329.6 → 327.6 |
| HF, 4 | 164.16 → 135.62 | 17.4%, 14.6%, 17.6% | 19.57 → 3.16 | 51.35 → 51.41 | 45.2 → 48.7 |
| Be, 2 | 10.06 → 9.49 | 6.0%, 4.5%, 5.7% | 0.38 → 0.12 | 3.22 → 3.12 | 15.1 → 15.2 |

The columns with arrows give baseline and candidate medians; the percentages use the wall times of each matched pair. Every water pair exceeded the 10% end-to-end threshold. HF and Be passed the 5% control-regression gate. Both builds produced identical Hamiltonian and result fingerprints, work counters, cache sizes and vector capacities. [Runs and probe measurements](../tests/benchmarks/fciqmc/diagonal_prefill_m3_r2.json)

To repeat a pair, build the baseline at `7def132a` and the candidate with the same Release flags. Run each binary through `tests/run_fciqmc_scaling_profile.sh BINARY FIXTURE THREADS OUTPUT_DIR`, with a separate output directory per invocation. Set `OMP_DYNAMIC=false` and the Accelerate, OpenBLAS and MKL thread limits to one. [Scaling runner](../tests/run_fciqmc_scaling_profile.sh)

### Thread selection

Use 8 OpenMP threads for the HF/6-31G 50k fixture on this M3. Three fixed-seed trajectories ran at each thread count, in 1/2/4/8 order. All 12 reproduced Hamiltonian `f032498d77f973b5`, result `ea152a6472c1789b`, the work counters in both phases and 105,050 final diagonal-cache entries.

| Threads | R1 median wall (s) | R2 median wall (s) | R2 wall range (s) | R2 speedup | Worker (s) | Merge (s) | Prefill (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 424.83 | 396.49 | 390.91–397.01 | 1.00× | 196.18 | 39.81 | 3.22 |
| 2 | 250.39 | 217.41 | 216.35–220.60 | 1.82× | 97.60 | 28.48 | 3.21 |
| 4 | 169.38 | 139.24 | 134.77–139.34 | 2.85× | 53.29 | 25.19 | 3.22 |
| 8 | 151.51 | 122.92 | 120.12–126.20 | 3.23× | 41.90 | 27.02 | 3.25 |

Worker, merge and prefill are median sampling-phase times. From four to eight threads, worker time fell by 11.39 s and merge time rose by 1.83 s. [Optimized sweep](../tests/benchmarks/fciqmc/hf50k_m3_r2.json) · [R1 sweep](../tests/benchmarks/fciqmc/hf50k_m3_r1.json) · [Thread sweep runner](../tests/run_fciqmc_e1_profile_ladder.sh)

### Small-population merge

Use four threads for the cached spherical Be fixture on this M3. When there are at most 4,096 tagged entries, the production merge now runs the levels with four or fewer pairs sequentially. For the 64-bin tree these are the last three levels. Each level keeps its pair slots and addition order. The Be trajectories peaked at 1,812 tagged entries and HF 50k at 50,522.

| Fixture | Threads | Previous wall (s) | New wall (s) | Paired wall reduction | Sampling merge (s) |
|---|---:|---:|---:|---:|---:|
| Be | 1 | 11.21 | 11.16 | 0.5% | 0.88 → 0.88 |
| Be | 2 | 9.44 | 8.53 | 9.6% | 2.02 → 1.53 |
| Be | 4 | 9.80 | 8.08 | 17.5% | 3.22 → 2.16 |
| Be | 8 | 13.81 | 9.84 | 28.9% | 5.22 → 2.90 |
| HF 50k | 8 | 120.06 | 119.70 | 1.3% | 26.70 → 26.72 |

Each row reports three full-length pairs in A/B, B/A, A/B order. At eight threads on Be the three reductions were 28.9%, 23.9% and 29.0%; with profiling disabled, three more pairs gave a 27.9% median reduction and identical printed estimators. The HF 50k control passed the 5% regression limit. All 30 profiled runs reproduced the archived Hamiltonians, trajectories, work counters, cache sizes and vector capacities. [Measurements and settings](../tests/benchmarks/fciqmc/small_tree_merge_m3_r2.json)

## The Cr2 target

**Active space.** Cr₂ at 1.68 Å in Cartesian STO-3G has 38 orbitals, more than the 31 the packed CI representation admits. With 18 frozen doubly occupied RHF orbitals and two excluded virtuals, only 18 active orbitals enter the bitstrings. Six alpha and six beta electrons give 344,622,096 determinants. The active space uses the contiguous RHF orbitals in their returned order. [Cr₂ input](../tests/inputs/exploratory/fciqmc/cr2_cas12_18_pilot.hfinp)

**Orbital character.** A checkpoint run reproduced Hamiltonian `3ffac35b4805c0ec`. In the AO overlap metric, the active MOs retained 11.9813 of the 12 nominal Cr 3d/4s valence directions by summed squared overlap (99.84%). The least well represented valence direction had squared overlap 0.9916. The active block also retained 5.9789 of the six 4p directions, and the two excluded MOs carried 1.9841 of the two Cartesian d-shell trace directions. The projection orthonormalizes atom-centered shells in the AO metric in this order: core (1s, 2s, 2p, 3s, 3p), valence (five traceless 3d functions and 4s per Cr), 4p, then the two Cartesian traces. [Orbital projection weights](../tests/benchmarks/fciqmc/cr2_population_spawn_m3_r3.json)

**Hamiltonian handoff checks.** On the short N₂ control, the previous binary, the new default path and explicit CAS(14,10) all gave Hamiltonian `3d0ae86188394164` and trajectory `af01f83434290758`. LiH/STO-3G CAS(2,4) tests both a frozen core and an excluded virtual. PySCF CASCI on Planck's all-MO FCIDUMP gave −7.863681638749 Eh on the same orbitals. To regenerate it, add `fcidump lih.FCIDUMP` to the fixture's SCF block, read it with `pyscf.tools.fcidump.to_scf("lih.FCIDUMP")`, and run `pyscf.mcscf.CASCI(mf, 4, 2).kernel(numpy.eye(6))`; the identity coefficients keep Planck's MO order. [LiH regression input](../tests/inputs/regression/post_hf/lih_fciqmc_cas24_sto3g.hfinp)

**Reference energies.** Starting from the all-MO FCIDUMP, the same 18 core orbitals were frozen and the next 18 RHF orbitals isolated. Within that space, singlet CASSCF trial states were optimized in 10 and 12 orbitals, starting from high-spin valence orbitals, with rotations kept inside the parent space. Both normalized trial vectors therefore belong to the target CAS(12,18), so their energies are upper bounds. Contracting their one- and two-body density matrices with the Hamiltonian reproduced each energy within 10⁻⁸ Eh.

| Parent space | Optimized trial orbitals | Variational energy (Eh) | SC-NEVPT2 correction (Eh) | Estimated parent energy (Eh) |
|---|---:|---:|---:|---:|
| CAS(12,18) | 10 | −2064.860005 | −0.052832 | −2064.912837 |
| CAS(12,18) | 12 | −2064.896580 | −0.035204 | −2064.931784 |
| CAS(12,12) control | 10 | −2064.503320 | −0.012267 | −2064.515586 |

**The CAS(12,18) ground energy is at most −2064.896580 Eh.** The two SC-NEVPT2 estimates differ by 18.95 mEh. In the CAS(12,12) control, the corrected estimate was 10.98 mEh above direct CI (−2064.526570 Eh). These checks give a scale for the sensitivity to the trial space; only the variational energies give a one-sided bound. To reproduce, run `tests/pyscf/cr2_fciqmc_accuracy.py` on the all-MO pilot FCIDUMP with `--trial-orbitals 10` and `--trial-orbitals 12`, adding `--parent-orbitals 12` to the 10-orbital command for the direct CI control. The largest reference calculation took 216 s and 310 MiB on this M3. [Reference calculation](../tests/pyscf/cr2_fciqmc_accuracy.py)

**The CAS(12,10) exact control.** Freezing the same 18 core orbitals and keeping the first ten active RHF orbitals gives a 44,100-determinant space whose CASCI energy is **−2064.494323848 Eh**. To reproduce, export the all-MO FCIDUMP from the Cr₂ pilot, read it with `pyscf.tools.fcidump.to_scf`, and run `mcscf.CASCI(mf, 10, (6, 6)).kernel(numpy.eye(38))` in the input MO order. For a stochastic control, change only `nactorb` to 10.

**The RHF determinant does not carry the wavefunction beyond ten active orbitals.** FCIQMC seeds and projects on the lowest-diagonal determinant, which here is the RHF determinant `0x3f/0x3f` (active orbitals 0–5 doubly occupied). The table gives its weight in the exact CASCI vector, in the same fixed RHF orbitals FCIQMC samples. The L1 share is the fraction of walkers a correctly sampled population would place on it.

| Space | RHF determinant c₀² | L1 share (walkers at 25k) | Rank | Largest determinant, c² | Determinants for 50 / 90 / 99% of the norm |
|---|---:|---:|---:|---|---:|
| CAS(12,10) | 0.538 | 5.5% (1,378) | 1 | RHF itself | 1 / 38 / 435 |
| CAS(12,12) | 0.0055 | 0.27% (66) | 18 | `0x82f/0x82f`, 0.557 | 1 / 74 / 1,622 |
| CAS(12,13) | 0.0002 | 0.034% (8) | 206 | `0x42f/0x42f`, 0.378 | 2 / 131 / 3,357 |

The CASCI energies are −2064.494324, −2064.526570 and −2064.746875 Eh. Each is an upper bound for CAS(12,18) in the same orbitals; the CASSCF bound, −2064.896580 Eh, is tighter. `0x82f/0x82f` empties active orbital 4 and doubly occupies orbital 11. `0x42f/0x42f` empties orbital 4 and doubly occupies orbital 10; orbitals 10 and 11 are nearly degenerate (Fock energies 0.3369 and 0.3385 Eh). The CAS(12,18) FCIQMC run at τ = 30 re-anchored to `0x42f/0x42f` on its own. Its diagonal energy is 92 mEh above the RHF determinant's (−2064.2537 vs −2064.3458 Eh), so the lowest-diagonal search picks the RHF determinant correctly by its own criterion. The RHF is a proper Aufbau state: the occupied active Fock energies are −0.39 to −0.11 Eh and the virtuals start at +0.19 Eh. The determinant with the lowest diagonal is simply not the one the correlated state is built on. The RHF determinant's weight falls by more than an order of magnitude with each added orbital: by CAS(12,13) its overlap with the ground state is c₀ = 0.014. Seeding there means the walkers start almost entirely in excited states, so projection needs long imaginary times, and the single-determinant estimator divides by a weight near zero. Which determinant dominates in CAS(12,18) is not known; CAS(12,14) has 9.0 million determinants. All these CASCI vectors use Planck's fixed RHF orbitals, so each embeds exactly in the CAS(12,18) determinant space. That makes the CAS(12,13) vector a ready trial and starting state. [Reference-weight script](../tests/pyscf/cr2_reference_weight.py)

**Exact imaginary-time propagation.** Applying the continuous imaginary-time exponential of the CAS(12,10) Hamiltonian to the seeded determinant gives the projected energy a run would converge to at each τ, with no sampling noise. The script checks each projected energy against the Rayleigh quotient of the state at half the projection time. [Finite-time reference](../tests/pyscf/cr2_fciqmc_projection.py)

| τ (Eh⁻¹) | Exact projected minus CASCI (mEh) | Mean L1 growth energy minus CASCI over the window (mEh) |
|---|---:|---:|
| 5 / 6 | +27.345 / +22.999 | −59.239 (τ 5–6) |
| 55 / 56 | +0.875 / +0.830 | −0.754 (τ 55–56) |
| 60 | +0.670 | −0.680 (τ 55–60) |
| 80 | +0.233 | −0.426 (τ 55–80) |

The seeded determinant is still relaxing at τ = 5, and even the exact projected and growth energies lie on opposite sides of the ground state there. Since the projected energy approaches from above, any sampling window ending before τ ≈ 50 carries a positive projection bias of order 10 mEh in CAS(12,10). CAS(12,18) is likely to need at least as long.

## Pre-fix Cr2 measurements

Everything in this section ran before `ed9ea6a6`. Findings 1, 2 and 4 affect every energy and error bar below, so the energies are not comparable with post-fix runs. The resource and timing figures still bound what the method costs. The deterministic results, the exact propagation above and the reference energies, are unaffected.

### First pilot: resources

Both runs used four threads, 10,000 target walkers, timestep 0.0005, initiator threshold 3, seed 20250901, one spawn attempt and 2,000 sampling steps; only equilibration changed. Wall time and peak RSS include SCF.

| Equilibration steps | Wall (s) | Peak RSS (MiB) | Shift (Eh) | Projected (Eh) | Usable projected samples |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 8.56 | 84.2 | −2067.269 ± 0.158 | −2063.837 ± 0.026 | 1,994 / 2,000 |
| 10,000 | 48.65 | 109.8 | −2068.514 ± 0.139 | −2062.651 ± 0.053 | 803 / 2,000 |

Both trajectories re-anchored after the seeded determinant lost weight, and the estimator gap grew from 3.43 to 5.86 Eh. These runs established resource feasibility only. [Profiles and timings](../tests/benchmarks/fciqmc/cr2_cas12_18_m3_r3.json)

### Walker population and spawn attempts

Seeds 20250901–20250903 ran with 2,000 equilibration and 2,000 sampling steps on four threads. Raising the population at one spawn attempt widened the shift–projected gap on every seed. Raising the attempts per occupied determinant kept the seeded reference, made every projected sample usable and narrowed the gap. All 15 gaps exceeded the driver's warning threshold.

| Target walkers | Draws per parent | Gap (Eh) | Usable samples / 2,000 | Mean reference weight | Wall (s) | Max RSS (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 1 | 2.00–3.73 | 1,899–2,000 | 15.5–25.1 | 7.5–8.4 | 84.3 |
| 25,000 | 1 | 2.29–4.17 | 1,236–2,000 | 22.9–40.1 | 15.7–17.6 | 124.9 |
| 50,000 | 1 | 2.47–4.43 | 1,236–2,000 | 32.2–56.9 | 30.3–33.3 | 209.5 |
| 10,000 | 4 | 1.00–1.01 | 2,000 | 90.5–133.5 | 14.6–15.0 | 76.1 |
| 10,000 | 16 | 0.89–0.99 | 2,000 | 177.3–183.7 | 39.2–40.4 | 73.7 |

Ranges span the three seeds. With four draws, shifts ranged from −2065.564 to −2065.528 Eh and projected energies from −2064.555 to −2064.515 Eh. [Per-seed results](../tests/benchmarks/fciqmc/cr2_population_spawn_m3_r3.json)

### Equilibration and population at four spawn attempts

Extending equilibration from 2,000 to 10,000 steps (sampling window τ = 1–2 to 5–6) at 10,000 walkers narrowed the gap on one seed and widened it on two. Raising the target to 25,000 walkers at 10,000 equilibration steps narrowed it on one seed and widened it on the other two. Mean reference weight fell to 5–13 walkers. Under the pre-fix single-step rule, two seeds re-anchored, and the replacement determinants differed between populations.

| Seed | Projected (Eh): 10k, τ 1–2 → 10k, τ 5–6 → 25k, τ 5–6 | Gap (Eh), same three runs | Usable samples / 2,000, τ 5–6 (10k → 25k) | Mean reference weight, τ 5–6 (10k → 25k) | 25k reference |
|---|---:|---:|---:|---:|---|
| 20250901 | −2064.549 → −2064.660 → −2064.693 | 1.002 → 0.682 → 0.816 | 2,000 → 2,000 | 12.0 → 13.3 | seeded |
| 20250902 | −2064.515 → −2064.241 → −2064.788 | 1.012 → 1.296 → 0.899 | 1,292 → 2,000 | 6.5 → 5.9 | changed |
| 20250903 | −2064.555 → −2064.473 → −2063.804 | 1.009 → 1.282 → 2.132 | 1,393 → 1,284 | 5.6 → 4.9 | changed |

The 25k runs took 152–163 s and peaked at 96.5–114.7 MiB. During sampling, workers took 87–88% of wall time and the initiator rule rejected about 98% of spawn attempts. [Equilibration runs](../tests/benchmarks/fciqmc/cr2_equilibration_m3_r3.json) · [25k runs](../tests/benchmarks/fciqmc/cr2_long_25k_m3_r3.json)

### The bound violation

Measured against the upper bound of −2064.896580 Eh, all three 25,000-walker projected energies at τ = 5–6 sat above it:

| Seed | Projected minus upper bound (Eh) | Projected blocked SE (Eh) | Shift minus 12-orbital SC-NEVPT2 estimate (Eh) | Accepted reference weight, mean |
|---|---:|---:|---:|---:|
| 20250901 | +0.203 | 0.013 | −0.577 | 13.26 |
| 20250902 | +0.108 | 0.086 | −0.756 | 5.88 |
| 20250903 | +1.092 | 0.102 | −1.005 | 4.88 |

The three four-draw projections at 10,000 walkers and τ = 1–2, which kept the seeded reference, also exceeded the bound, by 0.341–0.381 Eh. For seed 20250901, extending sampling from 2,000 to 10,000 steps (τ = 5–10) moved the projected energy from −2064.693174 to −2064.633191 ± 0.0313 Eh. Only 3,483 of the added 8,000 steps gave usable projections; finding 4 may account for this. The shift was −2065.646618 ± 0.0435 Eh. The run took 338.4 s and 79.5 MiB. [Accuracy comparisons and extended run](../tests/benchmarks/fciqmc/cr2_cas12_18_accuracy_m3_r3.json)

Candidate causes, none isolated: incomplete imaginary-time projection (see the exact propagation table), initiator bias at threshold 3, a reference holding 0.02–0.05% of the walkers, and findings 1, 2 and 4.

### Timestep, initiator and projection time in CAS(12,10)

The same 25,000-walker settings were run directly in CAS(12,10), where CASCI gives signed errors. All three seeds kept the seeded reference. Halving the timestep, with doubled step counts and shift interval to keep the same τ windows, did not reduce the projected error. Disabling the initiator rule moved the projected energy by only +0.17 to +0.37 mEh at τ = 5–6. Projected errors fell only once the window moved to τ = 55–56, which the exact propagation explains. Entries are shift / projected deviations from CASCI in mEh.

| Seed | dt 0.0005, threshold 3, τ 5–6 | dt 0.00025, threshold 3, τ 5–6 | Threshold 0, τ 5–6 | Threshold 0, τ 55–56 |
|---|---:|---:|---:|---:|
| 20250901 | −86.7 / +18.1 | — | −84.3 / +18.4 | +10.8 / +11.0 |
| 20250902 | −86.2 / +29.9 | — | −81.6 / +30.1 | +12.7 / +0.6 |
| 20250903 | −63.2 / +16.1 | — | −53.3 / +16.3 | +4.0 / +9.2 |
| Range | −86.7 to −63.2 / +16.1 to +29.9 | −75.9 to −65.3 / +24.0 to +35.2 | | |

All of these runs used every projected sample. The τ = 55–56 runs took 84–88 s. [Initiator and window runs](../tests/benchmarks/fciqmc/cr2_cas12_10_initiator_m3_r3.json) · [Timestep controls](../tests/benchmarks/fciqmc/cr2_long_25k_m3_r3.json)

### Sampling length and seed spread in CAS(12,10)

Settings: threshold 0, 110,000 equilibration steps, otherwise as above. The three original seeds extended sampling from 2,000 to 10,000 steps, and five fresh seeds (20250904–20250908) sampled 10,000 steps. Values are in mEh. SD is the sample standard deviation across seeds; RMS SE is the root mean square of the reported blocked errors.

| Seeds and sampling length | Mean shift error | Shift SD / RMS SE | Mean projected error | Projected SD / RMS SE |
|---|---:|---:|---:|---:|
| Original three, 2,000 steps | +9.14 | 4.56 / 17.16 | +6.94 | 5.53 / 1.59 |
| Original three, 10,000 steps | −1.55 | 5.06 / 7.79 | +2.52 | 4.13 / 3.26 |
| Five fresh, 10,000 steps | +0.40 | 7.75 / 7.51 | −6.02 | 9.82 / 2.48 |
| All eight, 10,000 steps | −0.33 | 6.53 / 7.62 | −2.82 | 8.92 / 2.80 |

Four of the five fresh projected estimates missed CASCI by more than twice their reported error, with signed errors from −14.86 to +7.33 mEh. This is the 4× understatement fixed by finding 2. All eight runs kept the seeded reference, with reference weights of 1,093–1,252 walkers. They took 90.7–95.3 s and peaked at 57.6 MiB; equilibration was about 90% of the time. [Per-seed results](../tests/benchmarks/fciqmc/cr2_cas12_10_sampling_m3_r3.json)

The five fresh seeds were then extended to 50,000 steps (τ = 55–80). The extension's errors are isolated by subtracting the prefix, weighting shifts by sample count and projected energies by the signed reference-weight sums.

| Sampling window | Mean projected error (mEh) | Projected SD / RMS SE (mEh) | Mean shift error (mEh) | Shift SD / RMS SE (mEh) |
|---|---:|---:|---:|---:|
| Prefix, 10,000 steps | −6.02 | 9.82 / 2.48 | +0.40 | 7.75 / 7.51 |
| Extension, 40,000 steps | +0.45 | 2.17 / — | −3.82 | 5.63 / — |
| Full, 50,000 steps | −0.83 | 2.61 / 3.41 | −2.97 | 5.72 / 3.48 |

Over the full window, four of five projected and four of five shift estimates fell within two reported errors. Seed 20250904's shift stayed 11.83 mEh below CASCI with a reported error of 3.15 mEh. Five times the sampling cost 1.30–1.32× the total wall time (122.3–125.0 s). [Per-seed results](../tests/benchmarks/fciqmc/cr2_cas12_10_sampling50k_m3_r3.json)

### Cost of CAS(12,12)

Seed 20250904 ran the first 12 active RHF orbitals (853,776 determinants) with 10,000 equilibration and 2,000 sampling steps, otherwise as in CAS(12,10). It took **90.7 s and 182.2 MiB**. Occupied parents averaged 51,178 per equilibration step and 94,048 per sampling step, and workers took 84.3% of sampling wall time. Of 752,381,920 sampling spawn attempts, 752,295,229 rounded to zero (99.988%). The seeded reference averaged 6.08 walkers on 1,635 of 2,000 usable samples. The last 2,000 steps cost **11.92 ms per step**, against an average budget of **3.75 ms per step** for a 110,000 + 50,000-step run, which would take about 31 minutes. Finding 1 explains the parent count and the rounding losses; the per-step cost on the fixed code has not been measured. [Settings and costs](../tests/benchmarks/fciqmc/cr2_cas12_12_cost_m3_r3.json)

### Spawn work by parent weight

The profiler groups each occupied determinant by its incoming |weight| relative to the spawn granularity, one walker in these runs. A surviving spawn is a nonzero contribution before merging. The replays reproduced the archived input, Hamiltonian and trajectory fingerprints.

| Sampling phase | Target walkers / draws | Initiator threshold | Visits and attempts below one walker | Absolute parent weight below one walker | Surviving spawns from those parents |
|---|---:|---:|---:|---:|---:|
| CAS(12,12), seed 20250904, τ = 5–6 | 25,000 / 4 | 0 | 96.1% | 80.4% | 78.2% |
| CAS(12,18), seed 20250901, τ = 1–2 | 10,000 / 1 | 3 | 93.6% | 50.9% | 16.9% |
| CAS(12,18), seed 20250903, τ = 5–6 | 25,000 / 4 | 3 | 93.4% | 62.0% | 46.6% |

These parents existed because surviving weights were never rounded (finding 1). After the fix, the CAS(12,10) control had no sampling parent below one walker. [Phase counts](../tests/benchmarks/fciqmc/cr2_parent_weight_m3_r3.json)

### Probabilistic spawning from low-weight parents

The experimental input `fciqmc_sample_low_weight_parents true` lets each incoming parent below the spawn granularity spawn with probability |w|/granularity, at the signed granularity as its spawn weight, so the expectation is unchanged. The diagonal step and the initiator test use the incoming weight. On the November 16 build, sampling excitation draws on the short control fell by 25.9%, and raw trajectories matched at 1, 2, 4 and 8 threads. [Qualification](../tests/benchmarks/fciqmc/cr2_low_weight_qualification_m3_r3.json)

A five-seed CAS(12,10) comparison at near-equal wall time used 50,000 baseline and 79,000 candidate sampling steps:

| Five-seed result | Baseline | Low-weight sampling |
|---|---:|---:|
| Sampling steps | 50,000 | 79,000 |
| Mean total wall time, s | 123.14 | 127.01 |
| Projected RMSE, mEh | 2.48 | 1.40 |
| Shift RMSE, mEh | 5.92 | 6.45 |
| Mean signed projected error, mEh | −0.833 | −0.829 |
| Mean signed shift error, mEh | −2.972 | −2.882 |

The option treated a symptom of finding 1. With occupation rounding in place no parent is below the granularity, so it no longer has any effect. [Calibration and per-seed results](../tests/benchmarks/fciqmc/cr2_low_weight_sampling_m3_r3.json)
