# FCIQMC Driver and Molecular Validation

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does `correlation fciqmc` become a working calculation, and what does it take to show it reproduces a chemical answer rather than a synthetic one?**

## Short answer

`run_fciqmc` mirrors `run_fci` and shares its integral transform, so the two paths cannot disagree about the Hamiltonian, only about how they solve it. FCIQMC reproduces exact FCI on N2/STO-3G (gated by `n2_fciqmc_sto3g`), the first validation on a real molecule at a walker population below saturation (0.69 walkers/determinant) rather than on a synthetic Hamiltonian or a trivially small system. This covers the driver entry point, input keywords, the N2/STO-3G validation, and the threading-determinism decision — step F5 of the ladder in `FCIQMC_RESEARCH_SCOPE.md`. Landed 2026-09-01/02. The layers beneath are `FCIQMC_SAMPLING_AND_DYNAMICS.md` (generator, propagator) and `FCIQMC_POPULATION_CONTROL.md` (shift control, estimators, initiator).

## Where the logic lives

- `src/post_hf/fciqmc_driver.{h,cpp}` — driver entry, estimators, warnings
- `src/post_hf/fci.{h,cpp}` (`build_all_mo_ci_setup`) — shared integral setup, extracted from `run_fci`
- `src/post_hf/ci/fciqmc.{h,cpp}` — the method itself
- `src/io/io.cpp` (`_scf_map`) — input keywords
- `tests/regression_cases.json` — `n2_fciqmc_sto3g`, `h2_fciqmc_threads1`, `h2_fciqmc_threads4`
- `FCIQMC_SAMPLING_AND_DYNAMICS.md`, `FCIQMC_POPULATION_CONTROL.md` — the layers beneath this one

## What invariants matter

### 1. Sharing the integral transform is what makes a sampling-vs-plumbing disagreement attributable

`build_all_mo_ci_setup` — validation, electron counts, the packed-orbital and `ci_max_dim` guards, `h_eff` and `ga` — was extracted from `run_fci`, a move rather than a copy. Both `run_fci` and `run_fciqmc` call it, and both Hamiltonian callbacks wrap the same `slater_condon_element`. Verified behaviour-neutral: the FCI regression cases pass and N2 still gives `-107.6529998854` digit-identical after the extraction.

Design rule:

- The two paths therefore cannot disagree about the Hamiltonian, only about how they solve it. Without that shared construction, a disagreement on a larger system would be ambiguous between a sampling artifact and a plumbing bug.

### 2. Every accepted input parameter must be verified to actually change the run

A keyword the parser accepts but silently ignores is worse than one it rejects outright. `fciqmc_initiator` looked inert at first; investigation showed it was correctly plumbed and the probe value used to test it was below the walker scale (5000 walkers on 4 determinants means every parent carries ~1250, so a threshold of 2 never fires) — the same fixture-saturation limit the population-control layer hit elsewhere.

Design rule:

- Verify each new keyword changes the run's output at a probe value sized to the system under test, not just that it parses without error. An inert-looking parameter needs its probe value checked against the actual scale of the quantities it gates before being trusted as working.

### 3. A small timestep makes a given step count a short time, not a long one

The first N2 attempt used `dt = 0.001` with 2000 equilibration steps, chosen conservatively for stability without checking what it buys in imaginary time. That is only tau = 2, at which 14-82% of an excited component survives, and the shift recovered only 74.7% of correlation as a result.

Design rule:

- When choosing an equilibration length, compute the resulting imaginary time (`dt * n_steps`), not just a step count that looks large in isolation.

### 4. A single-estimator implementation can report a perfect-looking wrong answer

At `dt = 0.010` the shift energy read 0.14 sigma from exact — apparently the best configuration — while the reference determinant was oscillating in sign (mean `|c0|` 91.75 against mean signed `c0` -7.50), the signature of the `dt*|H_ii - S| > 2` instability that drives `(1 - dt(H_ii - S))` below -1. The shift did not detect it because it responds to the total population, dominated by well-behaved determinants.

Design rule:

- Compute two estimators that share no arithmetic (shift and projected energy) and cross-check them. When they disagree, neither is automatically the broken one — instrument the disagreement rather than assuming which side is wrong.
- Warn on sign instability directly by comparing mean signed `c0` against mean `|c0|`, and state explicitly in the warning that the shift energy may still look converged despite it.

### 5. A failing diagnostic may be reporting a real problem, not a defect in the diagnostic

The projected energy was suspected of being a broken estimator, and it took three attempts to find the real cause. Two were wrong and instructive: (1) `c0` collapse — a falsifiable prediction (raising the population should recover it) that was refuted, since 10x more walkers made the deviation worse (1.01 to 1.98), which no sampling-noise problem does; (2) mean-of-ratios instead of ratio-of-sums — a genuine defect (the third appearance of `E[A/B] != E[A]/E[B]` in this project) but worth only 1.1x, not the observed size of the discrepancy. The real cause was the sign instability in invariant 4: the projected energy is a ratio anchored on the reference determinant, so a sign-flipping `c0` makes its denominator nearly cancel — it was correctly reporting a real problem with the run.

Design rule:

- Instrument the numerator and denominator of a suspect ratio-based estimator separately before theorizing about what is wrong with it. Two of three hypotheses here were plausible and both wrong; direct instrumentation settled it in one run.

### 6. Thread-count invariance requires partitioning the work, not the output

The research scope set the burden as "show why FCIQMC cannot do what the FCI sigma build did" for bitwise thread-count invariance, and that burden is not met: partitioning the *parents* by `hash(parent) % kBins` and merging bins in fixed order gives a result independent of the order threads visit parents, verified on a model of the spawn (in-order, reversed, and shuffled parent visits all identical). Binning by determinant is invariant even to the bin count itself, and `RandomSource::derive` already gives shard-count-independent streams, making this easier than the sigma build's case.

Design rule:

- Partition by the *work* (parents), never by the *output* (child determinant) — binning by the child fixes which accumulator receives a spawn but not the order arrivals reach it, so two threads spawning onto the same determinant would still race. This is the same lesson the FCI sigma build paid for.
- Gate the invariance property before the threading exists (`h2_fciqmc_threads1` / `h2_fciqmc_threads4` at `atol = 0.0`), so it passes trivially today and adding threads cannot silently break it later.

## What was found

1. **FCIQMC reproduces exact FCI on N2/STO-3G**, gated by `n2_fciqmc_sto3g` (extended suite, 69 s). 10 orbitals, 7 alpha / 7 beta, ndet = 14400, at 0.69 walkers per determinant — deliberately below saturation, so the population is a genuine sample rather than covering the space.

   | estimator | value | blocked error | deviation |
   |---|---|---|---|
   | shift | -107.6404846 | 3.89e-02 | **0.32 sigma** |
   | projected | -107.3108220 | 8.28e-01 | **0.41 sigma** |

   against exact FCI `-107.6529998854`. Verified non-vacuous: injecting an unstable timestep fails the gate on three independent grounds. Everything validated before this ran on a synthetic Hamiltonian, or on H2's 4 determinants where the walker population covers the whole space — this is the first test of the method itself rather than the plumbing.

2. **Eleven input keywords added**, each validated at parse time so a bad value fails naming the keyword, and each verified to actually change the run's output (see invariant 2).

3. **The seed is an input and is echoed** along with every other parameter, so a result is reproducible from its own log. The reproducibility contract holds through the real binary: seed 4242 twice gives `-1.1382560651` identically, seed 9999 gives `-1.1373518204`.

4. **The projected-energy investigation** (invariants 4 and 5) traced a real dynamical instability, not an estimator defect, and led to the sign-instability warning now built into the driver.

5. **The threading-determinism decision**: no exception. FCIQMC keeps bitwise thread-count invariance, gated ahead of the threading work actually existing.

## Validation strategy that should remain in place

- `n2_fciqmc_sto3g` — both estimators against exact FCI, with the sign-instability assertion (`not_contains: SIGN-UNSTABLE`) so an unstable run cannot pass on energy alone
- `h2_fciqmc_threads1` / `h2_fciqmc_threads4` — thread-count invariance at `atol = 0.0`, gating the property before any threading exists
- Fixed-seed reproducibility checks through the real binary
- Cross-checking the shift and projected energy against each other, not only against the exact reference — they share no arithmetic, so agreement between them is independent evidence

## Related but separate outcome: lessons on gate and build-verification mechanics

- **Gate assertions can be silent no-ops.** The sign-instability assertion was first written as `"forbidden"`, a key the runner ignores; the correct key is `not_contains`. This was checked rather than assumed, in a gate whose whole purpose is catching what would otherwise pass silently.
- **Build verification needs the actual condition, not a proxy.** A timestamp check passed while the binary still lacked the change (a relink mid-build can produce a binary newer than its inputs), and `strings | grep -c` returned 2 by matching error-message strings rather than the symbol itself. The fix is to test both that no build is running and that the exact symbol is present.
