# How FCIQMC is driven, and how it is validated on a real molecule

This answers: **how does `correlation fciqmc` become a working calculation, and
what does it take to show it reproduces a chemical answer rather than a synthetic
one?**

It covers the driver entry point, input keywords, the N2/STO-3G validation, and
the threading-determinism decision — step F5 of the ladder in
`FCIQMC_RESEARCH_SCOPE.md`. Landed 2026-09-01/02. The layers beneath are
`FCIQMC_SAMPLING_AND_DYNAMICS.md` (generator, propagator) and
`FCIQMC_POPULATION_CONTROL.md` (shift control, estimators, initiator).

## The result

**FCIQMC reproduces exact FCI on N2/STO-3G**, gated by `n2_fciqmc_sto3g`
(extended suite, 69 s). 10 orbitals, 7α/7β, ndet = 14 400, at 0.69 walkers per
determinant — deliberately below saturation, so the population is a genuine sample
rather than covering the space.

| estimator | value | blocked error | deviation |
|---|---|---|---|
| shift | −107.6404846 | 3.89e-02 | **0.32σ** |
| projected | −107.3108220 | 8.28e-01 | **0.41σ** |

against exact FCI `-107.6529998854`. Verified non-vacuous: injecting an unstable
timestep fails the gate on three independent grounds.

Everything validated before this ran on a *synthetic* Hamiltonian, or on H2's 4
determinants where the walker population covers the whole space. This is the first
test of the method rather than the plumbing.

## 1. Sharing the integral transform is what makes comparisons meaningful

`run_fciqmc` mirrors `run_fci`, and `build_all_mo_ci_setup` — validation, electron
counts, the packed-orbital and `ci_max_dim` guards, `h_eff` and `ga` — was
**extracted from `run_fci`**, a move rather than a copy. Both paths call it, and
the Hamiltonian callbacks wrap the same `slater_condon_element`.

**The two paths therefore cannot disagree about the Hamiltonian, only about how
they solve it.** Without that, a disagreement on a larger system would be
ambiguous — sampling, or plumbing? The extraction was verified behaviour-neutral:
the FCI regression cases pass and N2 still gives `-107.6529998854` digit-identical.

## 2. Input keywords, and the seed

Eleven keywords, each validated at parse time so a bad value fails naming the
keyword. **Every parameter was verified to change the run** — a keyword the parser
accepts but ignores is worse than one it rejects.

`fciqmc_initiator` looked inert at first; investigation showed it correctly plumbed
and the **probe value below the walker scale** (5000 walkers on 4 determinants
means every parent is ~1250, so a threshold of 2 never fires). Same
fixture-saturation limit the population-control layer hit.

**The seed is an input and is echoed**, along with every other parameter, so a
result is reproducible from its own log. The reproducibility contract holds through
the real binary: seed 4242 twice gives `-1.1382560651` identically, seed 9999 gives
`-1.1373518204`.

## 3. Two errors in running it, both instructive

**Equilibration.** The first N2 attempt used `dt = 0.001` with 2000 equilibration
steps — chosen conservatively for *stability*, without checking what it buys in
**imaginary time**. That is τ = 2, at which 14-82 % of an excited component
survives; the shift recovered only 74.7 % of correlation.

> **A small timestep makes a given step count a *short* time, not a long one.**

**Timestep, and the failure that hid inside a good-looking number.** At
`dt = 0.010` the shift energy read **0.14σ** from exact — apparently the best
configuration — while the reference determinant was **oscillating in sign**: mean
`|c₀|` 91.75 against mean *signed* `c₀` −7.50. That is the instability where
`dt·|H_ii − S| > 2` drives the diagonal factor `(1 − dt(H_ii − S))` below −1 and
the weight flips every step.

The shift does not detect it, because it responds to the *total* population, which
is dominated by well-behaved determinants. **A single-estimator implementation
would have reported a perfect-looking answer.**

## 4. The projected energy was never a broken estimator

It took three attempts to see that, and the first two were wrong:

1. **`c₀` collapse** — a falsifiable prediction (raise the population and it
   recovers) that was **refuted**: 10x more walkers made the answer *worse*
   (deviation 1.01 → 1.98), which no sampling-noise problem does.
2. **Mean of ratios instead of ratio of sums** — a genuine defect, and the third
   appearance of `E[A/B] ≠ E[A]/E[B]` in this project, but worth only 1.1x here.
3. **The real cause was the sign instability above.** The projected energy is a
   ratio anchored on the reference determinant, so a sign-flipping `c₀` makes its
   denominator nearly cancel. **It was correctly reporting a real problem with the
   run.**

At `dt = 0.001` the denominator is cleanly positive (mean signed `c₀` = 76.49) and
both estimators agree with exact FCI.

**The driver now warns on sign instability directly**, comparing mean signed `c₀`
against mean `|c₀|`, and says explicitly that *the shift energy may still look
converged*. The gate asserts `not_contains: SIGN-UNSTABLE`, so the instability
fails the case even if the energies happen to land.

## 5. The determinism decision: no exception

Every parallel path in Planck is bitwise thread-count-invariant. The research
scope set the burden as *show why FCIQMC cannot do what the FCI sigma build did*.

**It is not met.** Partitioning the **parents** by `hash(parent) % kBins` and
merging bins in fixed order gives a result independent of the order threads visit
parents — which is what invariance requires, since thread count only changes visit
order. Verified on a model of the spawn: in-order, reversed and shuffled parent
visits all identical.

Two things make it *easier* than the sigma build: binning by determinant is
invariant even to the **bin count** (each determinant maps to one bin regardless,
so its contributions accumulate in the same order — the sigma build binned by index
range, where a determinant could move between bins), and `RandomSource::derive`
already gives shard-count-independent streams.

**The trap:** binning by the **child** determinant is *not* sufficient. It fixes
which accumulator receives a spawn but not the order arrivals reach it, so two
threads spawning onto the same determinant still race. **The partition must be over
the work, not the output** — the same lesson the sigma build paid for.

**Gated before the threading exists**, by `h2_fciqmc_threads1` /
`h2_fciqmc_threads4`: the same input at `OMP_NUM_THREADS` 1 and 4 at `atol = 0.0`.
It passes trivially today — FCIQMC has zero `#pragma omp` — and that is the point:
the property is pinned so adding threads cannot silently break it.

## 6. Lessons

**Two estimators that share no arithmetic are worth their cost.** This is the
clearest demonstration in the project: at `dt = 0.010` the shift read 0.14σ from
exact while the dynamics were unstable, and only the projected energy — plus their
cross-check — exposed it. The corollary is that when they disagree, *neither* is
automatically the broken one.

**A failing diagnostic may be reporting a real problem.** Three rounds were spent
treating the projected energy as defective before instrumenting the numerator and
denominator separately, which showed the denominator was negative and settled it in
one run. **Instrument before theorising** — the first two hypotheses were both
plausible and both wrong.

**Gate assertions can be silent no-ops.** The sign-instability assertion was first
written as `"forbidden"`, a key the runner ignores; it is `not_contains`. Checked
rather than assumed, in a gate whose whole purpose is catching what would otherwise
pass silently.

**Build verification needs the actual condition.** A timestamp check passed while
the binary still lacked the change (a relink mid-build can produce a binary newer
than its inputs), and `strings | grep -c` returned 2 by matching *error-message
strings* rather than the symbol. Test build-not-running **and** exact symbol
present.

## Key code locations

| what | where |
|---|---|
| driver entry, estimators, warnings | `src/post_hf/fciqmc_driver.{h,cpp}` |
| shared integral setup | `build_all_mo_ci_setup`, `src/post_hf/fci.{h,cpp}` |
| the method | `src/post_hf/ci/fciqmc.{h,cpp}` |
| input keywords | `_scf_map`, `src/io/io.cpp` |
| N2 gate | `n2_fciqmc_sto3g`, `tests/regression_cases.json` |
| thread-invariance gate | `h2_fciqmc_threads1` / `h2_fciqmc_threads4` |
| the layers beneath | `FCIQMC_SAMPLING_AND_DYNAMICS.md`, `FCIQMC_POPULATION_CONTROL.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
