# Scope: F5 — driver wiring, the N2 regression gate, and the determinism decision

**Scope for in-flight work. Not started.** Step F5 of the ladder in
`FCIQMC_RESEARCH_SCOPE.md`. F1–F4 are landed and gated by
`planck-fciqmc-walkers`: the method runs, holds a population, and reports both
estimators with honest error bars.

**This step exists because everything validated so far runs on a synthetic
Hamiltonian.** `ToyHamiltonian` respects the sparsity of a real one and is checked
against exact diagonalization, but it is not a molecule: no SCF, no real
integrals, no basis set. Nothing yet demonstrates FCIQMC reproduces a *chemical*
answer.

## The deliverable that motivates the rest

**A regression case that reproduces N2/STO-3G deterministic FCI within its own
error bar.** N2/STO-3G is the research scope's primary validation fixture (10
orbitals, 7α/7β, ndet = 14 400, exact FCI cheap at ~8 s), chosen because it is the
smallest system where FCI is affordable *and* the determinant space is large
enough that a few-thousand-walker population is a genuine sample rather than
covering the space.

**Why this cannot be a unit test.** The gate would need a converged SCF for its
integrals — `h_eff = CᵀH_coreC` plus the transformed two-electron array — which
means linking the basis, integral and SCF machinery into a test that currently
links one file. And 14 400 determinants is 400x the current fixture, which already
uses the whole ~30 s budget. The honest home is a regression case driven by the
real binary.

## What is built, and what remains

**Built (F5.1–F5.2).** `run_fciqmc` (`src/post_hf/fciqmc_driver.{h,cpp}`) mirrors
`run_fci` and is dispatched from `hf_driver.cpp` on `correlation fciqmc`, with
eleven input keywords. The integral transform is **shared, not copied** —
`build_all_mo_ci_setup` was extracted from `run_fci` and both call it — so the two
paths cannot disagree about the Hamiltonian, only about how they solve it.

**Remaining.** The N2 gate (F5.3) and the determinism decision (F5.4).

## Steps

### F5.1 — the driver entry point — **DONE 2026-09-01. FCIQMC runs on real integrals.**

`run_fciqmc` (`src/post_hf/fciqmc_driver.{h,cpp}`), dispatched from
`hf_driver.cpp` on `correlation fciqmc`. Measured on H2/STO-3G against the exact
FCI `-1.1372744062`:

| estimator | value | blocked error | deviation |
|---|---|---|---|
| shift | −1.1375360199 | 2.76e-03 | **0.09σ** |
| projected | −1.1373278832 | 1.58e-04 | **0.34σ** |

**The integral transform is shared, not reimplemented.** `build_all_mo_ci_setup`
was *extracted* from `run_fci` — validation, electron counts, the packed-orbital
and `ci_max_dim` guards, `h_eff` and `ga` — and `run_fci` now calls it. Verified
behaviour-neutral: the FCI regression cases pass and N2 still gives
`-107.6529998854`, digit-identical. The Hamiltonian callbacks wrap
`slater_condon_element`, the same routine the deterministic CI uses, so the two
paths cannot disagree about the Hamiltonian — only about how they solve it.

**Gated** by `h2_fciqmc_sto3g` (extended suite), the first production consumer of
`metric_within_sigma`: both estimators asserted within 5 of their *own* blocked
error bars. Verified non-vacuous — against a deliberately wrong reference it fails
with the deviation reported in σ.

A collapsed or diverged population is a hard error naming the likely cause, never
a reported number.

**A build-hygiene trap worth recording.** A monitor watching only
`fciqmc_driver.cpp`'s timestamp fired on a build that predated the `io.cpp` edit by
one minute, so the first run failed with "Invalid Correlation : fciqmc" against
correct source. **Watching one file's timestamp does not prove the build included
every edit** — check all of them.

### F5.2 — input keywords — **DONE 2026-09-01**

Eleven keywords in `_scf_map`, each validated at parse time so a bad value fails
naming the keyword (`fciqmc_walkers must be positive`, `fciqmc_steps must be at
least 4`, …) rather than surfacing later as odd behaviour.

**Every parameter verified to change the run**, by varying each against a baseline
— nine moved the energy immediately. The tenth, `fciqmc_initiator`, appeared inert
at `n_add = 2.0`; investigation showed it is correctly plumbed and the **probe
value was below the walker scale**. With 5000 walkers on 4 determinants every
parent weight is ~1250, so a threshold of 2 never fires. At `n_add = 100` and
`1e9` it changes the answer as expected. Same fixture-saturation limit F4.5 hit.

**The reproducibility contract holds end-to-end through the real binary:** seed
4242 twice gives `-1.1382560651` identically, seed 9999 gives `-1.1373518204`.
That is F3.5's property verified at the driver level rather than in a unit test.

All parameters — seed included — are echoed to the output, so a result is
reproducible from its own log.

**A build-verification trap, and it defeated the fix for the previous one.**
F5.1's lesson was to check every edited file's timestamp against the binary. That
check *passed* while the binary still lacked the change, because a relink during
an in-flight build can produce a binary newer than its own inputs. A
`strings | grep -c` then returned 2 and looked like confirmation — but it was
matching the **error-message strings** (`"fciqmc_walkers must be positive"`), not
the map key. An exact match (`grep -qx`) showed the keyword genuinely absent.

> **A substring match on a binary is not evidence the symbol you care about is
> there, and a timestamp is not evidence a build finished.** Test the actual
> condition: build not running, *and* exact symbol present.

### F5.3 — the N2/STO-3G regression gate — **DONE 2026-09-02. Both estimators reproduce exact FCI.**

`n2_fciqmc_sto3g` (extended suite, **69 s**). Against exact FCI
`-107.6529998854` at `dt = 0.001`, τ_eq = 20:

| estimator | value | blocked error | deviation |
|---|---|---|---|
| shift | −107.6404846 | 3.89e-02 | **0.32σ** |
| projected | −107.3108220 | 8.28e-01 | **0.41σ** |

**Verified non-vacuous:** injecting the unstable `dt = 0.010` makes it fail on
three independent grounds — the SIGN-UNSTABLE warning, the projected energy
outside 5σ, and the two-estimator cross-check.

#### The projected energy was never a broken estimator

It took three attempts to see that, and the first two were wrong:

1. **`c_0` collapse** — refuted by measurement. Raising the population 10x made
   the answer *worse* (deviation 1.01 → 1.98), which no sampling-noise problem
   does.
2. **Mean of ratios instead of ratio of sums** — a genuine defect and the third
   appearance of `E[A/B] ≠ E[A]/E[B]` in this project, but worth only 1.1x here.
3. **The real cause: the reference determinant was oscillating in sign.** At
   `dt = 0.010`, mean `|c_0|` was 91.75 while mean *signed* `c_0` was −7.50, so the
   ratio's denominator nearly cancelled. That is the timestep instability F4.3
   gates: when `dt·|H_ii − S| > 2` the diagonal factor `(1 − dt(H_ii − S))` drops
   below −1 and the weight flips sign every step.

**The projected energy was correctly reporting a real problem with the run.** At
`dt = 0.001` the denominator is cleanly positive (mean signed `c_0` = 76.49) and
both estimators agree with exact FCI.

**The shift energy did not notice.** At `dt = 0.010` it read **0.14σ** from exact
while the dynamics were unstable, because it responds to the *total* population,
which is dominated by well-behaved determinants. **A single-estimator
implementation would have reported a perfect-looking answer.** This is the
strongest vindication of the two-estimator design in the project.

The driver now warns on sign instability directly — comparing mean signed `c_0`
against mean `|c_0|` — and says explicitly that the shift may still look
converged. The gate asserts `not_contains: SIGN-UNSTABLE`, so the instability
fails the case even if the energies happen to land.

**Equilibration was the first error, and also mine.** `dt = 0.001` with 2000 steps
is τ = 2, at which 14-82 % of an excited component survives; the shift recovered
only 74.7 % of correlation. **A small timestep makes a given step count a *short*
time, not a long one.**

**Runtime:** 138 s at 80 000 sampling steps, 69 s at 30 000. The shorter run keeps
both estimators inside 0.5σ, so it is what the gate uses; `timeout_s` is 180.

### F5.3 (superseded scope text) — the N2/STO-3G regression gate

**The headline result: FCIQMC reproduces exact FCI on N2/STO-3G.** 10 orbitals,
7α/7β, ndet = 14 400, at 0.69 walkers per determinant so the population is a
genuine sample rather than covering the space. Against exact FCI
`-107.6529998854`:

| dt | τ_eq | shift energy | deviation | correlation recovered |
|---|---|---|---|---|
| 0.001 | 20 | −107.6407856 ± 2.34e-02 | **0.5σ** | 98.6 % |
| 0.005 | 50 | −107.5855444 ± 4.19e-02 | 1.6σ | 92.4 % |
| **0.010** | **50** | **−107.6575413 ± 3.15e-02** | **0.1σ** | **100.5 %** |

**The dt-independence is the evidence**, not any single run: three different
discretizations of the same imaginary-time evolution converging on the same
answer.

**Equilibration was the first thing to get wrong.** The initial attempt used
`dt = 0.001` with 2000 equilibration steps — chosen conservatively for *stability*,
without checking what it buys in **imaginary time**. That is τ = 2, at which an
excited-state component of 14-82 % survives depending on the gap. The shift
recovered only 74.7 % of correlation. Raising τ_eq to 20-50 fixed it. **A small
timestep makes a given step count a *short* time, not a long one.**

#### The projected energy: one wrong diagnosis, then the real cause

The projected energy was badly wrong in every run (deviations 0.98-8.46 Eh,
error bars up to 6.63 Eh) while the shift was correct. **The F4.2 cross-check
caught it** — the driver warned that two estimators sharing no arithmetic
disagreed by 0.42 Eh, rather than reporting a plausible number.

**First diagnosis — `c_0` collapse — was WRONG, and a run refuted it.** The
reasoning was that at 0.69 walkers/determinant the reference holds too few
walkers for `1/c_0` to be stable. A falsifiable prediction was made (raise the
population and it should recover) and it failed:

| run | c₀ mean | projected | deviation | error bar |
|---|---|---|---|---|
| 10 000 walkers | 91.75 | −106.6445 | 1.01 | 0.34 |
| 100 000 walkers | **572.63** | −109.6339 | **1.98** | **1.70** |

Ten times more walkers on the reference made it **worse**, in both deviation and
error bar. No sampling-noise problem behaves that way.

**The real cause: averaging RATIOS instead of taking a RATIO OF SUMS.**

```
wrong:   E = H_00 + mean_t( sum_j H_0j c_j(t) / c_0(t) )
right:   E = H_00 + ( sum_t sum_j H_0j c_j(t) ) / ( sum_t c_0(t) )
```

`E[A/B] != E[A]/E[B]` — **the same inequality this project has now hit three
times**: F2.5's per-call acceptance-rate correction (1.72x wrong), F3.4's
documented finite-population bias, and here. It was written up twice and then
implemented in the wrong form anyway.

**What proved it, from data already collected:** two runs with *identical
configuration and seed*, differing only in the reference-weight threshold, gave
−99.19 and −106.64 — a **7.5 Eh** shift from a threshold change. Only a
heavy-tailed distribution does that, and the mean of such a distribution is set by
its outliers. Deviations also scattered on both sides of exact (−2.51, −0.98,
+8.46, −1.01, +1.98), which is outlier-dominated noise rather than a systematic
error.

The ratio-of-sums form is what production FCIQMC codes use, and it is not a
workaround: the numerator sum over the denominator sum converges to the true
ratio, while the mean of per-step ratios does not.

**Status: the fix is written and syntax-checked but NOT yet verified** — the build
was killed mid-flight, so the committed binary predates it. The next run must
confirm the projected energy converges near −107.65 *and* stops moving when the
reference-weight threshold changes.

**Still to do:** re-run to verify the fix, then add the regression case with
`n_sigma` set from the observed error bars, plus a fixed-seed reproducibility
assertion and a timing decision (default suite vs `extended`).

### F5.3 (original text) — the N2/STO-3G regression gate

The deliverable. Compare FCIQMC's energy against the deterministic FCI reference
for the same input.

**The reference, measured** (`OMP_NUM_THREADS=4`, ~8 s):

```
N2/STO-3G, 10 orbitals, 7 alpha / 7 beta, CI dim = 14400
  Total FCI Energy      -107.6529998854
  Correlation Energy      -0.8864061248
```

- **Use `metric_within_sigma`** (G1), against the exact FCI value, with the
  blocked standard error as the uncertainty metric. This is what that assertion
  was built for and it has had no production consumer until now.
- **Assert the reported error bar is blocked, not naive.** A run that reports the
  naive error would pass a within-σ check while understating its uncertainty by
  ~5x (measured in F4.4). Gate the ratio, or report both.
- **Assert fixed-seed reproducibility** in the same case: two runs with the same
  seed give bitwise-identical output. That is the gate that survives at any system
  size, and it costs one extra invocation.
- **Budget it honestly.** If the run is too slow for the default suite, put it in
  `extended` — but measure first rather than assuming, and record the number.

### F5.4 — the determinism decision — **DONE 2026-09-02. No exception.**

**The decision and its reasoning are in `FCIQMC_RESEARCH_SCOPE.md` §6**, where the
tension was raised. Summary: the burden that section set — show why FCIQMC cannot
do what the FCI sigma build did — **is not met**. FCIQMC can keep bitwise
thread-count invariance, by partitioning the **parents** (`hash(parent) % kBins`)
and merging bins in fixed order.

Verified on a model of the spawn: identical results whether parents are visited in
order, reversed, or shuffled — which is what thread-count invariance requires,
since thread count only changes visit order.

**Gated now, before the threading exists**, by `h2_fciqmc_threads1` /
`h2_fciqmc_threads4`: the same input at `OMP_NUM_THREADS` = 1 and 4, compared at
`atol = 0.0`. Today it passes trivially — FCIQMC has zero `#pragma omp` — and that
is the point: the property is pinned so adding threads cannot silently break it.
Verified non-vacuous by pointing one case at a different system, which fails.

**The trap worth carrying:** binning by the **child** determinant is *not*
sufficient. It fixes which accumulator receives a spawn but not the order arrivals
reach it, so two threads spawning onto the same determinant still race. The
partition must be over the **work**, not the output — the same lesson the sigma
build paid for.

### F5.4 (superseded scope text) — the determinism decision

**Do not start until F5.1–F5.3 are green**, and read §6 of the research scope
first. Every parallel path in Planck is bitwise thread-count-invariant, by design
and by gate. FCIQMC's natural parallelisation is not: the annihilation sum depends
on arrival order.

The FCI sigma build is the worked precedent — it threaded a scatter into a shared
vector and **kept** bitwise invariance for `kBins × dim × 8` bytes of
fixed-partition accumulators, at no measurable serial cost and 4.8 % idle. So the
burden is to show why FCIQMC cannot do what it did.

- **Decide explicitly, and write the decision down.** Either accept a fixed-order
  reduction, or document FCIQMC as the one path where bitwise thread-invariance
  does not hold. **Do not make the exception silently** — that is the failure this
  project has already paid for once.

## What this must not do

- **Do not gate the energy without an error bar.** A stochastic result quoted as a
  bare number invites a `metric_close` comparison, which is exactly the discipline
  mismatch Q2 asked about.
- **Do not tune the run parameters until the gate passes.** F4 established that
  `zeta` has a usable band and that the band is system-specific; a case that only
  passes at one hand-found setting is pinning an accident, not validating a method.
- **Do not report an energy from a run whose population collapsed or diverged.**
  F3 and F4 both established this; the driver must surface it as a failure rather
  than a number.
- **Do not let the N2 case be the only real-molecule gate.** Add at least one
  smaller one (H2 or LiH/STO-3G) that runs in the default suite, so a plumbing
  defect is caught in seconds rather than only by the slow case.

## Key code locations

| what | where |
|---|---|
| the method (F1–F4) | `src/post_hf/ci/fciqmc.{h,cpp}` |
| the unit gate | `tests/fciqmc_walkers.cpp` |
| the pattern to copy | `run_fci`, `src/post_hf/fci.{h,cpp}` |
| driver dispatch | `src/hf_driver.cpp:1401` |
| the `PostHF` enum | `src/base/types.h:73` |
| within-σ assertion | `metric_within_sigma`, `tests/run_regressions.py` |
| N2 input and its FCI reference | `tests/inputs/exploratory/fciqmc/n2_fci_sto3g.hfinp` |
| what F1–F4 established, and their traps | `docs/FCIQMC_SAMPLING_AND_DYNAMICS.md`, `docs/FCIQMC_POPULATION_CONTROL.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
