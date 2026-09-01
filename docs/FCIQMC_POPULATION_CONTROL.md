# How the walker population is controlled, and how the energy is trusted

This answers: **what holds an FCIQMC run at a usable size, what does it cost in
accuracy, and how do you know the energy it reports is right?**

It covers shift control, the two energy estimators, the timestep boundary, and
the initiator approximation — step F4 of the ladder in
`FCIQMC_RESEARCH_SCOPE.md`. Landed and gated by `planck-fciqmc-walkers` (~34 s).
Code in `src/post_hf/ci/fciqmc.{h,cpp}`; the sampling layer beneath it is
`FCIQMC_SAMPLING_AND_DYNAMICS.md`.

**Scope of validation:** everything here runs on a *synthetic* Hamiltonian that
respects a real one's sparsity and is checked against exact diagonalization. It is
not a molecule. Reproducing a chemical answer is F5's job
(`FCIQMC_F5_DRIVER_SCOPE.md`).

## 1. Why a fixed shift is not enough

`(1 − dt(H − S))` grows or shrinks the population exponentially unless `S` is the
ground-state energy — which is the unknown. Without control there is no affordable
run, no shift energy, and no way to see an unstable timestep.

## 2. The shift update needs two terms, not one

```
S(t) = S(t-A) - [ zeta * ln(N(t)/N(t-A)) + xi * ln(N(t)/N_target) ] / (A * dt)
```

**The standard single-term form never targets a population.** The `zeta` term
responds to the growth *rate*, so it stops exponential drift and then stabilises
wherever the run happened to be. Measured with `xi = 0`: the final population is
**proportional to the starting one** — 135.7x the target from every start across a
1000x range.

The `xi` term supplies the restoring force. With it the population lands on target
from both directions, and **the shift accuracy is unchanged** (3.2e-13 either
way), so the target term costs nothing.

### What `zeta` trades depends on which terms are present

With `xi = 0` it trades shift accuracy against population tightness. With the
target term doing the targeting, `zeta` becomes a **stability** parameter:

| ζ (ξ = 0.05) | peak/target | shift error | shift stdev |
|---|---|---|---|
| 0.0 | 20.36 | 4.6e-1 | 9.5 |
| 0.1 | 1.000 | 1.6e-10 | 3.9e-10 |
| 0.5 | 1.000 | 1.6e-10 | 3.9e-10 |
| 2.0 | 1.98 | 3.8 | 27.7 |
| 5.0 | **diverged** | | |

Both ends fail, which is the evidence the feedback is real. **The usable band is
system-specific** — the gain is `zeta/(A·dt)`, so a different `dt` moves the whole
band: 0.1–0.5 on a 20-determinant model, much higher on the 36-determinant one.
A gate pinned to one ζ pins an accident, so the gate asserts the *tradeoff*.

### The `A·dt` denominator is what makes ζ dimensionless

Dropping it **passed every behavioural check**, because it is equivalent to
rescaling ζ and ξ — which the tradeoff tests deliberately do not pin. It has its
own gate asserting the scaling directly: halving `dt` doubles the correction,
doubling the interval halves it.

> **A parameter's units cannot be gated by a test that only asserts the shape of a
> tradeoff in that parameter.**

## 3. Two estimators, and why that matters

**Projected energy** — `H_00 + Σ_j H_0j c_j / c_0`, from a ratio of walker weights
on the reference. Biased at finite population (see the sampling doc).

**Shift energy** — the time-average of `S` after equilibration, from the population
growth rate.

**They share no arithmetic**, which is what makes their agreement evidence rather
than a tautology. Measured across a 100x range of target populations:

| fixture | target 100 | target 1000 | target 10000 |
|---|---|---|---|
| 4 orbitals closed shell | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 5 orbitals open shell | 1.01e-09 | 1.01e-09 | 1.01e-09 |

**A gap of exactly zero is suspicious, so independence was verified rather than
assumed.** Perturbing *only* the projected energy by 1.0001 fails the cross-check
at a 7.97e-04 gap while the shift stays correct.

**Both are also pinned to the exact energy, not only to each other** — two
estimators can agree by sharing a common upstream defect (here, the propagator),
which agreement alone would never reveal.

### Equilibration is not optional

Averaging the transient in biases the result toward wherever the run started.
Starting 50x off target, discarding it improves the shift energy from **1.14e-02
to 2.19e-13** — a factor of 5×10¹⁰. The gate asserts that ratio, so if the fixture
ever starts equilibrating instantly the check reports itself as vacuous.

### Error bars must be blocked, never naive

A shift trajectory is strongly autocorrelated. The blocked error exceeds the naive
one by **4.7x** on a real trajectory here, and by up to 6.6x in the AR(1)
measurements behind `tests/blocking.py`. **Understating σ makes every downstream
gate pass**, which is the failure mode that matters. `blocked_standard_error` is
implemented in C++ and cross-checked against the validated Python at five
correlation strengths — identical to 1e-10 relative, so the two cannot drift.

## 4. The timestep boundary is observable only under control

F3 tried three times to assert that too large a `dt` breaks the propagation, and
every premise was false — the population does not collapse, the norm grows at
*every* `dt` by design, and the converged shape overlaps the true ground state even
at 5x the bound. Renormalizing each iteration turned it into a power iteration
whose dominant eigenvector stays the ground state.

**With the population controlled the boundary is sharp:**

| dt / diagonal bound | outcome |
|---|---|
| 0.10 – 0.26 | settles at target, shift −9.971196 |
| **0.30 – 0.60** | **diverges** |

**What this detects, precisely:** the boundary sits *below* the propagator's true
spectral limit (~0.44x the diagonal bound, itself 2.28x too large). So the
**controller** destabilises before the bare propagator would. This gates the
*controlled dynamics* — what a real run uses — and **the number must not be quoted
as the propagator's stability bound.**

Verified by isolating the controller: with `zeta = xi = 0` every timestep reports
"diverged", which is the exponential growth a frozen shift produces by design. The
transition exists only when the controller is active.

## 5. Discretization is what makes noise depend on population

`stochastic_round` was built in F1 and **never wired into the spawn**, leaving
weights continuous — which makes the propagator scale-invariant. Multiply the
population by *k*, every spawn scales by *k*, relative noise unchanged. Measured:
the blocked error was **4.2532e-02 at target populations of 500, 2000, 8000 and
32000 alike**, to five significant figures across a 64x range. That flat number is
what exposed it.

Spawns are now rounded to a `granularity` **stochastically** — rounding to nearest
would systematically discard sub-walker spawns and bias the energy. F1 gates that
property on `stochastic_round` itself; a separate test gates that the *spawn uses
it*, because swapping in `std::round` was otherwise invisible.

## 6. The initiator approximation

A spawn onto an **unoccupied** determinant is kept only if the parent's weight
exceeds `n_add`. In a space of 10⁸+ determinants most of the sign problem is
low-weight walkers scattering into determinants that should be empty; restricting
who may *colonize* suppresses that while leaving the established wavefunction free.

Occupancy is judged against the **incoming** population — using the partially-built
next one would make the rule depend on visit order.

**It is a biased approximation.** Never report an initiator energy without stating
the threshold.

## 7. Where the toy fixture runs out

Two of this step's intended gates could not be built on a 36-determinant fixture,
and both failures have the same cause.

**The initiator's `n_add → 0` convergence trend is not measurable.** The behaviour
is **binary**: below `n_add ≈ 100` every error is within one blocked σ of every
other; above ~300 the run is frozen with zero variance. The rule only fires on
spawns to unoccupied determinants, and at 5.5 walkers per determinant the space
fills within a few steps — so it never meaningfully fires until it freezes the
starting determinant outright.

**The stochastic error's population trend had to move below saturation.** With
discretization in, the error *rose* with population (3.87e-2 → 5.92e-2) at 14–889
walkers per determinant. Below ~1 walker per determinant it is clean:
3.65e-1 → 5.96e-2 over a 64x range.

> **This is the saturation trap the research scope names when it rejects H2 and
> water/STO-3G as FCIQMC fixtures, reached from the inside.** A space small enough
> to validate against exact diagonalization is often too small to exhibit the
> sampling behaviour being validated.

The convergence trend therefore belongs with the N2/STO-3G regression gate (F5.3),
where 14 400 determinants stay partially occupied at a realistic walker count.
Asserting it on the toy would mean tuning a fixture until a curve appeared.

## 8. Lessons that generalize

**A test can fail for a reason unrelated to the feature under test.** The
initiator's order-dependence check failed — and the control showed the
*propagator* already has insertion-order dependence, from hash-order iteration
against a shared RNG. Comparing against that control is what isolates the claim;
asserting an absolute that was never true would have sent the next reader hunting
a defect in the wrong place.

**A saturated fixture hides whole classes of defect.** A rule blocking *all*
spawns from low-weight parents was indistinguishable from the correct rule by
energy alone, because the occupancy condition rarely fired. It needed a direct
semantics test that pre-occupies the space.

**Statistical tests must be sized before they are trusted.** The spawn-bias test
first used spawns of ~0.04 walkers, making rounding near-binary with ~100 nonzero
events in 200k runs — it scattered **51 % on correct code**. Sizing spawns to
straddle the granularity fixed it, and it then caught the mutation at 43 %.

## Key code locations

| what | where |
|---|---|
| controller, estimators, initiator | `src/post_hf/ci/fciqmc.{h,cpp}` |
| the gate | `tests/fciqmc_walkers.cpp` (`planck-fciqmc-walkers`) |
| blocked error, cross-checked twin | `blocked_standard_error` / `tests/blocking.py` |
| the sampling layer beneath | `docs/FCIQMC_SAMPLING_AND_DYNAMICS.md` |
| the real-molecule gate this still needs | `docs/FCIQMC_F5_DRIVER_SCOPE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
