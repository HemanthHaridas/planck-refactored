# Scope: F4 — population control and the initiator approximation

**Scope for in-flight work. Not started.** Step F4 of the ladder in
`FCIQMC_RESEARCH_SCOPE.md`. F1–F3 are landed and gated
(`FCIQMC_SAMPLING_AND_DYNAMICS.md`): the method runs on a **fixed** shift, which
means the population grows or shrinks exponentially by construction.

F4 is what makes it usable: hold the population at a target, and make the walker
count affordable on spaces where full sampling is not.

## What is missing without it

With `S` fixed the population is not stationary. Three consequences:

1. **No shift energy.** The second standard estimator is the value of `S` that
   holds the population steady — undefined until `S` is controlled.
2. **No affordable run.** An uncontrolled population either dies or explodes;
   neither reaches a useful sampling time.
3. **No divergence gate.** F3 could not build one — renormalizing each iteration
   turns the propagation into a power iteration whose dominant eigenvector stays
   the ground state at *every* `dt` tried, so an unstable timestep has nowhere to
   show. **With population control it does**, which is why that gate was deferred
   here rather than dropped.

## The mechanism, and the number that decides whether it works

The update, applied every `A` iterations:

```
S(t) = S(t-A) - [ zeta * ln(N(t)/N(t-A)) + xi * ln(N(t)/N_target) ] / (A * dt)
```

**CORRECTION (found building F4.1). This scope originally gave only the first
term, and that update never targets a population.** The `zeta` term responds to
the growth *rate*, so it stops exponential drift — but it has no term proportional
to the population itself, and the run simply stabilises wherever it happened to
be. Measured: the final population is **proportional to the starting one**
(135.7x the target from every start, across a 1000x range of starts).

The `xi` term supplies the restoring force. With it the population lands on target
from both directions, and **the shift accuracy is unchanged** (3.2e-13 either
way) — so the target term costs nothing and is what makes "population control"
mean what it says.

**What `zeta` trades depends on whether the target term is present**, which is the
second correction. With `xi = 0` it trades shift accuracy against population
tightness. With the target term doing the targeting, `zeta` becomes a **stability**
parameter — too little leaves the shift oscillating, too much destabilises it:

| ζ (with ξ = 0.05) | peak/target | shift error | shift stdev |
|---|---|---|---|
| 0.0 | 20.36 | 4.6e-1 | 9.5 |
| 0.1 | 1.000 | 1.6e-10 | 3.9e-10 |
| 0.5 | 1.000 | 1.6e-10 | 3.9e-10 |
| 2.0 | 1.98 | 3.8 | 27.7 |
| 5.0 | **diverged** | | |

Both ends still fail, which is the evidence the feedback is real — but the gate
must assert *that*, not any value. **The usable band is system-specific**: it runs
0.1–0.5 on the 20-determinant scoping model and much higher on the 36-determinant
test Hamiltonian, because the feedback gain is `zeta/(A·dt)` and a different `dt`
moves the whole band. A test pinned to one ζ pins an accident.

## The check that makes this step self-validating

**The shift energy and the projected energy must agree**, and they are computed
from entirely different quantities — one from the population growth rate, the
other from a ratio of walker weights on the reference. Measured on the same model,
across a 100x range of target populations:

| target N | mean shift | projected energy | difference |
|---|---|---|---|
| 100 | −5.052818 | −5.052820 | 2.0e-6 |
| 1 000 | −5.052818 | −5.052820 | 2.0e-6 |
| 10 000 | −5.052818 | −5.052820 | 2.0e-6 |

**This is the strongest gate available at this step**, because the two estimators
share no arithmetic. A defect in one would have to be exactly mirrored in the
other to escape. Note it is also insensitive to the target population, which is
what makes it usable as a gate rather than another bias trend to characterize.

## Steps

### F4.1 — shift control on a fixed population — **DONE 2026-08-31**

`ShiftController` in `src/post_hf/ci/fciqmc.{h,cpp}`. Gated: the population
returns to target from 10x high and 10x low; the converged shift matches the exact
ground-state energy; the ζ tradeoff fails at both ends; the target term is
required; and the update has the right units.

**Three things this step corrected**, all above: the single-term update never
targets a population; what ζ trades depends on the target term's presence; and the
usable band is system-specific.

**A fourth, from mutation testing.** Dropping the `A·dt` denominator **passed every
other check** — it is equivalent to rescaling ζ and ξ, which the tradeoff tests
deliberately do not pin. The denominator is what makes ζ *dimensionless* and
transferable across `dt`, so it now has its own gate asserting the scaling
directly: halving `dt` doubles the correction, doubling the interval halves it.
**A parameter's units cannot be gated by a test that only asserts the shape of a
tradeoff in that parameter.**

### F4.1 (original text) — shift control on a fixed population

Implement the update above. Serial, deterministic propagator first, so the control
loop is separated from sampling noise exactly as F3.1 separated the dynamics.

- **Verify:** starting from a population far off target (10x high and 10x low),
  the population converges to the target and stays within a factor of ~2; the
  converged shift matches the exact ground-state energy to ~1e-5.
- **Verify the damping tradeoff exists:** small ζ gives an accurate shift and poor
  population control, large ζ the reverse. Assert both ends, not a single value —
  otherwise a no-op implementation that ignores ζ passes.

### F4.2 — the shift energy estimator — **DONE 2026-08-31**

`ShiftAverager` in `src/post_hf/ci/fciqmc.h`. Gated: the two estimators agree
across a 100x range of target populations on closed and open shell, both match the
exact energy, the equilibration cut demonstrably matters, and the averager's
semantics (discard accounting, one-sample NaN, constant-series zero) hold.

**Measured agreement** — gap between shift and projected energy:

| fixture | target 100 | target 1000 | target 10000 |
|---|---|---|---|
| 4 orbitals closed shell | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 5 orbitals open shell | 1.01e-09 | 1.01e-09 | 1.01e-09 |

**A gap of exactly zero is suspicious, so independence was verified rather than
assumed.** Perturbing *only* the projected energy by a factor of 1.0001 makes the
cross-check fail at a 7.97e-04 gap while the shift stays correct — the two are
genuinely separate computations, one from the population growth rate and one from
a ratio of walker weights.

Both are also pinned to the exact energy, not only to each other: two estimators
can agree by sharing a common upstream defect (here, the propagator), which
agreement alone would not reveal.

**The equilibration check is doing real work**: starting 50x off target, the cut
improves the shift energy from **1.14e-02 to 2.19e-13** — a factor of 5×10¹⁰. If
that ratio ever collapses, the fixture has started equilibrating too fast to test
the discard.

### F4.2 (original text) — the shift energy estimator, cross-checked against the projected energy

Report `E_shift` as the time-average of `S` after equilibration.

- **Verify:** `|E_shift − E_projected|` is small, on at least three target
  populations spanning 100x. This is the self-validating check above.
- **Verify the equilibration cut matters:** averaging from iteration 0 must give a
  *worse* answer than averaging after equilibration. If it does not, the run
  reached equilibrium immediately and the fixture is too easy to be testing
  anything.

### F4.3 — the deferred timestep divergence gate

Now that the population is controlled, an unstable `dt` has an observable
consequence: the shift cannot hold the population steady.

- **Verify:** below the stability bound the population converges to target; above
  it the control loop fails to hold it (population diverges or oscillates without
  settling). **This is the gate F3 could not construct** — record whether it works
  here, and if it still does not, say so rather than contriving a fixture.

### F4.4 — stochastic population control

Repeat F4.1–F4.2 with `propagate_stochastic`.

- **Verify:** the shift fluctuates around the exact energy rather than sitting on
  it; its blocked standard error (G2) shrinks as √N_walkers — the **slope**, not
  just the value, since that is what catches a biased sampler a mean-only check
  cannot see.
- **Verify against the deterministic result** from F4.1 within the blocked error
  bar, using `metric_within_sigma`.

### F4.5 — the initiator approximation (i-FCIQMC)

Spawns onto **unoccupied** determinants are accepted only from parents whose
weight exceeds `n_add`; spawns onto occupied determinants are always accepted.
This is what makes large spaces affordable — it suppresses the noise from
low-weight walkers wandering into an exponentially large space.

- **Verify:** it is a **controlled approximation**, so the initiator energy must
  converge to the exact energy as `n_add → 0` *and* as the population grows.
  Assert that trend; a single agreeing number proves nothing, exactly as with the
  projected-energy bias.
- **Verify it actually restricts:** at a large `n_add` the answer must be
  measurably *wrong*. An initiator implementation that changes nothing would pass
  a convergence-only test.

## What this must not do

- **Do not tune ζ until a test passes.** The damping band is a property of the
  system and the target; a test that only passes at one hand-found ζ is pinning an
  accident. Assert the tradeoff.
- **Do not gate the shift energy alone.** It is an average of a controlled
  quantity and can look stable while being wrong; the cross-check against the
  projected energy is what makes it evidence.
- **Do not report an energy from an uncontrolled or collapsed population.** F3
  already established that an estimator quoted from a population that has collapsed
  to a handful of walkers is noise with a small error bar — the most misleading
  possible output.
- **Do not let the initiator hide a sampling defect.** It suppresses low-weight
  spawns, which is exactly where a `p_gen` error shows up. Every F4.5 check must
  also pass with the initiator disabled.
- **No `omp atomic`, no completion-order reduction.** F4 is serial; F5 decides the
  parallel policy.

## Key code locations

| what | where |
|---|---|
| walker state, generator, propagator (F1–F3) | `src/post_hf/ci/fciqmc.{h,cpp}` |
| the gate | `tests/fciqmc_walkers.cpp` (`planck-fciqmc-walkers`) |
| projected energy, to cross-check the shift | `projected_energy` |
| blocked error bars for F4.4 | `tests/blocking.py` |
| the within-σ assertion | `metric_within_sigma`, `tests/run_regressions.py` |
| what F3 established, and its traps | `docs/FCIQMC_SAMPLING_AND_DYNAMICS.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
