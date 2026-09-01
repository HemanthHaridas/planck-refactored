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

The standard update, applied every `A` iterations:

```
S(t) = S(t - A) - (zeta / (A * dt)) * ln( N(t) / N(t - A) )
```

`zeta` is a damping parameter, and **it trades shift accuracy against population
control**. Measured on a 20-determinant model (`dt = 0.01`, `A = 5`, target
N = 1000, 4000 iterations, exact `E0 = -5.052827`):

| ζ | converged S | error vs E₀ | population overshoot |
|---|---|---|---|
| 0 (no control) | −2.000000 | 3.05 | 2×10⁵⁷ |
| 0.02 | −5.052804 | 2.2e-5 | 2062x |
| 0.1 | −5.052812 | 1.4e-5 | 3.6x |
| **0.5** | **−5.052813** | **1.4e-5** | **0.36x** |
| 2.0 | −5.044363 | 8.5e-3 | 0.05x |

Both ends fail in their own way: too little damping controls nothing, too much
biases the shift by 600x. **ζ ≈ 0.05–0.5 is the usable band on this model**, and
the gate should assert the tradeoff exists rather than pin one value — a run that
is insensitive to ζ is not doing population control.

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

### F4.1 — shift control on a fixed population

Implement the update above. Serial, deterministic propagator first, so the control
loop is separated from sampling noise exactly as F3.1 separated the dynamics.

- **Verify:** starting from a population far off target (10x high and 10x low),
  the population converges to the target and stays within a factor of ~2; the
  converged shift matches the exact ground-state energy to ~1e-5.
- **Verify the damping tradeoff exists:** small ζ gives an accurate shift and poor
  population control, large ζ the reverse. Assert both ends, not a single value —
  otherwise a no-op implementation that ignores ζ passes.

### F4.2 — the shift energy estimator, cross-checked against the projected energy

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
