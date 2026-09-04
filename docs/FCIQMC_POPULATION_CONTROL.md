# FCIQMC Population Control and Energy Estimators

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**What holds an FCIQMC run at a usable size, what does it cost in accuracy, and how do you know the energy it reports is right?**

## Short answer

A fixed shift cannot hold a population steady — `(1 - dt(H - S))` grows or shrinks the population exponentially unless `S` is the ground-state energy, which is unknown. The shift update needs two terms (a growth-rate term and an explicit population-target term), the two independent energy estimators (shift and projected) must be cross-checked because they share no arithmetic, the timestep divergence boundary is only observable once the population is controlled, spawns must be stochastically rounded so noise does not become population-independent, and the initiator approximation trades a controlled bias for tractability in large determinant spaces. This covers shift control, the two energy estimators, the timestep boundary, and the initiator approximation — step F4 of the ladder in `FCIQMC_RESEARCH_SCOPE.md`. Landed and gated by `planck-fciqmc-walkers` (~34 s).

**Scope of validation:** everything here runs on a synthetic Hamiltonian that respects a real one's sparsity and is checked against exact diagonalization. It is not a molecule. Reproducing a chemical answer is F5's job (`FCIQMC_DRIVER_AND_VALIDATION.md`).

## Where the logic lives

- `src/post_hf/ci/fciqmc.{h,cpp}` — controller, estimators, initiator
- `tests/fciqmc_walkers.cpp` (`planck-fciqmc-walkers`) — the gate
- `blocked_standard_error` / `tests/blocking.py` — the C++ implementation and its cross-checked Python twin
- `docs/FCIQMC_SAMPLING_AND_DYNAMICS.md` — the sampling layer beneath this one
- `docs/FCIQMC_DRIVER_AND_VALIDATION.md` — the real-molecule validation this step feeds into

## What invariants matter

### 1. The shift update needs a population-target term, not just a growth-rate term

The standard single-term shift update `S(t) = S(t-A) - zeta * ln(N(t)/N(t-A)) / (A*dt)` never targets a population — it responds only to the growth *rate*, so it stops exponential drift and then stabilizes wherever the run happened to be. Measured with `xi = 0`: the final population comes out proportional to the starting one (135.7x the target from every start across a 1000x range).

Design rule:

- Add a second term `xi * ln(N(t)/N_target) / (A*dt)` supplying an explicit restoring force toward the target population. With it, the population lands on target from both directions and the shift accuracy is unchanged (3.2e-13 either way) — the target term costs nothing.

### 2. What `zeta` trades depends on which other terms are present

With `xi = 0`, `zeta` trades shift accuracy against population tightness. With the target term present, `zeta` instead becomes a stability parameter: 0.0 leaves the shift oscillating at 4.6e-1 error, the usable band (0.1-0.5 on the tested fixture) gives shift error ~1.6e-10, and 2.0-5.0 destabilizes or diverges. Both ends fail, which is the evidence the feedback is real, but the usable band is system-specific: the gain is `zeta/(A*dt)`, so a different `dt` moves the whole band.

Design rule:

- Gate `zeta` on the *tradeoff shape* (both ends must fail), never on a pinned numeric value — a gate pinned to one zeta pins an accident of the fixture's `A*dt`.

### 3. A parameter's units cannot be gated by a test that only checks the shape of its tradeoff

Dropping the `A*dt` denominator from the shift update passed every behavioural check, because it is equivalent to rescaling `zeta` and `xi` — exactly what the tradeoff tests deliberately do not pin.

Design rule:

- Give a dimensioned parameter its own dedicated gate asserting the scaling directly (halving `dt` must double the correction, doubling the averaging interval must halve it), separate from any gate that only checks a qualitative tradeoff.

### 4. Two estimators that share no arithmetic are strong evidence only if independence is verified

The projected energy (a ratio of walker weights on the reference) and the shift energy (the time-average of `S` after equilibration, from the population growth rate) agree to 0.00e+00 (closed shell) and 1.01e-09 (open shell) across a 100x range of target populations. A gap of exactly zero is suspicious enough that independence was verified rather than assumed: perturbing only the projected energy by a factor of 1.0001 makes the cross-check fail at a 7.97e-04 gap while the shift stays correct.

Design rule:

- Pin both estimators to the exact energy independently, not only to each other — two estimators can agree by sharing a common upstream defect (here, the propagator), which agreement alone would never reveal.
- When two independent-looking quantities agree suspiciously well, verify the independence with a deliberate perturbation of only one side before trusting the agreement as evidence.

### 5. Error bars on an autocorrelated trajectory must be blocked, never naive

A shift trajectory is strongly autocorrelated; the blocked error exceeds the naive one by 4.7x on a real trajectory here, and by up to 6.6x in the AR(1) measurements behind `tests/blocking.py`.

Design rule:

- Always use a blocked (Flyvbjerg-Petersen-style) standard error for any autocorrelated stochastic trajectory. Understating sigma makes every downstream statistical gate pass, which is the failure mode that matters most.
- Cross-check any new-language port of the blocking algorithm against a previously validated implementation on synthetic data with a known answer (here, C++ `blocked_standard_error` against Python `tests/blocking.py` at five correlation strengths, agreeing to 1e-10 relative) so the two cannot silently drift apart.

### 6. Equilibration must be discarded, and the discard's value should be gated directly

Averaging the transient into the estimate biases the result toward wherever the run started. Starting 50x off target and discarding the transient improves the shift energy from 1.14e-02 to 2.19e-13 — a factor of 5e10.

Design rule:

- Gate the specific improvement ratio from discarding equilibration, not just the final answer — if the fixture ever starts equilibrating instantly, that gate should report itself as vacuous rather than silently passing for the wrong reason.

### 7. The timestep divergence boundary is only observable once the population is controlled

Three earlier attempts to show that too large a `dt` breaks the propagation all failed on false premises — the population does not collapse, the norm grows at every `dt` by design, and the converged shape overlaps the true ground state even at 5x the naive bound (renormalizing each iteration turns it into a power iteration whose dominant eigenvector stays the ground state). With the population controlled, the boundary becomes sharp: settles at target for `dt` in 0.10-0.26x the diagonal bound, diverges at 0.30-0.60x. This boundary sits below the propagator's true spectral limit (~0.44x the diagonal bound), so it detects the *controller* destabilizing, not the bare propagator's stability limit.

Design rule:

- Do not quote a controller-destabilization boundary as if it were the underlying propagator's stability bound — they are different quantities, and the controller's number is what a real run actually experiences.
- Isolate the controller's contribution by testing with the control terms zeroed out (`zeta = xi = 0`); if divergence still reports at every timestep, the transition being measured is a controller artifact, not evidence about the propagator itself.

### 8. Discretization is what makes sampling noise depend on population size, and it must be applied stochastically

`stochastic_round` existed but was never wired into the spawn step, leaving weights continuous, which makes the propagator scale-invariant: multiplying the population by k scales every spawn by k with relative noise unchanged. This was diagnosed by a suspiciously flat blocked error (4.2532e-02 identical to five significant figures across a 64x range of target populations).

Design rule:

- Round spawn weights to a fixed granularity *stochastically*, never to nearest — rounding to nearest systematically discards sub-walker spawns and biases the energy. Gate both that the rounding primitive is unbiased in isolation, and separately that the spawn step actually calls it (swapping in `std::round` was otherwise invisible to the existing gates).

### 9. The initiator approximation is a biased approximation and must always be reported with its threshold

A spawn onto an unoccupied determinant is kept only if the parent's weight exceeds `n_add`, judged against the *incoming* population (not a partially-built next population, which would make the rule depend on visit order). This suppresses the sign problem caused by low-weight walkers colonizing determinants that should stay empty, while leaving the established wavefunction free.

Design rule:

- Never report an initiator-approximation energy without stating the `n_add` threshold used — it is a biased estimate, not an exact one.
- Judge initiator occupancy against the population as it stood at the start of the step, not a version being mutated during the same step, to keep the rule order-independent.

## What was found

1. **The two-term shift update is required** — see invariants 1-3.
2. **Both estimators cross-validate to within 1e-9** across a 100x population range, with independence verified by deliberate perturbation — invariant 4.
3. **Blocked error bars are necessary**, cross-checked C++-vs-Python to 1e-10 relative — invariant 5.
4. **The timestep divergence boundary is measurable only under control**, and sits at 0.30-0.60x the diagonal bound, below the bare propagator's ~0.44x spectral limit — invariant 7.
5. **Stochastic rounding of spawns was missing and is now wired in and gated separately from the rounding primitive itself** — invariant 8.
6. **Two of this step's intended gates could not be built on the 36-determinant toy fixture, and both hit the same underlying cause: saturation.** The initiator's `n_add -> 0` convergence trend is not measurable — the behaviour is binary (below `n_add ~= 100` every error is within one blocked sigma of every other; above ~300 the run freezes with zero variance), because at 5.5 walkers per determinant the space fills within a few steps before the rule can meaningfully fire. The stochastic error's population trend had to be re-measured below saturation: with discretization in, error *rose* with population (3.87e-2 -> 5.92e-2) at 14-889 walkers per determinant, but is clean below ~1 walker per determinant (3.65e-1 -> 5.96e-2 over a 64x range). This is the saturation trap the research scope names when it rejects H2 and water/STO-3G as FCIQMC fixtures, reached from the inside: a space small enough to validate against exact diagonalization is often too small to exhibit the sampling behaviour being validated. The `n_add -> 0` convergence trend therefore belongs with the N2/STO-3G regression gate (F5.3, in `FCIQMC_DRIVER_AND_VALIDATION.md`), where 14400 determinants stay partially occupied at a realistic walker count.

## Validation strategy that should remain in place

- `planck-fciqmc-walkers` (`tests/fciqmc_walkers.cpp`), covering shift control tradeoffs, the dimensioned-parameter scaling gate, cross-estimator agreement, the equilibration-discard ratio, the controlled timestep-divergence boundary, and the stochastic-rounding wiring
- Cross-checking `blocked_standard_error` (C++) against `tests/blocking.py` (Python) whenever either is changed
- Re-deriving the N2/STO-3G gate's initiator convergence trend (F5.3) rather than attempting it on the saturated toy fixture

## Related but separate outcome: lessons that generalize beyond FCIQMC

- **A test can fail for a reason unrelated to the feature under test.** The initiator's order-dependence check failed, and investigation showed the *propagator* already has insertion-order dependence (hash-order iteration against a shared RNG) — unrelated to the initiator itself. Comparing against that control isolated the claim; asserting an absolute that was never true would have sent the next reader hunting a defect in the wrong place.
- **A saturated fixture hides whole classes of defect.** A rule blocking *all* spawns from low-weight parents (not just spawns to unoccupied determinants) was indistinguishable from the correct rule by energy alone, because the occupancy condition rarely fired on the toy fixture. It needed a direct semantics test that pre-occupies the space to expose the difference.
- **Statistical tests must be sized before they are trusted.** The spawn-bias test first used spawns of ~0.04 walkers, making rounding near-binary with ~100 nonzero events in 200k runs, which scattered 51% on correct code. Sizing spawns to straddle the granularity fixed it, and the same test then caught the intended mutation at 43%.
