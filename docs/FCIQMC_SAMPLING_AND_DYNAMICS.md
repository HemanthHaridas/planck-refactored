# FCIQMC Excitation Generator and Propagator

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**What does an FCIQMC iteration actually compute, and how do you know it is right when a wrong answer looks exactly like a right one?**

## Short answer

This layer needs unusual care because it fails silently: a generator whose reported `p_gen` disagrees with its actual sampling distribution produces a plausible, converged, wrong energy — walkers evolve, the population stabilizes, the shift settles, and the number is simply not the FCI answer, with no crash, no non-convergence, no NaN. The whole design is built around one rule: every layer is checked against an independent implementation, never against itself. This covers the excitation generator (`p_gen`) and the propagator (spawn, death, annihilation) — steps F2 and F3 of the ladder in `FCIQMC_RESEARCH_SCOPE.md`. Both are landed and gated by `planck-fciqmc-walkers` (~11 s). Code lives in `src/post_hf/ci/fciqmc.{h,cpp}`.

## Where the logic lives

- `src/post_hf/ci/fciqmc.{h,cpp}` — walker state, RNG, generator, propagator
- `tests/fciqmc_walkers.cpp` (`planck-fciqmc-walkers`) — the gate
- `slater_condon_element`, `ci.h:43` — `H_ij` for the spawn
- `build_ci_diagonal`, `ci.h:101` — `H_ii` for the death step
- `tests/{blocking,reproducibility}.py`, `metric_within_sigma` — statistical gate machinery
- `h2_fci_sto3g` — exact reference, total FCI `-1.1372744062`
- `tests/inputs/exploratory/fciqmc/cr2_casscf_target.hfinp` — the target case

## What invariants matter

### 1. Only single and double excitations connect determinants, and this sparsity is the entire method

The Hamiltonian is a two-body operator, so it cannot connect determinants differing in more than two orbitals. This fact is the entire sparsity the method rests on. Connection counts (verified by independent brute-force enumeration, not the combinatorial formula): H2/STO-3G 3, water/STO-3G 140, N2/STO-3G 609, Cr2 CAS(12,18) 7308.

Design rule:

- H2's exactly-3-connections property is what makes the excitation-generator layer gateable at all: the brute-force oracle can enumerate the full connection set exactly, with no statistical argument needed.

### 2. Every layer is validated against an independent implementation, never against itself

Three separate implementations exist for the same question ("what is connected to what, and with what probability"), each serving a distinct role: `enumerate_connections` is the brute-force oracle (nested loops, no index arithmetic to get wrong, deliberately slow, `p_gen` left at zero since enumeration is not sampling); `draw_uniform_excitation` enumerates per call and picks uniformly (`p_gen = 1/|connections|`, obviously correct but useless in production — Cr2 would enumerate 7308 connections per walker per iteration) and serves as the reference distribution the fast generator must reproduce; `draw_excitation` is the O(1) production generator.

Design rule:

- Never validate a fast/production implementation against itself. Build a slow, obviously-correct reference implementation first, and check the production path against it — the reference's simplicity (not its speed) is what makes it trustworthy.

### 3. Non-uniform `p_gen` is correct; the gate must test agreement, never uniformity

`draw_excitation` picks one of five classes (alpha single, beta single, alpha-alpha, beta-beta, alpha-beta double) with equal probability among the non-empty ones, then indexes within it — giving `p_gen = (1/n_live_classes) * (1/class_size)`, which varies 10x on water/STO-3G and 21x on N2/STO-3G. Non-uniformity is not the bug; what matters is that the *reported* `p_gen` matches the sampler's actual distribution.

Design rule:

- Test that a weighted sampler's reported probability agrees with its actual sampling frequency — never test for uniformity, which a correct weighted generator is not supposed to have.

### 4. Support and frequency are independent failure modes, and their independence only appears once `p_gen` is non-uniform

The gate checks both that each connection appears `N * p_gen` times within 5 sigma (frequency), and that the reachable set equals the oracle's exactly (support). For a uniform generator, a support hole redistributes probability and the frequency test alone catches it too (measured: dropping 1 of 140 connections showed at 54 sigma). For a weighted generator, a connection with `p_gen ~= 1e-6` that is never generated deviates by only ~0.6 sigma — invisible to any frequency test.

Design rule:

- On a non-uniform (weighted) sampler, a frequency-only gate is insufficient — a dedicated support check (does the reachable set match the oracle's exactly) is load-bearing and must be kept, not treated as redundant with the frequency check.

### 5. When a sampled quantity is used as a divisor, unbiasedness of the estimator is the wrong property to check

Restricted-space sampling (`draw_excitation_in_space`) uses rejection sampling and divides `p_gen` by the acceptance rate, which must be measured separately and passed in as a constant (`measure_acceptance_rate`), never estimated from the attempt count of the call that just succeeded. `E[p_gen * attempts]` is exactly the conditional probability — the per-call estimator is unbiased for `p_gen`, and every obvious check of it passes — but the spawn uses `|H_ij| / p_gen`, and `E[1/X] != 1/E[X]`. Measured at `p_accept = 0.3`: mean of `p_gen` correct to 0.1%, mean of `1/p_gen` 1.72x too large.

Design rule:

- When a sampled quantity feeds a division downstream, check the estimator of the specific quantity the consumer actually computes (here, `1/p_gen`), not the estimator of the sampled quantity itself (`p_gen`) — unbiasedness does not transfer across a nonlinear transform.
- Pin both halves (the unbiased `p_gen` estimator and the biased `1/p_gen` consumption) in one regression test, so the separate measurement of the acceptance rate cannot be simplified away on the grounds that the obvious check passes.

### 6. Annihilation is not a separate pass — it is what a signed, determinant-keyed accumulator already does

Adding `-w` to a determinant holding `+w` cancels it automatically inside `WalkerPopulation`, which is why the container is a map rather than a walker list, and why `compress()` (removing exact zeros left behind by annihilation) earns its place — without it, every determinant ever touched by one spawn would stay resident forever.

Design rule:

- Design the walker container's data structure so the required physics (annihilation) falls out of the storage choice itself, rather than being implemented as an explicit separate step that could disagree with the storage.

### 7. Validate the deterministic dynamics before any statistics enter

`propagate_deterministic` visits every connection via the oracle, so one call is exactly a matrix-vector product, checked against a hand-computed matvec to 1e-12. This separates "the dynamics are wrong" from "the sampling is wrong" before any statistical argument is needed — if the deterministic path does not match exactly, nothing built on top of it is worth debugging. `propagate_stochastic` then draws connections and reweights by `1/p_gen`; its mean must reproduce the deterministic result (verified within 5 sigma per component), which is a clean test of the division because it is a mean of a linear quantity carrying no ratio bias.

Design rule:

- Build and gate a deterministic (exhaustive, oracle-driven) version of any stochastic dynamics first, exactly against a hand-computed reference, before adding sampling on top.
- Keep any step with no meaningful variance to sample (here, the diagonal death step, one element per determinant) deterministic in both the deterministic and stochastic propagators — sampling it would add variance for nothing.

### 8. The projected energy is a biased ratio estimator, and the bias must be gated as a trend, not a single value

`E = H_00 + sum_j H_0j c_j / c_0` is a ratio of stochastic quantities, so `E[A/B] != E[A]/E[B]`. The bias falls roughly as 1/N (measured: `bias * N` roughly 3.80, 2.82, 2.38 at N = 800, 3200, 12800, flattening into measurement resolution by N = 51200) and is negative here — the direction that makes a result look more convincingly variational than it actually is.

Design rule:

- A small-population result agreeing suspiciously well (e.g., to 1e-6) with the exact answer is a reason for suspicion, not reassurance, given this estimator's known bias direction — gate the bias's 1/N trend across multiple population sizes, never a single value at one population.
- Report the projected energy as invalid (rather than a noise-over-noise number the caller cannot distinguish from a good one) whenever the reference weight `|c_0|` is too small to trust.

### 9. The diagonal timestep bound is necessary but not sufficient, and must not be mistaken for the true stability limit

`dt < 2/max|H_ii - S|` uses only the diagonal of `H - S`; true stability needs the spectral radius, to which off-diagonals contribute substantially (measured on a 36-determinant test Hamiltonian: diagonal-only bound 0.5714, true spectral bound 0.2509). The code keeps the diagonal form because the spectral radius of a 3e8-determinant Hamiltonian is not computable, but it must be documented as a limit to stay well under, not a safe target. It also has a second edge: it is computed from the currently-occupied determinants, so a run seeded with a single reference determinant can return infinity when that determinant's diagonal element equals the shift.

Design rule:

- Never treat the diagonal timestep bound as a tight stability guarantee — document it explicitly as a conservative limit to stay well under.
- Do not force a "too-large dt diverges" gate onto a fixture that does not actually exhibit the failure — three formulations here were each tried and measured to rest on a false premise (population does not collapse; norm grows at every dt by design; converged shape overlaps the true ground state at every dt tested). A gate belongs wherever the property is actually observable (here, under population control, in `FCIQMC_POPULATION_CONTROL.md`) rather than being contrived onto an unrelated layer.

### 10. Reproducibility must be gated on whole trajectories, and the negative control needs multiple mutation attempts to prove it is load-bearing

Same seed must give a bitwise-identical trajectory; different seed must give a different one, checked by digesting raw IEEE bits in determinant order (distinguishing a one-ulp difference) over an entire trajectory rather than a single step, since a defect that accumulates across steps would still match at any single step in isolation.

Design rule:

- Gate reproducibility on a full trajectory, not a single step, so state that drifts in over many iterations is actually caught.
- Verify that a negative control is actually catching something by trying multiple mutations against it — three plausible mutations here were instead caught by the statistical gates (destroying the variance those measure before reaching the reproducibility check), and it took a fourth mutation (an RNG that advances normally within a run but ignores its seed) to isolate what the reproducibility control specifically catches.

## What was found

1. **The excitation generator's `p_gen` is non-uniform by design and validated for agreement, not uniformity** — see invariants 3-4.
2. **The restricted-space acceptance-rate bias (1.72x) was found and fixed** by requiring the acceptance rate to be measured once and passed as a constant, never re-estimated per call — see invariant 5.
3. **The deterministic-then-stochastic validation order caught nothing wrong in the dynamics themselves** — `propagate_deterministic` matches a hand matvec to 1e-12, and `propagate_stochastic`'s mean matches it within 5 sigma per component.
4. **The projected-energy bias was characterized as a 1/N trend**, not a defect to eliminate, and the estimator now reports invalid rather than a misleadingly precise number at small `|c_0|` — see invariant 8.
5. **No "too-large dt diverges" gate exists at this layer, by a measured decision** — three attempts each rested on a false premise (see invariant 9); that property is gated one layer up, where it is actually observable.
6. **Reproducibility is gated on full trajectories with a mutation-verified negative control** — see invariant 10.

## Validation strategy that should remain in place

- `planck-fciqmc-walkers`, covering the excitation generator's frequency and support checks, the restricted-space acceptance-rate bias regression, the deterministic-vs-matvec and stochastic-vs-deterministic-mean checks, the projected-energy bias trend across population sizes, and whole-trajectory reproducibility with its negative control
- Brute-force `enumerate_connections` kept as the permanent independent oracle, never replaced by a self-consistency check against the production generator
- Deriving gate tolerances from the measurement (see the generalizable lessons below), never picking a round-number tolerance by hand

## Related but separate outcome: lessons on gate tolerance and fixture design that generalize beyond this layer

**Gate tolerances must be derived from the measurement, not chosen.** Every tolerance picked by hand in this work was wrong; every one derived was right. An absolute tolerance of 0.02 was vacuous — it sat at the size of the effect, so dropping `1/p_gen` entirely (the defect the test existed to catch) passed, since spawn magnitudes span 0.005 to 0.4 across excitation classes. A relative tolerance was noise-dominated, rejecting correct code at 51% because small components carry large fractional error. The standard error is the only scale correct for every component at once — with it, the two previously-passing mutations are caught at 5553 sigma and 226 sigma. The same wall applies from the other side: an apparent bias below the measurement's own resolution is noise, not a plateau, and asserting a tighter constant there would assert noise.

**A fixture must share the structure whose violation you intend to detect.** Too structureless: an i.i.d. population hid a real sampler bias at 0.58 sigma, while a trending population exposes the same mutation at 25.9 sigma. Too general: a synthetic Hamiltonian that filled every matrix entry is not a Hamiltonian — a real one is zero beyond a double excitation (9 of 35 pairs are unconnected at `n_act=4`), so a reference matvec built from a fully-dense synthetic Hamiltonian summed contributions the propagator correctly skipped, meaning the *test* was wrong, not the code. Too symmetric: every fixture was closed-shell, so an index bug that only manifests when alpha and beta counts differ had zero coverage — an equivalent mutant (a relabeling that legitimately passed) is what exposed the gap.

**Check the vacuity of the check.** Several tests here needed a companion assertion that the fixture can actually fail: that a restriction actually restricts, that the test values actually reassociate, that the machinery still accepts the correct implementation.
