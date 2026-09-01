# How FCIQMC's sampling and dynamics work, and how they are kept honest

This answers: **what does an FCIQMC iteration actually compute, and how do you
know it is right when a wrong answer looks exactly like a right one?**

It covers the excitation generator (`p_gen`) and the propagator (spawn, death,
annihilation) — steps F2 and F3 of the ladder in `FCIQMC_RESEARCH_SCOPE.md`. Both
are landed and gated by `planck-fciqmc-walkers` (~11 s). Code lives in
`src/post_hf/ci/fciqmc.{h,cpp}`.

## Why this layer needs unusual care

Every other part of this codebase fails loudly. This one fails **silently**.

A generator whose reported `p_gen` disagrees with its actual sampling distribution
produces a *plausible, converged, wrong* energy: walkers evolve, the population
stabilises, the shift settles, and the number is simply not the FCI answer. No
crash, no non-convergence, no NaN. Planck has been bitten by that class twice —
the spin-adapt default emitting a 4x-wrong correlation energy with every gate
green, and an invalid ERI symmetry table reaching the emitted C++ with a bogus
sign.

So the whole design is built around one rule: **every layer is checked against an
independent implementation, never against itself.**

## 1. The excitation generator

### What it must produce

Given a determinant `D`, draw a connected `D'` and report the probability that
*this call* would have produced *that* `D'`. The spawn then divides by it, so a
mis-reported `p_gen` biases the energy directly.

Only single and double excitations connect determinants — the Hamiltonian is a
two-body operator, so it cannot connect determinants differing in more than two
orbitals. **That fact is the entire sparsity the method rests on.**

| system | n_act | nα/nβ | connections |
|---|---|---|---|
| H2/STO-3G | 2 | 1/1 | **3** |
| water/STO-3G | 7 | 5/5 | 140 |
| N2/STO-3G | 10 | 7/7 | 609 |
| Cr₂ CAS(12,18) | 18 | 6/6 | 7308 |

Counts verified by independent brute-force enumeration, not from the combinatorial
formula. **H2 having only 3 connections is what makes the layer gateable**: the
oracle enumerates the full set exactly, with no statistical argument.

### The three implementations, and why there are three

**`enumerate_connections`** — the brute-force oracle. Nested loops over
occupied/virtual pairs, no index arithmetic to get wrong. Deliberately slow. It
exists so the generator is **never the only implementation of "what is connected
to what"**. `p_gen` is left at zero here: enumeration is not sampling, and filling
it in would invite exactly the self-consistency check that proves nothing.

**`draw_uniform_excitation`** — enumerates per call, picks uniformly,
`p_gen = 1/|connections|`. Obviously correct, and useless in production (Cr₂ would
enumerate 7308 connections per walker per iteration). It is the *reference
distribution* the fast generator must reproduce.

**`draw_excitation`** — the production generator. Picks one of five classes (α
single, β single, αα, ββ, αβ double) with equal probability among the non-empty
ones, then indexes within it. O(1): cost independent of connection count.

### `p_gen` is non-uniform, and that is correct

Equal class probability over unequal class sizes means

```
p_gen = (1 / n_live_classes) × (1 / class_size)
```

which varies **10x on water/STO-3G and 21x on N2/STO-3G**. Non-uniformity is not
the bug. A weighted generator deliberately biases toward cheap or important
excitations; what matters is that the *reported* `p_gen` matches the sampler's
actual distribution. **The gate therefore tests agreement, never uniformity.**

### Support and frequency are different failure modes

The gate checks both:

- **frequency** — each connection appears `N × p_gen` times, within 5σ;
- **support** — the reachable set equals the oracle's exactly.

**Their independence appears only once `p_gen` is non-uniform**, which was
measured rather than assumed. For a *uniform* generator a support hole
redistributes probability and the frequency test catches it too (dropping 1 of 140
connections showed at **54σ**). For a weighted one, a connection with
`p_gen ≈ 1e-6` that is never generated deviates by **~0.6σ** — invisible to any
frequency test. So the support check is load-bearing at the production generator
specifically.

### Restricted spaces: rejection sampling, and a trap that nearly shipped

Particle number per spin is conserved *structurally* — the generator annihilates
and creates once per channel — so it needs proof, not a filter. Symmetry is
different: a restricted CI space means some draws land outside it, and discarding
them while reporting the unrestricted `p_gen` silently suppresses every spawn out
of that space.

`draw_excitation_in_space` does rejection sampling and divides `p_gen` by the
acceptance rate. **The acceptance rate must be measured separately and passed in
as a constant** (`measure_acceptance_rate`), never estimated from the attempt count
of the call that just succeeded.

That distinction is a **1.72x bias**, measured. `E[p_gen × attempts]` is exactly
the conditional probability — the per-call estimator is *unbiased for `p_gen`*,
and every obvious check of it passes. But the spawn uses `|H_ij| / p_gen`, and
`E[1/X] ≠ 1/E[X]`. At `p_accept = 0.3`: mean of `p_gen` correct to 0.1 %, mean of
`1/p_gen` **1.72x too large**.

> **When a sampled quantity is used as a DIVISOR, unbiasedness of the estimator is
> the wrong property to check.** Check the estimator of the quantity the consumer
> actually computes.

A regression test pins both halves, so nobody simplifies the separate measurement
away on the grounds that the obvious check passes.

## 2. The propagator

Each iteration applies `(1 − dt(H − S))` to the population:

```
SPAWN       child weight = -dt * H_ij * c_i / p_gen(i->j)
DEATH       c_i *= (1 - dt * (H_ii - S))
ANNIHILATE  accumulate children into the population
```

**Annihilation is not a separate pass.** It is what accumulating signed weights
into a determinant-keyed map already does — adding `-w` to a determinant holding
`+w` cancels it. That is also why `WalkerPopulation` is a map rather than a walker
list, and why `compress()` earns its place: annihilation leaves exact zeros
behind, and every determinant ever touched by one spawn would otherwise stay
resident forever.

### Deterministic first, then stochastic

`propagate_deterministic` visits every connection via the oracle, so one call is
exactly a matrix-vector product and is checked against a hand-computed matvec to
**1e-12**. This separates *the dynamics are wrong* from *the sampling is wrong*
before any statistics enter — if it does not match exactly, nothing above it is
worth debugging.

`propagate_stochastic` then draws connections and reweights by `1/p_gen`. Its
**mean** must reproduce the deterministic result, which it does within 5σ per
component. That is a mean of a *linear* quantity, carrying no ratio bias, which is
what makes it a clean test of the division.

The death step stays deterministic in both: one diagonal element per determinant,
so sampling it would add variance for nothing.

### The projected energy is biased at finite population

`E = H_00 + Σ_j H_0j c_j / c_0` is a ratio of stochastic quantities, so
`E[A/B] ≠ E[A]/E[B]`. The bias falls roughly as 1/N:

| N | bias | bias × N |
|---|---|---|
| 800 | 4.75e-3 | 3.80 |
| 3 200 | 8.80e-4 | 2.82 |
| 12 800 | 1.86e-4 | 2.38 |
| 51 200 | 2.27e-4 | (resolution-limited) |

This is a published property of the estimator, not a defect — **which means a
small-population result agreeing to 1e-6 is suspicious, not reassuring.** The gate
asserts the *trend*, never a single value. The bias is **negative** here, the
direction that makes a result look more convincingly variational than it is.

`projected_energy` reports `valid = false` when `|c_0|` is small, rather than
returning noise-over-noise the caller cannot distinguish from a good number.

### The timestep bound is weaker than it looks

`dt < 2/max|H_ii − S|` is **necessary but not sufficient**. True stability needs
the spectral radius of `H − S`, to which off-diagonals contribute:

| bound | value on the 36-determinant test H |
|---|---|
| diagonal only | 0.5714 |
| true spectral | **0.2509** |

The code keeps the diagonal form because the spectral radius of a
3×10⁸-determinant Hamiltonian is not computable — but it is documented as a limit
to stay well under, not a safe target. It has a second edge: it is computed from
the **currently occupied** determinants while propagation spreads beyond them, so
a run seeded with one reference determinant sees one diagonal element and returns
**infinity** when that element equals the shift.

**There is deliberately no "too-large `dt` diverges" gate here.** Three
formulations were tried and each rested on a false premise, all measured: the
population does not collapse onto one determinant (max component 0.0716 from 1.5x
to 5x the bound); the norm does not diverge only above the bound (it grows at
*every* `dt`, since a shift below the ground-state energy produces exponential
growth by design); and the converged shape is not the wrong state (overlap
**1.000000** at every `dt` from 0.05x to 5x). Renormalizing each iteration turns
this into a power iteration whose dominant eigenvector stays the ground state.
**That gate belongs with population control**, where an unstable `dt` has
somewhere to show. Constructing a Hamiltonian contrived to fail would test the
fixture, not the code.

### Reproducibility is gated on whole trajectories

Same seed → bitwise-identical trajectory; different seed → different trajectory.
The digest runs over raw IEEE bits in determinant order and distinguishes a
**one-ulp** difference, so "identical" means identical rather than "similar
enough".

A *trajectory* rather than a step, because single-step checks cannot see a defect
that accumulates — each step feeds the next, so carried state drifts in over many
iterations while any single step still matches.

**The negative control is load-bearing, and it took four mutations to demonstrate
that.** Three were caught by the statistical gates instead (resetting the RNG per
call destroys the variance those measure, so it never reached the reproducibility
check). The case that isolates the control is an RNG that **advances normally
within a run but ignores its seed**: means, variance and `1/n_attempts` scaling all
correct, and only "different seeds must give different trajectories" fails.

## 3. Lessons that generalize beyond this layer

**Gate tolerances must be derived from the measurement, not chosen.** Every
tolerance picked by hand in this work was wrong; every one derived was right.

- An **absolute** tolerance was *vacuous*: 0.02 sat at the size of the effect, so
  dropping `1/p_gen` entirely — the defect the test existed to catch — **passed**.
  Spawn magnitudes span 0.005 to 0.4 across excitation classes.
- A **relative** tolerance was *noise-dominated*, rejecting correct code at 51 %
  because small components carry large fractional error.
- The **standard error** is the only scale correct for every component at once.
  With it, the two previously-passing mutations are caught at **5553σ** and
  **226σ**.
- The same wall from the other side: an apparent bias below the measurement's own
  resolution is noise, not a plateau. Asserting a tighter constant would assert
  noise.

**A fixture must share the structure whose violation you intend to detect.**
- Too *structureless*: an i.i.d. population hid a real sampler bias at 0.58σ; a
  trending one exposes the same mutation at 25.9σ.
- Too *general*: a synthetic Hamiltonian that filled every matrix entry is not a
  Hamiltonian — a real one is zero beyond a double excitation. 9 of 35 pairs are
  unconnected at `n_act=4`, so the reference matvec summed contributions the
  propagator correctly skipped, and the *test* was wrong.
- Too *symmetric*: every fixture was closed-shell, so an index bug that only
  manifests when α and β counts differ had zero coverage. An **equivalent mutant**
  — a relabeling that legitimately passed — is what exposed it.

**Check the vacuity of the check.** Several tests here needed a companion
assertion that the fixture can actually fail: that the restriction actually
restricts, that the test values actually reassociate, that the machinery still
*accepts* the correct implementation.

## Key code locations

| what | where |
|---|---|
| walker state, RNG, generator, propagator | `src/post_hf/ci/fciqmc.{h,cpp}` |
| the gate | `tests/fciqmc_walkers.cpp` (`planck-fciqmc-walkers`) |
| `H_ij` for the spawn | `slater_condon_element`, `ci.h:43` |
| `H_ii` for the death step | `build_ci_diagonal`, `ci.h:101` |
| statistical gate machinery | `tests/{blocking,reproducibility}.py`, `metric_within_sigma` |
| exact reference | `h2_fci_sto3g`, total FCI `-1.1372744062` |
| the target case | `tests/inputs/exploratory/fciqmc/cr2_casscf_target.hfinp` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
