# Scope: F2 — the excitation generator and `p_gen`

**Scope for in-flight work. Not started.** Step F2 of the implementation ladder in
`FCIQMC_RESEARCH_SCOPE.md`; F1 (walker container, RNG policy) is landed.

## Why this step gets its own scope

Every other step in the ladder fails loudly. This one fails **silently**.

An excitation generator whose returned probability disagrees with its actual
sampling distribution produces a **plausible, converged, wrong energy** — the
walkers still evolve, the population still stabilises, the shift still settles, and
the number is simply not the FCI answer. There is no crash, no non-convergence, no
NaN. This codebase has hit that failure class twice already (the spin-adapt default
emitting a 4x-wrong correlation energy while every gate stayed green; the invalid
ERI symmetry table reaching the emitted C++ with a bogus sign), and both times the
cost was measured in days of investigation.

**So the gate is built before the generator is used, not after**, and the gate is
brute-force enumeration rather than a self-consistency check.

## What the generator must do

Given a determinant `D`, draw a connected determinant `D'` and return the
probability that *this* call would have produced *that* `D'`.

```
struct Excitation {
    DetKey  det;        // the generated D'
    double  p_gen;      // probability this draw produced it
    double  phase;      // fermionic sign, from apply_annihilation/apply_creation
    bool    valid;      // false if the draw hit a dead end (see D3)
};
```

`p_gen` enters the FCIQMC spawn as `|H_ij| / p_gen`, so an over-reported `p_gen`
suppresses those spawns and an under-reported one inflates them. Neither is
detectable from the trajectory alone.

### The connection structure it samples over

Only single and double excitations connect determinants — a Slater-Condon element
vanishes beyond a double — so the space is:

| system | n_act | nα/nβ | singles | same-spin doubles | opposite-spin doubles | total |
|---|---|---|---|---|---|---|
| H2/STO-3G | 2 | 1/1 | 2 | 0 | 1 | **3** |
| water/STO-3G | 7 | 5/5 | 20 | 20 | 100 | 140 |
| N2/STO-3G | 10 | 7/7 | 42 | 126 | 441 | 609 |
| Cr₂ CAS(12,18) | 18 | 6/6 | 144 | 1980 | 5184 | 7308 |

**H2/STO-3G having 3 connections is what makes this step gateable**: the oracle can
enumerate the entire connection set exactly and compare against sampled
frequencies with no statistical argument at all.

### `p_gen` is not uniform, and that is fine

A natural two-stage generator — pick single-vs-double, then pick uniformly within
the class — gives:

| system | p(a given single) | p(a given double) | ratio |
|---|---|---|---|
| water/STO-3G | 0.025000 | 0.004167 | **6.0x** |
| N2/STO-3G | 0.011905 | 0.000882 | **13.5x** |

**Non-uniformity is not the bug.** The bug is a returned `p_gen` that does not match
the sampler's actual distribution. A generator may be as biased as it likes toward
cheap or important excitations — that is what weighted generators are *for* — so
long as it reports honestly. **F2's gate therefore tests agreement, never
uniformity.**

## Steps

Ordered so the cheapest kills the expensive ones. Each is independently verifiable.

### F2.1 — the brute-force oracle (no generator yet)

Enumerate **all** singles and doubles from a determinant, by direct nested loops
over occupied/virtual pairs, returning the full connection set with phases. Slow
and obviously correct; it exists only to check the sampler.

- **Verify:** on H2/STO-3G the oracle returns exactly the 3 connections in the
  table; on water/STO-3G exactly 140. Cross-check against the *independent*
  enumeration already in `apply_ci_hamiltonian` (`ci.cpp`) — every determinant the
  oracle produces must have a nonzero `slater_condon_element` with `D`, and every
  determinant the sigma build touches from `D` must appear in the oracle's set.
- **Why this first:** it is the measuring instrument. Building it before the
  generator means the generator is never the only implementation of "what is
  connected to what".

### F2.2 — a uniform generator over the full connection set

The simplest correct sampler: enumerate the connection set, draw one uniformly,
return `p_gen = 1/|connections|`. **Deliberately the slow, obviously-correct
version** — it enumerates on every call, which is exactly what production must not
do, and that is the point: F2.3 must reproduce its distribution while being fast.

- **Verify:** sampled frequency of each connection matches `p_gen` (chi-squared, or
  simply that every connection appears within a few σ of `N·p_gen` on H2 and water);
  `sum(p_gen)` over the enumerated set is 1.0 to machine precision.

### F2.3 — the production generator (no enumeration per call)

Pick a class (single / same-spin double / opposite-spin double), then pick orbitals
within it by direct index arithmetic — O(1) per draw, no enumeration.

- **Verify — and this is the load-bearing gate:** for every determinant in a small
  space, sample `N` times and check the **empirical frequency of each connection
  matches its returned `p_gen`** to within sampling error. Not that `p_gen` is
  uniform; that it is *correct*. Run it on both H2 (3 connections, exact) and water
  (140, statistical).
- **Also verify:** the connection *support* is identical to F2.1's oracle — nothing
  generated that is not connected, nothing connected that is never generated. A
  generator that silently cannot reach some excitations is the subtler half of this
  bug class.

  **Quantified while building F2.2, because the first attempt to demonstrate this
  failed.** For a *uniform* generator the support check is partly redundant: a
  support hole redistributes probability, so the frequency test catches it too
  (measured — dropping 1 of 140 connections showed up at **54σ**). The
  independence appears only once `p_gen` is **non-uniform**: a connection with
  `p_gen ≈ 1e-6` that is never generated deviates by **~0.6σ** over 400k draws,
  which no frequency test will flag. **So the support check is load-bearing at
  F2.3 specifically**, and a mutation test at F2.2 cannot demonstrate it — do not
  conclude from an F2.2 run that the two checks are interchangeable.

### F2.4 — mutation-verify the gate itself

The gate must be shown to fail before it is trusted. Inject, one at a time:

1. **A `p_gen` off by a constant factor** (e.g. return `p_gen/2`) — the frequency
   comparison must go red.
2. **A restricted generator** that never produces opposite-spin doubles — the
   support comparison must go red. (Note this is the F2.3 *"also verify"* item;
   a frequency-only gate passes this mutation, which is why both are required.)
3. **A class-probability mismatch** — draw singles with probability 0.3 but report
   `p_gen` as though it were 0.5.

Each must be caught by a **named** assertion, not merely by "something failed".

### F2.5 — spin and symmetry constraints

The generator must respect what the determinant space actually contains: particle
number per spin channel, and — when the CI space is symmetry-restricted — the
target irrep.

- **Verify:** every generated determinant has the same α and β popcount as its
  parent; on a symmetry-restricted space, every generated determinant is in the
  space (`det_lookup` finds it, or the RAS/irrep filter accepts it). A generator
  that produces out-of-space determinants is not automatically wrong — the spawn can
  discard them — **but it must then be reflected in `p_gen`**, and discarding
  silently while reporting the unrestricted `p_gen` is precisely the bias this scope
  exists to prevent.

**A trap found while building this, which nearly shipped.** The natural correction
is to divide `p_gen` by the acceptance rate, and the natural way to get that rate
is the attempt count of the call that just succeeded. **That estimator is biased in
the quantity that matters, and every obvious check of it passes.**

`E[p_gen × attempts]` is exactly the conditional probability — the estimator is
unbiased *for `p_gen`*. But the spawn uses `|H_ij| / p_gen`, and `E[1/X] ≠ 1/E[X]`
(Jensen). Measured at `p_accept = 0.3`: the mean of `p_gen` is correct to 0.1 %,
and the mean of `1/p_gen` is **1.72x too large** — every spawn out of a restricted
space over-weighted, with `p_gen`'s own mean looking perfect.

The fix is to measure the acceptance rate **once, separately** (`measure_acceptance_rate`)
and pass it in as a constant, so the correction is not a random variable. A
regression test pins both halves — that the per-call estimator *is* unbiased for
`p_gen`, and that it *is* biased in `1/p_gen` — so nobody simplifies the separate
measurement away on the grounds that the obvious check passes.

**Generalizable: when a sampled quantity is used as a divisor, unbiasedness of the
estimator is the wrong property to check.** Check the estimator of the quantity the
consumer actually computes.

## What this must not do

- **Do not gate `p_gen` by self-consistency.** Checking that the generator's own
  reported probabilities sum to 1 tests arithmetic, not correctness: a generator
  that samples the wrong distribution but reports it consistently passes. The gate
  must compare against **measured frequencies** and an **independent enumeration**.
- **Do not test only on H2.** 3 connections cannot exercise doubles between
  different occupied pairs, which is where the interesting bugs live. Water/STO-3G
  (140 connections, 5α/5β) is the minimum for that, and N2/STO-3G is the fixture the
  research scope names for the same reason.
- **Do not test only frequencies.** Support and frequency are different failure
  modes; mutation 2 above passes a frequency-only gate.
- **Do not optimize F2.3 before F2.2 exists.** The slow uniform generator is the
  reference distribution the fast one must reproduce. Writing only the fast one
  leaves nothing to compare against but the oracle's support.
- **Do not fold the fermionic phase into `p_gen`.** They are different quantities —
  `phase` is a sign from operator ordering, `p_gen` a probability. Conflating them
  produces sign errors that look like sampling bias.

## Key code locations

| what | where |
|---|---|
| walker state and RNG (F1, landed) | `src/post_hf/ci/fciqmc.{h,cpp}` |
| fermionic operators, with phase | `apply_annihilation` / `apply_creation`, `strings.h:94-95` |
| the independent enumeration to cross-check against | `apply_ci_hamiltonian`, `src/post_hf/ci/ci.cpp` |
| matrix elements, for the "is it connected" test | `slater_condon_element`, `ci.h:43` |
| the statistical gate machinery | `tests/{blocking,reproducibility}.py`, `metric_within_sigma` |
| smallest exact fixture (3 connections) | `tests/inputs/regression/post_hf/h2_fci_sto3g.hfinp` |
| the fixture with real doubles structure | `water_fci_sto3g`, and N2 under `exploratory/fciqmc/` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
