#!/usr/bin/env python3
"""A deliberately trivial Monte-Carlo estimator (FCIQMC scope, G4).

This is NOT physics and must never grow into any. Its only job is to exercise
the full statistical pipeline end to end -- seeded RNG, a correlated trajectory,
the blocking analysis, the within-sigma gate and the reproducibility gate -- on a
quantity whose exact answer is known in closed form.

The population mirrors the statistical *structure* of an FCIQMC run without any
of its machinery: a fixed set of "determinants" each carrying a weight, sampled
with a Markov chain so the trajectory is autocorrelated the way a real one is.
The exact answer is the plain weighted mean.
"""
from __future__ import annotations

import random
from typing import Sequence


def make_population(n: int, seed: int = 12345) -> list[float]:
    """A fixed synthetic 'H_ii'-like spectrum. Deterministic given the seed.

    The values TREND with index rather than being i.i.d., and that is
    load-bearing: with an i.i.d. population every sub-range has the same mean, so
    a sampler restricted to part of the space produces a bias of ~0.5 sigma and
    slides through a 3-sigma gate. Measured, before this was fixed: restricting
    the sampler to the first half shifted the mean by only 0.58 sigma and the G4
    mutation test came back GREEN on a genuinely biased sampler.

    A monotone trend makes any incomplete coverage of the space show up as a
    mean shift, which is what lets G4 detect the p_gen failure class it exists
    for. Real H_ii values are likewise not exchangeable across the determinant
    space -- low-energy determinants cluster -- so this is closer to the target
    than i.i.d. was, not merely more convenient.
    """
    rng = random.Random(seed)
    # Spread of the trend is ~4x the noise, so half-coverage is many sigma out.
    return [-14.0 + 8.0 * (i / max(1, n - 1)) + rng.gauss(0.0, 0.5) for i in range(n)]


def exact_mean(population: Sequence[float]) -> float:
    return sum(population) / len(population)


def sample_trajectory(
    population: Sequence[float],
    n_steps: int,
    seed: int,
    stay_prob: float = 0.9,
) -> list[float]:
    """A correlated Markov-chain sample of the population mean.

    At each step the walker either stays (with `stay_prob`) or jumps to a
    uniformly-chosen element. Staying is what makes the series autocorrelated --
    the same property a real trajectory has, and the reason the naive standard
    error would understate sigma here exactly as it does there.

    The chain is uniform over the population by construction (the jump target is
    uniform and staying does not bias which state is occupied), so the trajectory
    mean is an UNBIASED estimator of exact_mean(). That is what makes a failure
    of the 3-sigma gate meaningful rather than expected.
    """
    rng = random.Random(seed)
    n = len(population)
    idx = rng.randrange(n)
    out = []
    for _ in range(n_steps):
        if rng.random() >= stay_prob:
            idx = rng.randrange(n)
        out.append(population[idx])
    return out
