#!/usr/bin/env python3
"""Flyvbjerg-Petersen blocking analysis for correlated time series.

A naive standard error, sigma/sqrt(N), assumes independent samples. A Monte
Carlo trajectory is autocorrelated, so the naive estimate UNDERSTATES the true
uncertainty -- by sqrt(2*tau_int) for an AR(1) series. Understating sigma makes
every downstream statistical gate pass, which is the failure mode that matters:
see docs/FCIQMC_RESEARCH_SCOPE.md G2.

Blocking repeatedly halves the series by averaging adjacent pairs. Each
transformation leaves the mean unchanged but decorrelates the samples; the
blocked standard error rises with block size and plateaus once the block length
exceeds the correlation time. The plateau is the honest error bar.
"""
from __future__ import annotations

import math


def naive_standard_error(series: list[float]) -> float:
    """sigma/sqrt(N) -- correct only for independent samples."""
    n = len(series)
    if n < 2:
        return float("nan")
    mean = sum(series) / n
    var = sum((x - mean) ** 2 for x in series) / (n - 1)
    return math.sqrt(var / n)


def blocking_curve(series: list[float]) -> list[tuple[int, float, int]]:
    """Return [(block_size, standard_error, n_blocks)] over successive halvings.

    Stops when fewer than 4 blocks remain: a standard error computed from 2 or 3
    samples is itself so noisy that the plateau cannot be read from it.
    """
    curve: list[tuple[int, float, int]] = []
    data = list(series)
    block = 1
    while len(data) >= 4:
        curve.append((block, naive_standard_error(data), len(data)))
        data = [(data[i] + data[i + 1]) / 2.0 for i in range(0, len(data) - 1, 2)]
        block *= 2
    return curve


def blocked_standard_error(series: list[float]) -> float:
    """The plateau value of the blocking curve -- the honest standard error.

    Takes the maximum over the curve rather than fitting a plateau. The curve is
    monotonically rising until it plateaus and then becomes noisy at large block
    size, so the max is both simple and conservative: it can overestimate on a
    short series, never systematically underestimate, and an overestimate fails a
    gate loudly while an underestimate passes one silently.
    """
    curve = blocking_curve(series)
    if not curve:
        return float("nan")
    return max(se for _, se, _ in curve)


def integrated_autocorrelation_time(series: list[float]) -> float:
    """tau_int estimated from the ratio of blocked to naive variance.

    For a correlated series the true variance of the mean exceeds the naive one
    by a factor of 2*tau_int (with tau_int = 0.5 for independent samples).
    """
    naive = naive_standard_error(series)
    blocked = blocked_standard_error(series)
    if not math.isfinite(naive) or naive <= 0.0:
        return float("nan")
    return 0.5 * (blocked / naive) ** 2
