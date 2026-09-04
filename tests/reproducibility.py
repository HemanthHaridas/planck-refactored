#!/usr/bin/env python3
"""Fixed-seed reproducibility harness (FCIQMC scope, G3).

A stochastic method is only maintainable here if a rerun with the same seed
reproduces its trajectory BITWISE. That is the gate that survives at any system
size -- unlike the statistical gate, which needs a deterministic reference and so
only ever runs on small systems.

The harness is deliberately agnostic about what produces the trajectory: it takes
a callable and compares repeated invocations. That lets it be proven against a
DETERMINISTIC producer (the existing FCI) before anything stochastic relies on it.
"""
from __future__ import annotations

import struct
from typing import Callable, Iterable, Sequence


def trajectory_digest(values: Iterable[float]) -> str:
    """A digest over the exact bits of a float sequence.

    Uses struct '<d' rather than repr/round: two doubles differing in the last
    ulp must produce different digests, which is the entire point. A text
    comparison at print precision would hide exactly the reduction-order defects
    this codebase has been bitten by.
    """
    import hashlib

    h = hashlib.sha256()
    for v in values:
        h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


def check_reproducible(
    produce: Callable[[], Sequence[float]],
    n_runs: int = 3,
) -> tuple[bool, str]:
    """Run `produce` n_runs times; require every trajectory to be bit-identical.

    Returns (ok, detail). `produce` must be self-contained -- it is responsible
    for reseeding, so that a producer which accidentally carries state across
    calls is caught here rather than looking reproducible by luck.
    """
    if n_runs < 2:
        return False, "n_runs must be at least 2 to compare anything"
    digests = []
    lengths = []
    for _ in range(n_runs):
        traj = list(produce())
        digests.append(trajectory_digest(traj))
        lengths.append(len(traj))
    if len(set(lengths)) != 1:
        return False, f"trajectory lengths differ across runs: {lengths}"
    if len(set(digests)) != 1:
        return False, f"trajectories differ across runs: {sorted(set(digests))}"
    return True, f"{n_runs} runs identical ({digests[0][:16]}..., n={lengths[0]})"


def check_seed_sensitivity(
    produce_with_seed: Callable[[int], Sequence[float]],
    seed_a: int,
    seed_b: int,
) -> tuple[bool, str]:
    """Require DIFFERENT seeds to give different trajectories.

    Without this, a producer that ignores its seed entirely -- or one whose RNG
    is never actually consulted -- would sail through check_reproducible. A
    reproducibility gate that cannot fail proves nothing; this is its negative
    control.
    """
    da = trajectory_digest(produce_with_seed(seed_a))
    db = trajectory_digest(produce_with_seed(seed_b))
    if da == db:
        return False, f"seeds {seed_a} and {seed_b} gave IDENTICAL trajectories (seed ignored?)"
    return True, f"seeds {seed_a}/{seed_b} differ as required"
