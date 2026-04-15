"""Builders for cluster operators T1, T2, … Tn."""

from __future__ import annotations

import re
from fractions import Fraction
from math import factorial

from .indices import make_occ, make_vir, OCC_POOL, VIR_POOL, extend_pool
from .tensors import tn
from .sqops import create, annihilate
from .expr import OpTerm, Expr


def build_tn(n: int) -> Expr:
    """Build the n-body cluster operator Tn.

    T_n = (1/n!)^2  sum_{i1..in, a1..an}
          t_{i1..in}^{a1..an}  a+_{a1} … a+_{an} a_{in} … a_{i1}

    The prefactor (1/n!)^2 accounts for the antisymmetric summation
    over *n* occupied and *n* virtual dummy indices.
    """
    extend_pool(OCC_POOL, n)
    extend_pool(VIR_POOL, n)

    occ = tuple(make_occ(OCC_POOL[k], dummy=True) for k in range(n))
    vir = tuple(make_vir(VIR_POOL[k], dummy=True) for k in range(n))

    tensor = tn(n, vir, occ)
    creators = tuple(create(a) for a in vir)
    annihilators = tuple(annihilate(i) for i in reversed(occ))

    term = OpTerm(
        coeff=Fraction(1, factorial(n) ** 2),
        tensors=(tensor,),
        sqops=creators + annihilators,
        origin=("T", f"T{n}"),
    )
    return Expr([term])


# Keep the original convenience wrappers — they delegate to build_tn.
def build_t1() -> Expr:
    """T1 = sum_{ia} t_i^a a+_a a_i."""
    return build_tn(1)


def build_t2() -> Expr:
    """T2 = 1/4 sum_{ijab} t_{ij}^{ab} a+_a a+_b a_j a_i."""
    return build_tn(2)


def build_t3() -> Expr:
    """T3 = 1/36 sum_{ijkabc} t_{ijk}^{abc} a+_a a+_b a+_c a_k a_j a_i."""
    return build_tn(3)


# ── CC level string parsing ──────────────────────────────────────────

_RANK_LETTERS = {
    "s": 1,   # Singles
    "d": 2,   # Doubles
    "t": 3,   # Triples
    "q": 4,   # Quadruples
    "5": 5,   # Quintuples
    "6": 6,   # Sextuples (hextuples)
}


def parse_cc_level(level: str) -> list[int]:
    """Parse a CC method string into a sorted list of excitation ranks.

    Accepted forms:
      - Named: ``"ccd"``, ``"ccsd"``, ``"ccsdt"``, ``"ccsdtq"``, …
      - Numeric suffix: ``"ccsdtq5"``, ``"ccsdtq56"``
      - Explicit: ``"cc4"`` (includes T1–T4), ``"cc6"`` (T1–T6)

    Returns a sorted list of integers, e.g. ``[1, 2]`` for CCSD.
    """
    level = level.strip().lower()

    # "cc<N>" shorthand → T1 through TN
    m = re.fullmatch(r"cc(\d+)", level)
    if m:
        max_rank = int(m.group(1))
        if max_rank < 1:
            raise ValueError(f"CC rank must be ≥ 1, got {max_rank}")
        return list(range(1, max_rank + 1))

    if not level.startswith("cc"):
        raise ValueError(
            f"Unknown CC level '{level}'; expected 'cc' prefix"
        )
    body = level[2:]
    if not body:
        raise ValueError("Empty CC level after 'cc' prefix")

    ranks: list[int] = []
    for ch in body:
        if ch not in _RANK_LETTERS:
            raise ValueError(
                f"Unknown excitation character '{ch}' in '{level}'"
            )
        ranks.append(_RANK_LETTERS[ch])

    ranks.sort()
    return ranks


def build_cluster(level: str) -> Expr:
    """Build the cluster operator for a named method.

    Accepts any level string that :func:`parse_cc_level` understands.
    """
    ranks = parse_cc_level(level)
    result = build_tn(ranks[0])
    for r in ranks[1:]:
        result = result + build_tn(r)
    return result
