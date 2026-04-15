"""CCSDT equation generation driver."""

from __future__ import annotations

from ..generate import generate_cc_equations
from ..project import AlgebraTerm


def build_ccsdt_equations() -> dict[str, list[AlgebraTerm]]:
    """Generate spin-orbital CCSDT residuals.

    Returns energy, singles, doubles, and triples equations.
    """
    return generate_cc_equations("ccsdt")
