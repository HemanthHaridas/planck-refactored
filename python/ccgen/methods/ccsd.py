"""CCSD equation generation driver."""

from __future__ import annotations

from ..generate import generate_cc_equations
from ..project import AlgebraTerm


def build_ccsd_equations() -> dict[str, list[AlgebraTerm]]:
    """Generate spin-orbital CCSD energy, singles, and doubles residuals."""
    return generate_cc_equations("ccsd")
