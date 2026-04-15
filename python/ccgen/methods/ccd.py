"""CCD equation generation driver."""

from __future__ import annotations

from ..generate import generate_cc_equations
from ..project import AlgebraTerm


def build_ccd_equations() -> dict[str, list[AlgebraTerm]]:
    """Generate spin-orbital CCD energy and doubles residual."""
    return generate_cc_equations("ccd")
