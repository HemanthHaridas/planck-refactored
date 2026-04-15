"""Readable symbolic equation printer."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from ..project import AlgebraTerm, manifold_rank
from ..tensors import Tensor
from ..indices import Index

_EMPTY_INDICES: tuple[Index, ...] = ()


def _residual_name(target: str) -> str:
    """Map a manifold target name to a residual symbol."""
    if target == "energy":
        return "E_CC"
    try:
        rank = manifold_rank(target)
    except ValueError:
        return target
    return f"R{rank}"


def _format_coeff(c: Fraction) -> str:
    if c == 1:
        return "+ "
    if c == -1:
        return "- "
    if c > 0:
        return f"+ {c} "
    return f"- {-c} "


def _format_tensor(t: Tensor) -> str:
    idx_str = ",".join(i.name for i in t.indices)
    return f"{t.name}({idx_str})"


def _format_summed(indices: tuple[Index, ...]) -> str:
    if not indices:
        return ""
    names = ",".join(i.name for i in indices)
    return f"sum({names}) "


def format_term(term: AlgebraTerm) -> str:
    """Format a single AlgebraTerm as readable text."""
    parts = []
    parts.append(_format_coeff(term.coeff))
    parts.append(_format_summed(term.summed_indices))
    parts.append(" ".join(_format_tensor(f) for f in term.factors))
    return "".join(parts)


def format_residual(
    name: str,
    free_indices: tuple[Index, ...],
    terms: Sequence[AlgebraTerm],
) -> str:
    """Format a full residual equation as readable text."""
    idx_str = ",".join(i.name for i in free_indices)
    lines = [f"{name}({idx_str}) ="]
    for t in terms:
        lines.append(f"  {format_term(t)}")
    return "\n".join(lines)


def format_equations(equations: dict[str, list[AlgebraTerm]]) -> str:
    """Format a dict of {target: terms} as a complete equation listing."""
    blocks = []
    for target, terms in equations.items():
        if not terms:
            blocks.append(f"# {target}: (no terms)")
            continue

        if target == "energy":
            free: tuple[Index, ...] = _EMPTY_INDICES
        else:
            free = terms[0].free_indices if terms else _EMPTY_INDICES

        name = _residual_name(target)
        blocks.append(format_residual(name, free, terms))

    return "\n\n".join(blocks)
