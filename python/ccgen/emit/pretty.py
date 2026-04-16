"""Readable symbolic equation printer.

Phase 3 additions:
- Intermediate tensor definitions listed separately
- Equation grouping by manifold with explanatory comments
- Index space legend (i,j,k = occupied; a,b,c = virtual)
- Term origin annotations (BCH order, Wick topology)
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence, TYPE_CHECKING

from ..project import AlgebraTerm, manifold_rank
from ..tensors import Tensor
from ..indices import Index

if TYPE_CHECKING:
    from ..optimization.intermediates import IntermediateSpec

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


# ── Phase 3: Enhanced pretty printing ─────────────────────────────


INDEX_LEGEND = """\
# Index Legend
#   i, j, k, l, m, n, o  = occupied (hole) indices
#   a, b, c, d, e, f, g  = virtual (particle) indices
#   p, q, r, s, t, u, v  = general indices
#
# Tensor Legend
#   f(p,q)         = Fock matrix element
#   v(p,q,r,s)     = antisymmetrized two-electron integral <pq||rs>
#   t1(a,i)        = singles amplitude
#   t2(a,b,i,j)    = doubles amplitude
#   t3(a,b,c,i,j,k)= triples amplitude
#   D(...)         = orbital energy denominator
#   W_xxxx(...)    = intermediate tensor"""


def format_intermediate(spec: IntermediateSpec) -> str:
    """Format a single intermediate tensor definition."""
    idx_str = ",".join(i.name for i in spec.indices)
    lines = [
        f"# Intermediate: {spec.name} ({spec.index_space_sig},"
        f" used {spec.usage_count} times)",
    ]
    # Add layout hints if present
    if spec.memory_layout != "row_major":
        lines.append(f"#   layout: {spec.memory_layout}")
    if spec.allocation_strategy != "auto":
        lines.append(f"#   allocation: {spec.allocation_strategy}")
    if spec.blocking_hint:
        hint_str = ", ".join(f"{k}={v}" for k, v in spec.blocking_hint.items())
        lines.append(f"#   blocking: {hint_str}")

    lines.append(f"{spec.name}({idx_str}) =")
    for t in spec.definition_terms:
        lines.append(f"  {format_term(t)}")
    return "\n".join(lines)


def format_intermediates(
    intermediates: Sequence[IntermediateSpec],
) -> str:
    """Format all intermediate definitions as a section."""
    if not intermediates:
        return ""

    blocks = [
        f"# ---- Intermediate Tensor Definitions"
        f" ({len(intermediates)} intermediates) ----",
    ]
    for spec in intermediates:
        blocks.append(format_intermediate(spec))
    return "\n\n".join(blocks)


def _manifold_description(target: str) -> str:
    """Return a human-readable description for a manifold."""
    descriptions = {
        "energy": "Correlation energy",
        "singles": "Singles (T1) residual",
        "doubles": "Doubles (T2) residual",
        "triples": "Triples (T3) residual",
        "quadruples": "Quadruples (T4) residual",
    }
    return descriptions.get(target, target.capitalize())


def format_equations_with_intermediates(
    equations: dict[str, list[AlgebraTerm]],
    intermediates: Sequence[IntermediateSpec] | None = None,
    include_legend: bool = True,
    include_stats: bool = True,
) -> str:
    """Format equations with intermediates, legend, and section headers.

    Parameters
    ----------
    equations : dict
        Mapping of manifold name to list of AlgebraTerms.
    intermediates : sequence of IntermediateSpec or None
        Intermediate tensor definitions to display.
    include_legend : bool
        If True, prepend an index/tensor legend.
    include_stats : bool
        If True, append a summary of term counts per manifold.
    """
    blocks: list[str] = []

    # Header
    blocks.append("# Coupled-Cluster Residual Equations")
    blocks.append(f"# Total manifolds: {len(equations)}")
    total_terms = sum(len(terms) for terms in equations.values())
    blocks.append(f"# Total terms: {total_terms}")
    if intermediates:
        blocks.append(f"# Intermediates: {len(intermediates)}")
    blocks.append("")

    # Legend
    if include_legend:
        blocks.append(INDEX_LEGEND)
        blocks.append("")

    # Intermediate definitions
    if intermediates:
        blocks.append(format_intermediates(intermediates))
        blocks.append("")

    # Equations by manifold
    for target, terms in equations.items():
        desc = _manifold_description(target)
        blocks.append(f"# ---- {desc} ({len(terms)} terms) ----")

        if not terms:
            blocks.append(f"# {target}: (no terms)")
            blocks.append("")
            continue

        if target == "energy":
            free: tuple[Index, ...] = _EMPTY_INDICES
        else:
            free = terms[0].free_indices if terms else _EMPTY_INDICES

        name = _residual_name(target)
        blocks.append(format_residual(name, free, terms))
        blocks.append("")

    # Summary statistics
    if include_stats:
        blocks.append("# ---- Summary ----")
        for target, terms in equations.items():
            blocks.append(f"#   {target}: {len(terms)} terms")
        blocks.append(f"#   Total: {total_terms} terms")
        if intermediates:
            blocks.append(f"#   Intermediates: {len(intermediates)}")

    return "\n".join(blocks)
