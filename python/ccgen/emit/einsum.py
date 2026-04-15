"""NumPy einsum-style backend for validation and debugging."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from ..project import AlgebraTerm, manifold_rank
from ..tensors import Tensor
from ..indices import Index


def _residual_name(target: str) -> str:
    """Map a manifold target name to a residual symbol."""
    if target == "energy":
        return "E_CC"
    try:
        rank = manifold_rank(target)
    except ValueError:
        return target
    return f"R{rank}"


def _einsum_subscripts(
    factors: tuple[Tensor, ...], summed: tuple[Index, ...]
) -> str:
    """Build an einsum subscript string like 'abcd,cdij->abij'."""
    input_parts = []
    for fac in factors:
        input_parts.append("".join(i.name for i in fac.indices))

    all_indices: list[str] = []
    seen: set[str] = set()
    for fac in factors:
        for i in fac.indices:
            if i.name not in seen:
                all_indices.append(i.name)
                seen.add(i.name)

    summed_names = {i.name for i in summed}
    output = [n for n in all_indices if n not in summed_names]

    return ",".join(input_parts) + "->" + "".join(output)


def _coeff_str(c: Fraction) -> str:
    if c == 1:
        return ""
    if c == -1:
        return "-"
    f = float(c)
    if f == int(f):
        return str(int(f)) + " * "
    return str(c) + " * "


def format_term_einsum(term: AlgebraTerm, lhs: str = "R") -> str:
    """Format a single term as a numpy einsum call."""
    subscripts = _einsum_subscripts(term.factors, term.summed_indices)

    tensor_names = []
    for fac in term.factors:
        name = fac.name.upper()
        tensor_names.append(name)

    args = ", ".join([f"'{subscripts}'"] + tensor_names)
    coeff = _coeff_str(term.coeff)

    if term.coeff > 0:
        return f"{lhs} += {coeff}np.einsum({args})"
    else:
        neg_coeff = _coeff_str(-term.coeff)
        return f"{lhs} -= {neg_coeff}np.einsum({args})"


def format_residual_einsum(
    lhs: str,
    terms: Sequence[AlgebraTerm],
    free_indices: tuple[Index, ...] | None = None,
) -> str:
    """Format a full residual as numpy einsum statements."""
    lines = []

    if free_indices:
        shape_parts = []
        for idx in free_indices:
            if idx.space == "occ":
                shape_parts.append("no")
            elif idx.space == "vir":
                shape_parts.append("nv")
            else:
                shape_parts.append("n")
        shape = ", ".join(shape_parts)
        lines.append(f"{lhs} = np.zeros(({shape}))")
    else:
        lines.append(f"{lhs} = 0.0")

    for t in terms:
        lines.append(format_term_einsum(t, lhs))

    return "\n".join(lines)


def format_equations_einsum(equations: dict[str, list[AlgebraTerm]]) -> str:
    """Format all equations as numpy einsum code."""
    blocks = []
    for target, terms in equations.items():
        lhs = _residual_name(target)
        free = terms[0].free_indices if terms else ()
        blocks.append(format_residual_einsum(lhs, terms, free))
    return "\n\n".join(blocks)
