"""NumPy einsum-style backend for validation and debugging.

Supports optional intermediate tensor emission when used with
the optimization.intermediates module.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence, TYPE_CHECKING

from ..project import AlgebraTerm, manifold_rank
from ..tensors import Tensor
from ..indices import Index

try:
    import opt_einsum
    _HAS_OPT_EINSUM = True
except ImportError:
    _HAS_OPT_EINSUM = False

if TYPE_CHECKING:
    from ..optimization.intermediates import IntermediateSpec


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


def format_term_einsum(
    term: AlgebraTerm,
    lhs: str = "R",
    use_opt_einsum: bool = False,
) -> str:
    """Format a single term as a numpy einsum call.

    Parameters
    ----------
    use_opt_einsum : bool
        If True and opt_einsum is available, emit
        ``opt_einsum.contract(...)`` instead of ``np.einsum(...)``.
    """
    subscripts = _einsum_subscripts(term.factors, term.summed_indices)

    tensor_names = []
    for fac in term.factors:
        if fac.name == "D":
            tensor_names.append("D")
        else:
            name = fac.name.upper()
            tensor_names.append(name)

    if use_opt_einsum and _HAS_OPT_EINSUM:
        fn = "opt_einsum.contract"
        args = ", ".join(
            [f"'{subscripts}'"] + tensor_names + ["optimize='optimal'"]
        )
    else:
        fn = "np.einsum"
        args = ", ".join([f"'{subscripts}'"] + tensor_names)

    coeff = _coeff_str(term.coeff)

    if term.coeff > 0:
        return f"{lhs} += {coeff}{fn}({args})"
    else:
        neg_coeff = _coeff_str(-term.coeff)
        return f"{lhs} -= {neg_coeff}{fn}({args})"


def format_residual_einsum(
    lhs: str,
    terms: Sequence[AlgebraTerm],
    free_indices: tuple[Index, ...] | None = None,
    use_opt_einsum: bool = False,
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
        lines.append(format_term_einsum(t, lhs, use_opt_einsum=use_opt_einsum))

    return "\n".join(lines)


def format_equations_einsum(
    equations: dict[str, list[AlgebraTerm]],
    use_opt_einsum: bool = False,
) -> str:
    """Format all equations as numpy einsum code.

    Parameters
    ----------
    use_opt_einsum : bool
        If True and opt_einsum is installed, emit
        ``opt_einsum.contract(...)`` calls with ``optimize='optimal'``
        for automatic contraction path optimization.
    """
    blocks = []
    if use_opt_einsum and _HAS_OPT_EINSUM:
        blocks.append("import opt_einsum")
        blocks.append("")
    for target, terms in equations.items():
        lhs = _residual_name(target)
        free = terms[0].free_indices if terms else ()
        blocks.append(format_residual_einsum(
            lhs, terms, free, use_opt_einsum=use_opt_einsum,
        ))
    return "\n\n".join(blocks)


def format_intermediate_einsum(
    spec: IntermediateSpec,
) -> str:
    """Format a single intermediate build as numpy einsum statements."""
    lines: list[str] = []
    lines.append(f"# Intermediate {spec.name} ({spec.index_space_sig})")
    lines.append(format_residual_einsum(
        spec.name,
        spec.definition_terms,
        spec.indices,
    ))
    return "\n".join(lines)


def format_intermediates_einsum(
    intermediates: Sequence[IntermediateSpec],
) -> str:
    """Format all intermediate builds as numpy einsum code."""
    if not intermediates:
        return ""
    blocks = [f"# ── Intermediate tensor builds ({len(intermediates)}) ──"]
    for spec in intermediates:
        blocks.append(format_intermediate_einsum(spec))
    return "\n\n".join(blocks)


def format_equations_with_intermediates_einsum(
    equations: dict[str, list[AlgebraTerm]],
    intermediates: Sequence[IntermediateSpec] | None = None,
) -> str:
    """Format equations with optional intermediate builds as einsum code."""
    blocks: list[str] = []

    if intermediates:
        blocks.append(format_intermediates_einsum(intermediates))
        blocks.append("")

    blocks.append(format_equations_einsum(equations))
    return "\n\n".join(blocks)
