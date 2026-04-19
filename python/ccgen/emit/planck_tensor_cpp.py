"""Planck-specific C++ emitter for coupled-cluster tensor kernels.

The generic ``cpp_loops`` backend emits abstract ``F/V/R`` accessors. This
backend targets the concrete tensor objects already present in Planck's CC
implementation:

- ``CanonicalRHFCCReference`` for Fock blocks
- ``TensorCCBlockCache`` for ERI blocks
- ``DenominatorCache`` for orbital-energy denominators
- ``RCCSDAmplitudes`` / ``RCCSDTAmplitudes`` for cluster amplitudes
"""

from __future__ import annotations

from fractions import Fraction
import re
from typing import Sequence, TYPE_CHECKING

from ..cluster import parse_cc_level
from ..indices import Index
from ..lowering import (
    LoweredTensorFactor,
    RestrictedClosedShellTerm,
    lower_equations_restricted_closed_shell,
    lower_term_restricted_closed_shell,
)
from ..project import AlgebraTerm
from ..tensors import Tensor

if TYPE_CHECKING:
    from ..optimization.intermediates import IntermediateSpec


_CANONICAL_ERI_BLOCKS: dict[str, tuple[str, str, str, str]] = {
    "oooo": ("o", "o", "o", "o"),
    "ooov": ("o", "o", "o", "v"),
    "oovv": ("o", "o", "v", "v"),
    "ovov": ("o", "v", "o", "v"),
    "ovvo": ("o", "v", "v", "o"),
    "ovvv": ("o", "v", "v", "v"),
    "vvvv": ("v", "v", "v", "v"),
}

_ERI_SYMMETRY_PERMUTATIONS: tuple[tuple[tuple[int, int, int, int], int], ...] = (
    ((0, 1, 2, 3), +1),
    ((1, 0, 2, 3), -1),
    ((0, 1, 3, 2), -1),
    ((1, 0, 3, 2), +1),
    ((2, 3, 0, 1), +1),
    ((3, 2, 0, 1), -1),
    ((2, 3, 1, 0), -1),
    ((3, 2, 1, 0), +1),
)


def _space_char(idx: Index) -> str:
    if idx.space == "occ":
        return "o"
    if idx.space == "vir":
        return "v"
    return "g"


def _loop_bound(idx: Index) -> str:
    if idx.space == "occ":
        return "no"
    if idx.space == "vir":
        return "nv"
    return "n"


def _coeff_literal(coeff: Fraction) -> str:
    value = float(coeff)
    if value == int(value):
        integer = int(value)
        if integer == 1:
            return ""
        if integer == -1:
            return "-"
        return f"{integer} * "
    return f"{value} * "


def _tensor_type(rank: int) -> str:
    if rank == 0:
        return "double"
    if rank == 2:
        return "Tensor2D"
    if rank == 4:
        return "Tensor4D"
    if rank == 6:
        return "Tensor6D"
    if rank > 0:
        return "TensorND"
    raise ValueError(f"Unsupported tensor rank {rank} for Planck emitter")


def _is_supported_tensor_rank(rank: int) -> bool:
    return rank >= 0


def _dims_expr(indices: Sequence[Index], result_type: str | None = None) -> str:
    result_type = result_type or _tensor_type(len(indices))
    dims: list[str] = []
    for idx in indices:
        if idx.space == "occ":
            dims.append("no")
        elif idx.space == "vir":
            dims.append("nv")
        else:
            raise ValueError(
                "Planck emitter only supports occupied/virtual tensors, "
                f"got index {idx!r}"
            )
    if result_type == "TensorND":
        return "std::vector<int>{" + ", ".join(dims) + "}, 0.0"
    return ", ".join(dims) + ", 0.0"


def _inverse_permutation(perm: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(perm)
    for i, value in enumerate(perm):
        inverse[value] = i
    return tuple(inverse)


def _source_tensor(factor: Tensor | LoweredTensorFactor) -> Tensor:
    if isinstance(factor, LoweredTensorFactor):
        return factor.source
    return factor


def _access_indices(factor: Tensor | LoweredTensorFactor) -> tuple[Index, ...]:
    if isinstance(factor, LoweredTensorFactor):
        return factor.spatial_indices
    return _source_tensor(factor).indices


def _map_eri_tensor(tensor: Tensor | LoweredTensorFactor) -> tuple[int, str]:
    if isinstance(tensor, LoweredTensorFactor):
        if "g" in tensor.spatial_block:
            raise NotImplementedError(
                "General-space ERI blocks are not supported in Planck output: "
                f"{tensor.source!r}"
            )
        return (
            tensor.phase,
            f"mo_blocks.{tensor.spatial_block}("
            f"{', '.join(idx.name for idx in tensor.spatial_indices)})",
        )

    tensor_obj = _source_tensor(tensor)
    spaces = tuple(_space_char(idx) for idx in tensor_obj.indices)
    if "g" in spaces:
        raise NotImplementedError(
            "General-space ERI blocks are not supported in Planck output: "
            f"{tensor_obj!r}"
        )

    for block_name, block_spaces in _CANONICAL_ERI_BLOCKS.items():
        for perm, sign in _ERI_SYMMETRY_PERMUTATIONS:
            transformed = tuple(block_spaces[i] for i in perm)
            if transformed != spaces:
                continue

            inverse = _inverse_permutation(perm)
            reordered = [tensor_obj.indices[i].name for i in inverse]
            return sign, f"mo_blocks.{block_name}({', '.join(reordered)})"

    raise NotImplementedError(
        f"No Planck ERI block mapping available for pattern "
        f"{''.join(spaces)} in {tensor_obj!r}"
    )


def _map_factor(tensor: Tensor | LoweredTensorFactor) -> tuple[int, str]:
    tensor_obj = _source_tensor(tensor)
    indices = _access_indices(tensor)
    amplitude_match = re.fullmatch(r"t(\d+)", tensor_obj.name)

    if tensor_obj.name == "f":
        left, right = indices
        if left.space == "occ" and right.space == "occ":
            return 1, f"reference.f_oo({left.name}, {right.name})"
        if left.space == "vir" and right.space == "vir":
            return 1, f"reference.f_vv({left.name}, {right.name})"
        occ = next((idx for idx in indices if idx.space == "occ"), None)
        vir = next((idx for idx in indices if idx.space == "vir"), None)
        if occ is not None and vir is not None:
            return 1, f"reference.f_ov({occ.name}, {vir.name})"
        raise NotImplementedError(f"Unsupported Fock block for {tensor_obj!r}")

    if tensor_obj.name == "v":
        return _map_eri_tensor(tensor)

    if amplitude_match is not None:
        excitation_rank = int(amplitude_match.group(1))
        occ = [idx.name for idx in indices if idx.space == "occ"]
        vir = [idx.name for idx in indices if idx.space == "vir"]
        if len(occ) != excitation_rank or len(vir) != excitation_rank:
            raise ValueError(f"Invalid amplitude tensor layout for {tensor_obj!r}")
        if excitation_rank == 1:
            return 1, f"amplitudes.t1({occ[0]}, {vir[0]})"
        if excitation_rank == 2:
            return 1, f"amplitudes.t2({', '.join(occ + vir)})"
        if excitation_rank == 3:
            return 1, f"amplitudes.t3({', '.join(occ + vir)})"
        return 1, (
            f"amplitudes.tensor({excitation_rank})"
            f"({{{', '.join(occ + vir)}}})"
        )

    if tensor_obj.name == "D":
        occ = [idx.name for idx in indices if idx.space == "occ"]
        vir = [idx.name for idx in indices if idx.space == "vir"]
        rank = len(indices)
        excitation_rank = rank // 2
        if rank == 2 and len(occ) == 1 and len(vir) == 1:
            return 1, f"denominators.d1({occ[0]}, {vir[0]})"
        if rank == 4 and len(occ) == 2 and len(vir) == 2:
            return 1, f"denominators.d2({', '.join(occ + vir)})"
        if rank == 6 and len(occ) == 3 and len(vir) == 3:
            return 1, f"denominators.d3({', '.join(occ + vir)})"
        if rank > 0 and len(occ) == excitation_rank and len(vir) == excitation_rank:
            return 1, (
                f"denominators.tensor({excitation_rank})"
                f"({{{', '.join(occ + vir)}}})"
            )
        raise ValueError(f"Invalid denominator tensor layout for {tensor_obj!r}")

    if tensor_obj.name == "delta":
        left, right = indices
        return 1, f"(({left.name} == {right.name}) ? 1.0 : 0.0)"

    if tensor_obj.name.startswith("W_"):
        return 1, _target_expr(tensor_obj.name, indices)

    raise NotImplementedError(f"Unsupported tensor factor {tensor_obj!r}")


def emit_planck_term(
    term: AlgebraTerm | RestrictedClosedShellTerm,
    lhs: str = "result",
    indent: int = 4,
) -> str:
    """Emit a single algebraic term using Planck tensor accessors."""
    pad = " " * indent
    lines: list[str] = []

    if isinstance(term, RestrictedClosedShellTerm):
        free = list(term.canonical_free_indices)
        summed = list(term.canonical_summed_indices)
        factors: Sequence[Tensor | LoweredTensorFactor] = term.factors
    else:
        free = list(term.free_indices)
        summed = list(term.summed_indices)
        factors = term.factors

    for idx in free:
        lines.append(
            f"{pad}for (int {idx.name} = 0; {idx.name} < {_loop_bound(idx)}; ++{idx.name})"
        )

    lines.append(f"{pad}{{")

    sign = 1
    factor_exprs: list[str] = []
    for factor in factors:
        factor_sign, factor_expr = _map_factor(factor)
        sign *= factor_sign
        factor_exprs.append(factor_expr)

    coeff = term.coeff * sign
    product = " * ".join(factor_exprs) if factor_exprs else "1.0"

    target = _target_expr(lhs, free)

    if summed:
        lines.append(f"{pad}    double acc = 0.0;")
        for idx in summed:
            lines.append(
                f"{pad}    for (int {idx.name} = 0; {idx.name} < {_loop_bound(idx)}; ++{idx.name})"
            )
        lines.append(f"{pad}        acc += {_coeff_literal(coeff)}{product};")
        lines.append(f"{pad}    {target} += acc;")
    else:
        lines.append(f"{pad}    {target} += {_coeff_literal(coeff)}{product};")

    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _amplitude_type(method: str) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if max_rank >= 4:
        return "ArbitraryOrderRCCAmplitudes"
    if max_rank >= 3:
        return "RCCSDTAmplitudes"
    return "RCCSDAmplitudes"


def _denominator_type(method: str) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if max_rank >= 4:
        return "ArbitraryOrderDenominatorCache"
    return "DenominatorCache"


def _target_expr(lhs: str, indices: Sequence[Index]) -> str:
    if not indices:
        return lhs
    if len(indices) in (2, 4, 6):
        return f"{lhs}({', '.join(idx.name for idx in indices)})"
    return f"{lhs}({{{', '.join(idx.name for idx in indices)}}})"


def _kernel_name(method: str, target: str) -> str:
    if target == "energy":
        return f"compute_{method}_energy"
    return f"compute_{method}_{target}_residual"


def _emit_kernel(
    method: str,
    target: str,
    terms: Sequence[AlgebraTerm],
    lowered_terms: Sequence[RestrictedClosedShellTerm] | None = None,
    intermediates: Sequence[IntermediateSpec] | None = None,
    free_indices: Sequence[Index] | None = None,
) -> str:
    lowered_terms = tuple(lowered_terms or ())
    if free_indices is None:
        if lowered_terms:
            free_indices = lowered_terms[0].canonical_free_indices
        else:
            free_indices = terms[0].free_indices if terms else ()
    free_indices = tuple(free_indices)
    result_rank = len(free_indices)
    result_type = _tensor_type(result_rank)
    amplitude_type = _amplitude_type(method)
    denominator_type = _denominator_type(method)
    intermediate_map = {spec.name: spec for spec in intermediates or ()}

    lines: list[str] = []
    lines.append(f"{result_type} {_kernel_name(method, target)}(")
    lines.append("    const CanonicalRHFCCReference &reference,")
    lines.append("    const TensorCCBlockCache &mo_blocks,")
    lines.append(f"    const {denominator_type} &denominators,")
    lines.append(f"    const {amplitude_type} &amplitudes)")
    lines.append("{")
    lines.append("    const int no = reference.orbital_partition.n_occ;")
    lines.append("    const int nv = reference.orbital_partition.n_virt;")
    if result_rank == 0:
        lines.append("    double result = 0.0;")
    else:
        lines.append(
            f"    {result_type} result({_dims_expr(free_indices, result_type)});"
        )
    required_intermediates: list[IntermediateSpec] = []
    seen_intermediates: set[str] = set()
    for term in terms:
        for factor in term.factors:
            if factor.name in intermediate_map and factor.name not in seen_intermediates:
                seen_intermediates.add(factor.name)
                required_intermediates.append(intermediate_map[factor.name])
    if required_intermediates:
        lines.append("")
        lines.append("    // Build reused intermediates once for this kernel")
        for spec in required_intermediates:
            lines.append(
                f"    const auto {spec.name} = build_{spec.name}("
                "reference, mo_blocks, denominators, amplitudes);"
            )
    lines.append("")
    lines.append(f"    // {target} kernel ({len(terms)} terms)")
    emitted_terms: Sequence[AlgebraTerm | RestrictedClosedShellTerm]
    emitted_terms = lowered_terms if lowered_terms else terms
    for i, term in enumerate(emitted_terms, start=1):
        lines.append(f"    // Term {i}")
        lines.append(emit_planck_term(term, lhs="result", indent=4))
        lines.append("")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


def _emit_intermediate_builder(method: str, spec: IntermediateSpec) -> str:
    result_type = _tensor_type(spec.rank)
    amplitude_type = _amplitude_type(method)
    denominator_type = _denominator_type(method)
    lowered_definition_terms = tuple(
        lower_term_restricted_closed_shell(term, "reference")
        for term in spec.definition_terms
    )
    if lowered_definition_terms:
        builder_indices = lowered_definition_terms[0].canonical_free_indices
    else:
        builder_indices = spec.indices

    lines: list[str] = []
    lines.append(f"{result_type} build_{spec.name}(")
    lines.append("    const CanonicalRHFCCReference &reference,")
    lines.append("    const TensorCCBlockCache &mo_blocks,")
    lines.append(f"    const {denominator_type} &denominators,")
    lines.append(f"    const {amplitude_type} &amplitudes)")
    lines.append("{")
    lines.append("    const int no = reference.orbital_partition.n_occ;")
    lines.append("    const int nv = reference.orbital_partition.n_virt;")
    lines.append(
        f"    {result_type} result({_dims_expr(builder_indices, result_type)});"
    )
    lines.append("")
    lines.append(
        f"    // Intermediate {spec.name} ({spec.index_space_sig}, usage={spec.usage_count})"
    )
    for i, term in enumerate(lowered_definition_terms, start=1):
        lines.append(f"    // Definition term {i}")
        lines.append(emit_planck_term(term, lhs="result", indent=4))
        lines.append("")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


def emit_planck_translation_unit(
    method: str,
    equations: dict[str, list[AlgebraTerm]],
    intermediates: Sequence[IntermediateSpec] | None = None,
    lowered_equations: dict[str, list[RestrictedClosedShellTerm]] | None = None,
) -> str:
    """Emit a Planck-compatible C++ translation unit."""
    method = method.lower()
    lowered_equations = lowered_equations or lower_equations_restricted_closed_shell(
        equations
    )
    lines: list[str] = []
    lines.append("// Auto-generated by ccgen")
    lines.append(f"// Planck tensor kernels for {method.upper()}")
    lines.append("")
    lines.append('#include "post_hf/cc/tensor_backend.h"')
    lines.append("")
    lines.append("namespace HartreeFock::Correlation::CC")
    lines.append("{")
    lines.append("")

    if intermediates:
        for spec in intermediates:
            if not _is_supported_tensor_rank(spec.rank):
                continue
            lines.append(_emit_intermediate_builder(method, spec))
            lines.append("")

    for target, terms in equations.items():
        lines.append(
            _emit_kernel(
                method,
                target,
                terms,
                lowered_terms=lowered_equations.get(target),
                intermediates=intermediates,
            )
        )
        lines.append("")

    lines.append("} // namespace HartreeFock::Correlation::CC")
    return "\n".join(lines)
