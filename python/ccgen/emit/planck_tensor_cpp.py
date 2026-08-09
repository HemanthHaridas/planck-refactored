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


def _map_factor(
    tensor: Tensor | LoweredTensorFactor,
    intermediate_names: frozenset[str] | None = None,
    arbitrary_amplitudes: bool = False,
) -> tuple[int, str]:
    tensor_obj = _source_tensor(tensor)
    indices = _access_indices(tensor)
    # `t<rank>` is the reference Sz sector; `t<rank>_<tag>` (R3.1.3d) is a higher
    # independent Sz sector of the same amplitude (t4_aaabaaab), stored/read as a
    # distinct block. Both are rank-`\d+` amplitudes; the tag routes the read to
    # the sector's own tensor.
    amplitude_match = re.fullmatch(r"t(\d+)(?:_([ab]+))?", tensor_obj.name)

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
        sector_tag = amplitude_match.group(2)      # None for the reference sector
        occ = [idx.name for idx in indices if idx.space == "occ"]
        vir = [idx.name for idx in indices if idx.space == "vir"]
        if len(occ) != excitation_rank or len(vir) != excitation_rank:
            raise ValueError(f"Invalid amplitude tensor layout for {tensor_obj!r}")
        # The arbitrary-order runtime type (ArbitraryOrderRCCAmplitudes, rank ≥ 4
        # methods) exposes only `.tensor(rank)` — returning std::expected<view> —
        # with no t1/t2/t3 members. The view is bound ONCE per kernel/builder as a
        # local `t<rank>` (see _amplitude_view_bindings); the factor then indexes
        # that local with an initializer-list `t<rank>({...})`. The rank ≤ 3
        # tensor_backend types (RCCSD/RCCSDTAmplitudes) have direct t1/t2/t3(...)
        # accessors instead.
        if arbitrary_amplitudes:
            # R3.1.3d: a higher Sz sector reads from its own bound view
            # `t<rank>_<tag>` (sector_tag), the reference from `t<rank>`.
            view = (f"t{excitation_rank}_{sector_tag}" if sector_tag
                    else f"t{excitation_rank}")
            return 1, f"{view}({{{', '.join(occ + vir)}}})"
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

    # Extracted intermediates -- CSE "W_*", the tau/tau_c pseudo-amplitudes, and
    # (D7.3) the recognized dressed operators (Wmnij/Wabef/Wmbej/Fae/...) -- are
    # materialized locals built once per kernel; reference them by name.  Any
    # factor named in ``intermediate_names`` (the specs passed to the kernel)
    # resolves as such a local; the W_/tau prefixes stay as a fallback for the
    # CSE path that does not thread the name set.
    if (
        tensor_obj.name.startswith("W_")
        or tensor_obj.name == "tau"
        or (intermediate_names is not None and tensor_obj.name in intermediate_names)
    ):
        return 1, _target_expr(tensor_obj.name, indices)

    raise NotImplementedError(f"Unsupported tensor factor {tensor_obj!r}")


def emit_planck_term(
    term: AlgebraTerm | RestrictedClosedShellTerm,
    lhs: str = "result",
    indent: int = 4,
    intermediate_names: frozenset[str] | None = None,
    arbitrary_amplitudes: bool = False,
) -> str:
    """Emit a single algebraic term using Planck tensor accessors.

    ``intermediate_names`` lists the materialized intermediates in scope (the
    kernel's specs) so their factors resolve as local references (D7.3).
    ``arbitrary_amplitudes`` routes t-amplitude accessors through
    ``.tensor(rank)(...)`` for the arbitrary-order runtime type (rank ≥ 4)."""
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
        factor_sign, factor_expr = _map_factor(
            factor, intermediate_names, arbitrary_amplitudes)
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


def _amplitude_type(method: str, force_arbitrary: bool = False) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if force_arbitrary or max_rank >= 4:
        return "ArbitraryOrderRCCAmplitudes"
    if max_rank >= 3:
        return "RCCSDTAmplitudes"
    return "RCCSDAmplitudes"


def _amplitude_ranks_used(terms) -> list[tuple[int, str | None]]:
    """Distinct t-amplitude `(rank, sector_tag)` pairs referenced across `terms`,
    sorted. `sector_tag` is None for the reference Sz sector (`t<rank>`) and the
    tag string for a higher sector (`t<rank>_<tag>`, R3.1.3d). Drives the
    per-kernel view bindings for the arbitrary-order runtime."""
    used: set[tuple[int, str | None]] = set()
    for term in terms:
        for factor in term.factors:
            m = re.fullmatch(r"t(\d+)(?:_([ab]+))?", _source_tensor(factor).name)
            if m:
                used.add((int(m.group(1)), m.group(2)))
    return sorted(used, key=lambda rt: (rt[0], rt[1] or ""))


def _amplitude_view_bindings(terms, indent: int = 4) -> list[str]:
    """C++ lines binding one amplitude view per (rank, sector) used — the
    arbitrary-order runtime returns std::expected from its accessors, so the view
    is unwrapped once here and indexed in the loops. `.value()` is safe: the
    solver constructs every rank ≤ max before calling. The reference sector binds
    `t<rank> = amplitudes.tensor(rank)`; a higher Sz sector (R3.1.3d) binds
    `t<rank>_<tag> = amplitudes.sector_tensor(rank, "<tag>")`."""
    pad = " " * indent
    lines: list[str] = []
    for rank, tag in _amplitude_ranks_used(terms):
        if tag is None:
            lines.append(f"{pad}const auto t{rank} = amplitudes.tensor({rank}).value();")
        else:
            lines.append(
                f'{pad}const auto t{rank}_{tag} = '
                f'amplitudes.sector_tensor({rank}, "{tag}").value();')
    return lines


def _denominator_type(method: str, force_arbitrary: bool = False) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if force_arbitrary or max_rank >= 4:
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


def _target_rank(
    target: str,
    terms: Sequence[AlgebraTerm],
    lowered_terms: Sequence[RestrictedClosedShellTerm] | None = None,
) -> int:
    if target == "energy":
        return 0
    if lowered_terms:
        return len(lowered_terms[0].canonical_free_indices) // 2
    if terms:
        return len(terms[0].free_indices) // 2
    raise ValueError(f"Cannot infer excitation rank for target {target!r}")


def _emit_kernel(
    method: str,
    target: str,
    terms: Sequence[AlgebraTerm],
    lowered_terms: Sequence[RestrictedClosedShellTerm] | None = None,
    intermediates: Sequence[IntermediateSpec] | None = None,
    free_indices: Sequence[Index] | None = None,
    force_arbitrary: bool = False,
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
    amplitude_type = _amplitude_type(method, force_arbitrary)
    denominator_type = _denominator_type(method, force_arbitrary)
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
    intermediate_names = frozenset(intermediate_map)
    arbitrary = amplitude_type == "ArbitraryOrderRCCAmplitudes"

    # Large kernels (the spin-adapted CCSDTQ quadruples residuals are ~5000
    # statements each) are super-linear to optimize as one function -- an -O3
    # compile of the containing TU takes 40+ min. Split them into `_partN`
    # sub-functions each accumulating a slice of the terms into `result` (passed
    # by reference); the compiler then optimizes N bounded functions in ~linear
    # total time. Small kernels stay inline (byte-identical). The intermediate
    # builds and amplitude-view bindings are re-emitted per part -- cheap, local,
    # and keeps each part self-contained. `result_rank == 0` (energy) stays inline
    # (a scalar accumulator can't be passed the same way and is always tiny).
    if result_rank > 0 and len(emitted_terms) > _KERNEL_CHUNK_TERMS:
        return _emit_chunked_kernel(
            method, target, lines, emitted_terms, result_type, free_indices,
            denominator_type, amplitude_type, required_intermediates,
            intermediate_names, arbitrary)

    if arbitrary:
        bindings = _amplitude_view_bindings(emitted_terms)
        if bindings:
            lines.append("")
            lines.extend(bindings)
    for i, term in enumerate(emitted_terms, start=1):
        lines.append(f"    // Term {i}")
        lines.append(emit_planck_term(
            term, lhs="result", indent=4,
            intermediate_names=intermediate_names,
            arbitrary_amplitudes=arbitrary))
        lines.append("")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


# Terms-per-function threshold above which a residual kernel is split into
# `_partN` sub-functions (see _emit_kernel). Chosen so the largest CCSDTQ
# quadruples kernels (4613 / 6871 terms) split into ~a dozen bounded parts.
_KERNEL_CHUNK_TERMS = 512


def _emit_chunked_kernel(
    method, target, header_lines, emitted_terms, result_type, free_indices,
    denominator_type, amplitude_type, required_intermediates,
    intermediate_names, arbitrary,
):
    """Emit a large residual kernel as `_partN` sub-functions accumulating into a
    by-reference `result`, plus the main kernel that allocates `result`, calls
    each part, and returns it. Keeps per-function body size bounded so any -O
    level compiles in ~linear time. See _emit_kernel for why."""
    kernel = _kernel_name(method, target)
    n_parts = (len(emitted_terms) + _KERNEL_CHUNK_TERMS - 1) // _KERNEL_CHUNK_TERMS

    out: list[str] = []
    for p in range(n_parts):
        chunk = emitted_terms[p * _KERNEL_CHUNK_TERMS:(p + 1) * _KERNEL_CHUNK_TERMS]
        out.append(f"static void {kernel}_part{p}(")
        out.append(f"    {result_type} &result,")
        out.append("    const CanonicalRHFCCReference &reference,")
        out.append("    const TensorCCBlockCache &mo_blocks,")
        out.append(f"    const {denominator_type} &denominators,")
        out.append(f"    const {amplitude_type} &amplitudes)")
        out.append("{")
        out.append("    const int no = reference.orbital_partition.n_occ;")
        out.append("    const int nv = reference.orbital_partition.n_virt;")
        out.append("    (void)no; (void)nv;")
        if required_intermediates:
            for spec in required_intermediates:
                out.append(
                    f"    const auto {spec.name} = build_{spec.name}("
                    "reference, mo_blocks, denominators, amplitudes);")
        if arbitrary:
            bindings = _amplitude_view_bindings(chunk)
            out.extend(bindings)
        for i, term in enumerate(chunk, start=1):
            out.append(emit_planck_term(
                term, lhs="result", indent=4,
                intermediate_names=intermediate_names,
                arbitrary_amplitudes=arbitrary))
        out.append("}")
        out.append("")

    # main kernel: allocate result, call the parts, return it. header_lines holds
    # the signature + result allocation already; strip its trailing setup we don't
    # reuse (the parts do their own no/nv + intermediates) by appending calls.
    body = list(header_lines)
    for p in range(n_parts):
        body.append(
            f"    {kernel}_part{p}(result, reference, mo_blocks, denominators, amplitudes);")
    body.append("    return result;")
    body.append("}")
    return "\n".join(out + body)


def _emit_intermediate_builder(
    method: str,
    spec: IntermediateSpec,
    sibling_names: frozenset[str] | None = None,
    factor_body: bool = False,
    stride_order: bool = False,
    force_arbitrary: bool = False,
) -> str:
    result_type = _tensor_type(spec.rank)
    amplitude_type = _amplitude_type(method, force_arbitrary)
    denominator_type = _denominator_type(method, force_arbitrary)
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

    # M3.0: emit the builder body as a factored contraction tree (scratch-step
    # pairwise contractions) instead of one flat n-ary nest, when it factors
    # below the flat cost. Cuts the builder's own FLOP scaling (e.g.
    # W_t1t2v_oooovv o^5v^3 -> o^5v^2) at ~0.3x scratch memory. Falls back to the
    # flat lowered emit for single-step (<=2-factor) definitions.
    from ..optimization.factorize import factored_builder_steps
    steps = factored_builder_steps(spec, stride_order=stride_order)
    arbitrary = amplitude_type == "ArbitraryOrderRCCAmplitudes"
    if arbitrary:
        # bind amplitude views once; the factored steps and the flat definition
        # both reference the same leaf amplitudes, so bind from the definition.
        binding_terms = ([s for _, s in steps] if (factor_body and len(steps) > 1)
                         else lowered_definition_terms)
        bindings = _amplitude_view_bindings(binding_terms)
        if bindings:
            lines.extend(bindings)
            lines.append("")
    if factor_body and len(steps) > 1:
        scratch_names = frozenset(
            lhs for lhs, _ in steps if lhs != "result")
        names_in_scope = (sibling_names or frozenset()) | scratch_names
        for lhs, step in steps:
            if lhs != "result":
                stype = _tensor_type(len(step.free_indices))
                lines.append(
                    f"    {stype} {lhs}({_dims_expr(step.free_indices, stype)});")
            lines.append(emit_planck_term(
                step, lhs=lhs, indent=4, intermediate_names=names_in_scope,
                arbitrary_amplitudes=arbitrary))
            lines.append("")
    else:
        for i, term in enumerate(lowered_definition_terms, start=1):
            lines.append(f"    // Definition term {i}")
            lines.append(emit_planck_term(
                term, lhs="result", indent=4, intermediate_names=sibling_names,
                arbitrary_amplitudes=arbitrary))
            lines.append("")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


def _emit_arbitrary_order_kernel_bundle(
    method: str,
    equations: dict[str, list[AlgebraTerm]],
    lowered_equations: dict[str, list[RestrictedClosedShellTerm]],
    force_arbitrary: bool = False,
) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if max_rank < 4 and not force_arbitrary:
        return ""

    # Split targets into the per-rank REFERENCE residuals (residuals_by_rank) and
    # the higher-Sz SECTOR residuals (`quadruples_aaabaaab`, R3.1.3d), keyed
    # (rank, tag). The sector kernels are emitted as their own functions; B3
    # registers them in the bundle's `sector_residuals` + `sector_tags` so B1
    # allocates the sector amplitude block and B4 evaluates/updates it.
    ranked_targets: list[tuple[int, str]] = []
    sector_targets: list[tuple[int, str, str]] = []   # (rank, tag, target)
    for target, terms in equations.items():
        if target == "energy":
            continue
        rank = _target_rank(target, terms, lowered_equations.get(target))
        m = re.search(r"_([ab]+)$", target)
        if m:
            sector_targets.append((rank, m.group(1), target))
        else:
            ranked_targets.append((rank, target))
    ranked_targets.sort()
    sector_targets.sort()

    def _residual_lambda(target: str, indent: str) -> list[str]:
        return [
            f"{indent}[](",
            f"{indent}    const CanonicalRHFCCReference &reference,",
            f"{indent}    const TensorCCBlockCache &mo_blocks,",
            f"{indent}    const ArbitraryOrderDenominatorCache &denominators,",
            f"{indent}    const ArbitraryOrderRCCAmplitudes &amplitudes) -> TensorND",
            f"{indent}{{",
            f"{indent}    return to_tensor_nd("
            f"{_kernel_name(method, target)}(reference, mo_blocks, denominators, amplitudes));",
            f"{indent}}}",
        ]

    lines: list[str] = []
    lines.append(f"GeneratedArbitraryOrderKernels make_generated_{method}_kernels()")
    lines.append("{")
    lines.append("    GeneratedArbitraryOrderKernels kernels;")
    lines.append(f"    kernels.max_excitation_rank = {max_rank};")
    lines.append(f"    kernels.energy = {_kernel_name(method, 'energy')};")
    lines.append(f"    kernels.residuals_by_rank.reserve({max_rank});")
    for rank, target in ranked_targets:
        lines.append("    kernels.residuals_by_rank.push_back(")
        lines.extend(_residual_lambda(target, "        "))
        lines.append("        );")
    for rank, tag, target in sector_targets:
        lines.append(f'    kernels.sector_tags.push_back({{{rank}, "{tag}"}});')
        lines.append("    kernels.sector_residuals.push_back(")
        lines.append(f'        {{{rank}, "{tag}",')
        lines.extend(_residual_lambda(target, "         "))
        lines.append("        });")
    lines.append("    return kernels;")
    lines.append("}")
    return "\n".join(lines)


def emit_planck_translation_unit(
    method: str,
    equations: dict[str, list[AlgebraTerm]],
    intermediates: Sequence[IntermediateSpec] | None = None,
    lowered_equations: dict[str, list[RestrictedClosedShellTerm]] | None = None,
    factor_builder_bodies: bool = False,
    force_arbitrary: bool = False,
    spin_adapted: bool = False,
) -> str:
    """Emit a Planck-compatible C++ translation unit.

    ``factor_builder_bodies`` (M3.0): emit each intermediate's `build_W` as a
    factored contraction tree (scratch-step pairwise contractions) instead of one
    flat n-ary loop nest, cutting the builder's own FLOP scaling. Default off ->
    byte-identical flat emit.

    ``spin_adapted`` (R3.1.3d): the ``equations`` are ALREADY spatial RCC
    ``AlgebraTerm``s from ``spin_adapt_equations`` -- the genuine spatial
    contraction (2*(direct)-(exchange)), the whole point of the spin adaptation.
    Skip the relabel-only ``lower_equations_restricted_closed_shell`` (which would
    both re-apply the defect it was meant to replace AND crash on the multi-Sz
    target names like ``quadruples_aaabaaab``); emit the AlgebraTerms directly."""
    method = method.lower()
    if spin_adapted:
        lowered_equations = {}                  # emit AlgebraTerms directly
    else:
        lowered_equations = lowered_equations or lower_equations_restricted_closed_shell(
            equations
        )
    lines: list[str] = []
    lines.append("// Auto-generated by ccgen")
    lines.append(f"// Planck tensor kernels for {method.upper()}")
    lines.append("")
    lines.append('#include "post_hf/cc/tensor_backend.h"')
    if force_arbitrary or max(parse_cc_level(method), default=0) >= 4:
        lines.append('#include "post_hf/cc/generated_arbitrary_runtime.h"')
    lines.append("")
    lines.append("namespace HartreeFock::Correlation::CC")
    lines.append("{")
    lines.append("")

    if intermediates:
        sibling_names = frozenset(spec.name for spec in intermediates)
        for spec in intermediates:
            if not _is_supported_tensor_rank(spec.rank):
                continue
            lines.append(_emit_intermediate_builder(
                method, spec, sibling_names, factor_body=factor_builder_bodies,
                stride_order=factor_builder_bodies, force_arbitrary=force_arbitrary))
            lines.append("")

    for target, terms in equations.items():
        lines.append(
            _emit_kernel(
                method,
                target,
                terms,
                lowered_terms=lowered_equations.get(target),
                intermediates=intermediates,
                force_arbitrary=force_arbitrary,
            )
        )
        lines.append("")

    bundle_code = _emit_arbitrary_order_kernel_bundle(
        method,
        equations,
        lowered_equations,
        force_arbitrary=force_arbitrary,
    )
    if bundle_code:
        lines.append(bundle_code)
        lines.append("")

    lines.append("} // namespace HartreeFock::Correlation::CC")
    return "\n".join(lines)
