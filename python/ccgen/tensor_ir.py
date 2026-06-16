"""Backend-neutral contraction IR lowered from canonical algebra terms.

Provides ``BackendTerm`` (basic contraction metadata) and
``BackendTermEx`` (extended with optimization hints for BLAS pattern
detection, memory layout, and contraction ordering).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

from .indices import Index
from .project import AlgebraTerm, manifold_rank
from .tensors import Tensor


@dataclass(frozen=True)
class TensorSlot:
    """One indexed slot of a tensor factor inside a contraction term."""

    factor_index: int
    axis: int
    tensor_name: str
    index: Index


@dataclass(frozen=True)
class IndexContraction:
    """All tensor slots tied together by a shared symbolic index."""

    index: Index
    slots: tuple[TensorSlot, ...]
    is_free: bool


@dataclass(frozen=True)
class BackendTerm:
    """A backend-ready tensor contraction term."""

    lhs_name: str
    lhs_indices: tuple[Index, ...]
    coefficient: Fraction
    rhs_factors: tuple[Tensor, ...]
    contractions: tuple[IndexContraction, ...]
    summed_indices: tuple[Index, ...]
    free_indices: tuple[Index, ...]
    connected: bool

    @property
    def rank(self) -> int:
        return len(self.lhs_indices)

    def contraction_for(self, index: Index) -> IndexContraction | None:
        for contraction in self.contractions:
            if contraction.index == index:
                return contraction
        return None


# ── Extended IR with optimization hints ────────────────────────────


@dataclass(frozen=True)
class BLASHint:
    """Describes a recognized BLAS pattern in a contraction term.

    For GEMM patterns of the form C += alpha * op(A) * op(B):
    - ``pattern``: one of "gemm_nn", "gemm_nt", "gemm_tn", "gemm_tt"
      (N=no-transpose, T=transpose)
    - ``a_tensor``, ``b_tensor``: names of the A and B operands
    - ``a_indices``, ``b_indices``: index tuples for A and B
    - ``contraction_indices``: the shared summation indices (k-dimension)
    - ``m_indices``, ``n_indices``: the output dimensions
    """

    pattern: str  # "gemm_nn" | "gemm_nt" | "gemm_tn" | "gemm_tt"
    a_tensor: str
    b_tensor: str
    a_indices: tuple[Index, ...]
    b_indices: tuple[Index, ...]
    contraction_indices: tuple[Index, ...]
    m_indices: tuple[Index, ...]
    n_indices: tuple[Index, ...]


@dataclass(frozen=True)
class BackendTermEx(BackendTerm):
    """Extended backend term with optimization hints.

    Inherits all fields from ``BackendTerm`` and adds metadata that
    backend emitters can use to generate more efficient code:

    - ``memory_layout``: per-tensor storage order hints
    - ``blocking_hint``: suggested tile sizes per index
    - ``reuse_key``: identifies shared sub-contractions
    - ``computation_order``: recommended summation index ordering
    - ``blas_hint``: recognized BLAS pattern (GEMM, etc.)
    - ``estimated_flops``: estimated floating-point operations
    """

    memory_layout: dict[str, str] = field(default_factory=dict)
    blocking_hint: dict[str, int] = field(default_factory=dict)
    reuse_key: str | None = None
    computation_order: tuple[int, ...] = ()
    blas_hint: BLASHint | None = None
    estimated_flops: int = 0

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BackendTermEx):
            return NotImplemented
        return self is other


def default_lhs_name(target: str) -> str:
    """Map a target manifold name to a canonical residual symbol."""
    if target == "energy":
        return "E_CC"
    try:
        rank = manifold_rank(target)
    except ValueError:
        return target
    return f"R{rank}"


def lower_term(
    term: AlgebraTerm,
    target: str,
    lhs_name: str | None = None,
) -> BackendTerm:
    """Lower one algebraic term to explicit contraction metadata."""
    if lhs_name is None:
        lhs_name = default_lhs_name(target)

    slots_by_index: dict[Index, list[TensorSlot]] = {}
    for factor_index, factor in enumerate(term.factors):
        for axis, index in enumerate(factor.indices):
            slots_by_index.setdefault(index, []).append(TensorSlot(
                factor_index=factor_index,
                axis=axis,
                tensor_name=factor.name,
                index=index,
            ))

    ordered_indices = term.free_indices + term.summed_indices
    contractions = tuple(
        IndexContraction(
            index=index,
            slots=tuple(slots_by_index.get(index, ())),
            is_free=index in term.free_indices,
        )
        for index in ordered_indices
    )

    return BackendTerm(
        lhs_name=lhs_name,
        lhs_indices=term.free_indices,
        coefficient=term.coeff,
        rhs_factors=term.factors,
        contractions=contractions,
        summed_indices=term.summed_indices,
        free_indices=term.free_indices,
        connected=term.connected,
    )


def lower_terms(
    terms: Sequence[AlgebraTerm],
    target: str,
    lhs_name: str | None = None,
) -> list[BackendTerm]:
    """Lower a list of algebraic terms for one target manifold."""
    if lhs_name is None:
        lhs_name = default_lhs_name(target)
    return [
        lower_term(term, target=target, lhs_name=lhs_name)
        for term in terms
    ]


def lower_equations(
    equations: dict[str, list[AlgebraTerm]],
) -> dict[str, list[BackendTerm]]:
    """Lower a dict of canonical algebra equations to contraction IR."""
    return {
        target: lower_terms(terms, target=target)
        for target, terms in equations.items()
    }


# ── BLAS pattern detection ─────────────────────────────────────────


def _detect_gemm(term: BackendTerm) -> BLASHint | None:
    """Detect GEMM pattern: C += alpha * A(i,k) * B(k,j).

    A GEMM pattern requires exactly two RHS factors sharing one or
    more summation indices (the k-dimension), with the remaining
    indices mapping to the LHS (the m and n dimensions).
    """
    if len(term.rhs_factors) != 2:
        return None
    if not term.summed_indices:
        return None

    fa, fb = term.rhs_factors
    summed_set = frozenset(term.summed_indices)

    # Find contraction indices (shared between A and B and summed)
    a_indices = set(fa.indices)
    b_indices = set(fb.indices)
    contraction = a_indices & b_indices & summed_set

    if not contraction:
        return None

    # m-indices: in A but not contracted (appear in LHS)
    # n-indices: in B but not contracted (appear in LHS)
    m_idx = tuple(idx for idx in fa.indices if idx not in contraction)
    n_idx = tuple(idx for idx in fb.indices if idx not in contraction)
    k_idx = tuple(sorted(contraction, key=lambda x: (x.space, x.name)))

    if not m_idx and not n_idx:
        # Pure dot product, not GEMM
        return None

    # Determine transpose flags
    # Convention: "nn" means A is (m,k) and B is (k,n)
    # Check if contraction indices are trailing in A and leading in B
    a_positions = [i for i, idx in enumerate(fa.indices) if idx in contraction]
    b_positions = [i for i, idx in enumerate(fb.indices) if idx in contraction]

    a_trailing = a_positions == list(range(len(fa.indices) - len(k_idx), len(fa.indices)))
    b_leading = b_positions == list(range(len(k_idx)))

    if a_trailing and b_leading:
        pattern = "gemm_nn"
    elif a_trailing and not b_leading:
        pattern = "gemm_nt"
    elif not a_trailing and b_leading:
        pattern = "gemm_tn"
    else:
        pattern = "gemm_tt"

    return BLASHint(
        pattern=pattern,
        a_tensor=fa.name,
        b_tensor=fb.name,
        a_indices=fa.indices,
        b_indices=fb.indices,
        contraction_indices=k_idx,
        m_indices=m_idx,
        n_indices=n_idx,
    )


def _estimate_flops(term: BackendTerm) -> int:
    """Estimate FLOP count for a contraction term.

    Uses the product of all loop dimensions as a rough estimate.
    Each inner iteration performs one multiply-add (2 FLOPs) per
    tensor factor.
    """
    # Count unique indices and estimate dimension sizes
    # Use symbolic sizes: occ ~ 10, vir ~ 50 as defaults
    SIZE_EST = {"occ": 10, "vir": 50, "gen": 30}

    all_indices = set(term.free_indices) | set(term.summed_indices)
    total_iterations = 1
    for idx in all_indices:
        total_iterations *= SIZE_EST.get(idx.space, 30)

    # Each iteration: one multiply per factor pair + one add
    n_factors = len(term.rhs_factors)
    flops_per_iter = max(2 * n_factors - 1, 1)
    return total_iterations * flops_per_iter


def _optimal_contraction_order(term: BackendTerm) -> tuple[int, ...]:
    """Determine optimal summation index ordering.

    Heuristic: sum over indices with smaller dimension first
    (occupied before virtual) to minimize intermediate sizes.
    """
    SIZE_RANK = {"occ": 0, "vir": 1, "gen": 2}
    indexed = [
        (SIZE_RANK.get(idx.space, 2), idx.name, i)
        for i, idx in enumerate(term.summed_indices)
    ]
    indexed.sort()
    return tuple(i for _, _, i in indexed)


def lower_term_ex(
    term: AlgebraTerm,
    target: str,
    lhs_name: str | None = None,
    detect_blas: bool = True,
    tile_occ: int = 16,
    tile_vir: int = 16,
) -> BackendTermEx:
    """Lower one algebraic term to extended contraction IR with hints."""
    base = lower_term(term, target, lhs_name)

    blas = _detect_gemm(base) if detect_blas else None
    flops = _estimate_flops(base)
    order = _optimal_contraction_order(base)

    # Build blocking hints from free indices
    blocking: dict[str, int] = {}
    for idx in base.free_indices:
        if idx.space == "occ":
            blocking[idx.name] = tile_occ
        elif idx.space == "vir":
            blocking[idx.name] = tile_vir

    # Memory layout: default to row-major for all tensors
    layout: dict[str, str] = {}
    for fac in base.rhs_factors:
        if fac.name not in layout:
            layout[fac.name] = "row_major"

    return BackendTermEx(
        lhs_name=base.lhs_name,
        lhs_indices=base.lhs_indices,
        coefficient=base.coefficient,
        rhs_factors=base.rhs_factors,
        contractions=base.contractions,
        summed_indices=base.summed_indices,
        free_indices=base.free_indices,
        connected=base.connected,
        memory_layout=layout,
        blocking_hint=blocking,
        reuse_key=None,
        computation_order=order,
        blas_hint=blas,
        estimated_flops=flops,
    )


def lower_equations_ex(
    equations: dict[str, list[AlgebraTerm]],
    detect_blas: bool = True,
    tile_occ: int = 16,
    tile_vir: int = 16,
) -> dict[str, list[BackendTermEx]]:
    """Lower equations to extended contraction IR with optimization hints."""
    result: dict[str, list[BackendTermEx]] = {}
    for target, terms in equations.items():
        lhs_name = default_lhs_name(target)
        result[target] = [
            lower_term_ex(
                t, target, lhs_name,
                detect_blas=detect_blas,
                tile_occ=tile_occ,
                tile_vir=tile_vir,
            )
            for t in terms
        ]
    return result
