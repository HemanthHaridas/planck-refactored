"""Backend-neutral contraction IR lowered from canonical algebra terms."""

from __future__ import annotations

from dataclasses import dataclass
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
