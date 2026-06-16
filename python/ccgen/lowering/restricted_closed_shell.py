"""Restricted closed-shell lowering for spin-orbital algebra terms.

This module does not attempt a full symbolic spin summation yet. Instead, it
produces a backend-neutral spatial-orbital layout IR from the projected
spin-orbital equations:

- every factor carries an explicit occupied/virtual block signature
- a canonical spatial index order is derived with a stable occ/vir/gen split
- the permutation from source slots to spatial layout is recorded explicitly

That gives emitters and validation code a reliable bridge between the generic
ccgen algebra and restricted spatial backends such as Planck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ..indices import Index
from ..project import AlgebraTerm, manifold_rank
from ..tensors import Tensor

OrbitalModel = Literal["restricted_closed_shell"]

_CANONICAL_ERI_BLOCKS: dict[str, tuple[str, str, str, str]] = {
    "oooo": ("occ", "occ", "occ", "occ"),
    "ooov": ("occ", "occ", "occ", "vir"),
    "oovv": ("occ", "occ", "vir", "vir"),
    "ovov": ("occ", "vir", "occ", "vir"),
    "ovvo": ("occ", "vir", "vir", "occ"),
    "ovvv": ("occ", "vir", "vir", "vir"),
    "vvvv": ("vir", "vir", "vir", "vir"),
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


def _space_priority(space: str) -> int:
    order = {"occ": 0, "vir": 1, "gen": 2}
    return order.get(space, 3)


def _stable_spatial_indices(indices: Sequence[Index]) -> tuple[tuple[Index, ...], tuple[int, ...]]:
    """Return indices in stable spatial order and the source-slot permutation."""
    slots = list(enumerate(indices))
    slots.sort(key=lambda item: (_space_priority(item[1].space), item[0]))
    permutation = tuple(slot for slot, _ in slots)
    ordered = tuple(idx for _, idx in slots)
    return ordered, permutation


def _inverse_permutation(perm: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(perm)
    for i, value in enumerate(perm):
        inverse[value] = i
    return tuple(inverse)


def _lower_eri_indices(
    indices: Sequence[Index],
) -> tuple[tuple[Index, ...], tuple[int, ...], int]:
    spaces = tuple(idx.space for idx in indices)
    if "gen" in spaces:
        ordered, permutation = _stable_spatial_indices(indices)
        return ordered, permutation, 1

    for block_spaces in _CANONICAL_ERI_BLOCKS.values():
        for perm, phase in _ERI_SYMMETRY_PERMUTATIONS:
            transformed = tuple(block_spaces[i] for i in perm)
            if transformed != spaces:
                continue
            inverse = _inverse_permutation(perm)
            ordered = tuple(indices[i] for i in inverse)
            return ordered, inverse, phase

    ordered, permutation = _stable_spatial_indices(indices)
    return ordered, permutation, 1


def _space_signature(indices: Sequence[Index]) -> tuple[str, ...]:
    return tuple(idx.space for idx in indices)


@dataclass(frozen=True)
class LoweredTensorFactor:
    """A tensor factor lowered to restricted closed-shell spatial layout."""

    source: Tensor
    block_signature: tuple[str, ...]
    spatial_indices: tuple[Index, ...]
    spatial_permutation: tuple[int, ...]
    phase: int = 1

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def spatial_signature(self) -> tuple[str, ...]:
        return _space_signature(self.spatial_indices)

    @property
    def spatial_block(self) -> str:
        glyph = {"occ": "o", "vir": "v", "gen": "g"}
        return "".join(glyph.get(space, "x") for space in self.spatial_signature)


@dataclass(frozen=True)
class RestrictedClosedShellTerm:
    """A projected term annotated for restricted closed-shell backends."""

    source: AlgebraTerm
    manifold: str
    orbital_model: OrbitalModel
    factors: tuple[LoweredTensorFactor, ...]
    canonical_free_indices: tuple[Index, ...]
    canonical_summed_indices: tuple[Index, ...]
    excitation_rank: int

    @property
    def coeff(self):
        return self.source.coeff

    @property
    def free_indices(self) -> tuple[Index, ...]:
        return self.source.free_indices

    @property
    def summed_indices(self) -> tuple[Index, ...]:
        return self.source.summed_indices

    @property
    def connected(self) -> bool:
        return self.source.connected


def lower_tensor_factor_restricted_closed_shell(tensor: Tensor) -> LoweredTensorFactor:
    if tensor.name == "v" and tensor.rank == 4:
        spatial_indices, spatial_permutation, phase = _lower_eri_indices(
            tensor.indices
        )
    else:
        spatial_indices, spatial_permutation = _stable_spatial_indices(
            tensor.indices
        )
        phase = 1
    return LoweredTensorFactor(
        source=tensor,
        block_signature=_space_signature(tensor.indices),
        spatial_indices=spatial_indices,
        spatial_permutation=spatial_permutation,
        phase=phase,
    )


def lower_term_restricted_closed_shell(
    term: AlgebraTerm,
    manifold: str,
) -> RestrictedClosedShellTerm:
    canonical_free, _ = _stable_spatial_indices(term.free_indices)
    canonical_summed, _ = _stable_spatial_indices(term.summed_indices)
    return RestrictedClosedShellTerm(
        source=term,
        manifold=manifold,
        orbital_model="restricted_closed_shell",
        factors=tuple(
            lower_tensor_factor_restricted_closed_shell(factor)
            for factor in term.factors
        ),
        canonical_free_indices=canonical_free,
        canonical_summed_indices=canonical_summed,
        excitation_rank=manifold_rank(manifold),
    )


def lower_equations_restricted_closed_shell(
    equations: dict[str, list[AlgebraTerm]],
) -> dict[str, list[RestrictedClosedShellTerm]]:
    """Lower projected spin-orbital equations to restricted spatial layout IR."""
    lowered: dict[str, list[RestrictedClosedShellTerm]] = {}
    for manifold, terms in equations.items():
        lowered[manifold] = [
            lower_term_restricted_closed_shell(term, manifold)
            for term in terms
        ]
    return lowered
