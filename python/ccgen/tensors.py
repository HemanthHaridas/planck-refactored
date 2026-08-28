"""Tensor symbols with antisymmetry metadata and convenience constructors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .indices import Index


@dataclass(frozen=True, slots=True)
class Tensor:
    """A tensor factor appearing in an algebraic term."""

    name: str
    indices: tuple[Index, ...]
    antisym_groups: tuple[tuple[int, ...], ...] = ()
    _hash: int = field(init=False, repr=False, compare=False)
    _sort_key: tuple[object, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash((self.name, self.indices, self.antisym_groups)),
        )

    @property
    def rank(self) -> int:
        return len(self.indices)

    @property
    def sort_key(self) -> tuple[object, ...]:
        cached = self._sort_key
        if cached is not None:
            return cached
        cached = (
            self.name,
            tuple((idx.space, idx.name) for idx in self.indices),
        )
        object.__setattr__(self, "_sort_key", cached)
        return cached

    def __hash__(self) -> int:
        return self._hash

    def reindexed(self, mapping: dict[Index, Index]) -> Tensor:
        if not mapping:
            return self

        updated: list[Index] | None = None
        for pos, idx in enumerate(self.indices):
            mapped = mapping.get(idx)
            if mapped is None or mapped == idx:
                if updated is not None:
                    updated.append(idx)
                continue
            if updated is None:
                updated = list(self.indices[:pos])
            updated.append(mapped)

        if updated is None:
            return self
        return Tensor(self.name, tuple(updated), self.antisym_groups)

    def with_indices(self, new_indices: Sequence[Index]) -> Tensor:
        new_indices_tuple = tuple(new_indices)
        if new_indices_tuple == self.indices:
            return self
        return Tensor(self.name, new_indices_tuple, self.antisym_groups)

    def __repr__(self) -> str:
        idx_str = ",".join(repr(i) for i in self.indices)
        return f"{self.name}({idx_str})"


def f(p: Index, q: Index) -> Tensor:
    """Fock matrix element f_p^q."""
    return Tensor("f", (p, q))


def v(p: Index, q: Index, r: Index, s: Index) -> Tensor:
    """Antisymmetrized two-electron integral <pq||rs>."""
    return Tensor("v", (p, q, r, s), antisym_groups=((0, 1), (2, 3)))


def t1(a: Index, i: Index) -> Tensor:
    """Singles amplitude t_i^a."""
    return Tensor("t1", (a, i))


def t2(a: Index, b: Index, i: Index, j: Index) -> Tensor:
    """Doubles amplitude t_{ij}^{ab}."""
    return Tensor("t2", (a, b, i, j), antisym_groups=((0, 1), (2, 3)))


def t3(
    a: Index, b: Index, c: Index,
    i: Index, j: Index, k: Index,
) -> Tensor:
    """Triples amplitude t_{ijk}^{abc}."""
    return Tensor(
        "t3", (a, b, c, i, j, k),
        antisym_groups=((0, 1, 2), (3, 4, 5)),
    )


def tn(n: int, vir: tuple[Index, ...], occ: tuple[Index, ...]) -> Tensor:
    """Generic n-body cluster amplitude t_{i1...in}^{a1...an}.

    *vir* and *occ* must each have exactly *n* indices.  The tensor
    is stored with virtual indices first, then occupied, and both
    groups form antisymmetry groups.
    """
    if len(vir) != n or len(occ) != n:
        raise ValueError(
            f"tn({n}) requires {n} virtual and {n} occupied indices, "
            f"got {len(vir)} and {len(occ)}"
        )
    indices = vir + occ
    vir_group = tuple(range(n))
    occ_group = tuple(range(n, 2 * n))
    antisym = (vir_group, occ_group) if n > 1 else ()
    return Tensor(f"t{n}", indices, antisym_groups=antisym)


def delta(p: Index, q: Index) -> Tensor:
    """Kronecker delta_{pq}."""
    return Tensor("delta", (p, q))


def reindex_tensors(
    tensors: tuple[Tensor, ...],
    mapping: dict[Index, Index],
) -> tuple[Tensor, ...]:
    """Bulk reindex tensor factors, reusing original objects when possible."""
    if not mapping:
        return tensors

    updated: list[Tensor] | None = None
    for pos, tensor in enumerate(tensors):
        new_indices: list[Index] | None = None
        for idx_pos, idx in enumerate(tensor.indices):
            mapped = mapping.get(idx)
            if mapped is None or mapped == idx:
                if new_indices is not None:
                    new_indices.append(idx)
                continue
            if new_indices is None:
                new_indices = list(tensor.indices[:idx_pos])
            new_indices.append(mapped)

        if new_indices is None:
            if updated is not None:
                updated.append(tensor)
            continue
        if updated is None:
            updated = list(tensors[:pos])
        updated.append(Tensor(tensor.name, tuple(new_indices), tensor.antisym_groups))

    if updated is None:
        return tensors
    return tuple(updated)


# ── ERI index symmetries, defined ONCE ────────────────────────────────────
#
# A spatial (NON-antisymmetrized) physicist integral <pq|rs> over real orbitals
# has exactly four index symmetries, all carrying +1:
#
#     identity <pq|rs>,  particle swap <qp|sr>,
#     bra<->ket <rs|pq>, and their product <sr|qp>
#
# They cover all 16 four-index o/v patterns, so nothing is lost by restricting
# to them.
#
# The four SINGLE-swap relations <qp|rs> = -<pq|rs> and <pq|sr> = -<pq|rs> hold
# only for the ANTISYMMETRIZED <pq||rs> that spin-orbital equations use. Applying
# them to a spatial block reads a different integral with a bogus sign.
#
# THIS LIVES IN ONE PLACE ON PURPOSE. Three modules have independently needed
# this distinction -- emit/planck_tensor_cpp.py, optimization/dressing.py and
# lowering/restricted_closed_shell.py -- and each wrote its own table. Two got it
# right and wrote a warning comment; the third kept the 8-fold set, and that is
# how 41 of 288 emitted operator builders came to read the wrong ERI block
# (docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md, D4/D5). Two warning comments did
# not prevent a third copy from being wrong, so the table is shared and gated
# rather than restated.
SPATIAL_ERI_SYMMETRIES: tuple[tuple[tuple[int, int, int, int], int], ...] = (
    ((0, 1, 2, 3), +1),
    ((1, 0, 3, 2), +1),
    ((2, 3, 0, 1), +1),
    ((3, 2, 1, 0), +1),
)

# The full 8-fold group of the ANTISYMMETRIZED <pq||rs>. Valid ONLY on
# antisymmetrized integrals; the four members absent from the spatial set above
# are the odd-parity ones. Kept here beside the spatial set so the difference is
# visible rather than rediscovered.
ANTISYMMETRIZED_ERI_SYMMETRIES: tuple[tuple[tuple[int, int, int, int], int], ...] = (
    ((0, 1, 2, 3), +1),
    ((1, 0, 2, 3), -1),
    ((0, 1, 3, 2), -1),
    ((1, 0, 3, 2), +1),
    ((2, 3, 0, 1), +1),
    ((3, 2, 0, 1), -1),
    ((2, 3, 1, 0), -1),
    ((3, 2, 1, 0), +1),
)
