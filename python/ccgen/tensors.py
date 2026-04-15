"""Tensor symbols with antisymmetry metadata and convenience constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .indices import Index


@dataclass(frozen=True)
class Tensor:
    """A tensor factor appearing in an algebraic term."""

    name: str
    indices: tuple[Index, ...]
    antisym_groups: tuple[tuple[int, ...], ...] = ()

    @property
    def rank(self) -> int:
        return len(self.indices)

    def reindexed(self, mapping: dict[Index, Index]) -> Tensor:
        new_indices = tuple(mapping.get(i, i) for i in self.indices)
        return Tensor(self.name, new_indices, self.antisym_groups)

    def with_indices(self, new_indices: Sequence[Index]) -> Tensor:
        return Tensor(self.name, tuple(new_indices), self.antisym_groups)

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
