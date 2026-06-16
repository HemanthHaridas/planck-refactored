"""Typed orbital indices and canonical relabeling utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

OCC_POOL = list("ijklmno")
VIR_POOL = list("abcdefgh")
GEN_POOL = list("pqrstuvw")


def extend_pool(pool: list[str], needed: int) -> None:
    """Grow *pool* in place until it has at least *needed* entries.

    Extra names are formed by appending numeric suffixes to the last
    base letter (e.g. ``i0``, ``i1``, …).
    """
    if len(pool) >= needed:
        return
    base = pool[-1] if pool else "x"
    while len(pool) < needed:
        pool.append(f"{base}{len(pool)}")


@dataclass(frozen=True, order=True, slots=True)
class Index:
    """A single spin-orbital index."""

    name: str
    space: str
    is_dummy: bool = False
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.space not in ("occ", "vir", "gen"):
            raise ValueError(
                f"Invalid space '{self.space}'; expected occ/vir/gen"
            )
        object.__setattr__(
            self,
            "_hash",
            hash((self.name, self.space, self.is_dummy)),
        )

    def __hash__(self) -> int:
        return self._hash

    def as_dummy(self) -> Index:
        return Index(self.name, self.space, is_dummy=True)

    def as_free(self) -> Index:
        return Index(self.name, self.space, is_dummy=False)

    def renamed(self, new_name: str) -> Index:
        return Index(new_name, self.space, self.is_dummy)

    def __repr__(self) -> str:
        return self.name


def make_occ(name: str, dummy: bool = False) -> Index:
    return Index(name, "occ", is_dummy=dummy)


def make_vir(name: str, dummy: bool = False) -> Index:
    return Index(name, "vir", is_dummy=dummy)


def make_gen(name: str, dummy: bool = False) -> Index:
    return Index(name, "gen", is_dummy=dummy)


def _pool_for(space: str) -> list[str]:
    if space == "occ":
        return OCC_POOL
    if space == "vir":
        return VIR_POOL
    if space == "gen":
        return GEN_POOL
    raise ValueError(space)


def relabel_dummies(
    indices: Sequence[Index],
    free: frozenset[Index] | None = None,
) -> dict[Index, Index]:
    """Build a substitution map that renames dummy indices canonically."""
    if free is None:
        free = frozenset()

    reserved: dict[str, set[str]] = {"occ": set(), "vir": set(), "gen": set()}
    for idx in free:
        reserved[idx.space].add(idx.name)

    counters: dict[str, int] = {"occ": 0, "vir": 0, "gen": 0}
    mapping: dict[Index, Index] = {}

    for idx in indices:
        if idx in mapping:
            continue
        if not idx.is_dummy:
            continue
        pool = _pool_for(idx.space)
        while True:
            counters[idx.space] += 1
            extend_pool(pool, counters[idx.space])
            new_name = pool[counters[idx.space] - 1]
            if new_name not in reserved[idx.space]:
                break
        reserved[idx.space].add(new_name)
        mapping[idx] = Index(new_name, idx.space, is_dummy=True)

    return mapping


def apply_relabeling(
    indices: Sequence[Index],
    mapping: dict[Index, Index],
) -> tuple[Index, ...]:
    return tuple(mapping.get(idx, idx) for idx in indices)


def canonical_index_order(indices: Sequence[Index]) -> list[Index]:
    """Sort indices: occ before vir before gen, then alphabetically."""
    rank = {"occ": 0, "vir": 1, "gen": 2}
    return sorted(indices, key=lambda idx: (rank[idx.space], idx.name))
