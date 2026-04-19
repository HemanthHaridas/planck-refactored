"""Wick contraction engine for Fermi-vacuum matrix elements."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from .indices import Index
from .tensors import Tensor, delta
from .sqops import SQOp


@dataclass(frozen=True)
class WickResult:
    """One fully-contracted contribution."""

    sign: int
    deltas: tuple[tuple[Index, Index], ...]
    tensor_factors: tuple[Tensor, ...]
    block_edges: tuple[tuple[int, int], ...]


def _can_contract_signature(
    left_kind: str,
    left_space: str,
    right_kind: str,
    right_space: str,
) -> bool:
    if left_kind == "create" and right_kind == "annihilate":
        return left_space in ("occ", "gen") and right_space in ("occ", "gen")

    if left_kind == "annihilate" and right_kind == "create":
        return left_space in ("vir", "gen") and right_space in ("vir", "gen")

    return False


def _can_contract(left: SQOp, right: SQOp) -> bool:
    """Check whether *left* and *right* can form a nonzero contraction."""
    return _can_contract_signature(
        left.kind,
        left.index.space,
        right.kind,
        right.index.space,
    )


@lru_cache(maxsize=None)
def _can_fully_contract(
    signature: tuple[tuple[int, str, str, int], ...],
) -> bool:
    """Quick feasibility check: can all operators in *signature* pair up?

    Counts creators and annihilators by space.  A valid full contraction
    requires:
    - create+occ pairs with annihilate+occ (or gen)
    - annihilate+vir pairs with create+vir (or gen)
    - gen operators are flexible and can fill either role

    This is a necessary (not sufficient) condition — false positives are
    fine, false negatives would lose valid contractions.
    """
    if len(signature) % 2 != 0:
        return False

    # Count (kind, space) combinations, ignoring gen (which is flexible)
    create_occ = 0
    annihilate_occ = 0
    annihilate_vir = 0
    create_vir = 0
    gen_count = 0

    for _, kind, space, _ in signature:
        if space == "gen":
            gen_count += 1
        elif kind == "create" and space == "occ":
            create_occ += 1
        elif kind == "annihilate" and space == "occ":
            annihilate_occ += 1
        elif kind == "annihilate" and space == "vir":
            annihilate_vir += 1
        elif kind == "create" and space == "vir":
            create_vir += 1

    # Occ contractions: create_occ must pair with annihilate_occ
    # Vir contractions: annihilate_vir must pair with create_vir
    # Gen operators can fill any deficit
    occ_deficit = abs(create_occ - annihilate_occ)
    vir_deficit = abs(annihilate_vir - create_vir)

    return occ_deficit + vir_deficit <= gen_count


@lru_cache(maxsize=None)
def _wick_pairings(
    signature: tuple[tuple[int, str, str, int], ...],
) -> tuple[
    tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]],
    ...,
]:
    """Enumerate contraction topologies for a structural operator signature."""
    if not signature:
        return ((1, (), ()),)

    if not _can_fully_contract(signature):
        return ()

    first = signature[0]
    rest = signature[1:]
    _Pairing = tuple[
        int,
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
    ]
    results: list[_Pairing] = []

    for k, partner in enumerate(rest):
        if first[3] == partner[3]:
            continue
        if not _can_contract_signature(
            first[1], first[2], partner[1], partner[2],
        ):
            continue

        remaining = rest[:k] + rest[k + 1:]

        # Early pruning: check if remaining operators can fully contract
        if not _can_fully_contract(remaining):
            continue

        sign_factor = (-1) ** k
        for sub_sign, sub_pairs, sub_edges in _wick_pairings(remaining):
            pair = (first[0], partner[0])
            edge = (first[3], partner[3])
            results.append((
                sign_factor * sub_sign,
                (pair,) + sub_pairs,
                (edge,) + sub_edges,
            ))

    return tuple(results)


def wick_contract(
    sqops: Sequence[SQOp],
    tensors: tuple[Tensor, ...],
    block_ids: Sequence[int] | None = None,
) -> list[WickResult]:
    """Enumerate all fully-contracted Wick pairings."""
    if block_ids is None:
        block_ids = [0] * len(sqops)

    signature = tuple(
        (i, op.kind, op.index.space, bid)
        for i, (op, bid) in enumerate(zip(sqops, block_ids))
    )
    results: list[WickResult] = []

    for sign, pair_positions, edges in _wick_pairings(signature):
        deltas = tuple(
            (sqops[i].index, sqops[j].index) for i, j in pair_positions
        )
        results.append(WickResult(
            sign=sign,
            deltas=deltas,
            tensor_factors=tensors,
            block_edges=edges,
        ))

    return results


def apply_deltas(
    tensors: tuple[Tensor, ...],
    deltas: tuple[tuple[Index, Index], ...],
    protected: tuple[Index, ...] = (),
) -> tuple[Tensor, ...] | None:
    """Apply Kronecker deltas as index substitutions."""
    parent: dict[Index, Index] = {}
    protected_set = frozenset(protected)
    protected_order = {idx: pos for pos, idx in enumerate(protected)}

    def find(x: Index) -> Index:
        while x in parent and parent[x] != x:
            gp = parent.get(parent[x], parent[x])
            parent[x] = gp
            x = gp
        return x

    def _prefer(lhs: Index, rhs: Index) -> tuple[Index, Index]:
        lhs_protected = lhs in protected_set
        rhs_protected = rhs in protected_set

        if lhs_protected != rhs_protected:
            return (lhs, rhs) if lhs_protected else (rhs, lhs)

        if lhs_protected and rhs_protected:
            lhs_rank = protected_order[lhs]
            rhs_rank = protected_order[rhs]
            return (lhs, rhs) if lhs_rank <= rhs_rank else (rhs, lhs)

        if rhs.space != "gen" and lhs.space == "gen":
            return rhs, lhs
        if lhs.space == rhs.space and rhs.name < lhs.name:
            return rhs, lhs
        return lhs, rhs

    def union(a: Index, b: Index) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        if ra.space != "gen" and rb.space != "gen" and ra.space != rb.space:
            return False
        ra, rb = _prefer(ra, rb)
        parent[rb] = ra
        return True

    for p, q in deltas:
        if not union(p, q):
            return None

    all_indices: set[Index] = set()
    for t in tensors:
        all_indices.update(t.indices)
    all_indices.update(protected)

    components: dict[Index, set[Index]] = {}
    for idx in all_indices:
        components.setdefault(find(idx), set()).add(idx)

    extra_factors: list[Tensor] = []
    for root, members in components.items():
        protected_members = sorted(
            (idx for idx in members if idx in protected_set),
            key=lambda idx: protected_order[idx],
        )
        if len(protected_members) <= 1:
            continue
        rep = protected_members[0]
        for idx in protected_members[1:]:
            extra_factors.append(delta(rep, idx))

    mapping: dict[Index, Index] = {}
    for idx in all_indices:
        root = find(idx)
        if root != idx:
            if idx.is_dummy or root.is_dummy:
                mapping[idx] = Index(root.name, root.space, is_dummy=True)
            else:
                mapping[idx] = root

    transformed = tuple(t.reindexed(mapping) for t in tensors)
    if extra_factors:
        transformed = transformed + tuple(extra_factors)
    return transformed
