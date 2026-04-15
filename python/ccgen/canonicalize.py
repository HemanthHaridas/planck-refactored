"""Canonicalization: dummy relabeling, tensor normalization, term merging."""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

from .indices import Index, relabel_dummies
from .tensors import Tensor
from .project import AlgebraTerm


def _canonical_ordering_for_group(
    indices: list[Index],
    positions: tuple[int, ...],
) -> tuple[list[Index], int]:
    """Find the lexicographically smallest permutation within *positions*."""
    orig = [indices[p] for p in positions]
    best_perm = None
    best_key = None
    best_sign = 1

    for perm in permutations(range(len(positions))):
        reordered = [orig[p] for p in perm]
        key = tuple((idx.space, idx.name) for idx in reordered)
        inversions = 0
        for x in range(len(perm)):
            for y in range(x + 1, len(perm)):
                if perm[x] > perm[y]:
                    inversions += 1
        sign = (-1) ** inversions

        if best_key is None or key < best_key:
            best_key = key
            best_perm = reordered
            best_sign = sign

    assert best_perm is not None
    result = list(indices)
    for k, p in enumerate(positions):
        result[p] = best_perm[k]
    return result, best_sign


def canonicalize_tensor(tensor: Tensor) -> tuple[Tensor, int]:
    """Normalize a tensor's index ordering using its antisymmetry groups."""
    if not tensor.antisym_groups:
        return tensor, 1

    for group in tensor.antisym_groups:
        slots = [
            (tensor.indices[p].space, tensor.indices[p].name)
            for p in group
        ]
        if len(slots) != len(set(slots)):
            return tensor, 0

    indices = list(tensor.indices)
    total_sign = 1

    for group in tensor.antisym_groups:
        indices, sign = _canonical_ordering_for_group(indices, group)
        total_sign *= sign

    return tensor.with_indices(indices), total_sign


def _tensor_sort_key(t: Tensor) -> tuple[object, ...]:
    return (t.name, tuple((i.space, i.name) for i in t.indices))


def _all_indices_ordered(term: AlgebraTerm) -> list[Index]:
    seen: set[Index] = set()
    ordered: list[Index] = []
    for fac in term.factors:
        for idx in fac.indices:
            if idx not in seen:
                ordered.append(idx)
                seen.add(idx)
    return ordered


def relabel_term_dummies(term: AlgebraTerm) -> AlgebraTerm:
    """Relabel dummy (summed) indices canonically by first appearance."""
    free_set = frozenset(term.free_indices)
    all_idx = _all_indices_ordered(term)

    dummies_in_order = [
        idx.as_dummy() for idx in all_idx if idx not in free_set
    ]
    mapping = relabel_dummies(dummies_in_order, free=free_set)

    full_mapping: dict[Index, Index] = {}
    for idx in all_idx:
        if idx in free_set:
            continue
        dummy_ver = idx.as_dummy()
        if dummy_ver in mapping:
            full_mapping[idx] = mapping[dummy_ver]
        elif idx != dummy_ver:
            full_mapping[idx] = dummy_ver

    new_factors = tuple(f.reindexed(full_mapping) for f in term.factors)
    new_summed = tuple(
        sorted(
            set(full_mapping.get(s, s) for s in term.summed_indices),
            key=lambda x: (x.space, x.name),
        )
    )

    return AlgebraTerm(
        coeff=term.coeff,
        factors=new_factors,
        free_indices=term.free_indices,
        summed_indices=new_summed,
        connected=term.connected,
        provenance=term.provenance,
    )


def canonicalize_term(term: AlgebraTerm) -> AlgebraTerm:
    """Apply the full canonicalization pipeline to a single term."""
    sign = 1
    canon_factors = []
    for fac in term.factors:
        cf, s = canonicalize_tensor(fac)
        canon_factors.append(cf)
        sign *= s

    if sign == 0:
        return AlgebraTerm(
            coeff=0,
            factors=tuple(canon_factors),
            free_indices=term.free_indices,
            summed_indices=term.summed_indices,
            connected=term.connected,
            provenance=term.provenance,
        )

    term = AlgebraTerm(
        coeff=term.coeff * sign,
        factors=tuple(canon_factors),
        free_indices=term.free_indices,
        summed_indices=term.summed_indices,
        connected=term.connected,
        provenance=term.provenance,
    )

    # Relabel once before sorting so arbitrary fresh dummy suffixes do not
    # influence factor order, then sort and relabel again to normalize the
    # final first-appearance convention.
    term = relabel_term_dummies(term)
    canon_factors = list(term.factors)
    canon_factors.sort(key=_tensor_sort_key)

    term = AlgebraTerm(
        coeff=term.coeff,
        factors=tuple(canon_factors),
        free_indices=term.free_indices,
        summed_indices=term.summed_indices,
        connected=term.connected,
        provenance=term.provenance,
    )

    term = relabel_term_dummies(term)
    return term


def _term_signature(term: AlgebraTerm) -> tuple[object, ...]:
    fac_key = tuple(
        (f.name, tuple((i.space, i.name) for i in f.indices))
        for f in term.factors
    )
    free_key = tuple((i.space, i.name) for i in term.free_indices)
    return (fac_key, free_key)


def merge_like_terms(terms: Sequence[AlgebraTerm]) -> list[AlgebraTerm]:
    """Combine terms with identical tensor structure by adding coefficients."""
    buckets: dict[tuple[object, ...], AlgebraTerm] = {}
    order: list[tuple[object, ...]] = []

    for t in terms:
        sig = _term_signature(t)
        if sig in buckets:
            old = buckets[sig]
            buckets[sig] = AlgebraTerm(
                coeff=old.coeff + t.coeff,
                factors=old.factors,
                free_indices=old.free_indices,
                summed_indices=old.summed_indices,
                connected=old.connected and t.connected,
                provenance=old.provenance,
            )
        else:
            buckets[sig] = t
            order.append(sig)

    return [
        buckets[sig] for sig in order if buckets[sig].coeff != 0
    ]
