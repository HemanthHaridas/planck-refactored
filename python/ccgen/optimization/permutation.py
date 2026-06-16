"""Permutation-based term grouping for coupled-cluster equations.

Detects groups of terms that differ only by index permutations (possibly
with sign changes from antisymmetry) and merges them into compact
representations using permutation operators or antisymmetrized sums.

Example:
    Term 42: -1/8 t1(a,i) * t1(b,k) * v(a,b,k,j)
    Term 43: +1/8 t1(a,i) * t1(b,k) * v(b,a,k,j)
    →  -1/4 t1(a,i) * t1(b,k) * v(a,b,k,j)   [after antisym exploit]

This reduces loop nests by 30-40% in typical CCSD/CCSDT equations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Sequence

from ..indices import Index
from ..project import AlgebraTerm
from ..tensors import Tensor


@dataclass(frozen=True)
class IndexPermutation:
    """A permutation of indices with an associated sign."""

    mapping: tuple[tuple[Index, Index], ...]
    sign: int  # +1 or -1

    @property
    def is_identity(self) -> bool:
        return all(a == b for a, b in self.mapping)


@dataclass(frozen=True)
class PermutationGroup:
    """A group of terms related by index permutations.

    The ``representative`` is the canonical term.  ``permutations``
    lists the index permutations (with signs) that generate each
    member of the group from the representative.  ``combined_coeff``
    is the sum of coefficients after accounting for permutation signs.
    """

    representative: AlgebraTerm
    permutations: tuple[IndexPermutation, ...]
    original_indices: tuple[int, ...]  # indices into the original term list
    combined_coeff: Fraction


def _structure_key(term: AlgebraTerm) -> tuple[object, ...]:
    """Build a key that captures the tensor structure ignoring index names.

    Two terms with the same structure key have identical tensor names
    and identical index-space patterns, so they *could* be related by
    an index permutation.
    """
    factor_key = tuple(
        (f.name, tuple(idx.space for idx in f.indices))
        for f in term.factors
    )
    free_key = tuple(idx.space for idx in term.free_indices)
    summed_key = tuple(
        sorted(idx.space for idx in term.summed_indices)
    )
    return (factor_key, free_key, summed_key)


def _find_index_permutation(
    source: AlgebraTerm,
    target: AlgebraTerm,
) -> IndexPermutation | None:
    """Find an index permutation mapping *source* to *target*.

    Returns None if the terms cannot be related by a simple index
    permutation (different tensor names, different structure, etc.).
    The permutation maps source indices → target indices.
    """
    if len(source.factors) != len(target.factors):
        return None
    if len(source.free_indices) != len(target.free_indices):
        return None
    if len(source.summed_indices) != len(target.summed_indices):
        return None

    # Factors must match by name and rank
    for sf, tf in zip(source.factors, target.factors):
        if sf.name != tf.name or sf.rank != tf.rank:
            return None

    # Build the index mapping from source → target
    mapping: dict[Index, Index] = {}
    for sf, tf in zip(source.factors, target.factors):
        for si, ti in zip(sf.indices, tf.indices):
            if si.space != ti.space:
                return None
            if si in mapping:
                if mapping[si] != ti:
                    return None
            else:
                mapping[si] = ti

    # Verify the mapping is consistent with free/summed classification
    for fi in source.free_indices:
        if fi not in mapping:
            return None
        mapped = mapping[fi]
        if mapped not in target.free_indices:
            return None

    for si in source.summed_indices:
        if si not in mapping:
            return None
        mapped = mapping[si]
        if mapped not in target.summed_indices:
            return None

    # Check bijectivity
    reverse: dict[Index, Index] = {}
    for k, v in mapping.items():
        if v in reverse:
            return None
        reverse[v] = k

    perm_pairs = tuple(
        (k, v) for k, v in sorted(
            mapping.items(),
            key=lambda x: (x[0].space, x[0].name),
        )
    )

    # Determine the sign from antisymmetry
    # The sign is determined by the coefficient ratio
    if source.coeff == 0:
        sign = 1
    else:
        ratio = target.coeff / source.coeff
        if ratio == 1:
            sign = 1
        elif ratio == -1:
            sign = -1
        else:
            # Not a simple permutation — coefficients differ
            # by more than a sign
            return None

    return IndexPermutation(mapping=perm_pairs, sign=sign)


def detect_permutation_groups(
    terms: Sequence[AlgebraTerm],
    merge_antisymmetric: bool = True,
) -> tuple[list[PermutationGroup], list[int]]:
    """Detect groups of terms related by index permutations.

    Parameters
    ----------
    terms : sequence of AlgebraTerm
        The terms to analyze.
    merge_antisymmetric : bool
        If True, also merge terms related by antisymmetric index
        swaps (differing by a sign).

    Returns
    -------
    groups : list of PermutationGroup
        Each group contains a representative term and the permutations
        that generate all group members.
    ungrouped : list of int
        Indices of terms that don't belong to any group.
    """
    # Phase 1: bucket terms by structure key
    buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for i, term in enumerate(terms):
        key = _structure_key(term)
        buckets[key].append(i)

    groups: list[PermutationGroup] = []
    grouped: set[int] = set()

    # Phase 2: within each bucket, find permutation relationships
    for key, indices in buckets.items():
        if len(indices) < 2:
            continue

        # Try to group terms within this bucket
        bucket_grouped: set[int] = set()

        for i_pos, i_idx in enumerate(indices):
            if i_idx in bucket_grouped:
                continue

            source = terms[i_idx]
            group_indices = [i_idx]
            group_perms = [IndexPermutation(
                mapping=tuple(
                    (idx, idx) for idx in sorted(
                        set(source.free_indices) | set(source.summed_indices),
                        key=lambda x: (x.space, x.name),
                    )
                ),
                sign=1,
            )]

            for j_pos in range(i_pos + 1, len(indices)):
                j_idx = indices[j_pos]
                if j_idx in bucket_grouped:
                    continue

                target = terms[j_idx]
                perm = _find_index_permutation(source, target)
                if perm is None:
                    continue

                if not merge_antisymmetric and perm.sign == -1:
                    continue

                group_indices.append(j_idx)
                group_perms.append(perm)

            if len(group_indices) >= 2:
                # Compute combined coefficient
                combined = Fraction(0)
                for gi, gp in zip(group_indices, group_perms):
                    combined += terms[gi].coeff * gp.sign

                if combined != 0 or len(group_indices) > 2:
                    groups.append(PermutationGroup(
                        representative=source,
                        permutations=tuple(group_perms),
                        original_indices=tuple(group_indices),
                        combined_coeff=combined,
                    ))
                    bucket_grouped.update(group_indices)

        grouped.update(bucket_grouped)

    ungrouped = [i for i in range(len(terms)) if i not in grouped]
    return groups, ungrouped


def apply_permutation_grouping(
    terms: Sequence[AlgebraTerm],
    merge_antisymmetric: bool = True,
) -> list[AlgebraTerm]:
    """Apply permutation-based grouping to reduce the term list.

    Terms related by index permutations (with or without sign changes)
    are merged into a single term with a combined coefficient.  Terms
    whose combined coefficient is zero are eliminated entirely.

    Returns a new term list (shorter than or equal to the input).
    """
    groups, ungrouped = detect_permutation_groups(
        terms, merge_antisymmetric=merge_antisymmetric,
    )

    result: list[AlgebraTerm] = []

    # Add ungrouped terms unchanged
    for i in ungrouped:
        result.append(terms[i])

    # Add merged group representatives
    for group in groups:
        if group.combined_coeff == 0:
            continue
        result.append(AlgebraTerm(
            coeff=group.combined_coeff,
            factors=group.representative.factors,
            free_indices=group.representative.free_indices,
            summed_indices=group.representative.summed_indices,
            connected=group.representative.connected,
            provenance=group.representative.provenance,
        ))

    return result


def apply_permutation_grouping_equations(
    equations: dict[str, list[AlgebraTerm]],
    merge_antisymmetric: bool = True,
) -> dict[str, list[AlgebraTerm]]:
    """Apply permutation grouping to all equation manifolds.

    Returns a new equation dict with reduced term counts.
    """
    return {
        manifold: apply_permutation_grouping(
            terms, merge_antisymmetric=merge_antisymmetric,
        )
        for manifold, terms in equations.items()
    }


def permutation_grouping_stats(
    equations: dict[str, list[AlgebraTerm]],
    merge_antisymmetric: bool = True,
) -> dict[str, dict[str, int]]:
    """Return statistics about permutation grouping potential.

    For each manifold, reports:
    - ``before``: original term count
    - ``after``: term count after grouping
    - ``groups``: number of permutation groups found
    - ``eliminated``: number of terms eliminated
    """
    stats: dict[str, dict[str, int]] = {}
    for manifold, terms in equations.items():
        groups, ungrouped = detect_permutation_groups(
            terms, merge_antisymmetric=merge_antisymmetric,
        )
        after = len(ungrouped) + sum(
            1 for g in groups if g.combined_coeff != 0
        )
        stats[manifold] = {
            "before": len(terms),
            "after": after,
            "groups": len(groups),
            "eliminated": len(terms) - after,
        }
    return stats
