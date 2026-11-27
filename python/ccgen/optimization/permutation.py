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


# ── Symmetrize-once folding over the external indices ──────────────

# The spin-adapted rank-3 residual is the aabaab block: (ia) and (jb) alpha,
# (kc) beta. It is antisymmetric under a<->b and under i<->j separately and
# symmetric under both (measured to 4e-15 with an aab-correct t3,
# docs/CCGEN_GENERATED_CC_COST.md #9). No symmetry involves (kc).
AAB_TRIPLES_GROUP: tuple[tuple[dict[str, str], int], ...] = (
    ({}, 1),
    ({"a": "b", "b": "a"}, -1),
    ({"i": "j", "j": "i"}, -1),
    ({"a": "b", "b": "a", "i": "j", "j": "i"}, 1),
)


def _rename_free(term: AlgebraTerm, mapping: dict[str, str]) -> AlgebraTerm:
    if not mapping:
        return term

    def ren(idx: Index) -> Index:
        if idx.is_dummy or idx.name not in mapping:
            return idx
        return Index(mapping[idx.name], idx.space, idx.is_dummy)

    return AlgebraTerm(
        coeff=term.coeff,
        factors=tuple(Tensor(f.name, tuple(ren(i) for i in f.indices),
                             f.antisym_groups) for f in term.factors),
        free_indices=term.free_indices,
        summed_indices=term.summed_indices,
        connected=term.connected,
        provenance=term.provenance,
    )


def _term_text(term: AlgebraTerm) -> tuple:
    return tuple(sorted((f.name, tuple(i.name for i in f.indices))
                        for f in term.factors))


def fold_external_symmetry(
    terms: Sequence[AlgebraTerm],
    group: Sequence[tuple[dict[str, str], int]],
) -> list[AlgebraTerm]:
    """Return terms X with R = sum_g s_g g(X), for a residual R that satisfies
    g(R) = s_g R for every (g, s_g) in ``group``.

    Exact given that symmetry of R, with no requirement that the terms come in
    complete orbits: |G| R = sum_g s_g g(R), and for T = h(rep),
    sum_g s_g g(T) = s_h sum_g s_g g(rep). So each orbit contributes
    (sum_T c_T s_h / |G|) rep. ``group`` must be closed under composition and
    inverse, with s a homomorphism and s_{g^-1} = s_g (true for
    AAB_TRIPLES_GROUP and for every `spin_class_group`, whose 3-cycles are not
    their own inverses: whether the rename below is g or g^-1 does not matter,
    since both are in the group with the same sign). Orbits are matched on the
    term's text with dummies left as named, which finds fewer orbits than full
    canonicalization would but never a wrong one. Canonical-form keys were
    tried (2026-11-26): T triples 331 -> 306 terms, UCC CCSDT 802 -> 790, Λ
    residuals no change, at a canonicalizer cost that ran to minutes per
    large Λ-CCSDTQ term. Reverted; a cheaper complete canonical form would be
    the route.
    """
    names = {i.name for t in terms for i in t.free_indices}
    for mapping, _ in group:
        if not set(mapping) <= names:
            raise ValueError(f"group renames {sorted(set(mapping) - names)}, "
                             f"not free indices of these terms ({sorted(names)})")
    order: list[tuple] = []
    reps: dict[tuple, AlgebraTerm] = {}
    kappa: dict[tuple, Fraction] = {}
    for term in terms:
        images = [(_term_text(_rename_free(term, m)), m, s) for m, s in group]
        key, mapping, sign = min(images, key=lambda x: x[0])
        if key not in reps:
            order.append(key)
            reps[key] = _rename_free(term, mapping)
            kappa[key] = Fraction(0)
        kappa[key] += term.coeff * sign
    n = len(group)
    return [reps[k].scaled(kappa[k] / (n * reps[k].coeff))
            for k in order if kappa[k] != 0]


def spin_class_group(term: AlgebraTerm,
                     spins: dict[str, str] | None = None) -> tuple[tuple[dict[str, str], int], ...]:
    """The external symmetry group of a UCC residual block, from its spins.

    Free indices of the same space AND spin are identical particles, so the
    block is antisymmetric under any permutation within such a class. The group
    is the product of the classes' symmetric groups, each element signed by
    parity: order 4 for `doubles_aaaa` or `triples_aabaab`, 36 for
    `triples_aaaaaa`, 1 for `doubles_abab`. Never written by hand
    (docs/CCGEN_UCC_ORBIT_FOLD.md, invariant 1), and never swaps two
    indices of different spin: `abab` -> `baba` is a different block.

    Identity first, so a trivial group is ``(({}, 1),)``.

    ``spins`` gives the free indices' spins directly, for terms whose factors
    carry no spin tags (RCC, :func:`block_symmetry_group`); by default they are
    read from the UCC factor tags.
    """
    from ..spin import ucc_term_index_spins

    if spins is None:
        spins = ucc_term_index_spins(term)
    classes: dict[tuple[str, str], list[str]] = {}
    for idx in term.free_indices:
        if idx.name not in spins:
            raise ValueError(f"free index {idx.name!r} has no spin in {term!r}")
        classes.setdefault((idx.space, spins[idx.name]), []).append(idx.name)

    def _parity(perm: tuple[int, ...]) -> int:
        seen, sign = set(), 1
        for start in range(len(perm)):
            k, length = start, 0
            while k not in seen:
                seen.add(k)
                k = perm[k]
                length += 1
            if length and length % 2 == 0:
                sign = -sign
        return sign

    group: list[tuple[dict[str, str], int]] = [({}, 1)]
    for names in classes.values():
        factor = [({names[k]: names[p[k]] for k in range(len(names))
                    if p[k] != k}, _parity(p))
                  for p in permutations(range(len(names)))]
        group = [({**m1, **m2}, s1 * s2) for m1, s1 in group for m2, s2 in factor]
    return tuple(group)


def block_symmetry_group(term: AlgebraTerm, ucc: bool,
                         k_alpha: int | None = None) -> tuple[tuple[dict[str, str], int], ...]:
    """The symmetry group of one stored amplitude block, from its spins.

    UCC: :func:`spin_class_group` on the block's tags. RCC: the spatial block
    of sector ``k_alpha`` (α on the first ``k_alpha`` occupied/virtual pairs,
    β on the rest; ``spin.independent_spin_blocks``), whose free indices carry
    no tags. Its group is the same class product, plus, for a balanced sector
    (2 k_alpha = n), the closed-shell spin flip: the α pairs swapped with the
    β pairs, sign +. That gives the t2 pair symmetry x(i,j,a,b) = x(j,i,b,a)
    at rank 2 (abab) and AAB_TRIPLES_GROUP at rank 3 (aab), and the groups of
    every higher sector by the same rule.

    The Λ residual of a block is G symmetrized over it (docs/CC_LAMBDA_SCOPE.md).
    """
    if ucc:
        return spin_class_group(term)
    occ = [i.name for i in term.free_indices if i.space == "occ"]
    vir = [i.name for i in term.free_indices if i.space == "vir"]
    n = len(occ)
    if k_alpha is None:
        k_alpha = (n + 1) // 2
    spins = {nm: ("a" if j < k_alpha else "b") for names in (occ, vir)
             for j, nm in enumerate(names)}
    group = spin_class_group(term, spins)
    if 2 * k_alpha != n:
        return group
    flip = {}
    for names in (occ, vir):
        for j in range(k_alpha):
            flip[names[j]], flip[names[k_alpha + j]] = names[k_alpha + j], names[j]
    flipped = []
    for mapping, sign in group:
        h = {x: flip[mapping.get(x, x)] for x in occ + vir}
        flipped.append(({x: y for x, y in h.items() if x != y}, sign))
    return group + tuple(flipped)
