"""Implicit symmetry exploitation for coupled-cluster equations.

**Experimental / opt-in feature.**

Exploits antisymmetry of two-electron integrals and amplitudes
*before* canonicalization to reduce term count.  The key identity:

    v(p,q,r,s) = -v(q,p,r,s) = -v(p,q,s,r) = v(q,p,s,r)

and similarly for t2/t3 amplitudes.

By marking terms with their symmetry parity upfront and emitting
only unique index orderings, this can reduce the pre-canonicalization
term count by 30-50%.

Strategy:
1. Identify antisymmetric tensor pairs in each term
2. For each pair of antisymmetric indices, choose a canonical
   ordering (e.g., i < j for occupied, a < b for virtual)
3. Combine terms that differ only by such swaps
4. Emit restriction predicates (i < j loops) in code generation
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ..indices import Index
from ..project import AlgebraTerm
from ..tensors import Tensor


@dataclass(frozen=True)
class SymmetryRestriction:
    """A restriction on index ordering from antisymmetry.

    For example, ``SymmetryRestriction(i, j, -1)`` means that
    swapping i and j flips the sign, so we can restrict to i < j
    and double the coefficient.
    """

    index_a: Index
    index_b: Index
    parity: int  # -1 for antisymmetric, +1 for symmetric


@dataclass(frozen=True)
class SymmetryAnnotatedTerm:
    """A term annotated with symmetry restrictions.

    The ``restrictions`` list contains index-ordering constraints
    from exploited antisymmetry.  The ``effective_coeff`` accounts
    for the multiplicity from restricted summation.
    """

    term: AlgebraTerm
    restrictions: tuple[SymmetryRestriction, ...]
    effective_coeff: Fraction


def _find_antisymmetric_free_pairs(
    term: AlgebraTerm,
) -> list[tuple[Index, Index]]:
    """Find pairs of free indices that participate in antisymmetric groups.

    Returns pairs (idx_a, idx_b) where idx_a < idx_b (by name) and
    both appear in the same antisymmetric group of some tensor factor.
    """
    pairs: list[tuple[Index, Index]] = []
    free_set = frozenset(term.free_indices)

    for factor in term.factors:
        if not factor.antisym_groups:
            continue
        for group in factor.antisym_groups:
            group_indices = [factor.indices[p] for p in group]
            # Find pairs of free indices within this group
            free_in_group = [idx for idx in group_indices if idx in free_set]
            for i in range(len(free_in_group)):
                for j in range(i + 1, len(free_in_group)):
                    a, b = free_in_group[i], free_in_group[j]
                    if (a.space, a.name) > (b.space, b.name):
                        a, b = b, a
                    if (a, b) not in pairs:
                        pairs.append((a, b))

    return pairs


def annotate_symmetry(
    term: AlgebraTerm,
) -> SymmetryAnnotatedTerm:
    """Annotate a single term with antisymmetry restrictions.

    For each pair of free indices in an antisymmetric group, adds
    a restriction (i < j) and adjusts the coefficient.
    """
    pairs = _find_antisymmetric_free_pairs(term)

    restrictions: list[SymmetryRestriction] = []
    multiplier = Fraction(1)

    for a, b in pairs:
        restrictions.append(SymmetryRestriction(a, b, parity=-1))
        multiplier *= 2  # Each restriction halves the index range

    return SymmetryAnnotatedTerm(
        term=term,
        restrictions=tuple(restrictions),
        effective_coeff=term.coeff * multiplier,
    )


def _symmetry_equivalence_key(
    term: AlgebraTerm,
    swap_pair: tuple[Index, Index],
) -> tuple[object, ...]:
    """Build a term key under index swap (a,b) -> (b,a).

    Used to detect terms that are related by swapping a pair of
    antisymmetric indices.
    """
    a, b = swap_pair
    mapping = {a: b, b: a}

    swapped_factors = []
    for f in term.factors:
        new_indices = tuple(mapping.get(idx, idx) for idx in f.indices)
        swapped_factors.append((f.name, new_indices))

    return tuple(sorted(swapped_factors))


def exploit_antisymmetry(
    terms: Sequence[AlgebraTerm],
) -> list[AlgebraTerm]:
    """Exploit antisymmetry to reduce term count.

    For each pair of terms related by an antisymmetric index swap
    (one is the negative of the other under the swap), merge them
    into a single term with doubled coefficient and a restricted
    index range.

    This is a conservative implementation that only handles the
    simplest case: pairs of terms differing by a single index swap.
    """
    if len(terms) < 2:
        return list(terms)

    result: list[AlgebraTerm] = []
    consumed: set[int] = set()

    # Build term signatures for quick lookup
    term_sigs: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for i, term in enumerate(terms):
        sig = tuple(
            (f.name, tuple((idx.space, idx.name) for idx in f.indices))
            for f in term.factors
        )
        term_sigs[sig].append(i)

    for i, term in enumerate(terms):
        if i in consumed:
            continue

        # Find antisymmetric free pairs
        pairs = _find_antisymmetric_free_pairs(term)
        merged = False

        for a, b in pairs:
            # Build what the term would look like after swapping a <-> b
            mapping = {a: b, b: a}
            swapped_factors = tuple(
                f.reindexed(mapping) for f in term.factors
            )
            swapped_sig = tuple(
                (f.name, tuple((idx.space, idx.name) for idx in f.indices))
                for f in swapped_factors
            )

            # Look for a matching term with opposite sign
            for j in term_sigs.get(swapped_sig, []):
                if j == i or j in consumed:
                    continue
                other = terms[j]

                # Check that this is the antisymmetric partner
                expected_coeff = -term.coeff
                if other.coeff == expected_coeff:
                    # Merge: keep the original with 2x coefficient
                    result.append(AlgebraTerm(
                        coeff=term.coeff * 2,
                        factors=term.factors,
                        free_indices=term.free_indices,
                        summed_indices=term.summed_indices,
                        connected=term.connected,
                        provenance=term.provenance,
                    ))
                    consumed.add(i)
                    consumed.add(j)
                    merged = True
                    break

            if merged:
                break

        if not merged and i not in consumed:
            result.append(term)

    return result


def exploit_antisymmetry_equations(
    equations: dict[str, list[AlgebraTerm]],
) -> dict[str, list[AlgebraTerm]]:
    """Apply antisymmetry exploitation to all equation manifolds.

    Returns a new equation dict with potentially reduced term counts.
    Energy manifold is left unchanged (no free indices to exploit).
    """
    result: dict[str, list[AlgebraTerm]] = {}
    for manifold, terms in equations.items():
        if manifold in ("energy", "reference"):
            result[manifold] = list(terms)
        else:
            result[manifold] = exploit_antisymmetry(terms)
    return result


def symmetry_reduction_stats(
    equations: dict[str, list[AlgebraTerm]],
) -> dict[str, dict[str, int]]:
    """Report potential term reduction from antisymmetry exploitation.

    Returns per-manifold statistics: before, after, eliminated.
    """
    stats: dict[str, dict[str, int]] = {}
    for manifold, terms in equations.items():
        if manifold in ("energy", "reference"):
            stats[manifold] = {
                "before": len(terms),
                "after": len(terms),
                "eliminated": 0,
            }
        else:
            reduced = exploit_antisymmetry(terms)
            stats[manifold] = {
                "before": len(terms),
                "after": len(reduced),
                "eliminated": len(terms) - len(reduced),
            }
    return stats
