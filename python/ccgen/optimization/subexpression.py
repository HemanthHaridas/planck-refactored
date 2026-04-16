"""Common subexpression elimination (CSE) for CC equations.

Detects groups of terms that share identical contracted sub-products
and factors them into named temporaries, reducing redundant computation
and improving code readability.
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
class CSESpec:
    """Specification of a common subexpression to be factored out."""

    name: str
    indices: tuple[Index, ...]
    definition: AlgebraTerm
    usage_count: int
    index_space_sig: str

    @property
    def rank(self) -> int:
        return len(self.indices)


def _contraction_key(
    factors: tuple[Tensor, ...],
    summed: tuple[Index, ...],
) -> tuple[object, ...]:
    """Build a hashable key for the full contraction pattern of a term.

    Two terms that have the same contraction key perform the same
    tensor contraction on the RHS (same tensor names, same index
    connectivity, same summation pattern) — they may differ only in
    their numerical coefficient.
    """
    factor_key = tuple(
        (f.name, tuple((idx.space, idx.name) for idx in f.indices))
        for f in factors
    )
    summed_key = tuple(
        (idx.space, idx.name) for idx in summed
    )
    return (factor_key, summed_key)


def detect_common_subexpressions(
    terms: Sequence[AlgebraTerm],
    threshold: int = 2,
) -> list[tuple[list[int], tuple[object, ...]]]:
    """Find groups of terms with identical RHS contractions.

    Returns a list of (term_indices, contraction_key) where each group
    has at least *threshold* terms sharing the same contraction.
    """
    buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)

    for i, term in enumerate(terms):
        key = _contraction_key(term.factors, term.summed_indices)
        buckets[key].append(i)

    return [
        (indices, key)
        for key, indices in buckets.items()
        if len(indices) >= threshold
    ]


def factor_common_subexpressions(
    terms: Sequence[AlgebraTerm],
    threshold: int = 2,
) -> tuple[list[AlgebraTerm], list[CSESpec]]:
    """Factor out common subexpressions from a list of terms.

    For each group of terms sharing an identical RHS contraction,
    introduces a temporary variable and rewrites the terms to
    accumulate into it.

    Returns (rewritten_terms, cse_specs) where:
    - rewritten_terms: terms with CSE groups replaced by single terms
      referencing temporaries
    - cse_specs: definitions of the extracted temporaries

    Terms not participating in any CSE group are passed through unchanged.
    """
    groups = detect_common_subexpressions(terms, threshold=threshold)

    if not groups:
        return list(terms), []

    # Collect which term indices are consumed by CSE groups
    consumed: set[int] = set()
    cse_specs: list[CSESpec] = []
    replacement_terms: list[AlgebraTerm] = []

    for group_idx, (term_indices, key) in enumerate(groups):
        consumed.update(term_indices)

        # Sum up the coefficients
        total_coeff = Fraction(0)
        for ti in term_indices:
            total_coeff += terms[ti].coeff

        if total_coeff == 0:
            continue

        # Use the first term as the representative
        rep = terms[term_indices[0]]

        # Build the space signature
        ext_sorted = sorted(
            rep.free_indices,
            key=lambda x: (0 if x.space == "occ" else 1, x.name),
        )
        space_sig = "".join(
            "o" if idx.space == "occ" else "v" if idx.space == "vir" else "g"
            for idx in ext_sorted
        )

        cse_name = f"cse_{space_sig}_{group_idx + 1}"

        # The CSE definition is the contraction with coeff=1
        definition = AlgebraTerm(
            coeff=Fraction(1),
            factors=rep.factors,
            free_indices=rep.free_indices,
            summed_indices=rep.summed_indices,
            connected=rep.connected,
            provenance=rep.provenance,
        )

        cse_specs.append(CSESpec(
            name=cse_name,
            indices=rep.free_indices,
            definition=definition,
            usage_count=len(term_indices),
            index_space_sig=space_sig,
        ))

        # Replacement term: total_coeff * cse_tensor(free_indices)
        cse_tensor = Tensor(cse_name, rep.free_indices)
        replacement_terms.append(AlgebraTerm(
            coeff=total_coeff,
            factors=(cse_tensor,),
            free_indices=rep.free_indices,
            summed_indices=(),
            connected=rep.connected,
            provenance=rep.provenance,
        ))

    # Build output: unconsumed terms + replacement terms
    result = [t for i, t in enumerate(terms) if i not in consumed]
    result.extend(replacement_terms)

    return result, cse_specs


def apply_cse(
    equations: dict[str, list[AlgebraTerm]],
    threshold: int = 2,
) -> tuple[dict[str, list[AlgebraTerm]], dict[str, list[CSESpec]]]:
    """Apply CSE across all equation manifolds.

    Returns (rewritten_equations, cse_specs_by_manifold).
    """
    new_equations: dict[str, list[AlgebraTerm]] = {}
    all_specs: dict[str, list[CSESpec]] = {}

    for manifold, terms in equations.items():
        rewritten, specs = factor_common_subexpressions(
            terms, threshold=threshold,
        )
        new_equations[manifold] = rewritten
        if specs:
            all_specs[manifold] = specs

    return new_equations, all_specs
