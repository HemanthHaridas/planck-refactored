"""Canonicalization: dummy relabeling, tensor normalization, term merging."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from typing import Sequence

from .indices import (
    Index,
    OCC_POOL,
    VIR_POOL,
    GEN_POOL,
    extend_pool,
    relabel_dummies,
)
from .tensors import Tensor, reindex_tensors
from .project import AlgebraTerm

try:
    from . import _wickaccel
except ImportError:  # pragma: no cover - exercised without the extension
    _wickaccel = None


def _space_rank(space: str) -> int:
    if space == "occ":
        return 0
    if space == "vir":
        return 1
    return 2


def _space_code(space: str) -> int:
    return _space_rank(space)


def _pool_for_space(space: str) -> list[str]:
    if space == "occ":
        return OCC_POOL
    if space == "vir":
        return VIR_POOL
    return GEN_POOL


_POOL_SLOT_CACHE: dict[str, tuple[int, dict[str, int]]] = {
    "occ": (0, {}),
    "vir": (0, {}),
    "gen": (0, {}),
}


def _pool_slot_lookup(space: str) -> dict[str, int]:
    pool = _pool_for_space(space)
    cached_len, cached_lookup = _POOL_SLOT_CACHE[space]
    if cached_len == len(pool):
        return cached_lookup
    lookup = {name: pos for pos, name in enumerate(pool)}
    _POOL_SLOT_CACHE[space] = (len(pool), lookup)
    return lookup


@lru_cache(maxsize=None)
def _reserved_slot_sets(
    free_indices: tuple[Index, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    reserved: dict[str, set[int]] = {"occ": set(), "vir": set(), "gen": set()}
    for idx in free_indices:
        lookup = _pool_slot_lookup(idx.space)
        slot = lookup.get(idx.name)
        if slot is not None:
            reserved[idx.space].add(slot)
    return (
        tuple(sorted(reserved["occ"])),
        tuple(sorted(reserved["vir"])),
        tuple(sorted(reserved["gen"])),
    )


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

    if _wickaccel is not None:
        slot_order = {
            slot: pos for pos, slot in enumerate(sorted(
                {(idx.space, idx.name) for idx in tensor.indices},
                key=lambda item: (_space_rank(item[0]), item[1]),
            ))
        }
        codes = tuple(slot_order[(idx.space, idx.name)] for idx in tensor.indices)
        is_zero, sign, order = _wickaccel.canonicalize_tensor_layout(
            codes,
            tensor.antisym_groups,
        )
        if is_zero:
            return tensor, 0
        if tuple(order) == tuple(range(len(tensor.indices))):
            return tensor, sign
        return tensor.with_indices([tensor.indices[pos] for pos in order]), sign

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
    return t.sort_key


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

    if _wickaccel is not None:
        free_mask = tuple(idx in free_set for idx in all_idx)
        space_codes = tuple(_space_code(idx.space) for idx in all_idx)
        occ_reserved, vir_reserved, gen_reserved = _reserved_slot_sets(
            term.free_indices,
        )

        ordinals = _wickaccel.assign_dummy_ordinals(
            space_codes,
            free_mask,
            occ_reserved,
            vir_reserved,
            gen_reserved,
        )

        full_mapping: dict[Index, Index] = {}
        for idx, ordinal, is_free in zip(all_idx, ordinals, free_mask):
            if is_free:
                continue
            pool = _pool_for_space(idx.space)
            extend_pool(pool, ordinal + 1)
            full_mapping[idx] = Index(pool[ordinal], idx.space, is_dummy=True)

        new_factors = reindex_tensors(term.factors, full_mapping)
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

    new_factors = reindex_tensors(term.factors, full_mapping)
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
    """Apply the full canonicalization pipeline to a single term.

    Pipeline (factored):
      1. Canonicalize each tensor's indices via antisymmetry groups
      2. Sort tensor factors lexicographically
      3. Relabel dummy indices once by first-appearance convention

    Deferring relabeling until after sorting avoids a redundant
    relabeling pass — the single final relabel produces the same
    canonical form because both factor order and dummy names are
    normalized together.
    """
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

    # Sort factors before relabeling so that the factor order is
    # determined by tensor name + index *space* pattern only (not
    # arbitrary dummy names).  Then a single relabel pass normalizes
    # dummy names by first appearance in the sorted sequence.
    canon_factors.sort(key=_tensor_sort_key)

    term = AlgebraTerm(
        coeff=term.coeff * sign,
        factors=tuple(canon_factors),
        free_indices=term.free_indices,
        summed_indices=term.summed_indices,
        connected=term.connected,
        provenance=term.provenance,
    )

    term = relabel_term_dummies(term)
    return term


def _term_signature(term: AlgebraTerm) -> tuple[object, ...]:
    return (term.factors, term.free_indices)


def merge_like_terms(terms: Sequence[AlgebraTerm]) -> list[AlgebraTerm]:
    """Combine terms with identical tensor structure by adding coefficients."""
    buckets: dict[tuple[object, ...], AlgebraTerm] = {}
    order: list[tuple[object, ...]] = []

    for t in terms:
        merge_term_into_buckets(t, buckets, order)

    return [
        buckets[sig] for sig in order if buckets[sig].coeff != 0
    ]


def _exact_term_signature(term: AlgebraTerm) -> tuple[object, ...]:
    return (
        term.factors,
        term.free_indices,
        term.summed_indices,
        term.connected,
    )


def merge_exact_term_into_buckets(
    term: AlgebraTerm,
    buckets: dict[tuple[object, ...], AlgebraTerm],
    order: list[tuple[object, ...]],
) -> None:
    """Accumulate exact raw duplicates before canonicalization."""
    sig = _exact_term_signature(term)
    if sig in buckets:
        old = buckets[sig]
        buckets[sig] = AlgebraTerm(
            coeff=old.coeff + term.coeff,
            factors=old.factors,
            free_indices=old.free_indices,
            summed_indices=old.summed_indices,
            connected=old.connected and term.connected,
            provenance=old.provenance,
        )
        return

    buckets[sig] = term
    order.append(sig)


def merge_term_into_buckets(
    term: AlgebraTerm,
    buckets: dict[tuple[object, ...], AlgebraTerm],
    order: list[tuple[object, ...]],
) -> None:
    """Accumulate one canonical term into a merge bucket map."""
    sig = _term_signature(term)
    if sig in buckets:
        old = buckets[sig]
        buckets[sig] = AlgebraTerm(
            coeff=old.coeff + term.coeff,
            factors=old.factors,
            free_indices=old.free_indices,
            summed_indices=old.summed_indices,
            connected=old.connected and term.connected,
            provenance=old.provenance,
        )
        return

    buckets[sig] = term
    order.append(sig)


def term_is_zero_before_canonicalization(term: AlgebraTerm) -> bool:
    """Cheap zero test based on repeated antisymmetric tensor slots."""
    for factor in term.factors:
        if not factor.antisym_groups:
            continue
        for group in factor.antisym_groups:
            slots = [factor.indices[pos] for pos in group]
            if len(slots) != len(set(slots)):
                return True
    return False


# ── Orbital energy denominator collection ───────────────────────────


def _is_diagonal_fock(tensor: Tensor) -> bool:
    """True if *tensor* is f(x,x) with identical index name and space."""
    return (
        tensor.name == "f"
        and len(tensor.indices) == 2
        and tensor.indices[0].name == tensor.indices[1].name
        and tensor.indices[0].space == tensor.indices[1].space
    )


def _non_fock_signature(
    term: AlgebraTerm,
) -> tuple[tuple[object, ...], tuple[int, ...]] | None:
    """Return a hashable signature for the non-diagonal-Fock part of a term.

    If the term contains exactly one diagonal Fock factor among its
    factors, returns (signature_of_remaining_factors, indices_of_fock).
    Otherwise returns None.
    """
    fock_positions: list[int] = []
    for i, fac in enumerate(term.factors):
        if _is_diagonal_fock(fac):
            fock_positions.append(i)

    if len(fock_positions) != 1:
        return None

    # The diagonal Fock index must be a free index (not summed)
    fock_idx = term.factors[fock_positions[0]].indices[0]
    if fock_idx not in term.free_indices:
        return None

    remaining = tuple(
        f for i, f in enumerate(term.factors) if i not in fock_positions
    )
    sig = tuple(
        (f.name, tuple((idx.space, idx.name) for idx in f.indices))
        for f in remaining
    )
    free_key = tuple((idx.space, idx.name) for idx in term.free_indices)
    return (sig, free_key), tuple(fock_positions)


def collect_fock_diagonals(
    terms: Sequence[AlgebraTerm],
) -> list[AlgebraTerm]:
    """Collect diagonal Fock terms into orbital energy denominators.

    Scans *terms* for groups that differ only in which diagonal Fock
    element ``f(x,x)`` they carry (with the same amplitude factors).
    Groups are replaced by a single term with a ``D`` (denominator)
    tensor carrying the collected orbital energy indices.

    The denominator tensor ``D(a, b, i, j)`` represents the quantity
    ``(ε_a + ε_b - ε_i - ε_j)`` where virtual indices contribute
    ``+ε`` and occupied indices contribute ``-ε``.

    Terms that do not contain a diagonal Fock factor are passed through
    unchanged.
    """
    # Bucket terms by their non-Fock signature
    groups: dict[tuple[object, ...], list[tuple[AlgebraTerm, int]]] = (
        defaultdict(list)
    )
    passthrough: list[AlgebraTerm] = []
    passthrough_positions: list[int] = []

    for pos, term in enumerate(terms):
        result = _non_fock_signature(term)
        if result is None:
            passthrough.append(term)
            passthrough_positions.append(pos)
        else:
            sig, fock_positions = result
            groups[sig].append((term, fock_positions[0]))

    # Only collect groups with ≥2 terms (otherwise no benefit)
    collected: list[tuple[int, AlgebraTerm]] = []

    for sig, group in groups.items():
        if len(group) < 2:
            # Single diagonal-Fock term: keep as-is
            for term, _ in group:
                passthrough.append(term)
            continue

        # Collect: build a denominator tensor from all diagonal Fock
        # elements, then emit one term with D * amplitude
        first_term, first_fock_pos = group[0]
        remaining_factors = tuple(
            f for i, f in enumerate(first_term.factors)
            if i != first_fock_pos
        )

        # Gather denominator contributions: (index, sign)
        # Convention: vir → +1, occ → -1
        denom_parts: list[tuple[Index, Fraction]] = []
        base_coeff: Fraction | None = None

        for term, fock_pos in group:
            fock_tensor = term.factors[fock_pos]
            fock_idx = fock_tensor.indices[0]

            # Determine the sign contribution for this orbital energy
            if fock_idx.space == "vir":
                energy_sign = Fraction(1)
            else:  # occ
                energy_sign = Fraction(-1)

            # The coefficient of this term relative to the amplitude
            # should be ±|base_coeff|. Verify consistency.
            if base_coeff is None:
                base_coeff = abs(term.coeff)
                actual_sign = term.coeff / base_coeff
                if actual_sign != energy_sign:
                    # Signs don't follow the convention; abort this group
                    for t, _ in group:
                        passthrough.append(t)
                    break
            else:
                actual_sign = term.coeff / base_coeff
                if actual_sign != energy_sign:
                    # Inconsistent sign pattern; abort this group
                    for t, _ in group:
                        passthrough.append(t)
                    break

            denom_parts.append((fock_idx, energy_sign))
        else:
            # All terms validated; build the denominator tensor
            # Sort: occupied indices first, then virtual
            denom_parts.sort(
                key=lambda x: (0 if x[0].space == "occ" else 1, x[0].name)
            )
            denom_indices = tuple(idx for idx, _ in denom_parts)

            denom_tensor = Tensor("D", denom_indices)
            new_factors = (denom_tensor,) + remaining_factors

            collected_term = AlgebraTerm(
                coeff=base_coeff,
                factors=new_factors,
                free_indices=first_term.free_indices,
                summed_indices=first_term.summed_indices,
                connected=first_term.connected,
                provenance=first_term.provenance,
            )
            collected.append((0, collected_term))

    # Reconstruct output: passthrough terms in original order + collected
    result = list(passthrough) + [t for _, t in collected]
    return result
