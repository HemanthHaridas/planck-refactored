"""Intermediate tensor detection and equation rewriting.

Scans coupled-cluster residual equations for sub-contractions that
appear in multiple terms and extracts them as reusable intermediate
tensors (e.g. W_oovv, F_ov).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Sequence

from ..indices import Index, make_gen, make_occ, make_vir
from ..project import AlgebraTerm
from ..tensors import Tensor
from ..canonicalize import relabel_term_dummies

_SIZE_EST = {"occ": 30, "vir": 100, "gen": 50}


@dataclass(frozen=True)
class IntermediateSpec:
    """Specification for a detected intermediate tensor.

    Phase 3 additions: ``memory_layout``, ``blocking_hint``, and
    ``allocation_strategy`` carry hints for backend code generation.
    """

    name: str
    indices: tuple[Index, ...]
    definition_terms: tuple[AlgebraTerm, ...]
    usage_count: int
    index_space_sig: str
    usage_targets: tuple[str, ...] = ()
    memory_layout: str = "row_major"  # "row_major" | "col_major" | "blocked"
    blocking_hint: dict[str, int] | None = None  # {index_name: tile_size}
    allocation_strategy: str = "auto"  # "stack" | "malloc" | "external" | "auto"

    def __hash__(self) -> int:
        return hash((self.name, self.indices, self.index_space_sig))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntermediateSpec):
            return NotImplemented
        return (
            self.name == other.name
            and self.indices == other.indices
            and self.index_space_sig == other.index_space_sig
        )

    @property
    def rank(self) -> int:
        return len(self.indices)

    @property
    def estimated_elements(self) -> int:
        """Estimate number of elements (for allocation decisions).

        Uses symbolic sizes: occ ~ 30, vir ~ 100.
        """
        total = 1
        for idx in self.indices:
            total *= _SIZE_EST.get(idx.space, 50)
        return total

    @property
    def estimated_bytes(self) -> int:
        """Estimated memory in bytes (double precision)."""
        return self.estimated_elements * 8

    @property
    def definition_complexity(self) -> int:
        """Cheap proxy for downstream contraction savings."""
        return max(1, sum(len(term.factors) for term in self.definition_terms))

    @property
    def estimated_build_flops(self) -> int:
        """Cheap symbolic estimate of the cost to build the intermediate once."""
        total = 0
        for term in self.definition_terms:
            seen: set[Index] = set()
            term_cost = 1
            for factor in term.factors:
                for idx in factor.indices:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    term_cost *= _SIZE_EST.get(idx.space, 50)
            total += term_cost
        return max(1, total)

    @property
    def estimated_saved_flops(self) -> int:
        """Estimate total recomputation avoided by materializing the tensor."""
        return max(0, self.usage_count - 1) * self.estimated_build_flops

    @property
    def estimated_reuse_value(self) -> int:
        """Backwards-compatible value proxy for budgeted selection."""
        return self.estimated_saved_flops

    @property
    def selection_density(self) -> float:
        """Estimated flop savings per byte of storage."""
        return self.estimated_saved_flops / max(1, self.estimated_bytes)

    def with_layout_hints(
        self,
        memory_layout: str | None = None,
        blocking_hint: dict[str, int] | None = None,
        allocation_strategy: str | None = None,
    ) -> IntermediateSpec:
        """Return a copy with updated layout hints."""
        return IntermediateSpec(
            name=self.name,
            indices=self.indices,
            definition_terms=self.definition_terms,
            usage_count=self.usage_count,
            index_space_sig=self.index_space_sig,
            usage_targets=self.usage_targets,
            memory_layout=memory_layout or self.memory_layout,
            blocking_hint=blocking_hint or self.blocking_hint,
            allocation_strategy=allocation_strategy or self.allocation_strategy,
        )


def _subcontraction_signature(
    factors: tuple[Tensor, ...],
    summed: frozenset[Index],
) -> tuple[tuple[object, ...], str]:
    """Build a hashable signature for a sub-contraction.

    The signature is independent of the source term's concrete index names.
    It captures the canonicalized contraction topology of the sub-expression
    plus the external index-space pattern used for naming.
    """
    internal = _internal_indices(factors, summed)
    normalized = _normalize_subcontraction(factors, internal)
    space_sig = "".join(
        "o" if idx.space == "occ" else "v" if idx.space == "vir" else "g"
        for idx in normalized.free_indices
    )
    signature = (
        tuple(
            (f.name, tuple((idx.space, idx.name) for idx in f.indices))
            for f in normalized.factors
        ),
        tuple((idx.space, idx.name) for idx in normalized.free_indices),
        tuple((idx.space, idx.name) for idx in normalized.summed_indices),
    )
    return signature, space_sig


@dataclass(frozen=True)
class _SubContractionKey:
    """Hashable key for a sub-contraction pattern."""

    signature: tuple[object, ...]
    space_sig: str


def _extract_subcontractions(
    term: AlgebraTerm,
    min_factors: int = 2,
    max_factors: int | None = None,
) -> list[tuple[_SubContractionKey, tuple[int, ...]]]:
    """Extract all sub-contraction candidates from a single term.

    Returns (key, factor_indices) pairs.  Only sub-contractions that
    involve at least one internal summation are included.
    """
    n = len(term.factors)
    if max_factors is None:
        max_factors = min(n - 1, 4)  # don't extract the whole term

    summed_set = frozenset(term.summed_indices)
    results: list[tuple[_SubContractionKey, tuple[int, ...]]] = []

    for size in range(min_factors, max_factors + 1):
        for combo in combinations(range(n), size):
            sub_factors = tuple(term.factors[i] for i in combo)

            # Collect indices appearing in these factors
            sub_indices: Counter[Index] = Counter()
            for f in sub_factors:
                for idx in f.indices:
                    sub_indices[idx] += 1

            # Must have at least one internal contraction
            internal = [
                idx for idx in sub_indices
                if idx in summed_set and sub_indices[idx] >= 2
            ]
            if not internal:
                continue

            signature, space_sig = _subcontraction_signature(
                sub_factors, summed_set,
            )
            key = _SubContractionKey(
                signature=signature,
                space_sig=space_sig,
            )
            results.append((key, combo))

    return results


def _index_space_name(sig: str) -> str:
    """Convert a space signature like 'oovv' to a name like 'W_oovv'."""
    if not sig:
        return "W_scalar"
    return f"W_{sig}"


def detect_intermediates(
    equations: dict[str, list[AlgebraTerm]],
    threshold: int = 2,
    memory_budget_bytes: int | None = None,
    peak_memory_budget_bytes: int | None = None,
) -> list[IntermediateSpec]:
    """Detect reusable sub-contractions across all equation manifolds.

    Scans every term in *equations* for factor subsets that share
    internal contractions.  Sub-contractions appearing in at least
    *threshold* terms are returned as ``IntermediateSpec`` objects.

    Parameters
    ----------
    equations : dict mapping manifold name to list of AlgebraTerm
    threshold : minimum number of terms a sub-contraction must appear
                in to be reported (default 2)

    memory_budget_bytes : optional int
        If provided, greedily selects only the subset of detected
        intermediates that fits within the cumulative memory budget.
    peak_memory_budget_bytes : optional int
        If provided, selects a subset whose live intermediate set stays
        within the memory cap for every target manifold independently.

    Returns
    -------
    list of IntermediateSpec, sorted by usage count (descending)
    """
    # Phase 1: count how often each sub-contraction key appears
    key_count: Counter[_SubContractionKey] = Counter()
    key_examples: dict[_SubContractionKey, AlgebraTerm] = {}
    key_targets: dict[_SubContractionKey, set[str]] = {}

    for manifold, terms in equations.items():
        for term_idx, term in enumerate(terms):
            subs = _extract_subcontractions(term)
            seen_keys: set[_SubContractionKey] = set()
            for key, factor_indices in subs:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                key_count[key] += 1
                key_targets.setdefault(key, set()).add(manifold)
                if key not in key_examples:
                    sub_factors = tuple(term.factors[i] for i in factor_indices)
                    sub_indices: Counter[Index] = Counter()
                    for f in sub_factors:
                        for idx in f.indices:
                            sub_indices[idx] += 1
                    internal = frozenset(
                        idx for idx in sub_indices
                        if idx in frozenset(term.summed_indices) and sub_indices[idx] >= 2
                    )
                    key_examples[key] = _normalize_subcontraction(
                        sub_factors,
                        internal,
                    )

    # Phase 2: build IntermediateSpec for keys above threshold
    results: list[IntermediateSpec] = []
    name_counter: Counter[str] = Counter()

    for key, count in key_count.most_common():
        if count < threshold:
            break

        base_name = _index_space_name(key.space_sig)
        name_counter[base_name] += 1
        if name_counter[base_name] > 1:
            name = f"{base_name}_{name_counter[base_name]}"
        else:
            name = base_name

        definition = key_examples[key]

        results.append(IntermediateSpec(
            name=name,
            indices=definition.free_indices,
            definition_terms=(definition,),
            usage_count=count,
            index_space_sig=key.space_sig,
            usage_targets=tuple(sorted(key_targets.get(key, ()))),
        ))

    if peak_memory_budget_bytes is not None:
        return select_intermediates_for_peak_budget(results, peak_memory_budget_bytes)

    if memory_budget_bytes is not None:
        return select_intermediates_for_budget(results, memory_budget_bytes)

    return results


def total_estimated_bytes(
    intermediates: Sequence[IntermediateSpec],
) -> int:
    """Return the cumulative estimated footprint of materialized intermediates."""
    return sum(spec.estimated_bytes for spec in intermediates)


def peak_estimated_bytes_by_target(
    intermediates: Sequence[IntermediateSpec],
) -> dict[str, int]:
    """Return the cumulative intermediate footprint per usage target."""
    totals: dict[str, int] = {}
    for spec in intermediates:
        for target in spec.usage_targets:
            totals[target] = totals.get(target, 0) + spec.estimated_bytes
    return totals


def select_intermediates_for_budget(
    intermediates: Sequence[IntermediateSpec],
    memory_budget_bytes: int,
) -> list[IntermediateSpec]:
    """Select a subset of intermediates under a cumulative memory budget.

    The current selector is intentionally simple: it greedily prefers
    intermediates with the strongest reuse proxy, then breaks ties in favor
    of smaller tensors so tight budgets keep more reusable building blocks.
    """
    if memory_budget_bytes <= 0:
        return []

    ordered = sorted(
        intermediates,
        key=lambda spec: (
            -spec.selection_density,
            -spec.estimated_saved_flops,
            -spec.usage_count,
            spec.estimated_bytes,
            spec.rank,
            spec.name,
        ),
    )

    selected: list[IntermediateSpec] = []
    remaining = memory_budget_bytes
    for spec in ordered:
        if spec.estimated_bytes > remaining:
            continue
        selected.append(spec)
        remaining -= spec.estimated_bytes

    return sorted(
        selected,
        key=lambda spec: (-spec.usage_count, spec.estimated_bytes, spec.name),
    )


def select_intermediates_for_peak_budget(
    intermediates: Sequence[IntermediateSpec],
    peak_memory_budget_bytes: int,
) -> list[IntermediateSpec]:
    """Select intermediates while enforcing a per-target live-memory budget."""
    if peak_memory_budget_bytes <= 0:
        return []

    ordered = sorted(
        intermediates,
        key=lambda spec: (
            -spec.selection_density,
            -spec.estimated_saved_flops,
            -spec.usage_count,
            spec.estimated_bytes,
            len(spec.usage_targets),
            spec.rank,
            spec.name,
        ),
    )

    selected: list[IntermediateSpec] = []
    live_bytes_by_target: dict[str, int] = {}

    for spec in ordered:
        if any(
            live_bytes_by_target.get(target, 0) + spec.estimated_bytes
            > peak_memory_budget_bytes
            for target in spec.usage_targets
        ):
            continue
        selected.append(spec)
        for target in spec.usage_targets:
            live_bytes_by_target[target] = (
                live_bytes_by_target.get(target, 0) + spec.estimated_bytes
            )

    return sorted(
        selected,
        key=lambda spec: (-spec.usage_count, spec.estimated_bytes, spec.name),
    )


def _ordered_external(
    factors: tuple[Tensor, ...],
    internal: frozenset[Index],
) -> list[Index]:
    """Collect external indices in first-appearance order."""
    seen: set[Index] = set()
    result: list[Index] = []
    for f in factors:
        for idx in f.indices:
            if idx not in internal and idx not in seen:
                seen.add(idx)
                result.append(idx)
    return result


def _internal_indices(
    factors: tuple[Tensor, ...],
    summed: frozenset[Index],
) -> frozenset[Index]:
    """Return indices summed internally within the sub-contraction."""
    sub_indices: Counter[Index] = Counter()
    for f in factors:
        for idx in f.indices:
            sub_indices[idx] += 1
    return frozenset(
        idx for idx in sub_indices
        if idx in summed and sub_indices[idx] >= 2
    )


def _canonical_external_order(indices: Sequence[Index]) -> list[Index]:
    """Return external indices in Planck-friendly canonical order.

    The first-appearance order is still useful as a stable tiebreaker, but we
    need occupied indices before virtual indices so an intermediate named
    ``W_oovv`` is actually emitted and indexed as ``(i,j,a,b)`` rather than,
    for example, ``(a,i,b,j)``.
    """
    space_order = {"occ": 0, "vir": 1, "gen": 2}
    first_position = {idx: pos for pos, idx in enumerate(indices)}
    return sorted(
        indices,
        key=lambda idx: (space_order.get(idx.space, 99), first_position[idx]),
    )


def _next_canonical_free_index(space: str, offset: int) -> Index:
    if space == "occ":
        return make_occ(f"i{offset}" if offset else "i", dummy=False)
    if space == "vir":
        return make_vir(f"a{offset}" if offset else "a", dummy=False)
    return make_gen(f"p{offset}" if offset else "p", dummy=False)


def _normalize_subcontraction(
    factors: tuple[Tensor, ...],
    internal: frozenset[Index],
) -> AlgebraTerm:
    """Canonicalize a sub-contraction independent of source index names."""
    ordered_external = _canonical_external_order(_ordered_external(factors, internal))
    external_map: dict[Index, Index] = {}
    counts: dict[str, int] = {"occ": 0, "vir": 0, "gen": 0}
    normalized_free: list[Index] = []

    for idx in ordered_external:
        canonical = _next_canonical_free_index(idx.space, counts[idx.space])
        counts[idx.space] += 1
        external_map[idx] = canonical
        normalized_free.append(canonical)

    dummy_map = {idx: idx.as_dummy() for idx in internal}
    mapping = {**dummy_map, **external_map}
    normalized = AlgebraTerm(
        coeff=Fraction(1),
        factors=tuple(f.reindexed(mapping) for f in factors),
        free_indices=tuple(normalized_free),
        summed_indices=tuple(mapping[idx] for idx in internal),
        connected=True,
    )
    sorted_factors = tuple(
        sorted(
            normalized.factors,
            key=lambda tensor: (
                tensor.name,
                tuple((idx.space, idx.name) for idx in tensor.indices),
            ),
        )
    )
    return relabel_term_dummies(
        AlgebraTerm(
            coeff=normalized.coeff,
            factors=sorted_factors,
            free_indices=normalized.free_indices,
            summed_indices=normalized.summed_indices,
            connected=normalized.connected,
        )
    )


def build_intermediate_equations(
    intermediates: list[IntermediateSpec],
) -> dict[str, list[AlgebraTerm]]:
    """Return a dict mapping intermediate name → definition terms.

    Each intermediate's definition is a list of AlgebraTerm objects
    suitable for emission by the einsum or C++ backends.  The LHS
    of each equation is the intermediate tensor; the RHS factors
    are the original tensors being contracted.
    """
    result: dict[str, list[AlgebraTerm]] = {}
    for spec in intermediates:
        result[spec.name] = list(spec.definition_terms)
    return result


def rewrite_equations(
    equations: dict[str, list[AlgebraTerm]],
    intermediates: list[IntermediateSpec],
) -> dict[str, list[AlgebraTerm]]:
    """Rewrite equations substituting detected intermediates.

    For each intermediate, finds terms containing the matching
    sub-contraction and replaces the factor subset with a reference
    to the intermediate tensor.

    Returns a new equation dict (original is not modified).
    """
    if not intermediates:
        return equations

    result: dict[str, list[AlgebraTerm]] = {}
    for manifold, terms in equations.items():
        new_terms: list[AlgebraTerm] = []
        for term in terms:
            new_terms.append(_try_substitute(term, intermediates))
        result[manifold] = new_terms
    return result


def _try_substitute(
    term: AlgebraTerm,
    intermediates: list[IntermediateSpec],
) -> AlgebraTerm:
    """Try to substitute intermediates into a single term."""
    for spec in intermediates:
        if len(spec.definition_terms) != 1:
            continue
        defn = spec.definition_terms[0]
        match = _find_subfactors(
            term,
            defn.factors,
            defn.free_indices,
            defn.summed_indices,
        )
        if match is None:
            continue

        factor_indices, index_mapping = match
        remaining = [
            f for i, f in enumerate(term.factors)
            if i not in factor_indices
        ]

        # Build the intermediate tensor reference with mapped indices
        mapped_indices = tuple(
            index_mapping.get(idx, idx) for idx in spec.indices
        )
        intermediate_tensor = Tensor(spec.name, mapped_indices)
        new_factors = tuple([intermediate_tensor] + remaining)

        # Remove internal summed indices
        internal_summed = frozenset(
            index_mapping.get(idx, idx)
            for idx in defn.summed_indices
        )
        new_summed = tuple(
            idx for idx in term.summed_indices
            if idx not in internal_summed
        )

        return AlgebraTerm(
            coeff=term.coeff,
            factors=new_factors,
            free_indices=term.free_indices,
            summed_indices=new_summed,
            connected=term.connected,
            provenance=term.provenance,
        )

    return term


def _find_subfactors(
    term: AlgebraTerm,
    pattern_factors: tuple[Tensor, ...],
    pattern_free_indices: tuple[Index, ...] = (),
    pattern_summed_indices: tuple[Index, ...] = (),
) -> tuple[frozenset[int], dict[Index, Index]] | None:
    """Find factor subset in term matching the pattern by name.

    Returns (set of matched factor indices, index mapping from
    pattern to term) or None if no match.
    """
    pattern_names = [f.name for f in pattern_factors]
    term_by_name: dict[str, list[int]] = {}
    for i, f in enumerate(term.factors):
        term_by_name.setdefault(f.name, []).append(i)

    # Check all required tensor names are present
    for name in pattern_names:
        if name not in term_by_name or not term_by_name[name]:
            return None

    ordered_patterns = sorted(
        enumerate(pattern_factors),
        key=lambda item: (item[1].name, item[1].rank),
    )

    def _search(
        pos: int,
        used: frozenset[int],
        mapping: dict[Index, Index],
    ) -> tuple[frozenset[int], dict[Index, Index]] | None:
        if pos == len(ordered_patterns):
            matched_factors = tuple(term.factors[i] for i in sorted(used))
            actual_internal = _internal_indices(
                matched_factors,
                frozenset(term.summed_indices),
            )
            for idx in pattern_summed_indices:
                if mapping.get(idx) not in actual_internal:
                    return None
            for idx in pattern_free_indices:
                mapped = mapping.get(idx)
                if mapped is None or mapped in actual_internal:
                    return None
            return used, mapping

        _, pf = ordered_patterns[pos]
        for ti in term_by_name.get(pf.name, []):
            if ti in used:
                continue
            tf = term.factors[ti]
            if tf.rank != pf.rank:
                continue

            next_mapping = dict(mapping)
            inverse_mapping = {value: key for key, value in next_mapping.items()}
            conflict = False
            for pi, ti_idx in zip(pf.indices, tf.indices):
                if pi.space != ti_idx.space:
                    conflict = True
                    break
                mapped = next_mapping.get(pi)
                if mapped is not None and mapped != ti_idx:
                    conflict = True
                    break
                inverse = inverse_mapping.get(ti_idx)
                if inverse is not None and inverse != pi:
                    conflict = True
                    break
                next_mapping[pi] = ti_idx
                inverse_mapping[ti_idx] = pi
            if conflict:
                continue

            found = _search(pos + 1, used | frozenset({ti}), next_mapping)
            if found is not None:
                return found
        return None

    return _search(0, frozenset(), {})


# ── Phase 3: Memory layout and blocking annotation ────────────────


# Stack threshold: intermediates smaller than this (in bytes) use
# stack allocation; larger ones use malloc.
_STACK_THRESHOLD_BYTES = 256 * 1024  # 256 KB

# Default tile sizes
_DEFAULT_TILE_OCC = 16
_DEFAULT_TILE_VIR = 16


def annotate_layout_hints(
    intermediates: list[IntermediateSpec],
    tile_occ: int = _DEFAULT_TILE_OCC,
    tile_vir: int = _DEFAULT_TILE_VIR,
    stack_threshold: int = _STACK_THRESHOLD_BYTES,
) -> list[IntermediateSpec]:
    """Annotate intermediates with memory layout and blocking hints.

    For each intermediate:
    - Determines allocation strategy (stack vs malloc) based on size
    - Computes per-index tile sizes for cache-efficient blocking
    - Sets memory layout to row_major (default) or blocked for
      large intermediates

    Parameters
    ----------
    tile_occ, tile_vir : int
        Tile sizes for occupied and virtual index dimensions.
    stack_threshold : int
        Maximum size in bytes for stack allocation.
    """
    result: list[IntermediateSpec] = []

    for spec in intermediates:
        # Allocation strategy
        est_bytes = spec.estimated_bytes
        if est_bytes <= stack_threshold:
            alloc = "stack"
        else:
            alloc = "malloc"

        # Blocking hints per index
        blocking: dict[str, int] = {}
        for idx in spec.indices:
            if idx.space == "occ":
                blocking[idx.name] = tile_occ
            elif idx.space == "vir":
                blocking[idx.name] = tile_vir

        # Memory layout: use blocked for rank-4+ large tensors
        if spec.rank >= 4 and est_bytes > stack_threshold:
            layout = "blocked"
        else:
            layout = "row_major"

        result.append(spec.with_layout_hints(
            memory_layout=layout,
            blocking_hint=blocking,
            allocation_strategy=alloc,
        ))

    return result
