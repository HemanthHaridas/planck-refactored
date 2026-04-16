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

from ..indices import Index
from ..project import AlgebraTerm
from ..tensors import Tensor
from ..canonicalize import canonicalize_tensor


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
        SIZE_EST = {"occ": 30, "vir": 100, "gen": 50}
        total = 1
        for idx in self.indices:
            total *= SIZE_EST.get(idx.space, 50)
        return total

    @property
    def estimated_bytes(self) -> int:
        """Estimated memory in bytes (double precision)."""
        return self.estimated_elements * 8

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
            memory_layout=memory_layout or self.memory_layout,
            blocking_hint=blocking_hint or self.blocking_hint,
            allocation_strategy=allocation_strategy or self.allocation_strategy,
        )


def _subcontraction_signature(
    factors: tuple[Tensor, ...],
    summed: frozenset[Index],
) -> tuple[tuple[str, ...], str]:
    """Build a hashable signature for a sub-contraction.

    The signature is independent of dummy index names — it captures
    (sorted tensor names, index-space pattern).
    """
    factor_names = tuple(sorted(f.name for f in factors))

    # Collect all indices and classify as internal (summed within
    # the sub-contraction) vs external (free wrt the sub-contraction).
    all_indices: list[Index] = []
    for f in factors:
        all_indices.extend(f.indices)

    idx_count: Counter[Index] = Counter()
    for f in factors:
        for idx in f.indices:
            idx_count[idx] += 1

    # Internal = summed AND appears only within these factors
    # External = free indices or summed indices shared with other factors
    internal = frozenset(
        idx for idx in summed if idx_count[idx] >= 2
    )
    external = frozenset(all_indices) - internal

    # Space signature of external indices (the "shape" of the intermediate)
    ext_sorted = sorted(external, key=lambda x: (x.space, x.name))
    space_sig = "".join(
        "o" if idx.space == "occ" else "v" if idx.space == "vir" else "g"
        for idx in ext_sorted
    )

    return factor_names, space_sig


@dataclass(frozen=True)
class _SubContractionKey:
    """Hashable key for a sub-contraction pattern."""

    factor_names: tuple[str, ...]
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

            factor_names, space_sig = _subcontraction_signature(
                sub_factors, summed_set,
            )
            key = _SubContractionKey(
                factor_names=factor_names,
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

    Returns
    -------
    list of IntermediateSpec, sorted by usage count (descending)
    """
    # Phase 1: count how often each sub-contraction key appears
    key_count: Counter[_SubContractionKey] = Counter()
    key_examples: dict[_SubContractionKey, list[tuple[str, int, tuple[int, ...]]]] = {}

    for manifold, terms in equations.items():
        for term_idx, term in enumerate(terms):
            subs = _extract_subcontractions(term)
            seen_keys: set[_SubContractionKey] = set()
            for key, factor_indices in subs:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                key_count[key] += 1
                key_examples.setdefault(key, []).append(
                    (manifold, term_idx, factor_indices)
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

        # Use first example to determine external indices
        manifold, term_idx, factor_indices = key_examples[key][0]
        term = equations[manifold][term_idx]
        sub_factors = tuple(term.factors[i] for i in factor_indices)
        summed_set = frozenset(term.summed_indices)

        sub_indices: Counter[Index] = Counter()
        for f in sub_factors:
            for idx in f.indices:
                sub_indices[idx] += 1

        internal = frozenset(
            idx for idx in sub_indices
            if idx in summed_set and sub_indices[idx] >= 2
        )
        external_indices = tuple(
            idx for idx in _ordered_external(sub_factors, internal)
        )

        # Build a definition term from the sub-contraction
        internal_summed = tuple(sorted(
            internal, key=lambda x: (x.space, x.name),
        ))
        definition = AlgebraTerm(
            coeff=Fraction(1),
            factors=sub_factors,
            free_indices=external_indices,
            summed_indices=internal_summed,
            connected=True,
        )

        results.append(IntermediateSpec(
            name=name,
            indices=external_indices,
            definition_terms=(definition,),
            usage_count=count,
            index_space_sig=key.space_sig,
        ))

    return results


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
        match = _find_subfactors(term, defn.factors)
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

    # Greedy matching: pick first available factor for each pattern factor
    used: set[int] = set()
    matched: list[int] = []
    index_mapping: dict[Index, Index] = {}

    for pf in pattern_factors:
        found = False
        for ti in term_by_name.get(pf.name, []):
            if ti in used:
                continue
            tf = term.factors[ti]
            if tf.rank != pf.rank:
                continue
            # Build index mapping
            local_map: dict[Index, Index] = {}
            conflict = False
            for pi, ti_idx in zip(pf.indices, tf.indices):
                if pi in local_map:
                    if local_map[pi] != ti_idx:
                        conflict = True
                        break
                elif pi in index_mapping:
                    if index_mapping[pi] != ti_idx:
                        conflict = True
                        break
                else:
                    local_map[pi] = ti_idx
            if conflict:
                continue
            index_mapping.update(local_map)
            used.add(ti)
            matched.append(ti)
            found = True
            break
        if not found:
            return None

    return frozenset(matched), index_mapping


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
