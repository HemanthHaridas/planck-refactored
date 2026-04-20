"""Wick contraction engine for Fermi-vacuum matrix elements."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Sequence

from .connectivity import is_connected
from .indices import Index
from .tensors import Tensor, delta, reindex_tensors
from .sqops import SQOp

try:
    from . import _wickaccel
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    _wickaccel = None

_KIND_CREATE = 0
_KIND_ANNIHILATE = 1

_SPACE_OCC = 0
_SPACE_VIR = 1
_SPACE_GEN = 2

_POS_MASK = (1 << 16) - 1
_KIND_SHIFT = 16
_SPACE_SHIFT = 18
_BLOCK_SHIFT = 20
_EDGE_LOW_MASK = (1 << 20) - 1


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
    return _can_contract_codes(
        _encode_kind(left_kind),
        _encode_space(left_space),
        _encode_kind(right_kind),
        _encode_space(right_space),
    )


def _encode_kind(kind: str) -> int:
    return _KIND_CREATE if kind == "create" else _KIND_ANNIHILATE


def _encode_space(space: str) -> int:
    if space == "occ":
        return _SPACE_OCC
    if space == "vir":
        return _SPACE_VIR
    return _SPACE_GEN


def _can_contract_codes(
    left_kind: int,
    left_space: int,
    right_kind: int,
    right_space: int,
) -> bool:
    if left_kind == _KIND_CREATE and right_kind == _KIND_ANNIHILATE:
        return left_space in (_SPACE_OCC, _SPACE_GEN) and right_space in (
            _SPACE_OCC, _SPACE_GEN
        )

    if left_kind == _KIND_ANNIHILATE and right_kind == _KIND_CREATE:
        return left_space in (_SPACE_VIR, _SPACE_GEN) and right_space in (
            _SPACE_VIR, _SPACE_GEN
        )

    return False


def _pack_signature_op(
    pos: int,
    kind_code: int,
    space_code: int,
    block_id: int,
) -> int:
    return (
        pos
        | (kind_code << _KIND_SHIFT)
        | (space_code << _SPACE_SHIFT)
        | (block_id << _BLOCK_SHIFT)
    )


def _word_pos(word: int) -> int:
    return word & _POS_MASK


def _word_kind(word: int) -> int:
    return (word >> _KIND_SHIFT) & 0b11


def _word_space(word: int) -> int:
    return (word >> _SPACE_SHIFT) & 0b11


def _word_block(word: int) -> int:
    return word >> _BLOCK_SHIFT


def _pack_edge(a: int, b: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return (hi << _BLOCK_SHIFT) | lo


def _unpack_edge(word: int) -> tuple[int, int]:
    return word & _EDGE_LOW_MASK, word >> _BLOCK_SHIFT


def _can_contract(left: SQOp, right: SQOp) -> bool:
    """Check whether *left* and *right* can form a nonzero contraction."""
    return _can_contract_codes(
        _encode_kind(left.kind),
        _encode_space(left.index.space),
        _encode_kind(right.kind),
        _encode_space(right.index.space),
    )


def _signature_positions_can_contract(
    signature: tuple[int, ...],
    left_pos: int,
    right_pos: int,
) -> bool:
    lo, hi = sorted((left_pos, right_pos))
    left = signature[lo]
    right = signature[hi]
    return _can_contract_codes(
        _word_kind(left),
        _word_space(left),
        _word_kind(right),
        _word_space(right),
    )


def _analyze_signature_python(
    signature: tuple[int, ...],
    current_edges: tuple[int, ...],
    n_blocks: int,
    ignore_block: int | None,
) -> tuple[bool, int, tuple[int, ...]]:
    if not _can_fully_contract(signature):
        return False, -1, ()

    if n_blocks >= 0 and not _can_still_be_connected(
        signature,
        _normalize_edges(current_edges),
        n_blocks,
        ignore_block,
    ):
        return False, -1, ()

    pivot_pos = _choose_pivot_position(signature)
    if pivot_pos < 0:
        return False, -1, ()

    candidates = _candidate_partner_positions(signature, pivot_pos)
    if not candidates:
        return False, -1, ()
    return True, pivot_pos, candidates


@lru_cache(maxsize=None)
def _analyze_signature_cached(
    signature: tuple[int, ...],
    current_edges: tuple[int, ...],
    n_blocks: int,
    ignore_block: int | None,
) -> tuple[bool, int, tuple[int, ...]]:
    if _wickaccel is not None:
        return _wickaccel.analyze_signature(
            signature,
            current_edges,
            n_blocks,
            -1 if ignore_block is None else ignore_block,
        )
    return _analyze_signature_python(
        signature,
        current_edges,
        n_blocks,
        ignore_block,
    )


@lru_cache(maxsize=None)
def _can_fully_contract(
    signature: tuple[int, ...],
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

    for word in signature:
        kind = _word_kind(word)
        space = _word_space(word)
        if space == _SPACE_GEN:
            gen_count += 1
        elif kind == _KIND_CREATE and space == _SPACE_OCC:
            create_occ += 1
        elif kind == _KIND_ANNIHILATE and space == _SPACE_OCC:
            annihilate_occ += 1
        elif kind == _KIND_ANNIHILATE and space == _SPACE_VIR:
            annihilate_vir += 1
        elif kind == _KIND_CREATE and space == _SPACE_VIR:
            create_vir += 1

    # Occ contractions: create_occ must pair with annihilate_occ
    # Vir contractions: annihilate_vir must pair with create_vir
    # Gen operators can fill any deficit
    occ_deficit = abs(create_occ - annihilate_occ)
    vir_deficit = abs(annihilate_vir - create_vir)

    return occ_deficit + vir_deficit <= gen_count


_PAIRING_CACHE_MAX_OPS = 12


@lru_cache(maxsize=None)
def _wick_pairings_cached(
    signature: tuple[int, ...],
) -> tuple[
    tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]],
    ...,
]:
    """Enumerate contraction topologies for a structural operator signature."""
    return tuple(_iter_wick_pairings_uncached(signature))


@lru_cache(maxsize=None)
def _candidate_partner_positions(
    signature: tuple[int, ...],
    pivot_pos: int,
) -> tuple[int, ...]:
    pivot = signature[pivot_pos]
    candidates: list[int] = []
    for pos, partner in enumerate(signature):
        if pos == pivot_pos:
            continue
        if _word_block(pivot) == _word_block(partner):
            continue
        if _signature_positions_can_contract(signature, pivot_pos, pos):
            candidates.append(pos)
    return tuple(candidates)


@lru_cache(maxsize=None)
def _choose_pivot_position(
    signature: tuple[int, ...],
) -> int:
    best_pos = -1
    best_count: int | None = None
    best_span: int | None = None

    for pos in range(len(signature)):
        candidates = _candidate_partner_positions(signature, pos)
        count = len(candidates)
        if count == 0:
            return -1
        span = min(abs(partner - pos) for partner in candidates)
        if (
            best_count is None
            or count < best_count
            or (count == best_count and span < best_span)
            or (count == best_count and span == best_span and pos < best_pos)
        ):
            best_pos = pos
            best_count = count
            best_span = span

    return best_pos


def _remove_positions(
    signature: tuple[int, ...],
    left: int,
    right: int,
) -> tuple[int, ...]:
    lo, hi = sorted((left, right))
    return signature[:lo] + signature[lo + 1:hi] + signature[hi + 1:]


def _iter_wick_pairings_uncached(
    signature: tuple[int, ...],
    require_connected: bool = False,
    n_blocks: int | None = None,
    ignore_block: int | None = None,
    current_edges: tuple[int, ...] = (),
) -> Iterator[
    tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]
]:
    if not signature:
        if not require_connected or (
            n_blocks is not None and _pairing_is_connected(
                n_blocks,
                _normalize_edges(current_edges),
                ignore_block,
            )
        ):
            yield (1, (), ())
        return

    can_continue, pivot_pos, partner_positions = _analyze_signature_cached(
        signature,
        _normalize_edges(current_edges),
        n_blocks if require_connected and n_blocks is not None else -1,
        ignore_block if require_connected else None,
    )
    if not can_continue:
        return

    pivot = signature[pivot_pos]
    for partner_pos in partner_positions:
        partner = signature[partner_pos]
        remaining = _remove_positions(signature, pivot_pos, partner_pos)

        gap = abs(partner_pos - pivot_pos) - 1
        sign_factor = -1 if gap % 2 else 1
        pair = (_word_pos(pivot), _word_pos(partner))
        edge = _pack_edge(_word_block(pivot), _word_block(partner))
        next_edges = current_edges + (edge,)
        for sub_sign, sub_pairs, sub_edges in _iter_wick_pairings(
            remaining,
            require_connected=require_connected,
            n_blocks=n_blocks,
            ignore_block=ignore_block,
            current_edges=next_edges,
        ):
            yield (
                sign_factor * sub_sign,
                (pair,) + sub_pairs,
                (edge,) + sub_edges,
            )


def _iter_wick_pairings(
    signature: tuple[int, ...],
    require_connected: bool = False,
    n_blocks: int | None = None,
    ignore_block: int | None = None,
    current_edges: tuple[int, ...] = (),
) -> Iterator[
    tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]
]:
    """Yield contraction topologies without materializing huge pairing sets."""
    if not require_connected and len(signature) <= _PAIRING_CACHE_MAX_OPS:
        yield from _wick_pairings_cached(signature)
        return
    yield from _iter_wick_pairings_uncached(
        signature,
        require_connected=require_connected,
        n_blocks=n_blocks,
        ignore_block=ignore_block,
        current_edges=current_edges,
    )


def _normalize_edges(
    edges: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(sorted(edges))


def _relevant_blocks(
    n_blocks: int,
    ignore_block: int | None,
) -> tuple[int, ...]:
    return tuple(
        block for block in range(n_blocks) if block != ignore_block
    )


def _component_labels(
    blocks: tuple[int, ...],
    edges: tuple[int, ...],
) -> tuple[int, ...]:
    if not blocks:
        return ()

    index_of = {block: pos for pos, block in enumerate(blocks)}
    parent = list(range(len(blocks)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        a, b = _unpack_edge(edge)
        if a in index_of and b in index_of:
            union(index_of[a], index_of[b])

    labels = [find(i) for i in range(len(blocks))]
    remap: dict[int, int] = {}
    next_label = 0
    normalized: list[int] = []
    for label in labels:
        if label not in remap:
            remap[label] = next_label
            next_label += 1
        normalized.append(remap[label])
    return tuple(normalized)


@lru_cache(maxsize=None)
def _can_still_be_connected(
    signature: tuple[int, ...],
    current_edges: tuple[int, ...],
    n_blocks: int,
    ignore_block: int | None,
) -> bool:
    blocks = _relevant_blocks(n_blocks, ignore_block)
    component_by_pos = _component_labels(blocks, current_edges)
    n_components = len(set(component_by_pos))
    if n_components <= 1:
        return True

    component_of_block = {
        block: component_by_pos[pos] for pos, block in enumerate(blocks)
    }
    candidate_edges = set(current_edges)
    component_edges: set[tuple[int, int]] = set()
    external_capable_ops: set[int] = set()
    for i, left in enumerate(signature):
        for j in range(i + 1, len(signature)):
            right = signature[j]
            if _word_block(left) == _word_block(right):
                continue
            if _signature_positions_can_contract(signature, i, j):
                left_component = component_of_block.get(_word_block(left))
                right_component = component_of_block.get(_word_block(right))
                if left_component is None or right_component is None:
                    continue
                candidate_edges.add(_pack_edge(_word_block(left), _word_block(right)))
                if left_component != right_component:
                    component_edges.add((
                        min(left_component, right_component),
                        max(left_component, right_component),
                    ))
                    external_capable_ops.add(i)
                    external_capable_ops.add(j)

    if not component_edges:
        return False

    component_graph_edges = tuple(
        _pack_edge(a, b) for a, b in sorted(component_edges)
    )
    if not _pairing_is_connected(
        n_components,
        component_graph_edges,
        ignore_block=None,
    ):
        return False

    component_stub_counts = [0] * n_components
    for op_pos in external_capable_ops:
        component = component_of_block.get(_word_block(signature[op_pos]))
        if component is not None:
            component_stub_counts[component] += 1

    if any(count == 0 for count in component_stub_counts):
        return False

    if sum(component_stub_counts) < 2 * (n_components - 1):
        return False

    return _pairing_is_connected(
        n_blocks,
        tuple(sorted(candidate_edges)),
        ignore_block,
    )


def wick_contract(
    sqops: Sequence[SQOp],
    tensors: tuple[Tensor, ...],
    block_ids: Sequence[int] | None = None,
    require_connected: bool = False,
    n_blocks: int | None = None,
    ignore_block: int | None = None,
) -> Iterator[WickResult]:
    """Enumerate all fully-contracted Wick pairings."""
    if block_ids is None:
        block_ids = [0] * len(sqops)
    if n_blocks is None:
        n_blocks = max(block_ids, default=-1) + 1

    signature = tuple(
        _pack_signature_op(
            i,
            _encode_kind(op.kind),
            _encode_space(op.index.space),
            bid,
        )
        for i, (op, bid) in enumerate(zip(sqops, block_ids))
    )
    for sign, pair_positions, edges in _iter_wick_pairings(
        signature,
        require_connected=require_connected,
        n_blocks=n_blocks,
        ignore_block=ignore_block,
    ):
        if require_connected and not _pairing_is_connected(
            n_blocks,
            edges,
            ignore_block,
        ):
            continue
        deltas = tuple(
            (sqops[i].index, sqops[j].index) for i, j in pair_positions
        )
        yield WickResult(
            sign=sign,
            deltas=deltas,
            tensor_factors=tensors,
            block_edges=tuple(_unpack_edge(edge) for edge in edges),
        )


@lru_cache(maxsize=None)
def _pairing_is_connected(
    n_blocks: int,
    edges: tuple[int, ...],
    ignore_block: int | None,
) -> bool:
    return is_connected(
        n_blocks,
        tuple(_unpack_edge(edge) for edge in edges),
        ignore_block=ignore_block,
    )


@lru_cache(maxsize=None)
def _tensor_unique_indices(
    tensors: tuple[Tensor, ...],
) -> tuple[Index, ...]:
    """Return unique tensor indices in first-appearance order."""
    seen: set[Index] = set()
    ordered: list[Index] = []
    for tensor in tensors:
        for idx in tensor.indices:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
    return tuple(ordered)


@lru_cache(maxsize=None)
def _delta_layout_metadata(
    indices: tuple[Index, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...]]:
    slot_order = {
        slot: pos for pos, slot in enumerate(sorted(
            {(idx.space, idx.name) for idx in indices},
            key=lambda item: (
                0 if item[0] == "occ" else 1 if item[0] == "vir" else 2,
                item[1],
            ),
        ))
    }
    return (
        tuple(_encode_space(idx.space) for idx in indices),
        tuple(slot_order[(idx.space, idx.name)] for idx in indices),
        tuple(idx.is_dummy for idx in indices),
    )


def apply_deltas(
    tensors: tuple[Tensor, ...],
    deltas: tuple[tuple[Index, Index], ...],
    protected: tuple[Index, ...] = (),
) -> tuple[Tensor, ...] | None:
    """Apply Kronecker deltas as index substitutions."""
    tensor_indices = _tensor_unique_indices(tensors)
    all_indices: list[Index] = list(tensor_indices)
    index_pos: dict[Index, int] = {idx: pos for pos, idx in enumerate(all_indices)}

    def _ensure_index(idx: Index) -> int:
        pos = index_pos.get(idx)
        if pos is not None:
            return pos
        pos = len(all_indices)
        all_indices.append(idx)
        index_pos[idx] = pos
        return pos

    protected_pos = tuple(_ensure_index(idx) for idx in protected)
    delta_pos = tuple(
        (_ensure_index(lhs), _ensure_index(rhs))
        for lhs, rhs in deltas
    )
    protected_rank = {pos: rank for rank, pos in enumerate(protected_pos)}
    protected_by_slot = {
        (idx.space, idx.name): idx for idx in protected
    }
    if _wickaccel is not None and delta_pos:
        all_indices_tuple = tuple(all_indices)
        space_codes, name_codes, dummy_mask = _delta_layout_metadata(
            all_indices_tuple
        )
        ok, roots = _wickaccel.apply_deltas_layout(
            space_codes,
            name_codes,
            dummy_mask,
            tuple(protected_rank.get(pos, -1) for pos in range(len(all_indices))),
            delta_pos,
        )
        if not ok:
            return None
        root_by_pos = tuple(int(root) for root in roots)
    else:
        parent = list(range(len(all_indices)))

        def find(pos: int) -> int:
            while parent[pos] != pos:
                parent[pos] = parent[parent[pos]]
                pos = parent[pos]
            return pos

        def _prefer(lhs_pos: int, rhs_pos: int) -> tuple[int, int]:
            lhs_idx = all_indices[lhs_pos]
            rhs_idx = all_indices[rhs_pos]
            lhs_protected = lhs_pos in protected_rank
            rhs_protected = rhs_pos in protected_rank

            if lhs_protected != rhs_protected:
                return (lhs_pos, rhs_pos) if lhs_protected else (rhs_pos, lhs_pos)

            if lhs_protected and rhs_protected:
                lhs_rank = protected_rank[lhs_pos]
                rhs_rank = protected_rank[rhs_pos]
                return (
                    (lhs_pos, rhs_pos)
                    if lhs_rank <= rhs_rank
                    else (rhs_pos, lhs_pos)
                )

            if rhs_idx.space != "gen" and lhs_idx.space == "gen":
                return rhs_pos, lhs_pos
            if lhs_idx.space == rhs_idx.space and rhs_idx.name < lhs_idx.name:
                return rhs_pos, lhs_pos
            return lhs_pos, rhs_pos

        def union(lhs_pos: int, rhs_pos: int) -> bool:
            lhs_root = find(lhs_pos)
            rhs_root = find(rhs_pos)
            if lhs_root == rhs_root:
                return True
            lhs_idx = all_indices[lhs_root]
            rhs_idx = all_indices[rhs_root]
            if (
                lhs_idx.space != "gen"
                and rhs_idx.space != "gen"
                and lhs_idx.space != rhs_idx.space
            ):
                return False
            lhs_root, rhs_root = _prefer(lhs_root, rhs_root)
            parent[rhs_root] = lhs_root
            return True

        for lhs_pos, rhs_pos in delta_pos:
            if not union(lhs_pos, rhs_pos):
                return None
        root_by_pos = tuple(find(pos) for pos in range(len(all_indices)))

    extra_factors: list[Tensor] = []
    protected_components: dict[int, list[Index]] = {}
    for pos, idx in zip(protected_pos, protected):
        protected_components.setdefault(root_by_pos[pos], []).append(idx)
    for protected_members in protected_components.values():
        if len(protected_members) <= 1:
            continue
        rep = protected_members[0]
        for idx in protected_members[1:]:
            extra_factors.append(delta(rep, idx))

    mapping: dict[Index, Index] = {}
    for idx in tensor_indices:
        root_idx = all_indices[root_by_pos[index_pos[idx]]]
        protected_target = protected_by_slot.get((root_idx.space, root_idx.name))
        if protected_target is not None:
            mapped = protected_target
        elif idx.is_dummy or root_idx.is_dummy:
            target = root_idx
            mapped = Index(target.name, target.space, is_dummy=True)
        else:
            mapped = root_idx
        if mapped != idx:
            mapping[idx] = mapped

    transformed = reindex_tensors(tensors, mapping)
    if extra_factors:
        transformed = transformed + tuple(extra_factors)
    return transformed
