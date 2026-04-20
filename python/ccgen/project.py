"""Projection engine: project Hbar onto manifolds."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterator

from .indices import Index, make_occ, make_vir, OCC_POOL, VIR_POOL, extend_pool
from .tensors import Tensor
from .sqops import SQOp, create, annihilate
from .expr import OpTerm, Expr
from .wick import wick_contract, apply_deltas
from .connectivity import is_connected

try:
    from . import _wickaccel
except ImportError:  # pragma: no cover - exercised without the extension
    _wickaccel = None


@dataclass(frozen=True)
class AlgebraTerm:
    """A contracted tensor term after Wick projection."""

    coeff: Fraction
    factors: tuple[Tensor, ...]
    free_indices: tuple[Index, ...]
    summed_indices: tuple[Index, ...]
    connected: bool
    provenance: tuple[object, ...] = ()

    def scaled(self, s: Fraction | int) -> AlgebraTerm:
        return AlgebraTerm(
            self.coeff * Fraction(s),
            self.factors,
            self.free_indices,
            self.summed_indices,
            self.connected,
            self.provenance,
        )

    def with_factors(self, factors: tuple[Tensor, ...]) -> AlgebraTerm:
        return AlgebraTerm(
            self.coeff, factors, self.free_indices,
            self.summed_indices, self.connected, self.provenance,
        )

    def __repr__(self) -> str:
        c = str(self.coeff) if self.coeff != 1 else ""
        facs = " ".join(repr(f) for f in self.factors)
        tag = " [disconn]" if not self.connected else ""
        return f"{c} {facs}{tag}".strip()


def _proj_occ(names: str) -> list[Index]:
    return [make_occ(n, dummy=False) for n in names]


def _proj_vir(names: str) -> list[Index]:
    return [make_vir(n, dummy=False) for n in names]


def reference_projector() -> tuple[tuple[SQOp, ...], tuple[Index, ...]]:
    """<Phi_0| — no operators, no free indices."""
    return (), ()


def nfold_projector(n: int) -> tuple[tuple[SQOp, ...], tuple[Index, ...]]:
    """Build the n-fold excitation projector <Phi_{i1…in}^{a1…an}|.

    Returns (operator_string, free_indices) where the operators are
    a+_{i1} … a+_{in} a_{an} … a_{a1}  (creators on occ, annihilators
    on vir, annihilators in reverse order).
    """
    extend_pool(OCC_POOL, n)
    extend_pool(VIR_POOL, n)

    occ = [make_occ(OCC_POOL[k], dummy=False) for k in range(n)]
    vir = [make_vir(VIR_POOL[k], dummy=False) for k in range(n)]

    ops = tuple(create(i) for i in occ) + tuple(
        annihilate(a) for a in reversed(vir)
    )
    free = tuple(occ) + tuple(vir)
    return ops, free


def singles_projector() -> tuple[tuple[SQOp, ...], tuple[Index, ...]]:
    """<Phi_i^a| = <Phi_0| a+_i a_a."""
    return nfold_projector(1)


def doubles_projector() -> tuple[tuple[SQOp, ...], tuple[Index, ...]]:
    """<Phi_{ij}^{ab}| = <Phi_0| a+_i a+_j a_b a_a."""
    return nfold_projector(2)


def triples_projector() -> tuple[tuple[SQOp, ...], tuple[Index, ...]]:
    """<Phi_{ijk}^{abc}| = <Phi_0| a+_i a+_j a+_k a_c a_b a_a."""
    return nfold_projector(3)


# ── Manifold name ↔ excitation rank mapping ──────────────────────────

MANIFOLD_NAMES: dict[int, str] = {
    0: "energy",
    1: "singles",
    2: "doubles",
    3: "triples",
    4: "quadruples",
    5: "quintuples",
    6: "sextuples",
}

_NAME_TO_RANK: dict[str, int] = {v: k for k, v in MANIFOLD_NAMES.items()}
_NAME_TO_RANK["reference"] = 0


def manifold_rank(name: str) -> int:
    """Map a manifold name to its excitation rank."""
    if name in _NAME_TO_RANK:
        return _NAME_TO_RANK[name]
    raise ValueError(f"Unknown manifold '{name}'")


def manifold_name(rank: int) -> str:
    """Map an excitation rank to the canonical manifold name."""
    if rank in MANIFOLD_NAMES:
        return MANIFOLD_NAMES[rank]
    return f"rank{rank}"


def _get_projector(
    manifold: str,
) -> tuple[tuple[SQOp, ...], tuple[Index, ...]]:
    """Return the projector for any named manifold."""
    rank = manifold_rank(manifold)
    if rank == 0:
        return reference_projector()
    return nfold_projector(rank)


def _assign_block_ids(
    proj_ops: tuple[SQOp, ...],
    term: OpTerm,
) -> tuple[list[SQOp], list[int], int]:
    """Assign block IDs to the combined operator string."""
    combined: list[SQOp] = list(proj_ops)
    block_ids: list[int] = [0] * len(proj_ops)

    block = 1
    ops_per_tensor = []
    for tens in term.tensors:
        ops_per_tensor.append(tens.rank)

    idx = 0
    for n_ops in ops_per_tensor:
        for _ in range(n_ops):
            if idx < len(term.sqops):
                combined.append(term.sqops[idx])
                block_ids.append(block)
                idx += 1
        block += 1

    while idx < len(term.sqops):
        combined.append(term.sqops[idx])
        block_ids.append(block)
        idx += 1
        block += 1

    n_blocks = block
    return combined, block_ids, n_blocks


@lru_cache(maxsize=None)
def _term_signature_counts(
    signature: tuple[tuple[str, str], ...],
) -> tuple[int, int, int, int, int, int]:
    create_occ = 0
    create_vir = 0
    create_gen = 0
    annihilate_occ = 0
    annihilate_vir = 0
    annihilate_gen = 0

    for kind, space in signature:
        if kind == "create":
            if space == "occ":
                create_occ += 1
            elif space == "vir":
                create_vir += 1
            else:
                create_gen += 1
        else:
            if space == "occ":
                annihilate_occ += 1
            elif space == "vir":
                annihilate_vir += 1
            else:
                annihilate_gen += 1

    return (
        create_occ,
        create_vir,
        create_gen,
        annihilate_occ,
        annihilate_vir,
        annihilate_gen,
    )


@lru_cache(maxsize=None)
def _is_balanced_signature(
    signature: tuple[tuple[str, str], ...],
) -> bool:
    create_total = 0
    annihilate_total = 0
    for kind, _space in signature:
        if kind == "create":
            create_total += 1
        else:
            annihilate_total += 1
    return create_total == annihilate_total


@lru_cache(maxsize=None)
def _counts_can_fully_contract(
    create_occ: int,
    create_vir: int,
    create_gen: int,
    annihilate_occ: int,
    annihilate_vir: int,
    annihilate_gen: int,
) -> bool:
    """Necessary-condition feasibility check for a residual operator count tuple."""
    create_total = create_occ + create_vir + create_gen
    annihilate_total = annihilate_occ + annihilate_vir + annihilate_gen
    if create_total != annihilate_total:
        return False
    occ_deficit = abs(create_occ - annihilate_occ)
    vir_deficit = abs(annihilate_vir - create_vir)
    return occ_deficit + vir_deficit <= create_gen + annihilate_gen


@lru_cache(maxsize=None)
def _can_term_contribute_to_rank(
    signature: tuple[tuple[str, str], ...],
    rank: int,
) -> bool:
    """Quick necessary-condition filter before Wick expansion."""
    if not _is_balanced_signature(signature):
        return False
    if rank == 0:
        return True
    if len(signature) < 2 * rank:
        return False

    (
        create_occ,
        create_vir,
        create_gen,
        annihilate_occ,
        annihilate_vir,
        annihilate_gen,
    ) = _term_signature_counts(signature)

    # Projector creators on occupied indices must contract against
    # annihilate(occ/gen) slots in the term; projector annihilators on
    # virtual indices must contract against create(vir/gen) slots.
    if annihilate_occ + annihilate_gen < rank:
        return False
    if create_vir + create_gen < rank:
        return False

    min_create_gen = max(0, rank - create_vir)
    max_create_gen = min(rank, create_gen)
    min_annihilate_gen = max(0, rank - annihilate_occ)
    max_annihilate_gen = min(rank, annihilate_gen)

    for use_create_gen in range(min_create_gen, max_create_gen + 1):
        use_create_vir = rank - use_create_gen
        create_vir_left = create_vir - use_create_vir
        create_gen_left = create_gen - use_create_gen

        for use_annihilate_gen in range(
            min_annihilate_gen,
            max_annihilate_gen + 1,
        ):
            use_annihilate_occ = rank - use_annihilate_gen
            annihilate_occ_left = annihilate_occ - use_annihilate_occ
            annihilate_gen_left = annihilate_gen - use_annihilate_gen

            if _counts_can_fully_contract(
                create_occ,
                create_vir_left,
                create_gen_left,
                annihilate_occ_left,
                annihilate_vir,
                annihilate_gen_left,
            ):
                return True

    return False


@lru_cache(maxsize=None)
def _contributing_ranks(
    signature: tuple[tuple[str, str], ...],
    ranks: tuple[int, ...],
) -> tuple[int, ...]:
    if not _is_balanced_signature(signature):
        return ()
    return tuple(
        rank for rank in ranks if _can_term_contribute_to_rank(signature, rank)
    )


def _term_signature(term: OpTerm) -> tuple[tuple[str, str], ...]:
    return tuple((op.kind, op.index.space) for op in term.sqops)


def _term_block_kinds(
    term: OpTerm,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Group a term's operators into tensor/loose-operator connectivity blocks."""
    blocks: list[tuple[tuple[str, str], ...]] = []
    idx = 0
    for tens in term.tensors:
        block: list[tuple[str, str]] = []
        for _ in range(tens.rank):
            if idx >= len(term.sqops):
                break
            op = term.sqops[idx]
            block.append((op.kind, op.index.space))
            idx += 1
        if block:
            blocks.append(tuple(block))

    while idx < len(term.sqops):
        op = term.sqops[idx]
        blocks.append(((op.kind, op.index.space),))
        idx += 1

    return tuple(blocks)


def _blocks_can_contract(
    lhs: tuple[tuple[str, str], ...],
    rhs: tuple[tuple[str, str], ...],
) -> bool:
    for left_kind, left_space in lhs:
        for right_kind, right_space in rhs:
            if left_kind == "create" and right_kind == "annihilate":
                if left_space in ("occ", "gen") and right_space in ("occ", "gen"):
                    return True
            elif left_kind == "annihilate" and right_kind == "create":
                if left_space in ("vir", "gen") and right_space in ("vir", "gen"):
                    return True
    return False


def _projector_connectors(
    block: tuple[tuple[str, str], ...],
) -> tuple[bool, bool]:
    occ_side = any(
        kind == "annihilate" and space in ("occ", "gen")
        for kind, space in block
    )
    vir_side = any(
        kind == "create" and space in ("vir", "gen")
        for kind, space in block
    )
    return occ_side, vir_side


def _component_labels_from_adjacency(
    n_nodes: int,
    adjacency: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            union(node, neighbor)

    remap: dict[int, int] = {}
    labels: list[int] = []
    next_label = 0
    for node in range(n_nodes):
        root = find(node)
        if root not in remap:
            remap[root] = next_label
            next_label += 1
        labels.append(remap[root])
    return tuple(labels)


def _can_term_blocks_connect_to_rank(
    term: OpTerm,
    rank: int,
) -> bool:
    """Stronger necessary-condition filter using term block connectivity."""
    if rank == 0:
        return True

    blocks = _term_block_kinds(term)
    if not blocks:
        return False

    adjacency: list[list[int]] = [[] for _ in blocks]
    for i, lhs in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            if _blocks_can_contract(lhs, blocks[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    labels = _component_labels_from_adjacency(
        len(blocks),
        tuple(tuple(neighbors) for neighbors in adjacency),
    )
    component_count = max(labels, default=-1) + 1
    occ_only = 0
    vir_only = 0
    both = 0

    for component in range(component_count):
        component_blocks = [
            blocks[pos] for pos, label in enumerate(labels) if label == component
        ]
        occ_side = False
        vir_side = False
        for block in component_blocks:
            block_occ, block_vir = _projector_connectors(block)
            occ_side = occ_side or block_occ
            vir_side = vir_side or block_vir
        if not occ_side and not vir_side:
            return False
        if occ_side and vir_side:
            both += 1
        elif occ_side:
            occ_only += 1
        else:
            vir_only += 1

    if occ_only > rank or vir_only > rank:
        return False
    if occ_only + vir_only + both > 2 * rank:
        return False
    return True


def _term_can_contribute(term: OpTerm, manifold: str) -> bool:
    rank = manifold_rank(manifold)
    signature = _term_signature(term)
    if not _can_term_contribute_to_rank(signature, rank):
        return False
    return _can_term_blocks_connect_to_rank(term, rank)


def bucket_terms_by_manifold(
    hbar: Expr,
    manifolds: tuple[str, ...],
) -> dict[str, tuple[OpTerm, ...]]:
    """Preclassify BCH terms by the manifolds they can possibly hit."""
    by_manifold: dict[str, list[OpTerm]] = {name: [] for name in manifolds}
    ranks = tuple(manifold_rank(name) for name in manifolds)
    name_by_rank = {
        manifold_rank(name): name for name in manifolds
    }

    for term in hbar.terms:
        if term.coeff == 0:
            continue
        signature = _term_signature(term)
        for rank in _contributing_ranks(signature, ranks):
            if _can_term_blocks_connect_to_rank(term, rank):
                by_manifold[name_by_rank[rank]].append(term)

    return {
        name: tuple(terms) for name, terms in by_manifold.items()
    }


def _classify_indices(
    tensors: tuple[Tensor, ...],
    projector_free: tuple[Index, ...],
) -> tuple[tuple[Index, ...], tuple[Index, ...]]:
    """Separate free and summed indices from the tensor factors."""
    free_set = frozenset(projector_free)
    all_indices: list[Index] = []
    for t in tensors:
        all_indices.extend(t.indices)

    if _wickaccel is not None and all_indices:
        slot_order = {
            slot: pos for pos, slot in enumerate(sorted(
                {(idx.space, idx.name) for idx in all_indices},
                key=lambda item: (
                    0 if item[0] == "occ" else 1 if item[0] == "vir" else 2,
                    item[1],
                ),
            ))
        }
        summed_positions = _wickaccel.classify_summed_indices(
            tuple(0 if idx.space == "occ" else 1 if idx.space == "vir" else 2 for idx in all_indices),
            tuple(slot_order[(idx.space, idx.name)] for idx in all_indices),
            tuple(idx.is_dummy for idx in all_indices),
            tuple(idx in free_set for idx in all_indices),
        )
        return projector_free, tuple(all_indices[pos] for pos in summed_positions)

    summed: set[Index] = set()

    for idx in all_indices:
        if idx not in free_set:
            summed.add(idx)

    return projector_free, tuple(
        sorted(summed, key=lambda x: (x.space, x.name))
    )


def iter_projected_terms_from_terms(
    terms: tuple[OpTerm, ...] | list[OpTerm],
    manifold: str,
) -> Iterator[AlgebraTerm]:
    """Yield projected algebra terms for a given excitation manifold."""
    proj_ops, proj_free = _get_projector(manifold)

    for term in terms:
        all_ops = list(proj_ops) + list(term.sqops)
        n_c = sum(1 for op in all_ops if op.is_creator)
        n_a = sum(1 for op in all_ops if op.is_annihilator)
        if n_c != n_a:
            continue

        combined, block_ids, n_blocks = _assign_block_ids(proj_ops, term)
        ignore = 0 if manifold not in ("energy", "reference") else None
        contractions = wick_contract(
            combined,
            term.tensors,
            block_ids,
            require_connected=(manifold not in ("energy", "reference")),
            n_blocks=n_blocks,
            ignore_block=ignore,
        )

        for wr in contractions:
            reduced = apply_deltas(
                wr.tensor_factors,
                wr.deltas,
                protected=proj_free,
            )
            if reduced is None:
                continue
            conn = (
                True
                if manifold not in ("energy", "reference")
                else is_connected(n_blocks, wr.block_edges, ignore_block=ignore)
            )

            free_idx, summed_idx = _classify_indices(reduced, proj_free)
            overall_coeff = term.coeff * Fraction(wr.sign)

            yield AlgebraTerm(
                coeff=overall_coeff,
                factors=reduced,
                free_indices=free_idx,
                summed_indices=summed_idx,
                connected=conn,
                provenance=term.origin,
            )


def iter_projected_terms(hbar: Expr, manifold: str) -> Iterator[AlgebraTerm]:
    """Yield projected algebra terms for a given excitation manifold."""
    manifolds = (manifold,)
    buckets = bucket_terms_by_manifold(hbar, manifolds)
    yield from iter_projected_terms_from_terms(buckets[manifold], manifold)


def project(hbar: Expr, manifold: str) -> list[AlgebraTerm]:
    """Project Hbar onto a given excitation manifold."""
    return list(iter_projected_terms(hbar, manifold))
