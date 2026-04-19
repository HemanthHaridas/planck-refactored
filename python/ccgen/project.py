"""Projection engine: project Hbar onto manifolds."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from .indices import Index, make_occ, make_vir, OCC_POOL, VIR_POOL, extend_pool
from .tensors import Tensor
from .sqops import SQOp, create, annihilate
from .expr import OpTerm, Expr
from .wick import wick_contract, apply_deltas
from .connectivity import is_connected


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


def _classify_indices(
    tensors: tuple[Tensor, ...],
    projector_free: tuple[Index, ...],
) -> tuple[tuple[Index, ...], tuple[Index, ...]]:
    """Separate free and summed indices from the tensor factors."""
    free_set = frozenset(projector_free)
    all_indices: list[Index] = []
    for t in tensors:
        all_indices.extend(t.indices)

    summed: set[Index] = set()

    for idx in all_indices:
        if idx not in free_set:
            summed.add(idx)

    return projector_free, tuple(
        sorted(summed, key=lambda x: (x.space, x.name))
    )


def _same_slot(lhs: Index, rhs: Index) -> bool:
    return lhs.space == rhs.space and lhs.name == rhs.name


def _restore_free_indices(
    tensors: tuple[Tensor, ...],
    free_indices: tuple[Index, ...],
) -> tuple[Tensor, ...]:
    """Replace dummy/free-equivalent slots with canonical indices."""
    if not free_indices:
        return tensors

    mapping: dict[Index, Index] = {}
    free_by_slot = _free_slot_lookup(free_indices)

    for tensor in tensors:
        for idx in tensor.indices:
            slot = (idx.space, idx.name)
            free_idx = free_by_slot.get(slot)
            if free_idx is not None and idx != free_idx:
                mapping[idx] = free_idx

    if not mapping:
        return tensors
    return tuple(t.reindexed(mapping) for t in tensors)


@lru_cache(maxsize=None)
def _free_slot_lookup(
    free_indices: tuple[Index, ...],
) -> dict[tuple[str, str], Index]:
    return {(idx.space, idx.name): idx for idx in free_indices}


def project(hbar: Expr, manifold: str) -> list[AlgebraTerm]:
    """Project Hbar onto a given excitation manifold."""
    proj_ops, proj_free = _get_projector(manifold)
    results: list[AlgebraTerm] = []

    for term in hbar.terms:
        if term.coeff == 0:
            continue

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
            reduced = _restore_free_indices(reduced, proj_free)
            conn = (
                True
                if manifold not in ("energy", "reference")
                else is_connected(n_blocks, wr.block_edges, ignore_block=ignore)
            )

            free_idx, summed_idx = _classify_indices(reduced, proj_free)
            overall_coeff = term.coeff * Fraction(wr.sign)

            results.append(AlgebraTerm(
                coeff=overall_coeff,
                factors=reduced,
                free_indices=free_idx,
                summed_indices=summed_idx,
                connected=conn,
                provenance=term.origin,
            ))

    return results
