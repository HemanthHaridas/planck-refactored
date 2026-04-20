"""Symbolic multiplication, commutator, and BCH expansion."""

from __future__ import annotations

from fractions import Fraction
from math import factorial

from .indices import Index
from .expr import OpTerm, Expr


def _collect_indices(term: OpTerm) -> set[Index]:
    indices: set[Index] = set()
    for t in term.tensors:
        indices.update(t.indices)
    for op in term.sqops:
        indices.add(op.index)
    return indices


def _rename_dummies(term: OpTerm, avoid: set[Index]) -> OpTerm:
    """Rename all dummy indices in *term* that collide with *avoid*."""
    dummies = {idx for idx in _collect_indices(term) if idx.is_dummy}
    collisions = dummies & avoid
    if not collisions:
        return term

    used_names = {(idx.space, idx.name) for idx in dummies | avoid}
    mapping: dict[Index, Index] = {}
    counters: dict[tuple[str, str], int] = {}

    for idx in sorted(collisions, key=lambda item: (item.space, item.name)):
        key = (idx.space, idx.name)
        next_id = counters.get(key, 0)
        while True:
            candidate = f"{idx.name}_{next_id}"
            next_id += 1
            if (idx.space, candidate) not in used_names:
                break
        counters[key] = next_id
        used_names.add((idx.space, candidate))
        mapping[idx] = Index(candidate, idx.space, idx.is_dummy)
    return term.reindexed(mapping)


def multiply_terms(a: OpTerm, b: OpTerm) -> OpTerm:
    """Multiply two operator monomials (formal concatenation)."""
    a_indices = _collect_indices(a)
    b_safe = _rename_dummies(b, a_indices)
    return OpTerm(
        coeff=a.coeff * b_safe.coeff,
        tensors=a.tensors + b_safe.tensors,
        sqops=a.sqops + b_safe.sqops,
        origin=a.origin + b_safe.origin,
    )


def multiply(a: Expr, b: Expr) -> Expr:
    """Distribute multiplication over two expressions: A*B."""
    terms: list[OpTerm] = []
    for ta in a.terms:
        for tb in b.terms:
            terms.append(multiply_terms(ta, tb))
    return Expr(terms)


def commutator(a: Expr, b: Expr) -> Expr:
    """[A, B] = AB - BA (formal, no contractions)."""
    return multiply(a, b) - multiply(b, a)


def bch_expand(H: Expr, T: Expr, max_order: int = 4) -> Expr:
    """Compute the similarity-transformed Hamiltonian via finite BCH.

    H-bar = H + [H,T] + 1/2![[H,T],T] + ... + 1/n![...[H,T]...,T]
    """
    result = H.copy().combine_like_terms()
    current = H.copy().combine_like_terms()
    for n in range(1, max_order + 1):
        current = commutator(current, T).combine_like_terms()
        scale = Fraction(1, factorial(n))
        result = (result + current * scale).combine_like_terms()
    return result.drop_zeros().combine_like_terms()
