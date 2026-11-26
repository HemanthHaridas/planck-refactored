"""Λ (left-hand) CC equations by differentiating the CC Lagrangian.

    L(T, Λ) = E(T) + Σ_μ λ_μ R_μ(T)
    Λ residual_ν = ∂L/∂t_ν

E and R_μ are the T equations the generator already produces (the diagram
engine in production), so no new projection is needed: each Λ term is an L
term with one t factor removed and its indices freed. See
``docs/CC_LAMBDA_SCOPE.md``.

Conventions (fixed here, used everywhere):

* λ_n is tensor ``l<n>`` with indices (occ..., vir...), e.g. ``l2(i,j,a,b)``,
  antisymmetric within each group. This is PySCF's ``l1[i,a]``, ``l2[i,j,a,b]``.
* L sums λ over all index tuples with weight 1/(n!)², as T_n does, so
  Σ λ_μ R_μ over unique μ.
* :func:`differentiate` returns G = ∂L/∂T treating every element of T as
  independent, in t's layout (vir..., occ...). The Λ residual is its
  antisymmetrization over the occupied and over the virtual indices:
  ∂L/∂(unique t element) = (n!)² · A(G). Contracted with an antisymmetric
  direction D, Σ G·D = Σ A(G)·D, which is what the finite-difference gate
  checks.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial

from .canonicalize import canonicalize_term_to_fixed_point, merge_term_into_buckets
from .indices import Index
from .project import AlgebraTerm, _NAME_TO_RANK
from .tensors import Tensor


def _lambda_tensor(free: tuple[Index, ...]) -> Tensor:
    occ = [i for i in free if i.space == "occ"]
    vir = [i for i in free if i.space == "vir"]
    n = len(occ)
    assert n == len(vir), f"residual free indices are not an excitation: {free}"
    groups = (tuple(range(n)), tuple(range(n, 2 * n))) if n > 1 else ()
    return Tensor(f"l{n}", tuple(i.as_dummy() for i in occ + vir), groups)


def _as_dummy(term: AlgebraTerm, factors, free, summed, coeff) -> AlgebraTerm:
    return AlgebraTerm(Fraction(coeff), tuple(factors), tuple(free), tuple(summed),
                       True, term.provenance)


def lagrangian_terms(equations: dict[str, list[AlgebraTerm]]) -> list[AlgebraTerm]:
    """Scalar terms of L: the energy terms, and λ_n contracted onto each R_n
    term's free indices with weight 1/(n!)²."""
    terms = list(equations.get("energy", []))
    for manifold, residual in equations.items():
        if _NAME_TO_RANK.get(manifold, 0) == 0:
            continue
        for t in residual:
            free = t.free_indices
            dmap = {i: i.as_dummy() for i in free}
            factors = [f.reindexed(dmap) for f in t.factors] + [_lambda_tensor(free)]
            n = len(free) // 2
            terms.append(_as_dummy(t, factors, (),
                                   tuple(dmap[i] for i in free) + t.summed_indices,
                                   t.coeff / factorial(n) ** 2))
    return terms


def differentiate(terms: list[AlgebraTerm], rank: int) -> list[AlgebraTerm]:
    """G = ∂L/∂t<rank> over every element, canonicalized and merged.

    Free indices are named a, b, ... / i, j, ... in t's slot order, so the
    result evaluates in t's (vir..., occ...) layout."""
    name = f"t{rank}"
    raw = []
    for term in terms:
        for pos, fac in enumerate(term.factors):
            if fac.name != name:
                continue
            # Fresh names for every dummy, so the freed indices can take the
            # pool names a, b, i, j without colliding.
            fresh = {i: Index(f"_{i.space[0]}{k}", i.space, True)
                     for k, i in enumerate(term.summed_indices)}
            counts = {"occ": 0, "vir": 0}
            freed = {}
            for idx in fac.indices:
                pool = "ijklmno" if idx.space == "occ" else "abcdefgh"
                freed[fresh[idx]] = Index(pool[counts[idx.space]], idx.space, False)
                counts[idx.space] += 1
            rename = {old: freed.get(new, new) for old, new in fresh.items()}
            rest = [f.reindexed(rename) for k, f in enumerate(term.factors) if k != pos]
            free = tuple(rename[i] for i in fac.indices)
            summed = tuple(rename[i] for i in term.summed_indices if rename[i] not in free)
            raw.append(_as_dummy(term, rest, free, summed, term.coeff))
    buckets: dict = {}
    order: list = []
    for t in raw:
        merge_term_into_buckets(canonicalize_term_to_fixed_point(t), buckets, order)
    return [buckets[s] for s in order if buckets[s].coeff != 0]


def lambda_equations(method: str, engine: str = "diagram",
                     canonical_fock: bool = True) -> dict[str, list[AlgebraTerm]]:
    """Spin-orbital Λ equations for *method*, one G list per manifold name
    (``singles``, ``doubles``, ...). The Λ residual is A(sum of the terms)."""
    from .generate import generate_cc_equations, manifold_name, parse_cc_level

    equations = generate_cc_equations(method, engine=engine, canonical_fock=canonical_fock)
    lag = lagrangian_terms(equations)
    return {manifold_name(r): differentiate(lag, r) for r in parse_cc_level(method)}
