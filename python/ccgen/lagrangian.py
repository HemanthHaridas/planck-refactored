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
from .project import AlgebraTerm
from .tensors import Tensor


SPINS = ("gcc", "rcc", "ucc")


def _lambda_tensor(free: tuple[Index, ...], name: str, spin: str) -> Tensor:
    occ = [i for i in free if i.space == "occ"]
    vir = [i for i in free if i.space == "vir"]
    n = len(occ)
    assert n == len(vir), f"residual free indices are not an excitation: {free}"
    # Only the GCC λ carries its antisymmetry in the tensor; RCC (abab) and
    # UCC blocks are plain arrays, like their t factors, and an antisymmetric
    # same-spin UCC λ is imposed on the solution instead.
    groups = (tuple(range(n)), tuple(range(n, 2 * n))) if spin == "gcc" and n > 1 else ()
    return Tensor(name, tuple(i.as_dummy() for i in occ + vir), groups)


def _weight(manifold: str, n: int, spin: str) -> Fraction:
    """1/(n!)², except UCC: 1/(n_α! n_β!)², the number of spin-orbital tuples
    one stored block element stands for divided into 1/(n!)². With it the UCC
    Lagrangian is the GCC one restricted to stored blocks, so UCC λ is the
    spin-orbital λ (PySCF's UCCSD λ). For RCC any weight is valid; it only
    rescales λ."""
    if spin == "ucc":
        occ_tag = manifold.split("_", 1)[1][:n]
        na = occ_tag.count("a")
        return Fraction(1, (factorial(na) * factorial(n - na)) ** 2)
    return Fraction(1, factorial(n) ** 2)


def _lambda_name(manifold: str, n: int) -> str:
    """``l<n>``, plus the sector tag of a rank >= 4 residual key
    (``quadruples_aaabaaab`` -> ``l4_aaabaaab``), mirroring ``t4_aaabaaab``."""
    tag = manifold.split("_", 1)[1] if "_" in manifold else ""
    return f"l{n}_{tag}" if tag else f"l{n}"


def _as_dummy(term: AlgebraTerm, factors, free, summed, coeff) -> AlgebraTerm:
    return AlgebraTerm(Fraction(coeff), tuple(factors), tuple(free), tuple(summed),
                       True, term.provenance)


def lagrangian_terms(equations: dict[str, list[AlgebraTerm]],
                     spin: str = "gcc") -> list[AlgebraTerm]:
    """Scalar terms of L: the energy terms, and λ contracted onto each
    residual term's free indices (weight: :func:`_weight`).

    ``spin`` says what *equations* are: ``gcc`` (spin-orbital), ``rcc``
    (:func:`spin_adapt_equations`) or ``ucc`` (:func:`ucc_adapt_equations`).
    Every residual key gets its own λ, named after it: ``doubles`` -> ``l2``,
    ``quadruples_aaabaaab`` -> ``l4_aaabaaab``, ``doubles_abab`` -> ``l2_abab``."""
    assert spin in SPINS, spin
    terms = list(equations.get("energy", []))
    for manifold, residual in equations.items():
        if manifold == "energy":
            continue
        for t in residual:
            free = t.free_indices
            n = len(free) // 2
            dmap = {i: i.as_dummy() for i in free}
            lam = _lambda_tensor(free, _lambda_name(manifold, n), spin)
            factors = [f.reindexed(dmap) for f in t.factors] + [lam]
            terms.append(_as_dummy(t, factors, (),
                                   tuple(dmap[i] for i in free) + t.summed_indices,
                                   t.coeff * _weight(manifold, n, spin)))
    return terms


def differentiate(terms: list[AlgebraTerm], name: str | int) -> list[AlgebraTerm]:
    """G = ∂L/∂(amplitude *name*) over every element, canonicalized and merged.
    An int n means ``t<n>``; a sector amplitude is named (``t4_aaabaaab``).

    Free indices are named a, b, ... / i, j, ... in t's slot order, so the
    result evaluates in t's (vir..., occ...) layout."""
    if isinstance(name, int):
        name = f"t{name}"
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


def lambda_equations(method: str, engine: str = "diagram", canonical_fock: bool = True,
                     spin: str = "gcc") -> dict[str, list[AlgebraTerm]]:
    """Λ equations for *method*, one G list per residual key, the derivative
    with respect to the amplitude that key solves for (``doubles`` -> t2,
    ``quadruples_aaabaaab`` -> t4_aaabaaab, ``doubles_abab`` -> t2_abab).

    The Λ residual is the sum of the terms, symmetrized over the amplitude's
    own symmetry: antisymmetrized for GCC and same-spin UCC blocks, the pair
    symmetry for RCC. ``rcc``: λ is the multiplier of the spatial
    parametrization (not PySCF's RCCSD λ). ``gcc`` and ``ucc``: λ is PySCF's
    (GCCSD, UCCSD) λ."""
    from .generate import generate_cc_equations

    equations = generate_cc_equations(method, engine=engine, canonical_fock=canonical_fock)
    if spin == "rcc":
        from .spin import spin_adapt_equations
        equations = spin_adapt_equations(equations)
    elif spin == "ucc":
        from .spin import ucc_adapt_equations
        equations = ucc_adapt_equations(equations)
    lag = lagrangian_terms(equations, spin)
    out = {}
    for key, residual in equations.items():
        if key == "energy":
            continue
        n = len(residual[0].free_indices) // 2
        tag = key.split("_", 1)[1] if "_" in key else ""
        out[key] = differentiate(lag, f"t{n}_{tag}" if tag else f"t{n}")
    return out
