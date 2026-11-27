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
    return [t for block in differentiate_blocks(terms, name).values() for t in block]


def differentiate_blocks(terms: list[AlgebraTerm], name: str) -> dict[str, list[AlgebraTerm]]:
    """∂L/∂(factor *name*), grouped by the factor's space pattern (``oo``,
    ``ov``, ``oovv``, ...): one list per block, each canonicalized and merged.

    Free indices are in the factor's slot order. An amplitude has one block;
    f and v have several, and the densities are these derivatives
    (:func:`density_terms`)."""
    raw: dict[str, list[AlgebraTerm]] = {}
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
            block = "".join(i.space[0] for i in fac.indices)
            raw.setdefault(block, []).append(_as_dummy(term, rest, free, summed, term.coeff))
    out = {}
    for block, ts in raw.items():
        buckets: dict = {}
        order: list = []
        for t in ts:
            merge_term_into_buckets(canonicalize_term_to_fixed_point(t), buckets, order)
        kept = [buckets[s] for s in order if buckets[s].coeff != 0]
        if kept:
            out[block] = kept
    return out


def density_terms(lag: list[AlgebraTerm], spin: str = "gcc") -> dict[str, dict[str, list[AlgebraTerm]]]:
    """The correlation parts of the unrelaxed densities, as ∂L/∂(integral):
    ``{name: {block: terms}}`` for every f and v tensor of *spin*'s
    vocabulary (``f``, ``v``; UCC ``f_aa``, ``v_abab``, ...).

    At converged T and Λ, ∂L/∂f_pq is the correlation 1-RDM γ_qp (f enters L
    linearly), and ∂L/∂v the matching two-body piece. Generate L from the
    general-Fock equations (``canonical_fock=False``): the f_ov terms that
    canonical Fock drops are exactly what D_ov is made of. Assembling total
    RDMs (reference part, and f's own dependence on v) is the caller's."""
    names = sorted({f.name for t in lag for f in t.factors if f.name[0] in "fv"})
    return {n: differentiate_blocks(lag, n) for n in names}


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
    return lambda_residual_terms(equations, spin)


def lambda_residual_terms(equations: dict[str, list[AlgebraTerm]], spin: str = "gcc",
                          occ_first: bool = False) -> dict[str, list[AlgebraTerm]]:
    """G per residual key of the (already adapted) T *equations*; see
    :func:`lambda_equations`. ``occ_first`` reorders each term's free indices to
    (occ..., vir...), λ's own layout, which is what the C++ emitter writes the
    residual in (``l2(i,j,a,b)``); the default keeps t's (vir..., occ...)."""
    lag = lagrangian_terms(equations, spin)
    out = {}
    for key, residual in equations.items():
        if key == "energy":
            continue
        n = len(residual[0].free_indices) // 2
        tag = key.split("_", 1)[1] if "_" in key else ""
        terms = differentiate(lag, f"t{n}_{tag}" if tag else f"t{n}")
        if occ_first:
            terms = [AlgebraTerm(t.coeff, t.factors,
                                 tuple(sorted(t.free_indices, key=lambda i: i.space != "occ")),
                                 t.summed_indices, t.connected, t.provenance)
                     for t in terms]
        out[key] = terms
    return out


def lambda_diagonal_scale(terms: list[AlgebraTerm], group, ucc: bool = False) -> float:
    """c with R_Λ ≈ b − c·D·λ on the diagonal, D = Σ ε_occ − Σ ε_vir: the Λ
    Jacobi step is λ += R / (c D), the T step with its denominators scaled
    (docs/CC_LAMBDA_SCOPE.md, 3c).

    *terms* are one block's G (occ-first free indices, as
    :func:`lambda_residual_terms` gives them) and *group* its symmetry group;
    R = Σ_g s_g g(G). Only the f·λ terms carry the diagonal (canonical Fock,
    so f is block-diagonal). Evaluated with distinct orbital energies and a
    symmetrized unit λ at an element whose indices are all distinct, so only
    the identity maps it to itself and λ there is 1.
    """
    # Pure Python, sparse: this runs in the build-time generator, which needs
    # no numpy. λ is one symmetrized unit element and f is diagonal, so every
    # array here has at most |G| nonzeros.
    free = terms[0].free_indices
    names = [i.name for i in free]
    n = len(names) // 2
    no = n + 1
    eps = {"a": [(p + 1) * 0.37 - 3.0 for p in range(2 * no)],
           "b": [(p + 1) * 0.29 - 2.0 for p in range(2 * no)]}
    # This block's own λ: the one its f·λ terms read.
    lam_name = None
    for t in terms:
        if len(t.factors) == 2 and {f.name[0] for f in t.factors} == {"f", "l"}:
            lam = next(f for f in t.factors if f.name[0] == "l")
            if len(lam.indices) == 2 * n:      # this rank's λ, not an f_ov·λ_{n±1}
                lam_name = lam.name
                break
    if lam_name is None:
        raise ValueError("no f·λ term in this Λ block; cannot form its diagonal")
    spins = {}
    if ucc:
        from .spin import ucc_term_index_spins
        for t in terms:
            spins.update(ucc_term_index_spins(t))
    # The probe element: position j in each space, so all indices are distinct.
    k = {nm: j for half in (names[:n], names[n:]) for j, nm in enumerate(half)}

    # λ = Σ_g s_g g(unit): g(unit) is 1 where y[g(name)] = k[name].
    lam = {}
    for m, s in group:
        y = {m.get(nm, nm): k[nm] for nm in names}
        key = tuple(y[nm] for nm in names)
        lam[key] = lam.get(key, 0.0) + s

    def f_diag(f, pos):
        sp = f.name.split("_")[1][0] if "_" in f.name else "a"
        return eps[sp][pos if f.indices[0].space == "occ" else no + pos]

    out = {}
    for t in terms:
        # Only f·(this block's λ) reaches the diagonal; f_ov·λ of another rank
        # (general-Fock equations) does not.
        if (len(t.factors) != 2 or {f.name[0] for f in t.factors} != {"f", "l"}
                or lam_name not in {f.name for f in t.factors}):
            continue
        fock = next(f for f in t.factors if f.name[0] == "f")
        lfac = next(f for f in t.factors if f.name[0] == "l")
        if fock.indices[0].space != fock.indices[1].space:
            continue                            # f_ov: zero on a canonical reference
        for key, val in lam.items():
            bound = {}
            if any(bound.setdefault(i, v) != v for i, v in zip(lfac.indices, key)):
                continue                        # λ read with a repeated index
            a, b = fock.indices
            pos = bound.get(a, bound.get(b))    # f diagonal: both ends equal
            if any(bound.setdefault(i, pos) != pos for i in (a, b)):
                continue
            okey = tuple(bound[i] for i in t.free_indices)
            out[okey] = out.get(okey, 0.0) + float(t.coeff) * f_diag(fock, pos) * val
    # R = Σ_g s_g g(out) at the probe: g(out)[at] = out[at with names mapped].
    at = {nm: k[nm] for nm in names}
    R = sum(s * out.get(tuple(at[m.get(nm, nm)] for nm in names), 0.0) for m, s in group)

    def e(nm):
        sp = spins.get(nm, "a") if ucc else "a"
        return eps[sp][k[nm] if nm in "ijklmno" else no + k[nm]]
    D = sum(e(nm) for nm in names[:n]) - sum(e(nm) for nm in names[n:])
    # A ratio of exact rationals (the λ weight times |G|); drop the roundoff.
    return round(float(-R / D), 12)
