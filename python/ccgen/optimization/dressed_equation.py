"""Dressed-equation transcription + verification (Option B).

The automatic dressing-discovery path (A1-A3) proved the CCSD residual's dressed
pieces are inseparable: the raw R2 residual is not a coverable sum of
independent operator*rest instances (see vault Open Work -> ccgen dressed
intermediates, "exact-cover disproven").  So dressing is done by CURATED
TEMPLATES: transcribe the whole dressed R1/R2/... equation and verify the
assembled dressed equation equals the raw generated residual, exactly.

This module is the RANK-AGNOSTIC machinery.  The per-method dressed equations
(the human content) live alongside it.  Extensibility: the verifier works on
any transcribed equation of any rank; only the transcription is method-specific
(CCSD is textbook Stanton-Gauss; CCSDT/CCSDTQ add their own W-intermediates and
equation terms, which no closed form covers for arbitrary rank).

A "dressed term" is an ordinary AlgebraTerm whose factors may include named
operators (Wmnij, Wabef, Fae, ...) and pseudo-amplitudes (tau, tau_tilde)
alongside primitive t1/t2/v/f.  Verification expands every operator via its
definition and every pseudo-amplitude via t2 + written*t1t1, canonicalizes with
the ERI-exchange-aware key, and compares the result to the raw residual as a
coefficient-keyed multiset.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from ..indices import Index, make_occ, make_vir
from ..project import AlgebraTerm
from ..tensors import Tensor, reindex_tensors
from ..tensors import t1 as _t1, t2 as _t2, v as _v, f as _f
from .tau import TAU_NAME, TAU_CONTRACTED_NAME, tau
from .dressing import (
    TAU_TILDE_NAME,
    DressedOperator,
    _eri_canonical,
    _expand_pseudo_amplitude_in_term,
    seeded_operators,
)


def _operator_table() -> dict[str, DressedOperator]:
    return {op.name: op for op in seeded_operators()}


def _expand_one_operator_factor(
    term: AlgebraTerm,
    operators: dict[str, DressedOperator],
) -> list[AlgebraTerm] | None:
    """Replace the first operator factor in ``term`` with its definition.

    Returns the list of expanded terms (one per definition term of the
    operator), with the operator's block indices renamed to the factor's actual
    indices and internal dummies made unique.  Returns None if the term carries
    no operator factor.
    """
    pos = next(
        (k for k, f in enumerate(term.factors) if f.name in operators), None
    )
    if pos is None:
        return None

    ofac = term.factors[pos]
    op = operators[ofac.name]
    others = tuple(f for k, f in enumerate(term.factors) if k != pos)

    # Map operator block indices -> the factor's actual indices.
    if len(op.block) != len(ofac.indices):
        raise ValueError(
            f"operator {op.name} block arity {len(op.block)} != factor arity "
            f"{len(ofac.indices)}"
        )
    block_map = {b: a for b, a in zip(op.block, ofac.indices)}

    out: list[AlgebraTerm] = []
    for dt in op.definition_terms:
        ren: dict[Index, Index] = {}
        for idx in dt.free_indices:  # block
            ren[idx] = block_map[idx]
        for idx in dt.summed_indices:  # internal dummy -> unique
            ren[idx] = Index(f"__{op.name}_{idx.name}", idx.space, True)
        new_amp = reindex_tensors(dt.factors, ren)
        internal = tuple(ren[i] for i in dt.summed_indices)
        out.append(AlgebraTerm(
            coeff=term.coeff * dt.coeff,
            factors=tuple(new_amp) + others,
            free_indices=term.free_indices,
            summed_indices=tuple(sorted(
                set(term.summed_indices) | set(internal),
                key=lambda x: (x.space, x.name))),
            connected=term.connected,
        ))
    return out


def expand_dressed_term(
    term: AlgebraTerm,
    operators: dict[str, DressedOperator] | None = None,
) -> list[AlgebraTerm]:
    """Fully expand all operator and pseudo-amplitude factors in one term.

    Iterates to a fixed point: expand operators to their definitions, then
    expand tau/tau_tilde to t2 + written*t1t1, until only primitive factors
    (t1/t2/v/f) remain.
    """
    operators = operators or _operator_table()
    frontier = [term]
    changed = True
    while changed:
        changed = False
        nxt: list[AlgebraTerm] = []
        for t in frontier:
            has_op = any(f.name in operators for f in t.factors)
            if has_op:
                expanded = _expand_one_operator_factor(t, operators)
                nxt.extend(expanded)
                changed = True
                continue
            has_pseudo = any(
                f.name in (TAU_NAME, TAU_TILDE_NAME, TAU_CONTRACTED_NAME)
                for f in t.factors
            )
            if has_pseudo:
                pieces = _expand_pseudo_amplitude_in_term(t)
                nxt.extend(pieces)
                changed = changed or (len(pieces) != 1)
                continue
            nxt.append(t)
        frontier = nxt
    return frontier


def expand_then_adapt(equations, adapter=None, operators=None):
    """Expand a dressed manifold to primitives in GCC, THEN spin-adapt it (V1.1e.1).

    This is the pinned order for validating a dressed spatial equation, and it is the
    one Decision 5 implies (``GCC -> dress -> adapt``). The rejected alternative --
    adapting the operator definitions and the residual separately, then expanding the
    adapted dressed manifold against an adapted operator table -- is measurably worse
    on dressed CCSD (mismatches vs the adapted raw residual):

        configuration                    energy  singles  doubles
        adapt-then-verify  (REJECTED)         0       13       61
        expand-then-adapt  (THIS)             0        0       14

    Two reasons beyond the raw counts. (1) Expansion introduces operator-internal
    dummy indices (``__Wmnij_e``, ``__Wabef_m``); doing it in GCC keeps those out of
    the adapter, which keys spin blocks on slot structure. (2) An adapted operator
    table means the SAME operator is adapted once per definition and again per usage
    site, so any orientation sensitivity is applied twice, inconsistently.

    ``adapter`` defaults to :func:`ccgen.spin.spin_adapt_equations`; pass
    ``ucc_adapt_equations`` for UCC. Returns the adapted primitive manifold, directly
    comparable to ``adapter(raw)`` via :func:`raw_multiset`.

    NOTE: the residual doubles=14 is a real open defect, root-caused to `v` bra<->ket
    orientation sensitivity in the adapter (V1.1e.2), NOT to this ordering. Pinning the
    order here is what makes that residue a single reproducible number."""
    from ..spin import spin_adapt_equations

    fn = adapter or spin_adapt_equations
    expanded = {
        manifold: [p for t in terms for p in expand_dressed_term(t, operators)]
        for manifold, terms in equations.items()
    }
    return fn(expanded)


def verify_adapted_dressed_equation(dressed, raw, adapter=None, operators=None):
    """Does a dressed manifold, expanded-then-adapted, equal the adapted raw residual?

    Returns ``{manifold: {key: delta}}`` holding only the manifolds that mismatch, so
    an empty dict means exact. The per-manifold split is what lets a failure name
    ``doubles`` instead of "the equation" (V1.1e.1); per-OPERATOR localization is
    V1.1e.3.

    Both sides go through the same adapter, so this compares adapted-to-adapted -- the
    dressed side is never credited with a symmetry fold the adapted output does not
    actually carry."""
    from ..spin import spin_adapt_equations

    fn = adapter or spin_adapt_equations
    got = expand_then_adapt(dressed, adapter=fn, operators=operators)
    want = fn(raw)

    out: dict = {}
    for manifold in set(got) | set(want):
        a = raw_multiset(got.get(manifold, []))
        b = raw_multiset(want.get(manifold, []))
        diff = {}
        for key in set(a) | set(b):
            d = a.get(key, Fraction(0)) - b.get(key, Fraction(0))
            if d:
                diff[key] = d
        if diff:
            out[manifold] = diff
    return out


def dressed_multiset(
    terms: Sequence[AlgebraTerm],
    operators: dict[str, DressedOperator] | None = None,
    spatial: bool = False,
) -> dict[tuple, Fraction]:
    """ERI-canonical multiset of a dressed equation, fully expanded.

    Expands every dressed term to primitives, then sums coefficients per
    ERI-canonical key (folding bra<->ket exchange).  This is the identity of the
    dressed equation, directly comparable to raw_multiset of the generated
    residual.
    """
    operators = operators or _operator_table()
    acc: dict[tuple, Fraction] = {}
    for term in terms:
        for prim in expand_dressed_term(term, operators):
            key, coeff = _eri_canonical(prim, spatial=spatial)
            acc[key] = acc.get(key, Fraction(0)) + coeff
    return {k: v for k, v in acc.items() if v != 0}


def raw_multiset(terms: Sequence[AlgebraTerm],
                 spatial: bool = False) -> dict[tuple, Fraction]:
    """ERI-canonical multiset of a raw generated residual.

    ``spatial=True`` folds only the four relations a non-antisymmetrized <pq|rs> has.
    Required for spin-adapted input; the 8-fold default equates spatial terms that are
    not equal.
    """
    acc: dict[tuple, Fraction] = {}
    for term in terms:
        key, coeff = _eri_canonical(term, spatial=spatial)
        acc[key] = acc.get(key, Fraction(0)) + coeff
    return {k: v for k, v in acc.items() if v != 0}


def verify_dressed_equation(
    dressed_terms: Sequence[AlgebraTerm],
    raw_terms: Sequence[AlgebraTerm],
    operators: dict[str, DressedOperator] | None = None,
    spatial: bool = False,
) -> tuple[bool, dict[tuple, Fraction]]:
    """Does a transcribed dressed equation equal the raw residual, exactly?

    Returns ``(ok, diff)`` where ``diff`` maps each mismatched canonical key to
    (dressed_coeff - raw_coeff); ``ok`` is True iff diff is empty.  The diff is
    the actionable output while transcribing -- it names exactly which primitive
    contributions are over- or under-counted.
    """
    dressed = dressed_multiset(dressed_terms, operators, spatial=spatial)
    raw = raw_multiset(raw_terms, spatial=spatial)
    diff: dict[tuple, Fraction] = {}
    for k in set(dressed) | set(raw):
        d = dressed.get(k, Fraction(0)) - raw.get(k, Fraction(0))
        if d != 0:
            diff[k] = d
    return (not diff), diff


# ---------------------------------------------------------------------------
# CCSD dressed equations (curated transcription -- the method-specific content)
# ---------------------------------------------------------------------------
#
# Standard spin-orbital Stanton-Gauss CCSD (JCP 94, 4334 (1991)):
#
#   R1_ai = f_ai + t1_ei Fae - t1_am Fmi + t1_em Wmaei/(Wmbej) ...  [R1 below]
#   R2_abij = <ab||ij>
#           + P(ab)[ t2_aeij ( Fbe - 1/2 t1_bm Fme ) ]
#           - P(ij)[ t2_abim ( Fmj + 1/2 t1_ej Fme ) ]
#           + 1/2 tau_abmn Wmnij + 1/2 tau_efij Wabef
#           + P(ij)P(ab)[ t2_aeim Wmbej - t1_ei t1_am <mb||ej> ]
#           + P(ij)[ t1_ei <ab||ej> ] - P(ab)[ t1_am <mb||ij> ]
#
# where P(pq) X = X_pq - X_qp.  The Fme-correction terms inside the Fae/Fmi
# brackets are NOT folded into the operators here (our Fae/Fmi are the pure
# blocks), so they are written explicitly.  Verified against the raw generated
# doubles by `verify_dressed_equation`; the diff drives transcription until 0.


def _op(name: str, *idx: Index) -> Tensor:
    return Tensor(name, tuple(idx))


def ccsd_dressed_r2() -> list[AlgebraTerm]:
    """Transcribed dressed CCSD R2 (externals a,b,i,j; open-shell spin-orbital).

    WORK IN PROGRESS -- does NOT yet diff to 0 against the raw generated
    doubles.  The framework (expand_dressed_term / verify_dressed_equation) is
    complete and correct; this transcription still needs its spin-orbital
    conventions reconciled with ccgen's generated residual.  Known convention
    gaps found so far: free-index listing order (fixed in _eri_canonical via
    _free_order_normalized), and the ccgen residual writes the bare ERI as
    <ij||ab> (v(i,j,a,b)) not <ab||ij>.  Remaining: sign conventions on the
    Fock/ERI terms and the P(pq) antisymmetrizer expansions -- drive with the
    `verify_dressed_equation(...)` diff until it reaches 0.

    Returns a list of dressed AlgebraTerms (operators/pseudo-amps + primitives).
    """
    a, b = make_vir("a"), make_vir("b")
    i, j = make_occ("i"), make_occ("j")
    m, n = make_occ("m"), make_occ("n")
    e, ff = make_vir("e"), make_vir("f")

    def T(coeff, facs, summed):
        return AlgebraTerm(
            coeff=Fraction(coeff) if not isinstance(coeff, Fraction) else coeff,
            factors=tuple(facs), free_indices=(a, b, i, j),
            summed_indices=tuple(summed), connected=True,
        )

    terms: list[AlgebraTerm] = []

    # <ab||ij>
    terms.append(T(1, [_v(a, b, i, j)], []))

    # P(ab) t2_aeij Fae  (pure Fae; the -1/2 t1 Fme correction handled below)
    terms.append(T(1, [_t2(a, e, i, j), _op("Fae", b, e)], [e]))
    terms.append(T(-1, [_t2(b, e, i, j), _op("Fae", a, e)], [e]))
    # -P(ab) 1/2 t2_aeij t1_bm Fme
    terms.append(T(Fraction(-1, 2), [_t2(a, e, i, j), _t1(b, m), _op("Fme", m, e)], [e, m]))
    terms.append(T(Fraction(1, 2), [_t2(b, e, i, j), _t1(a, m), _op("Fme", m, e)], [e, m]))

    # -P(ij) t2_abim Fmi
    terms.append(T(-1, [_t2(a, b, i, m), _op("Fmi", m, j)], [m]))
    terms.append(T(1, [_t2(a, b, j, m), _op("Fmi", m, i)], [m]))
    # -P(ij) 1/2 t2_abim t1_ej Fme
    terms.append(T(Fraction(-1, 2), [_t2(a, b, i, m), _t1(e, j), _op("Fme", m, e)], [m, e]))
    terms.append(T(Fraction(1, 2), [_t2(a, b, j, m), _t1(e, i), _op("Fme", m, e)], [m, e]))

    # 1/2 tau_abmn Wmnij + 1/2 tau_efij Wabef
    terms.append(T(Fraction(1, 2), [tau(a, b, m, n), _op("Wmnij", m, n, i, j)], [m, n]))
    terms.append(T(Fraction(1, 2), [tau(e, ff, i, j), _op("Wabef", a, b, e, ff)], [e, ff]))

    # P(ij)P(ab) [ t2_aeim Wmbej - t1_ei t1_am <mb||ej> ]
    for si, (ii, jj) in [(1, (i, j)), (-1, (j, i))]:
        for sa, (aa, bb) in [(1, (a, b)), (-1, (b, a))]:
            s = si * sa
            terms.append(T(s, [_t2(aa, e, ii, m), _op("Wmbej", m, bb, e, jj)], [e, m]))
            terms.append(T(-s, [_t1(e, ii), _t1(aa, m), _v(m, bb, e, jj)], [e, m]))

    # P(ij) t1_ei <ab||ej>
    terms.append(T(1, [_t1(e, i), _v(a, b, e, j)], [e]))
    terms.append(T(-1, [_t1(e, j), _v(a, b, e, i)], [e]))
    # -P(ab) t1_am <mb||ij>
    terms.append(T(-1, [_t1(a, m), _v(m, b, i, j)], [m]))
    terms.append(T(1, [_t1(b, m), _v(m, a, i, j)], [m]))

    return terms
