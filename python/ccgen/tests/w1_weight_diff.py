"""W1 -- diff ccgen's per-structure t1-weights against the W0 reference table.

Companion to `docs/CCGEN_RAW_GENERATION_WEIGHT_SCOPE.md`.  W0
(`w0_primitive_weights.py`) is the ground-truth `dict[structure -> coeff]` from
the PySCF-validated reference.  W1 builds the SAME dict from ccgen's generated
doubles residual and diffs.  The diff is nonzero exactly on the mis-weighted
structures -- the precise list every prior attempt lacked.

Both sides are reduced to a shared canonical key with
`dressing._eri_canonical`, which folds the ERI's 8-fold exchange symmetry and
free-index listing order (the convention gaps between the reference's einsum
form and ccgen's raw AlgebraTerms).  A structure = that canonical key; the
value = summed coefficient.

Only t1-containing structures are compared -- the t1-free residual (CCD +
bare-ERI + f*t2) is already exact (see the doc).  ccgen's `f*t1*t2` terms are
Fock-driven and vanish under canonical Fock; the reference table is written for
that canonical case, so they are excluded here.

ponytail: reuses _eri_canonical instead of a bespoke key; the only new code is
the W0-einsum -> AlgebraTerm converter, which the shared canonical key then
grades.
"""

from __future__ import annotations

from fractions import Fraction

from ccgen.generate import generate_cc_equations
from ccgen.indices import Index, make_occ, make_vir
from ccgen.project import AlgebraTerm
from ccgen.tensors import Tensor, t1, t2, v
from ccgen.optimization.dressing import _eri_canonical
from ccgen.tests.w0_primitive_weights import PRIMITIVE_T1_TERMS

# einsum index letter -> orbital space (matches gccsd_reference's convention).
_OCC = set("ijmn")
_VIR = set("abef")


def _index(letter: str) -> Index:
    return make_occ(letter) if letter in _OCC else make_vir(letter)


def _factor_from_operand(key: str, letters: str) -> Tensor:
    """Map one W0 einsum operand to a ccgen Tensor.

    W0 layouts:  t1 -> [occ, vir];  t2 -> [occ, occ, vir, vir];
    ERI block (e.g. 'oovv') -> v(p,q,r,s) over those spaces directly.
    ccgen layouts: t1(a,i)=[vir,occ]; t2(a,b,i,j)=[vir,vir,occ,occ]; v same order.
    """
    idx = [_index(c) for c in letters]
    if key == "t1":  # [o, v] -> t1(v, o)
        return t1(idx[1], idx[0])
    if key == "t2":  # [o, o, v, v] -> t2(v, v, o, o)
        return t2(idx[2], idx[3], idx[0], idx[1])
    # ERI block: letters already name the four slots in <pq||rs> order.
    return v(idx[0], idx[1], idx[2], idx[3])


def _term_from_w0(coeff: Fraction, subs: str, keys: tuple[str, ...]) -> AlgebraTerm:
    ins, out = subs.split("->")
    operand_letters = ins.split(",")
    factors = tuple(
        _factor_from_operand(k, lets) for k, lets in zip(keys, operand_letters)
    )
    all_letters = set("".join(operand_letters))
    free_letters = set(out)
    summed_letters = sorted(all_letters - free_letters)
    free = tuple(_index(c) for c in out)
    summed = tuple(_index(c) for c in summed_letters)
    return AlgebraTerm(
        coeff=coeff, factors=factors, free_indices=free,
        summed_indices=summed, connected=True,
    )


def _multiset(terms) -> dict[tuple, Fraction]:
    acc: dict[tuple, Fraction] = {}
    for term in terms:
        key, c = _eri_canonical(term)
        acc[key] = acc.get(key, Fraction(0)) + c
    return {k: val for k, val in acc.items() if val != 0}


def w0_multiset() -> dict[tuple, Fraction]:
    return _multiset(_term_from_w0(c, s, k) for c, s, k in PRIMITIVE_T1_TERMS)


# ccgen t1-structure classes to compare (t1-free residual is already exact;
# f*t1*t2 is Fock-driven and excluded -- see module docstring).
_T1_CLASSES = {
    frozenset(("t1", "v")),
    frozenset(("t1", "t1", "v")),
    frozenset(("t1", "t1", "t1", "v")),
    frozenset(("t1", "t1", "t1", "t1", "v")),
    frozenset(("t1", "t2", "v")),
    frozenset(("t1", "t1", "t2", "v")),
}


def ccgen_multiset() -> dict[tuple, Fraction]:
    doubles = generate_cc_equations("ccsd")["doubles"]
    t1_terms = [
        t for t in doubles
        if frozenset(x.name for x in t.factors) in _T1_CLASSES
    ]
    return _multiset(t1_terms)


# NOTE on coefficient diffing: comparing the multisets' _eri_canonical
# coefficients directly is NOT reliable across the reference-einsum vs raw-ccgen
# conventions -- the exchange fixed-point flags the CORRECT t1*v singles with a
# spurious +/-2 sign diff. The multisets are used only to cross-check STRUCTURE
# grouping (0 missing / 0 spurious); the authoritative per-structure verdict is
# numeric, below.

# --- numeric per-structure comparison (the authoritative W1 verdict) ---------
#
# Group both sides by canonical key, evaluate each group's residual tensor on
# shared random inputs, and diff numerically. Immune to _eri_canonical's sign
# bookkeeping: a wrong sign only registers if it is a REAL contribution error.

def _group_by_key(terms) -> dict[tuple, list]:
    groups: dict[tuple, list] = {}
    for term in terms:
        key, _ = _eri_canonical(term)
        groups.setdefault(key, []).append(term)
    return groups


def numeric_diff(seed: int = 11) -> dict[tuple, float]:
    """Per-structure ||ccgen_group - ref_group|| on shared inputs.

    A structure is mis-weighted iff its ccgen contribution differs numerically
    from the reference contribution for the SAME canonical structure. Returns
    {canonical_key -> max|diff|} over structures where it exceeds tolerance.
    """
    from ccgen.tests.test_gccsd_gate import _shared_inputs, _ccgen_r2

    g, f, t1v, t2v = _shared_inputs(seed)
    ref_groups = _group_by_key(
        _term_from_w0(c, s, k) for c, s, k in PRIMITIVE_T1_TERMS
    )
    doubles = generate_cc_equations("ccsd")["doubles"]
    gen_groups = _group_by_key(
        t for t in doubles
        if frozenset(x.name for x in t.factors) in _T1_CLASSES
    )
    out: dict[tuple, float] = {}
    import numpy as np

    for key in set(ref_groups) | set(gen_groups):
        r_ref = (_ccgen_r2(ref_groups[key], g, f, t1v, t2v)
                 if key in ref_groups else 0.0)
        r_gen = (_ccgen_r2(gen_groups[key], g, f, t1v, t2v)
                 if key in gen_groups else 0.0)
        d = float(np.max(np.abs(r_gen - r_ref)))
        if d > 1e-9:
            out[key] = d
    return out


def _fmt_key(key: tuple) -> str:
    # Canonical key is a nested tuple; render the factor names + index spaces.
    return str(key)


def _classes_of(key: tuple) -> str:
    factor_names = sorted(fac[0] for fac in key[0])
    return "*".join(factor_names)


def report() -> None:
    ref = w0_multiset()
    gen = ccgen_multiset()
    print("=== structure grouping cross-check (coeff key) ===")
    print(f"reference structures: {len(ref)}")
    print(f"ccgen structures:     {len(gen)}")
    print(f"  in ref only:  {len(set(ref) - set(gen))}")
    print(f"  in ccgen only:{len(set(gen) - set(ref))}\n")

    nd = numeric_diff()
    print("=== numeric per-structure verdict (authoritative) ===")
    print(f"mis-weighted structures: {len(nd)}\n")
    from collections import Counter
    by_class = Counter(_classes_of(k) for k in nd)
    for cls, n in sorted(by_class.items()):
        print(f"  {n:2d}  {cls}")
    print()
    for key, dv in sorted(nd.items(), key=lambda kv: -kv[1]):
        print(f"[{_classes_of(key)}] max|diff|={dv:.4f}")
        print(f"       {_fmt_key(key)}")


def test_w1_grouping_is_sound():
    # The canonical key groups reference and ccgen structures onto the SAME set
    # (0 missing / 0 spurious) -- so any numeric diff is a real weight error,
    # not a structure that only one side has. This is the doc's stop-condition
    # check: a clean small set, not a spread across unrelated structures.
    ref, gen = w0_multiset(), ccgen_multiset()
    assert set(ref) == set(gen), (
        f"structure sets differ: {len(set(ref) - set(gen))} ref-only, "
        f"{len(set(gen) - set(ref))} ccgen-only"
    )


def test_w1_numeric_diff_is_the_known_t1t2_bug():
    # The authoritative verdict: the mis-weighted structures are exactly the
    # t1*t1*t2*v ladder cross-terms -- the singles (t1*v etc.) are numerically
    # correct and must NOT appear. The W2b fix (still open -- see the doc) makes
    # this diff empty; flip to `assert not nd` then.
    nd = numeric_diff()
    assert nd, "expected nonzero numeric t1-weight diff (the known ccgen bug)"
    classes = {_classes_of(k) for k in nd}
    assert classes == {"t1*t1*t2*v"}, (
        f"unexpected mis-weighted classes (converter/convention bug?): {classes}"
    )
    assert len(nd) == 6, f"expected 6 mis-weighted structures, got {len(nd)}"


if __name__ == "__main__":
    report()
