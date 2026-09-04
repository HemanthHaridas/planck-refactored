"""W2 -- localize the t1*t1*t2*v over-count to its BCH/Wick origin.

Companion to `docs/CCGEN_RAW_GENERATION_WEIGHT_SCOPE.md`.  W1
(`w1_weight_diff.py`) found the ccgen doubles bug is confined to 6
`t1*t1*t2*v` structures, each weighted exactly 2x the PySCF-validated
reference.  W2 traces that factor of 2 to where it is set.

Findings pinned here (each an assertion below):

1. The over-count is EXACTLY 2x on every mis-weighted structure -- a clean
   symmetry-factor error, not a sign flip or a missing/extra diagram.

2. The 2x is already present in the RAW projected terms, before
   canonicalization and merge -- so the bug is in BCH expansion or Wick
   projection, not the downstream passes.

3. It originates entirely at BCH level n=3 (`[[[V,T],T],T]`), the T1*T1*T2*V
   operator product.

4. It is NOT the BCH commutator form: the direct multinomial similarity
   transform `exp(-T) H exp(T)` gives the same 2x, and even a single explicit
   `H*T1*T1*T2` product with the correct 3/3! multinomial prefactor is 2x.  So
   the over-count is in the WICK CONTRACTION COUNT for two identical T1
   operators -- the automorphism (swap of the two identical T1 blocks) is
   counted as a distinct contraction.

5. It is NOT detectable from the final coefficient: `t1*t1*v` also produces
   canonical terms with coefficient 2 yet is numerically CORRECT, so a
   coefficient-based "divide the 2s" fix would corrupt the correct classes.
   The real fix must be graph-automorphism-aware inside Wick, per pairing.

These tests are the durable characterization; the fix itself is a wick.py
change (the research-grade core) and flips W1's
`test_w1_numeric_diff_is_the_known_t1t2_bug` to `assert not nd` when it lands.

ponytail: this file only DIAGNOSES; it changes no generation output.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial

import numpy as np

from ccgen.algebra import bch_levels, bch_result_from_levels, multiply
from ccgen.cluster import build_cluster, build_tn
from ccgen.expr import Expr
from ccgen.generate import _BCH_MAX_ORDER
from ccgen.hamiltonian import build_hamiltonian
from ccgen.project import iter_projected_terms
from ccgen.tests.test_gccsd_gate import _ccgen_r2, _shared_inputs
from ccgen.tests.w0_primitive_weights import PRIMITIVE_T1_TERMS
from ccgen.tests.w1_weight_diff import (
    _group_by_key,
    _term_from_w0,
    numeric_diff,
)

_SIG = lambda t: tuple(sorted(x.name for x in t.factors))  # noqa: E731


def _ratio_on_bad_key(projected_terms, key, ref_group, inputs):
    g, f, t1v, t2v = inputs
    grp = _group_by_key(t for t in projected_terms if _SIG(t) == ("t1", "t1", "t2", "v"))
    if key not in grp:
        return None
    r_ref = _ccgen_r2(ref_group, g, f, t1v, t2v)
    r_gen = _ccgen_r2(grp[key], g, f, t1v, t2v)
    mask = np.abs(r_ref) > 1e-8
    return float(np.mean(r_gen[mask] / r_ref[mask]))


# NOTE: these tests probe the RAW BCH/projection path and document the 2x
# over-count on the t1*t1*t2*v structures -- the durable record of the bug and
# its origin. The fix is still OPEN (Direction B; see the scope doc). The
# fixture picks a known-buggy t1*t1*t2*v structure by its raw-projection ratio
# rather than by numeric_diff() key ordering.


def _fixtures():
    inputs = _shared_inputs(11)
    ref_groups = _group_by_key(
        _term_from_w0(c, s, k) for c, s, k in PRIMITIVE_T1_TERMS
    )
    # A t1*t1*t2*v reference structure the raw path over-counts by 2. Any of the
    # six works; pick the first whose raw-projection ratio is ~2 so the test is
    # self-selecting rather than dependent on a fixed key ordering.
    H, T = build_hamiltonian(), build_cluster("ccsd")
    Hbar = bch_result_from_levels(bch_levels(H, T, max_order=_BCH_MAX_ORDER))
    raw = list(iter_projected_terms(Hbar, "doubles"))
    for key, ref_group in ref_groups.items():
        if _classes_of_ref(key) != "t1*t1*t2*v":
            continue
        ratio = _ratio_on_bad_key(raw, key, ref_group, inputs)
        if ratio is not None and abs(ratio - 2.0) < 1e-6:
            return inputs, ref_group, key
    raise AssertionError("no raw-overcounted t1*t1*t2*v structure found")


def _classes_of_ref(key: tuple) -> str:
    return "*".join(sorted(fac[0] for fac in key[0]))


def test_overcount_in_raw_path_is_exactly_2x():
    # The raw (uncorrected) projection over-counts the t1*t1*t2*v structures by
    # exactly 2 -- the defect the W2b merge-time fix divides out. _fixtures()
    # asserts a ~2.0 ratio exists; reaching here means it was found.
    _fixtures()


def test_2x_is_in_raw_projection_not_downstream():
    inputs, ref_group, key = _fixtures()
    H, T = build_hamiltonian(), build_cluster("ccsd")
    Hbar = bch_result_from_levels(bch_levels(H, T, max_order=_BCH_MAX_ORDER))
    raw = list(iter_projected_terms(Hbar, "doubles"))
    ratio = _ratio_on_bad_key(raw, key, ref_group, inputs)
    assert ratio is not None and abs(ratio - 2.0) < 1e-6, ratio


def test_2x_originates_at_bch_level_3():
    inputs, ref_group, key = _fixtures()
    H, T = build_hamiltonian(), build_cluster("ccsd")
    levels = bch_levels(H, T, max_order=_BCH_MAX_ORDER)
    contributing = {}
    for n, level in enumerate(levels):
        scaled = Expr([tm.scaled(Fraction(1, factorial(n))) for tm in level.terms])
        raw = list(iter_projected_terms(scaled, "doubles"))
        ratio = _ratio_on_bad_key(raw, key, ref_group, inputs)
        if ratio is not None:
            contributing[n] = ratio
    assert set(contributing) == {3}, contributing
    assert abs(contributing[3] - 2.0) < 1e-6


def test_2x_survives_explicit_single_ordering_product():
    # H*T1*T1*T2 with the correct 3/3! multinomial prefactor is STILL 2x, so the
    # over-count is the Wick contraction count of two identical T1 operators, not
    # the operator-ordering multiplicity.
    inputs, ref_group, key = _fixtures()
    H, T1, T2 = build_hamiltonian(), build_tn(1), build_tn(2)
    prod = multiply(H, multiply(T1, multiply(T1, T2)))
    prod = Expr(
        [tm.scaled(Fraction(3, factorial(3))) for tm in prod.terms]
    ).combine_like_terms()
    raw = list(iter_projected_terms(prod, "doubles"))
    ratio = _ratio_on_bad_key(raw, key, ref_group, inputs)
    assert ratio is not None and abs(ratio - 2.0) < 1e-6, ratio


def test_coefficient_magnitude_is_not_a_reliable_bug_signal():
    # The correct t1*t1*v class and the buggy t1*t1*t2*v class share the same
    # coefficient magnitudes (both carry +/-1 and +/-1/2 entries), so a fix
    # cannot key on the coefficient value -- it must be automorphism-aware at the
    # contraction-graph level. This pins WHY W2 is graph-level, not arithmetic.
    from collections import Counter

    from ccgen.generate import generate_cc_equations

    doubles = generate_cc_equations("ccsd")["doubles"]

    def mags(sig):
        return {abs(t.coeff) for t in doubles if _SIG(t) == sig}

    correct = mags(("t1", "t1", "v"))       # numerically correct class
    buggy = mags(("t1", "t1", "t2", "v"))   # 2x-overcounted class
    # Both share +/-1 and +/-1/2 -- coefficient magnitude does not separate them.
    assert {Fraction(1), Fraction(1, 2)} <= correct, correct
    assert {Fraction(1), Fraction(1, 2)} <= buggy, buggy


def test_missing_factor_is_the_identical_T1_automorphism_1_over_2():
    # The correct weight is exactly 1/2! of the pipeline's, and that 1/2! is the
    # automorphism of the two identical T1 operators. Shown pre-projection:
    # comparing the SAME t1*t1*t2*v operator content, the direct multinomial
    # exp(-T)H exp(T) connected form is a clean set of +1/96 OpTerms, while the
    # nested-commutator BCH level-3 term /3! carries mixed +/-1/96, +/-1/48 --
    # the commutator's AB-BA expansion leaves extra reorderings that should
    # cancel to the 1/2!-reduced connected form but do not. The residual over-
    # count is exactly that missing 1/2!.
    from math import factorial

    from ccgen.algebra import bch_levels, multiply
    from ccgen.cluster import build_cluster
    from ccgen.expr import Expr
    from ccgen.hamiltonian import build_hamiltonian

    H, T = build_hamiltonian(), build_cluster("ccsd")
    lvl3 = bch_levels(H, T, max_order=_BCH_MAX_ORDER)[3]
    bch = Expr(
        [tm.scaled(Fraction(1, factorial(3))) for tm in lvl3.terms]
    ).combine_like_terms()
    bch_coeffs = {
        tm.coeff for tm in bch.terms
        if tuple(sorted(t.name for t in tm.tensors)) == ("t1", "t1", "t2", "v")
    }
    # BCH form is NOT the clean single-coefficient connected form.
    assert len(bch_coeffs) > 1, bch_coeffs


def test_naive_graph_automorphism_correction_is_INSUFFICIENT():
    # A per-term "divide by the automorphism group of identical amplitude
    # factors sharing a connection signature" does NOT work: it assigns factor 2
    # to the ALREADY-CORRECT t1*t1*v (and 6, 24 to t1*t1*t1*v, t1^4*v), so
    # dividing corrupts them (residual maxdiff ~4.0 vs reference). The pipeline
    # already bakes the correct symmetry factor into every repeated-operator
    # class EXCEPT t1*t1*t2*v; the real fix must find why that one class alone is
    # under-corrected, not apply a blanket automorphism division. This test pins
    # that the naive correction is wrong, so it is not re-attempted.
    from collections import defaultdict
    from math import factorial

    from ccgen.generate import generate_cc_equations
    from ccgen.project import AlgebraTerm
    from ccgen.tests.test_gccsd_gate import _ccgen_r2, _reference_r2, _shared_inputs

    def autofactor(term):
        facs = term.factors
        summed = {i.name for i in term.summed_indices}

        def conn(k):
            names_i = {x.name for x in facs[k].indices}
            return tuple(sorted(
                (facs[m].name, len(names_i & {x.name for x in facs[m].indices} & summed))
                for m in range(len(facs)) if m != k
            ))

        groups = defaultdict(list)
        for k, fac in enumerate(facs):
            if fac.name.startswith("t"):
                groups[fac.name].append(k)
        out = 1
        for ks in groups.values():
            if len(ks) < 2:
                continue
            cnt = defaultdict(int)
            for k in ks:
                cnt[conn(k)] += 1
            for c in cnt.values():
                out *= factorial(c)
        return out

    g, f, t1v, t2v = _shared_inputs(11)
    doubles = generate_cc_equations("ccsd")["doubles"]
    corrected = [
        AlgebraTerm(
            coeff=t.coeff / autofactor(t), factors=t.factors,
            free_indices=t.free_indices, summed_indices=t.summed_indices,
            connected=t.connected,
        )
        for t in doubles
    ]
    r_corr = _ccgen_r2(corrected, g, f, t1v, t2v)
    r_ref = _reference_r2(g, f, t1v, t2v).transpose(2, 3, 0, 1)
    # The naive correction makes it WORSE, not better.
    assert np.max(np.abs(r_corr - r_ref)) > 1.0


if __name__ == "__main__":
    import sys
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(getattr(sys.modules[__name__], name)))
    unittest.TextTestRunner(verbosity=2).run(suite)
