"""V1.1e.2 gate: `ucc_integrate_term_antisym` must be ORIENTATION-INVARIANT.

The adapter's output must depend on what the algebra IS, not on how it is WRITTEN.
Two `AlgebraTerm`s equal under the 8-fold ERI symmetry fold (`_eri_canonical`) must
spin-integrate to the same total coefficient per (block, spatial content).

LANDED (e.2.1): `_orientation_normalized` reorients every rank-4 `v` to one canonical
member of its 8-fold ERI orbit before the lines are read, folding that reorientation's
parity into the returned sign. Minimal reproducer, lifted verbatim from the dressed-CCSD
doubles manifold (the expansion side vs the raw side of one ERI-canonical key):

    expansion:      v(k,b,c,j) t2(a,c,i,k)   integrated to  2, now 0
    raw:         -1 v(j,c,k,b) t2(a,c,i,k)   integrated to  0, still 0

These are the same term: `v(j,c,k,b)` is `v(k,b,c,j)` bra<->ket-exchanged (2,3,0,1)
and then bra-swapped, and for antisymmetric `v` that swap costs -1 -- exactly the -1
the raw side carries. `_eri_canonical` confirms it: same key, same folded coefficient.

The mechanism is `_line_pairs`, which pairs slot k with slot k+n (the physicist
<pq|rs> convention). Read positionally the two writings present DIFFERENT line
structures -- `k-c, b-j` versus `j-k, c-b` -- so `_antisym_to_allowed` re-derives its
sign from written slot order and treats one integral as two independent inputs.

Per spin case the pre-fix divergence was precise, and this file pins the mechanism
rather than only pinning totals: the SAME 6 of 16 cases survived in both writings, into
the SAME blocks, but 4 of those 6 carried OPPOSITE SIGNS -- a writing-dependent,
inconsistent pattern. Post-fix the flip is UNIFORM across all 6 (the orbit parity), and
the per-TERM totals agree. A fix that made the totals agree by changing which cases
survive would be wrong and is still caught here.

SCOPE OF THE FIX. This makes the adapter orientation-invariant, which was necessary but
is NOT sufficient to close V1.1e: the dressed-vs-raw adapted doubles residual still
shows 14 mismatches, unchanged by this fix, with the repeated-same-name-factor signature
(t1t1v, t2t2v, t1t1t1t1v). That is a separate defect -- see e.2.5 in
docs/CCGEN_V11E2_ORIENTATION_INVARIANCE_SCOPE.md.

Also pinned: the bra<->ket exchange ALONE is harmless (a 256-case sweep finds no
divergence), because it maps lines p-r, q-s to r-p, s-q -- the same lines. The defect
requires exchange COMPOSED with a within-group swap. That is why the fix cannot be
"also try the exchange".
"""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.indices import make_occ, make_vir  # noqa: E402
from ccgen.optimization.dressing import _eri_canonical  # noqa: E402
from ccgen.project import AlgebraTerm  # noqa: E402
from ccgen.spin import (  # noqa: E402
    SpinIndex,
    _antisym_to_allowed,
    _closed_shell_representative_block,
    _line_pairs,
    _residual_template,
    ucc_integrate_target,
)
from ccgen.tensors import t2, v  # noqa: E402

A, B, C = make_vir("a"), make_vir("b"), make_vir("c")
I, J, K = make_occ("i"), make_occ("j"), make_occ("k")


def _doubles_term(coeff, v_factor):
    """The reproducer term: `coeff * v_factor * t2(a,c,i,k)`, externals (a,b,i,j)."""
    return AlgebraTerm(
        coeff=coeff,
        factors=(v_factor, t2(A, C, I, K)),
        free_indices=(A, B, I, J),
        summed_indices=(C, K),
        connected=True,
    )


# The two writings of one integral, as they appear in the real manifolds.
EXPANSION_WRITING = _doubles_term(1, v(K, B, C, J))
RAW_WRITING = _doubles_term(-1, v(J, C, K, B))


def _integrated_sum(term):
    """Total spin-integrated coefficient of `term` on the closed-shell block."""
    template = _residual_template("doubles", [term])
    block = _closed_shell_representative_block(template)
    return sum(st.coeff for st in ucc_integrate_target([term], block))


def _verdicts(v_factor):
    """{spin string: (sign, block) or None} for a v factor over all 16 spin labels."""
    out = {}
    for spins in itertools.product("ab", repeat=4):
        label = {
            "k": SpinIndex(K, spins[0]),
            "b": SpinIndex(B, spins[1]),
            "c": SpinIndex(C, spins[2]),
            "j": SpinIndex(J, spins[3]),
        }
        got = _antisym_to_allowed(v_factor, label)
        out["".join(spins)] = (
            None if got is None
            else (got[0], "".join(x.spin for x in got[1]))
        )
    return out


class PremiseTests(unittest.TestCase):
    """The reproducer is only meaningful if the two writings really are one term."""

    def test_the_two_writings_are_algebraically_equal(self):
        key_e, coeff_e = _eri_canonical(EXPANSION_WRITING)
        key_r, coeff_r = _eri_canonical(RAW_WRITING)
        self.assertEqual(key_e, key_r, "the two writings are not the same integral")
        self.assertEqual(coeff_e, coeff_r, "folded coefficients differ")

    def test_the_writings_present_different_line_structures(self):
        """The mechanism: `_line_pairs` reads slot k with slot k+n, so the two
        writings expose different lines to the adapter."""
        def lines(factor):
            return {
                frozenset((factor.indices[p].name, factor.indices[q].name))
                for p, q in _line_pairs(factor)
            }

        self.assertEqual(lines(v(K, B, C, J)),
                         {frozenset(("k", "c")), frozenset(("b", "j"))})
        self.assertEqual(lines(v(J, C, K, B)),
                         {frozenset(("j", "k")), frozenset(("c", "b"))})

    def test_bra_ket_exchange_alone_is_harmless(self):
        """Ruled out as the cause, so a fix does not go looking there: exchanging
        (2,3,0,1) maps lines p-r, q-s to r-p, s-q -- the same lines -- so it never
        changes a verdict. Swept over all occ/vir slot patterns x all spin labels."""
        names = "pqrs"
        makers = {"o": make_occ, "v": make_vir}
        divergent = 0
        total = 0
        for pattern in itertools.product("ov", repeat=4):
            indices = [makers[ch](nm) for ch, nm in zip(pattern, names)]
            written = v(*indices)
            exchanged = written.with_indices(
                tuple(written.indices[p] for p in (2, 3, 0, 1)))
            for spins in itertools.product("ab", repeat=4):
                label = {nm: SpinIndex(ix, sp)
                         for nm, ix, sp in zip(names, indices, spins)}
                a = _antisym_to_allowed(written, label)
                b = _antisym_to_allowed(exchanged, label)
                total += 1
                if (a is None) != (b is None):
                    divergent += 1
                elif a is not None:
                    sa = (a[0], "".join(x.spin for x in a[1]))
                    sb = (b[0], "".join(x.spin for x in b[1]))
                    if sa != sb:
                        divergent += 1
        self.assertEqual(total, 256)
        self.assertEqual(
            divergent, 0,
            "bra<->ket exchange alone now diverges; the scoped root cause "
            "(exchange COMPOSED with a within-group swap) may no longer hold")


class OrientationDivergenceMechanismTests(unittest.TestCase):
    """Per-FACTOR behavior after the e.2.1 fix.

    Note what is and is not required here. Per-factor verdicts still DIFFER between
    the two writings, and correctly so: `v(j,c,k,b) = -v(k,b,c,j)` as a factor, and
    the raw term carries the compensating -1 in its own coefficient. The invariant is
    per-TERM (see :class:`OrientationInvarianceTests`), which is why a per-factor
    equality assertion would be the wrong target.

    What the fix guarantees per factor is that both writings are reoriented to the
    SAME canonical member of their 8-fold ERI orbit, so the block each lands in --
    and hence which spatial tensor slice the kernel reads -- no longer depends on how
    the caller wrote it.
    """

    def setUp(self):
        self.f1 = _verdicts(v(K, B, C, J))
        self.f2 = _verdicts(v(J, C, K, B))

    def test_survival_is_identical_across_writings(self):
        self.assertEqual(
            {s for s, d in self.f1.items() if d is None},
            {s for s, d in self.f2.items() if d is None},
        )

    def test_six_of_sixteen_cases_survive(self):
        self.assertEqual(sum(1 for d in self.f1.values() if d is not None), 6)

    def test_blocks_are_identical_where_both_survive(self):
        """The load-bearing per-factor guarantee: same block, so the same spatial
        slice is read regardless of writing."""
        for spins, d1 in self.f1.items():
            if d1 is None:
                continue
            with self.subTest(spins=spins):
                self.assertEqual(d1[1], self.f2[spins][1])

    def test_factor_signs_still_differ_by_the_orbit_parity(self):
        """Not a defect: the two writings differ by one antisymmetry sign as factors,
        so every surviving case flips. The term coefficient carries the compensating
        sign, and `OrientationInvarianceTests` is where that is checked.

        Before the fix only 4 of the 6 flipped (the mixed-spin ones) -- an
        inconsistent, writing-dependent pattern. A UNIFORM flip is the signature of a
        clean global reorientation."""
        flipped = {
            spins for spins, d1 in self.f1.items()
            if d1 is not None and d1[0] != self.f2[spins][0]
        }
        surviving = {s for s, d in self.f1.items() if d is not None}
        self.assertEqual(flipped, surviving,
                         "the orbit-parity flip should be uniform across all "
                         "surviving cases, not restricted to the mixed-spin ones")


class OrientationInvarianceTests(unittest.TestCase):
    """The invariant V1.1e.2 establishes: equal algebra integrates equally.

    Was 2 vs 0 before e.2.1; now both integrate to 0.
    """

    def test_equal_algebra_integrates_equally(self):
        self.assertEqual(
            _integrated_sum(EXPANSION_WRITING),
            _integrated_sum(RAW_WRITING),
        )

    def test_both_writings_integrate_to_zero(self):
        """Pinned absolutely, not just relatively: a fix that made both sides agree on
        some other value would satisfy the equality above while changing answers.
        Zero is what the pre-fix raw writing already gave, and the full pyscf
        `test_spin` suite (93 tests, S1/S2/S4 + rank-6/rank-8 FCI-limit gates) confirms
        the adapted residual is semantically unchanged."""
        self.assertEqual(_integrated_sum(EXPANSION_WRITING), 0)
        self.assertEqual(_integrated_sum(RAW_WRITING), 0)


if __name__ == "__main__":
    unittest.main()
