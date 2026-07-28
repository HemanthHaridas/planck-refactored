"""S0 gate: the spin-adaptation index model + single-term spin labeling.

Structural checks only (S0 is labeling, not coefficient integration): a term's
indices get spins consistent along shared lines, free indices take the requested
external block, and summed indices enumerate 2^(#distinct summed names) cases.
Coefficient algebra (UCC blocks = S1, RCC alpha=beta collapse = S2) is later.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.indices import make_occ, make_vir  # noqa: E402
from ccgen.spin import SpinIndex, spin_label_cases, SPINS  # noqa: E402


class SpinIndexTests(unittest.TestCase):
    def test_wraps_base_and_validates_spin(self):
        a = make_vir("a")
        sa = SpinIndex(a, "a")
        self.assertEqual(sa.name, "a")
        self.assertEqual(sa.space, "vir")
        self.assertEqual(sa.spin, "a")
        self.assertEqual(repr(sa), "aa")
        with self.assertRaises(ValueError):
            SpinIndex(a, "x")

    def test_identity_is_base_plus_spin(self):
        i = make_occ("i")
        self.assertEqual(SpinIndex(i, "a"), SpinIndex(make_occ("i"), "a"))
        self.assertNotEqual(SpinIndex(i, "a"), SpinIndex(i, "b"))


class SpinLabelCasesTests(unittest.TestCase):
    """Gate on the pp-ladder doubles term 1/2 t2(c,d,i,j) v(c,d,a,b):
    free i,j,a,b ; summed c,d."""

    def _pp_ladder(self):
        terms = [
            t for t in generate_cc_equations("ccd")["doubles"]
            if tuple(sorted(f.name for f in t.factors)) == ("t2", "v")
        ]
        pp = [
            t for t in terms
            if [i.name for i in t.factors[0].indices] == ["c", "d", "i", "j"]
        ]
        self.assertEqual(len(pp), 1)
        return pp[0]

    def test_case_count_is_two_to_the_summed_names(self):
        pp = self._pp_ladder()
        cases = spin_label_cases(pp, {"i": "a", "j": "b", "a": "a", "b": "b"})
        # two distinct summed names (c, d) -> 2^2 = 4 spin cases
        self.assertEqual(len(cases), 4)

    def test_free_indices_take_the_requested_external_block(self):
        pp = self._pp_ladder()
        ext = {"i": "a", "j": "b", "a": "a", "b": "b"}
        for label in spin_label_cases(pp, ext):
            for name, spin in ext.items():
                self.assertEqual(label[name].spin, spin, name)

    def test_shared_summed_name_has_one_consistent_spin(self):
        # c appears in BOTH t2 and v; it is ONE line and must carry ONE spin in
        # each case (the contracted line preserves spin). spin_label_cases keys
        # by name, so this holds by construction -- assert it explicitly.
        pp = self._pp_ladder()
        for label in spin_label_cases(pp, {"i": "a", "j": "b", "a": "a", "b": "b"}):
            # every index NAME maps to exactly one SpinIndex
            self.assertIn("c", label)
            self.assertIn("d", label)
            self.assertIn(label["c"].spin, SPINS)

    def test_summed_spins_are_enumerated_exhaustively(self):
        pp = self._pp_ladder()
        cases = spin_label_cases(pp, {"i": "a", "j": "b", "a": "a", "b": "b"})
        cd = {(label["c"].spin, label["d"].spin) for label in cases}
        self.assertEqual(cd, {("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")})

    def test_missing_external_spin_raises(self):
        pp = self._pp_ladder()
        with self.assertRaises(ValueError):
            spin_label_cases(pp, {"i": "a", "j": "b", "a": "a"})  # missing b


if __name__ == "__main__":
    unittest.main()
