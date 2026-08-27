"""D5.6: the ERI symmetry tables must stay one definition, and stay correct.

A spatial <pq|rs> has four index symmetries, all +1. The four single-swap
relations are true only for the antisymmetrized <pq||rs>.

Three modules have independently needed this distinction, and each wrote its own
table. Two got it right AND wrote a warning comment; the third kept the 8-fold
set. Two warning comments did not prevent the third copy from being wrong -- it
cost 41 of 288 emitted operator builders reading the wrong ERI block with a
bogus sign (D4/D5 in docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md), and before that
a 52 % energy defect that "passed every symbolic check"
(optimization/dressing.py:1912).

Hence a test, not a third comment.
"""
from __future__ import annotations

import itertools
import pathlib
import re
import unittest

import numpy as np

from ccgen.tensors import (
    ANTISYMMETRIZED_ERI_SYMMETRIES,
    SPATIAL_ERI_SYMMETRIES,
)


class EriSymmetryTableTests(unittest.TestCase):
    def test_spatial_relations_hold_on_a_spatial_integral(self):
        """Not asserted from the definition -- verified on a real tensor."""
        rng = np.random.default_rng(20260827)
        n = 6
        v = rng.standard_normal((n, n, n, n))
        # the symmetries a spatial physicist <pq|rs> over real orbitals has
        v = v + v.transpose(1, 0, 3, 2)
        v = v + v.transpose(2, 3, 0, 1)
        for perm, sign in SPATIAL_ERI_SYMMETRIES:
            self.assertEqual(sign, +1, f"{perm} must carry +1")
            self.assertLess(
                float(np.abs(v.transpose(perm) - sign * v).max()), 1e-12,
                f"{perm} is not a symmetry of a spatial <pq|rs>")

    def test_the_odd_relations_are_false_on_a_spatial_integral(self):
        """The control. If these held, the whole distinction would be moot."""
        rng = np.random.default_rng(7)
        n = 6
        v = rng.standard_normal((n, n, n, n))
        v = v + v.transpose(1, 0, 3, 2)
        v = v + v.transpose(2, 3, 0, 1)
        odd = [p for p, s in ANTISYMMETRIZED_ERI_SYMMETRIES if s < 0]
        self.assertEqual(len(odd), 4)
        for perm in odd:
            self.assertGreater(
                float(np.abs(v.transpose(perm) + v).max()), 1e-3,
                f"{perm} appears to hold on a spatial integral -- fixture is wrong")

    def test_spatial_is_exactly_the_parity_plus_one_subset(self):
        self.assertEqual(
            set(SPATIAL_ERI_SYMMETRIES),
            {(p, s) for p, s in ANTISYMMETRIZED_ERI_SYMMETRIES if s > 0})

    def test_no_module_redefines_a_signed_table(self):
        """One definition of the SIGNED tables. A copy is how this happened.

        Matches on shape, not on a name, so renaming the constant does not
        evade it: any module-level `(perm, sign)` 4-index table outside
        `tensors.py` is an offender.
        """
        root = pathlib.Path(__file__).resolve().parents[1]
        signed_entry = re.compile(
            r"\(\s*\(\s*\d\s*,\s*\d\s*,\s*\d\s*,\s*\d\s*\)\s*,\s*[+-]?1\s*\)")
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "tensors.py" or "tests" in path.parts:
                continue
            if len(signed_entry.findall(path.read_text())) >= 4:
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(
            offenders, [],
            "signed ERI symmetry table defined outside ccgen/tensors.py: "
            + ", ".join(offenders))

    def test_unsigned_permutation_sets_agree_with_the_signed_tables(self):
        """`optimization/dressing.py` keeps its own UNSIGNED permutation sets.

        They are deliberately NOT folded into the shared constants: they are
        sets used for canonicalization and enumeration, with parity computed
        separately (`_perm_parity`), not `(perm, sign)` pairs used for value
        reads. Merging two things of different shape would be a false
        unification.

        But they must still agree about WHICH permutations belong to which
        basis -- that is the invariant this whole defect violated -- so the
        membership is gated here rather than left to a comment.
        """
        from ccgen.optimization import dressing as D

        self.assertEqual(
            set(D._ERI_PERMUTATIONS_SPATIAL),
            {p for p, _ in SPATIAL_ERI_SYMMETRIES},
            "dressing's spatial permutation set disagrees with "
            "SPATIAL_ERI_SYMMETRIES")
        self.assertEqual(
            set(D._ERI_PERMUTATIONS),
            {p for p, _ in ANTISYMMETRIZED_ERI_SYMMETRIES},
            "dressing's full permutation set disagrees with "
            "ANTISYMMETRIZED_ERI_SYMMETRIES")

    def test_consumers_use_the_shared_spatial_table(self):
        from ccgen.emit import planck_tensor_cpp as E
        from ccgen.lowering import restricted_closed_shell as L
        self.assertIs(E._ERI_SYMMETRY_PERMUTATIONS, SPATIAL_ERI_SYMMETRIES)
        self.assertIs(L._ERI_SYMMETRY_PERMUTATIONS, SPATIAL_ERI_SYMMETRIES)


if __name__ == "__main__":
    unittest.main()
