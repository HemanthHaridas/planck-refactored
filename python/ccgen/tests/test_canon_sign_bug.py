"""T1.2a: isolate the name-overload false-zero bug in canonicalize_tensor.

When a tensor's antisymmetric pair holds two DISTINCT index objects that share a
name -- a free external and a summed dummy both called ``i`` -- the degeneracy
check keys on ``(space, name)``, sees a repeated slot, and returns sign 0,
falsely zeroing a legitimate term. This is the root cause of the seed-dependent
under-count in the ERI t1*t2 doubles residual (see the diagram/gate notes).

The distinguishing fact: ``is_dummy`` differs between the two indices, so they
are separable; the sign logic just does not look.

T1.2b fixed the bug; these are now hard assertions (the two that were
expectedFailure now pass).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.tensors import Tensor  # noqa: E402
from ccgen.indices import make_occ, make_vir  # noqa: E402
from ccgen.canonicalize import canonicalize_tensor  # noqa: E402


class CanonicalizeSignBugTests(unittest.TestCase):
    def test_genuine_degeneracy_is_still_zero(self):
        # Control: the SAME object twice is a real antisymmetry zero. Must stay 0
        # so the fix does not over-correct.
        i = make_occ("i", dummy=True)
        a = make_vir("a", dummy=True)
        b = make_vir("b", dummy=True)
        t = Tensor("t2", (a, b, i, i), antisym_groups=((0, 1), (2, 3)))
        _, sign = canonicalize_tensor(t)
        self.assertEqual(sign, 0)

    def test_distinct_same_name_indices_are_not_zero(self):
        # The bug: i_free and i_sum are DIFFERENT indices sharing the name "i".
        # t2(a,b,i_free,i_sum) is NOT antisymmetry-zero, but canonicalize_tensor
        # returns sign 0.
        i_free = make_occ("i", dummy=False)
        i_sum = make_occ("i", dummy=True)
        a = make_vir("a", dummy=True)
        b = make_vir("b", dummy=False)
        t = Tensor("t2", (a, b, i_free, i_sum), antisym_groups=((0, 1), (2, 3)))
        _, sign = canonicalize_tensor(t)
        self.assertNotEqual(sign, 0)

    def test_eri_pattern_from_the_real_bug(self):
        # The exact trigger from the conflated t1*t2*v terms: an ERI ket pair
        # holding a summed i_0 and a free/summed same-named i. Both in one
        # antisym pair (slots 2,3) -> the false zero.
        i_free = make_occ("i", dummy=False)
        i_sum = make_occ("i", dummy=True)
        i0 = make_occ("i_0", dummy=True)
        b = make_vir("b", dummy=True)
        t = Tensor("v", (b, i0, i_free, i_sum), antisym_groups=((0, 1), (2, 3)))
        _, sign = canonicalize_tensor(t)
        self.assertNotEqual(sign, 0)

    def test_different_names_in_a_pair_are_never_zeroed(self):
        # Control: distinct names in a pair are fine (no false zero possible).
        i0 = make_occ("i_0", dummy=True)
        i1 = make_occ("i_1", dummy=True)
        b = make_vir("b", dummy=True)
        t = Tensor("v", (b, i0, i1, i0), antisym_groups=((0, 1), (2, 3)))
        # slots 2,3 = (i1, i0): distinct names -> not zero.
        _, sign = canonicalize_tensor(t)
        self.assertNotEqual(sign, 0)


if __name__ == "__main__":
    unittest.main()
