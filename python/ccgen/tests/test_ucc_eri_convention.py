"""U5.4 -- the ERI convention between ccgen and the C++ block cache.

THE CONVENTION, and it is pinned here because having it implicit on BOTH sides is
exactly what let the two halves disagree undetected through U3, U4 and U5.0-U5.3b:

    every array in the UCC block cache stores the PLAIN <pq|rs>;
    the emitter writes any antisymmetrization it needs into the emitted text.

WHAT WENT WRONG WITHOUT IT. The UCC algebra means the ANTISYMMETRIZED <pq||rs>
when it writes `v_aaaa`: `ucc_integrate_term_antisym` is documented as integrating
"for REAL ANTISYMMETRIC tensors", and the energy manifold carries ONE same-spin
term at coefficient 1/4 with no exchange partner, which only balances for <pq||rs>.
`build_ucc_spin_block_cache_from_eri` stores a plain `transform_eri` result. Both
sides were individually correct and self-consistent; nothing checked they agreed.

Measured cost of that gap: `ucc2` on B/STO-3G returned E_corr = -0.0705299626
against hand-written UCCSD's -0.0402694793. The `abab` channel was right (it has
no exchange partner, so it is unaffected) and both same-spin channels were wrong,
which is why the ratio was 1.7515 rather than a clean factor.

WHY THE EMITTER SIDE CARRIES IT. `<pq||rs> = <pq|rs> - <pq|sr>` needs slots 2 and
3 to be interchangeable, which holds only when they carry the same spin. For a
mixed block the exchange partner is a different SHAPE (`oovv_abab` is
(noa,nob,nva,nvb); its partner would be (noa,nob,nvb,nva)) and the array cannot
hold it. So antisymmetrizing the cache is not one transform over a vocabulary but
a conditional over a subset, leaving one accessor with two meanings. It would also
contradict a rule `ucc_blocks.cpp` already states, and silently redefine what three
landed C++ gates assert on.
"""

from __future__ import annotations

import re
import unittest

from ccgen.emit.planck_tensor_cpp import _block_needs_explicit_exchange
from ccgen.generate import print_cpp_planck


class ExchangePredicateTests(unittest.TestCase):
    """The rule itself, stated once and asserted directly."""

    def test_same_spin_blocks_need_the_exchange(self):
        for tag in ("aaaa", "bbbb"):
            self.assertTrue(_block_needs_explicit_exchange(tag), tag)

    def test_mixed_blocks_do_not(self):
        """A mixed block's exchange partner is a different tensor shape, and the
        algebra gives it no exchange partner anyway (coefficient 1, not 1/4)."""
        for tag in ("abab", "abba", "aabb"):
            self.assertFalse(_block_needs_explicit_exchange(tag), tag)

    def test_rcc_path_is_untouched(self):
        """`block_tag is None` is the closed-shell path, whose equations already
        carry their own exchange structure."""
        self.assertFalse(_block_needs_explicit_exchange(None))


class EmittedExchangeTests(unittest.TestCase):
    """The rule as it actually reaches the emitted text."""

    @classmethod
    def setUpClass(cls):
        cls.src = print_cpp_planck("ccsd", ucc=True)

    def test_same_spin_eri_reads_carry_an_exchange_partner(self):
        for tag in ("aaaa", "bbbb"):
            pairs = re.findall(
                r"\(v_" + tag + r"_([a-z]{4})\(([^)]*)\) - v_" + tag + r"_\1\(([^)]*)\)",
                self.src)
            self.assertGreater(len(pairs), 0, f"no exchange emitted for {tag}")
            for _space, direct, exchanged in pairs:
                d = [x.strip() for x in direct.split(",")]
                e = [x.strip() for x in exchanged.split(",")]
                self.assertEqual(
                    e, [d[0], d[1], d[3], d[2]],
                    "the exchange partner must swap the LAST TWO slots only")

    def test_mixed_eri_reads_carry_no_exchange(self):
        """The discriminating assertion: `abab` must be left alone. A rule applied
        spin-blindly would produce a shape-invalid read here, and this is the only
        block where that is visible."""
        self.assertEqual(
            re.findall(r"\(v_abab_[a-z]{4}\([^)]*\) - ", self.src), [])

    def test_fock_reads_carry_no_exchange(self):
        """Two-index blocks are plain on both sides already."""
        self.assertEqual(re.findall(r"\(f_[ab]+_[a-z]+\([^)]*\) - ", self.src), [])

    def test_rcc_emit_is_byte_identical(self):
        import hashlib
        for method, digest in (
            ("ccsd", "af74826e253415a261f9b57efd4ed906827ef0c70cb9da6989e0f941d3b9f656"),
            ("ccsdt", "775e185b5ab27566b639ea1db0c4ef7debfb4fc2374957445e17277b67d479a6"),
        ):
            with self.subTest(method=method):
                self.assertEqual(
                    hashlib.sha256(print_cpp_planck(method).encode()).hexdigest(),
                    digest)


if __name__ == "__main__":
    unittest.main()
