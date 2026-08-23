"""U4.3: the `--ucc` / `ucc=True` switch reaches a runnable all-sectors bundle.

`ucc_adapt_equations` had **zero non-test callers** until this switch existed --
the whole UCC emit path was reachable only from tests. This gates the wiring:
the flag exists, it produces an all-sectors bundle, it is mutually exclusive with
`--spin-adapt`, and it is default-off so the shipped build is unaffected.

WHY MUTUALLY EXCLUSIVE RATHER THAN ORDERED. `spin_adapt` and `ucc` both resolve
spin, in OPPOSITE directions: adaptation collapses the blocks into one spatial
tensor per rank, UCC keeps them resolved as separate arrays. Running both would
collapse and then attempt to re-resolve, which is not a composition in either
order. Raising follows the `dress_operators` / `factorize_tau` precedent, where
silent precedence had disguised the hazard.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import print_cpp_planck  # noqa: E402


class UccSwitchTests(unittest.TestCase):
    def test_ucc_emits_an_all_sectors_bundle(self):
        """Every target block-tagged, so no per-rank reference residual exists."""
        tu = print_cpp_planck("ccsd", ucc=True)
        tags = set(re.findall(r'sector_tags\.push_back\(\{(\d+), "([ab]+)"\}\)', tu))
        self.assertEqual(
            tags,
            {("1", "aa"), ("1", "bb"), ("2", "aaaa"), ("2", "abab"), ("2", "bbbb")})
        self.assertEqual(
            re.findall(r"residuals_by_rank\.push_back", tu), [],
            "a UCC bundle must push no per-rank reference residual -- that is what "
            "makes it all-sectors, and what U4.0 taught the runtime to accept")

    def test_ucc_reads_spin_blocked_storage(self):
        """The U3 routing must survive the switch: no untagged reads of either kind."""
        tu = print_cpp_planck("ccsd", ucc=True)
        self.assertEqual(re.findall(r"mo_blocks\.[a-z]{4}\(", tu), [])
        self.assertEqual(re.findall(r"reference\.f_[ov]{2}\(", tu), [])

    def test_ucc_and_spin_adapt_are_mutually_exclusive(self):
        with self.assertRaises(ValueError) as caught:
            print_cpp_planck("ccsd", ucc=True, spin_adapt=True)
        self.assertIn("mutually exclusive", str(caught.exception))

    def test_default_emit_is_unchanged(self):
        """Default off, so the shipped build cannot move. Compares the whole TU
        rather than a marker: a marker would miss a change anywhere else."""
        import hashlib

        for method in ("ccsd", "ccsdt"):
            with self.subTest(method=method):
                plain = print_cpp_planck(method)
                self.assertNotIn("sector_tensor(1, \"aa\")", plain,
                                 "the default emit must carry no UCC blocks")
                # Pin the exact bytes; these are the hashes from before the switch
                # was added, re-verified after.
                digest = hashlib.sha256(plain.encode()).hexdigest()[:16]
                self.assertEqual(
                    digest,
                    {"ccsd": "af74826e253415a2",
                     "ccsdt": "775e185b5ab27566"}[method],
                    f"the default {method} emit changed; --ucc must be inert when off")

    def test_intermediates_are_forced_off_under_ucc(self):
        """CSE is unvalidated on spin-RESOLVED terms for the same reason it is on
        spatial ones, and UCC multiplies the term count by the block count, so the
        compile-time argument is strictly worse. Forced off, not an error, matching
        the spin_adapt precedent."""
        tu = print_cpp_planck("ccsd", ucc=True, include_intermediates=True)
        self.assertEqual(re.findall(r"\bbuild_W\w*\(", tu), [],
                         "no intermediate builders may be emitted on the UCC path")


if __name__ == "__main__":
    unittest.main()
