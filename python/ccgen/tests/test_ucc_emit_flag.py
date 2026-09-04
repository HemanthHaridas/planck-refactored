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

    def test_spin_adapted_emit_is_unchanged(self):
        """The OTHER RCC path, and the one a UCC change is most likely to break.

        The pin above covers `spin_adapt=False` only, and that gap shipped a
        regression: U3b.2a keyed its spin map on "does this term have tagged
        factors" rather than on `ucc`, which is true for the SPIN-ADAPTED path
        too -- its rank-4 sector amplitudes are named `t4_aaabaaab`. Spin-adapted
        kernels got `noa`/`nvb` loop bounds over a `no`/`nv` preamble and the TU
        stopped compiling, while the pin above stayed green throughout.

        Both engines, because they are not the same text: `engine="diagram"` and
        the default differ by 2038 lines at rank 2 (identical byte length, which
        is a coincidence and exactly why the length check alone is not enough).
        The generator and every UCC gate use `diagram`; the flag matrix in
        `test_emit_flag_matrix.py` pins the default. Neither covered the other.

        RANK 4 IS THE ONE THAT MATTERS, and it is why this test is not just the
        rank-2/3 pin extended. The `t4_aaabaaab` sector names that trigger the
        defect do not exist below rank 4: measured, ranks 2 and 3 emit ZERO
        spin-adapted terms with a non-empty spin map, so a ranks-2-and-3 version
        of this test passes under the exact regression it exists to catch. That
        was verified by mutation, not assumed -- and it is the third time in this
        step that a gate written at a convenient rank turned out vacuous.
        `ccsdtq` costs ~4s here, so there is no reason to leave it out.
        """
        import hashlib

        expected = {
            ("ccsd", None): "44705c8ad85f951c",
            ("ccsd", "diagram"): "4d1ab40e5a75fb19",
            ("ccsdt", None): "7d5dc96aeebb2141",
            ("ccsdt", "diagram"): "792a73c904403849",
            # The discriminating point: rank 4 is where the sector amplitudes live.
            ("ccsdtq", "diagram"): "d6e0f38aba1e6961",
        }
        for (method, engine), digest in expected.items():
            with self.subTest(method=method, engine=engine):
                kwargs = {"spin_adapt": True}
                if engine is not None:
                    kwargs["engine"] = engine
                text = print_cpp_planck(method, **kwargs)
                self.assertEqual(
                    hashlib.sha256(text.encode()).hexdigest()[:16], digest,
                    f"the spin-adapted {method} emit (engine={engine}) changed")

    def test_ucc_symbols_do_not_collide_with_rcc(self):
        """U5.0: a UCC TU must be linkable ALONGSIDE the RCC one for the method.

        MEASURED AT RANK 3, NOT RANK 2, and that distinction is the point. Before
        this landed:

            ccsd    1 collision   compute_ccsd_energy
            ccsdt   2 collisions  compute_ccsdt_energy,
                                  make_generated_ccsdt_kernels

        Rank 2 understates it: RCC emits no kernel bundle below the arbitrary
        floor, so `make_generated_*_kernels` exists on only one side there and a
        rank-2-only check would have found half the collision and looked
        conclusive. `force_arbitrary=True` puts a bundle on the RCC side too.
        """
        for method, rcc_kwargs in (("ccsd", {}), ("ccsdt", {"force_arbitrary": True})):
            with self.subTest(method=method):
                rcc = self._defined_symbols(print_cpp_planck(method, **rcc_kwargs))
                ucc = self._defined_symbols(print_cpp_planck(method, ucc=True))
                self.assertTrue(rcc and ucc, "a TU defined no symbols at all")
                self.assertEqual(
                    sorted(rcc & ucc), [],
                    f"the {method} RCC and UCC translation units define the same "
                    f"symbol(s); they would not link into one binary")

    def test_ucc_bundle_factory_is_distinctly_named(self):
        """The factory the registry will call must be reachable without ambiguity."""
        ucc = self._defined_symbols(print_cpp_planck("ccsd", ucc=True))
        self.assertIn("make_generated_ucc_ccsd_kernels", ucc)
        self.assertNotIn("make_generated_ccsd_kernels", ucc)

    @staticmethod
    def _defined_symbols(tu: str) -> set[str]:
        """Function definitions in an emitted TU (top-level, so column 0)."""
        return set(re.findall(r"^\w[\w:<>&,* ]*? (\w+)\(", tu, re.M))

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
