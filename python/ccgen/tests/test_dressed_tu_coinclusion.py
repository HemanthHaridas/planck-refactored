"""V1.3.0/V1.3.2: two dressed TUs must be co-includable in one translation unit.

`generated_kernel_registry.cpp` `#include`s each generated TU into ONE translation unit, so
builder symbols from different methods share a scope.

**Pre-V1.3.2 the dressed builders were unsuffixed** (`build_tau`, `build_Wmnij`, …) in every
method, and whether that collided depended on the configuration:

- **Non-arbitrary** (`RCCSDAmplitudes` vs `RCCSDTAmplitudes`): differing amplitude parameter
  types made them **overloads** — co-inclusion compiled.
- **`force_arbitrary=True`** (both `ArbitraryOrderRCCAmplitudes`): identical signatures made
  them **redefinitions** — 5 errors, one per builder.

The failing case was the mode the registry actually uses, so the hazard was conditional on
exactly the target configuration and invisible in the mode probed first.

**V1.3.2 fixed the mechanism** rather than restricting scope around it: `_builder_symbol` names
every builder `build_<name>_<method>`, so the names are disjoint and the identical signatures no
longer matter. The chosen route (b) over "restrict dressing to one rank and enforce it", because
the collision is a property of the naming scheme, not of how many ranks are enabled — a scope
restriction would leave the trap armed for whoever enabled a second dressed rank.

These tests keep BOTH halves of the original asymmetry, because it is the reason the suffix is
required rather than cosmetic: the identical-signature hazard still exists underneath, and if
signatures ever diverge per method the justification changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import print_cpp_planck  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
# Intermediate names; the emitted symbol is `build_<name>_<method>` (V1.3.2).
DRESSED_INTERMEDIATES = ("tau", "tau_c", "Wmnij", "Wabef", "Wmbej")


def _compiler():
    cxx = os.environ.get("CXX", "c++")
    return cxx if shutil.which(cxx) else None


class _CoInclude(unittest.TestCase):
    """Shared harness: emit two dressed TUs and object-compile them together."""

    def _require_toolchain(self):
        cxx = _compiler()
        if cxx is None:
            self.skipTest("no C++ compiler available")
        eigen = REPO / "build" / "_deps" / "eigen-src"
        if not eigen.is_dir():
            self.skipTest("Eigen fetch not present (configure the build first)")
        return cxx, eigen

    def _co_compile(self, **flags):
        """Object-compile ccsd+ccsdt dressed TUs in one TU; return (returncode, stderr).

        Object compilation, not `-fsyntax-only`: the redefinition diagnostic is what is being
        measured, and the point of this file is that a weaker check reports success here.
        """
        cxx, eigen = self._require_toolchain()
        first = print_cpp_planck("ccsd", **flags)
        second = print_cpp_planck("ccsdt", **flags)
        with tempfile.TemporaryDirectory() as work:
            (Path(work) / "first.cpp").write_text(first)
            (Path(work) / "second.cpp").write_text(second)
            combined = Path(work) / "both.cpp"
            combined.write_text(
                f'#include "{work}/first.cpp"\n#include "{work}/second.cpp"\n')
            proc = subprocess.run(
                [cxx, "-std=c++23", "-c", "-o", os.devnull, "-w",
                 "-I", str(REPO / "src"), "-I", str(eigen), str(combined)],
                capture_output=True, text=True, timeout=1800,
            )
        return proc.returncode, proc.stderr


class NonArbitraryCoInclusionTests(_CoInclude):
    """Distinct amplitude types make the builders overloads, so co-inclusion is legal."""

    def test_builders_differ_only_in_amplitude_type(self):
        """The mechanism, asserted at the source so the compile result is explained rather
        than merely observed."""
        flags = {"dress_operators": True, "spin_adapt": True}
        signatures = {}
        for method, expected in (("ccsd", "RCCSDAmplitudes"),
                                 ("ccsdt", "RCCSDTAmplitudes")):
            text = print_cpp_planck(method, **flags)
            start = text.index(f"build_tau_{method}(")
            signatures[method] = text[text.index("(", start):text.index(")", start)]
            with self.subTest(method=method):
                self.assertIn(expected, signatures[method])
        self.assertNotEqual(signatures["ccsd"], signatures["ccsdt"])

    def test_co_inclusion_compiles(self):
        rc, stderr = self._co_compile(dress_operators=True, spin_adapt=True)
        self.assertEqual(rc, 0, f"expected overloads to co-compile:\n{stderr[-1200:]}")
        self.assertNotIn("redefinition", stderr)


class ArbitraryOrderCoInclusionTests(_CoInclude):
    """The case that used to clash: identical signatures, resolved by V1.3.2's suffixing.

    Under `force_arbitrary` every method's builders take `ArbitraryOrderRCCAmplitudes` and
    `ArbitraryOrderDenominatorCache`, so the SIGNATURES are identical and the pre-V1.3.2
    unsuffixed names made co-inclusion a redefinition -- measured, 5 errors, one per dressed
    builder. V1.3.2 suffixes each builder with its method (`build_tau_ccsd`), so the names are
    disjoint and the identical signatures no longer matter.

    The signature test below is kept: it is the reason the suffix is REQUIRED rather than
    cosmetic, so if signatures ever diverge per method the suffix's justification changes.
    """

    def test_builders_still_share_an_identical_signature(self):
        """The underlying hazard has not gone away -- only the names disambiguate it."""
        flags = {"dress_operators": True, "spin_adapt": True, "force_arbitrary": True}
        signatures = set()
        for method in ("ccsd", "ccsdt"):
            text = print_cpp_planck(method, **flags)
            start = text.index(f"build_tau_{method}(")
            signatures.add(text[text.index("(", start):text.index(")", start)])
            with self.subTest(method=method):
                self.assertIn("ArbitraryOrderRCCAmplitudes", text)
        self.assertEqual(len(signatures), 1,
                         "arbitrary-order builders should still share one signature; if not, "
                         "re-read _builder_symbol's rationale")

    def test_builder_names_are_method_disjoint(self):
        flags = {"dress_operators": True, "spin_adapt": True, "force_arbitrary": True}
        by_method = {}
        for method in ("ccsd", "ccsdt"):
            text = print_cpp_planck(method, **flags)
            by_method[method] = set(
                re.findall(r"^\w[\w:<>, ]*? (build_\w+)\($", text, re.M))
            with self.subTest(method=method):
                self.assertTrue(by_method[method], "no builders emitted")
                for name in by_method[method]:
                    self.assertTrue(name.endswith(f"_{method}"),
                                    f"{name} is not method-suffixed")
        self.assertEqual(by_method["ccsd"] & by_method["ccsdt"], set(),
                         "builder names must not overlap between methods")

    def test_co_inclusion_now_compiles(self):
        """V1.3.2's payoff: the configuration that produced 5 redefinitions is clean."""
        rc, stderr = self._co_compile(
            dress_operators=True, spin_adapt=True, force_arbitrary=True)
        self.assertEqual(rc, 0, f"expected suffixing to resolve the clash:\n{stderr[-1500:]}")
        self.assertNotIn("redefinition", stderr)


class AnchorIsNotCoIncludedTests(unittest.TestCase):
    """Why rank 3 is the V1.3 anchor: its TU is never co-included with another dressed TU."""

    def test_rank3_tu_has_a_single_dedicated_consumer(self):
        consumer = (REPO / "src" / "post_hf" / "cc" / "tensor_backend.cpp").read_text()
        self.assertIn("generated/cc/ccsdt_planck_generated.cpp", consumer)

        registry = (REPO / "src" / "post_hf" / "cc"
                    / "generated_kernel_registry.cpp").read_text()
        includes = re.findall(r'#include "generated/cc/(\w+)\.cpp"', registry)
        self.assertNotIn("ccsdt_planck_generated", includes,
                         "the plain rank-3 TU must not also be registry-co-included")

    def test_rank2_tu_has_no_consumer(self):
        """`ccsd_planck_generated.cpp` is generated but compiled into nothing, which is why
        the anchor is rank 3 rather than the rank-2 case every V1 measurement used."""
        hits = subprocess.run(
            ["grep", "-rl", "ccsd_planck_generated.cpp", str(REPO / "src")],
            capture_output=True, text=True).stdout.split()
        self.assertEqual(hits, [], f"expected no consumer, found: {hits}")


if __name__ == "__main__":
    unittest.main()
