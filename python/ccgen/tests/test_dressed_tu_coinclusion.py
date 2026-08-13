"""V1.3.0: when two dressed TUs may share one translation unit, and when they may not.

`generated_kernel_registry.cpp` `#include`s each generated TU into ONE translation unit, and
dressed builders carry unsuffixed names (`build_tau`, `build_Wmnij`, ...) in every method. That
looks like a guaranteed redefinition clash — and it is not, which is exactly why this is pinned:

- **Non-arbitrary** (`RCCSDAmplitudes` vs `RCCSDTAmplitudes`): the builders differ in their
  amplitude parameter type, so they are **overloads**. Co-inclusion compiles cleanly.
- **`force_arbitrary=True`** (both take `ArbitraryOrderRCCAmplitudes`): signatures are
  identical, so they are **redefinitions**. Co-inclusion fails with 5 errors, one per builder.

The second case is the mode the registry actually uses, so the collision is conditional on
precisely the configuration V1.3 targets, and invisible in the mode probed first. Pinning both
halves keeps the asymmetry from being misremembered as "we checked, co-inclusion works" — that
was true only where it does not matter.

Consequence recorded by these tests: the V1.3 anchor is `ccsdt_planck_generated.cpp` via
`tensor_backend.cpp`, a single non-co-included TU with a method-specific amplitude type, so the
collision cannot arise for it. Dressing the *registry* path needs V1.3.2 to resolve naming first.
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
DRESSED_BUILDERS = ("build_tau", "build_tau_c",
                    "build_Wmnij", "build_Wabef", "build_Wmbej")


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
            start = text.index("build_tau(")
            signatures[method] = text[start:text.index(")", start)]
            with self.subTest(method=method):
                self.assertIn(expected, signatures[method])
        self.assertNotEqual(signatures["ccsd"], signatures["ccsdt"])

    def test_co_inclusion_compiles(self):
        rc, stderr = self._co_compile(dress_operators=True, spin_adapt=True)
        self.assertEqual(rc, 0, f"expected overloads to co-compile:\n{stderr[-1200:]}")
        self.assertNotIn("redefinition", stderr)


class ArbitraryOrderCoInclusionTests(_CoInclude):
    """Identical signatures make them redefinitions — the registry's actual mode."""

    def test_builders_share_an_identical_signature(self):
        flags = {"dress_operators": True, "spin_adapt": True, "force_arbitrary": True}
        signatures = set()
        for method in ("ccsd", "ccsdt"):
            text = print_cpp_planck(method, **flags)
            start = text.index("build_tau(")
            signatures.add(text[start:text.index(")", start)])
            with self.subTest(method=method):
                self.assertIn("ArbitraryOrderRCCAmplitudes", text)
        self.assertEqual(len(signatures), 1,
                         "arbitrary-order builders should share one signature")

    def test_co_inclusion_fails_with_one_error_per_builder(self):
        """Fails, and specifically on the five dressed builders — not for some other reason.

        This is the constraint V1.3.2 must resolve before the registry path can be dressed.
        """
        rc, stderr = self._co_compile(
            dress_operators=True, spin_adapt=True, force_arbitrary=True)
        self.assertNotEqual(rc, 0, "expected identical signatures to clash")
        clashing = {name for name in DRESSED_BUILDERS
                    if re.search(rf"redefinition of '{name}'", stderr)}
        self.assertEqual(clashing, set(DRESSED_BUILDERS),
                         f"expected all five builders to clash; got {sorted(clashing)}")


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
