"""O4: a chunked kernel must build each operator EXACTLY ONCE.

The defect this pins (found 2026-08-30 while threading the CC residual):
`_emit_kernel` emitted the intermediate builds as `const auto` locals
unconditionally, and then, for any kernel above `_KERNEL_CHUNK_TERMS`, delegated
to `_emit_chunked_kernel`, which builds the same operators again into the
`<kernel>_ops` struct that the `_partN` functions actually read. So every chunked
kernel built every operator TWICE, and the first set was dead -- never referenced,
because the parts take `ops`.

H5 introduced the struct hoist and did not remove the emission it superseded. It
cost ~6 % of the rank-3 solve (4.9 s serial on HF/6-31G, 88 operators) and scales
with operator count, so rank 4 (894 operators) stands to lose more.

**It was invisible to every existing gate**, and that is the interesting part: the
duplicate is semantically a no-op, so energies, residuals and the value gates were
all correct. Only wall-clock and a reading of the emitted text could see it. This
gate reads the text.
"""
from __future__ import annotations

import re
import unittest


class ChunkedKernelBuildsOnceTests(unittest.TestCase):
    @staticmethod
    def _emit():
        from ccgen.generate import print_cpp_planck
        return print_cpp_planck("ccsdt", dressing="derived", spin_adapt=True,
                                force_arbitrary=True)

    def _entry_body(self, src, kernel="compute_ccsdt_triples_residual"):
        """The main kernel function body: signature through `return result;`."""
        m = re.search(r"^Tensor\dD %s\(.*?\n\}" % re.escape(kernel),
                      src, re.S | re.M)
        self.assertIsNotNone(m, f"no entry function {kernel}")
        return m.group(0)

    def test_no_operator_is_built_twice_in_a_chunked_kernel(self):
        try:
            src = self._emit()
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")

        body = self._entry_body(src)
        builds = re.findall(r"(build_W_[A-Za-z0-9_]+)\(reference", body)
        dupes = {n for n in builds if builds.count(n) > 1}
        self.assertEqual(
            dupes, set(),
            f"{len(dupes)} operators are built more than once in the chunked "
            f"kernel entry point, e.g. {sorted(dupes)[:3]}. The `ops` struct is "
            "what the _partN functions read; a second `const auto` build is dead "
            "work. See _emit_kernel's `chunked` guard.")

    def test_the_chunked_entry_has_no_dead_const_auto_builds(self):
        """The stronger, more direct form: no `const auto` build at all there.

        Everything the parts need arrives through `ops`. A `const auto W_... =
        build_W_...` in the entry function is by construction unreferenced --
        stated separately from the duplicate check because a future change could
        emit one WITHOUT a matching struct entry, which the check above would
        miss.
        """
        try:
            src = self._emit()
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")

        body = self._entry_body(src)
        agg = body.find("_ops ops{")
        self.assertGreater(agg, 0, "no ops aggregate -- kernel is not chunked?")
        locals_before = re.findall(
            r"const auto (W_[A-Za-z0-9_]+) = build_W_", body[:agg])
        self.assertEqual(
            locals_before, [],
            f"{len(locals_before)} dead `const auto` operator builds precede the "
            f"ops aggregate, e.g. {locals_before[:3]}")

    def test_the_gate_is_not_vacuous(self):
        """The kernel under test must actually be chunked and actually build ops.

        Without this, an emitter change that stopped chunking (or stopped using
        the struct) would make both checks above pass while testing nothing.
        """
        try:
            src = self._emit()
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")

        self.assertIn("compute_ccsdt_triples_residual_part0(", src,
                      "triples kernel is not chunked -- gate is vacuous")
        body = self._entry_body(src)
        n = len(re.findall(r"build_W_[A-Za-z0-9_]+\(reference", body))
        self.assertGreater(n, 10,
                           f"only {n} operator builds in the entry point; the "
                           "gate needs a kernel that actually builds operators")


if __name__ == "__main__":
    unittest.main()
