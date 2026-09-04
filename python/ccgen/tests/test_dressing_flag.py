"""W3.1: the `--dressing {none,recognized,derived}` axis on the kernel emitter.

The load-bearing property is that **nothing that exists today changes meaning**:
`--dress-operators` must remain an exact alias for `--dressing recognized`, and
the default must remain the undressed emit. Silently changing what an existing
flag emits would make an old command line reproduce different kernels -- the
failure this enum was chosen to avoid.

All three values emit, and all three differ: `derived` was refused at W3.1 and
wired at W3.2, so the assertion here is that it produces its own translation
unit rather than silently falling back to another route.
"""
import pathlib
import subprocess
import sys
import tempfile
import unittest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "generate_planck_cc_kernels.py"


def _run(out_dir, *args):
    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(out_dir),
         "--methods", "ccsd", "--engine", "diagram", *args],
        capture_output=True, text=True)


class DressingFlagTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _emit(self, sub, *args):
        out = self.tmp / sub
        proc = _run(out, *args)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        files = sorted(out.glob("*.cpp"))
        self.assertTrue(files, f"no TU emitted for {args}")
        return b"".join(f.read_bytes() for f in files)

    def test_dress_operators_is_an_exact_alias(self):
        """The deprecated spelling must emit byte-identical kernels."""
        self.assertEqual(self._emit("a", "--dress-operators"),
                         self._emit("b", "--dressing", "recognized"))

    def test_default_is_undressed_and_explicit_none_matches(self):
        self.assertEqual(self._emit("c"), self._emit("d", "--dressing", "none"))

    def test_dressing_changes_the_output(self):
        """Guards against the enum being accepted and then ignored."""
        self.assertNotEqual(self._emit("e", "--dressing", "none"),
                            self._emit("f", "--dressing", "recognized"))

    def test_derived_emits_its_own_translation_unit(self):
        """W3.2 wired `derived`; it was refused at W3.1.

        This test previously asserted the refusal. That assertion was correct
        for W3.1 and is exactly what caught the behaviour change when W3.2
        landed -- kept as a positive assertion rather than deleted, so the route
        cannot silently regress to a no-op.
        """
        self.assertNotEqual(self._emit("g", "--dressing", "derived"),
                            self._emit("j", "--dressing", "none"))

    def test_both_spellings_together_is_an_error(self):
        proc = _run(self.tmp / "h", "--dressing", "recognized", "--dress-operators")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("alias", proc.stderr)

    def test_factorize_tau_mutual_exclusion_survives(self):
        proc = _run(self.tmp / "i", "--dressing", "recognized", "--factorize-tau")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("mutually exclusive", proc.stderr)


if __name__ == "__main__":
    unittest.main()
