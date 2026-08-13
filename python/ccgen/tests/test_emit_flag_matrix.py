"""V1.2.0: pin `print_cpp_planck`'s output across its flag matrix.

V1.2 refactors a function with four interacting flags (`dress_operators`, `spin_adapt`,
`factorize_tau`, `force_arbitrary`) to remove an early return. This is the net placed
BEFORE that change: it records what every currently-reachable combination emits, so a
"wiring" edit that quietly alters an existing path is caught rather than assumed benign.

Two things are pinned per combination, because either alone is weak:

- the full-text SHA-256, which catches any content change; and
- the byte length, which is what a human reads in a diff and what the scope docs quote.

A length that holds while the hash moves means the emit was reordered or a token swapped --
exactly the kind of change a length-only check would wave through.

WHAT IS DELIBERATELY NOT PINNED. `dress_operators=True` combined with `spin_adapt`,
`factorize_tau`, or `force_arbitrary`: those are unreachable today (the early return fires
first), so there is no status quo to record. `test_dress_operators_currently_ignores_other_
flags` pins the *observable consequence* of that unreachability instead, which is the thing
V1.2.1 changes and V1.2.4 must guard.
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import print_cpp_planck  # noqa: E402

METHOD = "ccsd"

# (label, kwargs, byte length, sha256) measured on the current tree.
BASELINES = (
    ("bare", {}, 37216,
     "af74826e253415a261f9b57efd4ed906827ef0c70cb9da6989e0f941d3b9f656"),
    ("spin_adapt", {"spin_adapt": True}, 65431,
     "44705c8ad85f951cbebb532a2fe60ea5418feddfbc28b8523ee2768fd12e0fd4"),
    ("factorize_tau", {"factorize_tau": True}, 37413,
     "05a0b6d055700c7f3e2ae929dc14e2a8a8f8a010e668b7e249c2999e6e441765"),
    ("force_arbitrary", {"force_arbitrary": True}, 37175,
     "bf1e083d5759120621369de2fe1572a926fbf96d21d886387d452d81afe6363e"),
    ("spin_adapt+force_arbitrary",
     {"spin_adapt": True, "force_arbitrary": True}, 64265,
     "f3a85400f3178fabb06f1ba674e51a5fb9e963d437b70ffda32f900c98e7ca2f"),
    ("dress_operators", {"dress_operators": True}, 27960,
     "1c52c36178906c9b1916a9cfca67b6cf7da8176f35b7e50033fc3229a91c01b5"),
)


def _emit(**kwargs) -> str:
    return print_cpp_planck(METHOD, **kwargs)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FlagMatrixBaselineTests(unittest.TestCase):
    """Every currently-reachable flag combination keeps its exact output."""

    @classmethod
    def setUpClass(cls):
        cls.emitted = {label: _emit(**kwargs) for label, kwargs, _, _ in BASELINES}

    def test_full_text_hashes_are_unchanged(self):
        """The strong check: any content change moves the digest."""
        for label, _, _, digest in BASELINES:
            with self.subTest(combination=label):
                self.assertEqual(_sha(self.emitted[label]), digest)

    def test_byte_lengths_are_unchanged(self):
        """The readable check, and the figure the scope docs quote. Kept alongside the
        hash so a failure says *how much* moved, not just that something did."""
        for label, _, length, _ in BASELINES:
            with self.subTest(combination=label):
                self.assertEqual(len(self.emitted[label]), length)

    def test_outputs_are_pairwise_distinct(self):
        """Guards against the length checks passing because two flags collapsed onto the
        same output -- a refactor that made a flag a no-op would otherwise look fine."""
        digests = {label: _sha(text) for label, text in self.emitted.items()}
        self.assertEqual(len(set(digests.values())), len(digests),
                         f"two combinations emit identical text: {digests}")

    def test_emission_is_deterministic(self):
        """The hashes below are only meaningful if emit is reproducible in-process."""
        for label, kwargs, _, _ in BASELINES:
            with self.subTest(combination=label):
                self.assertEqual(_sha(_emit(**kwargs)), _sha(self.emitted[label]))


class DressedPathReachabilityTests(unittest.TestCase):
    """What the early return currently makes unreachable.

    These pin the STATUS QUO, not desired behavior. V1.2.1 removes the early return and
    V1.2.4 adds explicit guards, at which point these tests should be updated to assert
    the new contract -- they exist so that transition is deliberate rather than silent.
    """

    def test_dress_operators_currently_ignores_other_flags(self):
        """`factorize_tau` / `spin_adapt` / `force_arbitrary` are silently dropped under
        dressing today, because the early return fires before any of them is read.

        This is the hazard V1.2.4 addresses: the parent scope records dress x tau as
        "already mutually exclusive", but the exclusion is unreachability, not a guard --
        so removing the early return ACTIVATES tau rather than continuing to exclude it.
        """
        dressed_only = _emit(dress_operators=True)
        for extra in ({"factorize_tau": True},
                      {"spin_adapt": True},
                      {"force_arbitrary": True}):
            with self.subTest(extra=extra):
                self.assertEqual(_emit(dress_operators=True, **extra), dressed_only)

    def test_dressed_output_carries_its_builders(self):
        """Anchors what the dressed path is, so a regression in V1.2.1 that emitted an
        undressed TU at the right byte count could not slip through."""
        text = _emit(dress_operators=True)
        for builder in ("build_tau", "build_Wmnij", "build_Wabef", "build_Wmbej"):
            with self.subTest(builder=builder):
                self.assertIn(builder, text)


if __name__ == "__main__":
    unittest.main()
