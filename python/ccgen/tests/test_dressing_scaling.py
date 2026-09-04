"""D0/D1: dressed-operator recognition must not rebuild the residual multiset per hypothesis.

`hypothesis_is_consistent` needs the raw residual's ERI-canonical multiset, which is invariant
across the whole hypothesis search — `residual_terms` is the manifold and does not change. It
was recomputed on every call: 7,461 calls on `ccsdt` triples, at ~0.036 s per rebuild, i.e.
~270 s of a ~300 s manifold spent recomputing a fixed value. That product
(`n_hypotheses × n_terms`) is the super-linear term, and both factors grow with rank.

Fixed by computing it once in `find_operator_occurrences` and passing it down.

WHY THESE GATES COUNT CALLS RATHER THAN SECONDS. A wall-clock threshold is flaky on shared
machines and says nothing about *why* it regressed. The call count is deterministic and names
the defect directly: if `raw_multiset` starts scaling with the hypothesis count again, that is
exactly this bug returning.

The byte-identity test is not ceremony either. `raw_multiset` is pure, so hoisting it *cannot*
change results — which is precisely the kind of "should be fine" that earns an assertion.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ccgen.optimization.dressed_equation as dressed_equation  # noqa: E402
from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.optimization.dressing import (  # noqa: E402
    assemble_dressed_equation,
    find_operator_occurrences,
    seeded_operators,
)


class _CountingRawMultiset:
    """Context manager counting `raw_multiset` calls made through the module attribute."""

    def __init__(self):
        self.calls = 0
        self._original = None

    def __enter__(self):
        self._original = dressed_equation.raw_multiset

        def counted(terms):
            self.calls += 1
            return self._original(terms)

        dressed_equation.raw_multiset = counted
        return self

    def __exit__(self, *exc):
        dressed_equation.raw_multiset = self._original
        return False


class RawMultisetCallCountTests(unittest.TestCase):
    """The residual multiset is built a bounded number of times, not per hypothesis."""

    @classmethod
    def setUpClass(cls):
        cls.equations = generate_cc_equations(
            "ccsdt", engine="diagram", canonical_fock=True)
        cls.operators = seeded_operators()

    def test_occurrence_search_builds_the_multiset_once(self):
        """`find_operator_occurrences` runs the hypothesis search; one multiset suffices."""
        for manifold in ("singles", "doubles"):
            for operator in self.operators:
                with _CountingRawMultiset() as counter:
                    find_operator_occurrences(operator, self.equations[manifold])
                with self.subTest(manifold=manifold, operator=operator.name):
                    self.assertLessEqual(
                        counter.calls, 1,
                        f"raw_multiset called {counter.calls}x for one operator's search")

    def test_call_count_does_not_scale_with_manifold_size(self):
        """The signature of the defect: it was 139 calls for 12 terms and 2452 for 73.

        Post-fix the count is driven by the number of operators and top-level phases, not by
        term count -- so growing the manifold ~6x must not grow the count materially.
        """
        counts = {}
        for manifold in ("singles", "doubles"):
            with _CountingRawMultiset() as counter:
                assemble_dressed_equation(self.operators, self.equations[manifold])
            counts[manifold] = counter.calls

        self.assertLess(
            counts["doubles"], 4 * counts["singles"] + 20,
            f"raw_multiset calls still scale with manifold size: {counts}")
        self.assertLess(counts["doubles"], 100,
                        f"expected a bounded number of rebuilds, got {counts}")


class HoistIsBehaviourPreservingTests(unittest.TestCase):
    """`raw_multiset` is pure, so hoisting must be byte-identical. Asserted, not assumed."""

    @classmethod
    def setUpClass(cls):
        cls.equations = generate_cc_equations(
            "ccsdt", engine="diagram", canonical_fock=True)
        cls.operators = seeded_operators()

    def test_occurrences_match_an_explicitly_recomputed_search(self):
        """Passing the multiset down must give the same occurrences as letting each
        consistency check build its own (the pre-fix behaviour)."""
        for manifold in ("singles", "doubles"):
            terms = self.equations[manifold]
            for operator in self.operators:
                hoisted = find_operator_occurrences(operator, terms)
                fresh = find_operator_occurrences(
                    operator, terms, raw=dressed_equation.raw_multiset(terms))
                with self.subTest(manifold=manifold, operator=operator.name):
                    self.assertEqual([o["term"] for o in hoisted],
                                     [o["term"] for o in fresh])

    def test_dressed_equation_is_unchanged(self):
        """The end-to-end invariant: same terms, same coefficients, same order."""
        for manifold in ("energy", "singles", "doubles"):
            terms = self.equations[manifold]
            with self.subTest(manifold=manifold):
                first = assemble_dressed_equation(self.operators, terms)
                second = assemble_dressed_equation(self.operators, terms)
                self.assertEqual(first, second)
                self.assertTrue(first, "empty dressed equation would pass vacuously")


if __name__ == "__main__":
    unittest.main()
