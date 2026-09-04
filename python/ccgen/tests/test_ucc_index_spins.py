"""U3b.0: the spin of every index in a resolved UCC term.

The emitter writes loop bounds from an `Index`, and an Index carries only a SPACE
(occ/vir) -- never a spin. The U1 bridge drops the spin label by design, because
RCC stores one spatial tensor per rank and has nothing to route. Under UCC the
bound differs per spin (`noa` vs `nob`), so it must be recovered elsewhere.

It is recoverable from the FACTORS: a resolved factor carries its block in its name
(`t2_abab`, `v_aaaa`, `f_bb`) and slot k of that factor carries spin tag[k].

WHY THIS IS ITS OWN STEP RATHER THAN PART OF THE EMITTER CHANGE. If two factors
disagreed about an index's spin, the slot mapping would be wrong -- the R3.1.2
failure mode, where a factor indexes the wrong slice and the residual comes out
near-zero rather than obviously broken. That is a much larger problem than loop
bounds, and it is far cheaper to find here, on the symbolic manifold, than in a
C++ kernel. So the map is built and gated before anything consumes it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.spin import (  # noqa: E402
    ucc_adapt_equations,
    ucc_term_index_spins,
)


def _ucc(method: str = "ccsd"):
    return ucc_adapt_equations(generate_cc_equations(method))


class IndexSpinMapTests(unittest.TestCase):
    def test_every_index_of_every_term_is_assigned(self):
        """An unassigned index would silently keep a spin-blind bound -- the exact
        defect U3b exists to remove, reintroduced one index at a time."""
        for method in ("ccsd", "ccsdt"):
            with self.subTest(method=method):
                for target, terms in _ucc(method).items():
                    if target == "energy":
                        continue
                    for term in terms:
                        spins = ucc_term_index_spins(term)
                        used = {idx.name
                                for factor in term.factors
                                for idx in factor.indices}
                        self.assertEqual(
                            used - set(spins), set(),
                            f"{target}: indices with no spin assigned")

    def test_no_index_receives_two_spins(self):
        """The load-bearing assertion. A conflict means two factors disagree about
        an index, i.e. the slot mapping is wrong -- and `ucc_term_index_spins`
        raises rather than picking one, so this also pins that it raises."""
        total = 0
        for method in ("ccsd", "ccsdt"):
            with self.subTest(method=method):
                for target, terms in _ucc(method).items():
                    if target == "energy":
                        continue
                    for term in terms:
                        total += len(ucc_term_index_spins(term))
        self.assertGreater(total, 0, "no assignments at all; fixture drift")

    def test_assignment_count_is_the_measured_one(self):
        """Pins the CCSD number so a manifold change is visible rather than silent."""
        total = sum(len(ucc_term_index_spins(t))
                    for target, terms in _ucc().items() if target != "energy"
                    for t in terms)
        self.assertEqual(total, 2346)

    def test_the_tag_is_read_POSITIONALLY_not_by_space(self):
        """The detail most likely to be got wrong when re-deriving this.

        `t2_abab`'s slots are (vir, vir, occ, occ) -- virtuals first -- so the tag
        does NOT read "occ half then vir half" at the factor level. It reads
        straight across the slots as they appear. A space-grouped reading gives the
        same answer for `aaaa`/`bbbb` and a DIFFERENT one for `abab`, so a
        same-spin check cannot catch the mistake.
        """
        for target, terms in _ucc().items():
            if target != "doubles_abab":
                continue
            for term in terms:
                for factor in term.factors:
                    name = factor.name if hasattr(factor, "name") else ""
                    root, _, tag = name.rpartition("_")
                    if not root or not tag or any(c not in "ab" for c in tag):
                        continue
                    if len(factor.indices) != 4 or tag != "abab":
                        continue
                    spaces = tuple(i.space for i in factor.indices)
                    if spaces != ("vir", "vir", "occ", "occ"):
                        continue
                    spins = ucc_term_index_spins(term)
                    # Positional: slot 0 (a vir) is 'a', slot 2 (an occ) is 'a'.
                    self.assertEqual(spins[factor.indices[0].name], "a")
                    self.assertEqual(spins[factor.indices[1].name], "b")
                    self.assertEqual(spins[factor.indices[2].name], "a")
                    self.assertEqual(spins[factor.indices[3].name], "b")
                    return
        self.skipTest("no virtuals-first t2_abab factor in the manifold")

    def test_a_conflicting_slot_mapping_raises(self):
        """Falsifiability: the map must REJECT a contradiction rather than resolve
        it. Built by hand, because the real manifold has none (that is the point)."""
        class FakeIndex:
            def __init__(self, name, space):
                self.name, self.space = name, space

        class FakeFactor:
            def __init__(self, name, indices):
                self.name, self.indices = name, indices

        class FakeTerm:
            def __init__(self, factors):
                self.factors = factors

        i = FakeIndex("i", "occ")
        a = FakeIndex("a", "vir")
        term = FakeTerm([
            FakeFactor("f_aa", [i, a]),      # says i is alpha
            FakeFactor("f_bb", [i, a]),      # says i is beta
        ])
        with self.assertRaises(ValueError) as caught:
            ucc_term_index_spins(term)
        self.assertIn("disagree", str(caught.exception))

    def test_a_slot_count_mismatch_raises(self):
        """A tag whose length does not match the factor's slots is a mapping bug,
        not something to truncate."""
        class FakeIndex:
            def __init__(self, name, space):
                self.name, self.space = name, space

        class FakeFactor:
            def __init__(self, name, indices):
                self.name, self.indices = name, indices

        class FakeTerm:
            def __init__(self, factors):
                self.factors = factors

        term = FakeTerm([FakeFactor("t2_abab", [FakeIndex("i", "occ")])])
        with self.assertRaises(ValueError) as caught:
            ucc_term_index_spins(term)
        self.assertIn("slot mapping", str(caught.exception))

    def test_rcc_terms_yield_an_empty_map(self):
        """Calling this on the RCC path must return nothing rather than guess --
        bare `v`/`f`/`t2` carry no block, and inventing one would be a defect."""
        from ccgen.spin import spin_adapt_equations

        rcc = spin_adapt_equations(generate_cc_equations("ccsd"))
        for target, terms in rcc.items():
            if target == "energy":
                continue
            for term in terms:
                self.assertEqual(ucc_term_index_spins(term), {})


if __name__ == "__main__":
    unittest.main()
