"""V1.1f: index-space validity of intermediate specs (the CSE trap).

`validate_intermediate_specs` asserts an `IntermediateSpec`'s index metadata is
self-consistent: the `index_space_sig` matches the slot spaces character-for-character,
rank agrees, no slot is repeated, every definition term carries the spec's own free
indices in the same order, and (optionally) every slot's space is buildable against the
reference partition.

WHY. This is the assertion the `--spin-adapt` CSE path never got, and the reason
`include_intermediates` is force-disabled there (`e0f3849`): CSE derives an intermediate's
indices from a *syntactic* pattern match, so it can mislabel occ/vir, and the `sig` drives
the emitted buffer's dimensions -- a sig claiming `oooo` for a `vvoo` tensor allocates and
indexes the wrong shape, silently. Dressed operators should be immune (their indices come
from a recognized physical operator with a declared block), and measured they are: all five
dressed specs validate clean. But both ride the same `IntermediateSpec`, so the point is to
have the check for whatever rides it next.

Half of this file is therefore NEGATIVE cases. A validator that never fires is worthless,
so each check is exercised against a spec deliberately broken in exactly that one way.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import _dress_operator_equations, generate_cc_equations  # noqa: E402
from ccgen.indices import make_occ, make_vir  # noqa: E402
from ccgen.optimization.intermediates import (  # noqa: E402
    IntermediateSpec,
    validate_intermediate_spec,
    validate_intermediate_specs,
)
from ccgen.project import AlgebraTerm  # noqa: E402
from ccgen.tensors import t2  # noqa: E402

A, B = make_vir("a"), make_vir("b")
I, J = make_occ("i"), make_occ("j")
PARTITION = {"occ": 3, "vir": 4}


def _spec(**overrides):
    """A minimal valid rank-4 vvoo spec; override one field to break one check."""
    indices = overrides.pop("indices", (A, B, I, J))
    term = AlgebraTerm(
        coeff=1,
        factors=(t2(A, B, I, J),),
        free_indices=overrides.pop("term_free", indices),
        summed_indices=(),
        connected=True,
    )
    base = {
        "name": "W_test",
        "indices": indices,
        "definition_terms": (term,),
        "usage_count": 1,
        "index_space_sig": "vvoo",
    }
    base.update(overrides)
    return IntermediateSpec(**base)


class ValidSpecTests(unittest.TestCase):
    def test_the_baseline_spec_is_valid(self):
        self.assertEqual(validate_intermediate_spec(_spec(), PARTITION), [])

    def test_partition_is_optional(self):
        self.assertEqual(validate_intermediate_spec(_spec()), [])


class SigMismatchTests(unittest.TestCase):
    """Check 1 -- the CSE trap itself."""

    def test_wrong_sig_is_caught(self):
        problems = validate_intermediate_spec(_spec(index_space_sig="oooo"), PARTITION)
        self.assertTrue(any("does not match slot spaces" in p for p in problems))

    def test_transposed_sig_is_caught(self):
        """`oovv` vs `vvoo` -- same character multiset, wrong order. A multiset check
        would miss this; the emitted buffer would have transposed dimensions."""
        problems = validate_intermediate_spec(_spec(index_space_sig="oovv"), PARTITION)
        self.assertTrue(any("does not match slot spaces" in p for p in problems))

    def test_short_sig_is_caught(self):
        problems = validate_intermediate_spec(_spec(index_space_sig="vvo"), PARTITION)
        self.assertTrue(any("len(index_space_sig)" in p for p in problems))


class RepeatedSlotTests(unittest.TestCase):
    """Check 3 -- a duplicated slot means the tensor is really a lower-rank trace."""

    def test_repeated_index_is_caught(self):
        problems = validate_intermediate_spec(
            _spec(indices=(A, A, I, J), term_free=(A, A, I, J)), PARTITION)
        self.assertTrue(any("repeated index slot" in p for p in problems))


class DefinitionTermTests(unittest.TestCase):
    """Check 4 -- a permuted definition term writes a transpose into the buffer."""

    def test_permuted_free_indices_are_caught(self):
        problems = validate_intermediate_spec(_spec(term_free=(B, A, I, J)), PARTITION)
        self.assertTrue(any("order differs from" in p for p in problems))

    def test_wrong_free_index_set_is_caught(self):
        problems = validate_intermediate_spec(_spec(term_free=(A, B, I, I)), PARTITION)
        self.assertTrue(any("does not match" in p for p in problems))


class ReferencePartitionTests(unittest.TestCase):
    """Check 5 -- internally consistent but unbuildable."""

    def test_zero_extent_space_is_caught(self):
        problems = validate_intermediate_spec(_spec(), {"occ": 3, "vir": 0})
        self.assertTrue(any("extent" in p for p in problems))

    def test_missing_space_is_caught(self):
        problems = validate_intermediate_spec(_spec(), {"occ": 3})
        self.assertTrue(any("extent" in p for p in problems))


class SpecListTests(unittest.TestCase):
    """List-level checks that per-spec validation cannot see."""

    def test_duplicate_names_are_caught(self):
        problems = validate_intermediate_specs([_spec(), _spec()], PARTITION)
        self.assertTrue(any("duplicate spec name" in p for p in problems))

    def test_forward_reference_is_caught(self):
        """The emitter materializes builders in list order, so referencing a spec
        defined later is a use-before-def."""
        from ccgen.tensors import Tensor

        later = _spec(name="W_later")
        earlier = dataclasses.replace(
            _spec(name="W_earlier"),
            definition_terms=(AlgebraTerm(
                coeff=1,
                factors=(Tensor("W_later", (A, B, I, J)),),
                free_indices=(A, B, I, J),
                summed_indices=(),
                connected=True,
            ),),
        )
        problems = validate_intermediate_specs([earlier, later], PARTITION)
        self.assertTrue(any("defined later in the list" in p for p in problems))

    def test_correct_order_is_accepted(self):
        from ccgen.tensors import Tensor

        first = _spec(name="W_first")
        second = dataclasses.replace(
            _spec(name="W_second"),
            definition_terms=(AlgebraTerm(
                coeff=1,
                factors=(Tensor("W_first", (A, B, I, J)),),
                free_indices=(A, B, I, J),
                summed_indices=(),
                connected=True,
            ),),
        )
        self.assertEqual(validate_intermediate_specs([first, second], PARTITION), [])

    def test_self_reference_is_caught(self):
        from ccgen.tensors import Tensor

        spec = dataclasses.replace(
            _spec(name="W_self"),
            definition_terms=(AlgebraTerm(
                coeff=1,
                factors=(Tensor("W_self", (A, B, I, J)),),
                free_indices=(A, B, I, J),
                summed_indices=(),
                connected=True,
            ),),
        )
        problems = validate_intermediate_specs([spec], PARTITION)
        self.assertTrue(any("references itself" in p for p in problems))


class DressedSpecsAreValidTests(unittest.TestCase):
    """The V1.1f gate proper: the real dressed specs must validate clean."""

    @classmethod
    def setUpClass(cls):
        eqs = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        _, cls.specs = _dress_operator_equations(eqs)

    def test_all_dressed_specs_are_valid(self):
        self.assertEqual(validate_intermediate_specs(self.specs, PARTITION), [])

    def test_the_expected_specs_are_present(self):
        """Guards against the gate passing vacuously on an empty list."""
        self.assertEqual({s.name for s in self.specs},
                         {"tau", "tau_c", "Wmnij", "Wabef", "Wmbej"})

    def test_dependency_order_puts_tau_before_its_consumers(self):
        order = [s.name for s in self.specs]
        for consumer in ("Wmnij", "Wabef", "Wmbej"):
            for term in next(s for s in self.specs if s.name == consumer).definition_terms:
                for factor in term.factors:
                    if factor.name in ("tau", "tau_c"):
                        with self.subTest(consumer=consumer, dep=factor.name):
                            self.assertLess(order.index(factor.name),
                                            order.index(consumer))


if __name__ == "__main__":
    unittest.main()
