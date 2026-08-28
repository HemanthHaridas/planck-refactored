"""O1 -- the symbolic transpose predicate agrees with a numeric oracle.

`symbolic_transpose` decides, on the SHAPE KEY alone, whether two derived
operators are the same contraction up to a permutation of the operator's slots.
That is the merge rule `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` (O1) needs to
recover the reuse the D6 split cost, and it must be decidable symbolically --
the emitter cannot compare arrays.

These tests hold it against a NUMERIC oracle (materialize both operators, try
every axis permutation) across two fixtures and three seeds. The two-fixture
sweep matters: `no=3, nv=4` and `no=4, nv=3` invert the asymmetry, so a
permutation that only appears to work because two extents coincide is caught.

Status, measured: EXACT on BOTH bases, at both `canonical_fock` settings -- no
false merges and no misses. Spatial got there in two steps: O2.0 corrected the
oracle's fixture (48 -> 18 misses; ~30 were oracle false positives, not
predicate gaps), and O2.2 added the amplitudes' own symmetry (18 -> 0).

Exactness is only meaningful alongside `PredicateStillDiscriminatesTests`: a
predicate that merged everything would also report zero misses.
"""

from __future__ import annotations

import itertools
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.optimization.factorize import manifold_operators  # noqa: E402
from ccgen.optimization.operator_identity import symbolic_transpose  # noqa: E402
from ccgen.spin import spin_adapt_equations  # noqa: E402
from ccgen.tests.residual_eval import (  # noqa: E402
    random_tensors,
    residual_einsum,
    ucc_closed_shell_tensors,
)

FIXTURES = ((3, 4), (4, 3))
SEEDS = (0, 1, 2)


def _tensors(no, nv, seed, spatial):
    """The fixture matching the basis under test.

    `random_tensors` antisymmetrizes `t2` and `v`. That is `<pq||rs>` -- correct
    for GCC, WRONG for spatial, where neither is antisymmetric. Running the
    spatial oracle on it made 30 pairs look equivalent that are equal only for
    that fixture (48 misses -> 18 on the correct one), an oracle false positive
    rather than a predicate gap. Recorded as O2.0 in
    `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`.
    """
    if spatial:
        return ucc_closed_shell_tensors(no, nv, seed=seed)[1]
    return random_tensors(no, nv, seed=seed)


def _build(spec, tensors, no, nv):
    out = None
    for dt in spec.definition_terms:
        arr = residual_einsum(dt, no, nv, tensors=tensors)
        ext = ([i for i in dt.free_indices if i.space == "vir"]
               + [i for i in dt.free_indices if i.space == "occ"])
        pos = {idx: k for k, idx in enumerate(ext)}
        if set(pos) != set(spec.indices):
            return None
        arr = np.transpose(arr, [pos[i] for i in spec.indices])
        out = arr if out is None else out + arr
    return out


def _numeric_transpose(sp1, sp2, no, nv, seed, spatial):
    tensors = _tensors(no, nv, seed, spatial)
    a1, a2 = _build(sp1, tensors, no, nv), _build(sp2, tensors, no, nv)
    if a1 is None or a2 is None:
        return None
    for p in itertools.permutations(range(a1.ndim)):
        t = np.transpose(a1, p)
        if t.shape == a2.shape and np.allclose(t, a2, atol=1e-12):
            return p
    return None


def _same_family_pairs(terms):
    ops = manifold_operators(terms, include_reuse=False)
    fam = defaultdict(list)
    for o in ops:
        fam["_".join(o.name.split("_")[:-1])].append(o)
    return [(a, b) for ms in fam.values() for a, b in itertools.combinations(ms, 2)]


def _oracle(a, b, spatial=False):
    """Numeric verdict, required CONSISTENT across fixtures and seeds.

    Returns (found, detail). An inconsistent verdict means the numeric check
    itself is fixture-dependent, which disqualifies it as an oracle for that
    pair rather than convicting the predicate."""
    verdicts = {_numeric_transpose(a, b, no, nv, s, spatial) is not None
                for (no, nv) in FIXTURES for s in SEEDS}
    if len(verdicts) != 1:
        return None, "numeric verdict varies by fixture/seed"
    return verdicts.pop(), ""


class SymbolicMatchesNumericTests(unittest.TestCase):

    def _check(self, label, terms, spatial, allow_misses):
        pairs = _same_family_pairs(terms)
        self.assertGreater(len(pairs), 10, f"{label}: too few pairs to be meaningful")
        false_merges, misses, unstable = [], [], []
        for a, b in pairs:
            found, why = _oracle(a, b, spatial=spatial)
            if found is None:
                unstable.append((a.name, b.name, why)); continue
            sym = symbolic_transpose(a, b, spatial=spatial) is not None
            if sym and not found:
                false_merges.append((a.name, b.name))
            elif found and not sym:
                misses.append((a.name, b.name))
        # A FALSE MERGE is never acceptable: it binds one array to two different
        # contractions, which is exactly the defect D6 fixed.
        self.assertEqual(
            false_merges, [],
            f"{label}: predicate claims equivalence the oracle denies:\n"
            + "\n".join(f"  {x} ~ {y}" for x, y in false_merges[:5]))
        self.assertEqual(unstable, [], f"{label}: unstable oracle: {unstable[:3]}")
        if not allow_misses:
            self.assertEqual(
                misses, [],
                f"{label}: {len(misses)}/{len(pairs)} equivalences missed:\n"
                + "\n".join(f"  {x} ~ {y}" for x, y in misses[:5]))
        return len(pairs), len(misses)

    def test_gcc_is_exact(self):
        """No false merges AND no misses, at both canonical_fock settings."""
        for cf in (True, False):
            with self.subTest(canonical_fock=cf):
                self._check(f"GCC cf={cf}",
                            generate_cc_equations("ccsd", canonical_fock=cf)["doubles"],
                            spatial=False, allow_misses=False)

    def test_spatial_is_exact(self):
        """Spatial: no false merges AND no misses, since O2.2.

        Was 48/49 misses on the spin-orbital fixture, 18/19 once O2.0 corrected
        the oracle, and 0 once O2.2 modelled `t2[abij] = t2[baji]`."""
        for cf in (True, False):
            with self.subTest(canonical_fock=cf):
                self._check(
                    f"SPATIAL cf={cf}",
                    spin_adapt_equations(
                        generate_cc_equations("ccsd", canonical_fock=cf))["doubles"],
                    spatial=True, allow_misses=False)


class PredicateStillDiscriminatesTests(unittest.TestCase):
    """Exactness could also be reached by a predicate that merges everything.
    These pin that it does not -- without them, `test_*_is_exact` going green
    proves nothing."""

    def _families(self, terms):
        from collections import defaultdict
        from ccgen.optimization.factorize import manifold_operators
        fam = defaultdict(list)
        for o in manifold_operators(terms, include_reuse=False):
            fam["_".join(o.name.split("_")[:-1])].append(o)
        return fam

    def test_not_every_same_family_pair_merges(self):
        for label, terms, sp in (
            ("GCC", generate_cc_equations("ccsd", canonical_fock=True)["doubles"], False),
            ("spatial", spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=True))["doubles"], True),
        ):
            pairs = _same_family_pairs(terms)
            merged = sum(1 for a, b in pairs
                         if symbolic_transpose(a, b, spatial=sp))
            with self.subTest(label):
                self.assertGreater(merged, 0, "predicate merges nothing")
                self.assertLess(merged, len(pairs),
                                "predicate merges EVERY pair -- it is vacuous")

    def test_operators_of_different_families_never_merge(self):
        """A cross-family merge would bind unrelated contractions to one array."""
        import itertools as it
        for label, terms, sp in (
            ("GCC", generate_cc_equations("ccsd", canonical_fock=True)["doubles"], False),
            ("spatial", spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=True))["doubles"], True),
        ):
            fam = self._families(terms)
            keys = list(fam)
            bad = [(a.name, b.name)
                   for k1, k2 in it.combinations(keys, 2)
                   for a in fam[k1] for b in fam[k2]
                   if symbolic_transpose(a, b, spatial=sp)]
            with self.subTest(label):
                self.assertEqual(bad, [], f"cross-family merges: {bad[:3]}")


class CanonicalShapeTests(unittest.TestCase):
    """O4: `canonical_shape` partitions operators exactly as the pairwise
    predicate does — one canonicalization per operator instead of one comparison
    per pair.

    It is NOT yet wired into `_shape_tag`. Naming operators by it merges them,
    but a merged operator is only correct if each CALL SITE reads the shared
    array with permuted indices, and `rewrite_term_factorized` currently emits
    each site's own canonical index order. Doing the naming half alone
    reintroduces the D6 defect — measured, 11 GCC doubles terms stop reproducing
    their source. See O4 in `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`."""

    def test_partition_matches_the_pairwise_predicate(self):
        from collections import defaultdict
        import itertools as it
        from ccgen.optimization.factorize import manifold_operators
        from ccgen.optimization.operator_identity import canonical_shape
        for label, terms, sp, expect in (
            ("GCC", generate_cc_equations("ccsd", canonical_fock=True)["doubles"],
             False, 19),
            ("spatial", spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=True))["doubles"],
             True, 31),
        ):
            ops = manifold_operators(terms, include_reuse=False)
            classes = defaultdict(list)
            for o in ops:
                classes[canonical_shape(o, sp)].append(o)
            with self.subTest(label):
                self.assertEqual(len(classes), expect)
                # every co-classified pair is pairwise-equivalent, and every
                # pairwise-equivalent pair is co-classified
                for members in classes.values():
                    for a, b in it.combinations(members, 2):
                        self.assertIsNotNone(
                            symbolic_transpose(a, b, spatial=sp),
                            f"{a.name}/{b.name} share a canonical shape but are "
                            f"not pairwise equivalent")
                keys = list(classes)
                for k1, k2 in it.combinations(keys, 2):
                    for a in classes[k1]:
                        for b in classes[k2]:
                            self.assertIsNone(
                                symbolic_transpose(a, b, spatial=sp),
                                f"{a.name}/{b.name} are equivalent but fall in "
                                f"different canonical classes")


class MergePlanTests(unittest.TestCase):
    """O4.1: the merge plan is computable, total, and inert.

    Inert is the point of this step. The plan says which operators COULD share
    an array and how each maps onto its representative; nothing consumes it yet.
    O4.2 makes call sites permute, O4.3 merges the names — separately, because
    doing both at once broke 11 GCC terms with no way to tell which half."""

    def _ops(self, spatial, manifold="doubles"):
        from ccgen.optimization.factorize import manifold_operators
        eqs = generate_cc_equations("ccsd", canonical_fock=True)
        if spatial:
            eqs = spin_adapt_equations(eqs)
        return manifold_operators(eqs[manifold], include_reuse=False)

    def test_plan_covers_every_operator(self):
        """Total, so no caller has to special-case a representative."""
        from ccgen.optimization.operator_identity import merge_plan
        for label, sp in (("GCC", False), ("spatial", True)):
            ops = self._ops(sp)
            plan = merge_plan(ops, sp)
            with self.subTest(label):
                self.assertEqual(set(plan), {o.name for o in ops})

    def test_plan_matches_the_canonical_partition(self):
        """Its classes are `canonical_shape`'s classes -- 19 GCC / 31 spatial,
        the same numbers O3 measured by union-find over the pairwise predicate."""
        from ccgen.optimization.operator_identity import merge_plan
        for label, sp, expect in (("GCC", False, 19), ("spatial", True, 31)):
            plan = merge_plan(self._ops(sp), sp)
            with self.subTest(label):
                self.assertEqual(len({r for r, _ in plan.values()}), expect)

    def test_representatives_are_self_mapped_and_deterministic(self):
        from ccgen.optimization.operator_identity import merge_plan
        for label, sp in (("GCC", False), ("spatial", True)):
            ops = self._ops(sp)
            plan = merge_plan(ops, sp)
            with self.subTest(label):
                for name, (rep, perm) in plan.items():
                    if name == rep:
                        self.assertEqual(perm, tuple(range(len(perm))),
                                         f"{name} is its own rep but not identity")
                # rep is the lexicographic minimum of its class, so reordering
                # the input cannot change the plan
                self.assertEqual(plan, merge_plan(list(reversed(ops)), sp))

    def test_the_permutations_are_not_all_identity(self):
        """If every permutation were the identity, O4.2 and O4.3 would be
        vacuous and their gates would prove nothing. Measured: 8 non-identity
        on GCC doubles, 19 on spatial."""
        from ccgen.optimization.operator_identity import merge_plan
        for label, sp, least in (("GCC", False, 8), ("spatial", True, 19)):
            plan = merge_plan(self._ops(sp), sp)
            nonid = sum(1 for _, (_, p) in plan.items()
                        if p != tuple(range(len(p))))
            with self.subTest(label):
                self.assertGreaterEqual(nonid, least)


class SignPreservationTests(unittest.TestCase):
    """The restriction that makes the predicate sound on GCC.

    Four of the eight `_ERI_PERMUTATIONS` are odd and hold only up to -1.
    Including them produced two false merges on GCC. This pins the reason, so a
    future edit that reaches for the full orbit fails here with the cause named."""

    def test_only_sign_preserving_eri_permutations_are_folded(self):
        from ccgen.optimization.dressing import (
            _ERI_PERMUTATIONS, _ERI_PERMUTATIONS_SPATIAL, _perm_parity)
        from ccgen.optimization.operator_identity import v_variants

        self.assertEqual(sum(1 for p in _ERI_PERMUTATIONS if _perm_parity(p) == 1), 4)
        self.assertTrue(all(_perm_parity(p) == 1 for p in _ERI_PERMUTATIONS_SPATIAL))
        # a 4-slot v factor yields exactly the 4 even rewritings, not 8
        shape = (("v", ("0", "1", "2", "3")),)
        self.assertEqual(len(v_variants(shape, spatial=False)), 4)


if __name__ == "__main__":
    unittest.main()
