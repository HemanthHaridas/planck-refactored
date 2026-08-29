"""D5 -- does a factorized rewrite reproduce its source term NUMERICALLY?

The factorizer (`factorize.py`, the *derivation* dressing route) has 47 tests and
none of them check value. `tree_preserves_term` checks leaf/index bookkeeping;
`test_budgeted_rewrite_is_exact` compares a factor `Counter`, which is blind to
index order by construction. A route can pass both and still not compute its own
equation -- and it does.

This is the instrument the retirement decision never had. It is deliberately RED:
21 of 41 rewritten GCC `ccsd` doubles terms disagree with their source. GCC is the
control basis -- there is no spin adaptation to blame there, which is what makes
the failure a factorizer defect rather than a composition one.

WHAT IT ISOLATES. Every failure comes from the MANIFOLD REPRESENTATIVE, not from
the contraction tree: `manifold_operators` keeps one definition per operator NAME
(`rep = specs[0]`) and binds it at every call site, while `_derived_name` builds
that name from sorted factor names + block signature and discards slot order.
Evaluated with each term's OWN `identify_node` spec instead, all 21 are exact.
That is measured here, not asserted -- see `test_per_term_specs_are_exact`.

FIXTURE, load-bearing: `no=3, nv=4`. On a square fixture (`no == nv`) a
transposed tensor read stays in bounds and returns a wrong number silently; the
asymmetric extents make it raise or mismatch instead. The dressed-route
retirement learned this the expensive way.

See `docs/CCGEN_TWO_DRESSING_ROUTES.md` (D5/D6).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.optimization.factorize import (  # noqa: E402
    Derived,
    _contraction_shape,
    emittable_operators,
    manifold_operators,
    rewrite_term_factorized,
    seeded_fingerprints,
    select_operators_by_savings,
)
from ccgen.tests.residual_eval import (  # noqa: E402
    random_tensors,
    residual_einsum,
    ucc_closed_shell_tensors,
)

NO, NV = 3, 4


def _tensors(no, nv, seed=0, spatial=False):
    """The fixture matching the BASIS under test.

    `random_tensors` antisymmetrizes `t2` and `v` -- correct for GCC
    (`<pq||rs>`), wrong for spatial input, where neither is antisymmetric. Using
    it on spatial terms lets a check pass for a property of the fixture rather
    than of the code; O2.0 in `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` records
    finding exactly that. The `spatial` bundle of `ucc_closed_shell_tensors`
    carries the right relations (`t2[abij] == t2[baji]`, `<pq|rs> == <qp|sr>`).
    """
    if spatial:
        return ucc_closed_shell_tensors(no, nv, seed=seed)[1]
    return random_tensors(no, nv, seed=seed)


def _build_operator(spec, tensors):
    """Materialize `spec` as an array in its own declared SLOT order.

    `residual_einsum` emits `[vir_ext..., occ_ext...]`; a spec's `indices` are
    its slot order (e.g. `(i,j,k,a)`). Storing the einsum layout and reading it
    back as slot order is a PROBE bug that would masquerade as the defect, so
    the transpose is explicit. Returns None if slots and definition disagree --
    that is a real defect, reported as a disagreement rather than an exception.
    """
    out = None
    for dt in spec.definition_terms:
        arr = residual_einsum(dt, NO, NV, tensors=tensors)
        ext = ([i for i in dt.free_indices if i.space == "vir"]
               + [i for i in dt.free_indices if i.space == "occ"])
        pos = {idx: k for k, idx in enumerate(ext)}
        if set(pos) != set(spec.indices):
            return None
        arr = np.transpose(arr, [pos[i] for i in spec.indices])
        out = arr if out is None else out + arr
    return out


def _canon(term, arr):
    """(array, key) in a canonical free-index order, so two terms that agree up
    to external ordering are compared in the same layout."""
    ext = ([i for i in term.free_indices if i.space == "vir"]
           + [i for i in term.free_indices if i.space == "occ"])
    order = sorted(range(len(ext)), key=lambda k: str(ext[k]))
    return np.transpose(arr, order), tuple(str(ext[k]) for k in order)


def _eval_rewritten(term, defs, tensors):
    materialized = dict(tensors)
    for f in term.factors:
        if f.name in defs and f.name not in materialized:
            arr = _build_operator(defs[f.name], tensors)
            if arr is None:
                return None
            materialized[f.name] = arr
    return residual_einsum(term, NO, NV, tensors=materialized)


def _rewrites(manifold_terms):
    """(source term, rewritten term, shared-representative defs) per rewritten term."""
    ops = manifold_operators(manifold_terms, include_reuse=False)
    _, keep = select_operators_by_savings(ops, savings_fraction=1.0)
    shared = {o.name: o for o in ops}
    for t in manifold_terms:
        r = rewrite_term_factorized(t, keep_operators=keep)
        if tuple(f.name for f in r.factors) == tuple(f.name for f in t.factors):
            continue  # single-step term, nothing factored out
        yield t, r, shared


def _disagreements(manifold_terms, per_term: bool = False, spatial: bool = False):
    """Terms whose rewrite does not reproduce them. `per_term=True` uses each
    term's own `identify_node` spec instead of the manifold representative.
    `spatial=True` selects the spatial fixture (see `_tensors`)."""
    tensors = _tensors(NO, NV, seed=0, spatial=spatial)
    fps = seeded_fingerprints()
    bad, total = [], 0
    for t, r, shared in _rewrites(manifold_terms):
        total += 1
        defs = dict(shared)
        if per_term:
            for _nt, res in emittable_operators(t, fps):
                if isinstance(res, Derived):
                    defs[res.spec.name] = res.spec
        src, ksrc = _canon(t, residual_einsum(t, NO, NV, tensors=tensors))
        raw = _eval_rewritten(r, defs, tensors)
        if raw is None:
            bad.append((t, "operator slots disagree with its definition")); continue
        got, kgot = _canon(r, raw)
        if ksrc != kgot or src.shape != got.shape:
            bad.append((t, f"external layout {kgot} != {ksrc}")); continue
        d = float(np.linalg.norm(src - got))
        if d > 1e-10 * max(1.0, float(np.linalg.norm(src))):
            bad.append((t, f"||diff|| = {d:.3e}"))
    return bad, total


class ProbeControlsTests(unittest.TestCase):
    """A value gate that fails its own controls is measuring the probe. These run
    the SAME evaluation path as the real gate, on inputs whose answer is known."""

    def test_evaluation_is_deterministic(self):
        terms = generate_cc_equations("ccsd")["doubles"]
        tensors = random_tensors(NO, NV, seed=0)
        a = residual_einsum(terms[0], NO, NV, tensors=tensors)
        b = residual_einsum(terms[0], NO, NV, tensors=tensors)
        self.assertTrue(np.array_equal(a, b))

    def test_untouched_terms_agree_through_the_same_path(self):
        """Terms the factorizer leaves alone must round-trip exactly. If this
        fails, every number the gate reports is about the probe."""
        terms = generate_cc_equations("ccsd")["doubles"]
        ops = manifold_operators(terms, include_reuse=False)
        _, keep = select_operators_by_savings(ops, savings_fraction=1.0)
        shared = {o.name: o for o in ops}
        tensors = random_tensors(NO, NV, seed=0)
        checked = 0
        for t in terms:
            r = rewrite_term_factorized(t, keep_operators=keep)
            if tuple(f.name for f in r.factors) != tuple(f.name for f in t.factors):
                continue
            checked += 1
            src, _ = _canon(t, residual_einsum(t, NO, NV, tensors=tensors))
            got, _ = _canon(r, _eval_rewritten(r, shared, tensors))
            self.assertLessEqual(
                float(np.linalg.norm(src - got)),
                1e-12 * max(1.0, float(np.linalg.norm(src))),
                f"untouched term did not round-trip: {t!r}")
        self.assertGreater(checked, 0, "no untouched terms -- control is vacuous")

    def test_the_gate_actually_rewrites_something(self):
        """Guards the other direction: if nothing is rewritten, a green value
        gate would mean nothing at all."""
        terms = generate_cc_equations("ccsd")["doubles"]
        self.assertGreater(sum(1 for _ in _rewrites(terms)), 20)


class FactorizedRewritePreservesValueTests(unittest.TestCase):
    """The gate. RED on the current tree -- that is the point."""

    def _assert_clean(self, label, terms, spatial):
        bad, total = _disagreements(terms, spatial=spatial)
        self.assertEqual(
            bad, [],
            f"{label}: {len(bad)}/{total} rewritten terms do not reproduce "
            f"their source:\n"
            + "\n".join(f"  {why}: {t!r}" for t, why in bad[:5]))

    def test_gcc_rewrite_is_value_preserving(self):
        """GCC is the CONTROL basis: no spin adaptation, so a failure here is a
        factorizer defect and cannot be blamed on composition.

        Both `canonical_fock` settings: `generate_cc_equations` defaults to
        False while the emitter defaults to True, so testing one leaves the
        other -- including the one production emits -- ungated."""
        for cf in (True, False):
            eqs = generate_cc_equations("ccsd", canonical_fock=cf)
            for manifold in ("singles", "doubles"):
                with self.subTest(canonical_fock=cf, manifold=manifold):
                    self._assert_clean(f"GCC cf={cf} {manifold}",
                                       eqs[manifold], spatial=False)

    def test_spatial_rewrite_is_value_preserving(self):
        """Spatial, on the SPATIAL fixture (see `_tensors`)."""
        from ccgen.spin import spin_adapt_equations
        for cf in (True, False):
            eqs = spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=cf))
            for manifold in ("singles", "doubles"):
                with self.subTest(canonical_fock=cf, manifold=manifold):
                    self._assert_clean(f"SPATIAL cf={cf} {manifold}",
                                       eqs[manifold], spatial=True)


class FactorOrderValueTests(unittest.TestCase):
    """A's probe: shuffling FACTOR ORDER must not change what a rewrite computes.

    `test_operator_set_invariant_under_factor_order` (test_factorize.py) asserts
    that the derived operator MULTISET is a function of the terms alone. It has
    been red since `7bdfdaf1`, and the reason is not a naming wobble: shuffling
    factor order produces **genuinely different decompositions**. Measured on
    rank-3 triples, the operator COUNT is invariant (963 every seed) but 16-22
    operator names differ, and inspecting the specs they are different
    contractions, not one contraction misnamed:

        base:  t2(a,e,j,k) v(d,e,l,m) t1(b,l) t1(d,i)
        shuf:  t2(a,e,i,j) v(d,e,l,m) t1(c,m) t1(d,k)

    That raised the question this class answers, and it was the one that could
    have reclassified those six red gates from test debt to a live defect: the
    value gate above only ever runs on the UNSHUFFLED manifold, so whether a
    shuffled decomposition still reproduces its source terms was unmeasured --
    and that is precisely the property that must hold if factor order can steer
    the factorizer.

    **It holds. 0 disagreements across 4 seeds**, GCC doubles + triples and
    spatial singles + doubles. So the factorizer reaches different but equally
    valid trees, and the invariance gate is asserting something stronger than
    correctness requires. Nothing here says which invariant SHOULD be asserted --
    that is the open question in docs/CCGEN_RED_TESTS_SCOPE.md -- but it does say
    the emitted values are not at risk either way.

    Non-vacuity is asserted rather than assumed: the shuffle must actually change
    the operator set, and terms must actually be rewritten. Without both, this
    would compare a decomposition against itself and pass for free.
    """

    @staticmethod
    def _shuffled(terms, seed):
        import random

        from ccgen.project import AlgebraTerm

        random.seed(seed)
        return [
            AlgebraTerm(t.coeff,
                        tuple(random.sample(list(t.factors), len(t.factors))),
                        t.free_indices, t.summed_indices, t.connected,
                        t.provenance)
            for t in terms
        ]

    def _assert_order_invariant_value(self, label, terms, spatial):
        for seed in range(4):
            with self.subTest(label=label, seed=seed):
                bad, total = _disagreements(self._shuffled(terms, seed),
                                            spatial=spatial)
                self.assertGreater(total, 0,
                                   f"{label} seed {seed}: nothing was rewritten")
                self.assertEqual(
                    bad, [],
                    f"{label} seed {seed}: {len(bad)}/{total} shuffled terms do "
                    f"not reproduce their source -- factor order changes VALUE, "
                    f"not just the decomposition:\n" +
                    "\n".join(f"  {w}" for _t, w in bad[:5]))

    def test_gcc_value_survives_factor_shuffling(self):
        eqs = generate_cc_equations("ccsdt", engine="diagram",
                                    canonical_fock=True)
        for manifold in ("doubles", "triples"):
            self._assert_order_invariant_value(f"GCC {manifold}", eqs[manifold],
                                               spatial=False)

    def test_spatial_value_survives_factor_shuffling(self):
        """Spatial is what production emits, on the spatial fixture.

        `ccsd` rather than `ccsdt`: the spatial fixture carries no `t3`, which is
        the same limit `test_spatial_rewrite_is_value_preserving` works under.
        """
        from ccgen.spin import spin_adapt_equations

        eqs = spin_adapt_equations(generate_cc_equations("ccsd",
                                                         canonical_fock=True))
        for manifold in ("singles", "doubles"):
            self._assert_order_invariant_value(f"SPATIAL {manifold}",
                                               eqs[manifold], spatial=True)

    def test_the_shuffle_actually_changes_the_decomposition(self):
        """Without this, the two gates above could pass on identical input."""
        from collections import Counter

        from ccgen.optimization.factorize import Reuse, identify_tree

        def opset(terms):
            c = Counter()
            for t in terms:
                for _n, r in identify_tree(t):
                    c[r.op_name if isinstance(r, Reuse) else r.name] += 1
            return c

        terms = generate_cc_equations("ccsdt", engine="diagram",
                                      canonical_fock=True)["triples"]
        base = opset(terms)
        moved = 0
        for seed in range(4):
            got = opset(self._shuffled(terms, seed))
            self.assertEqual(sum(base.values()), sum(got.values()),
                             f"seed {seed}: operator COUNT changed; the premise "
                             "that only the decomposition differs is wrong")
            moved += sum((got - base).values()) + sum((base - got).values())
        self.assertGreater(moved, 0,
                           "no seed changed the operator set -- the value gates "
                           "above are comparing a decomposition against itself")


class PermutedCallSiteTests(unittest.TestCase):
    """O4.2: with `merge_plan_map`, a call site reads its class REPRESENTATIVE's
    array through a permutation — and still reproduces its own term.

    This is the step that proves the merge is safe BEFORE the names merge
    (O4.3). Each operator still owns its own array here, so a wrong permutation
    shows up as a value failure that cannot be blamed on sharing. Doing both at
    once was tried and reverted: 11 GCC terms broke with no way to attribute
    the failure.

    KNOWN LIMIT, stated because it would otherwise read as coverage this gate
    does not have: every permutation `merge_plan` produces on `ccsd` is a single
    two-element swap and therefore SELF-INVERSE (8 on GCC, 19 on spatial,
    measured). Applying `perm` backwards is consequently a no-op on this data
    and the gate cannot detect it. It does catch dropping the permutation and
    applying a wrong one — both verified by sabotage. A manifold with a 3-cycle
    would close the gap; rank 3+ is the place to look."""

    def _check(self, label, terms, spatial):
        from ccgen.optimization.factorize import (
            manifold_operators, rewrite_term_factorized)
        from ccgen.optimization.operator_identity import merge_plan

        ops = manifold_operators(terms, include_reuse=False)
        plan = merge_plan(ops, spatial=spatial)
        by_name = {o.name: o for o in ops}
        # keep_operators=None (hoist everything) rather than a savings budget.
        # With `savings_fraction=1.0` the spatial keep set is 13 operators and
        # contains almost none of the 19 with a non-identity permutation, so the
        # permuted path is never taken and this gate would pass vacuously --
        # caught by the `permuted > 0` assertion below. Unrestricted, 17 of the
        # permuted operators are hoisted.
        keep = None
        tensors = _tensors(NO, NV, 0, spatial=spatial)

        checked = permuted = 0
        for t in terms:
            r = rewrite_term_factorized(
                t, keep_operators=keep, merge_plan_map=plan)
            if tuple(f.name for f in r.factors) == tuple(f.name for f in t.factors):
                continue
            checked += 1
            # O4.5 made the rewrite emit the REPRESENTATIVE's name as well as
            # its index order, so `f.name` is already a representative here and
            # `plan[f.name]` is always the identity. Count the permutation by
            # comparing against the un-permuted rewrite instead.
            plainr = rewrite_term_factorized(t, keep_operators=keep)
            for f, pf in zip(r.factors, plainr.factors):
                if f.name.startswith("W_") and (
                        f.indices != pf.indices or f.name != pf.name):
                    permuted += 1
            materialized = dict(tensors)
            for f in r.factors:
                if not f.name.startswith("W_") or f.name in materialized:
                    continue
                self.assertIn(f.name, by_name,
                              f"{f.name} is referenced but was not emitted")
                arr = _build_operator(by_name[f.name], tensors)
                self.assertIsNotNone(arr, f"{f.name} slots disagree with its definition")
                materialized[f.name] = arr
            src, ksrc = _canon(t, residual_einsum(t, NO, NV, tensors=tensors))
            got, kgot = _canon(r, residual_einsum(r, NO, NV, tensors=materialized))
            self.assertEqual(ksrc, kgot, f"{label}: external layout changed")
            self.assertLessEqual(
                float(np.linalg.norm(src - got)),
                1e-10 * max(1.0, float(np.linalg.norm(src))),
                f"{label}: permuted call site does not reproduce {t!r}")
        return checked, permuted

    def test_permuted_sites_reproduce_their_terms(self):
        from ccgen.spin import spin_adapt_equations
        for label, terms, sp in (
            ("GCC", generate_cc_equations("ccsd", canonical_fock=True)["doubles"], False),
            ("spatial", spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=True))["doubles"], True),
        ):
            with self.subTest(label):
                checked, permuted = self._check(label, terms, sp)
                self.assertGreater(checked, 10, "too few rewrites to be meaningful")
                # If nothing were permuted this gate would be a re-run of the
                # plain value gate and would prove nothing about O4.2.
                self.assertGreater(
                    permuted, 0,
                    f"{label}: no call site was actually permuted -- gate vacuous")

    def test_plan_changes_the_rewrite(self):
        """Passing the plan must actually alter call sites; otherwise O4.2 is a
        no-op wearing a gate. Measured: 9 GCC doubles terms change."""
        from ccgen.optimization.factorize import (
            manifold_operators, rewrite_term_factorized,
            select_operators_by_savings)
        from ccgen.optimization.operator_identity import merge_plan
        terms = generate_cc_equations("ccsd", canonical_fock=True)["doubles"]
        ops = manifold_operators(terms, include_reuse=False)
        plan = merge_plan(ops, spatial=True)
        _, keep = select_operators_by_savings(ops, savings_fraction=1.0)
        changed = sum(
            1 for t in terms
            if rewrite_term_factorized(t, keep_operators=keep, merge_plan_map=plan)
            != rewrite_term_factorized(t, keep_operators=keep))
        self.assertGreaterEqual(changed, 9)


class MergedOperatorsTests(unittest.TestCase):
    """O4.3: with the names merged, the surviving representatives still
    reproduce every term.

    `manifold_operators_with_plan` returns the merged specs and the call-site
    plan TOGETHER, deliberately: emitting merged specs while call sites read in
    their own slot order is the reverted first attempt (11 GCC terms broke).
    Handing them back as a pair is what stops the two halves being separated."""

    def _check(self, label, terms, spatial):
        from ccgen.optimization.factorize import (
            manifold_operators, manifold_operators_with_plan,
            rewrite_term_factorized)

        ops, plan = manifold_operators_with_plan(
            terms, include_reuse=False, spatial=spatial)
        unmerged = manifold_operators(terms, include_reuse=False)
        by_name = {o.name: o for o in ops}
        tensors = _tensors(NO, NV, 0, spatial=spatial)

        # anti-vacuity: merging must actually remove operators, else this gate
        # is a re-run of the un-merged one
        self.assertLess(len(ops), len(unmerged),
                        f"{label}: merging removed nothing")

        checked = 0
        for t in terms:
            r = rewrite_term_factorized(
                t, keep_operators=None, merge_plan_map=plan)
            if tuple(f.name for f in r.factors) == tuple(f.name for f in t.factors):
                continue
            checked += 1
            materialized = dict(tensors)
            for f in r.factors:
                if not f.name.startswith("W_") or f.name in materialized:
                    continue
                rep = plan[f.name][0] if f.name in plan else f.name
                self.assertIn(rep, by_name,
                              f"{label}: {f.name} maps to {rep}, which was not emitted")
                arr = _build_operator(by_name[rep], tensors)
                self.assertIsNotNone(arr)
                materialized[f.name] = arr
            src, _ = _canon(t, residual_einsum(t, NO, NV, tensors=tensors))
            got, _ = _canon(r, residual_einsum(r, NO, NV, tensors=materialized))
            self.assertLessEqual(
                float(np.linalg.norm(src - got)),
                1e-10 * max(1.0, float(np.linalg.norm(src))),
                f"{label}: merged operator does not reproduce {t!r}")
        self.assertGreater(checked, 10)
        return len(unmerged), len(ops)

    def test_merged_operators_reproduce_every_term(self):
        from ccgen.spin import spin_adapt_equations
        for label, terms, sp, expect in (
            ("GCC", generate_cc_equations("ccsd", canonical_fock=True)["doubles"],
             False, (27, 19)),
            ("spatial", spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=True))["doubles"],
             True, (59, 31)),
        ):
            with self.subTest(label):
                self.assertEqual(self._check(label, terms, sp), expect)

    def test_default_path_does_not_merge(self):
        """Merging is opt-in. A caller that does not ask for it must see the
        un-merged set, because it will not be permuting its call sites."""
        from ccgen.optimization.factorize import manifold_operators
        terms = generate_cc_equations("ccsd", canonical_fock=True)["doubles"]
        self.assertEqual(len(manifold_operators(terms, include_reuse=False)), 27)


class HigherRankMergeTests(unittest.TestCase):
    """O5: the merge holds at rank 3 and rank 4, not just `ccsd`.

    Asserted separately because this codebase has twice established that rank 3
    is not a proxy for rank 4 — the tensor-accessor fix left rank 4 completely
    unchanged while giving rank 3 a 206x speedup, and the rank-3 solver defect
    did not generalize either. Rank 4 also uses different tensor types and a
    different code path.

    Rank 4 is the strongest evidence the merge is sound: 2536 rewritten
    quadruples terms, 254 -> 69 operators, zero disagreements.
    """

    def _check(self, method, manifold, expect_ops):
        from ccgen.optimization.factorize import (
            manifold_operators, manifold_operators_with_plan,
            rewrite_term_factorized)

        eqs = generate_cc_equations(method, engine="diagram", canonical_fock=True)
        terms = eqs[manifold]
        unmerged = manifold_operators(terms, include_reuse=False)
        ops, plan = manifold_operators_with_plan(
            terms, include_reuse=False, spatial=True)
        self.assertEqual((len(unmerged), len(ops)), expect_ops)

        by_name = {o.name: o for o in ops}
        tensors = _tensors(NO, NV, 0, spatial=False)
        checked = 0
        for t in terms:
            r = rewrite_term_factorized(
                t, keep_operators=None, merge_plan_map=plan)
            if tuple(f.name for f in r.factors) == tuple(f.name for f in t.factors):
                continue
            checked += 1
            materialized = dict(tensors)
            for f in r.factors:
                if not f.name.startswith("W_") or f.name in materialized:
                    continue
                self.assertIn(f.name, by_name,
                              f"{f.name} referenced but not emitted")
                arr = _build_operator(by_name[f.name], tensors)
                self.assertIsNotNone(arr)
                materialized[f.name] = arr
            src, _ = _canon(t, residual_einsum(t, NO, NV, tensors=tensors))
            got, _ = _canon(r, residual_einsum(r, NO, NV, tensors=materialized))
            self.assertLessEqual(
                float(np.linalg.norm(src - got)),
                1e-10 * max(1.0, float(np.linalg.norm(src))),
                f"{method}/{manifold}: merged operator does not reproduce {t!r}")
        return checked

    def test_rank3_triples_merge_preserves_value(self):
        # (80, 39) at canonical_fock=True, which is what the emitter uses;
        # the default (canonical_fock=False) set gives (83, 42).
        self.assertGreater(self._check("ccsdt", "triples", (80, 39)), 300)

    def test_rank4_quadruples_merge_preserves_value(self):
        """Slow (~1 min) and deliberately kept: rank 4 is the case rank 3 has
        repeatedly failed to predict in this codebase."""
        self.assertGreater(self._check("ccsdtq", "quadruples", (254, 69)), 2000)


class SharedRepresentativeIsTheDefectTests(unittest.TestCase):
    """D6: localize the failure. Measured, not assumed -- this is what says the
    contraction TREES are sound and only the one-definition-per-NAME reuse is not."""

    def test_per_term_specs_are_exact(self):
        """Every term's OWN `identify_node` spec reproduces it. Green today;
        if it ever goes red the defect has moved into the tree itself and D6's
        framing is wrong."""
        terms = generate_cc_equations("ccsd")["doubles"]
        bad, total = _disagreements(terms, per_term=True)
        self.assertEqual(
            bad, [],
            f"{len(bad)}/{total} terms fail even with their OWN spec -- the "
            f"defect is not (only) the shared representative:\n"
            + "\n".join(f"  {why}: {t!r}" for t, why in bad[:5]))

    def test_shared_and_per_term_now_agree(self):
        """D6 FIXED: the shared representative is as good as each term's own spec.

        Before the fix this asserted the opposite (shared strictly worse: 21 vs
        0 on GCC, 39 vs 0 on spatial) -- that WAS the diagnosis. Folding the
        contraction shape into the operator name makes one name denote exactly
        one definition, so the two routes coincide. Kept in this direction so a
        regression that reintroduces name collisions fails here as well as in
        the value gates above."""
        for label, terms in (
            ("GCC", generate_cc_equations("ccsd")["doubles"]),
            ("spatial", __import__(
                "ccgen.spin", fromlist=["spin_adapt_equations"]
             ).spin_adapt_equations(generate_cc_equations("ccsd"))["doubles"]),
        ):
            sp = label == "spatial"
            shared_bad, _ = _disagreements(terms, spatial=sp)
            per_term_bad, _ = _disagreements(terms, per_term=True, spatial=sp)
            self.assertEqual(
                len(shared_bad), len(per_term_bad),
                f"{label}: shared-representative binding ({len(shared_bad)}) "
                f"differs from per-term ({len(per_term_bad)}) -- an operator "
                f"name denotes more than one definition again")

    def test_one_name_one_contraction_shape(self):
        """The invariant the D6 fix establishes, asserted directly rather than
        through its numeric consequence: across a manifold, every operator name
        carries exactly one contraction SHAPE.

        Shape, not literal indices. `(i,j,k,a)` and `(i,j,k,b)` are the same
        operator called at two sites — that sharing is the point of the
        factorizer and the fix deliberately keeps it. What must not recur is two
        different CONTRACTIONS under one name, which is what silently emitted a
        single `build_W` for both."""
        from collections import defaultdict

        from ccgen.spin import spin_adapt_equations

        for label, eqs in (("GCC", generate_cc_equations("ccsd")),
                           ("spatial", spin_adapt_equations(
                               generate_cc_equations("ccsd")))):
            for manifold, terms in eqs.items():
                seen = defaultdict(set)
                for t in terms:
                    for _nt, r in emittable_operators(t, seeded_fingerprints()):
                        if not isinstance(r, Derived):
                            continue
                        seen[r.spec.name].add(
                            _contraction_shape(r.spec.definition_terms[0]))
                clashes = {k: v for k, v in seen.items() if len(v) > 1}
                self.assertFalse(
                    clashes,
                    f"{label}/{manifold}: {len(clashes)} operator name(s) carry "
                    f"multiple contraction shapes, e.g. "
                    f"{next(iter(clashes)) if clashes else ''}")

    def test_one_emitted_name_one_canonical_shape_when_merged(self):
        """O4.4: the same invariant, restated for the MERGED path.

        Under merging the un-merged form above no longer describes what is
        wanted: an emitted name deliberately covers several RAW shapes — that is
        the merge (measured: 8 names on GCC, 21 on spatial). What must still
        hold, and what actually guards correctness, is one name per CANONICAL
        shape: two contractions that are NOT transpose-equivalent must never
        share a `build_W`.

        This is additional coverage, not a replacement. The un-merged gate above
        still applies to the default path, which is what most callers use.
        """
        from collections import defaultdict

        from ccgen.optimization.factorize import manifold_operators_with_plan
        from ccgen.optimization.operator_identity import canonical_shape
        from ccgen.spin import spin_adapt_equations

        for label, eqs, sp in (
            ("GCC", generate_cc_equations("ccsd", canonical_fock=True), False),
            ("spatial", spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=True)), True),
        ):
            for manifold, terms in eqs.items():
                if manifold in ("energy", "reference"):
                    continue
                _ops, plan = manifold_operators_with_plan(
                    terms, include_reuse=False, spatial=sp)
                raw = defaultdict(set)
                canon = defaultdict(set)
                for t in terms:
                    for _nt, r in emittable_operators(t, seeded_fingerprints()):
                        if not isinstance(r, Derived):
                            continue
                        rep = plan[r.spec.name][0]
                        raw[rep].add(_contraction_shape(r.spec.definition_terms[0]))
                        canon[rep].add(canonical_shape(r.spec, sp))
                clashes = {k: v for k, v in canon.items() if len(v) > 1}
                with self.subTest(label=label, manifold=manifold):
                    self.assertFalse(
                        clashes,
                        f"{label}/{manifold}: {len(clashes)} emitted name(s) "
                        f"cover more than one CANONICAL shape — non-equivalent "
                        f"contractions sharing a build_W")

            # anti-vacuity: if no name covered several raw shapes, merging did
            # nothing here and the assertion above is the un-merged one again.
            merged_names = sum(1 for v in raw.values() if len(v) > 1)
            self.assertGreater(
                merged_names, 0,
                f"{label}: no emitted name covers multiple raw shapes — "
                f"nothing merged, so this gate proves nothing")


if __name__ == "__main__":
    unittest.main()
