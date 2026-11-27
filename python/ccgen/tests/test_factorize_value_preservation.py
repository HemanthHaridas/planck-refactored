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
    that is the open question in docs/CCGEN_RED_TESTS.md -- but it does say
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
             False, (38, 28)),  # 2026-11-21: counts rose when the emit path stopped matching the retired route's seeded operators (EMIT_FINGERPRINTS, docs/CCGEN_GENERATED_CC_COST.md); those sub-contractions are now derived operators.
            ("spatial", spin_adapt_equations(
                generate_cc_equations("ccsd", canonical_fock=True))["doubles"],
             True, (81, 45)),
        ):
            with self.subTest(label):
                self.assertEqual(self._check(label, terms, sp), expect)

    def test_default_path_does_not_merge(self):
        """Merging is opt-in. A caller that does not ask for it must see the
        un-merged set, because it will not be permuting its call sites."""
        from ccgen.optimization.factorize import manifold_operators
        terms = generate_cc_equations("ccsd", canonical_fock=True)["doubles"]
        # 2026-11-21: counts rose when the emit path stopped matching the retired route's seeded operators (EMIT_FINGERPRINTS, docs/CCGEN_GENERATED_CC_COST.md); those sub-contractions are now derived operators.
        self.assertEqual(len(manifold_operators(terms, include_reuse=False)), 38)


class HigherRankMergeTests(unittest.TestCase):
    """O5: the merge holds at rank 3 and rank 4, not just `ccsd`.

    Asserted separately because this codebase has twice established that rank 3
    is not a proxy for rank 4 — the tensor-accessor fix left rank 4 completely
    unchanged while giving rank 3 a 206x speedup, and the rank-3 solver defect
    did not generalize either. Rank 4 also uses different tensor types and a
    different code path.

    Rank 4 is the strongest evidence the merge is sound: 2536 rewritten
    quadruples terms, 268 -> 79 operators (254 -> 69 before the 2026-11-21
    seeded-operator fix), zero disagreements.
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
        # canonical_fock=True, which is what the emitter uses. Was (80, 39).
        # 2026-11-21: counts rose when the emit path stopped matching the retired route's seeded operators (EMIT_FINGERPRINTS, docs/CCGEN_GENERATED_CC_COST.md); those sub-contractions are now derived operators.
        self.assertGreater(self._check("ccsdt", "triples", (94, 49)), 300)

    def test_rank4_quadruples_merge_preserves_value(self):
        """Slow (~1 min) and deliberately kept: rank 4 is the case rank 3 has
        repeatedly failed to predict in this codebase."""
        self.assertGreater(self._check("ccsdtq", "quadruples", (268, 79)), 2000)


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


# --- UCC -------------------------------------------------------------------
#
# O6 named this gate as the precondition for trusting the derived route on
# spin-blocked input. Every orbital count differs (noa != nva != nob != nvb),
# so a slot read with the wrong spin or space raises or mismatches instead of
# silently staying in bounds -- the same reason the GCC fixture is 3x4.
UCC_DIMS = {"noa": 3, "nva": 5, "nob": 4, "nvb": 6}


def _ucc_eval(term, tensors, operators):
    """`ucc_residual_einsum` with derived operators resolved by name.

    Operators are stored in their declared slot order and read directly;
    everything else goes through `ucc_resolve_factor`, which slices `v`/`f`
    by each slot's (space, spin)."""
    import string

    from ccgen.tests.residual_eval import ucc_resolve_factor

    pool = iter(string.ascii_letters)
    letters = {}
    for idx in list(term.free_indices) + list(term.summed_indices):
        letters[idx] = next(pool)
    subs, ops = [], []
    for f in term.factors:
        subs.append("".join(letters[i] for i in f.indices))
        ops.append(operators[f.name] if f.name in operators
                   else ucc_resolve_factor(f, tensors, UCC_DIMS))
    ext = ([i for i in term.free_indices if i.space == "vir"]
           + [i for i in term.free_indices if i.space == "occ"])
    out = np.einsum(",".join(subs) + "->" + "".join(letters[i] for i in ext),
                    *ops, optimize=True)
    return out * float(term.coeff)


def _ucc_build_operators(specs, tensors):
    built = {}
    for spec in specs:          # dependency-ordered, like the emitter's builders
        arr = None
        for dt in spec.definition_terms:
            part = _ucc_eval(dt, tensors, built)
            ext = ([i for i in dt.free_indices if i.space == "vir"]
                   + [i for i in dt.free_indices if i.space == "occ"])
            pos = {idx: k for k, idx in enumerate(ext)}
            assert set(pos) == set(spec.indices), spec.name
            part = np.transpose(part, [pos[i] for i in spec.indices])
            arr = part if arr is None else arr + part
        built[spec.name] = arr
    return built


def _ucc_disagreements(method, mutate=None):
    """(bad, rewritten, total) for the production UCC composition:
    `ucc_adapt_equations` -> `factorize_equations(spatial=False,
    merge_transposes=True)`, exactly what `print_cpp_planck` runs."""
    from ccgen.optimization.factorize import factorize_equations
    from ccgen.spin import ucc_adapt_equations
    from ccgen.tests.residual_eval import ucc_random_tensors

    eqs = ucc_adapt_equations(
        generate_cc_equations(method, engine="diagram", canonical_fock=True))
    rewritten, specs = factorize_equations(
        eqs, spatial=False, merge_transposes=True)
    specs = list(specs or [])
    if mutate is not None:
        specs = mutate(specs)
    tensors = ucc_random_tensors(UCC_DIMS["noa"], UCC_DIMS["nva"],
                                 UCC_DIMS["nob"], UCC_DIMS["nvb"], seed=0)
    built = _ucc_build_operators(specs, tensors)
    bad, n_rewritten, total = [], 0, 0
    for target, terms in eqs.items():
        assert len(terms) == len(rewritten[target]), target
        for t, r in zip(terms, rewritten[target]):
            total += 1
            if any(f.name in built for f in r.factors):
                n_rewritten += 1
            src, ksrc = _canon(t, _ucc_eval(t, tensors, {}))
            got, kgot = _canon(r, _ucc_eval(r, tensors, built))
            if ksrc != kgot or src.shape != got.shape:
                bad.append((target, t, f"layout {kgot} != {ksrc}"))
                continue
            d = float(np.linalg.norm(src - got))
            if d > 1e-10 * max(1.0, float(np.linalg.norm(src))):
                bad.append((target, t, f"||diff|| = {d:.3e}"))
    return bad, n_rewritten, total


class UCCRewritePreservesValueTests(unittest.TestCase):
    """The derived route on spin-blocked UCC terms, with transpose merging on.

    UCC factor names carry their spin block (`v_abab`), which
    `tensor_symmetries` does not list, so merging treats them as opaque: it
    can miss a merge, never make one across blocks. This gate is what checks
    that claim, and the rest of the rewrite, by value."""

    def _assert_clean(self, method, min_rewritten):
        bad, n_rewritten, total = _ucc_disagreements(method)
        self.assertGreater(n_rewritten, min_rewritten,
                           f"{method}: too few rewritten terms -- gate is vacuous")
        self.assertEqual(
            bad, [],
            f"UCC {method}: {len(bad)}/{total} terms do not reproduce:\n"
            + "\n".join(f"  {tg} {why}: {t!r}" for tg, t, why in bad[:5]))

    def test_ccsd_rewrite_is_value_preserving(self):
        self._assert_clean("ccsd", 200)

    def test_ccsdt_rewrite_is_value_preserving(self):
        self._assert_clean("ccsdt", 1500)

    def test_gate_catches_a_transposed_operator(self):
        """Mutation control: swap two same-spin, same-space slots of one
        operator's declared order, so every shape stays legal and only the
        values move. The gate must report it as a disagreement."""
        from ccgen.spin import ucc_term_index_spins

        def swap_one(specs):
            for k, s in enumerate(specs):
                idx = list(s.indices)
                spin = ucc_term_index_spins(s.definition_terms[0])
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        if (idx[a].space == idx[b].space
                                and spin[idx[a].name] == spin[idx[b].name]):
                            idx[a], idx[b] = idx[b], idx[a]
                            import dataclasses
                            specs = list(specs)
                            specs[k] = dataclasses.replace(s, indices=tuple(idx))
                            return specs
            raise AssertionError("no operator with two same-space slots")
        bad, _, _ = _ucc_disagreements("ccsd", mutate=swap_one)
        self.assertGreater(len(bad), 0, "a transposed operator went unnoticed")


def _ucc_block(terms, tensors, operators):
    """Sum of `terms` in one layout, keyed by free-index name (sorted)."""
    total, key = None, None
    for t in terms:
        arr, k = _canon(t, _ucc_eval(t, tensors, operators))
        assert key in (None, k), (key, k)
        key = k
        total = arr if total is None else total + arr
    return total, key


def _apply_group_element(arr, names, mapping):
    """g(R)[names] = R[mapping(names)], the C++ symmetrize pass's reading."""
    src = [mapping.get(n, n) for n in names]
    return np.transpose(arr, [names.index(s) for s in src])


def _ucc_fold_disagreements(method, corrupt=None, merge=False, drop=None):
    """(bad, folded_targets, terms_before, terms_after) for the production UCC
    composition with the orbit fold: adapt -> fold -> factorize -> symmetrize,
    compared per block against the unfolded residual.

    ``merge`` (U3b) adds the operand merges `print_cpp_planck` runs after the
    factorizer (`merge_operand_reads`), and then checks every block, folded
    or not, since a merge touches all of them; ``drop`` names a block whose
    first term is removed after merging (mutation control)."""
    from ccgen.optimization.factorize import factorize_equations, merge_operand_reads
    from ccgen.optimization.permutation import (
        fold_external_symmetry, spin_class_group)
    from ccgen.spin import ucc_adapt_equations
    from ccgen.tests.residual_eval import ucc_random_tensors

    eqs = ucc_adapt_equations(
        generate_cc_equations(method, engine="diagram", canonical_fock=True))
    folded, groups = dict(eqs), {}
    for target, terms in eqs.items():
        if terms and terms[0].free_indices:
            g = spin_class_group(terms[0])
            if len(g) > 1:
                folded[target] = fold_external_symmetry(terms, g)
                groups[target] = list(g)
    if corrupt is not None:
        groups = corrupt(groups)
    rewritten, specs = factorize_equations(
        folded, spatial=False, merge_transposes=True)
    if merge:
        rewritten, specs = merge_operand_reads(rewritten, specs, ucc=True)
        groups = {t: groups.get(t, [({}, 1)]) for t in eqs
                  if t != "energy" and eqs[t]}
    if drop is not None:
        rewritten = dict(rewritten)
        rewritten[drop] = rewritten[drop][1:]
    tensors = ucc_random_tensors(UCC_DIMS["noa"], UCC_DIMS["nva"],
                                 UCC_DIMS["nob"], UCC_DIMS["nvb"], seed=0)
    built = _ucc_build_operators(list(specs or []), tensors)
    bad = []
    for target, terms in eqs.items():
        if target not in groups:
            continue
        ref, names = _ucc_block(terms, tensors, {})
        x, xnames = _ucc_block(rewritten[target], tensors, built)
        assert names == xnames, (target, names, xnames)
        got = sum(s * _apply_group_element(x, list(names), m)
                  for m, s in groups[target])
        d = float(np.linalg.norm(got - ref))
        if d > 1e-10 * max(1.0, float(np.linalg.norm(ref))):
            bad.append((target, f"||diff|| = {d:.3e}"))
    before = sum(len(eqs[t]) for t in groups)
    after = sum(len(rewritten[t]) for t in groups)
    return bad, sorted(groups), before, after


class UCCMergeTests(unittest.TestCase):
    """U3b: the operand merges (R2, R2b) on UCC, composed exactly as
    `print_cpp_planck` runs them (adapt -> fold -> factorize -> merge), every
    block against the plain residual (docs/CCSD_T_SIZE_SCOPE.md)."""

    def test_ccsd_merge_is_value_preserving(self):
        bad, targets, before, after = _ucc_fold_disagreements("ccsd", merge=True)
        self.assertEqual(len(targets), 5)
        self.assertEqual(bad, [])

    def test_ccsdt_merge_is_value_preserving(self):
        bad, targets, _, _ = _ucc_fold_disagreements("ccsdt", merge=True)
        self.assertEqual(len(targets), 9)
        self.assertEqual(bad, [])

    def test_gate_catches_a_dropped_merged_term(self):
        bad, _, _, _ = _ucc_fold_disagreements("ccsd", merge=True, drop="doubles_abab")
        self.assertEqual([t for t, _ in bad], ["doubles_abab"])


class UCCOrbitFoldTests(unittest.TestCase):
    """The UCC external-symmetry fold (docs/CCGEN_UCC_ORBIT_FOLD.md),
    composed with the derived operators exactly as `print_cpp_planck` runs it.
    The fixture's amplitudes are antisymmetric within each same-spin class,
    which is the property the fold's exactness rests on."""

    def test_group_orders_follow_spin_classes(self):
        from ccgen.optimization.permutation import spin_class_group
        from ccgen.spin import ucc_adapt_equations
        eqs = ucc_adapt_equations(generate_cc_equations(
            "ccsdt", engine="diagram", canonical_fock=True))
        orders = {t: len(spin_class_group(eqs[t][0]))
                  for t in eqs if eqs[t] and eqs[t][0].free_indices}
        self.assertEqual(orders, {
            "singles_aa": 1, "singles_bb": 1,
            "doubles_aaaa": 4, "doubles_abab": 1, "doubles_bbbb": 4,
            "triples_aaaaaa": 36, "triples_aabaab": 4,
            "triples_abbabb": 4, "triples_bbbbbb": 36})

    def test_ccsd_fold_is_value_preserving(self):
        bad, targets, before, after = _ucc_fold_disagreements("ccsd")
        self.assertEqual(targets, ["doubles_aaaa", "doubles_bbbb"])
        self.assertLess(after, before)
        self.assertEqual(bad, [])

    def test_ccsdt_fold_is_value_preserving(self):
        bad, targets, before, after = _ucc_fold_disagreements("ccsdt")
        self.assertEqual(len(targets), 6)
        self.assertLess(after, before / 2)
        self.assertEqual(bad, [])

    def test_gate_catches_a_wrong_sign(self):
        """Mutation control: flip the sign of one odd element of the
        order-36 group. The block must go red."""
        def flip(groups):
            g = list(groups["triples_aaaaaa"])
            k = next(i for i, (_m, s) in enumerate(g) if s < 0)
            g[k] = (g[k][0], 1)
            return {**groups, "triples_aaaaaa": g}
        bad, _, _, _ = _ucc_fold_disagreements("ccsdt", corrupt=flip)
        self.assertEqual([t for t, _ in bad], ["triples_aaaaaa"])
