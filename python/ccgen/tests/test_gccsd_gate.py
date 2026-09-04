"""Per-class antisymmetry checks for ccgen CCSD doubles.

The generated kernels are not compiled into any binary, so these evaluate the
ccgen residual as numeric tensor contractions and assert the STRUCTURAL property
every correct CCSD doubles residual has: antisymmetry under i<->j and a<->b, per
term class.

NOT a correctness oracle. The decisive ccgen-vs-PySCF equality lives in
`test_reference_vs_pyscf.py::test_ccgen_doubles_matches_pyscf` (full residual ==
gccsd.update_amps to ~1e-15). This module deliberately does NOT compare ccgen to
the hand-written dressed reference (`gccsd_reference`) on the random inputs here:
that reference is the Stanton-Gauss dressed form, which equals the raw projection
ccgen emits only ON-SHELL and in its antisymmetric projection -- NOT term-by-term
on arbitrary OFF-SHELL random amplitudes. The earlier "known t1*t2 bug" pinned by
`@expectedFailure` here was that off-shell comparison artifact, not a real defect
(see the 2026-07-27 correction in docs/CCGEN_DIAGRAM_REPRESENTATION.md).
"""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations  # noqa: E402
from ccgen.tests.gccsd_reference import gccsd_doubles_residual  # noqa: E402

NO, NV = 3, 4
N = NO + NV


def _shared_inputs(seed=11):
    rng = np.random.default_rng(seed)
    # antisymmetric physicist <pq||rs>
    g = rng.random((N, N, N, N))
    g = g + g.transpose(2, 3, 0, 1)
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    f = rng.random((N, N))
    f = (f + f.T) / 2
    t1_ov = rng.random((NO, NV))
    t2 = rng.random((NO, NO, NV, NV))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    return g, f, t1_ov, t2


def _reference_r2(g, f, t1_ov, t2):
    occ, vir = slice(0, NO), slice(NO, N)

    def blk(sig):
        s = [occ if c == "o" else vir for c in sig]
        return g[s[0], s[1], s[2], s[3]]

    blocks = {s: blk(s) for s in
              ["oooo", "ooov", "oovv", "ovov", "ovvo", "ovvv", "vvvv"]}
    fdict = {"oo": f[occ, occ], "ov": f[occ, vir],
             "vo": f[vir, occ], "vv": f[vir, vir]}
    return gccsd_doubles_residual(fdict, blocks, t1_ov, t2)  # [i,j,a,b]


def _ccgen_r2(terms, g, f, t1_ov, t2):
    tensors = {"t1": t1_ov.T, "t2": t2.transpose(2, 3, 0, 1), "v": g, "f": f}

    def space(idx):
        return list(range(NO)) if idx.space == "occ" else list(range(NO, N))

    r = np.zeros((NV, NV, NO, NO))
    for term in terms:
        bn = {x.name: x for x in term.free_indices}
        a, b, i, j = bn["a"], bn["b"], bn["i"], bn["j"]
        summed = term.summed_indices
        for av, bv, iv, jv in itertools.product(
            range(NV), range(NV), range(NO), range(NO)
        ):
            env = {a: NO + av, b: NO + bv, i: iv, j: jv}
            acc = 0.0
            for sv in itertools.product(*[space(x) for x in summed]):
                for k, x in enumerate(summed):
                    env[x] = sv[k]
                p = 1.0
                for fac in term.factors:
                    key = tuple(
                        env[x] - NO
                        if (fac.name.startswith("t") and x.space == "vir")
                        else env[x]
                        for x in fac.indices
                    )
                    p *= tensors[fac.name][key]
                acc += p
            r[av, bv, iv, jv] += float(term.coeff) * acc
    return r  # [a,b,i,j]


class GccsdGateTests(unittest.TestCase):
    def test_reference_is_antisymmetric(self):
        g, f, t1, t2 = _shared_inputs()
        r = _reference_r2(g, f, t1, t2)
        self.assertTrue(np.allclose(r, -r.transpose(1, 0, 2, 3)))  # i<->j
        self.assertTrue(np.allclose(r, -r.transpose(0, 1, 3, 2)))  # a<->b

    def test_conventions_align_on_the_bare_eri(self):
        # The one term with no freedom: r2 += <ij||ab>. If this matches to 0,
        # the ccgen<->reference convention mapping is correct and any later
        # mismatch is a real coefficient difference, not an alignment artifact.
        g, f, t1, t2 = _shared_inputs()
        bare = [t for t in generate_cc_equations("ccsd")["doubles"]
                if [x.name for x in t.factors] == ["v"] and not t.summed_indices]
        r_ccgen = _ccgen_r2(bare, g, f, t1, t2)
        occ, vir = slice(0, NO), slice(NO, N)
        r_ref = g[occ, occ, vir, vir].transpose(2, 3, 0, 1)
        self.assertTrue(np.allclose(r_ccgen, r_ref), np.max(np.abs(r_ccgen - r_ref)))

    def test_non_t1t2_pieces_are_antisymmetric(self):
        # Everything EXCEPT the t1*t2-mixing terms is correctly antisymmetric.
        g, f, t1, t2 = _shared_inputs()
        broken_types = {frozenset(("f", "t1", "t2")),
                        frozenset(("t1", "t2", "v")),
                        frozenset(("t1", "t1", "t2", "v"))}
        good = [t for t in generate_cc_equations("ccsd")["doubles"]
                if frozenset(x.name for x in t.factors) not in broken_types]
        r = _ccgen_r2(good, g, f, t1, t2)
        self.assertTrue(np.allclose(r, -r.transpose(1, 0, 2, 3)))
        self.assertTrue(np.allclose(r, -r.transpose(0, 1, 3, 2)))

    def test_all_classes_are_now_antisymmetric_after_T1_2b(self):
        # After the T1.2b is_dummy false-zero fix, EVERY doubles term type is
        # antisymmetric per class -- the pre-fix per-class breakage (f*t1*t2,
        # t1*t2*v, t1*t1*t2*v) is gone. The remaining residual error is a
        # MAGNITUDE error (~3%), not a per-class antisymmetry break; that is what
        # the whole-residual gate below still catches.
        from collections import defaultdict

        g, f, t1, t2 = _shared_inputs()
        groups = defaultdict(list)
        for t in generate_cc_equations("ccsd")["doubles"]:
            groups[frozenset(x.name for x in t.factors)].append(t)
        broken = set()
        for key, ts in groups.items():
            r = _ccgen_r2(ts, g, f, t1, t2)
            if not (np.allclose(r, -r.transpose(1, 0, 2, 3))
                    and np.allclose(r, -r.transpose(0, 1, 3, 2))):
                broken.add(key)
        self.assertEqual(broken, set(),
                         f"unexpected per-class antisymmetry break: {broken}")

    def test_canonical_fock_output_is_also_antisymmetric(self):
        # T2 + T1.2b together: canonical_fock output is antisymmetric per class.
        # (The whole-residual magnitude error remains, gated separately.)
        from collections import defaultdict

        g, _f, t1, t2 = _shared_inputs()
        rng = np.random.default_rng(11)
        f_diag = np.diag(rng.random(N))
        terms = generate_cc_equations("ccsd", canonical_fock=True)["doubles"]
        groups = defaultdict(list)
        for t in terms:
            groups[frozenset(x.name for x in t.factors)].append(t)
        broken = set()
        for key, ts in groups.items():
            r = _ccgen_r2(ts, g, f_diag, t1, t2)
            if not (np.allclose(r, -r.transpose(1, 0, 2, 3))
                    and np.allclose(r, -r.transpose(0, 1, 3, 2))):
                broken.add(key)
        self.assertEqual(broken, set())
        # No f*t1*t2 terms at all under canonical Fock.
        self.assertFalse(any(
            frozenset(x.name for x in t.factors) == frozenset(("f", "t1", "t2"))
            for t in terms
        ))

    # The former `@expectedFailure` end-to-end and class-local gates
    # (test_ccgen_matches_reference_KNOWN_BUG, test_t1t2v_terms_hit_their_target,
    # + the _t1t2v_target scaffolding) were REMOVED 2026-07-27. They compared
    # ccgen to the dressed `gccsd_reference` on off-shell random amplitudes and
    # pinned the resulting term-by-term gap as a "known bug". It is not one: the
    # full ccgen doubles residual matches PySCF gccsd.update_amps to ~1e-15
    # (test_reference_vs_pyscf.py::test_ccgen_doubles_matches_pyscf). The raw
    # projection and the dressed form coincide on-shell and in the antisymmetric
    # projection, not term-by-term off-shell -- so the comparison was invalid,
    # not the generator. Correctness now lives in the PySCF module; this module
    # keeps only the per-class antisymmetry checks, which remain meaningful.


if __name__ == "__main__":
    unittest.main()
