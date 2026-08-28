"""D4.4: every emitted `build_W_*` must compute the tensor its OWN spec declares.

This is the gate the derivation route never had. Every other gate in this suite
validates Python objects — the rewrite, the specs, the operator reuse, the
per-term algebra — and all of them pass while the emitted C++ computes a
different tensor. The gap is the emitter's RENDERING, and only reading the
emitted text can close it.

The defect this was written against (D4, 2026-08-26): 8 of 65 derived-operator
builders on spatial `ccsdt` read the wrong ERI block. Example:

    spec def : t1(c,j) v(i,c,a,k)          <ic|ak>  = ovvo(i,c,a,k)
    emitted  : -t1({j,c}) * mo_blocks.ovov(i, c, k, a)

i.e. the emitter applied `<ic|ak> = -<ic|ka>`, which is the ANTISYMMETRY
relation. It holds for antisymmetrized `<pq||rs>` and is FALSE for the spatial,
non-antisymmetrized integrals these kernels index. `ovov` and `ovvo` differ by
3.9e-01 on the fixture, and the relative error reaches 8.8 -- larger than the
quantity being computed.

`planck_tensor_cpp.py` carries a comment naming this exact pair as FIXED (the
"R4" note); that fix covers the exchange-partner read, not the derived-operator
builder path. Hence a gate rather than trust.
"""
from __future__ import annotations

import unittest

import numpy as np


NO, NV = 3, 4


def _spatial_tensors(no, nv):
    """A spatial (NON-antisymmetrized) fixture.

    Deliberately not `random_tensors`, which antisymmetrizes `v` -- under an
    antisymmetric `v` the very relation this gate exists to catch becomes TRUE,
    and the gate would pass vacuously. That trap is the reason this helper is
    written out rather than reused.
    """
    rng = np.random.default_rng(20260827)
    nmo = no + nv
    v = rng.standard_normal((nmo, nmo, nmo, nmo)) * 0.1
    # Only the symmetries a real spatial ERI has: <pq|rs> = <qp|sr> = <rs|pq>.
    v = v + v.transpose(1, 0, 3, 2)
    v = v + v.transpose(2, 3, 0, 1)
    o, w = slice(0, no), slice(no, nmo)
    return {
        "t1": rng.standard_normal((nv, no)) * 0.05,
        "t2": rng.standard_normal((nv, nv, no, no)) * 0.05,
        "t3": rng.standard_normal((nv, nv, nv, no, no, no)) * 0.02,
        "v": v,
        "f": np.diag(rng.standard_normal(nmo)),
        "_blocks": {
            "oooo": v[o, o, o, o], "ooov": v[o, o, o, w],
            "oovv": v[o, o, w, w], "ovov": v[o, w, o, w],
            "ovvo": v[o, w, w, o], "ovvv": v[o, w, w, w],
            "vvvv": v[w, w, w, w], "vovv": v[w, o, w, w],
            "vvov": v[w, w, o, w], "vvvo": v[w, w, w, o],
            "oovo": v[o, o, w, o], "ovoo": v[o, w, o, o],
            "vooo": v[w, o, o, o], "voov": v[w, o, o, w],
            "vovo": v[w, o, w, o], "vvoo": v[w, w, o, o],
        },
    }


class EmittedBuilderMatchesSpecTests(unittest.TestCase):
    def _emit_and_check(self, method):
        from ccgen.generate import generate_cc_equations, print_cpp_planck
        from ccgen.spin import spin_adapt_equations
        from ccgen.optimization.factorize import (
            manifold_operators, select_operators_by_savings)
        from ccgen.tests import test_factorize_value_preservation as G
        from ccgen.tests.emitted_cpp_eval import build_emitted_operators

        src = print_cpp_planck(method, dressing="derived", spin_adapt=True,
                               force_arbitrary=True)

        tensors = _spatial_tensors(NO, NV)
        blocks = tensors.pop("_blocks")
        fock = tensors["f"]
        o, w = slice(0, NO), slice(NO, NO + NV)
        env = dict(blocks)
        env.update({"f_oo": fock[o, o], "f_ov": fock[o, w], "f_vv": fock[w, w]})
        # C++ amplitude layout is (occ..., virt...); ccgen's is (vir..., occ...).
        env["t1"] = tensors["t1"].T
        env["t2"] = np.transpose(tensors["t2"], (2, 3, 0, 1))
        env["t3"] = np.transpose(tensors["t3"], (3, 4, 5, 0, 1, 2))

        emitted = build_emitted_operators(src, method, NO, NV, env.get)

        eqs = spin_adapt_equations(
            generate_cc_equations(method, canonical_fock=True))
        specs = {}
        for manifold, terms in eqs.items():
            if manifold in ("energy", "reference"):
                continue
            for spec in manifold_operators(terms, include_reuse=False):
                specs[spec.name] = spec

        G.NO, G.NV = NO, NV
        bad, checked = [], 0
        for name, arr in sorted(emitted.items()):
            spec = specs.get(name)
            if spec is None:
                continue
            want = G._build_operator(spec, tensors)
            if want is None or want.shape != arr.shape:
                bad.append((name, "shape/slot mismatch"))
                continue
            checked += 1
            scale = max(float(np.abs(want).max()), 1.0)
            d = float(np.abs(arr - want).max())
            if d > 1e-10 * scale:
                bad.append((name, f"rel={d / scale:.2e}  def={spec.definition_terms[0]!r}"))
        return checked, bad

    def test_the_gate_is_not_vacuous(self):
        """The fixture must have spatial symmetry but NOT antisymmetry.

        `random_tensors` antisymmetrizes `v`; under an antisymmetric `v` the
        emitter's `<ic|ak> = -<ic|ka>` is TRUE and this gate passes while the
        defect is present. Measured: 0/288 builders disagree on an
        antisymmetrized fixture, 41/288 on this one -- so the failures are
        specifically the antisymmetry misuse, not a parser or layout artifact.
        """
        t = _spatial_tensors(NO, NV)
        v, blocks = t["v"], t["_blocks"]
        self.assertLess(float(np.abs(v - v.transpose(1, 0, 3, 2)).max()), 1e-12)
        self.assertLess(float(np.abs(v - v.transpose(2, 3, 0, 1)).max()), 1e-12)
        self.assertGreater(float(np.abs(v + v.transpose(0, 1, 3, 2)).max()), 1e-3,
                           "fixture is antisymmetric -- the gate would be vacuous")
        self.assertGreater(
            float(np.abs(blocks["ovov"].transpose(0, 1, 3, 2) - blocks["ovvo"]).max()),
            1e-3, "ovov and ovvo coincide -- the gate cannot see the defect")

    def test_ccsdt_derived_builders_compute_their_specs(self):
        try:
            checked, bad = self._emit_and_check("ccsdt")
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")
        self.assertGreater(checked, 0, "no builders were checked -- vacuous")
        self.assertEqual(
            bad, [],
            f"{len(bad)}/{checked} emitted builders do not compute their own "
            f"spec:\n" + "\n".join(f"  {n}: {why}" for n, why in bad[:10]))


if __name__ == "__main__":
    unittest.main()
