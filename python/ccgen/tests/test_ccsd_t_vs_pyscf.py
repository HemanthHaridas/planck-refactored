"""CCSD(T) step-1 gate (docs/CCSD_T_SCOPE.md): the generated W and V triples
projections, evaluated at PySCF's converged GCCSD amplitudes, must give PySCF's
spin-orbital (T) energy, E(T) = (1/36) sum (W + V) W / D.

Run with the PySCF interpreter:

    tests/pyscf/.venv/bin/python -m unittest ccgen.tests.test_ccsd_t_vs_pyscf
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pyscf import gto, scf
    from pyscf.cc import gccsd, gccsd_t
    _HAVE_PYSCF = True
except ImportError:  # pragma: no cover - depends on the pyscf venv
    _HAVE_PYSCF = False

from ccgen.methods.ccsd_t import build_ccsd_t_equations  # noqa: E402
from ccgen.tests.residual_eval import residual_einsum  # noqa: E402

WATER = "O 0 0 0.117176; H 0 0.756950 -0.468703; H 0 -0.756950 -0.468703"
H2O2_C1 = ("O 0.000 0.717 -0.041; O 0.021 -0.717 0.032; "
           "H 0.851 0.887 0.463; H -0.802 -0.911 0.395")
OH = "O 0 0 0; H 0 0 0.97"


def _triples(terms, no, nv, tensors):
    return sum(residual_einsum(t, no, nv, tensors=tensors) for t in terms)


def ccgen_and_pyscf_t(atom, basis, spin=0, charge=0):
    from ccgen.tests.test_reference_vs_pyscf import _full_antisym_v

    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = scf.GHF(mol).run(conv_tol=1e-12)
    cc = gccsd.GCCSD(mf)
    cc.conv_tol = 1e-11
    cc.conv_tol_normt = 1e-9
    cc.kernel()
    eris = cc.ao2mo()
    no, nmo = cc.nocc, cc.nmo
    nv = nmo - no
    tensors = {
        "t1": cc.t1.T, "t2": cc.t2.transpose(2, 3, 0, 1),
        "v": _full_antisym_v(eris, nmo, no), "f": np.asarray(eris.fock),
    }
    eqs = build_ccsd_t_equations()
    w = _triples(eqs["W"], no, nv, tensors)  # [a,b,c,i,j,k]
    v = _triples(eqs["V"], no, nv, tensors)
    e = np.asarray(eris.fock).diagonal()
    eo, ev = e[:no], e[no:]
    d = (eo[None, None, None, :, None, None] + eo[None, None, None, None, :, None]
         + eo[None, None, None, None, None, :] - ev[:, None, None, None, None, None]
         - ev[None, :, None, None, None, None] - ev[None, None, :, None, None, None])
    e_t = float(np.sum((w + v) * w / d)) / 36.0
    return e_t, float(gccsd_t.kernel(cc, eris, cc.t1, cc.t2, verbose=0))


def _denominator(eo, ev):
    return (eo[None, None, None, :, None, None] + eo[None, None, None, None, :, None]
            + eo[None, None, None, None, None, :] - ev[:, None, None, None, None, None]
            - ev[None, :, None, None, None, None] - ev[None, None, :, None, None, None])


def _half_antisymmetrize_virtuals(x):
    """1/2 sum_P sign(P) x[P(abc), ijk]: the closed-shell aaa block from aab."""
    out = np.zeros_like(x)
    for perm, sign in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
                       ((1, 0, 2), -1), ((0, 2, 1), -1), ((2, 1, 0), -1)):
        out += sign * x.transpose(*perm, 3, 4, 5)
    return 0.5 * out


def restricted_and_pyscf_t(atom, basis):
    """Closed-shell E(T) two ways from the restricted lowering, and PySCF's.

    ``direct``: W and V integrated on the aaa and aab blocks separately.
    ``one_block``: aab only, with aaa from the virtual antisymmetrizer -- the
    form the restricted C++ kernel will use."""
    from pyscf import ao2mo, cc
    from pyscf.cc import ccsd_t
    from ccgen.methods.ccsd_t import build_ccsd_t_restricted
    from ccgen.spin import _adapt_on_block

    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-11
    mycc.kernel()
    no, nmo = mycc.nocc, mycc.nmo
    nv = nmo - no
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), nmo)
    tensors = {"t1": mycc.t1.T, "t2": mycc.t2.transpose(2, 3, 0, 1),
               "v": eri.transpose(0, 2, 1, 3)}  # physicist <pq|rs>
    eris = mycc.ao2mo()
    # PySCF's (T) takes D from diag(eris.fock), not mf.mo_energy; the two differ
    # by the SCF residual (1.9e-11 in E(T) on C1 H2O2 6-31G).
    e = np.asarray(eris.fock).diagonal()
    d = _denominator(e[:no], e[no:])

    def s(w, v):
        return float(np.sum((w + v) * w / d))

    so = build_ccsd_t_equations()
    aaa = {k: _triples(_adapt_on_block(so[k], dict(zip("abcijk", "aaaaaa"))),
                       no, nv, tensors) for k in ("W", "V")}
    rs = build_ccsd_t_restricted()
    w, v = (_triples(rs[k], no, nv, tensors) for k in ("W", "V"))
    s_aab = s(w, v)
    direct = s(aaa["W"], aaa["V"]) / 18 + s_aab / 2
    one_block = s(_half_antisymmetrize_virtuals(w), _half_antisymmetrize_virtuals(v)) / 18 + s_aab / 2
    return direct, one_block, float(ccsd_t.kernel(mycc, eris, verbose=0))


@unittest.skipUnless(_HAVE_PYSCF, "PySCF not importable")
class RestrictedCCSDTriplesVsPySCF(unittest.TestCase):
    def check(self, atom, basis):
        direct, one_block, ref = restricted_and_pyscf_t(atom, basis)
        print(f"\n  {atom[:12]:12s} {basis:7s} direct {direct - ref:.1e} "
              f"one-block {one_block - ref:.1e} (pyscf {ref:.12f})")
        self.assertGreater(abs(ref), 1e-6)
        self.assertAlmostEqual(direct, ref, delta=1e-11)
        self.assertAlmostEqual(one_block, ref, delta=1e-11)

    def test_water_631g(self):
        self.check(WATER, "6-31g")

    def test_h2o2_c1_631g(self):
        self.check(H2O2_C1, "6-31g")


def unrestricted_and_pyscf_t(atom, basis, spin, charge=0):
    """Open-shell E(T) from the four-block unrestricted lowering, and PySCF's
    UCCSD(T), both at PySCF's converged UCCSD amplitudes."""
    from pyscf import ao2mo, cc
    from pyscf.cc import uccsd_t
    from ccgen.methods.ccsd_t import build_ccsd_t_unrestricted
    from ccgen.tests.residual_eval import ucc_residual_einsum

    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    # A tight orbital gradient matters here: at conv_tol 1e-12 alone, water+
    # missed PySCF by 2.0e-11 (the residual off-diagonal Fock); at 1e-10 in the
    # gradient the two agree to 1e-15.
    mf = scf.UHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-10
    mycc.kernel()
    eris = mycc.ao2mo()
    ca, cb = mf.mo_coeff
    noa, nob = mol.nelec
    nva, nvb = ca.shape[1] - noa, cb.shape[1] - nob

    def phys(c1, c2):
        g = ao2mo.general(mol, (c1, c1, c2, c2), compact=False).reshape(
            c1.shape[1], c1.shape[1], c2.shape[1], c2.shape[1])
        return g.transpose(0, 2, 1, 3)

    vaa, vbb = phys(ca, ca), phys(cb, cb)
    t1a, t1b = mycc.t1
    t2aa, t2ab, t2bb = mycc.t2
    tensors = {
        "t1_aa": t1a.T, "t1_bb": t1b.T,
        "t2_aaaa": t2aa.transpose(2, 3, 0, 1), "t2_abab": t2ab.transpose(2, 3, 0, 1),
        "t2_bbbb": t2bb.transpose(2, 3, 0, 1),
        "v_aaaa": vaa - vaa.transpose(0, 1, 3, 2), "v_bbbb": vbb - vbb.transpose(0, 1, 3, 2),
        "v_abab": phys(ca, cb),
    }
    dims = dict(noa=noa, nva=nva, nob=nob, nvb=nvb)
    # D from diag(Fock), as PySCF's (T) uses (see the restricted finding).
    eps = {"a": np.asarray(eris.focka).diagonal(), "b": np.asarray(eris.fockb).diagonal()}
    nocc = {"a": noa, "b": nob}

    blocks = build_ccsd_t_unrestricted()
    e_t = 0.0
    for tag, weight in (("aaaaaa", 1 / 36), ("bbbbbb", 1 / 36),
                        ("aabaab", 1 / 4), ("abbabb", 1 / 4)):
        w = sum(ucc_residual_einsum(t, dims, tensors) for t in blocks[f"W_{tag}"])
        v = sum(ucc_residual_einsum(t, dims, tensors) for t in blocks[f"V_{tag}"])
        ev = [eps[sp][nocc[sp]:] for sp in tag[:3]]
        eo = [eps[sp][:nocc[sp]] for sp in tag[3:]]
        d = (-ev[0][:, None, None, None, None, None] - ev[1][None, :, None, None, None, None]
             - ev[2][None, None, :, None, None, None] + eo[0][None, None, None, :, None, None]
             + eo[1][None, None, None, None, :, None] + eo[2][None, None, None, None, None, :])
        e_t += weight * float(np.sum((w + v) * w / d))
    return e_t, float(uccsd_t.kernel(mycc, eris, verbose=0))


@unittest.skipUnless(_HAVE_PYSCF, "PySCF not importable")
class UnrestrictedCCSDTriplesVsPySCF(unittest.TestCase):
    def check(self, atom, basis, spin, charge=0):
        ours, ref = unrestricted_and_pyscf_t(atom, basis, spin, charge)
        print(f"\n  {atom[:12]:12s} {basis:7s} ours {ours:.12f} pyscf {ref:.12f} "
              f"diff {ours - ref:.1e}")
        self.assertGreater(abs(ref), 1e-6)
        self.assertAlmostEqual(ours, ref, delta=1e-11)

    def test_oh_doublet_631g(self):
        self.check(OH, "6-31g", spin=1)

    def test_water_cation_631g(self):
        self.check(WATER, "6-31g", spin=1, charge=1)


@unittest.skipUnless(_HAVE_PYSCF, "PySCF not importable")
class CCSDTriplesVsPySCF(unittest.TestCase):
    def check(self, *args, **kw):
        ours, ref = ccgen_and_pyscf_t(*args, **kw)
        print(f"\n  {args[0][:12]:12s} {args[1]:7s} ccgen {ours:.12f} pyscf {ref:.12f} "
              f"diff {ours - ref:.1e}")
        self.assertGreater(abs(ref), 1e-6)  # a zero (T) would pass vacuously
        self.assertAlmostEqual(ours, ref, delta=1e-11)

    def test_water_sto3g(self):
        self.check(WATER, "sto-3g")

    def test_water_631g(self):
        self.check(WATER, "6-31g")

    def test_h2o2_c1_sto3g(self):
        self.check(H2O2_C1, "sto-3g")

    def test_oh_doublet_631g(self):
        # Open shell through GHF: same spin-orbital (T), no restriction.
        self.check(OH, "6-31g", spin=1)


if __name__ == "__main__":
    unittest.main()
