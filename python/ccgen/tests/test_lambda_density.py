"""Unrelaxed CC densities as ∂L/∂(integral) (docs/CC_LAMBDA_SCOPE.md, step 3).

Restricted CCSD: the generated ∂L/∂f and ∂L/∂v, at PySCF's converged t and
our own spatial λ (solved by GMRES), assembled into total RDMs by the chain
rule through

    E = E_ref(h, g) + L(f(h, g), v(g)),   f = h + Σ_i [2(pq|ii) − (pi|iq)],
    v_pqrs = <pq|rs> = (pr|qs),

must equal PySCF's ``make_rdm1``/``make_rdm2`` (built from PySCF's λ) element
by element, in PySCF's convention E = Σ h D + ½ Σ (pq|rs) Γ. The densities do
not depend on how λ is parametrized, so this tests the spatial λ, the density
terms and the assembly together.
"""

from __future__ import annotations

import os
import string
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from test_lambda_spatial import H2O2, WATER, _pyscf_rccsd, _solve_lambda, _spatial_lagrangian  # noqa: E402

from ccgen.lagrangian import density_terms  # noqa: E402

try:
    import pyscf  # noqa: F401
    import scipy  # noqa: F401
    HAVE_PYSCF = True
except ImportError:
    HAVE_PYSCF = False


def _eval_slots(term, no, nv, tensors):
    """One term to an array in its free indices' own order (the derivative's
    slot order), not residual_einsum's virtuals-first layout."""
    n = no + nv
    sl = {"occ": slice(0, no), "vir": slice(no, n)}
    letters = dict(zip(list(term.free_indices) + list(term.summed_indices),
                       string.ascii_letters))
    subs, ops = [], []
    for f in term.factors:
        subs.append("".join(letters[i] for i in f.indices))
        a = tensors[f.name]
        ops.append(a[tuple(sl[i.space] for i in f.indices)] if f.name in ("v", "f") else a)
    out = "".join(letters[i] for i in term.free_indices)
    return float(term.coeff) * np.einsum(",".join(subs) + "->" + out, *ops, optimize=True)


def _full(blocks, no, nv, rank, tensors):
    """Place each space block of a derivative into one (nmo,)*rank array."""
    n = no + nv
    sl = {"o": slice(0, no), "v": slice(no, n)}
    out = np.zeros((n,) * rank)
    for block, terms in blocks.items():
        out[tuple(sl[c] for c in block)] += sum(_eval_slots(t, no, nv, tensors) for t in terms)
    return out


def _sym_ab(g):
    """(αα|ββ): only the swaps within each pair are symmetries."""
    g = 0.5 * (g + g.transpose(1, 0, 2, 3))
    return 0.5 * (g + g.transpose(0, 1, 3, 2))


def _sym8(g):
    g = 0.5 * (g + g.transpose(1, 0, 2, 3))
    g = 0.5 * (g + g.transpose(0, 1, 3, 2))
    return 0.5 * (g + g.transpose(2, 3, 0, 1))


def rcc_rdms(dens, no, nv, tensors):
    """Total spin-summed RDMs (MO basis, PySCF convention) from ∂L/∂f, ∂L/∂v."""
    n = no + nv
    Lf = _full(dens["f"], no, nv, 2, tensors)
    Lv = _full(dens["v"], no, nv, 4, tensors)
    occ = np.arange(no)

    D = Lf.copy()
    D[occ, occ] += 2.0
    D = 0.5 * (D + D.T)

    # Y = ∂E/∂(pq|rs). v[a,b,c,d] = g[a,c,b,d].
    Y = Lv.transpose(0, 2, 1, 3).copy()
    for i in occ:
        for j in occ:
            Y[i, i, j, j] += 2.0
            Y[i, j, j, i] -= 1.0
        Y[:, :, i, i] += 2.0 * Lf
        Y[:, i, i, :] -= Lf
    return D, _sym8(2.0 * Y), n


@unittest.skipUnless(HAVE_PYSCF, "needs PySCF and SciPy")
class RCCDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _eqs, cls.lag, cls.G = _spatial_lagrangian(canonical_fock=False)
        cls.dens = density_terms(cls.lag, "rcc")

    def _check(self, atom, basis):
        from pyscf.cc import ccsd_rdm

        mycc, _eris, tensors, no, nv = _pyscf_rccsd(atom, basis)
        self.assertLess(_solve_lambda(self.G, no, nv, tensors), 1e-9)
        D, G2, _n = rcc_rdms(self.dens, no, nv, tensors)

        l1, l2 = mycc.solve_lambda()
        D_ref = ccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2)
        G_ref = ccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
        d1 = np.abs(D - 0.5 * (D_ref + D_ref.T)).max()
        d2 = np.abs(G2 - _sym8(G_ref)).max()
        self.assertLess(d1, 1e-8, f"1-RDM differs by {d1:.1e}")
        self.assertLess(d2, 1e-8, f"2-RDM differs by {d2:.1e}")

        from pyscf import ao2mo
        mf = mycc._scf
        C = mf.mo_coeff
        h = C.T @ mf.get_hcore() @ C
        g = ao2mo.kernel(mf.mol, C, compact=False).reshape((no + nv,) * 4)
        e = np.sum(h * D) + 0.5 * np.sum(g * G2) + mf.energy_nuc()
        self.assertLess(abs(e - mycc.e_tot), 1e-10, "E from the RDMs")
        return d1, d2

    def test_water_sto3g(self):
        self._check(WATER, "sto-3g")

    def test_h2o2_c1_631g(self):
        self._check(H2O2, "6-31g")


if __name__ == "__main__":
    unittest.main()


# ── UCC ──────────────────────────────────────────────────────────────────

def _ucc_eval_slots(term, dims, tensors):
    from ccgen.tests.residual_eval import ucc_resolve_factor

    letters = dict(zip(list(term.free_indices) + list(term.summed_indices),
                       string.ascii_letters))
    subs = ["".join(letters[i] for i in f.indices) for f in term.factors]
    ops = [ucc_resolve_factor(f, tensors, dims) for f in term.factors]
    out = "".join(letters[i] for i in term.free_indices)
    return float(term.coeff) * np.einsum(",".join(subs) + "->" + out, *ops, optimize=True)


def _ucc_full(blocks, name, dims):
    """One UCC integral derivative (``f_aa``, ``v_abab``, ...) as a full array
    over each slot's own spin orbital space."""
    tag = name.split("_")[1]
    no = {"a": dims["noa"], "b": dims["nob"]}
    nv = {"a": dims["nva"], "b": dims["nvb"]}
    return np.zeros(tuple(no[s] + nv[s] for s in tag)), no, tag


def ucc_rdms(dens, dims, tensors):
    """Total RDMs in PySCF's UCC convention:
    E = Σ h_a Da + Σ h_b Db + ½ Σ g_aa Γaa + Σ g_ab Γab + ½ Σ g_bb Γbb."""
    full = {}
    for name, blocks in dens.items():
        arr, no, tag = _ucc_full(blocks, name, dims)
        for block, terms in blocks.items():
            idx = tuple(slice(0, no[s]) if c == "o" else slice(no[s], None)
                        for c, s in zip(block, tag))
            arr[idx] += sum(_ucc_eval_slots(t, dims, tensors) for t in terms)
        full[name] = arr
    Lfa, Lfb = full["f_aa"], full["f_bb"]
    oa, ob = np.arange(dims["noa"]), np.arange(dims["nob"])

    Da, Db = Lfa.copy(), Lfb.copy()
    Da[oa, oa] += 1.0
    Db[ob, ob] += 1.0

    # ∂E/∂g. v_aaaa[p,q,r,s] = g_aa[p,r,q,s] − g_aa[p,s,q,r]; v_abab[p,q,r,s] = g_ab[p,r,q,s].
    def same(Lv):
        return Lv.transpose(0, 2, 1, 3) - Lv.transpose(0, 3, 1, 2)
    Yaa, Ybb = same(full["v_aaaa"]), same(full["v_bbbb"])
    Yab = full["v_abab"].transpose(0, 2, 1, 3).copy()
    for Y, Lf, occ in ((Yaa, Lfa, oa), (Ybb, Lfb, ob)):
        for i in occ:
            for j in occ:
                Y[i, i, j, j] += 0.5
                Y[i, j, j, i] -= 0.5
            Y[:, :, i, i] += Lf
            Y[:, i, i, :] -= Lf
    for i in oa:
        Yab[i, i, :, :] += Lfb
        for j in ob:
            Yab[i, i, j, j] += 1.0
    for j in ob:
        Yab[:, :, j, j] += Lfa

    sym1 = lambda d: 0.5 * (d + d.T)  # noqa: E731
    return (sym1(Da), sym1(Db)), (_sym8(2 * Yaa), _sym_ab(Yab), _sym8(2 * Ybb))


@unittest.skipUnless(HAVE_PYSCF, "needs PySCF and SciPy")
class UCCDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ccgen.generate import generate_cc_equations
        from ccgen.lagrangian import lagrangian_terms
        from ccgen.spin import ucc_adapt_equations

        eqs = ucc_adapt_equations(
            generate_cc_equations("ccsd", engine="diagram", canonical_fock=False))
        cls.dens = density_terms(lagrangian_terms(eqs, "ucc"), "ucc")

    def _check(self, *system):
        from pyscf import ao2mo
        from pyscf.cc import uccsd_rdm
        from test_lambda_ucc import _pyscf_uccsd

        tensors, dims, mycc = _pyscf_uccsd(*system, return_cc=True)
        (Da, Db), (Gaa, Gab, Gbb) = ucc_rdms(self.dens, dims, tensors)
        l1, l2 = mycc.l1, mycc.l2
        Da_r, Db_r = uccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2)
        Gaa_r, Gab_r, Gbb_r = uccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
        sym1 = lambda d: 0.5 * (d + d.T)  # noqa: E731
        d1 = max(np.abs(Da - sym1(Da_r)).max(), np.abs(Db - sym1(Db_r)).max())
        d2 = max(np.abs(Gaa - _sym8(Gaa_r)).max(), np.abs(Gbb - _sym8(Gbb_r)).max(),
                 np.abs(Gab - _sym_ab(Gab_r)).max())
        self.assertLess(d1, 1e-8, f"1-RDM differs by {d1:.1e}")
        self.assertLess(d2, 1e-8, f"2-RDM differs by {d2:.1e}")

        mf = mycc._scf
        Ca, Cb = mf.mo_coeff
        mol = mf.mol
        ha, hb = (C.T @ mf.get_hcore() @ C for C in (Ca, Cb))
        na, nb = Ca.shape[1], Cb.shape[1]
        gaa = ao2mo.general(mol, (Ca,) * 4, compact=False).reshape((na,) * 4)
        gbb = ao2mo.general(mol, (Cb,) * 4, compact=False).reshape((nb,) * 4)
        gab = ao2mo.general(mol, (Ca, Ca, Cb, Cb), compact=False).reshape(na, na, nb, nb)
        e = (np.sum(ha * Da) + np.sum(hb * Db) + 0.5 * np.sum(gaa * Gaa)
             + np.sum(gab * Gab) + 0.5 * np.sum(gbb * Gbb) + mf.energy_nuc())
        self.assertLess(abs(e - mycc.e_tot), 1e-10, "E from the RDMs")
        return d1, d2, e - mycc.e_tot

    def test_ch3_sto3g(self):
        from test_lambda_ucc import CH3
        self._check(CH3, "sto-3g", 0, 1)

    def test_water_cation_631g(self):
        self._check(WATER, "6-31g", 1, 1)
