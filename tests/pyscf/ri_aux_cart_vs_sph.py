"""Cartesian vs spherical auxiliary basis with a spherical orbital basis
(docs/RI_SPHERICAL_AUX.md): aux function counts and RHF fitting error
for each, against the exact energy. Bases read from Planck's GBS files.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_aux_cart_vs_sph.py
"""
import pathlib

import numpy as np
from pyscf import df, gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]


def cart_aux_jk(mol, aux_basis):
    """get_jk for a spherical mol, fitted with a Cartesian auxiliary basis."""
    molc = mol.copy()
    molc.cart = True
    molc.build()
    auxmol = molc.copy()
    auxmol.basis = aux_basis
    auxmol.build()
    assert molc.cart and auxmol.cart
    naux, nc, ns = auxmol.nao, molc.nao, mol.nao
    j3c = df.incore.aux_e2(molc, auxmol, intor="int3c2e", aosym="s1").reshape(nc, nc, naux)
    c = molc.cart2sph_coeff()                                   # (nc, ns)
    j3c = np.einsum("pi,pqQ,qj->ijQ", c, j3c, c, optimize=True).reshape(ns * ns, naux)
    L = np.linalg.cholesky(auxmol.intor("int2c2e"))
    B = np.linalg.solve(L, j3c.T)                               # (naux, ns*ns)
    Bq = B.reshape(naux, ns, ns)

    def get_jk(mol_, dm, hermi=1, with_j=True, with_k=True, omega=None):
        dms = np.asarray(dm).reshape(-1, ns, ns)
        vj = np.array([((B @ d.ravel()) @ B).reshape(ns, ns) for d in dms])
        vk = np.array([np.einsum("Qij,jk,Qlk->il", Bq, d, Bq, optimize=True) for d in dms])
        return vj.reshape(np.shape(dm)), vk.reshape(np.shape(dm))
    return get_jk


MOLS = {
    "water": "O 0 0 0\nH 0.758602 0 0.504284\nH -0.758602 0 0.504284",
    "h2o2": "O 0 0.7 -0.05\nO 0.02 -0.71 0.04\nH 0.88 0.9 0.42\nH -0.65 -0.95 0.78",
    "benzene": "C 0 1.3915 0\nC 1.2051 0.6958 0\nC 1.2051 -0.6958 0\nC 0 -1.3915 0\n"
               "C -1.2051 -0.6958 0\nC -1.2051 0.6958 0\nH 0 2.4715 0\nH 2.1404 1.2358 0\n"
               "H 2.1404 -1.2358 0\nH 0 -2.4715 0\nH -2.1404 -1.2358 0\nH -2.1404 1.2358 0",
}
PAIRS = (("6-31g*", "def2-universal-JKFIT"), ("cc-pVDZ", "cc-pVTZ-JKFIT"),
         ("cc-pVDZ", "cc-pVDZ-RIFIT"))


def load(name, elements):
    return {e: parse_gaussian.load(str(ROOT / "basis-sets" / name), e) for e in elements}


for orb, auxname in PAIRS:
    for name, xyz in MOLS.items():
        els = {line.split()[0] for line in xyz.splitlines()}
        mol = gto.M(atom=xyz, cart=False, verbose=0, basis=load(orb, els))
        aux = load(auxname, els)
        n_cart = gto.M(atom=xyz, cart=True, verbose=0, basis=aux).nao
        n_sph = gto.M(atom=xyz, cart=False, verbose=0, basis=aux).nao

        def energy(kind):
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-11
            if kind == "sph":
                mf = mf.density_fit(auxbasis=aux)  # PySCF: aux follows mol.cart
            elif kind == "cart":
                mf.get_jk = cart_aux_jk(mol, aux)
            return mf.kernel()

        ex, es, ec = energy("exact"), energy("sph"), energy("cart")
        print(f"{orb:8s} {auxname:21s} {name:8s} naux cart {n_cart:4d} sph {n_sph:4d} "
              f"({n_cart / n_sph:.2f}x)  fit err cart {ec - ex:+.2e} sph {es - ex:+.2e}  "
              f"sph-cart {es - ec:+.2e}", flush=True)
