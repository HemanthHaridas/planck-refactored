"""Cartesian vs spherical auxiliary basis with a spherical orbital basis
(docs/RI_SPHERICAL_AUX_SCOPE.md): aux function counts and RHF fitting error
for each, against the exact energy. Bases read from Planck's GBS files.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_aux_cart_vs_sph.py
"""
import pathlib
import sys

from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ri_jk_references import ROOT, cart_aux_jk  # noqa: E402

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
