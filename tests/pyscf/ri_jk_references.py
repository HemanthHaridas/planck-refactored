"""References for the RI-JK SCF gates (docs/RI_JK_ROUTINE_SCOPE.md steps 0-2).

PySCF density fitting on each tests/inputs/regression/ri/*_rijk_* input: the
orbital basis and the auxiliary basis (mp2_ri_basis) both read from Planck's
GBS files, Planck's CODATA 2018 Bohr. Also prints the exact (non-fitted)
energy, so the fitting error is on record.

Planck's auxiliary basis is always Cartesian. PySCF makes it spherical for a
spherical molecule, which is a different fit, so spherical inputs use
cart_aux_jk: 3-center integrals over Cartesian orbitals and Cartesian aux,
orbital indices transformed with cart2sph_coeff, fed to the SCF as get_jk.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_jk_references.py [AUX_BASIS]
AUX_BASIS overrides each input's mp2_ri_basis (e.g. def2-universal-JKFIT).
"""
import pathlib
import re
import sys

import numpy as np
from pyscf import df, gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)


def field(text, key):
    m = re.search(rf"^\s*{key}\s+(\S+)", text, re.M)
    return m.group(1) if m else None


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


def run(path):
    text = path.read_text()
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    elements = {line.split()[0] for line in coords[2:]}
    gbs = str(ROOT / "basis-sets" / field(text, "basis"))
    aux = str(ROOT / "basis-sets" / (sys.argv[1] if len(sys.argv) > 1 else field(text, "mp2_ri_basis")))
    cart = field(text, "basis_type") != "spherical"
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=cart, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    energies = []
    for fitted in (True, False):
        mf = (scf.RHF if field(text, "scf_type") == "rhf" else scf.UHF)(mol)
        aux_basis = {e: parse_gaussian.load(aux, e) for e in elements}
        if fitted and cart:
            mf = mf.density_fit(auxbasis=aux_basis)
        elif fitted:
            mf.get_jk = cart_aux_jk(mol, aux_basis)
        mf.init_guess = "1e"  # Planck guess hcore
        mf.conv_tol = 1e-12
        mf.kernel()
        # From 1e, H2O2+ UHF lands on a saddle 0.023 Eh up; Planck's hcore
        # run reaches the stable minimum. Follow internal instabilities so
        # both codes are compared at the minimum.
        if isinstance(mf, scf.uhf.UHF):
            for _ in range(5):
                mo, _, stable, _ = mf.stability(return_status=True)
                if stable:
                    break
                mf.kernel(mf.make_rdm1(mo, mf.mo_occ))
            assert stable, path.stem
        energies.append(mf.e_tot)
        assert mf.converged, path.stem
    print(f"{path.stem:34s} E_RI = {energies[0]:.10f}  E_exact = {energies[1]:.10f}  "
          f"fit error = {energies[0] - energies[1]:.3e}")


if __name__ == "__main__":
    for p in sorted((ROOT / "tests/inputs/regression/ri").glob("*_rijk_*.hfinp")):
        run(p)
