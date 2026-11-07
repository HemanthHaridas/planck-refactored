"""PySCF references for the def2 ECP post-HF gates (docs/DEF2_ECP.md).

HI / def2-SVP, cart=False, basis and ECP from Planck's own GBS (load_ecp from
def2_ecp_references.py), Planck's CODATA 2018 Bohr: RHF, CCSD and CASSCF,
all converged tightly so the comparison is not limited by convergence.
Run: tests/pyscf/.venv/bin/python tests/pyscf/def2_ecp_post_hf_references.py
"""
import pathlib
import sys

from pyscf import cc, gto, mcscf, scf
from pyscf.gto.basis import parse_gaussian

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from def2_ecp_references import load_ecp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
GBS = str(ROOT / "basis-sets" / "def2-SVP")

mol = gto.M(atom="H 0 0 0; I 0 0 1.609", unit=BOHR_ANGSTROM, cart=False, verbose=0,
            basis={e: parse_gaussian.load(GBS, e) for e in ("H", "I")}, ecp={"I": load_ecp(GBS, "I")})
mf = scf.RHF(mol)
mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
e_hf = mf.kernel()
print(f"RHF    E = {e_hf:.10f}  nelec = {mol.nelectron}")

ccsd = cc.CCSD(mf)
ccsd.conv_tol, ccsd.conv_tol_normt = 1e-12, 1e-10
ccsd.kernel()
print(f"CCSD   E_corr = {ccsd.e_corr:.10f}  converged = {ccsd.converged}")

if len(sys.argv) > 1:   # CASSCF(ncas, nelecas) from the command line
    ncas, nelecas = int(sys.argv[1]), int(sys.argv[2])
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol, mc.conv_tol_grad = 1e-12, 1e-7
    mc.kernel()
    print(f"CASSCF({nelecas},{ncas}) E = {mc.e_tot:.10f}  converged = {mc.converged}")
