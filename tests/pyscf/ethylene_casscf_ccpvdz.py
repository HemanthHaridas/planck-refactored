"""
PySCF reference: ethylene CAS(2,2)/cc-pVDZ
Matches Planck input: tests/inputs/casscf_tests/ethylene_casscf_ccpvdz.hfinp

Geometry: planar ethylene (Planck input geometry, Angstrom)
Active space: CAS(2e, 2o) — pi, pi* (HOMO, LUMO)
Planck reference energy: -77.9524855976 Eh

Note: PySCF selects active orbitals as the ncas orbitals centered on the
HOMO/LUMO gap by default. For planar ethylene with nactele=2, nactorb=2,
this naturally picks the pi and pi* orbitals. If PySCF picks different
orbitals and the energy is wrong, sort_mo can be used to fix the selection:
    mo = mc.sort_mo([homo_idx, lumo_idx])
    mc.kernel(mo)
"""

from pyscf import gto, scf, mcscf

CASE = "ethylene_casscf_ccpvdz"
PLANCK_ENERGY = -77.9524855976
TOLERANCE = 1e-5

mol = gto.Mole()
mol.atom = """
C  -0.669500   0.000000   0.000000
C   0.669500   0.000000   0.000000
H  -1.233698   0.927942   0.000000
H  -1.233698  -0.927942   0.000000
H   1.233698   0.000000   0.927942
H   1.233698   0.000000  -0.927942
"""
mol.basis = "cc-pvdz"
mol.charge = 0
mol.spin = 0
mol.cart = True  # match Planck 'basis_type cartesian'
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

mc = mcscf.CASSCF(mf, 2, 2)
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-6
mc.kernel()

e_casscf = mc.e_tot
delta = abs(e_casscf - PLANCK_ENERGY)
status = "PASS" if delta < TOLERANCE else "FAIL"

print(f"CASE: {CASE}")
print(f"HF_ENERGY:     {mf.e_tot:.10f} Eh")
print(f"CASSCF_ENERGY: {e_casscf:.10f} Eh")
print(f"PLANCK_ENERGY: {PLANCK_ENERGY:.10f} Eh")
print(f"DELTA:         {delta:.2e} Eh")
print(f"STATUS:        {status}")
