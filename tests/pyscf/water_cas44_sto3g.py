"""
PySCF reference: H2O CAS(4,4)/STO-3G
Matches Planck input: tests/inputs/casscf_tests/water_cas44_sto3g.hfinp

Geometry: C2v water (Planck input geometry, Angstrom)
Active space: CAS(4e, 4o) — two lone pairs + two OH bonding/antibonding MOs
Planck reference energy: -74.9760171760 Eh
"""

from pyscf import gto, scf, mcscf

CASE = "water_cas44_sto3g"
PLANCK_ENERGY = -74.9760171760
TOLERANCE = 1e-5

mol = gto.Mole()
mol.atom = """
O   0.000000   0.000000   0.117176
H   0.000000   0.757005  -0.468704
H   0.000000  -0.757005  -0.468704
"""
mol.basis = "sto-3g"
mol.charge = 0
mol.spin = 0
mol.cart = True  # match Planck 'basis_type cartesian'
mol.symmetry = True
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

mc = mcscf.CASSCF(mf, 4, 4)
mc = mc.newton()
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
