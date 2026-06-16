"""
PySCF reference: H2 CAS(2,2)/STO-3G
Matches Planck input: tests/benchmarks/casscf/pyscf_reference/h2_cas22_sto3g.hfinp

Geometry: H-H at 0.740 Angstrom (Planck input geometry)
Active space: CAS(2e, 2o) — sigma_g, sigma_u* (HOMO + LUMO)
Planck reference energy: -1.1372838351 Eh
"""

from pyscf import gto, scf, mcscf

CASE = "h2_cas22_sto3g"
PLANCK_ENERGY = -1.1372838351
TOLERANCE = 1e-5

mol = gto.Mole()
mol.atom = """
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
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

mc = mcscf.CASSCF(mf, 2, 2)
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
