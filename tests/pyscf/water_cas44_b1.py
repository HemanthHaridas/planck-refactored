"""
PySCF reference: H2O CAS(4,4)/STO-3G targeting B1 symmetry
Matches Planck input: tests/benchmarks/casscf/pyscf_reference/water_cas44_b1.hfinp

Geometry: C2v water (Planck input geometry, Angstrom)
Active space: CAS(4e, 4o)
Target irrep: B1 (lowest B1 state, an excited state of water)
Planck reference energy: -74.5856163677 Eh

Note: The B1 target gives a CASSCF energy above the RHF energy because
the ground state of water has A1 symmetry; this constrains the CI
wavefunction to the B1 sector.
"""

from pyscf import gto, scf, mcscf

CASE = "water_cas44_b1"
PLANCK_ENERGY = -74.5856163677
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
mol.cart = True   # match Planck 'basis_type cartesian'
mol.symmetry = True  # C2v — needed to target B1 irrep
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

mc = mcscf.CASSCF(mf, 4, 4)
mc = mc.newton()
mc.fcisolver.wfnsym = "B1"
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
