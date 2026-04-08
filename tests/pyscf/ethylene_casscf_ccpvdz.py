"""
PySCF reference: ethylene CAS(2,2)/cc-pVDZ
Matches Planck input: tests/inputs/casscf_tests/ethylene_casscf_ccpvdz.hfinp

Geometry: planar ethylene (Planck input geometry, Angstrom)
Active space: CAS(2e, 2o) — pi, pi* (HOMO, LUMO)
Planck reference energy: -77.9524855977 Eh

Symmetry is enabled for RHF to match Planck's D2h-aware HF reference.
CASSCF runs without symmetry because PySCF's default active-orbital
selection changes under symmetry for this system (same issue as
ethylene_casscf_321g — D2h irrep constraints cause a different orbital
pairing than the symmetry-free HOMO/LUMO selection that matches Planck).
"""

from pyscf import gto, scf, mcscf

CASE = "ethylene_casscf_ccpvdz"
PLANCK_ENERGY = -77.9524855977
TOLERANCE = 1e-5

GEOMETRY = """
C  -0.669500   0.000000   0.000000
C   0.669500   0.000000   0.000000
H  -1.233698   0.927942   0.000000
H  -1.233698  -0.927942   0.000000
H   1.233698   0.000000   0.927942
H   1.233698   0.000000  -0.927942
"""


def build_molecule(*, symmetry: bool) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = GEOMETRY
    mol.basis = "cc-pvdz"
    mol.charge = 0
    mol.spin = 0
    mol.cart = True  # match Planck 'basis_type cartesian'
    mol.symmetry = symmetry
    mol.verbose = 0
    mol.build()
    return mol


mf = scf.RHF(build_molecule(symmetry=True))
mf.init_guess = "hcore"
mf.conv_tol = 1e-12
mf.max_cycle = 200
mf.kernel()

casscf_mol = build_molecule(symmetry=False)
mf_nosym = scf.RHF(casscf_mol)
mf_nosym.init_guess = "hcore"
mf_nosym.conv_tol = 1e-12
mf_nosym.max_cycle = 200
mf_nosym.kernel()

mc = mcscf.CASSCF(mf_nosym, 2, 2)
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
