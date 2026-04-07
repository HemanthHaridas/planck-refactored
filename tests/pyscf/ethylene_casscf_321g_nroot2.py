"""
PySCF reference: ethylene CAS(2,2)/3-21G — second-root variant
Matches Planck input: tests/inputs/casscf_tests/ethylene_casscf_321g_nroot2.hfinp

Geometry: planar ethylene (Planck input geometry, Angstrom)
Active space: CAS(2e, 2o) — pi, pi* (HOMO, LUMO)
Planck reference energy: -77.5145223872 Eh

This case exercises the same geometry and active space as ethylene_casscf_321g
but via a different nroots code path in Planck. The ground-state energy should
be identical to the single-root result. As in the single-root script, the RHF
energy is reported from a symmetry-enabled PySCF reference so it matches the
Planck RHF root.
"""

from pyscf import gto, scf, mcscf

CASE = "ethylene_casscf_321g_nroot2"
PLANCK_ENERGY = -77.5145223872
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
    mol.basis = "3-21g"
    mol.charge = 0
    mol.spin = 0
    mol.cart = True  # match Planck 'basis_type cartesian'
    mol.symmetry = symmetry
    mol.verbose = 0
    mol.build()
    return mol


rhf_mol = build_molecule(symmetry=True)
rhf = scf.RHF(rhf_mol)
rhf.init_guess = "hcore"
rhf.conv_tol = 1e-12
rhf.max_cycle = 200
rhf.kernel()

casscf_mol = build_molecule(symmetry=False)
mf = scf.RHF(casscf_mol)
mf.conv_tol = 1e-12
mf.max_cycle = 200
mf.kernel()

mc = mcscf.CASSCF(mf, 2, 2)
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-6
mc.kernel()

e_casscf = mc.e_tot
delta = abs(e_casscf - PLANCK_ENERGY)
status = "PASS" if delta < TOLERANCE else "FAIL"

print(f"CASE: {CASE}")
print(f"HF_ENERGY:     {rhf.e_tot:.10f} Eh")
print(f"CASSCF_ENERGY: {e_casscf:.10f} Eh")
print(f"PLANCK_ENERGY: {PLANCK_ENERGY:.10f} Eh")
print(f"DELTA:         {delta:.2e} Eh")
print(f"STATUS:        {status}")
