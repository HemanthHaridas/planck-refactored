"""
PySCF reference: ethylene CAS(2,2)/3-21G
Matches Planck input: tests/inputs/casscf_tests/ethylene_casscf_321g.hfinp

Geometry: planar ethylene (Planck input geometry, Angstrom)
Active space: CAS(2e, 2o) — pi, pi* (HOMO, LUMO)
Planck reference energy: -77.5145223872 Eh

Planck preserves the D2d RHF symmetry for this case, while symmetry-free
PySCF collapses to a lower-symmetry RHF stationary point. This script reports
the symmetry-enabled PySCF RHF energy directly, but keeps the historical
symmetry-free CASSCF path because it already matches the checked-in Planck
CAS(2,2) benchmark.

PySCF selects active orbitals as the ncas orbitals centered on the HOMO/LUMO
gap by default. For planar ethylene with nactele=2, nactorb=2, this naturally
selects the pi and pi* orbitals. If PySCF picks different orbitals and the
energy is wrong, sort_mo can be used to fix the selection:
    mo = mc.sort_mo([homo_idx, lumo_idx])
    mc.kernel(mo)
"""

from pyscf import gto, scf, mcscf

CASE = "ethylene_casscf_321g"
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
