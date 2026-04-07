"""
PySCF reference: H2O SA-CASSCF(4,4)/STO-3G, 2 roots equal weights
Matches Planck input: tests/inputs/casscf_tests/water_cas44_sto3g_sa2.hfinp

Geometry: C2v water (Planck input geometry, Angstrom)
Active space: CAS(4e, 4o)
Roots: 2, weights: [0.5, 0.5]
Planck SA-weighted energy: -74.7751377977 Eh

The SA-weighted energy is E_SA = 0.5*E_0 + 0.5*E_1.
Planck prints and converges to this quantity.
"""

from pyscf import gto, scf, mcscf

CASE = "water_cas44_sto3g_sa2"
PLANCK_SA_ENERGY = -74.7751377977
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
mol.symmetry = False
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

weights = [0.5, 0.5]
mc = mcscf.CASSCF(mf, 4, 4)
mc = mc.state_average_(weights)
mc = mc.newton()
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-6
mc.kernel()

e_states = mc.e_states
e_sa = sum(w * e for w, e in zip(weights, e_states))
delta = abs(e_sa - PLANCK_SA_ENERGY)
status = "PASS" if delta < TOLERANCE else "FAIL"

print(f"CASE: {CASE}")
print(f"HF_ENERGY:       {mf.e_tot:.10f} Eh")
print(f"ROOT_0_ENERGY:   {e_states[0]:.10f} Eh")
print(f"ROOT_1_ENERGY:   {e_states[1]:.10f} Eh")
print(f"SA_ENERGY:       {e_sa:.10f} Eh")
print(f"PLANCK_SA_ENERGY:{PLANCK_SA_ENERGY:.10f} Eh")
print(f"DELTA:           {delta:.2e} Eh")
print(f"STATUS:          {status}")
