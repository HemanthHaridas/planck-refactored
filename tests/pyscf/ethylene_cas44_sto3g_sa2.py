"""
PySCF reference: twisted ethylene SA-CASSCF(4,4)/STO-3G, 2 roots equal weights
Matches Planck input: tests/inputs/casscf_tests/ethylene_cas44_sto3g_sa2.hfinp

Geometry: 90-degree twisted ethylene (Planck input geometry, Angstrom).
One CH2 group lies in the XZ plane; the other lies in the YZ plane.
This geometry breaks the pi bond, making the ground and first excited states
nearly degenerate — a root-tracking stress test.

Active space: CAS(4e, 4o)
Roots: 2, weights: [0.5, 0.5]
Planck SA-weighted energy: -77.0034974301 Eh

Note on orbital selection: at the 90-degree twisted geometry the pi system is
broken. PySCF selects ncas active orbitals centered on the HOMO/LUMO gap. For
CAS(4,4) this should include the two C 2p orbitals that become degenerate at 90
degrees plus the adjacent sigma/sigma* pair. If PySCF converges to a different
local minimum or picks different orbitals, use sort_mo to specify the correct
active space before calling kernel.
"""

from pyscf import gto, scf, mcscf

CASE = "ethylene_cas44_sto3g_sa2"
PLANCK_SA_ENERGY = -77.0034974301
TOLERANCE = 1e-5

mol = gto.Mole()
mol.atom = """
C   0.000000   0.000000   0.000000
C   0.000000   0.000000   1.340000
H   0.920000   0.000000  -0.540000
H  -0.920000   0.000000  -0.540000
H   0.000000   0.920000   1.880000
H   0.000000  -0.920000   1.880000
"""
mol.basis = "sto-3g"
mol.charge = 0
mol.spin = 0
mol.cart = True  # match Planck 'basis_type cartesian'
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

weights = [0.5, 0.5]
mc = mcscf.CASSCF(mf, 4, 4)
mc = mc.state_average_(weights)
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
