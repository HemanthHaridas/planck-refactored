"""
PySCF reference: twisted ethylene SA-CASSCF(4,4)/STO-3G, 2 roots equal weights
Matches Planck input:
tests/inputs/casscf_tests/ethylene_cas44_sto3g_sa2.hfinp

Geometry: 90-degree twisted ethylene (Planck input geometry, Angstrom).
One CH2 group lies in the XZ plane; the other lies in the YZ plane.
This geometry breaks the pi bond, making the ground and first excited states
nearly degenerate — a root-tracking stress test.

Active space: CAS(4e, 4o)
Roots: 2, weights: [0.5, 0.5]
Planck RHF energy:        -76.8522465545 Eh  (symmetry-enabled RHF, from .log)
Planck SA-weighted energy: -77.0034974301 Eh

The D2-symmetry RHF MOs are used as the initial orbital guess for the SA-CASSCF
to match Planck's starting point. The FCI solver runs without symmetry constraints
(mol.symmetry = False for CASSCF) so that both SA roots can mix freely across D2
irreps — necessary because the two near-degenerate states at 90° twist have
different spatial symmetries and constraining to any single D2 irrep causes the
SA optimizer to produce a badly non-degenerate root pair (~0.22 Eh apart).
"""

from pyscf import gto, scf, mcscf

CASE = "ethylene_cas44_sto3g_sa2"
PLANCK_SA_ENERGY = -77.0034974301
TOLERANCE = 1e-5

GEOMETRY = """
C   0.000000   0.000000   0.000000
C   0.000000   0.000000   1.340000
H   0.920000   0.000000  -0.540000
H  -0.920000   0.000000  -0.540000
H   0.000000   0.920000   1.880000
H   0.000000  -0.920000   1.880000
"""


def build_molecule(*, symmetry: bool) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = GEOMETRY
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.cart = True  # match Planck 'basis_type cartesian'
    mol.symmetry = symmetry
    mol.verbose = 0
    mol.build()
    return mol


# D2-symmetry RHF — matches Planck's starting point; MOs used as init guess
mf_sym = scf.RHF(build_molecule(symmetry=True))
mf_sym.conv_tol = 1e-12
mf_sym.max_cycle = 200
mf_sym.kernel()

# SA-CASSCF: symmetry-free mol so all D2 irreps can mix freely across both roots
mf_c1 = scf.RHF(build_molecule(symmetry=False))
mf_c1.conv_tol = 1e-12
mf_c1.max_cycle = 200
mf_c1.kernel()

weights = [0.5, 0.5]
mc = mcscf.CASSCF(mf_c1, 4, 4)
mc = mc.state_average_(weights)
mc = mc.newton()
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-6
mc.kernel(mf_sym.mo_coeff)  # seed from D2 RHF to match Planck's orbital basin

e_states = mc.e_states
e_sa = sum(w * e for w, e in zip(weights, e_states))
delta = abs(e_sa - PLANCK_SA_ENERGY)
status = "PASS" if delta < TOLERANCE else "FAIL"

print(f"CASE: {CASE}")
print(f"HF_ENERGY:       {mf_sym.e_tot:.10f} Eh")
print(f"ROOT_0_ENERGY:   {e_states[0]:.10f} Eh")
print(f"ROOT_1_ENERGY:   {e_states[1]:.10f} Eh")
print(f"SA_ENERGY:       {e_sa:.10f} Eh")
print(f"PLANCK_SA_ENERGY:{PLANCK_SA_ENERGY:.10f} Eh")
print(f"DELTA:           {delta:.2e} Eh")
print(f"STATUS:          {status}")
