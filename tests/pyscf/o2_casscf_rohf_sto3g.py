"""
PySCF reference: O2 triplet CAS(8,6)/STO-3G on a ROHF reference
Matches Planck input: tests/inputs/regression/post_hf/o2_casscf_rohf_sto3g.hfinp

Geometry: O2 at r = 1.208 Angstrom (Planck input geometry, Angstrom)
Reference: ROHF, triplet (mol.spin = 2)
Active space: CAS(8e, 6o) — full O2 valence above the 1s/2s-like core
             core = 4 doubly-occupied orbitals; active split (n_alpha, n_beta) = (5, 3)
"""

from pyscf import gto, scf, mcscf

CASE = "o2_casscf_rohf_sto3g"
TOLERANCE = 1e-5

mol = gto.Mole()
mol.atom = """
O    0.000000    0.000000   -0.604000
O    0.000000    0.000000    0.604000
"""
mol.basis = "sto-3g"
mol.charge = 0
mol.spin = 2  # triplet: n_alpha - n_beta = 2 (multiplicity 3)
mol.cart = True  # match Planck 'basis_type cartesian'
mol.symmetry = False
mol.verbose = 0
mol.build()

mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# CAS(8e, 6o): 6 active orbitals, 8 active electrons.
# Pin the active alpha/beta split to (5, 3) to match Planck's
# n_alpha_act = (nactele + (mult-1))/2 = 5, n_beta_act = 3.
mc = mcscf.CASSCF(mf, 6, (5, 3))
mc = mc.newton()
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-6
mc.kernel()

e_rohf = mf.e_tot
e_casscf = mc.e_tot

print(f"CASE: {CASE}")
print(f"ROHF_ENERGY:   {e_rohf:.10f} Eh")
print(f"CASSCF_ENERGY: {e_casscf:.10f} Eh")
