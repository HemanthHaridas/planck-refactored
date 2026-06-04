"""
PySCF reference: OH doublet CAS(5,4)/STO-3G on a ROHF reference
Matches Planck input: tests/inputs/regression/post_hf/oh_casscf_rohf_sto3g.hfinp

Geometry: OH at r = 0.970 Angstrom (Planck input geometry, Angstrom)
Reference: ROHF, doublet (mol.spin = 1)
Active space: CAS(5e, 4o) — full OH valence above the O 1s-like core
             core = 2 doubly-occupied orbitals; active split (n_alpha, n_beta) = (3, 2)
Exercises the odd-electron path: n_alpha_act - n_beta_act = 1.
"""

from pyscf import gto, scf, mcscf

CASE = "oh_casscf_rohf_sto3g"
TOLERANCE = 1e-5

mol = gto.Mole()
mol.atom = """
O    0.000000    0.000000    0.000000
H    0.000000    0.000000    0.970000
"""
mol.basis = "sto-3g"
mol.charge = 0
mol.spin = 1  # doublet: n_alpha - n_beta = 1 (multiplicity 2)
mol.cart = True  # match Planck 'basis_type cartesian'
mol.symmetry = False
mol.verbose = 0
mol.build()

mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# CAS(5e, 4o): 4 active orbitals, 5 active electrons.
# Pin the active alpha/beta split to (3, 2) to match Planck's
# n_alpha_act = (nactele + (mult-1))/2 = 3, n_beta_act = 2.
mc = mcscf.CASSCF(mf, 4, (3, 2))
mc = mc.newton()
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-6
mc.kernel()

e_rohf = mf.e_tot
e_casscf = mc.e_tot

print(f"CASE: {CASE}")
print(f"ROHF_ENERGY:   {e_rohf:.10f} Eh")
print(f"CASSCF_ENERGY: {e_casscf:.10f} Eh")
