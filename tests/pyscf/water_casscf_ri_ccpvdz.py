"""
PySCF reference: H2O DF-CAS(4,4)/cc-pVDZ (density-fitted / RI)
Matches Planck input: tests/inputs/regression/ri/water_casscf_ri_ccpvdz.hfinp

Same geometry / active space / symmetry basin as the conventional gate
water_cas44_ccpvdz.py, but with density fitting turned on for both the SCF
and the CASSCF, using the cc-pVDZ-RI auxiliary basis (PySCF's name for the
same Weigend/Turbomole fitting set Planck loads as cc-pVDZ-RIFIT).

Planck RI-CASSCF energy: -76.0449911146 Eh

Tolerance note: 5e-5, looser than the conventional gate's 1e-5. Both codes
use the same aux basis and land in the same basin (conventional CASSCF
-76.0440 in both), so the residual ~2.7e-5 spread is the genuine cross-code
difference in how the density fitting itself is applied (metric lindep
threshold, fitting-contraction order), not a basin or active-space
disagreement. The RI-induced *shift* from conventional agrees to ~3e-6
between the two codes, which is the real correctness signal.
"""

from pyscf import gto, scf, mcscf

CASE = "water_casscf_ri_ccpvdz"
PLANCK_ENERGY = -76.0449911146
TOLERANCE = 5e-5
AUXBASIS = "cc-pvdz-ri"

mol = gto.Mole()
mol.atom = """
O   0.000000   0.000000   0.117176
H   0.000000   0.757005  -0.468704
H   0.000000  -0.757005  -0.468704
"""
mol.basis = "cc-pvdz"
mol.charge = 0
mol.spin = 0
mol.cart = True  # match Planck 'basis_type cartesian'
mol.symmetry = True  # match Planck use_symm .true. and the conventional gate
mol.verbose = 0
mol.build()

mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
mf.conv_tol = 1e-12
mf.kernel()

mc = mcscf.DFCASSCF(mf, 4, 4, auxbasis=AUXBASIS)
mc = mc.newton()
mc.conv_tol = 1e-9
mc.conv_tol_grad = 1e-6
mc.kernel()

e_casscf = mc.e_tot
delta = abs(e_casscf - PLANCK_ENERGY)
status = "PASS" if delta < TOLERANCE else "FAIL"

print(f"CASE: {CASE}")
print(f"HF_ENERGY(DF):  {mf.e_tot:.10f} Eh")
print(f"CASSCF_ENERGY:  {e_casscf:.10f} Eh")
print(f"PLANCK_ENERGY:  {PLANCK_ENERGY:.10f} Eh")
print(f"DELTA:          {delta:.2e} Eh")
print(f"STATUS:         {status}")
