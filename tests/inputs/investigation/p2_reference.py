"""P2 reference: PySCF RCCSDTQ + FCI for the two viable systems.

Compare Planck's generated rank-4 energy against RCCSDTQ (NOT FCI). The
CCSDTQ-FCI gap is printed to confirm the exactness confound is absent.
"""
from math import comb
from pyscf import gto, scf, fci
from pyscf.cc import rccsdtq

SYSTEMS = {
    "ch4_sto3g": ("C 0 0 0; H 0 0 1.09; H 1.03 0 -0.36; "
                  "H -0.51 0.89 -0.36; H -0.51 -0.89 -0.36", "sto-3g"),
    "h2o_631g":  ("O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24", "6-31g"),
}

for name, (atom, basis) in SYSTEMS.items():
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mol.cart = True                       # match Planck's basis_type cartesian
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
    no = mol.nelectron // 2; nv = mol.nao - no
    q = comb(no, 4) * comb(nv, 4) if no >= 4 and nv >= 4 else 0
    print(f"=== {name}  no={no} nv={nv}  distinct quadruples={q} ===")
    print(f"  RHF     = {mf.e_tot:.10f}")
    c = rccsdtq.RCCSDTQ(mf); c.verbose = 0; c.kernel()
    e_q = mf.e_tot + c.e_corr
    print(f"  RCCSDTQ = {e_q:.10f}   <-- COMPARE PLANCK AGAINST THIS")
    try:
        e_fci = fci.FCI(mf).kernel()[0]
        print(f"  FCI     = {e_fci:.10f}")
        print(f"  CCSDTQ - FCI = {e_q - e_fci:+.3e}  (nonzero => not an exactness case)")
    except Exception as exc:
        print(f"  FCI     skipped: {type(exc).__name__}")
    print()
