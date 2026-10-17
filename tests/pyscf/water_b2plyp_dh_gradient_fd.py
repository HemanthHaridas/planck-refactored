"""Exact FD reference for Planck's double-hybrid PT2 gradient correction.

FD the DH energy in PySCF: E(R) = E_KS_hyb(R) + 0.27*E_MP2_on_KS(R). This is
the same total-energy function Planck's own FD driver differentiates, so the
printed `0.27*d(E_corr)/dR` is exactly what Planck's `*corr` must equal (see
docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md, N3.5.7.9).

Why this and not `pyscf.grad.mp2` on the KS object: PySCF's grad/mp2.py builds
its Z-vector `fvind` from `mp._scf.get_veff(mol, dm + dm.T)` -- the full
NONLINEAR KS get_veff on a small non-idempotent trial density, not the linear
gen_response. That CPHF matrix is numerically singular (cond 1.6e18; the stock
Krylov solver raises "Krylov solver failed to converge"), and patching the three
get_veff sites to the linear response still lands 5.3e-3 from the FD truth --
22x worse than Planck. PySCF has no validated DH gradient; do not use it as one.

NB PySCF's vendored libxc has no `B2PLYP` alias, so the functional must be
spelled out in its explicit hybrid form below.

Run:  tests/pyscf/.venv/bin/python tests/pyscf/water_b2plyp_dh_gradient_fd.py

Reference values (STO-3G cart, grids.level=6, h=1e-3 Bohr), 0.27*dE_corr/dR:
    Atom 1 (O):  0.0000000000  0.0000000000  0.0098812248
    Atom 2 (H): -0.0052017424 -0.0000000000 -0.0049406117
    Atom 3 (H):  0.0052017424  0.0000000000 -0.0049406117
Planck reproduces these to 2.4e-4 Ha/Bohr; the shortfall is the open term.
"""
import numpy as np
from pyscf import gto, dft, mp
BASE = [("O",(0.0,0.0,0.0)),("H",(0.758602,0.0,0.504284)),("H",(-0.758602,0.0,0.504284))]
BOHR = 0.52917721092
def energy(coords_ang):
    mol = gto.M(atom=[(a,tuple(c)) for (a,_),c in zip(BASE,coords_ang)],
                basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
    ks = dft.RKS(mol); ks.xc="0.53*HF + 0.47*B88, 0.73*LYP"
    ks.grids.level=6; ks.conv_tol=1e-12
    e = ks.kernel()
    pt = mp.MP2(ks); pt.kernel()
    return e + 0.27*pt.e_corr, e, pt.e_corr
c0 = np.array([c for _,c in BASE])
e0,eks0,ec0 = energy(c0)
print("E_total =",e0," E_KS =",eks0," E_corr =",ec0)
h_bohr = 1e-3; h = h_bohr*BOHR
g = np.zeros((3,3)); gks = np.zeros((3,3)); gc = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        cp=c0.copy(); cp[i,j]+=h; ep,ekp,ecp = energy(cp)
        cm=c0.copy(); cm[i,j]-=h; em,ekm,ecm = energy(cm)
        g[i,j]=(ep-em)/(2*h_bohr); gks[i,j]=(ekp-ekm)/(2*h_bohr); gc[i,j]=0.27*(ecp-ecm)/(2*h_bohr)
def show(t,G):
    print(t)
    for i,r in enumerate(G): print(f"  Atom {i+1}: {r[0]: .10f} {r[1]: .10f} {r[2]: .10f}")
show("FD total DH gradient:",g); show("FD KS-hybrid part:",gks)
show("FD 0.27*d(E_corr)/dR  <-- what Planck's *corr must equal:",gc)
