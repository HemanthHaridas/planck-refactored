"""Exact FD reference for Planck's DH PT2 gradient -- LOW-SYMMETRY fixture.

Same recipe as water_b2plyp_dh_gradient_fd.py, on a distorted C1 H2O2. The
point is the DEGREES OF FREEDOM: water/C2v leaves only 3 independent gradient
components, so any 3 candidate terms span the target exactly and a
least-squares fit is meaningless (it returned coefficients -0.99 / 8.4 / 23011
with residual 1e-18). H2O2/C1 gives 12 components, 6 independent after
translation+rotation -- enough to IDENTIFY a term rather than merely fit it.

That is what settled N3.5.7.10: XC_II scores cos 0.946 at coefficient 1.078
against the target here, where on water it looked like an ambiguous 0.62.

Energy cross-check (STO-3G cart, grids.level=6):
    PySCF E_corr -0.1005921150   Planck -0.1005920516   (6.3e-8)
Reference 0.27*dE_corr/dR:
    Atom 1 (O):  0.0108845041 -0.0187604810  0.0065567734
    Atom 2 (O): -0.0102507639  0.0184098059  0.0024251590
    Atom 3 (H): -0.0087020440 -0.0000122891 -0.0053664809
    Atom 4 (H):  0.0080683038  0.0003629645 -0.0036154513
"""
import numpy as np
from pyscf import gto, dft, mp
BASE = [("O",(0.0,0.717,-0.041)),("O",(0.021,-0.717,0.032)),("H",(0.851,0.887,0.463)),("H",(-0.802,-0.911,0.395))]
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
g = np.zeros((4,3)); gks = np.zeros((4,3)); gc = np.zeros((4,3))
for i in range(4):
    for j in range(3):
        cp=c0.copy(); cp[i,j]+=h; ep,ekp,ecp = energy(cp)
        cm=c0.copy(); cm[i,j]-=h; em,ekm,ecm = energy(cm)
        g[i,j]=(ep-em)/(2*h_bohr); gks[i,j]=(ekp-ekm)/(2*h_bohr); gc[i,j]=0.27*(ecp-ecm)/(2*h_bohr)
def show(t,G):
    print(t)
    for i,r in enumerate(G): print(f"  Atom {i+1}: {r[0]: .10f} {r[1]: .10f} {r[2]: .10f}")
show("FD total DH gradient:",g); show("FD KS-hybrid part:",gks)
show("FD 0.27*d(E_corr)/dR  <-- what Planck's *corr must equal:",gc)
