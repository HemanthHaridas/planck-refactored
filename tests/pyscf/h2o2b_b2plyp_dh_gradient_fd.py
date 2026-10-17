"""Second, independently distorted C1 H2O2 -- the CROSS-FIXTURE control.

Companion to h2o2_b2plyp_dh_gradient_fd.py. A 12-component cos-similarity is
still not decisive on its own; what separates a real term from a fit artifact
is scoring the SAME coefficient on two independent geometries.

Validated on the known answer: XC_II scores cos 0.946 / scale 1.078 on
fixture 1 and cos 0.942 / scale 1.015 here. Three candidates for the remaining
31% were killed by this test (see N3.5.7.11), including one that looked
promising on fixture 1 alone (vhf_s1occ KS swap: scale -1.11 there, -0.59 here).

Energy cross-check: PySCF E_corr -0.0868897932 vs Planck -0.0868898054.
Reference 0.27*dE_corr/dR:
    Atom 1 (O):  0.0078814736 -0.0150026978  0.0070023282
    Atom 2 (O): -0.0067137931  0.0163644737  0.0020907430
    Atom 3 (H): -0.0074556688 -0.0015844464 -0.0051435764
    Atom 4 (H):  0.0062879882  0.0002226714 -0.0039494945
"""
import numpy as np
from pyscf import gto, dft, mp
BASE = [("O",(0.021,0.690,-0.093)),("O",(-0.037,-0.702,0.061)),("H",(0.792,0.951,0.402)),("H",(-0.741,-0.859,0.487))]
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
