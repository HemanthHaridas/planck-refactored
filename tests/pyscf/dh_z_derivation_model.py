"""Differentiable model whose energy is NON-VARIATIONAL in C, so a Z-vector is
forced rather than assumed.

Setup mirrors the DH case: a variational "SCF" fixes C (minimising E_scf), then
a second functional E_pt2 is evaluated on those C WITHOUT re-optimising. dE/dR
then needs dC/dR, and the Z-vector is the device that avoids computing it.

3 AOs / 1 occ / 2 virt -- enough for a real ov block and a 2-component Z.
"""
import numpy as np

ALPHA = np.array([0.7, 1.1, 1.7])
NOCC  = 1

def S_ao(R):
    n=len(ALPHA); S=np.zeros((n,n))
    for m in range(n):
        for k in range(n):
            a,b=ALPHA[m],ALPHA[k]; d=R[m]-R[k]
            S[m,k]=(2*np.sqrt(a*b)/(a+b))**1.5*np.exp(-a*b/(a+b)*(d@d))
    return S

def H_ao(R):
    n=len(ALPHA); H=np.zeros((n,n))
    S=S_ao(R)
    for m in range(n):
        for k in range(n):
            a,b=ALPHA[m],ALPHA[k]; d=R[m]-R[k]
            H[m,k]=-(a+b)*0.30*np.exp(-0.5*a*b/(a+b)*(d@d))-0.35*S[m,k]
    return H

def scf(R):
    """Variational: C minimises Tr[P H] subject to C^T S C = I."""
    S=S_ao(R); H=H_ao(R)
    s,U=np.linalg.eigh(S); X=U@np.diag(s**-0.5)@U.T
    eps,Cp=np.linalg.eigh(X@H@X)
    return X@Cp, eps

def V_ao(R):
    """A geometry-dependent 2-index 'interaction' standing in for the ERIs."""
    n=len(ALPHA); V=np.zeros((n,n))
    for m in range(n):
        for k in range(n):
            a,b=ALPHA[m],ALPHA[k]; d=R[m]-R[k]
            V[m,k]=0.6*np.exp(-0.8*a*b/(a+b)*(d@d))
    return V

def E_pt2(R, C=None, eps=None):
    """MP2-SHAPED and GAUGE-INVARIANT: sum_ia |v_ia|^2 / (eps_i - eps_a).

    Invariant under occ-occ and virt-virt rotations because the numerator is a
    full o-v block contraction and the denominator uses the eigenvalues. That
    invariance is what makes a Z-vector formulation well posed -- the previous
    version of this model lacked it and was ill-posed (see the scope doc)."""
    if C is None: C,eps = scf(R)
    if eps is None: _,eps = scf(R)
    v = C.T @ V_ao(R) @ C
    tot=0.0
    for i in range(NOCC):
        for a in range(NOCC,len(ALPHA)):
            tot += v[i,a]**2/(eps[i]-eps[a])
    return float(tot)
