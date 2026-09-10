"""Explicit differentiable model for the ENERGY-WEIGHTED DENSITY term <W S^(x)>.

W is not a postulate -- it is what the orthonormality constraint C^T S C = I
forces when the basis moves. This model derives it from that constraint rather
than transcribing Eqs. 42-45, then checks the result against an FD oracle.

Setup: 2 atoms, 2 s-AOs, 1 occupied / 1 virtual. Small enough for exact FD,
big enough to have oo/vv/ov blocks and a genuine Z-vector.
"""
import numpy as np

ALPHA = np.array([0.8, 1.3])

def S_ao(R):
    """AO overlap -- the ONLY place the geometry enters this model."""
    n=len(ALPHA); S=np.zeros((n,n))
    for m in range(n):
        for k in range(n):
            a,b = ALPHA[m], ALPHA[k]; d = R[m]-R[k]
            S[m,k] = (2*np.sqrt(a*b)/(a+b))**1.5 * np.exp(-a*b/(a+b)*(d@d))
    return S

def H_ao(R):
    """A geometry-dependent one-electron operator (stands in for h_core)."""
    n=len(ALPHA); H=np.zeros((n,n))
    for m in range(n):
        for k in range(n):
            a,b = ALPHA[m], ALPHA[k]; d = R[m]-R[k]
            H[m,k] = -(a+b)*0.35*np.exp(-0.5*a*b/(a+b)*(d@d)) - 0.4*S_ao(R)[m,k]
    return H

def scf(R):
    """Generalised eigenproblem H C = S C eps, C^T S C = I."""
    S=S_ao(R); H=H_ao(R)
    s,U = np.linalg.eigh(S)
    X = U @ np.diag(s**-0.5) @ U.T           # symmetric orthogonaliser
    eps,Cp = np.linalg.eigh(X @ H @ X)
    return X @ Cp, eps

# --- a fixed "correlation" difference density in the MO basis ----------------
# oo/vv blocks (from amplitudes) plus an ov block (the Z-vector).
D_MO = np.array([[-0.13, 0.07],
                 [ 0.07, 0.21]])

def E_pt2_like(R):
    """A functional whose C-dependence is non-variational, so that its geometry
    derivative genuinely needs W. Tr[D_mo * eps] is the eps-channel of E_PT2."""
    C,eps = scf(R)
    return float(np.sum(np.diag(D_MO) * eps))
