"""Derive W^PT2 from ORTHONORMALITY, check <W S^(x)> against an FD oracle.

Companion to dh_xc_derivation_check.py. W is not a postulate -- it is what
C^T S C = I forces when the basis moves -- so this derives it from that
constraint instead of transcribing Eqs. 42-45, then compares with the C++.

  d eps_p/dR = <p|dH/dR|p> - eps_p <p|dS/dR|p>     (generalised Hellmann-Feynman)

so for E = sum_p D_pp eps_p:

  dE/dR = sum_p D_pp <p|dH/dR|p>        <- "operator moves"
        - sum_p D_pp eps_p <p|dS/dR|p>  <- orthonormality == -<W S^(x)>

RESULT: max|SUM - FD| = 2.475e-12 (rel 1.0e-10). The two channels are each
~0.2 and cancel to 0.035 -- a 6x cancellation, so W's sign and factor are
tightly constrained, which is what makes the mutation test below meaningful.

VERDICT vs Planck (see the companion mutation tests in the scope doc):
  * the DIAGONAL (oo/vv) zeta weight is VERIFIED -- Planck's 0.5*(eps_p+eps_q)
    reproduces the oracle at 2.475e-12, while a 2x factor mutation fails at
    1.4e-1 and a sign flip at 2.9e-1.
  * the OFF-DIAGONAL (ov/vo) blocks are BEYOND this model: constraining them
    needs a functional non-variational in C, which then needs a Z-vector the
    model does not have. Measured directly -- diagonal D passes at 1.1e-11
    while full D fails at 1.8e-3, and that residual IS the missing dC/dR
    channel, not a defect in Planck's W.

Run: tests/pyscf/.venv/bin/python tests/pyscf/dh_w_derivation_check.py
"""
import numpy as np
import dh_w_derivation_model as M

R0 = np.array([[0.0,0.0,0.0],[0.0,0.0,1.4]])

def fd_total(R,h=1e-6):
    g=np.zeros_like(R)
    for a in range(R.shape[0]):
        for q in range(3):
            Rp=R.copy(); Rp[a,q]+=h; Rm=R.copy(); Rm[a,q]-=h
            g[a,q]=(M.E_pt2_like(Rp)-M.E_pt2_like(Rm))/(2*h)
    return g

def dmat(fn,R,a,q,h=1e-6):
    Rp=R.copy(); Rp[a,q]+=h; Rm=R.copy(); Rm[a,q]-=h
    return (fn(Rp)-fn(Rm))/(2*h)

def channels(R):
    C,eps = M.scf(R)
    n=len(eps)
    # W in the AO basis: C (D .* zeta) C^T with zeta_pp = eps_p
    zeta = np.zeros((n,n))
    for p in range(n): zeta[p,p] = eps[p]
    W_ao = C @ (M.D_MO*zeta) @ C.T
    op = np.zeros_like(R); ov = np.zeros_like(R)
    for a in range(R.shape[0]):
        for q in range(3):
            dH = dmat(M.H_ao,R,a,q); dS = dmat(M.S_ao,R,a,q)
            # operator-moves channel: sum_p D_pp <p|dH|p>
            op[a,q] = float(np.sum(np.diag(M.D_MO) * np.diag(C.T@dH@C)))
            # orthonormality channel: -<W S^(x)>
            ov[a,q] = -float(np.sum(W_ao*dS))
    return op, ov, W_ao

fd = fd_total(R0)
op, ov, W = channels(R0)
tot = op+ov
print("W^PT2 derivation: channels vs FD oracle\n")
print(f"  operator-moves  <D dH>   |g|={np.linalg.norm(op):.6e}")
print(f"  orthonormality -<W S^x>  |g|={np.linalg.norm(ov):.6e}")
print(f"\n  SUM  |g|={np.linalg.norm(tot):.6e}")
print(f"  FD   |g|={np.linalg.norm(fd):.6e}")
print(f"\n  max|SUM - FD| = {np.abs(tot-fd).max():.3e}  rel = {np.abs(tot-fd).max()/max(np.abs(fd).max(),1e-30):.3e}")
print("\n  W_ao (derived) =\n", np.round(W,8))

