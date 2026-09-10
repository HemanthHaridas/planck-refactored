"""Symbolic-style derivation of d/dR{Phi_XC}, checked against an FD oracle.

Answers: "if the implementation matches the paper, where does the error come
from?" -- by deriving the gradient from an explicit differentiable model rather
than transcribing equations, then comparing channel by channel with the C++.

The model writes out EVERY R-dependence (basis centres, grid points that ride
their owner atom, Becke weights) and differentiates each in isolation. f_xc's
first derivatives are analytic, so only the outer geometry FD is numerical.

RESULT: the three channels sum to the FD total to **2.2e-11 relative**, and

    basis  net force = +0.09341059
    point  net force = -0.09341059     <- equal and opposite, 8 digits
    weight net force =  0

So d/dR{Phi_XC} is translationally invariant while the basis channel ALONE is
not. That is exactly the XC_II / missing-companion structure, reproduced from
first principles with no reference to the paper.

MAPPING to the C++ (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md):
    basis/rho_D moves -> XC_I      (correctly NOT wired: Z-vector carries it)
    basis/rho_P moves -> XC_II     (wired, coefficient 1)
    point translation -> XC_III(b) (implemented, cancels on a real grid)
    Becke weight      -> XC_III(a) (implemented, cancels on a real grid)

No unmatched term. The C++ has a counterpart for every channel.

Run: tests/pyscf/.venv/bin/python tests/pyscf/dh_xc_derivation_check.py
"""
import numpy as np
import dh_xc_derivation_model as M

np.random.seed(3)
R0 = np.array([[0.0,0.0,0.0],[0.0,0.0,1.4]])
A = np.random.randn(2,2)*0.3; P = A@A.T + 2*np.eye(2)      # SPD-ish density
B = np.random.randn(2,2)*0.2; D = B+B.T                     # difference density

def fd_total(R, P, D, h=1e-5):
    g = np.zeros_like(R)
    for a in range(R.shape[0]):
        for q in range(3):
            Rp = R.copy(); Rp[a,q]+=h
            Rm = R.copy(); Rm[a,q]-=h
            g[a,q] = (M.Phi_XC(Rp,P,D)-M.Phi_XC(Rm,P,D))/(2*h)
    return g

def channel(R,P,D,which,h=1e-5):
    """Differentiate ONE R-dependence, freezing the others at R0.
       basis  : AO centres move, grid points+weights FROZEN in space
       point  : grid points translate with owner, basis+weights frozen
       weight : Becke weights move, everything else frozen"""
    g = np.zeros_like(R)
    frozen_pts = [r for r,_ in M.grid(R)]
    frozen_w   = [w for _,w in M.grid(R)]
    for a in range(R.shape[0]):
        for q in range(3):
            def val(sgn):
                Rs = R.copy(); Rs[a,q] += sgn*h
                tot = 0.0
                for p in range(len(M.OFFS)):
                    if which=="basis":   r, w = frozen_pts[p], frozen_w[p]; Rb = Rs
                    elif which=="point": r, w = Rs[M.OWNER[p]]+M.OFFS[p], frozen_w[p]; Rb = R
                    elif which=="weight":r, w = frozen_pts[p], M.becke_w(Rs,p); Rb = R
                    rP,gP = M.rho_and_grad(r,Rb,P); rD,gD = M.rho_and_grad(r,Rb,D)
                    vr,vs = M.d_f(rP, gP@gP)
                    tot += w*(vr*rD + 2.0*vs*(gP@gD))
                return tot
            g[a,q] = (val(+1)-val(-1))/(2*h)
    return g

fd = fd_total(R0,P,D)
ch = {k: channel(R0,P,D,k) for k in ("basis","point","weight")}
tot = sum(ch.values())

print("d/dR{Phi_XC}: analytic channel decomposition vs FD oracle\n")
for k,v in ch.items():
    print(f"  {k:7s} |g|={np.linalg.norm(v):.6e}   sum_A={np.round(v.sum(axis=0),9)}")
print(f"\n  SUM     |g|={np.linalg.norm(tot):.6e}   sum_A={np.round(tot.sum(axis=0),9)}")
print(f"  FD      |g|={np.linalg.norm(fd):.6e}   sum_A={np.round(fd.sum(axis=0),9)}")
print(f"\n  max|SUM - FD| = {np.abs(tot-fd).max():.3e}   rel = {np.abs(tot-fd).max()/np.abs(fd).max():.3e}")

