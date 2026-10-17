"""Explicit differentiable model of Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>.

Every R-dependence is written out: basis centres, grid points (which move with
their owner atom), and Becke weights. Nothing is held fixed by convention.

This is the object whose d/dR the DH PT2 gradient's XC term is supposed to be.
Small enough (2 atoms, 1 s-function each, a handful of grid points) that FD is
exact to ~1e-9 and every channel can be isolated.
"""
import numpy as np

# ---- basis: one normalised s-Gaussian per atom -------------------------------
ALPHA = np.array([0.8, 1.3])

def phi(r, R, mu):
    """AO mu evaluated at point r (both 3-vectors); R = (natom,3) centres."""
    a = ALPHA[mu]
    d = r - R[mu]
    return (2*a/np.pi)**0.75 * np.exp(-a*(d@d))

def grad_phi_r(r, R, mu):
    """d phi / d r  (gradient wrt the FIELD POINT)."""
    a = ALPHA[mu]
    d = r - R[mu]
    return -2*a*d * phi(r, R, mu)

# ---- grid: points ride with their owner atom, Becke weights depend on R ------
# offsets are fixed relative to the owner: r_p = R[owner_p] + offset_p
OFFS = np.array([[0.30,-0.20, 0.10],[-0.15, 0.25,-0.30],[ 0.05, 0.10, 0.40],
                 [-0.35,-0.10, 0.20],[ 0.20, 0.30,-0.15],[ 0.10,-0.40, 0.05]])
OWNER = np.array([0,0,0,1,1,1])
WBASE = np.array([0.9,1.1,1.0,0.8,1.2,1.05])

def becke_w(R, p):
    """A smooth, genuinely R-dependent partition weight (stands in for Becke).
    Real Becke uses inter-atomic distances; so does this."""
    r = R[OWNER[p]] + OFFS[p]
    d0 = np.linalg.norm(r - R[0]); d1 = np.linalg.norm(r - R[1])
    mu = (d0 - d1)/np.linalg.norm(R[0]-R[1])
    s = 0.5*(1 - np.tanh(2.0*mu))          # smooth switching, in (0,1)
    return WBASE[p] * (s if OWNER[p] == 0 else (1-s))

def grid(R):
    return [(R[OWNER[p]] + OFFS[p], becke_w(R, p)) for p in range(len(OFFS))]

# ---- densities on the grid ---------------------------------------------------
def rho_and_grad(r, R, M):
    """rho(r) and grad_r rho(r) for AO-basis matrix M (symmetric)."""
    v  = np.array([phi(r, R, m) for m in range(len(ALPHA))])
    gv = np.array([grad_phi_r(r, R, m) for m in range(len(ALPHA))])
    rho = v @ M @ v
    grd = 2.0 * (gv.T @ (M @ v))
    return rho, grd

# ---- a GGA-like functional; only its derivatives matter ----------------------
def f_xc(rho, sigma):
    return -0.75*rho**(4/3) - 0.05*sigma/(1.0+rho)

def d_f(rho, sigma):
    """ANALYTIC first derivatives of f_xc -- no nested FD."""
    vr = -1.0*rho**(1/3) + 0.05*sigma/(1.0+rho)**2
    vs = -0.05/(1.0+rho)
    return vr, vs

def Phi_XC(R, P, D):
    """sum_munu D_munu <mu| V_xc[rho_P] |nu>, as a grid quadrature.
       <mu|V_xc|nu> = int w [ vrho*phi_mu*phi_nu
                            + 2*vsigma*(grad rho_P . grad(phi_mu phi_nu)) ]"""
    tot = 0.0
    for r, w in grid(R):
        rP, gP = rho_and_grad(r, R, P)
        rD, gD = rho_and_grad(r, R, D)
        vr, vs = d_f(rP, gP@gP)
        tot += w * (vr*rD + 2.0*vs*(gP@gD))
    return tot
