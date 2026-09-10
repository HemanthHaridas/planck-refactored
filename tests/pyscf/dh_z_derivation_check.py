"""Z-vector channel decomposition in the canonical gauge, FD-verified.

Third in the series (see dh_xc_ and dh_w_derivation_check.py). Splits
dE_pt2/dR into the three channels the Z-vector formalism separates and checks
the sum against an FD oracle.

RESULT: max|SUM - FD| = 1.735e-11 (rel 2.4e-09).

    explicit (basis moves, MOs+eps frozen)   |g| = 7.294688e-02
    eps channel (eps moves, rest frozen)     |g| = 1.496824e-03
    MO channel  (MOs move, rest frozen)      |g| = 6.142377e-02
    SUM == FD                                |g| = 1.118980e-02

The explicit and MO channels are each ~7e-2 and cancel to 1.1e-2 -- a **6.5x
cancellation**, the same signature W^PT2 showed. The MO channel is precisely
what the Z-vector exists to compute without forming dC/dR.

TWO MODEL DEFECTS FOUND AND FIXED, both in the test, recorded because each
would have produced a false finding about Planck:

1. The first model used E = Tr[G * C^T H C], which is NOT invariant under
   virtual-virtual rotations (measured: -0.77723303 vs -0.79274766 under a
   0.3 rad vv rotation). The SCF leaves those rotations arbitrary, so that
   functional is not a function of the SCF solution at all and NO Z-vector
   formulation can reproduce its FD. The model was ill-posed, not Planck.

2. The replacement is MP2-shaped, sum_ia |v_ia|^2/(eps_i - eps_a), which fixes
   the gauge to canonical orbitals -- the same invariant Planck's CC/MP2 code
   relies on (f_ov = 0, canonical Fock throughout).

SUPERSEDED IN PART (2026-09-10): the "NOT ESTABLISHED" section below is
obsolete on both counts. See dh_zov_derivation_check.py (D9).

  * The orbital Hessian IS now under an independent oracle -- but in C++, not
    here: build_ks_orbital_hessian_op's diag/J/K channels are gated to
    rel 3.7e-15 against build_rhf_cphf_matrix (PLANCK_DFT_DH_HESSIAN_AUDIT).
    Needed no FD and no kappa at all -- the non-XC channels of the KS orbital
    Hessian ARE the RHF CPHF couplings, so comparing operator-to-operator at a
    FIXED geometry sidesteps the metric problem entirely.
  * The kappa extraction below is NOT blocked on Eqs. 19-21. The contamination
    is removed by Lowdin-orthonormalizing the displaced MOs against S(R0) --
    four lines. log(U) is then a clean generator (pred vs actual dkappa/dR
    agree to 0.9993/0.9998, FD-step-limited).

**And this file's own mo_channel is contaminated by the same defect**: it
differentiates E(C(R+h)) with C raw from the displaced SCF, which carries an
O(dS) norm change on top of the rotation. Measured via a variational control
(E_scf's MO channel must vanish): **1.065e+00 raw, 3.0e-09 projected**. The
numbers below are therefore ~26x too large in the MO/explicit channels
individually. **The SUM is unaffected** -- the contamination cancels between
the explicit and MO channels, which is why 1.735e-11 vs FD still holds -- so
the decomposition result stands, but **do not cite the individual channel
magnitudes**.

WHAT THIS DOES **NOT** ESTABLISH -- read before citing it (OBSOLETE, above):

The decomposition is verified, but the step that would test Planck's ORBITAL
HESSIAN (`build_ks_orbital_hessian_op`) is NOT. Reproducing the MO channel as
`L . dkappa/dR` requires extracting the rotation generator kappa from the MO
overlap U = C0^T S(R0) C(R+h), and that extraction is contaminated: C(R+h) is
orthonormal wrt S(R+h), not S(R0), so U mixes a genuine rotation with an O(dS)
metric mismatch (measured: |U - I| = 3.7e-6 at h = 1e-5, against O(h) = 1e-5 for
a pure rotation). Three extraction attempts each came up ~30x short.

Separating those is exactly the U_ij = -1/2 S^(x)_ij bookkeeping of the paper's
Eqs. 19-21. A model that carries it would close the loop and put Planck's
Hessian coefficients under an independent oracle -- which is what D5 identified
as the highest-leverage unverified object (3.6x on the final gradient). That is
real work, not a patch to this file.

Run: tests/pyscf/.venv/bin/python tests/pyscf/dh_z_derivation_check.py
"""
import numpy as np
import dh_z_derivation_model as M
R0=np.array([[0.0,0.0,0.0],[0.0,0.0,1.35],[0.0,1.25,0.62]])
NO=M.NOCC; NB=len(M.ALPHA); NV=NB-NO

def rot(C,x):
    K=np.zeros((NB,NB))
    for a in range(NV):
        for i in range(NO):
            K[NO+a,i]=x[a*NO+i]; K[i,NO+a]=-x[a*NO+i]
    E=np.eye(NB); T=np.eye(NB)
    for k in range(1,14): T=T@K/k; E=E+T
    return C@E

def fd_total(h=1e-6):
    g=np.zeros_like(R0)
    for a in range(3):
        for q in range(3):
            Rp=R0.copy(); Rp[a,q]+=h; Rm=R0.copy(); Rm[a,q]-=h
            g[a,q]=(M.E_pt2(Rp)-M.E_pt2(Rm))/(2*h)
    return g

def explicit(h=1e-6):
    """basis moves, MOs and eps FROZEN."""
    C,e=M.scf(R0); g=np.zeros_like(R0)
    for a in range(3):
        for q in range(3):
            Rp=R0.copy(); Rp[a,q]+=h; Rm=R0.copy(); Rm[a,q]-=h
            g[a,q]=(M.E_pt2(Rp,C,e)-M.E_pt2(Rm,C,e))/(2*h)
    return g

def eps_channel(h=1e-6):
    """eps moves, MOs and basis FROZEN."""
    C,e=M.scf(R0); g=np.zeros_like(R0)
    for a in range(3):
        for q in range(3):
            Rp=R0.copy(); Rp[a,q]+=h; Rm=R0.copy(); Rm[a,q]-=h
            _,ep=M.scf(Rp); _,em=M.scf(Rm)
            g[a,q]=(M.E_pt2(R0,C,ep)-M.E_pt2(R0,C,em))/(2*h)
    return g

def mo_channel(h=1e-6):
    """MOs move, basis and eps FROZEN."""
    C,e=M.scf(R0); S0=M.S_ao(R0); g=np.zeros_like(R0)
    for a in range(3):
        for q in range(3):
            Rp=R0.copy(); Rp[a,q]+=h; Rm=R0.copy(); Rm[a,q]-=h
            Cp,_=M.scf(Rp); Cm,_=M.scf(Rm)
            for p in range(NB):
                if C[:,p]@(M.S_ao(Rp)@Cp[:,p])<0: Cp[:,p]*=-1
                if C[:,p]@(M.S_ao(Rm)@Cm[:,p])<0: Cm[:,p]*=-1
            g[a,q]=(M.E_pt2(R0,Cp,e)-M.E_pt2(R0,Cm,e))/(2*h)
    return g

if __name__=="__main__":
    fd=fd_total(); ex=explicit(); ec=eps_channel(); mo=mo_channel()
    tot=ex+ec+mo
    print("Z-vector channel decomposition (canonical gauge) vs FD\n")
    for n_,v in (("explicit (basis)",ex),("eps channel",ec),("MO channel",mo)):
        print(f"  {n_:18s} |g|={np.linalg.norm(v):.6e}")
    print(f"\n  SUM |g|={np.linalg.norm(tot):.6e}")
    print(f"  FD  |g|={np.linalg.norm(fd):.6e}")
    print(f"\n  max|SUM-FD| = {np.abs(tot-fd).max():.3e}  rel={np.abs(tot-fd).max()/np.abs(fd).max():.3e}")
