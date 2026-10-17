"""d/dR{Phi_XC} on a REAL molecular grid, real basis, real libxc.

D6 established the CHANNEL STRUCTURE of this derivative on a toy model
(2 s-functions, 6 hand-placed points, a tanh stand-in for Becke). That was
enough to show the channels sum to FD, but it could not test the MAPPING onto
the C++'s three-way split, because the toy model's "basis" channel moves the
AO centres in rho_P and rho_D TOGETHER while the C++ splits them:

    C++ XC_I   <- basis centres move inside rho_D
    C++ XC_II  <- basis centres move inside rho_P
    C++ XC_III <- grid points translate + Becke weights respond

dh_xc_derivation_check.py's mapping table asserts XC_I is "correctly NOT wired:
the Z-vector carries it". **That assertion has never been tested**, and it is
the difference between two expressions: production wires XC_II alone, while
kXcAll (documented as "the full d/dR{Phi_XC} the FD gate verifies") measures
17x WORSE against the DH gradient's own FD target. Both cannot be right.

This model settles it the only way that is decisive: differentiate the SAME
expression the C++ claims to, on the SAME kind of grid, and require exactness.

Phi_XC(R) = sum_munu D_munu <mu| V_xc[rho_P] |nu>
          = int w(r;R) [ vrho(rho_P) rho_D + 2 vsigma(rho_P) grad_rho_P . grad_rho_D ]

Every R-dependence is explicit: AO centres, grid-point positions (which ride
their owner atom), and Becke weights. P and D are held FIXED as AO matrices --
their orbital response is a separate channel the Z-vector owns, and conflating
it with the geometric derivative is precisely what is being tested.
"""
import numpy as np
from pyscf import gto, dft

# Small enough that a full 6-component FD is affordable, real enough that the
# grid is a genuine Becke/Treutler-Ahlrichs product with real pruning.
ATOMS = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8))]
BASIS = "sto-3g"
XC = "pbe"          # a real GGA, so vsigma is live
GRID_LEVEL = 3


def build_mol(R):
    m = gto.Mole()
    m.atom = [(a, tuple(R[i])) for i, (a, _) in enumerate(ATOMS)]
    m.basis = BASIS
    m.unit = "Bohr"
    m.spin = 1
    m.build(verbose=0)
    return m


def make_grid(R):
    """Real Becke grid: coords AND weights, both R-dependent."""
    m = build_mol(R)
    g = dft.gen_grid.Grids(m)
    g.level = GRID_LEVEL
    g.build()
    return g.coords, g.weights, m


def ao_on(mol, coords):
    """AO values and first derivatives at arbitrary points."""
    return dft.numint.eval_ao(mol, coords, deriv=1)   # (4, npts, nao)


def rho_grad(ao, M):
    """rho and grad_rho on the grid for AO-basis matrix M (symmetric)."""
    v = ao[0]                                   # (npts, nao)
    rho = np.einsum("pi,ij,pj->p", v, M, v)
    grd = 2.0 * np.einsum("xpi,ij,pj->px", ao[1:4], M, v)
    return rho, grd


def xc_derivs(rho, grad):
    """First derivatives of the real functional, from libxc. PySCF's GGA
    eval_xc takes the (4, npts) [rho, dx, dy, dz] array, not (rho, sigma)."""
    rho4 = np.vstack([rho, grad.T])            # (4, npts)
    vxc = dft.libxc.eval_xc(XC, rho4, spin=0, deriv=1)[1]
    return vxc[0], vxc[1]        # vrho, vsigma


def Phi_XC(coords, weights, mol_basis, P, D):
    """The scalar being differentiated. mol_basis supplies the AO CENTRES;
    coords/weights supply the QUADRATURE. Keeping them separate arguments is
    what makes the individual channels reachable."""
    ao = ao_on(mol_basis, coords)
    rP, gP = rho_grad(ao, P)
    rD, gD = rho_grad(ao, D)
    vr, vs = xc_derivs(rP, gP)
    integ = vr * rD + 2.0 * vs * np.einsum("px,px->p", gP, gD)
    return float(np.dot(weights, integ))


def Phi_XC_split(coords, weights, mol_P, mol_D, P, D):
    """Same integrand, but rho_P and rho_D read their AO centres from
    DIFFERENT molecules. Setting mol_P != mol_D is how the C++'s XC_I /
    XC_II split is reproduced exactly rather than approximated."""
    aoP = ao_on(mol_P, coords)
    aoD = ao_on(mol_D, coords)
    rP, gP = rho_grad(aoP, P)
    rD, gD = rho_grad(aoD, D)
    vr, vs = xc_derivs(rP, gP)
    integ = vr * rD + 2.0 * vs * np.einsum("px,px->p", gP, gD)
    return float(np.dot(weights, integ))
