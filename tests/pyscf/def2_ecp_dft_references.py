"""PySCF references for the def2 ECP DFT gates (docs/DEF2_ECP.md step 6).

Spherical def2-SVP with the basis and ECP read from Planck's own GBS file
(load_ecp from def2_ecp_references.py), Planck's CODATA 2018 Bohr, and the
unpruned 150 x 590 grid of the inputs. Prints, per input: the energy; the
analytic gradient (grid response on) for `calculation gradient`; the lowest UKS
orbital-Hessian eigenvalue (gen_g_hop_uhf, Planck's A + B) when
`stability_check` is on; the four lowest Casida roots for `calculation tddft`.
PySCF's gradient grid response is patched for ECP atoms (see below).
Run: tests/pyscf/.venv/bin/python tests/pyscf/def2_ecp_dft_references.py [case ...]
"""
import pathlib
import re
import sys

import numpy as np
from pyscf import dft, gto, lib
from pyscf.gto.basis import parse_gaussian
from pyscf.soscf import newton_ah

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from def2_ecp_references import load_ecp  # noqa: E402



def patch_pyscf_grid_response_radii():
    """Work around a PySCF defect (docs/PYSCF_ECP_GRID_RESPONSE_SCOPE.md).

    grad.rks.grids_response_becke looks Bragg radii up by mol.atom_charges(),
    which is Z - n_core on an ECP atom, while the energy's Becke partition
    (radi.treutler_atomic_radii_adjust) uses the proton number. The gradient
    then differentiates a different partition (AuH.H2O B3LYP: 1.1e-5 Ha/Bohr
    off PySCF's own energy FD). Present on PySCF master and hotfix as of
    2026-11-06; remove this once upstream is fixed.
    """
    import inspect
    import textwrap
    import pyscf.grad.rks as grks
    from pyscf.dft import radi
    src = textwrap.dedent(inspect.getsource(grks.grids_response_becke))
    old = "charges = mol.atom_charges()"
    if old not in src:
        return  # fixed upstream
    ns = dict(vars(grks))
    ns["elements_proton"] = radi.elements_proton
    exec(src.replace(old, "charges = [elements_proton(x) for x in mol.elements]"), ns)
    grks.grids_response_becke = ns["grids_response_becke"]


patch_pyscf_grid_response_radii()

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/dft"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
HARTREE_EV = 27.211386245988
XC = {"pbe": "pbe,pbe", "b3lyp": "b3lyp"}


def lowest(hop, hdiag, nroots=4):
    x0 = np.zeros((nroots, hdiag.size))
    for k, i in enumerate(np.argsort(hdiag)[:nroots]):
        x0[k, i] = 1.0
    e, _ = lib.davidson(hop, x0, lambda dx, e, x0: dx / (hdiag - e + 1e-8), nroots=nroots,
                        tol=1e-12, max_cycle=200)
    return min(e)


def run(path):
    text = path.read_text()
    key = lambda k: re.search(rf"^\s*{k}\s+(\S+)", text, re.M)
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    gbs = str(ROOT / "basis-sets" / key("basis").group(1))
    elements = {line.split()[0] for line in coords[2:]}
    ecp = {e: load_ecp(gbs, e) for e in elements}
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=False, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements},
                ecp={e: v for e, v in ecp.items() if v})
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
    mf.xc = XC[key("exchange").group(1)]
    mf.grids.atom_grid, mf.grids.prune = (150, 590), None
    guess = {"hcore": "1e", "sad": "atom"}[key("guess").group(1)]
    # The input's density tolerance bounds the orbital gradient too: HI+ has a
    # hole in a degenerate pi pair, so the orbitals rotate freely in that pair
    # and only a loose tolerance (1e-6, as in Planck's input) can be met.
    tol_density = float(key("tol_density").group(1))
    mf.init_guess, mf.conv_tol, mf.conv_tol_grad = guess, 1e-12, max(1e-9, tol_density)
    mf.max_cycle = 300
    e = mf.kernel()
    if not mf.converged:
        # A hole in a degenerate pi pair (HI+) makes DIIS wander inside the
        # degenerate manifold; finish with the second-order solver.
        mf = mf.newton()
        e = mf.kernel(mf.mo_coeff, mf.mo_occ)
    assert mf.converged, path.stem
    print(f"{path.stem:42s} E = {e:.10f}")
    calc = key("calculation").group(1)
    if calc == "gradient":
        g = mf.nuc_grad_method()
        g.grid_response = True
        for i, row in enumerate(g.kernel(), 1):
            print(f"  Atom {i}: " + "  ".join(f"{v:+.12f}" for v in row))
    if key("stability_check"):
        _, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
        print(f"  UKS -> UKS  lambda_min {lowest(hop, hdiag):+.7e}")
    if calc == "tddft":
        td = mf.TDDFT()
        td.nstates, td.conv_tol = 4, 1e-10
        td.kernel()
        for k, (w, f) in enumerate(zip(td.e, td.oscillator_strength()), 1):
            print(f"  root {k}: {w * HARTREE_EV:.7f} eV  f = {f:.6f}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["hi_rks_pbe_def2svp_ecp", "auh_h2o_rks_b3lyp_def2svp_ecp_gradient",
                             "hi_cation_uks_pbe_def2svp_ecp_stability", "hi_rks_b3lyp_def2svp_ecp_tddft"]
    for name in names:
        run(CASES / f"{name}.hfinp")
