"""PySCF references for spherical KS stability (docs/SPHERICAL_DFT.md step 4).

Lowest orbital-Hessian eigenvalue of each channel, cart=False, Planck's own GBS,
its CODATA 2018 Bohr and its unpruned 150 x 590 grid. Planck reports A + B, so
(docs/KS_STABILITY.md, eigenvalue convention):
  UKS -> UKS   gen_g_hop_uhf                  x1
  RKS -> RKS   gen_g_hop_rhf                  /2
  RKS -> UKS   _gen_hop_rhf_external (3rd)    x1
Run: tests/pyscf/.venv/bin/python tests/pyscf/spherical_ks_stability_references.py
"""
import pathlib
import re

import numpy as np
from pyscf import dft, gto, lib
from pyscf.gto.basis import parse_gaussian
from pyscf.scf import stability
from pyscf.soscf import newton_ah

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/spherical"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
XC = {"b3lyp": "b3lyp", "pbe": "pbe,pbe"}


def lowest(hop, hdiag, nroots=4):
    # Four roots: a symmetry-breaking mode is reached only from starts that touch it.
    x0 = np.zeros((nroots, hdiag.size))
    for k, i in enumerate(np.argsort(hdiag)[:nroots]):
        x0[k, i] = 1.0
    e, _ = lib.davidson(hop, x0,
                        lambda dx, e, x0: dx / (hdiag - e + 1e-8), nroots=nroots,
                        tol=1e-12, max_cycle=200)
    return min(e)


def run(path):
    text = path.read_text()
    basis = re.search(r"^\s*basis\s+(\S+)", text, re.M).group(1)
    exchange = re.search(r"^\s*exchange\s+(\S+)", text, re.M).group(1)
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    atoms = "\n".join(coords[2:])
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {line.split()[0] for line in coords[2:]}
    mol = gto.M(atom=atoms, unit=BOHR_ANGSTROM, cart=False, verbose=0, charge=charge,
                spin=mult - 1, basis={e: parse_gaussian.load(gbs, e) for e in elements})
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
    mf.xc = XC[exchange]
    mf.grids.atom_grid = (150, 590)
    mf.grids.prune = None
    mf.init_guess = "1e"  # Planck's guess hcore
    mf.conv_tol = 1e-12
    e = mf.kernel()
    assert mf.converged, path.stem
    print(f"{path.stem:42s} E = {e:.10f}")
    if mult == 1:
        _, hop, hdiag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
        print(f"  RKS -> RKS  (singlet) {lowest(hop, hdiag) / 2:+.7f}")
        _, _, hop, hdiag = stability._gen_hop_rhf_external(mf)
        print(f"  RKS -> UKS  (triplet) {lowest(hop, hdiag):+.7f}")
    else:
        _, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
        print(f"  UKS -> UKS            {lowest(hop, hdiag):+.7f}")


for name in ("water_rks_b3lyp_spherical_ccpvtz", "water_cation_uks_pbe_spherical_ccpvtz",
             "h2o2_cation_uks_pbe_spherical_631gd"):
    run(CASES / f"{name}.hfinp")


def broken_symmetry_follow(path):
    """RKS, its triplet eigenvalue, then UKS from a HOMO/LUMO-mixed guess (+-45 deg)."""
    text = path.read_text()
    basis = re.search(r"^\s*basis\s+(\S+)", text, re.M).group(1)
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    gbs = str(ROOT / "basis-sets" / basis)
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=False, verbose=0,
                basis={"H": parse_gaussian.load(gbs, "H")})
    rks = dft.RKS(mol, xc="pbe,pbe")
    rks.grids.atom_grid, rks.grids.prune = (150, 590), None
    rks.init_guess, rks.conv_tol = "1e", 1e-12
    print(f"{path.stem:42s} E(RKS) = {rks.kernel():.10f}")
    _, _, hop, hdiag = stability._gen_hop_rhf_external(rks)
    print(f"  RKS -> UKS  (triplet) {lowest(hop, hdiag):+.7f}")
    homo = int(np.count_nonzero(rks.mo_occ)) - 1
    mos = []
    for sign in (1.0, -1.0):
        c = rks.mo_coeff.copy()
        h, l = c[:, homo].copy(), c[:, homo + 1].copy()
        c[:, homo] = (h + sign * l) / np.sqrt(2.0)
        c[:, homo + 1] = (l - sign * h) / np.sqrt(2.0)
        mos.append(c)
    uks = dft.UKS(mol, xc="pbe,pbe")
    uks.grids.atom_grid, uks.grids.prune = (150, 590), None
    uks.conv_tol = 1e-12
    occ = np.array([rks.mo_occ / 2, rks.mo_occ / 2])
    e = uks.kernel(uks.make_rdm1(mos, occ))
    s2 = uks.spin_square()[0]
    print(f"  broken-symmetry UKS   E = {e:.10f}  <S^2> = {s2:.4f}")


broken_symmetry_follow(CASES / "h2_rks_pbe_triplet_follow_spherical_ccpvdz.hfinp")
