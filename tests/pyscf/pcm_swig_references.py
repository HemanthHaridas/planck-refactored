"""References for the SWIG C-PCM gates (docs/PCM_GEOMETRY_SCOPE.md step 1).

PySCF's default C-PCM (SWIG, lebedev_order 29 = 302 points, vdw_scale 1.2,
modified Bondi, eps 78.3553) on each tests/inputs/regression/*/*pcm_swig*
input, cart=True, Planck's GBS and its CODATA 2018 Bohr; DFT on Planck's
unpruned 150 x 590 grid.
  tests/pyscf/.venv/bin/python tests/pyscf/pcm_swig_references.py
"""
import pathlib
import re

from pyscf import dft, gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)


def field(text, key):
    m = re.search(rf"^\s*{key}\s+(\S+)", text, re.M)
    return m.group(1) if m else None


def run(path):
    text = path.read_text()
    basis, scf_type, exchange = field(text, "basis"), field(text, "scf_type"), field(text, "exchange")
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {line.split()[0] for line in coords[2:]}
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=True, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    if exchange:
        mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
        mf.xc = exchange
        mf.grids.atom_grid = (150, 590)
        mf.grids.prune = None
    else:
        mf = (scf.RHF if scf_type == "rhf" else scf.UHF)(mol)
    mf = mf.PCM()
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.eps = 78.3553
    mf.with_solvent.lebedev_order = 29
    mf.init_guess = "1e"
    mf.conv_tol = 1e-12
    e = mf.kernel()
    assert mf.converged, path.stem
    print(f"{path.stem:42s} E = {e:.10f}  G_pcm = {mf.with_solvent.e:.10f}  "
          f"npts = {len(mf.with_solvent.surface['grid_coords'])}")


for p in sorted(ROOT.glob("tests/inputs/regression/*/*pcm_swig*.hfinp")):
    run(p)
