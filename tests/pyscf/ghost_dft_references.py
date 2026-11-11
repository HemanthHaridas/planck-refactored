"""References for the ghost-atom DFT gates (docs/DFT_COUNTERPOISE_SCOPE.md step 1).

PySCF RKS/UKS with cart=True, Planck's GBS, its CODATA 2018 Bohr and its
unpruned 150 x 590 grid, for tests/inputs/regression/dft/*ghost*.hfinp.

A ghost atom's grid is built with its element's proton number, as in Planck
and in PySCF >= 2.14 (pyscf/pyscf#3302). PySCF 2.13 built it as Z = 0 (radial
count, Treutler xi, pruning and Becke radius), which #3302 fixed as a bug. To
get the fixed behaviour on 2.13 the grid is built on the same molecule with
the ghosts made real; the electronic problem keeps the ghosts.
  tests/pyscf/.venv/bin/python tests/pyscf/ghost_dft_references.py
"""
import pathlib
import re

from pyscf import dft, gto
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/dft"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
XC = {"pbe": "pbe", "b3lyp": "b3lypg"}  # Planck's B3LYP is libxc's (VWN-RPA)


def field(text, key):
    return re.search(rf"^\s*{key}\s+(\S+)", text, re.M).group(1)


def run(path):
    text = path.read_text()
    basis, exchange, guess = field(text, "basis"), field(text, "exchange"), field(text, "guess")
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    atoms = [line.split() for line in coords[2:]]
    real = [(re.sub(r"^Gh\((\w+)\)$", r"\1", a[0]), a[1:]) for a in atoms]
    ghost = [("ghost-" + s if a[0].startswith("Gh(") else s, xyz) for a, (s, xyz) in zip(atoms, real)]
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {s for s, _ in real}
    bas = {e: parse_gaussian.load(gbs, e) for e in elements}

    def mol(spec, chg, spin):
        return gto.M(atom="\n".join(f"{s} {' '.join(x)}" for s, x in spec), unit=BOHR_ANGSTROM,
                     cart=True, verbose=0, charge=chg, spin=spin, basis=bas)

    grids = dft.gen_grid.Grids(mol(real, 0, 0))
    grids.atom_grid = (150, 590)
    grids.prune = None
    grids.build()

    m = mol(ghost, charge, mult - 1)
    mf = (dft.RKS if mult == 1 else dft.UKS)(m)
    mf.xc = XC[exchange]
    mf.grids = grids
    mf.init_guess = "minao" if guess == "sad" else "1e"
    mf.conv_tol = 1e-12
    e = mf.kernel()
    assert mf.converged
    print(f"{path.stem}: E = {e:.10f}  (nao {m.nao}, npts {len(grids.weights)})")


for p in sorted(CASES.glob("*ghost*.hfinp")):
    run(p)
