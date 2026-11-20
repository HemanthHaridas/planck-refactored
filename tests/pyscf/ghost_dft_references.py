"""References for the ghost-atom DFT gates (docs/DFT_COUNTERPOISE.md §2).

PySCF RKS/UKS with Planck's GBS, its CODATA 2018 Bohr and its unpruned
150 x 590 grid, for tests/inputs/regression/dft/*ghost*.hfinp.

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


def parse(path):
    """Input fields plus atoms as [(element, xyz strings, is_ghost)]."""
    text = path.read_text()
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    atoms = []
    for line in coords[2:]:
        tok = line.split()
        g = re.fullmatch(r"Gh\((\w+)\)", tok[0])
        atoms.append((g.group(1) if g else tok[0], tok[1:4], bool(g)))
    guess = re.search(r"^\s*guess\s+(\S+)", text, re.M)
    return dict(text=text, basis=field(text, "basis"), xc=XC[field(text, "exchange")],
                cart=field(text, "basis_type") != "spherical", charge=charge, mult=mult,
                guess=guess.group(1) if guess else "hcore", atoms=atoms)


def energy(atoms, charge, mult, xc, basis, cart, guess="hcore"):
    """KS energy of `atoms` [(element, xyz, is_ghost)], ghost grids at the element's Z."""
    bas = {e: parse_gaussian.load(str(ROOT / "basis-sets" / basis), e) for e, _, _ in atoms}

    def mol(ghosts, chg, spin):
        spec = "\n".join(f"{'ghost-' if g and ghosts else ''}{e} {' '.join(x)}" for e, x, g in atoms)
        return gto.M(atom=spec, unit=BOHR_ANGSTROM, cart=cart, verbose=0,
                     charge=chg, spin=spin, basis=bas)

    nelec = sum(gto.charge(e) for e, _, _ in atoms)
    grids = dft.gen_grid.Grids(mol(False, 0, nelec % 2))  # ghosts made real
    grids.atom_grid = (150, 590)
    grids.prune = None
    grids.build()

    m = mol(True, charge, mult - 1)
    mf = (dft.RKS if mult == 1 else dft.UKS)(m)
    mf.xc = xc
    mf.grids = grids
    mf.init_guess = "minao" if guess == "sad" else "1e"
    mf.conv_tol = 1e-12
    e = mf.kernel()
    assert mf.converged
    return e


if __name__ == "__main__":
    for p in sorted(CASES.glob("*ghost*.hfinp")):
        s = parse(p)
        e = energy(s["atoms"], s["charge"], s["mult"], s["xc"], s["basis"], s["cart"], s["guess"])
        print(f"{p.stem}: E = {e:.10f}")
