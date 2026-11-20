#!/usr/bin/env python3
"""Orientation dependence of a PySCF KS energy on an unpruned Becke grid.

External check for the partition floor in docs/DFT_DEFAULT_GRID_SCOPE.md:
same water chains and rotations as tests/dft_grid_orientation_audit.py,
Treutler-Ahlrichs radial, Becke partition with Treutler radii adjustment
(PySCF defaults), no pruning. Prints max |E(R) - E(I)| per system.
"""
import sys
from pathlib import Path

import numpy as np
from pyscf import dft, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dft_grid_orientation_audit import ROTATIONS, water_chain  # noqa: E402


def energy(coords, rot, radial, angular):
    atoms = []
    for line in coords.splitlines():
        el, *xyz = line.split()
        p = np.array([float(t) for t in xyz])
        if rot is not None:
            p = np.array(rot) @ p
        atoms.append((el, p))
    mol = gto.M(atom=atoms, basis='6-31g*', cart=True, unit='Angstrom', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.grids.atom_grid = (radial, angular)
    mf.grids.prune = None
    mf.conv_tol = 1e-11
    return mf.kernel()


def main():
    systems = {'water1': water_chain(1, 3.0), 'water4': water_chain(4, 3.0)}
    for radial, angular in ((135, 590), (135, 974)):
        for name, coords in systems.items():
            e0 = energy(coords, None, radial, angular)
            de = max(abs(energy(coords, r, radial, angular) - e0) for r in ROTATIONS)
            print(f"{name:8s} {radial}x{angular} unpruned  max|dE| {de:.3e}", flush=True)


if __name__ == '__main__':
    main()
