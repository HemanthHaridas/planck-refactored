#!/usr/bin/env python3

import json
import numpy as np
import struct
import sys
from collections import defaultdict
from pathlib import Path

from pyscf import df, gto


def _normalize_exponents(line: str) -> str:
    return line.replace("D", "E").replace("d", "E")


def _drop_redundant_primitives(shells):
    """Mirror of drop_redundant_primitives (src/basis/gaussian.cpp), which
    read_gbs_basis applies to every orbital basis since K1 (2026-10-25): a
    primitive of a contracted shell is dropped when a single-primitive shell
    of the same label has the same exponent. The span is unchanged but the
    contracted functions are not, so the element-wise comparison in
    tests/ri_3c_eri.cpp needs the same contraction."""
    def same(a, b):
        return abs(a - b) <= 1e-12 * abs(a)

    def free_shell(label, expn):
        return any(len(p) == 1 and lab == label and same(p[0][0], expn) for lab, p, _ in shells)

    out = []
    for label, prims, scale in shells:
        if len(prims) >= 2:
            kept = [p for p in prims if not free_shell(label, p[0])]
            prims = kept or prims
        out.append((label, prims, scale))
    return out


def _parse_gbs_element(path: Path, symbol: str, drop_redundant: bool = False) -> str:
    shells = []
    current = None
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("!"):
            continue
        if line == "****":
            current = None
            continue

        parts = line.split()
        if len(parts) == 2 and parts[1] == "0" and parts[0].isalpha():
            current = parts[0]
            continue

        if current != symbol:
            continue

        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed shell header in {path}: {line}")
        label = parts[0].upper()
        nprim = int(parts[1])
        scale = float(parts[2]) if len(parts) > 2 else 1.0

        prims = []
        for _ in range(nprim):
            raw = _normalize_exponents(lines[i].strip()).split()
            i += 1
            prims.append((float(raw[0]), [float(v) for v in raw[1:]]))
        shells.append((label, prims, scale))

    if drop_redundant:
        shells = _drop_redundant_primitives(shells)

    blocks: list[str] = []
    for label, prims, scale in shells:
        if label == "SP":
            s_block = [f"{symbol} S"]
            p_block = [f"{symbol} P"]
            for expn, (cs, cp, *_) in prims:
                s_block.append(f"  {expn:.16g} {cs * scale:.16g}")
                p_block.append(f"  {expn:.16g} {cp * scale:.16g}")
            blocks.append("\n".join(s_block))
            blocks.append("\n".join(p_block))
        else:
            block = [f"{symbol} {label}"]
            for expn, (coeff, *_) in prims:
                block.append(f"  {expn:.16g} {coeff * scale:.16g}")
            blocks.append("\n".join(block))

    if not blocks:
        raise ValueError(f"Basis element {symbol} not found in {path}")
    return "\n".join(blocks) + "\n"


def _load_basis_map(path: Path, symbols: list[str], drop_redundant: bool = False) -> dict[str, object]:
    basis = {}
    for sym in symbols:
        basis[sym] = gto.basis.parse(_parse_gbs_element(path, sym, drop_redundant), symb=sym, optimize=False)
    return basis


def _cart_key_list(mol) -> list[str]:
    out: list[str] = []
    prev_shell = None
    shell_counts: dict[int, int] = defaultdict(int)
    shell_occ = -1
    for atom_idx, sym, nl, comp in mol.cart_labels(fmt=False):
        shell_id = (atom_idx, nl)
        if shell_id != prev_shell:
            shell_occ = shell_counts[atom_idx]
            shell_counts[atom_idx] += 1
            prev_shell = shell_id
        out.append(f"{atom_idx}:{shell_occ}:{comp}")
    return out


def _pair_index(i: int, j: int) -> int:
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: export_ri_3c_pyscf.py INPUT.json OUTPUT.bin", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    cfg = json.loads(input_path.read_text())

    atoms = [(atom["symbol"], tuple(atom["coords_bohr"])) for atom in cfg["atoms"]]
    symbols = sorted({sym for sym, _ in atoms})
    # Orbital basis only: read_ri_basis does not drop primitives.
    basis = _load_basis_map(Path(cfg["basis_file"]), symbols, drop_redundant=True)
    auxbasis = _load_basis_map(Path(cfg["aux_basis_file"]), symbols)

    mol = gto.Mole()
    mol.verbose = 0
    mol.unit = "Bohr"
    mol.atom = atoms
    mol.basis = basis
    mol.cart = True
    mol.build()

    auxmol = df.addons.make_auxmol(mol, auxbasis)
    j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e_cart", aosym="s1")
    ao_diag = mol.intor("int1e_ovlp_cart").diagonal()
    aux_diag = auxmol.intor("int1e_ovlp_cart").diagonal()

    planck_ao_keys = cfg.get("planck_ao_keys")
    planck_aux_keys = cfg.get("planck_aux_keys")
    if planck_ao_keys is not None and planck_aux_keys is not None:
        pyscf_ao_keys = _cart_key_list(mol)
        pyscf_aux_keys = _cart_key_list(auxmol)
        ao_lookup = {key: idx for idx, key in enumerate(pyscf_ao_keys)}
        aux_lookup = {key: idx for idx, key in enumerate(pyscf_aux_keys)}

        ao_perm = [ao_lookup[key] for key in planck_ao_keys]
        aux_perm = [aux_lookup[key] for key in planck_aux_keys]

        nao = len(planck_ao_keys)
        j3c_planck = j3c[:, :, aux_perm].copy()
        j3c_planck = j3c_planck[ao_perm, :, :]
        j3c_planck = j3c_planck[:, ao_perm, :]
        rows = []
        for mu in range(nao):
            for nu in range(mu + 1):
                rows.append(j3c_planck[mu, nu, :])
        j3c = np.ascontiguousarray(rows, dtype="float64")
        ao_diag = ao_diag[ao_perm]
        aux_diag = aux_diag[aux_perm]

    j3c = j3c.astype("float64", copy=False, order="C")
    ao_diag = ao_diag.astype("float64", copy=False)
    aux_diag = aux_diag.astype("float64", copy=False)

    with output_path.open("wb") as fh:
        fh.write(struct.pack("<QQQQ", j3c.shape[0], j3c.shape[1], ao_diag.shape[0], aux_diag.shape[0]))
        # Eigen::MatrixXd reads raw payloads in column-major order, so write
        # the 2D matrix in Fortran order to preserve (row, col) indexing.
        fh.write(j3c.tobytes(order="F"))
        fh.write(ao_diag.tobytes(order="C"))
        fh.write(aux_diag.tobytes(order="C"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
