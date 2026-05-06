from __future__ import annotations

from pathlib import Path
from typing import Any

from pyscf import gto


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {".true.", "true", "yes", "1"}


def parse_input_file(path: Path) -> dict[str, Any]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("%begin_"):
            current = lower[len("%begin_") :]
            sections[current] = []
            continue
        if lower.startswith("%end_"):
            current = None
            continue
        if current is not None:
            sections[current].append(stripped)

    parsed: dict[str, Any] = {}
    for name in ("control", "scf", "geom", "dft"):
        values: dict[str, str] = {}
        for entry in sections.get(name, []):
            parts = entry.split()
            if parts:
                values[parts[0].lower()] = " ".join(parts[1:])
        parsed[name] = values

    coords_lines = sections.get("coords", [])
    if len(coords_lines) < 2:
        raise ValueError(f"{path} is missing a valid coords section")
    natoms = int(coords_lines[0].split()[0])
    charge, multiplicity = (int(x) for x in coords_lines[1].split()[:2])
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for entry in coords_lines[2 : 2 + natoms]:
        symbol, x, y, z = entry.split()[:4]
        atoms.append((symbol, (float(x), float(y), float(z))))
    parsed["coords"] = {
        "natoms": natoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "atoms": atoms,
    }
    return parsed


def build_molecule(spec: dict[str, Any]) -> gto.Mole:
    control = spec["control"]
    geom = spec["geom"]
    coords = spec["coords"]
    basis_name = control["basis"].lower()
    basis_aliases = {
        "cc-pvdz-unc": "cc-pvdz",
    }

    mol = gto.Mole()
    mol.atom = "\n".join(
        f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        for symbol, xyz in coords["atoms"]
    )
    mol.basis = basis_aliases.get(basis_name, control["basis"])
    mol.charge = coords["charge"]
    mol.spin = coords["multiplicity"] - 1
    mol.cart = control.get("basis_type", "cartesian").lower() == "cartesian"
    mol.symmetry = parse_bool(geom.get("use_symm", ".false."))
    mol.unit = "Bohr" if geom.get("coord_units", "angstrom").lower() == "bohr" else "Angstrom"
    mol.verbose = 0
    mol.build()
    return mol


def grid_level(name: str) -> int:
    return {
        "coarse": 0,
        "normal": 1,
        "fine": 3,
        "ultrafine": 5,
    }.get(name.lower(), 1)
