#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from pyscf import cc, dft, gto, mp, scf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "pyscf" / "regression_reference_cases.json"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "pyscf" / "regression_references.json"

LABEL_PATTERNS = {
    "CASSCF_ENERGY": re.compile(r"^CASSCF_ENERGY:\s*([-+0-9Ee.]+)", re.MULTILINE),
}


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            current = lower[len("%begin_"):]
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


def functional_string(dft_section: dict[str, str]) -> str:
    exchange = dft_section.get("exchange", "").strip().lower()
    correlation = dft_section.get("correlation", "").strip().lower()
    if exchange in {"b3lyp", "pbe0", "pbeh"}:
        return {"b3lyp": "b3lyp", "pbe0": "pbe0", "pbeh": "pbe0"}[exchange]
    if exchange == "lda":
        exchange = "slater"
    if correlation == "vwn":
        correlation = "vwn5"
    return f"{exchange},{correlation}"


def grid_level(name: str) -> int:
    return {
        "coarse": 0,
        "normal": 1,
        "fine": 3,
        "ultrafine": 5,
    }.get(name.lower(), 1)


def run_input_case(case: dict[str, Any]) -> dict[str, float]:
    spec = parse_input_file(REPO_ROOT / case["input"])
    mol = build_molecule(spec)
    scf_section = spec["scf"]
    dft_section = spec["dft"]
    scf_type = scf_section.get("scf_type", "rhf").lower()
    correlation = scf_section.get("correlation", "").lower()
    tol_energy = float(scf_section.get("tol_energy", "1e-10"))

    uses_dft = bool(dft_section)
    if uses_dft:
        if scf_type in {"rhf", "rks"}:
            mf: Any = dft.RKS(mol)
        elif scf_type in {"uhf", "uks"}:
            mf = dft.UKS(mol)
        else:
            raise ValueError(f"Unsupported DFT scf_type {scf_type!r} for {case['id']}")
        mf.xc = functional_string(dft_section)
        mf.grids.level = grid_level(dft_section.get("grid", dft_section.get("grid_level", "normal")))
    else:
        if scf_type == "rhf":
            mf = scf.RHF(mol)
        elif scf_type == "uhf":
            mf = scf.UHF(mol)
        else:
            raise ValueError(f"Unsupported SCF type {scf_type!r} for {case['id']}")

    mf.conv_tol = tol_energy
    mf.kernel()

    metrics: dict[str, float] = {}
    if uses_dft:
        metrics["dft_total_energy"] = float(mf.e_tot)
        return metrics

    metrics["rhf_total_energy"] = float(mf.e_tot)
    if not correlation:
        return metrics

    if correlation == "rmp2":
        corr, *_ = mp.MP2(mf).kernel()
        metrics["mp2_total_energy"] = float(mf.e_tot + corr)
    elif correlation == "ump2":
        corr, *_ = mp.UMP2(mf).kernel()
        metrics["mp2_total_energy"] = float(mf.e_tot + corr)
    elif correlation == "ccsdt":
        corr, *_ = cc.RCCSDT(mf).kernel()
        metrics["rccsdt_total_energy"] = float(mf.e_tot + corr)
    elif correlation == "uccsd":
        corr, *_ = cc.UCCSD(mf).kernel()
        metrics["uccsd_total_energy"] = float(mf.e_tot + corr)
    elif correlation == "uccsdt":
        corr, *_ = cc.UCCSDT(mf).kernel()
        metrics["uccsdt_total_energy"] = float(mf.e_tot + corr)
    else:
        raise ValueError(f"Unsupported correlation method {correlation!r} for {case['id']}")
    return metrics


def run_script_case(case: dict[str, Any], python_executable: Path) -> dict[str, float]:
    proc = subprocess.run(
        [str(python_executable), str(REPO_ROOT / case["script"])],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"PySCF script failed for {case['id']}:\n{output}")

    metrics: dict[str, float] = {}
    for metric_name, label in case["parse_labels"].items():
        pattern = LABEL_PATTERNS[label]
        match = pattern.search(output)
        if match is None:
            raise RuntimeError(f"Could not parse {label} from {case['script']}")
        metrics[metric_name] = float(match.group(1))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PySCF references for Planck regressions")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    selected = set(args.case)
    python_executable = Path(args.python).expanduser()

    references: dict[str, dict[str, dict[str, float]]] = {}
    for case in manifest["cases"]:
        if selected and case["id"] not in selected:
            continue
        print(f"[PySCF] {case['id']}", file=sys.stderr)
        if case["mode"] == "input":
            metrics = run_input_case(case)
        elif case["mode"] == "script":
            metrics = run_script_case(case, python_executable)
        else:
            raise ValueError(f"Unknown mode {case['mode']!r}")

        payload: dict[str, dict[str, float]] = {}
        for metric_name, expected in metrics.items():
            entry = {"expected": float(expected)}
            if metric_name in case.get("tolerances", {}):
                entry["atol"] = float(case["tolerances"][metric_name])
            payload[metric_name] = entry
        references[case["id"]] = payload

    output_path = Path(args.output)
    output_path.write_text(json.dumps({"cases": references}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
