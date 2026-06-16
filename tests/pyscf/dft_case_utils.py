#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pyscf import dft, mp

from input_utils import build_molecule, grid_level, parse_input_file

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANCK_DFT_EXE = REPO_ROOT / "build/planck-dft"
DFT_TOTAL_PATTERN = re.compile(r"^\[INF\]\s+DFT Energy\s*:\s+([-+0-9Ee\.]+)\s+Eh", re.MULTILINE)


def parse_last_float(pattern: re.Pattern[str], text: str, label: str) -> float:
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from output")
    return float(matches[-1])


def manual_reference_spec(exchange_name: str) -> tuple[str, float]:
    exchange = exchange_name.strip().lower()
    if exchange == "b3lyp":
        return "b3lyp", 0.0
    if exchange == "hse06":
        return "hse06", 0.0
    if exchange == "b2plyp":
        return "0.53*HF + 0.47*B88, 0.73*LYP", 0.27
    raise ValueError(
        f"PySCF reference mapping is not implemented for exchange functional {exchange_name!r}"
    )


def run_planck_dft(input_path: Path) -> tuple[float, float]:
    if not PLANCK_DFT_EXE.exists():
        raise RuntimeError(f"Planck DFT executable not found: {PLANCK_DFT_EXE}")

    start = time.perf_counter()
    proc = subprocess.run(
        [str(PLANCK_DFT_EXE), str(input_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            "Planck DFT run failed\n"
            f"exit code: {proc.returncode}\n"
            "---- output ----\n"
            f"{output}"
        )

    return parse_last_float(DFT_TOTAL_PATTERN, output, "DFT total energy"), elapsed


def with_grid_override(input_path: Path, grid_name: str) -> str:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_dft = False
    replaced = False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("%begin_dft"):
            in_dft = True
            out.append(line)
            continue
        if stripped.startswith("%end_dft"):
            if in_dft and not replaced:
                out.append(f"    grid                {grid_name}")
            in_dft = False
            out.append(line)
            continue
        if in_dft and stripped.startswith("grid"):
            out.append(f"    grid                {grid_name}")
            replaced = True
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def run_planck_dft_with_grid(input_path: Path, grid_name: str) -> tuple[float, float]:
    contents = with_grid_override(input_path, grid_name)
    with tempfile.NamedTemporaryFile("w", suffix=".hfinp", delete=False) as handle:
        handle.write(contents)
        temp_path = Path(handle.name)
    try:
        return run_planck_dft(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def run_pyscf_reference(input_path: Path, grid_name: str | None = None) -> dict[str, Any]:
    spec = parse_input_file(input_path)
    if grid_name is not None:
        spec["dft"]["grid"] = grid_name
    mol = build_molecule(spec)
    dft_section = spec["dft"]
    scf_section = spec["scf"]
    xc_string, pt2_scale = manual_reference_spec(dft_section["exchange"])

    if scf_section.get("scf_type", "rhf").lower() not in {"rhf", "rks"}:
        raise ValueError(f"Only closed-shell RKS/RHF DFT references are supported for {input_path}")

    mf = dft.RKS(mol)
    mf.conv_tol = float(scf_section.get("tol_energy", "1e-10"))
    mf.grids.level = grid_level(dft_section.get("grid", dft_section.get("grid_level", "normal")))
    mf.xc = xc_string

    start = time.perf_counter()
    scf_energy = float(mf.kernel())
    pt2_energy = 0.0
    if abs(pt2_scale) > 0.0:
        pt2_energy = float(mp.MP2(mf).kernel()[0])
    total_energy = scf_energy + pt2_scale * pt2_energy
    elapsed = time.perf_counter() - start

    return {
        "scf_energy": scf_energy,
        "pt2_energy": pt2_energy,
        "pt2_scale": pt2_scale,
        "total_energy": total_energy,
        "elapsed_s": elapsed,
        "xc_string": xc_string,
    }
