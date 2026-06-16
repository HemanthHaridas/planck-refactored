#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy
from pyscf import dft
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import uks as uks_grad

from dft_case_utils import PLANCK_DFT_EXE, REPO_ROOT
from input_utils import build_molecule, grid_level, parse_input_file

ATOM_LINE_RE = re.compile(
    r"Atom\s+(\d+)\s*:\s*([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
)


def prepare_planck_input(input_path: Path, calculation: str, grid_name: str, use_symm: bool) -> str:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    current: str | None = None
    replaced_calc = False
    replaced_grid = False
    replaced_symm = False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("%begin_"):
            current = stripped[len("%begin_") :]
            out.append(line)
            continue
        if stripped.startswith("%end_"):
            if current == "control" and not replaced_calc:
                out.append(f"    calculation {calculation}")
            if current == "dft" and not replaced_grid:
                out.append(f"    grid                {grid_name}")
            if current == "geom" and not replaced_symm:
                out.append(f"    use_symm    {'.true.' if use_symm else '.false.'}")
            current = None
            out.append(line)
            continue

        if current == "control" and stripped.startswith("calculation"):
            out.append(f"    calculation {calculation}")
            replaced_calc = True
            continue
        if current == "dft" and stripped.startswith("grid"):
            out.append(f"    grid                {grid_name}")
            replaced_grid = True
            continue
        if current == "geom" and stripped.startswith("use_symm"):
            out.append(f"    use_symm    {'.true.' if use_symm else '.false.'}")
            replaced_symm = True
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def run_planck_gradient(input_path: Path) -> str:
    contents = prepare_planck_input(
        input_path,
        calculation="gradient",
        grid_name="fine",
        use_symm=False,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".hfinp", delete=False) as handle:
        handle.write(contents)
        temp_path = Path(handle.name)
    env = os.environ.copy()
    env["PLANCK_DFT_GRADIENT_DEBUG"] = "1"
    env["PLANCK_DFT_ALLOW_RS_WORKFLOWS"] = "1"
    try:
        proc = subprocess.run(
            [str(PLANCK_DFT_EXE), str(temp_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            "Planck gradient run failed\n"
            f"exit code: {proc.returncode}\n"
            "---- output ----\n"
            f"{output}"
        )
    return output


def parse_gradient_block(lines: list[str], start_index: int) -> numpy.ndarray:
    rows: list[list[float]] = []
    for line in lines[start_index + 1 :]:
        match = ATOM_LINE_RE.search(line)
        if match:
            rows.append(
                [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            )
            continue
        if rows:
            break
    if not rows:
        raise RuntimeError(f"Could not parse gradient block after line: {lines[start_index]}")
    return numpy.asarray(rows)


def parse_planck_components(output: str) -> dict[str, numpy.ndarray]:
    lines = output.splitlines()
    components: dict[str, numpy.ndarray] = {}
    for idx, line in enumerate(lines):
        if "DFT Gradient Debug :" in line and "component (Ha/Bohr)" in line:
            label = line.split("DFT Gradient Debug :", 1)[1].strip().split(" component", 1)[0]
            components[label] = parse_gradient_block(lines, idx)
        elif "Nuclear Gradient (Ha/Bohr) :" in line:
            components["Reported-total"] = parse_gradient_block(lines, idx)
    return components


def contract_rks_block(vmat: numpy.ndarray, dm: numpy.ndarray, aoslices, natm: int) -> numpy.ndarray:
    result = numpy.zeros((natm, 3))
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2:]
        result[ia] += numpy.einsum("xij,ij->x", vmat[:, p0:p1], dm[p0:p1]) * 2.0
    return result


def contract_uks_block(vmat: numpy.ndarray, dm: numpy.ndarray, aoslices, natm: int) -> numpy.ndarray:
    result = numpy.zeros((natm, 3))
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2:]
        result[ia] += numpy.einsum("sxij,sij->x", vmat[:, :, p0:p1], dm[:, p0:p1]) * 2.0
    return result


def pyscf_rks_components(input_path: Path) -> dict[str, numpy.ndarray]:
    spec = parse_input_file(input_path)
    spec["dft"]["grid"] = "fine"
    spec["geom"]["use_symm"] = ".false."
    mol = build_molecule(spec)

    mf = dft.RKS(mol)
    mf.xc = "hse06"
    mf.conv_tol = float(spec["scf"].get("tol_energy", "1e-10"))
    mf.grids.level = grid_level(spec["dft"]["grid"])
    mf.kernel()

    grad_obj = mf.nuc_grad_method()
    grad_obj.grid_response = True

    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = grad_obj._tag_rdm1(dm0, mo_coeff, mo_occ)
    dme0 = grad_obj.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    hcore_deriv = grad_obj.hcore_generator(mol)
    s1 = grad_obj.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()

    ni = mf._numint
    grids, _ = rks_grad._initialize_grids(grad_obj)
    exc_grid, xcvmat = rks_grad.get_vxc_full_response(
        ni,
        mol,
        grids,
        mf.xc,
        dm0,
        max_memory=max(2000, grad_obj.max_memory * 0.9),
        verbose=0,
    )

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    vj, vk = grad_obj.get_jk(mol, dm0)
    vk_full = vk * hyb
    vk_lr = numpy.zeros_like(vk)
    if omega != 0:
        vk_lr = grad_obj.get_k(mol, dm0, omega=omega) * (alpha - hyb)
    vk = vk_full + vk_lr
    jk_vmat = vj - 0.5 * vk

    core_pulay = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2:]
        h1ao = hcore_deriv(ia)
        core_pulay[ia] += numpy.einsum("xij,ij->x", h1ao, dm0)
        core_pulay[ia] -= numpy.einsum("xij,ij->x", s1[:, p0:p1], dme0[p0:p1]) * 2.0

    j_grad = contract_rks_block(vj, dm0, aoslices, mol.natm)
    k_full_grad = contract_rks_block(-0.5 * vk_full, dm0, aoslices, mol.natm)
    k_lr_grad = contract_rks_block(-0.5 * vk_lr, dm0, aoslices, mol.natm)
    jk_grad = j_grad + k_full_grad + k_lr_grad
    xc_matrix_grad = contract_rks_block(xcvmat, dm0, aoslices, mol.natm)
    xc_grid = xc_matrix_grad + exc_grid
    nuclear = grad_obj.grad_nuc(mol)
    hf_like = core_pulay + jk_grad + nuclear
    total = hf_like + xc_grid

    return {
        "PySCF core+Pulay": core_pulay,
        "PySCF J": j_grad,
        "PySCF K-full": k_full_grad,
        "PySCF K-lr": k_lr_grad,
        "PySCF J/K": jk_grad,
        "PySCF XC-grid": xc_grid,
        "PySCF HF-like": hf_like,
        "PySCF KS-total": total,
        "PySCF nuclear": nuclear,
    }


def pyscf_uks_components(input_path: Path) -> dict[str, numpy.ndarray]:
    spec = parse_input_file(input_path)
    spec["dft"]["grid"] = "fine"
    spec["geom"]["use_symm"] = ".false."
    mol = build_molecule(spec)

    mf = dft.UKS(mol)
    mf.xc = "hse06"
    mf.conv_tol = float(spec["scf"].get("tol_energy", "1e-10"))
    mf.grids.level = grid_level(spec["dft"]["grid"])
    mf.kernel()

    grad_obj = mf.nuc_grad_method()
    grad_obj.grid_response = True

    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = grad_obj._tag_rdm1(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
    dme0 = grad_obj.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]
    hcore_deriv = grad_obj.hcore_generator(mol)
    s1 = grad_obj.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()

    ni = mf._numint
    grids, _ = rks_grad._initialize_grids(grad_obj)
    exc_grid, xcvmat = uks_grad.get_vxc_full_response(
        ni,
        mol,
        grids,
        mf.xc,
        dm0,
        max_memory=max(2000, grad_obj.max_memory * 0.9),
        verbose=0,
    )

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    vj, vk = grad_obj.get_jk(mol, dm0)
    vk_full = vk * hyb
    vk_lr = numpy.zeros_like(vk)
    if omega != 0:
        vk_lr = grad_obj.get_k(mol, dm0, omega=omega) * (alpha - hyb)
    vk = vk_full + vk_lr
    vj_total = vj[0] + vj[1]
    jk_vmat = numpy.asarray((vj_total - vk[0], vj_total - vk[1]))

    core_pulay = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2:]
        h1ao = hcore_deriv(ia)
        core_pulay[ia] += numpy.einsum("xij,ij->x", h1ao, dm0_sf)
        core_pulay[ia] -= numpy.einsum("xij,ij->x", s1[:, p0:p1], dme0_sf[p0:p1]) * 2.0

    j_vmat = numpy.asarray((vj_total, vj_total))
    j_grad = contract_uks_block(j_vmat, dm0, aoslices, mol.natm)
    k_full_grad = contract_uks_block(-vk_full, dm0, aoslices, mol.natm)
    k_lr_grad = contract_uks_block(-vk_lr, dm0, aoslices, mol.natm)
    jk_grad = j_grad + k_full_grad + k_lr_grad
    xc_matrix_grad = contract_uks_block(xcvmat, dm0, aoslices, mol.natm)
    xc_grid = xc_matrix_grad + exc_grid
    nuclear = grad_obj.grad_nuc(mol)
    hf_like = core_pulay + jk_grad + nuclear
    total = hf_like + xc_grid

    return {
        "PySCF core+Pulay": core_pulay,
        "PySCF J": j_grad,
        "PySCF K-full": k_full_grad,
        "PySCF K-lr": k_lr_grad,
        "PySCF J/K": jk_grad,
        "PySCF XC-grid": xc_grid,
        "PySCF HF-like": hf_like,
        "PySCF KS-total": total,
        "PySCF nuclear": nuclear,
    }


def format_matrix(name: str, matrix: numpy.ndarray) -> str:
    lines = [name]
    for idx, row in enumerate(matrix, start=1):
        lines.append(
            f"  Atom {idx:>2d}: {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}"
        )
    return "\n".join(lines)


def compare_components(planck: dict[str, numpy.ndarray], pyscf: dict[str, numpy.ndarray]) -> str:
    comparisons = [
        ("HF-core+Pulay", "PySCF core+Pulay"),
        ("HF-2e-Coulomb", "PySCF J"),
        ("HF-2e-Exchange-Full", "PySCF K-full"),
        ("HF-2e-Exchange-LR", "PySCF K-lr"),
        ("HF-2e", "PySCF J/K"),
        ("HF-nuclear", "PySCF nuclear"),
        ("HF-like", "PySCF HF-like"),
        ("XC-grid", "PySCF XC-grid"),
        ("KS-total", "PySCF KS-total"),
    ]
    lines: list[str] = []
    for planck_key, pyscf_key in comparisons:
        if planck_key not in planck or pyscf_key not in pyscf:
            continue
        diff = planck[planck_key] - pyscf[pyscf_key]
        max_abs = float(numpy.abs(diff).max())
        lines.append(format_matrix(f"Planck {planck_key}", planck[planck_key]))
        lines.append(format_matrix(pyscf_key, pyscf[pyscf_key]))
        lines.append(format_matrix(f"Delta {planck_key}", diff))
        lines.append(f"  max |delta| = {max_abs:.8e} Ha/Bohr")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Planck input file")
    args = parser.parse_args()

    planck_output = run_planck_gradient(args.input)
    planck_components_raw = parse_planck_components(planck_output)
    required = {"HF-like", "XC-grid", "KS-total"}
    missing = required - set(planck_components_raw)
    if missing:
        raise RuntimeError(f"Missing Planck debug components: {sorted(missing)}")

    spec = parse_input_file(args.input)
    scf_type = spec["scf"].get("scf_type", "rhf").lower()
    if scf_type in {"rhf", "rks"}:
        pyscf_components = pyscf_rks_components(args.input)
    elif scf_type in {"uhf", "uks"}:
        pyscf_components = pyscf_uks_components(args.input)
    else:
        raise ValueError(f"Unsupported SCF type for comparison: {scf_type}")

    print(compare_components(planck_components_raw, pyscf_components))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
