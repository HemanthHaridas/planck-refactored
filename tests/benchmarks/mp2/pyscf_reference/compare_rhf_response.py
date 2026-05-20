#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from functools import reduce
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PYSCF_DIR = REPO_ROOT / "tests" / "pyscf"
LOCAL_PYSCF_PYTHON = PYSCF_DIR / ".venv" / "bin" / "python"

if str(PYSCF_DIR) not in sys.path:
    sys.path.insert(0, str(PYSCF_DIR))


def ensure_pyscf_runtime() -> None:
    try:
        import pyscf  # noqa: F401
    except ModuleNotFoundError:
        current = Path(sys.executable).resolve()
        if LOCAL_PYSCF_PYTHON.exists() and current != LOCAL_PYSCF_PYTHON.resolve():
            proc = subprocess.run(
                [str(LOCAL_PYSCF_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
                check=False,
            )
            raise SystemExit(proc.returncode)
        raise


ensure_pyscf_runtime()

from pyscf import lib, mp  # noqa: E402
from pyscf.ao2mo import _ao2mo  # noqa: E402
from pyscf.grad import mp2 as pyscf_grad_mp2  # noqa: E402
from pyscf.mp import mp2 as pyscf_mp2  # noqa: E402
from pyscf.scf import cphf  # noqa: E402

from benchmark import CASE_INPUTS, build_mean_field, parse_mp2_frozen  # noqa: E402
from input_utils import parse_input_file  # noqa: E402


HEADER_RE = re.compile(r"^PLANCK_RHF_RESPONSE\s+(\S+)\s+(\d+)\s+(\d+)\s*$")
ELEM_RE = re.compile(r"^PLANCK_RHF_RESPONSE_ELEM\s+(\S+)\s+(\d+)\s+(\d+)\s+([-+0-9Ee\.]+)\s*$")


def run_planck_response(executable: Path, input_path: Path) -> dict[str, np.ndarray]:
    env = dict(os.environ)
    env["PLANCK_DEBUG_RHF_RESPONSE"] = "1"
    proc = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Planck run failed for {input_path} with exit {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    mats: dict[str, np.ndarray] = {}
    for line in proc.stdout.splitlines():
        header = HEADER_RE.match(line.strip())
        if header:
            mats[header.group(1)] = np.zeros((int(header.group(2)), int(header.group(3))), dtype=float)
            continue
        elem = ELEM_RE.match(line.strip())
        if elem:
            mats[elem.group(1)][int(elem.group(2)), int(elem.group(3))] = float(elem.group(4))

    if not mats:
        raise RuntimeError("No PLANCK_RHF_RESPONSE rows were found in Planck output.")
    return mats


def build_pyscf_response(input_path: Path) -> dict[str, np.ndarray]:
    spec = parse_input_file(input_path)
    scf_spec = spec["scf"]
    mf = build_mean_field(spec)
    frozen = parse_mp2_frozen(scf_spec.get("mp2_frozen"))
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for {input_path}")

    post = mp.MP2(mf, frozen=frozen)
    post.verbose = 0
    post.level_shift = float(scf_spec.get("mp2_level_shift", "0.0"))
    post.conv_tol = float(scf_spec.get("mp2_conv_tol", "1e-7"))
    post.conv_tol_normt = float(scf_spec.get("mp2_conv_tol_normt", "1e-5"))
    post.max_cycle = int(scf_spec.get("mp2_max_cycle", "50"))
    post.diis_space = int(scf_spec.get("mp2_diis_space", "6"))
    post.with_t2 = True
    post.kernel()

    t2 = post.t2
    doo, dvv = pyscf_mp2._gamma1_intermediates(post, t2)
    OA, VA, OF, VF = pyscf_grad_mp2._index_frozen_active(post.get_frozen_mask(), post.mo_occ)
    if len(OF) or len(VF):
        raise RuntimeError("Frozen-orbital comparison is not implemented in this script.")
    mo_coeff = post.mo_coeff
    mo_energy = post._scf.mo_energy
    mo_occ = post.mo_occ
    nocc = int(np.count_nonzero(mo_occ > 0))
    nmo = mo_coeff.shape[1]
    nvirt = nmo - nocc
    nao = mo_coeff.shape[0]

    dm1mo = np.zeros((nmo, nmo))
    dm1mo[:nocc, :nocc] = doo + doo.T
    dm1mo[nocc:, nocc:] = dvv + dvv.T
    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = post._scf.get_veff(post.mol, dm1) * 2.0

    orbo = post.mo_coeff[:, OA]
    orbv = post.mo_coeff[:, VA]
    part_dm2 = _ao2mo.nr_e2(
        t2.reshape(nocc**2, nvirt**2),
        np.asarray(orbv.T, order="F"),
        (0, nao, 0, nao),
        "s1",
        "s1",
    ).reshape(nocc, nocc, nao, nao)
    part_dm2 = part_dm2.transpose(0, 2, 3, 1) * 4.0 - part_dm2.transpose(0, 3, 2, 1) * 2.0

    offsetdic = post.mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    imat = np.zeros((nao, nao))
    max_memory = max(0, post.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))
    for atom in range(post.mol.natm):
        shl0, shl1, p0, _ = offsetdic[atom]
        ip1 = p0
        for b0, b1, nf in pyscf_grad_mp2._shell_prange(post.mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum("pi,iqrj->pqrj", orbo[ip0:ip1], part_dm2)
            dm2buf += lib.einsum("qi,iprj->pqrj", orbo, part_dm2[:, ip0:ip1])
            dm2buf = lib.einsum("pqrj,sj->pqrs", dm2buf, orbo)
            dm2buf = dm2buf + dm2buf.transpose(0, 1, 3, 2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1, nao, nao)).reshape(nf, nao, -1)
            dm2buf[:, :, diagidx] *= 0.5
            shls_slice = (b0, b1, 0, post.mol.nbas, 0, post.mol.nbas, 0, post.mol.nbas)
            eri0 = post.mol.intor("int2e", aosym="s2kl", shls_slice=shls_slice)
            imat += lib.einsum("ipx,iqx->pq", eri0.reshape(nf, nao, -1), dm2buf)

    imat = reduce(np.dot, (mo_coeff.T, imat, post._scf.get_ovlp(), mo_coeff)) * -1.0
    rhs = reduce(np.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    rhs += imat[:nocc, nocc:].T - imat[nocc:, :nocc]
    rhs_vec = (-rhs).reshape(-1, order="C")

    def fvind(x):
        x = x.reshape((nvirt, nocc))
        dm = reduce(np.dot, (mo_coeff[:, nocc:], x, mo_coeff[:, :nocc].T))
        v = post._scf.get_veff(post.mol, dm + dm.T)
        v = reduce(np.dot, (mo_coeff[:, nocc:].T, v, mo_coeff[:, :nocc]))
        return v * 2.0

    a = np.zeros((nvirt * nocc, nvirt * nocc))
    for col in range(nvirt * nocc):
        x = np.zeros((nvirt, nocc))
        x.reshape(-1, order="C")[col] = 1.0
        a[:, col] = (np.diag((mo_energy[nocc:, None] - mo_energy[:nocc]).reshape(-1, order="C")))[:, col]
        a[:, col] -= fvind(x).reshape(-1, order="C")

    z = cphf.solve(fvind, mo_energy, mo_occ, rhs, max_cycle=30)[0]
    return {
        "A": a,
        "rhs": rhs,
        "rhs_vec": rhs_vec.reshape(-1, 1),
        "z": z,
    }


def summarize(name: str, planck: np.ndarray, pyscf: np.ndarray) -> None:
    diff = planck - pyscf
    max_abs = float(np.max(np.abs(diff)))
    rms = float(math.sqrt(np.mean(diff * diff)))
    worst = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print(f"{name}")
    print(f"  shape      = {planck.shape[0]}x{planck.shape[1]}")
    print(f"  max_abs    = {max_abs:.9e}")
    print(f"  rms        = {rms:.9e}")
    print(
        f"  worst elem = ({worst[0]}, {worst[1]})  "
        f"planck={planck[worst]:.12e}  pyscf={pyscf[worst]:.12e}  diff={diff[worst]:.12e}"
    )


def estimate_sign_vector(planck_rhs: np.ndarray, pyscf_rhs: np.ndarray) -> np.ndarray:
    flat_p = planck_rhs.reshape(-1, order="C")
    flat_r = pyscf_rhs.reshape(-1, order="C")
    sign = np.ones_like(flat_p)
    for idx, (lhs, rhs) in enumerate(zip(flat_p, flat_r)):
        if abs(lhs) > 1e-10 and abs(rhs) > 1e-10:
            sign[idx] = 1.0 if lhs * rhs >= 0.0 else -1.0
    return sign


def apply_sign_alignment_to_matrix(a: np.ndarray, sign: np.ndarray) -> np.ndarray:
    return sign[:, None] * a * sign[None, :]


def apply_sign_alignment_to_ov(mat: np.ndarray, sign: np.ndarray) -> np.ndarray:
    return (mat.reshape(-1, order="C") * sign).reshape(mat.shape, order="C")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Planck and PySCF restricted RHF response objects.")
    parser.add_argument("--case", default="water_rmp2_gradient_sto3g", choices=sorted(CASE_INPUTS))
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args()

    input_path = CASE_INPUTS[args.case]
    executable = args.build_dir / "hartree-fock"
    if not executable.exists():
        raise SystemExit(f"hartree-fock binary not found at {executable}")

    planck = run_planck_response(executable, input_path)
    pyscf = build_pyscf_response(input_path)

    print("Restricted RHF response objects: Planck vs PySCF")
    for name in ["A", "rhs", "rhs_vec", "z"]:
        summarize(name, planck[name], pyscf[name])

    sign = estimate_sign_vector(planck["rhs"], pyscf["rhs"])
    print("Gauge-aligned using sign vector from rhs overlap")
    summarize("A_aligned", apply_sign_alignment_to_matrix(planck["A"], sign), pyscf["A"])
    summarize("rhs_aligned", apply_sign_alignment_to_ov(planck["rhs"], sign), pyscf["rhs"])
    summarize("z_aligned", apply_sign_alignment_to_ov(planck["z"], sign), pyscf["z"])

    # Additional orientation checks for debugging flattening conventions
    alt_z = np.linalg.solve(pyscf["A"], planck["rhs_vec"].reshape(-1)).reshape(pyscf["z"].shape, order="C")
    summarize("z_from_pyscf_A_planck_rhs", alt_z, pyscf["z"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
