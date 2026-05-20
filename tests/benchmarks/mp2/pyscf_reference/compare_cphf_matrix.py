#!/usr/bin/env python3
"""Compare Planck's CPHF matrix against PySCF's."""
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
from pyscf.grad import mp2 as pyscf_grad_mp2  # noqa: E402
from pyscf.mp import mp2 as pyscf_mp2  # noqa: E402

from benchmark import (  # noqa: E402
    CASE_INPUTS,
    build_mean_field,
    parse_mp2_frozen,
)
from input_utils import parse_input_file  # noqa: E402


HEADER_RE = re.compile(r"^PLANCK_RHF_RESPONSE\s+(\S+)\s+(\d+)\s+(\d+)\s*$")
ELEM_RE = re.compile(r"^PLANCK_RHF_RESPONSE_ELEM\s+(\S+)\s+(\d+)\s+(\d+)\s+([-+0-9Ee\.]+)\s*$")


def run_planck_matrices(executable: Path, input_path: Path) -> dict[str, np.ndarray]:
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

    dims: dict[str, tuple[int, int]] = {}
    mats: dict[str, np.ndarray] = {}
    for line in proc.stdout.splitlines():
        header = HEADER_RE.match(line.strip())
        if header:
            name = header.group(1)
            rows = int(header.group(2))
            cols = int(header.group(3))
            dims[name] = (rows, cols)
            mats[name] = np.zeros((rows, cols), dtype=float)
            continue
        elem = ELEM_RE.match(line.strip())
        if elem:
            name = elem.group(1)
            mats[name][int(elem.group(2)), int(elem.group(3))] = float(elem.group(4))

    if not mats:
        raise RuntimeError("No PLANCK_RHF_RESPONSE rows were found in Planck output.")
    return mats


def build_pyscf_cphf(input_path: Path) -> dict[str, np.ndarray]:
    """Build CPHF matrix and RHS using PySCF."""
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
    with_frozen = pyscf_grad_mp2.has_frozen_orbitals(post)
    if with_frozen:
        raise RuntimeError("Frozen-orbital CPHF comparison is not implemented in this script.")

    mo_coeff = post.mo_coeff
    mo_energy = post._scf.mo_energy
    mo_occ = post.mo_occ
    nocc = int(np.count_nonzero(mo_occ > 0))
    nmo = mo_coeff.shape[1]
    nvirt = nmo - nocc
    hf_dm1 = post._scf.make_rdm1(post.mo_coeff, post.mo_occ)

    dm1_corr_mo = np.zeros((nmo, nmo))
    dm1_corr_mo[:nocc, :nocc] = doo + doo.T
    dm1_corr_mo[nocc:, nocc:] = dvv + dvv.T

    dm1_corr_ao = reduce(np.dot, (mo_coeff, dm1_corr_mo, mo_coeff.T))
    vhf = post._scf.get_veff(post.mol, dm1_corr_ao) * 2.0

    grad = post.nuc_grad_method()
    mol = grad.mol
    OA, VA, OF, VF = pyscf_grad_mp2._index_frozen_active(post.get_frozen_mask(), post.mo_occ)
    assert len(OF) == 0 and len(VF) == 0
    orbo = post.mo_coeff[:, OA]
    orbv = post.mo_coeff[:, VA]
    nao = orbo.shape[0]

    from pyscf.ao2mo import _ao2mo  # local import to mirror grad code

    part_dm2 = _ao2mo.nr_e2(
        t2.reshape(nocc**2, nvirt**2),
        np.asarray(orbv.T, order="F"),
        (0, nao, 0, nao),
        "s1",
        "s1",
    ).reshape(nocc, nocc, nao, nao)
    part_dm2 = part_dm2.transpose(0, 2, 3, 1) * 4.0 - part_dm2.transpose(0, 3, 2, 1) * 2.0

    offsetdic = mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    imat = np.zeros((nao, nao))
    max_memory = max(0, post.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))
    for atom in range(mol.natm):
        shl0, shl1, p0, _ = offsetdic[atom]
        ip1 = p0
        for b0, b1, nf in pyscf_grad_mp2._shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum("pi,iqrj->pqrj", orbo[ip0:ip1], part_dm2)
            dm2buf += lib.einsum("qi,iprj->pqrj", orbo, part_dm2[:, ip0:ip1])
            dm2buf = lib.einsum("pqrj,sj->pqrs", dm2buf, orbo)
            dm2buf = dm2buf + dm2buf.transpose(0, 1, 3, 2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1, nao, nao)).reshape(nf, nao, -1)
            dm2buf[:, :, diagidx] *= 0.5

            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri0 = mol.intor("int2e", aosym="s2kl", shls_slice=shls_slice)
            imat += lib.einsum("ipx,iqx->pq", eri0.reshape(nf, nao, -1), dm2buf)

    imat = reduce(np.dot, (mo_coeff.T, imat, post._scf.get_ovlp(), mo_coeff)) * -1.0
    xvo = reduce(np.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    xvo += imat[:nocc, nocc:].T - imat[nocc:, :nocc]

    print("=" * 70)
    print("PySCF RHS (xvo) breakdown:")
    print(f"  vhf part shape:    {vhf.shape}")
    vhf_mo = reduce(np.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    print(f"  vhf_mo shape:      {vhf_mo.shape}")
    print(f"  vhf_mo max:        {np.max(np.abs(vhf_mo)):.10e}")
    imat_term = imat[:nocc, nocc:].T - imat[nocc:, :nocc]
    print(f"  imat term shape:   {imat_term.shape}")
    print(f"  imat term max:     {np.max(np.abs(imat_term)):.10e}")
    print(f"  xvo max:           {np.max(np.abs(xvo)):.10e}")
    print("=" * 70)

    def fvind(x):
        x = x.reshape((nvirt, nocc))
        dm = reduce(np.dot, (mo_coeff[:, nocc:], x, mo_coeff[:, :nocc].T))
        v = post._scf.get_veff(post.mol, dm + dm.T)
        v = reduce(np.dot, (mo_coeff[:, nocc:].T, v, mo_coeff[:, :nocc]))
        return v * 2.0

    from pyscf.scf import cphf

    # Build the full CPHF matrix by applying fvind to a one-hot basis
    A_matrix = np.zeros((nvirt * nocc, nvirt * nocc))
    for idx in range(nvirt * nocc):
        one_hot = np.zeros(nvirt * nocc)
        one_hot[idx] = 1.0
        A_matrix[:, idx] = fvind(one_hot).ravel()

    # Add diagonal energy differences
    for a in range(nvirt):
        for i in range(nocc):
            idx = a * nocc + i
            A_matrix[idx, idx] += mo_energy[nocc + a] - mo_energy[i]

    z = cphf.solve(fvind, mo_energy, mo_occ, xvo, max_cycle=30)[0]

    return {
        "A": A_matrix,
        "xvo": xvo,
        "z": z,
    }


def summarize_diff(name: str, planck: np.ndarray, pyscf: np.ndarray) -> None:
    diff = planck - pyscf
    max_abs = float(np.max(np.abs(diff)))
    rms = float(math.sqrt(np.mean(diff * diff)))
    worst = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print(f"\n{name}")
    print(f"  shape      = {planck.shape}")
    print(f"  max_abs    = {max_abs:.9e}")
    print(f"  rms        = {rms:.9e}")
    if len(worst) == 2:
        print(
            f"  worst elem = ({worst[0]}, {worst[1]})  "
            f"planck={planck[worst]:.12e}  pyscf={pyscf[worst]:.12e}  diff={diff[worst]:.12e}"
        )
    else:
        print(f"  worst elem = {worst}  planck={planck[worst]:.12e}  pyscf={pyscf[worst]:.12e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Planck and PySCF CPHF matrices.")
    parser.add_argument("--case", default="water_rmp2_gradient_sto3g", choices=sorted(CASE_INPUTS))
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args()

    input_path = CASE_INPUTS[args.case]
    executable = args.build_dir / "hartree-fock"
    if not executable.exists():
        raise SystemExit(f"hartree-fock binary not found at {executable}")

    print("Extracting Planck CPHF matrices...")
    planck_mats = run_planck_matrices(executable, input_path)

    print("Building PySCF CPHF matrices...")
    pyscf_mats = build_pyscf_cphf(input_path)

    print("\n" + "=" * 70)
    print("CPHF Matrix Comparison: Planck vs PySCF")
    print("=" * 70)

    if "A" in planck_mats and "A" in pyscf_mats:
        summarize_diff("A (CPHF matrix)", planck_mats["A"], pyscf_mats["A"])

    if "rhs" in planck_mats and "xvo" in pyscf_mats:
        summarize_diff("rhs (CPHF RHS)", planck_mats["rhs"], pyscf_mats["xvo"])

    if "z" in planck_mats and "z" in pyscf_mats:
        summarize_diff("z (CPHF solution)", planck_mats["z"], pyscf_mats["z"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
