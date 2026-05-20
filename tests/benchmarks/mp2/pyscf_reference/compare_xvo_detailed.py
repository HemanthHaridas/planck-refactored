#!/usr/bin/env python3
"""Compare RHS construction in detail."""
from __future__ import annotations

import os
import re
import subprocess
from functools import reduce
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]


def extract_planck_matrices(executable: Path, input_path: Path) -> dict:
    """Extract debug matrices from Planck."""
    env = dict(os.environ)
    env["PLANCK_DEBUG_RMP2_MATRICES"] = "1"
    proc = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    HEADER_RE = re.compile(r"^PLANCK_RMP2_MATRIX\s+(\S+)\s+(\d+)\s+(\d+)\s*$")
    ELEM_RE = re.compile(r"^PLANCK_RMP2_MATRIX_ELEM\s+(\S+)\s+(\d+)\s+(\d+)\s+([-+0-9Ee\.]+)\s*$")

    mats = {}
    for line in proc.stdout.splitlines():
        header = HEADER_RE.match(line.strip())
        if header:
            name = header.group(1)
            rows = int(header.group(2))
            cols = int(header.group(3))
            mats[name] = np.zeros((rows, cols), dtype=float)
            continue
        elem = ELEM_RE.match(line.strip())
        if elem:
            name = elem.group(1)
            mats[name][int(elem.group(2)), int(elem.group(3))] = float(elem.group(4))

    return mats


def main():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tests" / "pyscf"))

    # Import after path is set
    from pyscf import lib, mp  # noqa: E402
    from pyscf.grad import mp2 as pyscf_grad_mp2  # noqa: E402
    from pyscf.mp import mp2 as pyscf_mp2  # noqa: E402
    from benchmark import CASE_INPUTS, build_mean_field, parse_mp2_frozen  # noqa: E402
    from input_utils import parse_input_file  # noqa: E402

    case = "water_rmp2_gradient_sto3g"
    input_path = CASE_INPUTS[case]
    executable = REPO_ROOT / "build" / "hartree-fock"

    print("=" * 70)
    print("Extracting Planck matrices...")
    print("=" * 70)
    planck_mats = extract_planck_matrices(executable, input_path)

    print("\nExtracted matrices from Planck:")
    for name in sorted(planck_mats.keys()):
        mat = planck_mats[name]
        print(f"  {name}: shape={mat.shape}, max={np.max(np.abs(mat)):.10e}")

    print("\n" + "=" * 70)
    print("Building PySCF reference...")
    print("=" * 70)

    spec = parse_input_file(input_path)
    scf_spec = spec["scf"]
    mf = build_mean_field(spec)
    frozen = parse_mp2_frozen(scf_spec.get("mp2_frozen"))
    mf.kernel()

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

    mo_coeff = post.mo_coeff
    mo_energy = post._scf.mo_energy
    mo_occ = post.mo_occ
    nocc = int(np.count_nonzero(mo_occ > 0))
    nmo = mo_coeff.shape[1]
    nvirt = nmo - nocc

    dm1_corr_mo = np.zeros((nmo, nmo))
    dm1_corr_mo[:nocc, :nocc] = doo + doo.T
    dm1_corr_mo[nocc:, nocc:] = dvv + dvv.T

    dm1_corr_ao = reduce(np.dot, (mo_coeff, dm1_corr_mo, mo_coeff.T))
    vhf = post._scf.get_veff(post.mol, dm1_corr_ao) * 2.0

    grad = post.nuc_grad_method()
    mol = grad.mol
    OA, VA, OF, VF = pyscf_grad_mp2._index_frozen_active(post.get_frozen_mask(), post.mo_occ)
    orbo = post.mo_coeff[:, OA]
    orbv = post.mo_coeff[:, VA]
    nao = orbo.shape[0]

    from pyscf.ao2mo import _ao2mo

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

    vhf_mo = reduce(np.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    imat_term = imat[:nocc, nocc:].T - imat[nocc:, :nocc]
    xvo_pyscf = vhf_mo + imat_term

    print("\nPySCF RHS (xvo) components:")
    print(f"  vhf_mo: shape={vhf_mo.shape}")
    print(f"    max={np.max(np.abs(vhf_mo)):.12e}")
    print(f"    values:\n{vhf_mo}")
    print(f"\n  imat[:nocc, nocc:]: shape={imat[:nocc, nocc:].shape}")
    print(f"    max={np.max(np.abs(imat[:nocc, nocc:])):.12e}")
    print(f"    values:\n{imat[:nocc, nocc:]}")
    print(f"\n  imat[nocc:, :nocc]: shape={imat[nocc:, :nocc].shape}")
    print(f"    max={np.max(np.abs(imat[nocc:, :nocc])):.12e}")
    print(f"    values:\n{imat[nocc:, :nocc]}")
    print(f"\n  imat_term (T - rest): shape={imat_term.shape}")
    print(f"    max={np.max(np.abs(imat_term)):.12e}")
    print(f"    values:\n{imat_term}")
    print(f"\n  xvo (final): shape={xvo_pyscf.shape}")
    print(f"    max={np.max(np.abs(xvo_pyscf)):.12e}")
    print(f"    values:\n{xvo_pyscf}")

    if "P_ao" in planck_mats:
        print("\n" + "=" * 70)
        print("Comparing P_ao matrices...")
        print("=" * 70)
        planck_p_ao = planck_mats["P_ao"]
        print(f"Planck P_ao: max={np.max(np.abs(planck_p_ao)):.10e}")
        print(f"PySCF P_ao (vhf contraction basis): max={np.max(np.abs(vhf)):.10e}")


if __name__ == "__main__":
    main()
