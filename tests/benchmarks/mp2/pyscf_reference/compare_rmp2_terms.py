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
from typing import Any

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

from pyscf import lib, mp, scf  # noqa: E402
from pyscf.ao2mo import _ao2mo  # noqa: E402
from pyscf.grad import mp2 as pyscf_grad_mp2  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402
from pyscf.mp import mp2 as pyscf_mp2  # noqa: E402

from benchmark import (  # noqa: E402
    CASE_INPUTS,
    build_mean_field,
    build_rotation,
    extract_input_coords,
    parse_mp2_frozen,
    parse_planck_standard_coords,
)
from input_utils import parse_input_file  # noqa: E402


TERM_ROW_RE = re.compile(
    r"^PLANCK_RMP2_TERM_ROW\s+(\S+)\s+(\d+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s*$"
)


def rotate_rows(rows: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.asarray([rotation @ row for row in rows], dtype=float)


def run_planck_terms(executable: Path, input_path: Path) -> tuple[dict[str, np.ndarray], list[list[float]] | None]:
    env = dict(os.environ)
    env["PLANCK_DEBUG_RMP2_TERMS"] = "1"
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

    rows: dict[str, dict[int, list[float]]] = {}
    for line in proc.stdout.splitlines():
        match = TERM_ROW_RE.match(line.strip())
        if not match:
            continue
        name = match.group(1)
        atom = int(match.group(2))
        rows.setdefault(name, {})[atom] = [
            float(match.group(3)),
            float(match.group(4)),
            float(match.group(5)),
        ]

    if not rows:
        raise RuntimeError("No PLANCK_RMP2_TERM rows were found in Planck output.")

    terms = {
        name: np.asarray([atoms[idx] for idx in range(1, len(atoms) + 1)], dtype=float)
        for name, atoms in rows.items()
    }
    standard_coords = parse_planck_standard_coords(proc.stdout, next(iter(terms.values())).shape[0])
    return terms, standard_coords


def _shell_prange(mol, start: int, stop: int, blksize: int):
    ang = mol._bas[start:stop, pyscf_grad_mp2.gto.ANG_OF]
    if mol.cart:
        dims = (ang + 1) * (ang + 2) // 2 * mol._bas[start:stop, pyscf_grad_mp2.gto.NCTR_OF]
    else:
        dims = (ang * 2 + 1) * mol._bas[start:stop, pyscf_grad_mp2.gto.NCTR_OF]
    nao = 0
    ib0 = start
    for ib, now in zip(range(start, stop), dims):
        nao += now
        if nao > blksize:
            yield ib0, ib, nao - now
            ib0 = ib
            nao = now
    yield ib0, stop, nao


def compute_pyscf_rmp2_terms(input_path: Path) -> dict[str, np.ndarray]:
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

    grad = post.nuc_grad_method()
    mol = grad.mol
    t2 = post.t2
    doo, dvv = pyscf_mp2._gamma1_intermediates(post, t2)

    with_frozen = pyscf_grad_mp2.has_frozen_orbitals(post)
    OA, VA, OF, VF = pyscf_grad_mp2._index_frozen_active(post.get_frozen_mask(), post.mo_occ)
    orbo = post.mo_coeff[:, OA]
    orbv = post.mo_coeff[:, VA]
    nao, nocc_active = orbo.shape
    nvir_active = orbv.shape[1]

    part_dm2 = _ao2mo.nr_e2(
        t2.reshape(nocc_active**2, nvir_active**2),
        np.asarray(orbv.T, order="F"),
        (0, nao, 0, nao),
        "s1",
        "s1",
    ).reshape(nocc_active, nocc_active, nao, nao)
    part_dm2 = part_dm2.transpose(0, 2, 3, 1) * 4.0 - part_dm2.transpose(0, 3, 2, 1) * 2.0

    hf_dm1 = post._scf.make_rdm1(post.mo_coeff, post.mo_occ)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    natm = mol.natm

    two_e = np.zeros((natm, 3))
    vhf1_blocks = np.zeros((natm, 3, nao, nao))
    imat = np.zeros((nao, nao))

    max_memory = max(0, post.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))

    for atom in range(natm):
        shl0, shl1, p0, p1 = offsetdic[atom]
        ip1 = p0
        vhf = np.zeros((3, nao, nao))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
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

            eri1 = mol.intor("int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice).reshape(3, nf, nao, -1)
            two_e[atom] -= np.einsum("xijk,ijk->x", eri1, dm2buf) * 2.0

            for comp in range(3):
                eri1_full = lib.unpack_tril(eri1[comp].reshape(nf * nao, -1)).reshape(nf, nao, nao, nao)
                vhf[comp] += np.einsum("ijkl,ij->kl", eri1_full, hf_dm1[ip0:ip1])
                vhf[comp] -= np.einsum("ijkl,il->kj", eri1_full, hf_dm1[ip0:ip1]) * 0.5
                vhf[comp, ip0:ip1] += np.einsum("ijkl,kl->ij", eri1_full, hf_dm1)
                vhf[comp, ip0:ip1] -= np.einsum("ijkl,jk->il", eri1_full, hf_dm1) * 0.5
        vhf1_blocks[atom] = vhf

    mo_coeff = post.mo_coeff
    mo_energy = post._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = np.count_nonzero(post.mo_occ > 0)
    imat = reduce(np.dot, (mo_coeff.T, imat, post._scf.get_ovlp(), mo_coeff)) * -1.0

    dm1mo = np.zeros((nmo, nmo))
    if with_frozen:
        dco = imat[OF[:, None], OA] / (mo_energy[OF, None] - mo_energy[OA])
        dfv = imat[VF[:, None], VA] / (mo_energy[VF, None] - mo_energy[VA])
        dm1mo[OA[:, None], OA] = doo + doo.T
        dm1mo[OF[:, None], OA] = dco
        dm1mo[OA[:, None], OF] = dco.T
        dm1mo[VA[:, None], VA] = dvv + dvv.T
        dm1mo[VF[:, None], VA] = dfv
        dm1mo[VA[:, None], VF] = dfv.T
    else:
        dm1mo[:nocc, :nocc] = doo + doo.T
        dm1mo[nocc:, nocc:] = dvv + dvv.T

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = post._scf.get_veff(mol, dm1) * 2.0
    xvo = reduce(np.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    xvo += imat[:nocc, nocc:].T - imat[nocc:, :nocc]

    dm1mo += pyscf_grad_mp2._response_dm1(post, xvo)
    imat[nocc:, :nocc] = imat[:nocc, nocc:].T
    im1 = reduce(np.dot, (mo_coeff, imat, mo_coeff.T))

    mf_grad = post._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    zeta = lib.direct_sum("i+j->ij", mo_energy, mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[:nocc].reshape(-1, 1)
    zeta = reduce(np.dot, (mo_coeff, zeta * dm1mo, mo_coeff.T))

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = np.dot(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
    vhf_s1occ = reduce(np.dot, (p1, post._scf.get_veff(mol, dm1 + dm1.T), p1))

    dm1p = hf_dm1 + dm1 * 2.0
    dm1 = dm1 + hf_dm1
    zeta = zeta + rhf_grad.make_rdm1e(mo_energy, mo_coeff, post.mo_occ)

    terms: dict[str, np.ndarray] = {
        "two_e": two_e,
        "h1": np.zeros((natm, 3)),
        "h1_kinetic": np.zeros((natm, 3)),
        "h1_nuc_a": np.zeros((natm, 3)),
        "h1_nuc_c": np.zeros((natm, 3)),
        "s_im1": np.zeros((natm, 3)),
        "s_zeta": np.zeros((natm, 3)),
        "s_vhf": np.zeros((natm, 3)),
        "vhf1": np.zeros((natm, 3)),
        "vhf1_rs": np.zeros((natm, 3)),
        "vhf1_rq": np.zeros((natm, 3)),
        "vhf1_pq": np.zeros((natm, 3)),
        "vhf1_ps": np.zeros((natm, 3)),
    }

    h_kin = -mol.intor("int1e_ipkin", comp=3)
    h_nuc = -mol.intor("int1e_ipnuc", comp=3)
    for atom in range(natm):
        _, _, p0, p1 = offsetdic[atom]
        with mol.with_rinv_at_nucleus(atom):
            vrinv = mol.intor("int1e_iprinv", comp=3)
            vrinv *= -mol.atom_charge(atom)
        h1_kin = np.zeros((3, nao, nao))
        h1_nuca = np.zeros((3, nao, nao))
        h1_kin[:, p0:p1] += h_kin[:, p0:p1]
        h1_nuca[:, p0:p1] += h_nuc[:, p0:p1]
        h1_kin = h1_kin + h1_kin.transpose(0, 2, 1)
        h1_nuca = h1_nuca + h1_nuca.transpose(0, 2, 1)
        h1_nucc = vrinv + vrinv.transpose(0, 2, 1)

        terms["s_im1"][atom] += np.einsum("xij,ij->x", s1[:, p0:p1], im1[p0:p1])
        terms["s_im1"][atom] += np.einsum("xji,ij->x", s1[:, p0:p1], im1[:, p0:p1])
        terms["h1_kinetic"][atom] += np.einsum("xij,ji->x", h1_kin, dm1)
        terms["h1_nuc_a"][atom] += np.einsum("xij,ji->x", h1_nuca, dm1)
        terms["h1_nuc_c"][atom] += np.einsum("xij,ji->x", h1_nucc, dm1)
        terms["h1"][atom] += terms["h1_kinetic"][atom] + terms["h1_nuc_a"][atom] + terms["h1_nuc_c"][atom]
        terms["s_zeta"][atom] -= np.einsum("xij,ij->x", s1[:, p0:p1], zeta[p0:p1])
        terms["s_zeta"][atom] -= np.einsum("xji,ij->x", s1[:, p0:p1], zeta[:, p0:p1])
        terms["s_vhf"][atom] -= np.einsum("xij,ij->x", s1[:, p0:p1], vhf_s1occ[p0:p1]) * 2.0
        terms["vhf1"][atom] -= np.einsum("xij,ij->x", vhf1_blocks[atom], dm1p)

    ip1 = 0
    for atom in range(natm):
        shl0, shl1, p0, _ = offsetdic[atom]
        ip1 = p0
        max_memory = max(0, post.max_memory - lib.current_memory()[0])
        blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor("int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice).reshape(3, nf, nao, -1)
            for comp in range(3):
                eri1_full = lib.unpack_tril(eri1[comp].reshape(nf * nao, -1)).reshape(nf, nao, nao, nao)
                terms["vhf1_rs"][atom, comp] -= np.einsum("ijkl,ij,kl->", eri1_full, hf_dm1[ip0:ip1], dm1p)
                terms["vhf1_rq"][atom, comp] += 0.5 * np.einsum("ijkl,il,kj->", eri1_full, hf_dm1[ip0:ip1], dm1p)
                terms["vhf1_pq"][atom, comp] -= np.einsum("ijkl,kl,ij->", eri1_full, hf_dm1, dm1p[ip0:ip1])
                terms["vhf1_ps"][atom, comp] += 0.5 * np.einsum("ijkl,jk,il->", eri1_full, hf_dm1, dm1p[ip0:ip1])

    terms["electronic"] = (
        terms["two_e"] + terms["h1"] + terms["s_im1"] + terms["s_zeta"] + terms["s_vhf"] + terms["vhf1"]
    )
    return terms


def sorted_errors(planck_terms: dict[str, np.ndarray], pyscf_terms: dict[str, np.ndarray]) -> list[tuple[str, float, float, tuple[int, int]]]:
    rows: list[tuple[str, float, float, tuple[int, int]]] = []
    for name, planck in planck_terms.items():
        if name not in pyscf_terms:
            continue
        diff = planck - pyscf_terms[name]
        max_abs = float(np.max(np.abs(diff)))
        rms = float(math.sqrt(np.mean(diff * diff)))
        worst = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        rows.append((name, max_abs, rms, (int(worst[0]), int(worst[1]))))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Planck and PySCF RMP2 gradient terms.")
    parser.add_argument("--case", default="water_rmp2_gradient_sto3g", choices=sorted(CASE_INPUTS))
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    args = parser.parse_args()

    input_path = CASE_INPUTS[args.case]
    executable = args.build_dir / "hartree-fock"
    if not executable.exists():
        raise SystemExit(f"hartree-fock binary not found at {executable}")

    planck_terms, standard_coords = run_planck_terms(executable, input_path)
    pyscf_terms = compute_pyscf_rmp2_terms(input_path)

    if standard_coords is not None:
        spec = parse_input_file(input_path)
        rotation = build_rotation(extract_input_coords(spec), standard_coords)
        pyscf_terms = {name: rotate_rows(rows, rotation) for name, rows in pyscf_terms.items()}

    ranking = sorted_errors(planck_terms, pyscf_terms)
    print("Sorted RMP2 term errors (Planck vs PySCF)")
    print(f"{'term':<10} {'max_abs_err':>16} {'rms_err':>16} {'worst component':>20}")
    for name, max_abs, rms, (atom, axis) in ranking:
        axis_name = "xyz"[axis]
        print(f"{name:<10} {max_abs:>16.9e} {rms:>16.9e} {f'atom {atom + 1}, {axis_name}':>20}")

    print()
    for name, _, _, _ in ranking:
        planck = planck_terms[name]
        pyscf = pyscf_terms[name]
        diff = planck - pyscf
        print(name)
        print("Planck", np.array2string(planck, precision=10, separator=", "))
        print("PySCF ", np.array2string(pyscf, precision=10, separator=", "))
        print("Diff  ", np.array2string(diff, precision=10, separator=", "))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
