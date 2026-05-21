"""Canonical PySCF reproduction of the RMP2 gradient intermediates.

Every comparison script used to inline its own copy of this chain. It now lives
in one place: :func:`build_intermediates` walks ``part_dm2`` -> ``dm2buf`` ->
``Imat`` -> ``Xvo`` -> ``z`` -> relaxed density, and :func:`build_terms`
reproduces the per-atom gradient term decomposition. Both mirror
``pyscf/grad/mp2.py`` and Planck's ``src/post_hf/mp2_gradient.cpp`` so the two
can be diffed stage by stage.

Closed-shell, no frozen orbitals (the regime of the RMP2 gradient cases).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np

from ._runtime import ensure_pyscf_runtime

ensure_pyscf_runtime()

from pyscf import lib, mp  # noqa: E402
from pyscf.ao2mo import _ao2mo  # noqa: E402
from pyscf.grad import mp2 as pyscf_grad_mp2  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402
from pyscf.mp import mp2 as pyscf_mp2  # noqa: E402
from pyscf.scf import cphf  # noqa: E402

from benchmark import build_mean_field, parse_mp2_frozen  # noqa: E402
from input_utils import parse_input_file  # noqa: E402


@dataclass
class MP2Context:
    """Converged SCF + MP2 plus the orbital partition used everywhere below."""

    post: object  # pyscf.mp.MP2
    mol: object
    mo_coeff: np.ndarray
    mo_energy: np.ndarray
    mo_occ: np.ndarray
    nocc: int
    nvirt: int
    nmo: int
    nao: int
    orbo: np.ndarray
    orbv: np.ndarray
    t2: np.ndarray
    doo: np.ndarray
    dvv: np.ndarray
    hf_dm1: np.ndarray


def build_context(input_path: Path) -> MP2Context:
    """Run PySCF SCF+MP2 for a Planck ``.hfinp`` and collect shared quantities."""

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

    if pyscf_grad_mp2.has_frozen_orbitals(post):
        raise RuntimeError("Frozen-orbital reference is not implemented in rmp2_grad.")

    t2 = post.t2
    doo, dvv = pyscf_mp2._gamma1_intermediates(post, t2)
    mo_coeff = post.mo_coeff
    mo_occ = post.mo_occ
    nocc = int(np.count_nonzero(mo_occ > 0))
    nmo = mo_coeff.shape[1]
    OA, VA, _, _ = pyscf_grad_mp2._index_frozen_active(post.get_frozen_mask(), mo_occ)
    orbo = mo_coeff[:, OA]
    orbv = mo_coeff[:, VA]
    return MP2Context(
        post=post,
        mol=post.nuc_grad_method().mol,
        mo_coeff=mo_coeff,
        mo_energy=post._scf.mo_energy,
        mo_occ=mo_occ,
        nocc=nocc,
        nvirt=nmo - nocc,
        nmo=nmo,
        nao=orbo.shape[0],
        orbo=orbo,
        orbv=orbv,
        t2=t2,
        doo=doo,
        dvv=dvv,
        hf_dm1=post._scf.make_rdm1(mo_coeff, mo_occ),
    )


def _part_dm2(ctx: MP2Context) -> np.ndarray:
    part = _ao2mo.nr_e2(
        ctx.t2.reshape(ctx.nocc**2, ctx.nvirt**2),
        np.asarray(ctx.orbv.T, order="F"),
        (0, ctx.nao, 0, ctx.nao),
        "s1",
        "s1",
    ).reshape(ctx.nocc, ctx.nocc, ctx.nao, ctx.nao)
    return part.transpose(0, 2, 3, 1) * 4.0 - part.transpose(0, 3, 2, 1) * 2.0


def _shell_prange(mol, start: int, stop: int, blksize: int):
    yield from pyscf_grad_mp2._shell_prange(mol, start, stop, blksize)


def _accumulate_imat_and_de(ctx: MP2Context, part_dm2: np.ndarray, *, want_de: bool):
    """Walk PySCF's atom/shell loop, accumulating Imat (and the 2e gradient)."""

    mol = ctx.mol
    nao = ctx.nao
    offsetdic = mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    imat = np.zeros((nao, nao))
    de = np.zeros((mol.natm, 3)) if want_de else None
    max_memory = max(0, ctx.post.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))

    for atom in range(mol.natm):
        shl0, shl1, p0, _ = offsetdic[atom]
        ip1 = p0
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum("pi,iqrj->pqrj", ctx.orbo[ip0:ip1], part_dm2)
            dm2buf += lib.einsum("qi,iprj->pqrj", ctx.orbo, part_dm2[:, ip0:ip1])
            dm2buf = lib.einsum("pqrj,sj->pqrs", dm2buf, ctx.orbo)
            dm2buf = dm2buf + dm2buf.transpose(0, 1, 3, 2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1, nao, nao)).reshape(nf, nao, -1)
            dm2buf[:, :, diagidx] *= 0.5

            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri0 = mol.intor("int2e", aosym="s2kl", shls_slice=shls_slice)
            imat += lib.einsum("ipx,iqx->pq", eri0.reshape(nf, nao, -1), dm2buf)
            if want_de:
                eri1 = mol.intor(
                    "int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice
                ).reshape(3, nf, nao, -1)
                de[atom] -= np.einsum("xijk,ijk->x", eri1, dm2buf) * 2.0
    return imat, de


def _fvind(ctx: MP2Context):
    mo, nocc = ctx.mo_coeff, ctx.nocc

    def apply(x):
        x = x.reshape((ctx.nvirt, nocc))
        dm = reduce(np.dot, (mo[:, nocc:], x, mo[:, :nocc].T))
        v = ctx.post._scf.get_veff(ctx.post.mol, dm + dm.T)
        return reduce(np.dot, (mo[:, nocc:].T, v, mo[:, :nocc])) * 2.0

    return apply


def build_intermediates(input_path: Path) -> dict[str, np.ndarray]:
    """Full PySCF intermediate chain, keyed to match Planck's debug-dump names.

    Returns A (CPHF matrix), xvo (CPHF rhs), z, and the response-density chain
    (corr_relaxed_mo, P_ao, dm1_corr_relaxed_ao, dm1p) plus dm2buf-related stats.
    """

    ctx = build_context(input_path)
    mo, nocc, nmo = ctx.mo_coeff, ctx.nocc, ctx.nmo

    dm1_corr_mo = np.zeros((nmo, nmo))
    dm1_corr_mo[:nocc, :nocc] = ctx.doo + ctx.doo.T
    dm1_corr_mo[nocc:, nocc:] = ctx.dvv + ctx.dvv.T
    dm1_corr_ao = reduce(np.dot, (mo, dm1_corr_mo, mo.T))
    vhf = ctx.post._scf.get_veff(ctx.post.mol, dm1_corr_ao) * 2.0

    part_dm2 = _part_dm2(ctx)
    imat, _ = _accumulate_imat_and_de(ctx, part_dm2, want_de=False)
    imat_mo = reduce(np.dot, (mo.T, imat, ctx.post._scf.get_ovlp(), mo)) * -1.0

    xvo = reduce(np.dot, (mo[:, nocc:].T, vhf, mo[:, :nocc]))
    xvo += imat_mo[:nocc, nocc:].T - imat_mo[nocc:, :nocc]

    fvind = _fvind(ctx)
    A = np.zeros((ctx.nvirt * nocc, ctx.nvirt * nocc))
    for idx in range(ctx.nvirt * nocc):
        one_hot = np.zeros(ctx.nvirt * nocc)
        one_hot[idx] = 1.0
        A[:, idx] = fvind(one_hot).ravel()
    for a in range(ctx.nvirt):
        for i in range(nocc):
            A[a * nocc + i, a * nocc + i] += ctx.mo_energy[nocc + a] - ctx.mo_energy[i]

    z = cphf.solve(fvind, ctx.mo_energy, ctx.mo_occ, xvo, max_cycle=30)[0]

    corr_relaxed_mo = dm1_corr_mo.copy()
    corr_relaxed_mo[nocc:, :nocc] = z
    corr_relaxed_mo[:nocc, nocc:] = z.T

    p_mo = np.zeros((nmo, nmo))
    p_mo[:nocc, :nocc] = 2.0 * np.eye(nocc)
    p_mo += corr_relaxed_mo
    p_ao = reduce(np.dot, (mo, p_mo, mo.T))
    dm1_corr_relaxed_ao = reduce(np.dot, (mo, corr_relaxed_mo, mo.T))
    dm1p = ctx.hf_dm1 + dm1_corr_relaxed_ao * 2.0

    return {
        "A": A,
        "xvo": xvo,
        "rhs": xvo,  # alias: Planck dumps this block as "rhs"
        "z": z,
        "imat_mo": imat_mo,
        "imat_top_right": imat_mo[:nocc, nocc:],  # occ-virt block
        "imat_bottom_left": imat_mo[nocc:, :nocc],  # virt-occ block
        "corr_relaxed_mo": corr_relaxed_mo,
        "P_ao": p_ao,
        "dm1_corr_relaxed_ao": dm1_corr_relaxed_ao,
        "dm1p": dm1p,
    }


def build_terms(input_path: Path) -> dict[str, np.ndarray]:
    """Per-atom gradient term decomposition matching Planck's PLANCK_RMP2_TERM_ROW.

    Faithful re-implementation of ``pyscf/grad/mp2.py`` ``grad_elec``, split into
    the same named terms Planck prints (two_e, h1[_kinetic/_nuc_a/_nuc_c], s_im1,
    s_zeta, s_vhf, vhf1[_rs/_rq/_pq/_ps], electronic).
    """

    ctx = build_context(input_path)
    mol, nao, nocc, nmo = ctx.mol, ctx.nao, ctx.nocc, ctx.nmo
    mo, mo_energy = ctx.mo_coeff, ctx.mo_energy
    natm = mol.natm
    hf_dm1 = ctx.hf_dm1
    offsetdic = mol.offset_nr_by_atom()

    part_dm2 = _part_dm2(ctx)
    imat, two_e = _accumulate_imat_and_de(ctx, part_dm2, want_de=True)

    # vhf1 HF-part blocks (per atom, per component).
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    vhf1_blocks = np.zeros((natm, 3, nao, nao))
    max_memory = max(0, ctx.post.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))
    for atom in range(natm):
        shl0, shl1, p0, _ = offsetdic[atom]
        ip1 = p0
        vhf = np.zeros((3, nao, nao))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor(
                "int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice
            ).reshape(3, nf, nao, -1)
            for comp in range(3):
                eri1_full = lib.unpack_tril(eri1[comp].reshape(nf * nao, -1)).reshape(nf, nao, nao, nao)
                vhf[comp] += np.einsum("ijkl,ij->kl", eri1_full, hf_dm1[ip0:ip1])
                vhf[comp] -= np.einsum("ijkl,il->kj", eri1_full, hf_dm1[ip0:ip1]) * 0.5
                vhf[comp, ip0:ip1] += np.einsum("ijkl,kl->ij", eri1_full, hf_dm1)
                vhf[comp, ip0:ip1] -= np.einsum("ijkl,jk->il", eri1_full, hf_dm1) * 0.5
        vhf1_blocks[atom] = vhf

    imat = reduce(np.dot, (mo.T, imat, ctx.post._scf.get_ovlp(), mo)) * -1.0

    dm1mo = np.zeros((nmo, nmo))
    dm1mo[:nocc, :nocc] = ctx.doo + ctx.doo.T
    dm1mo[nocc:, nocc:] = ctx.dvv + ctx.dvv.T

    dm1 = reduce(np.dot, (mo, dm1mo, mo.T))
    vhf = ctx.post._scf.get_veff(mol, dm1) * 2.0
    xvo = reduce(np.dot, (mo[:, nocc:].T, vhf, mo[:, :nocc]))
    xvo += imat[:nocc, nocc:].T - imat[nocc:, :nocc]
    dm1mo += pyscf_grad_mp2._response_dm1(ctx.post, xvo)

    imat[nocc:, :nocc] = imat[:nocc, nocc:].T
    im1 = reduce(np.dot, (mo, imat, mo.T))

    mf_grad = ctx.post._scf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)

    zeta = lib.direct_sum("i+j->ij", mo_energy, mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[:nocc].reshape(-1, 1)
    zeta = reduce(np.dot, (mo, zeta * dm1mo, mo.T))

    dm1 = reduce(np.dot, (mo, dm1mo, mo.T))
    p1 = np.dot(mo[:, :nocc], mo[:, :nocc].T)
    vhf_s1occ = reduce(np.dot, (p1, ctx.post._scf.get_veff(mol, dm1 + dm1.T), p1))

    dm1p = hf_dm1 + dm1 * 2.0
    dm1 = dm1 + hf_dm1
    zeta = zeta + rhf_grad.make_rdm1e(mo_energy, mo, ctx.mo_occ)

    terms = {
        name: np.zeros((natm, 3))
        for name in (
            "h1", "h1_kinetic", "h1_nuc_a", "h1_nuc_c",
            "s_im1", "s_zeta", "s_vhf",
            "vhf1", "vhf1_rs", "vhf1_rq", "vhf1_pq", "vhf1_ps",
        )
    }
    terms["two_e"] = two_e

    h_kin = -mol.intor("int1e_ipkin", comp=3)
    h_nuc = -mol.intor("int1e_ipnuc", comp=3)
    for atom in range(natm):
        _, _, p0, p1_ = offsetdic[atom]
        with mol.with_rinv_at_nucleus(atom):
            vrinv = mol.intor("int1e_iprinv", comp=3) * -mol.atom_charge(atom)
        h1_kin = np.zeros((3, nao, nao))
        h1_nuca = np.zeros((3, nao, nao))
        h1_kin[:, p0:p1_] += h_kin[:, p0:p1_]
        h1_nuca[:, p0:p1_] += h_nuc[:, p0:p1_]
        h1_kin = h1_kin + h1_kin.transpose(0, 2, 1)
        h1_nuca = h1_nuca + h1_nuca.transpose(0, 2, 1)
        h1_nucc = vrinv + vrinv.transpose(0, 2, 1)

        terms["s_im1"][atom] += np.einsum("xij,ij->x", s1[:, p0:p1_], im1[p0:p1_])
        terms["s_im1"][atom] += np.einsum("xji,ij->x", s1[:, p0:p1_], im1[:, p0:p1_])
        terms["h1_kinetic"][atom] += np.einsum("xij,ji->x", h1_kin, dm1)
        terms["h1_nuc_a"][atom] += np.einsum("xij,ji->x", h1_nuca, dm1)
        terms["h1_nuc_c"][atom] += np.einsum("xij,ji->x", h1_nucc, dm1)
        terms["h1"][atom] += terms["h1_kinetic"][atom] + terms["h1_nuc_a"][atom] + terms["h1_nuc_c"][atom]
        terms["s_zeta"][atom] -= np.einsum("xij,ij->x", s1[:, p0:p1_], zeta[p0:p1_])
        terms["s_zeta"][atom] -= np.einsum("xji,ij->x", s1[:, p0:p1_], zeta[:, p0:p1_])
        terms["s_vhf"][atom] -= np.einsum("xij,ij->x", s1[:, p0:p1_], vhf_s1occ[p0:p1_]) * 2.0
        terms["vhf1"][atom] -= np.einsum("xij,ij->x", vhf1_blocks[atom], dm1p)

    for atom in range(natm):
        shl0, shl1, p0, _ = offsetdic[atom]
        ip1 = p0
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor(
                "int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice
            ).reshape(3, nf, nao, -1)
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


def total_gradient(input_path: Path) -> np.ndarray:
    """PySCF analytic RMP2 nuclear gradient, ``(natom, 3)``."""

    ctx = build_context(input_path)
    return ctx.post.nuc_grad_method().kernel()
