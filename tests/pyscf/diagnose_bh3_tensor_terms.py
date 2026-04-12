#!/usr/bin/env python3
from __future__ import annotations

import functools
import numpy as np

from pyscf import cc, gto, scf
from pyscf.cc import rccsdt, rccsdt_highm
from pyscf.cc.rccsdt import _einsum


def build_molecule():
    mol = gto.Mole()
    mol.atom = """
B   0.000000  0.000000  0.000000
H   1.190000  0.000000  0.000000
H  -0.595000  1.030570  0.000000
H  -0.595000 -1.030570  0.000000
"""
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.cart = True
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    return mol


def same_spin(p: int, q: int) -> bool:
    return (p % 2) == (q % 2)


def spatial_index(p: int) -> int:
    return p // 2


def t3_p201(t3, i, j, k, a, b, c):
    return (
        2.0 * t3[i, j, k, a, b, c]
        - t3[i, j, k, b, a, c]
        - t3[i, j, k, c, b, a]
    )


def t3_p422(t3, i, j, k, a, b, c):
    return (
        4.0 * t3[i, j, k, a, b, c]
        - 2.0 * t3[i, j, k, a, c, b]
        - 2.0 * t3[i, j, k, b, a, c]
        + t3[i, j, k, b, c, a]
        + t3[i, j, k, c, a, b]
        - 2.0 * t3[i, j, k, c, b, a]
    )


def build_spin_system(mf, eris):
    nmo = mf.mo_coeff.shape[1]
    nso = 2 * nmo
    nocc = 2 * mf.mol.nelectron // 2
    nvirt = nso - nocc
    fock_mo = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff

    fock = np.zeros((nso, nso))
    eri = np.zeros((nso, nso, nso, nso))

    for p in range(nso):
        for q in range(nso):
            if same_spin(p, q):
                fock[p, q] = fock_mo[spatial_index(p), spatial_index(q)]

    for p in range(nso):
        for r in range(nso):
            for q in range(nso):
                for s in range(nso):
                    if same_spin(p, q) and same_spin(r, s):
                        eri[p, r, q, s] = eris.pppp[
                            spatial_index(p),
                            spatial_index(r),
                            spatial_index(q),
                            spatial_index(s),
                        ]
    return nocc, nvirt, fock, eri


def build_dressed_system(fock, eri, nocc, t1):
    nmo = fock.shape[0]
    nvirt = nmo - nocc
    x = np.eye(nmo)
    y = np.eye(nmo)
    x[nocc:, :nocc] -= t1.T
    y[:nocc, nocc:] += t1

    dressed_fock = fock.copy()
    for r in range(nmo):
        for s in range(nmo):
            for i in range(nocc):
                for a in range(nvirt):
                    va = nocc + a
                    dressed_fock[r, s] += 2.0 * eri[r, i, s, va] * t1[i, a]
                    dressed_fock[r, s] -= eri[r, i, va, s] * t1[i, a]
    dressed_fock = x @ dressed_fock @ y.T

    einsum = functools.partial(_einsum, "numpy")
    t1_eris = einsum("tvuw,pt->pvuw", eri, x)
    t1_eris = einsum("pvuw,rv->pruw", t1_eris, x)
    t1_eris = np.ascontiguousarray(t1_eris.transpose(2, 3, 0, 1))
    t1_eris = einsum("uwpr,qu->qwpr", t1_eris, y)
    t1_eris = einsum("qwpr,sw->qspr", t1_eris, y)
    t1_eris = np.ascontiguousarray(t1_eris.transpose(2, 3, 0, 1))
    return dressed_fock, t1_eris


def build_sd_intermediates(nocc, t1_fock, t1_eris, t2):
    nvirt = t1_fock.shape[0] - nocc
    einsum = functools.partial(_einsum, "numpy")

    F_vv = t1_fock[nocc:, nocc:].copy()
    einsum("kldc,kldb->bc", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_vv, alpha=-2.0, beta=1.0)
    einsum("klcd,kldb->bc", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_vv, alpha=1.0, beta=1.0)

    F_oo = t1_fock[:nocc, :nocc].copy()
    einsum("lkcd,ljcd->kj", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_oo, alpha=2.0, beta=1.0)
    einsum("lkdc,ljcd->kj", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_oo, alpha=-1.0, beta=1.0)

    W_oooo = t1_eris[:nocc, :nocc, :nocc, :nocc].copy()
    einsum("klcd,ijcd->klij", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_oooo, alpha=1.0, beta=1.0)

    W_ovvo = -t1_eris[:nocc, nocc:, nocc:, :nocc].copy()
    einsum("klcd,ilad->kaci", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=-1.0, beta=1.0)
    einsum("kldc,ilad->kaci", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=0.5, beta=1.0)
    einsum("klcd,ilda->kaci", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovvo, alpha=0.5, beta=1.0)

    W_ovov = -t1_eris[:nocc, nocc:, :nocc, nocc:].copy()
    einsum("kldc,liad->kaic", t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovov, alpha=0.5, beta=1.0)

    return F_oo, F_vv, W_oooo, W_ovvo, W_ovov


def build_sd_residuals(nocc, t1_fock, t1_eris, F_oo, F_vv, W_oooo, W_ovvo, W_ovov, t2):
    einsum = functools.partial(_einsum, "numpy")
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)

    r1 = t1_fock[nocc:, :nocc].T.copy()
    einsum("kc,ikac->ia", t1_fock[:nocc, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum("akcd,ikcd->ia", t1_eris[nocc:, :nocc, nocc:, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum("klic,klac->ia", t1_eris[:nocc, :nocc, :nocc, nocc:], c_t2, out=r1, alpha=-1.0, beta=1.0)

    r2 = 0.5 * t1_eris[nocc:, nocc:, :nocc, :nocc].T.copy()
    einsum("bc,ijac->ijab", F_vv, t2, out=r2, alpha=1.0, beta=1.0)
    einsum("kj,ikab->ijab", F_oo, t2, out=r2, alpha=-1.0, beta=1.0)
    einsum("abcd,ijcd->ijab", t1_eris[nocc:, nocc:, nocc:, nocc:], t2, out=r2, alpha=0.5, beta=1.0)
    einsum("klij,klab->ijab", W_oooo, t2, out=r2, alpha=0.5, beta=1.0)
    einsum("kajc,ikcb->ijab", W_ovov, t2, out=r2, alpha=1.0, beta=1.0)
    einsum("kaci,kjcb->ijab", W_ovvo, t2, out=r2, alpha=-2.0, beta=1.0)
    einsum("kaic,kjcb->ijab", W_ovov, t2, out=r2, alpha=1.0, beta=1.0)
    einsum("kaci,jkcb->ijab", W_ovvo, t2, out=r2, alpha=1.0, beta=1.0)
    return r1, r2


def add_t3_feedback(nocc, t1_fock, t1_eris, r1, r2, t3):
    for i in range(nocc):
        for a in range(t3.shape[3]):
            for j in range(nocc):
                for k in range(nocc):
                    for b in range(t3.shape[4]):
                        for c in range(t3.shape[5]):
                            r1[i, a] += 0.5 * t1_eris[j, k, nocc + b, nocc + c] * t3_p422(t3, k, i, j, c, a, b)

    for i in range(nocc):
        for j in range(nocc):
            for a in range(t3.shape[3]):
                for b in range(t3.shape[4]):
                    for k in range(nocc):
                        for c in range(t3.shape[5]):
                            r2[i, j, a, b] += 0.5 * t1_fock[k, nocc + c] * t3_p201(t3, k, i, j, c, a, b)
                    for k in range(nocc):
                        for c in range(t3.shape[4]):
                            for d in range(t3.shape[5]):
                                r2[i, j, a, b] += t1_eris[nocc + b, k, nocc + c, nocc + d] * t3_p201(t3, k, i, j, d, a, c)
                    for k in range(nocc):
                        for l in range(nocc):
                            for c in range(t3.shape[5]):
                                r2[i, j, a, b] -= t1_eris[j, k, l, nocc + c] * t3_p201(t3, k, i, j, c, a, b)


def report_diff(name, lhs, rhs):
    diff = np.max(np.abs(lhs - rhs))
    print(f"{name:24s} {diff:.6e}")
    return diff


if __name__ == "__main__":
    mol = build_molecule()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    mycc = cc.RCCSDT(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()

    eris = mycc.ao2mo(mycc.mo_coeff)
    nocc = mycc.nocc
    t1, t2, t3 = mycc.tamps

    nocc_so, nvirt_so, so_fock, so_eri = build_spin_system(mf, eris)
    print(f"PySCF t1 shape:        {t1.shape}")
    print(f"PySCF t2 shape:        {t2.shape}")
    print(f"PySCF t3 shape:        {t3.shape}")
    print(f"Planck staged t1 size: ({nocc_so}, {nvirt_so})")
    print(f"Planck staged t2 size: ({nocc_so}, {nocc_so}, {nvirt_so}, {nvirt_so})")
    print(f"Planck staged t3 size: ({nocc_so}, {nocc_so}, {nocc_so}, {nvirt_so}, {nvirt_so}, {nvirt_so})")

    if t1.shape != (nocc_so, nvirt_so):
        print()
        print("DIAGNOSIS: local PySCF RCCSDT uses restricted spatial-orbital amplitudes,")
        print("while the current Planck no-fallback tensor rewrite is assembling a")
        print("spin-orbital dressed system. A direct term-by-term comparison is")
        print("therefore invalid at the first amplitude layer.")
        raise SystemExit(0)

    dressed_fock, dressed_eri = build_dressed_system(so_fock, so_eri, nocc_so, t1)

    imds = rccsdt._IMDS()
    rccsdt.update_t1_fock_eris(mycc, imds, t1, eris)
    report_diff("t1_fock", dressed_fock, imds.t1_fock)
    report_diff("t1_eris", dressed_eri, imds.t1_eris)

    F_oo, F_vv, W_oooo, W_ovvo, W_ovov = build_sd_intermediates(nocc_so, dressed_fock, dressed_eri, t2)
    imds2 = rccsdt._IMDS()
    imds2.t1_fock = imds.t1_fock.copy()
    imds2.t1_eris = imds.t1_eris.copy()
    rccsdt.intermediates_t1t2(mycc, imds2, t2)
    report_diff("F_oo", F_oo, imds2.F_oo)
    report_diff("F_vv", F_vv, imds2.F_vv)
    report_diff("W_oooo", W_oooo, imds2.W_oooo)
    report_diff("W_ovvo", W_ovvo, imds2.W_ovvo)
    report_diff("W_ovov", W_ovov, imds2.W_ovov)

    r1, r2 = build_sd_residuals(nocc_so, dressed_fock, dressed_eri, F_oo, F_vv, W_oooo, W_ovvo, W_ovov, t2)
    py_r1, py_r2 = rccsdt.compute_r1r2(mycc, imds2, t2)
    report_diff("r1(no t3)", r1, py_r1)
    report_diff("r2(no t3)", r2, py_r2)

    add_t3_feedback(nocc_so, dressed_fock, dressed_eri, r1, r2, t3)
    py_r1_t3, py_r2_t3 = py_r1.copy(), py_r2.copy()
    rccsdt_highm.r1r2_add_t3_(mycc, imds, py_r1_t3, py_r2_t3, t3)
    report_diff("r1(with t3)", r1, py_r1_t3)
    report_diff("r2(with t3)", r2, py_r2_t3)
