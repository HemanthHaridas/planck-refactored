"""Numpy transcription of the hand-written GCCSD doubles residual.

Ground-truth gate for ccgen's generated CCSD doubles equations. The C++ solver
`src/post_hf/cc/ccsd.cpp` is spin-orbital GCCSD, validated against PySCF, and
builds an explicitly antisymmetric r2 via Stanton-Gauss tau intermediates. The
generated kernels are not compiled into any binary, so the comparison has to be
Python-side: evaluate BOTH residuals as tensor contractions on the same random
amplitudes/integrals and diff the arrays.

Transcribed verbatim from `build_intermediates` / `build_residuals` (ccsd.cpp
lines 240-423). The C++ assumes a canonical HF reference (diagonal Fock), so its
`fae`/`fmi` carry `eps` on the diagonal; here we pass a general symmetric Fock
`f` and put its ov/vo/oo/vv blocks in directly, which is the same object ccgen's
`f(p,q)` denotes. Only the DOUBLES residual is transcribed -- that is where the
ccgen bug lives.

Not imported by the generator; test-only.
"""

from __future__ import annotations

import numpy as np


def build_tau(t1, t2):
    """tau_{ij}^{ab} = t2 + (t1_i^a t1_j^b - t1_i^b t1_j^a)."""
    pair = np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
    return t2 + pair


def build_tau_tilde(t1, t2):
    """tau_tilde = t2 + 1/2 (t1_i^a t1_j^b - t1_i^b t1_j^a)."""
    pair = np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
    return t2 + 0.5 * pair


def gccsd_doubles_residual(f, blocks, t1, t2):
    """The hand-written GCCSD r2(i,j,a,b), as a contraction.

    ``f`` is the general Fock in blocks: ``f["oo"], f["ov"], f["vo"], f["vv"]``.
    ``blocks`` holds the antisymmetrized spin-orbital integrals keyed by space
    signature (``oooo, ooov, oovv, ovov, ovvo, ovvv, vvvv``), each ``⟨pq||rs⟩``.
    """
    oovv = blocks["oovv"]
    ooov = blocks["ooov"]
    ovvv = blocks["ovvv"]
    ovov = blocks["ovov"]
    ovvo = blocks["ovvo"]
    oooo = blocks["oooo"]
    vvvv = blocks["vvvv"]

    tau = build_tau(t1, t2)
    tau_t = build_tau_tilde(t1, t2)

    # ---- Fae (ccsd.cpp 262-274), general Fock vv block on the diagonal ----
    fae = f["vv"].copy()
    fae += np.einsum("mf,mafe->ae", t1, ovvv)          # + t1(m,f) ovvv(m,a,f,e)
    fae -= 0.5 * np.einsum("mnaf,mnef->ae", tau_t, oovv)

    # ---- Fmi (ccsd.cpp 276-288) ----
    fmi = f["oo"].copy()
    fmi += np.einsum("ne,mnie->mi", t1, ooov)
    fmi += 0.5 * np.einsum("inef,mnef->mi", tau_t, oovv)

    # ---- Fme (ccsd.cpp 256-260): the ov Fock block plus t1 driving ----
    fme = f["ov"].copy()
    fme += np.einsum("nf,mnef->me", t1, oovv)

    # ---- Wmnij (ccsd.cpp 290-303) ----
    wmnij = oooo.copy()
    wmnij += np.einsum("je,mnie->mnij", t1, ooov)
    wmnij -= np.einsum("ie,mnje->mnij", t1, ooov)
    wmnij += 0.25 * np.einsum("ijef,mnef->mnij", tau, oovv)

    # ---- Wabef (ccsd.cpp 305-318) ----
    wabef = vvvv.copy()
    wabef += np.einsum("mb,maef->abef", t1, ovvv)
    wabef -= np.einsum("ma,mbef->abef", t1, ovvv)
    wabef += 0.25 * np.einsum("mnab,mnef->abef", tau, oovv)

    # ---- Wmbej (ccsd.cpp 320-336) ----
    wmbej = ovvo.copy()
    wmbej += np.einsum("jf,mbef->mbej", t1, ovvv)
    wmbej += np.einsum("nb,mnje->mbej", t1, ooov)
    wmbej -= np.einsum("jnfb,mnef->mbej", 0.5 * t2, oovv)
    wmbej -= np.einsum("jf,nb,mnef->mbej", t1, t1, oovv)

    # ---- r2 (ccsd.cpp 378-423) ----
    r2 = oovv.copy()
    # Fae terms  (P(ab))
    r2 += np.einsum("ijae,be->ijab", t2, fae)
    r2 -= np.einsum("ijbe,ae->ijab", t2, fae)
    # Fmi terms  (P(ij))
    r2 -= np.einsum("imab,mj->ijab", t2, fmi)
    r2 += np.einsum("jmab,mi->ijab", t2, fmi)
    # ladders
    r2 += 0.5 * np.einsum("mnab,mnij->ijab", tau, wmnij)
    r2 += 0.5 * np.einsum("ijef,abef->ijab", tau, wabef)
    # ring  (P(ij)P(ab))
    r2 += np.einsum("imae,mbej->ijab", t2, wmbej)
    r2 -= np.einsum("imbe,maej->ijab", t2, wmbej)
    r2 -= np.einsum("jmae,mbei->ijab", t2, wmbej)
    r2 += np.einsum("jmbe,maei->ijab", t2, wmbej)
    # t1*t1*ovov correction (P(ij)P(ab))
    r2 += np.einsum("ie,ma,mbje->ijab", t1, t1, ovov)
    r2 -= np.einsum("ie,mb,maje->ijab", t1, t1, ovov)
    r2 -= np.einsum("je,ma,mbie->ijab", t1, t1, ovov)
    r2 += np.einsum("je,mb,maie->ijab", t1, t1, ovov)
    # singles ovvv (P(ij))
    r2 += np.einsum("ie,jeba->ijab", t1, ovvv)
    r2 -= np.einsum("je,ieba->ijab", t1, ovvv)
    # singles ooov (P(ab))
    r2 -= np.einsum("ma,ijmb->ijab", t1, ooov)
    r2 += np.einsum("mb,ijma->ijab", t1, ooov)

    return r2
