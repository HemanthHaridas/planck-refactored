"""W0 -- primitive-level per-structure weight table for the reference r2.

Companion to `docs/CCGEN_RAW_GENERATION_WEIGHT_SCOPE.md`.  Expands the
hand-written, PySCF-validated GCCSD doubles reference (`gccsd_reference.py`)
into its individual t1-containing primitive contractions, each carrying a
rational coefficient and an einsum "structure signature".  This is the
ground-truth weight table W1 diffs ccgen against.

Every intermediate (fae/fmi/fme/wmnij/wabef/wmbej, and tau/tau_tilde) is folded
into r2 by hand, and each resulting `t1*t2*v`, `t1*t1*t2*v`, `t1*v`, `t1*t1*v`
term is listed explicitly below.  The check `verify()` evaluates the table
numerically and asserts it reproduces the reference's t1-part exactly (the
t1-part is obtained by differencing the reference at t1 and at t1=0).

Structure signature = (sorted factor names, einsum subscript string).  The
einsum string names which tensor carries which external, on which ERI block,
so it is directly comparable to ccgen's per-term structure.

ponytail: table is transcribed by hand from gccsd_reference; the numeric
verify() is the guard -- any transcription slip fails it.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

# Each entry: (coeff, einsum_subscripts, operand_keys).  Output index order is
# always "ijab" (the reference's r2 layout).  Operand keys:
#   t1        -> t1[o,v]
#   t2        -> t2[o,o,v,v]
#   <block>   -> antisymmetrized ERI slice, e.g. "ovvv" -> g[o,v,v,v]
# so a term is coeff * einsum(subs, *operands) accumulated into r2[i,j,a,b].
#
# Derivation: r2 with every intermediate expanded, keeping only terms that
# contain at least one t1.  Grouped by their origin comment.
PRIMITIVE_T1_TERMS: list[tuple[Fraction, str, tuple[str, ...]]] = [
    # ---- Fae into P(ab) t2 Fae ----------------------------------------------
    # fae += t1(m,f) ovvv(m,a,f,e)   -> +t2(ijae) t1(mf) ovvv(m,b,f,e)  [P(ab)]
    (Fraction(1), "ijae,mf,mbfe->ijab", ("t2", "t1", "ovvv")),
    (Fraction(-1), "ijbe,mf,mafe->ijab", ("t2", "t1", "ovvv")),
    # r2 += t2(ijae) fae(b,e); fae(b,e) -= 1/2 tau_t(mn,bf) oovv(mn,ef).
    # tau_t t1t1 half on (b,f): 1/2 (t1(m,b)t1(n,f) - t1(m,f)t1(n,b)).
    # so contribution = -1/4 t2(ijae)(t1(mb)t1(nf)-t1(mf)t1(nb))oovv(mnef), P(ab).
    (Fraction(-1, 4), "ijae,mb,nf,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(1, 4), "ijae,mf,nb,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(1, 4), "ijbe,ma,nf,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(-1, 4), "ijbe,mf,na,mnef->ijab", ("t2", "t1", "t1", "oovv")),

    # ---- Fmi into -P(ij) t2 Fmi ---------------------------------------------
    # fmi += t1(n,e) ooov(m,n,i,e)  -> -t2(imab) t1(ne) ooov(m,n,j,e) [-P(ij)]
    (Fraction(-1), "imab,ne,mnje->ijab", ("t2", "t1", "ooov")),
    (Fraction(1), "jmab,ne,mnie->ijab", ("t2", "t1", "ooov")),
    # r2 -= t2(imab) fmi(m,j); fmi(m,j) += 1/2 tau_t(jn,ef) oovv(mn,ef).
    # tau_t t1t1 half on (j,n): 1/2 (t1(j,e)t1(n,f) - t1(j,f)t1(n,e)).
    # contribution = -1/4 t2(imab)(t1(je)t1(nf)-t1(jf)t1(ne))oovv(mnef), -P(ij).
    (Fraction(-1, 4), "imab,je,nf,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(1, 4), "imab,jf,ne,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(1, 4), "jmab,ie,nf,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(-1, 4), "jmab,if,ne,mnef->ijab", ("t2", "t1", "t1", "oovv")),

    # ---- Fme correction: none here (Fme has t1 via oovv but enters only ----
    #      inside Fae/Fmi as tau_t; already covered above). ---------------------

    # ---- 1/2 tau_abmn Wmnij --------------------------------------------------
    # tau(ab,mn) = t2 + (t1(m,a)t1(n,b)-t1(m,b)t1(n,a))  [note tau externals a,b sit on m,n slots]
    # Wmnij t1 pieces: +t1(je)ooov(mnie) - t1(ie)ooov(mnje) + 1/4 tau(ijef)oovv(mnef)
    #
    # (A) t2 * Wmnij-t1pieces:
    (Fraction(1, 2), "mnab,je,mnie->ijab", ("t2", "t1", "ooov")),
    (Fraction(-1, 2), "mnab,ie,mnje->ijab", ("t2", "t1", "ooov")),
    # (B) t2 * 1/4 tau(ijef) t1t1 piece of tau -> 1/8 t2(mnab)(t1(ie)t1(jf)-t1(if)t1(je))oovv(mnef)
    (Fraction(1, 8), "mnab,ie,jf,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(-1, 8), "mnab,if,je,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    # (C) tau-t1t1(ab) * oooo:  1/2 (t1(ma)t1(nb)-t1(mb)t1(na)) oooo(mnij)
    (Fraction(1, 2), "ma,nb,mnij->ijab", ("t1", "t1", "oooo")),
    (Fraction(-1, 2), "mb,na,mnij->ijab", ("t1", "t1", "oooo")),
    # (D) tau-t1t1(ab) * Wmnij-t1pieces (t1t1t1) :
    (Fraction(1, 2), "ma,nb,je,mnie->ijab", ("t1", "t1", "t1", "ooov")),
    (Fraction(-1, 2), "ma,nb,ie,mnje->ijab", ("t1", "t1", "t1", "ooov")),
    (Fraction(-1, 2), "mb,na,je,mnie->ijab", ("t1", "t1", "t1", "ooov")),
    (Fraction(1, 2), "mb,na,ie,mnje->ijab", ("t1", "t1", "t1", "ooov")),
    # (E) tau-t1t1(ab) * 1/4 tau(ijef)-t1t1 -> t1t1t1t1:
    (Fraction(1, 8), "ma,nb,ie,jf,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    (Fraction(-1, 8), "ma,nb,if,je,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    (Fraction(-1, 8), "mb,na,ie,jf,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    (Fraction(1, 8), "mb,na,if,je,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    # (F) tau-t1t1(ab) * 1/4 tau(ijef)-t2 cross term -> t1t1*t2:
    (Fraction(1, 8), "ma,nb,ijef,mnef->ijab", ("t1", "t1", "t2", "oovv")),
    (Fraction(-1, 8), "mb,na,ijef,mnef->ijab", ("t1", "t1", "t2", "oovv")),
    # note: tau_abmn * 1/4 tau_ijef oovv where BOTH are t2 -> t2*t2, no t1, skip.

    # ---- 1/2 tau_efij Wabef --------------------------------------------------
    # tau(ef,ij) = t2 + (t1(i,e)t1(j,f)-t1(i,f)t1(j,e))
    # Wabef t1 pieces: +t1(mb)ovvv(maef) - t1(ma)ovvv(mbef) + 1/4 tau(mnab)oovv(mnef)
    #
    # (A) t2(ijef) * Wabef-t1pieces:
    (Fraction(1, 2), "ijef,mb,maef->ijab", ("t2", "t1", "ovvv")),
    (Fraction(-1, 2), "ijef,ma,mbef->ijab", ("t2", "t1", "ovvv")),
    # (B) t2(ijef) * 1/4 tau(mnab)t1t1 piece -> 1/8 t2(ijef)(t1(ma)t1(nb)-t1(mb)t1(na))oovv(mnef)
    (Fraction(1, 8), "ijef,ma,nb,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(-1, 8), "ijef,mb,na,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    # (C) tau-t1t1(ij) * vvvv: 1/2 (t1(ie)t1(jf)-t1(if)t1(je)) vvvv(abef)
    (Fraction(1, 2), "ie,jf,abef->ijab", ("t1", "t1", "vvvv")),
    (Fraction(-1, 2), "if,je,abef->ijab", ("t1", "t1", "vvvv")),
    # (D) tau-t1t1(ij) * Wabef-t1pieces (t1t1t1):
    (Fraction(1, 2), "ie,jf,mb,maef->ijab", ("t1", "t1", "t1", "ovvv")),
    (Fraction(-1, 2), "ie,jf,ma,mbef->ijab", ("t1", "t1", "t1", "ovvv")),
    (Fraction(-1, 2), "if,je,mb,maef->ijab", ("t1", "t1", "t1", "ovvv")),
    (Fraction(1, 2), "if,je,ma,mbef->ijab", ("t1", "t1", "t1", "ovvv")),
    # (E) tau-t1t1(ij) * 1/4 tau(mnab)-t1t1 -> t1^4:
    (Fraction(1, 8), "ie,jf,ma,nb,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    (Fraction(-1, 8), "ie,jf,mb,na,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    (Fraction(-1, 8), "if,je,ma,nb,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    (Fraction(1, 8), "if,je,mb,na,mnef->ijab", ("t1", "t1", "t1", "t1", "oovv")),
    # (F) tau-t1t1(ij) * 1/4 tau(mnab)-t2 cross term -> t1t1*t2:
    (Fraction(1, 8), "ie,jf,mnab,mnef->ijab", ("t1", "t1", "t2", "oovv")),
    (Fraction(-1, 8), "if,je,mnab,mnef->ijab", ("t1", "t1", "t2", "oovv")),

    # ---- P(ij)P(ab) t2 Wmbej -------------------------------------------------
    # Wmbej t1 pieces: +t1(jf)ovvv(mbef) + t1(nb)ooov(mnje) - t1(jf)t1(nb)oovv(mnef)
    # r2 signs from the four (ii,jj)x(aa,bb) permutations, each: +t2(aa e ii m) Wmbej(m bb e jj)
    #   term1: (i,j,a,b)+  (j,i,a,b)-  (i,j,b,a)-  (j,i,b,a)+
    # (a) + t1(jf) ovvv(mbef):
    (Fraction(1), "imae,jf,mbef->ijab", ("t2", "t1", "ovvv")),
    (Fraction(-1), "jmae,if,mbef->ijab", ("t2", "t1", "ovvv")),
    (Fraction(-1), "imbe,jf,maef->ijab", ("t2", "t1", "ovvv")),
    (Fraction(1), "jmbe,if,maef->ijab", ("t2", "t1", "ovvv")),
    # (b) + t1(nb) ooov(mnje):
    (Fraction(1), "imae,nb,mnje->ijab", ("t2", "t1", "ooov")),
    (Fraction(-1), "jmae,nb,mnie->ijab", ("t2", "t1", "ooov")),
    (Fraction(-1), "imbe,na,mnje->ijab", ("t2", "t1", "ooov")),
    (Fraction(1), "jmbe,na,mnie->ijab", ("t2", "t1", "ooov")),
    # (c) - t1(jf) t1(nb) oovv(mnef):
    (Fraction(-1), "imae,jf,nb,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(1), "jmae,if,nb,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(1), "imbe,jf,na,mnef->ijab", ("t2", "t1", "t1", "oovv")),
    (Fraction(-1), "jmbe,if,na,mnef->ijab", ("t2", "t1", "t1", "oovv")),

    # ---- P(ij)P(ab) t1 t1 ovov correction -----------------------------------
    (Fraction(1), "ie,ma,mbje->ijab", ("t1", "t1", "ovov")),
    (Fraction(-1), "ie,mb,maje->ijab", ("t1", "t1", "ovov")),
    (Fraction(-1), "je,ma,mbie->ijab", ("t1", "t1", "ovov")),
    (Fraction(1), "je,mb,maie->ijab", ("t1", "t1", "ovov")),

    # ---- P(ij) t1 ovvv singles ----------------------------------------------
    (Fraction(1), "ie,jeba->ijab", ("t1", "ovvv")),
    (Fraction(-1), "je,ieba->ijab", ("t1", "ovvv")),

    # ---- P(ab) t1 ooov singles ----------------------------------------------
    (Fraction(-1), "ma,ijmb->ijab", ("t1", "ooov")),
    (Fraction(1), "mb,ijma->ijab", ("t1", "ooov")),
]


def _operands(t1, t2, g, NO, N):
    occ, vir = slice(0, NO), slice(NO, N)

    def blk(sig):
        s = [occ if c == "o" else vir for c in sig]
        return g[s[0], s[1], s[2], s[3]]

    def get(key):
        if key == "t1":
            return t1
        if key == "t2":
            return t2
        return blk(key)

    return get


def evaluate_table(t1, t2, g, NO, N):
    """Sum every primitive term into r2[i,j,a,b]."""
    get = _operands(t1, t2, g, NO, N)
    r = np.zeros((NO, NO, NV_of(t1), NV_of(t1)))
    for coeff, subs, keys in PRIMITIVE_T1_TERMS:
        operands = [get(k) for k in keys]
        r += float(coeff) * np.einsum(subs, *operands)
    return r


def NV_of(t1):
    return t1.shape[1]


def verify(seed=11):
    """Table reproduces the reference's t1-part exactly."""
    from ccgen.tests.gccsd_reference import gccsd_doubles_residual

    NO, NV = 3, 4
    N = NO + NV
    rng = np.random.default_rng(seed)
    g = rng.random((N, N, N, N))
    g = g + g.transpose(2, 3, 0, 1)
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    f = rng.random((N, N))
    f = (f + f.T) / 2
    t1 = rng.random((NO, NV))
    t2 = rng.random((NO, NO, NV, NV))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    occ, vir = slice(0, NO), slice(NO, N)

    def blk(sig):
        s = [occ if c == "o" else vir for c in sig]
        return g[s[0], s[1], s[2], s[3]]

    blocks = {s: blk(s) for s in
              ["oooo", "ooov", "oovv", "ovov", "ovvo", "ovvv", "vvvv"]}
    fd = {"oo": f[occ, occ], "ov": f[occ, vir],
          "vo": f[vir, occ], "vv": f[vir, vir]}
    full = gccsd_doubles_residual(fd, blocks, t1, t2)
    zero = gccsd_doubles_residual(fd, blocks, np.zeros_like(t1), t2)
    t1part = full - zero

    table = evaluate_table(t1, t2, g, NO, N)
    return float(np.max(np.abs(table - t1part))), np.linalg.norm(t1part)


def test_w0_table_reproduces_reference_t1_part():
    diff, _ = verify()
    assert diff < 1e-10, diff


if __name__ == "__main__":
    diff, norm = verify()
    print(f"maxdiff = {diff:.3e}   (t1part norm = {norm:.3f})")
    assert diff < 1e-10, f"W0 table does not reproduce reference t1-part: {diff}"
    print("W0 OK: primitive weight table reproduces the reference t1-part.")
