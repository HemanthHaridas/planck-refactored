"""Numerical residual evaluator for CC AlgebraTerms (D3.2 gate).

Test-side only: this evaluates a symbolic ``AlgebraTerm`` (or a bag of them) on
random amplitude/integral tensors and returns the residual array indexed by the
term's external indices.  It is the gate the diagram-path assembly must pass --
two term sets are equivalent iff their residual arrays agree, which is stronger
than matching diagram ids or norms (both of which passed wrong assemblies during
D3.2b).

Not imported by the generator; nothing here changes emitted code.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np


def random_tensors(no: int, nv: int, seed: int = 0):
    """A consistent set of antisymmetrized amplitudes and integrals.

    Returns a dict keyed by factor name.  ``t1..t3`` are amplitudes in
    (vir..., occ...) layout; ``v`` is the full antisymmetric ``<pq||rs>`` over
    the combined ``n = no + nv`` orbital space; ``f`` is a general Fock matrix.
    Amplitudes and ``v`` are antisymmetrized so the evaluator sees physical
    symmetry, which is what makes the residual comparison meaningful.
    """
    rng = np.random.default_rng(seed)
    n = no + nv

    t1 = rng.random((nv, no))

    t2 = rng.random((nv, nv, no, no))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)

    t3 = rng.random((nv, nv, nv, no, no, no))
    t3 = _antisymmetrize_block(t3, (0, 1, 2))
    t3 = _antisymmetrize_block(t3, (3, 4, 5))

    t4 = rng.random((nv, nv, nv, nv, no, no, no, no))
    t4 = _antisymmetrize_block(t4, (0, 1, 2, 3))
    t4 = _antisymmetrize_block(t4, (4, 5, 6, 7))

    v = rng.random((n, n, n, n))
    v = v + v.transpose(2, 3, 0, 1)
    v = v - v.transpose(1, 0, 2, 3)
    v = v - v.transpose(0, 1, 3, 2)
    # Re-impose: the two antisym projections above do not commute with the
    # bra<->ket one, so a single pass leaves a residual. Real <pq||rs> satisfies
    # all three (checked against pyscf); without this, any comparison of two
    # writings related by the exchange reports a spurious difference.
    v = v + v.transpose(2, 3, 0, 1)

    f = rng.random((n, n))
    f = f + f.T

    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "v": v, "f": f}


def ucc_random_tensors(noa: int, nva: int, nob: int, nvb: int, seed: int = 0):
    """F1 -- a spin-resolved tensor bundle for the UCC numeric gate.

    Sibling of :func:`random_tensors`, not a generalization of it: that one has
    seven consumers whose signature must not move, and UCC needs per-spin spaces
    of DIFFERENT sizes plus per-block ERIs.

    Keyed by the names the UCC bridge emits (`t1_aa`, `t2_abab`, `v_aaaa`, ...).
    Layout follows the bridge's own output -- amplitudes are ``(vir..., occ...)``
    and a block tag's FIRST half indexes the virtual slots, its SECOND half the
    occupied ones. So `t2_abab` is ``(nva, nvb, noa, nob)``.

    Symmetry is per block, and getting this wrong is the easiest way to build a
    fixture that silently disagrees with PySCF:

    * ``aaaa`` / ``bbbb`` are antisymmetric within the bra pair and within the
      ket pair independently -- both slots are the same spin space, so the swap
      is a real permutation of identical particles.
    * ``abab`` is **not** antisymmetrized across the halves: its two slots are
      different spin spaces. On a non-square case the transpose is not even
      shape-legal.
    * ERIs carry ``<pq||rs> = <rs||pq>`` in every block; the within-bra/ket
      antisymmetry only in the same-spin blocks.
    """
    rng = np.random.default_rng(seed)
    na, nb = noa + nva, nob + nvb

    def _same_spin_t2(nv, no):
        a = rng.random((nv, nv, no, no))
        a = a - a.transpose(1, 0, 2, 3)
        return a - a.transpose(0, 1, 3, 2)

    def _same_spin_v(n):
        v = rng.random((n, n, n, n))
        v = v + v.transpose(2, 3, 0, 1)
        v = v - v.transpose(1, 0, 2, 3)
        v = v - v.transpose(0, 1, 3, 2)
        # the antisym projections do not commute with the bra<->ket one, so a
        # single pass leaves a residual -- same re-impose random_tensors needs.
        return v + v.transpose(2, 3, 0, 1)

    def _mixed_v(n_bra, n_ket):
        # <a b || a b>: bra<->ket symmetric, no within-half antisymmetry (the two
        # slots of each half are different spin spaces).
        v = rng.random((n_bra, n_ket, n_bra, n_ket))
        return v + v.transpose(2, 3, 0, 1)

    def _sym_fock(n):
        f = rng.random((n, n))
        return f + f.T

    return {
        "t1_aa": rng.random((nva, noa)),
        "t1_bb": rng.random((nvb, nob)),
        "t2_aaaa": _same_spin_t2(nva, noa),
        "t2_abab": rng.random((nva, nvb, noa, nob)),
        "t2_bbbb": _same_spin_t2(nvb, nob),
        "v_aaaa": _same_spin_v(na),
        "v_bbbb": _same_spin_v(nb),
        "v_abab": _mixed_v(na, nb),
        "f_aa": _sym_fock(na),
        "f_bb": _sym_fock(nb),
    }


def ucc_closed_shell_tensors(no: int, nv: int, seed: int = 0):
    """F2.3 -- a closed-shell bundle where the UCC blocks and the RCC spatial
    tensors describe the SAME physics, so the two evaluators are comparable.

    Returns ``(ucc_blocks, spatial)``. This is a second fixture rather than a
    reuse of :func:`ucc_random_tensors`, and the reason is the whole point of the
    gate: that one draws every block INDEPENDENTLY, which violates the relations
    below by construction. Feed it to both sides and the comparison fails for a
    reason that has nothing to do with the evaluator.

    The relations are what ``collapse_amplitudes`` / ``collapse_integrals``
    invert -- a same-spin block is the spatial one antisymmetrized against its
    own exchange:

        t2_aaaa = t2 - t2.transpose(1,0,2,3)          (same for bbbb)
        v_aaaa  = v  - v.transpose(0,1,3,2)           (same for bbbb)
        t1_aa = t1_bb = t1        f_aa = f_bb = f     v_abab = v   t2_abab = t2

    The spatial tensors also carry the symmetries a real closed-shell reference
    has -- ``t2[abij] = t2[baji]`` and ``<pq|rs> = <qp|sr> = <rs|pq>``. Measured:
    the oracle holds to ~8e-13 **without** them, because the RCC/UCC identity is
    an algebraic property of the two term sets rather than of the tensors they
    contract. They are kept so the fixture describes a physically reachable
    reference -- do not read their presence as load-bearing for the comparison.

    The CLOSURE relations above are load-bearing, and are the part a mutation
    catches: flipping the sign in ``v_aaaa`` breaks every paired target.

    ``no != nv`` at every call site: a square case hides a transposed axis, the
    trap recorded in ``CCGEN_RANK3_KERNEL_AND_SOLVER.md``.
    """
    rng = np.random.default_rng(seed)
    n = no + nv

    t1 = rng.random((nv, no))

    t2 = rng.random((nv, nv, no, no))
    t2 = t2 + t2.transpose(1, 0, 3, 2)          # t2[abij] = t2[baji]

    v = rng.random((n, n, n, n))
    v = v + v.transpose(1, 0, 3, 2)             # <pq|rs> = <qp|sr>
    v = v + v.transpose(2, 3, 0, 1)             # bra <-> ket

    f = rng.random((n, n))
    f = f + f.T

    ucc = {
        "t1_aa": t1, "t1_bb": t1,
        "t2_abab": t2,
        "t2_aaaa": t2 - t2.transpose(1, 0, 2, 3),
        "t2_bbbb": t2 - t2.transpose(1, 0, 2, 3),
        "v_abab": v,
        "v_aaaa": v - v.transpose(0, 1, 3, 2),
        "v_bbbb": v - v.transpose(0, 1, 3, 2),
        "f_aa": f, "f_bb": f,
    }
    return ucc, {"t1": t1, "t2": t2, "v": v, "f": f}

def ucc_resolve_factor(factor, tensors, dims):
    """F2.1 -- resolve one UCC factor to the array slice it denotes.

    The RCC evaluator picks a factor's array by SPACE alone (`occ`/`vir` slices of
    one spin-free tensor). Under UCC the array also depends on SPIN, which after
    F2.0b is carried in the factor's own name: `v_abab`, `t2_aaaa`, `f_bb`.

    The tag is **positional** -- character k is slot k's spin, independent of that
    slot's space. Verified on the emitted vocabulary: `v_abab` occurs with 13
    different space patterns, all sharing the one tag.

    So the rule is: look the block up by name, then slice axis k by
    (space of slot k, spin of slot k). Amplitudes are stored per block with no
    combined space, so they need no slicing -- only the shape check that their
    tag and dims agree.

    ``dims`` is ``{"noa","nva","nob","nvb"}``. Raises on a block the fixture does
    not carry, rather than falling back to a spin-free array: a silent fallback
    is how a wrong-block read would survive.
    """
    name = factor.name
    if name not in tensors:
        raise KeyError(
            f"ucc_resolve_factor: no block {name!r} in the tensor bundle "
            f"(have {sorted(tensors)}). A UCC factor must name its spin block.")
    array = tensors[name]

    root, _, tag = name.partition("_")
    if not tag:
        raise ValueError(
            f"ucc_resolve_factor: factor {name!r} carries no spin block; the UCC "
            f"bridge must tag every factor (F2.0b).")
    if len(tag) != len(factor.indices):
        raise ValueError(
            f"ucc_resolve_factor: block tag {tag!r} has {len(tag)} spins but "
            f"{name!r} has {len(factor.indices)} slots.")

    if root.startswith("t"):
        # amplitudes are stored per block, already in (vir..., occ...) layout
        return array

    # v / f are stored over the combined per-spin orbital space, so each axis is
    # sliced by that slot's own (space, spin).
    bounds = {"a": (dims["noa"], dims["nva"]), "b": (dims["nob"], dims["nvb"])}
    sl = []
    for idx, spin in zip(factor.indices, tag):
        if spin not in bounds:
            raise ValueError(f"ucc_resolve_factor: bad spin {spin!r} in {name!r}")
        no_s, _nv_s = bounds[spin]
        sl.append(slice(0, no_s) if idx.space == "occ" else slice(no_s, None))
    return array[tuple(sl)]


def ucc_residual_einsum(term, dims, tensors):
    """F2.2b+c -- evaluate one UCC term to its residual array via one einsum.

    The UCC sibling of :func:`residual_einsum`; only the operand lookup differs,
    via :func:`ucc_resolve_factor`. Output layout is the same convention,
    ``R[vir_ext..., occ_ext...]``, so the two are directly comparable -- which is
    what the F2.3 closed-shell oracle rests on.

    Each axis is sized from its own index's spin, so ONE code path yields
    different shapes per block: `doubles_abab` is ``(nva, nvb, noa, nob)`` while
    `doubles_bbbb` is ``(nvb, nvb, nob, nob)``.

    ``dims`` is ``{"noa","nva","nob","nvb"}``; ``tensors`` is a block-keyed
    bundle (:func:`ucc_random_tensors`, or F2.3's closed-shell fixture).

    ``ucc_term_spins`` is called for its per-term consistency check only -- the
    operands do NOT come from it. F2.1 reads each factor's block off its own
    name, so an index's spin is never needed to slice; what the map catches is a
    term whose spin integration produced two spins for one index, which would
    make the term unevaluable block-wise regardless of the slicing.
    """
    import string

    from ..spin import ucc_term_spins

    ucc_term_spins(term)

    letters = {}
    pool = iter(string.ascii_lowercase + string.ascii_uppercase)
    for idx in list(term.free_indices) + list(term.summed_indices):
        letters[idx] = next(pool)

    subs, ops = [], []
    for f in term.factors:
        subs.append("".join(letters[i] for i in f.indices))
        ops.append(ucc_resolve_factor(f, tensors, dims))

    ext_vir = [i for i in term.free_indices if i.space == "vir"]
    ext_occ = [i for i in term.free_indices if i.space == "occ"]
    out = "".join(letters[i] for i in ext_vir + ext_occ)
    result = np.einsum(",".join(subs) + "->" + out, *ops, optimize=True)
    return result * float(term.coeff)


def residual_einsum(term, no: int, nv: int, tensors=None, seed: int = 0):
    """Evaluate one term to its residual array via a single ``np.einsum``.

    Equivalent to :func:`residual_of` for a single term, but orders of magnitude
    faster for high-rank terms (rank-4 doubles/quadruples): the pure-Python
    nested loop in ``residual_of`` is O(dims^externals * dims^summed), while
    einsum runs the optimized contraction. Output layout is
    ``R[vir_ext..., occ_ext...]`` (virtuals first, then occupieds, each in
    first-appearance order) -- identical to ``residual_of``.

    ``v`` / ``f`` are the full ``n``-space tensors, sliced to each factor's index
    spaces; ``t1..t4`` are block amplitudes used directly.
    """
    import string

    if tensors is None:
        tensors = random_tensors(no, nv, seed)
    n = no + nv
    occ, vir = slice(0, no), slice(no, n)

    letters = {}
    pool = iter(string.ascii_lowercase + string.ascii_uppercase)
    for idx in list(term.free_indices) + list(term.summed_indices):
        letters[idx] = next(pool)

    subs, ops = [], []
    for f in term.factors:
        subs.append("".join(letters[i] for i in f.indices))
        if f.name in ("v", "f"):
            sl = tuple(occ if i.space == "occ" else vir for i in f.indices)
            ops.append(tensors[f.name][sl])
        else:
            ops.append(tensors[f.name])

    ext_vir = [i for i in term.free_indices if i.space == "vir"]
    ext_occ = [i for i in term.free_indices if i.space == "occ"]
    out = "".join(letters[i] for i in ext_vir + ext_occ)
    result = np.einsum(",".join(subs) + "->" + out, *ops, optimize=True)
    return result * float(term.coeff)


def _signed_perms(k: int):
    import math

    base = list(range(k))
    for perm in itertools.permutations(base):
        # parity of the permutation
        sign = 1
        seen = [False] * k
        for start in range(k):
            if seen[start]:
                continue
            length = 0
            j = start
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                length += 1
            if length % 2 == 0:
                sign = -sign
        yield perm, sign


def _antisymmetrize_block(arr, axes):
    """Antisymmetrize ``arr`` over the given axes (a contiguous index block)."""
    out = np.zeros_like(arr)
    axes = list(axes)
    for perm, sign in _signed_perms(len(axes)):
        order = list(range(arr.ndim))
        for slot, a in enumerate(axes):
            order[a] = axes[perm[slot]]
        out = out + sign * arr.transpose(order)
    return out


def _slices(no: int, nv: int):
    occ = list(range(no))
    vir = list(range(no, no + nv))
    return occ, vir


def residual_of(terms, no: int, nv: int, tensors=None, seed: int = 0):
    """Evaluate one term or a list of terms to a residual array.

    The result is indexed ``R[vir_ext..., occ_ext...]`` in the term's external
    layout (virtuals first, then occupieds, each in first-appearance order).
    Summed indices are contracted.  A list sums the per-term residuals, matching
    how the emitter accumulates terms at runtime.
    """
    if not isinstance(terms, (list, tuple)):
        terms = [terms]
    if tensors is None:
        tensors = random_tensors(no, nv, seed)
    occ, vir = _slices(no, nv)

    def space(idx):
        return occ if idx.space == "occ" else vir

    # Output shape from the first term's externals (all terms in a diagram share
    # the same external structure).
    first = terms[0]
    ext_vir = [i for i in first.free_indices if i.space == "vir"]
    ext_occ = [i for i in first.free_indices if i.space == "occ"]
    shape = tuple([nv] * len(ext_vir) + [no] * len(ext_occ))
    R = np.zeros(shape)

    for term in terms:
        free = list(term.free_indices)
        summed = list(term.summed_indices)
        tv = [i for i in free if i.space == "vir"]
        to = [i for i in free if i.space == "occ"]
        for fvals in itertools.product(*[range(len(space(i))) for i in free]):
            env = {i: space(i)[fvals[k]] for k, i in enumerate(free)}
            acc = 0.0
            for svals in itertools.product(*[space(i) for i in summed]):
                for k, i in enumerate(summed):
                    env[i] = svals[k]
                p = 1.0
                for fac in term.factors:
                    A = tensors[fac.name]
                    key = tuple(
                        env[i] - no if (fac.name.startswith("t") and i.space == "vir")
                        else env[i]
                        for i in fac.indices
                    )
                    p *= A[key]
                acc += p
            out_index = tuple([env[i] - no for i in tv] + [env[i] for i in to])
            R[out_index] += float(term.coeff) * acc
    return R
