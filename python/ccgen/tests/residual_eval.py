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
    v = v - v.transpose(1, 0, 2, 3)
    v = v - v.transpose(0, 1, 3, 2)

    f = rng.random((n, n))

    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "v": v, "f": f}


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
