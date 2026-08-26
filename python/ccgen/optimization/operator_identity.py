"""O1: transpose-equivalence of derived operators, decided SYMBOLICALLY.

Two derived operators can be the same contraction stored under different names,
differing only by a permutation of the operator's own slots. Merging such pairs
is how the D6 split (`docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`) recovers reuse
without giving up the correctness D6 bought.

This module decides that question on the SHAPE KEY, not by comparing arrays. A
numeric merge would be a coincidence on one random draw at one fixture; the
emitter needs a predicate it can evaluate on the symbolic form.

Three sources of freedom, all required (measured on GCC ccsd doubles, 23
same-family pairs, against a numeric oracle at two fixtures x three seeds):

- slot permutation                      -- the transpose itself
- summed-label permutation              -- `_contraction_shape` numbers internal
  indices per-shape, so the numbering does not align across two shapes
- the ERI's own SIGN-PRESERVING symmetries

That last restriction is load-bearing. `_ERI_PERMUTATIONS` is the symmetry
ORBIT of <pq||rs>, and four of its eight members are odd -- they hold only up to
a factor of -1. Folding all eight produced two FALSE MERGES on GCC (`W_t2v_vv`,
`W_t1t1v_oo`): the predicate claimed equivalence for arrays that differ in sign.
That is the same blind spot that let a 52% energy defect pass every symbolic
check (see `_ERI_PERMUTATIONS_SPATIAL`'s comment in `dressing.py`), reached
independently here.
"""

from __future__ import annotations

import itertools

def relabel(shape, perm, sperm=None):
    """Apply a slot permutation (and optionally a summed-label permutation).

    perm[k] = new label of old slot k. `sperm` does the same for `S<k>` labels:
    two shapes can be the same contraction while numbering their internal summed
    indices differently, and the canonical numbering in `_contraction_shape` is
    chosen per-shape, so it does not align across shapes. Entries are re-sorted
    (the key is order-invariant by construction)."""
    out = []
    for name, slots in shape:
        new_slots = []
        for x in slots:
            if x.startswith("S"):
                new_slots.append(f"S{sperm[int(x[1:])]}" if sperm else x)
            else:
                new_slots.append(str(perm[int(x)]))
        out.append((name, tuple(new_slots)))
    return tuple(sorted(out))


def tensor_symmetries(spatial: bool) -> dict:
    """Sign-preserving index permutations, per tensor name (O2.1).

    Only SIGN-PRESERVING relations belong here. A relation holding up to -1 is
    not usable for merging: two operators differing by a sign are not one
    operator, and binding them to a single array is wrong at every call site.
    Folding the odd ERI permutations produced exactly that -- two false merges
    on GCC -- which is the same blind spot that let a 52% dressed-energy defect
    pass every symbolic check (see `_ERI_PERMUTATIONS_SPATIAL` in `dressing.py`).

    `v`: the parity-+1 members of the ERI symmetry group, taken from
    `dressing.py` rather than re-derived so both sides share one contract.

    `t2` (O2.2): the amplitudes carry symmetry too, and `v`'s pattern does NOT
    transfer. Spatial `t2[abij] = t2[baji]` -- the SIMULTANEOUS pair swap only.
    The single-pair swaps `(1,0,2,3)` / `(0,1,3,2)` are not symmetries of
    spatial `t2` at all (they are antisymmetries of the spin-orbital one), so
    they are absent by intent, not by oversight.

    GCC `t2` is antisymmetric under each single swap and symmetric under their
    product; the product is the same `(1,0,3,2)`, so the entry is basis-
    independent even though the reasoning differs.
    """
    from ccgen.optimization.dressing import (
        _ERI_PERMUTATIONS, _ERI_PERMUTATIONS_SPATIAL, _perm_parity)

    v_perms = tuple(pm for pm in (
        _ERI_PERMUTATIONS_SPATIAL if spatial else _ERI_PERMUTATIONS)
        if _perm_parity(pm) == 1)
    return {
        "v": v_perms,
        "t2": ((0, 1, 2, 3), (1, 0, 3, 2)),
    }


def v_variants(shape, spatial):
    """Every rewriting of `shape` under its factors' own index symmetries.

    Named for `v` because that was the only entry when it was written; it now
    walks the whole `tensor_symmetries` table. A factor whose name is absent is
    treated as opaque (no rewriting), which is the safe default -- an unmodelled
    symmetry costs a missed merge, never a false one.
    """
    table = tensor_symmetries(spatial)
    idx = [(k, table[n]) for k, (n, sl) in enumerate(shape)
           if n in table and len(sl) == 4]
    if not idx:
        return {shape}
    out = set()
    for combo in itertools.product(*(perms for _, perms in idx)):
        ent = list(shape)
        for (k, _), pm in zip(idx, combo):
            n, sl = ent[k]
            ent[k] = (n, tuple(sl[i] for i in pm))
        out.add(tuple(sorted(ent)))
    return out


def symbolic_transpose(sp1, sp2, spatial: bool = False):
    """The slot permutation making `sp1` and `sp2` the same operator, or None.

    `perm[k] = j` means slot k of sp1 corresponds to slot j of sp2, i.e. one
    stored array serves both if the call site reads it transposed. Slot SPACES
    must correspond (an occ slot cannot become a vir slot), which is also what
    makes the transpose well-typed.

    Sound but INCOMPLETE on spatial input: it models `v`'s symmetries but not
    the amplitudes' (`t2(a,b,i,j) = t2(b,a,j,i)`), so it misses ~48 of 229
    same-family spatial pairs that a numeric oracle finds. Every disagreement is
    a MISS, never a false merge -- see the module docstring and O2.
    """
    from .factorize import _contraction_shape

    s1, s2 = (_contraction_shape(x.definition_terms[0]) for x in (sp1, sp2))
    n = len(sp1.indices)
    if n != len(sp2.indices):
        return None
    sp1_spaces = [i.space for i in sp1.indices]
    sp2_spaces = [i.space for i in sp2.indices]
    targets = v_variants(s2, spatial)
    nsum = 1 + max(
        [int(x[1:]) for _, sl in s1 for x in sl if x.startswith("S")] or [-1])
    for perm in itertools.permutations(range(n)):
        if any(sp1_spaces[k] != sp2_spaces[perm[k]] for k in range(n)):
            continue
        for sperm in itertools.permutations(range(nsum)):
            if relabel(s1, perm, sperm) in targets:
                return perm
    return None




def canonical_shape_of_term(node_term, spatial: bool = False):
    """`canonical_shape` for a bare node term (its free indices ARE its slots)."""
    from .factorize import _contraction_shape

    shape = _contraction_shape(node_term)
    spaces = [i.space for i in node_term.free_indices]
    return _canonicalize(shape, spaces, spatial)


def _canonicalize(shape, spaces, spatial):
    n = len(spaces)
    nsum = 1 + max(
        [int(x[1:]) for _, sl in shape for x in sl if x.startswith("S")] or [-1])
    best = None
    for perm in itertools.permutations(range(n)):
        if any(spaces[k] != spaces[perm[k]] for k in range(n)):
            continue
        for sperm in itertools.permutations(range(nsum)):
            for variant in v_variants(relabel(shape, perm, sperm), spatial):
                if best is None or variant < best:
                    best = variant
    return (tuple(spaces), best)


def canonical_shape(spec, spatial: bool = False):
    """O4: the orbit representative of `spec`'s contraction shape.

    Two operators are the same object iff their canonical shapes AND slot-space
    patterns agree — which is exactly `symbolic_transpose(a, b) is not None`,
    computed once per operator instead of once per pair. Keying the operator
    NAME on this makes transpose-equivalent contractions collapse at the point
    they are named, so nothing downstream has to know about merge classes.

    The slot-space pattern is part of the key, not an afterthought: two shapes
    can canonicalize identically while their slots differ in occ/vir, and a
    single array cannot serve both.
    """
    from .factorize import _contraction_shape

    shape = _contraction_shape(spec.definition_terms[0])
    return _canonicalize(shape, [i.space for i in spec.indices], spatial)


def merge_plan(specs, spatial: bool = False) -> dict:
    """O4.1: group `specs` into transpose-equivalence classes and, for each,
    record how every member maps onto its class representative.

    Returns ``{spec.name: (representative_name, permutation)}`` covering EVERY
    input spec — a representative maps to itself under the identity, so callers
    never special-case it. ``permutation[k] = j`` means slot k of this operator
    is slot j of the representative, i.e. a call site that would have read
    ``W_member(x0..xn)`` reads ``W_rep`` with its indices in that order.

    The representative is the lexicographically smallest name in the class, so
    the plan is deterministic across runs and independent of `specs` order.

    This is the whole of O4.1: computing the plan changes nothing on its own.
    Applying it takes two further steps that must NOT be combined — the call
    sites must permute (O4.2) BEFORE the names merge (O4.3). Doing both at once
    was tried and reverted: 11 GCC doubles terms stopped reproducing their
    source and the value gate could not say which half was at fault. See O4 in
    `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`.
    """
    from collections import defaultdict

    classes: dict = defaultdict(list)
    for spec in specs:
        classes[canonical_shape(spec, spatial)].append(spec)

    plan: dict = {}
    for members in classes.values():
        rep = min(members, key=lambda o: o.name)
        for m in members:
            if m.name == rep.name:
                plan[m.name] = (rep.name, tuple(range(len(m.indices))))
                continue
            perm = symbolic_transpose(m, rep, spatial=spatial)
            if perm is None:  # pragma: no cover - canonical_shape agreed, so
                raise AssertionError(          # the predicate must too
                    f"{m.name} shares a canonical shape with {rep.name} but no "
                    f"permutation maps between them")
            plan[m.name] = (rep.name, perm)
    return plan
