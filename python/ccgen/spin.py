"""Spin-adaptation layer (S0): the index model + single-term spin labeling.

ccgen derives every equation in spin-orbital (GCC) form -- `indices.Index` has a
space (occ/vir/gen) but NO spin. Mapping those equations to restricted (RCC) or
unrestricted (UCC) spatial-orbital form is *spin integration*: each spin-orbital
index p is a (spatial index, spin sigma in {a,b}) pair, and the spin-orbital
equation is summed over spin.

DESIGN DECISION (S0): the spin layer is kept ISOLATED from the GCC path. It does
NOT add a spin field to `Index` (which is baked into every canonicalize / wick /
diagram hash and equality -- perturbing it would risk the validated GCC path).
Instead it wraps a spatial `Index` in a lightweight `SpinIndex` and operates on
`AlgebraTerm`s produced by generation. Generation, canonicalization, and the
diagram engine are untouched.

S0 delivers only the model + the single-term spin *labeling* (assign a spin to
every index consistently along shared lines, enumerate the summed-index spin
cases). The coefficient algebra -- UCC block coefficients (S1) and the RCC
alpha=beta collapse (S2) -- is later work. So the S0 gate is STRUCTURAL: labels
are consistent along shared names and the summed indices enumerate the right
spin cases; it does not yet assert collapsed coefficients.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .indices import Index

SPINS = ("a", "b")  # alpha, beta


@dataclass(frozen=True)
class SpinIndex:
    """A spatial index plus a spin label. Wraps a GCC `Index` without touching
    it -- the spatial identity (`base`) is the original occ/vir index; `spin` is
    "a" or "b". Two spin-orbital lines are the same iff same base name AND same
    spin."""

    base: Index
    spin: str

    def __post_init__(self) -> None:
        if self.spin not in SPINS:
            raise ValueError(f"spin must be one of {SPINS}, got {self.spin!r}")

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def space(self) -> str:
        return self.base.space

    def __repr__(self) -> str:
        return f"{self.base.name}{self.spin}"


def _distinct_names(term) -> tuple[list[Index], list[Index]]:
    """(free, summed) index objects of a term, de-duplicated by name -- a
    repeated NAME is one physical line and gets one spin."""
    seen: dict[str, Index] = {}
    free: list[Index] = []
    summed: list[Index] = []
    free_names = {i.name for i in term.free_indices}
    for fac in term.factors:
        for idx in fac.indices:
            if idx.name in seen:
                continue
            seen[idx.name] = idx
            (free if idx.name in free_names else summed).append(idx)
    return free, summed


def spin_label_cases(term, external_spins: dict[str, str]):
    """Enumerate the spin-labeled forms of one GCC term for a fixed choice of
    EXTERNAL (free-index) spins (S0).

    Every distinct index NAME in the term is one spin-orbital line and carries a
    single spin. Free names take their spin from `external_spins`; each summed
    name is enumerated over both spins (the spin sum). Returns a list of
    ``{index_name: SpinIndex}`` maps, one per summed-spin assignment -- i.e.
    ``2 ** (number of distinct summed names)`` cases.

    This is labeling only: it does NOT yet produce spin-blocked tensors or
    integrate coefficients (S1/S2). It is the structural primitive both build on.
    """
    free, summed = _distinct_names(term)
    missing = [i.name for i in free if i.name not in external_spins]
    if missing:
        raise ValueError(f"external_spins missing free indices: {missing}")

    base_by_name = {i.name: i for i in free + summed}
    summed_names = [i.name for i in summed]

    cases = []
    for combo in itertools.product(SPINS, repeat=len(summed_names)):
        assignment = dict(zip(summed_names, combo))
        label = {}
        for name, base in base_by_name.items():
            spin = external_spins[name] if name in external_spins else assignment[name]
            label[name] = SpinIndex(base, spin)
        cases.append(label)
    return cases


# ── S1.0 the UCC block model: spin conservation per line ──────────────
#
# A spin-orbital tensor block is nonzero iff spin is CONSERVED along every
# excitation / interaction LINE. ccgen orders each tensor's indices as n virtual
# slots followed by n occupied slots (t1: [v,o]; t2: [v,v,o,o]; t3: [v,v,v,o,o,o];
# f: [p,q]; v = <pq||rs>: [v,v,o,o] but the physicist antisym lines pair p-r, q-s).
# So for a rank-2n tensor the lines pair slot k with slot k+n, and the block
# exists iff spin(slot k) == spin(slot k+n) for every k in 0..n-1.
#
# This one rule covers every tensor kind ccgen emits -- there is no per-tensor
# lookup table. It is the spin-conservation content of the UCC block structure
# (t1 -> a,b; t2 -> aa,bb,ab; f -> a,b; v -> aaaa,bbbb,abab,baba), derived from
# physics, matching PySCF's case-tagged UCC blocks.


def _line_pairs(factor) -> list[tuple[int, int]]:
    """Slot pairs (k, k+n) for a rank-2n tensor -- the excitation/interaction
    lines along which spin is conserved. Raises for odd rank (no line pairing)."""
    r = len(factor.indices)
    if r % 2 != 0:
        raise ValueError(f"{factor.name}: odd rank {r} has no line pairing")
    n = r // 2
    return [(k, k + n) for k in range(n)]


def block_exists(factor, label: dict) -> bool:
    """Whether the spin-labeled *factor* is a nonzero UCC block: spin is
    conserved along every line (slot k and slot k+n carry the same spin).
    ``label`` maps index name -> SpinIndex (from :func:`spin_label_cases`)."""
    spins = [label[i.name].spin for i in factor.indices]
    return all(spins[a] == spins[b] for a, b in _line_pairs(factor))


def resolve_block(factor, label: dict) -> tuple[str, bool]:
    """Resolve the spin-labeled *factor* to a (block_tag, exists) pair (S1.1).

    ``block_tag`` is the per-slot spin string in the factor's own index order
    (e.g. ``t2`` slots [v,v,o,o] with spins a,b,a,b -> "abab"); ``exists`` is
    :func:`block_exists`. A forbidden block still gets a tag (for diagnostics)
    but ``exists=False`` and S1.2 will drop it. Coefficient integration and the
    physicist->chemist naming are S1.2/S1.4, not here."""
    tag = "".join(label[i.name].spin for i in factor.indices)
    return tag, block_exists(factor, label)


# ── S1.2 single-term UCC integration ──────────────────────────────────
#
# For a fixed EXTERNAL block (a spin assignment to the free indices), a GCC term
# contributes the SUM over its summed-index spin cases (S0) of the cases where
# EVERY factor is a valid block (S1.1). Each surviving case is one spatial term:
# same GCC coefficient, factors tagged with their spin block, spatial indices.
#
# The coefficient here is the RAW GCC coefficient. Whether the spatial block
# combinatorics multiply it by a factor (e.g. the GCC 1/2 on a summed
# spin-orbital antisym pair becoming 1 when the surviving spatial case has
# distinct-spin summed indices) is deliberately NOT decided here -- it is the
# thing the numeric gates settle: the spin-orbital identity (S1.2 test) and PySCF
# `uccsd.update_amps` (S1.3). S1.2 emits the surviving cases faithfully; the gates
# tell us if the block model already makes the coefficients come out right.


@dataclass(frozen=True)
class SpinFactor:
    """One factor of a spin-integrated (spatial) term: the tensor name, its spin
    block tag (per-slot spins in the factor's index order), and the SpinIndex per
    slot (spatial index + spin)."""

    name: str
    block: str
    indices: tuple


@dataclass(frozen=True)
class SpinTerm:
    """A single spin-integrated spatial term: coefficient + tagged factors, for a
    named external block (the spin assignment of the free indices)."""

    coeff: object
    external_block: tuple
    factors: tuple


def ucc_integrate_term(term, external_spins: dict):
    """Integrate one GCC ``AlgebraTerm`` into the UCC ``SpinTerm``s that
    contribute to the given external block (S1.2).

    Enumerates the summed-spin cases (:func:`spin_label_cases`), keeps only cases
    where every factor is a valid block (:func:`block_exists`), and emits one
    ``SpinTerm`` per surviving case carrying the GCC coefficient. The external
    block is the sorted (free-name, spin) tuple; each factor is tagged with its
    resolved block."""
    ext_block = tuple(sorted(external_spins.items()))
    out = []
    for label in spin_label_cases(term, external_spins):
        if not all(block_exists(f, label) for f in term.factors):
            continue
        factors = tuple(
            SpinFactor(
                name=f.name,
                block=resolve_block(f, label)[0],
                indices=tuple(label[i.name] for i in f.indices),
            )
            for f in term.factors
        )
        out.append(SpinTerm(coeff=term.coeff, external_block=ext_block, factors=factors))
    return out


# ── S1.2' antisymmetry-correct integration (real integrals) ───────────
#
# WHY THIS EXISTS. `ucc_integrate_term` above DROPS every summed-spin case in
# which any factor lands in a spin-forbidden block (`block_exists` is False).
# That is correct ONLY when the tensors are spin-conserving-per-line, i.e. their
# forbidden blocks are exactly zero. Real CC tensors are ANTISYMMETRIC: their
# "forbidden" (spin-broken-line) blocks are NONZERO -- they carry the exchange.
# On real integrals the drop silently discards that exchange, so the plain
# `ucc_integrate_term` fails the S2.1 identity (~0.06 on water/STO-3G) and the
# whole S2 collapse validated only on the synthetic spin-conserving fixture.
#
# THE FIX. A forbidden block of an antisymmetric rank-4 tensor is not zero -- it
# equals an ALLOWED block reached by swapping the bra pair (slots 0,1) and/or the
# ket pair (slots 2,3), each swap contributing a sign -1. So instead of dropping
# the case, re-express each factor into its allowed block with that sign. The -1
# signs ARE the exchange (-K); with the S2 collapse + merge they become the RCC
# `2J - K` combinations. Verified: `ucc_integrate_term_antisym` summed over the
# whole CCD/CCSD manifold reproduces the GCC abab/aa residual on REAL
# antisymmetric water/STO-3G integrals to ~2e-17 (singles + doubles).


def _antisym_to_allowed(factor, label):
    """Map a spin-labeled factor to its allowed (spin-conserving-per-line) block
    via antisymmetry, returning ``(sign, indices)`` or ``None`` if genuinely zero.

    rank-2 (f, t1): one line (slots 0-1); allowed iff the two spins match, else
    zero (no swap can fix a single line). rank-4 (t2, v): lines 0-2 and 1-3;
    antisymmetric in the bra pair (0,1) and ket pair (2,3), so try the identity,
    the bra swap (sign -1), the ket swap (sign -1), and both (sign +1), returning
    the first that conserves spin on both lines."""
    idx = [label[i.name] for i in factor.indices]
    if len(idx) == 2:
        return (1, idx) if idx[0].spin == idx[1].spin else None

    def conserves(ix):
        return ix[0].spin == ix[2].spin and ix[1].spin == ix[3].spin

    candidates = [
        (1, idx),
        (-1, [idx[1], idx[0], idx[2], idx[3]]),   # bra swap
        (-1, [idx[0], idx[1], idx[3], idx[2]]),   # ket swap
        (1, [idx[1], idx[0], idx[3], idx[2]]),    # both
    ]
    for sign, cix in candidates:
        if conserves(cix):
            return (sign, cix)
    return None


def ucc_integrate_term_antisym(term, external_spins: dict):
    """Integrate one GCC ``AlgebraTerm`` for REAL antisymmetric tensors (S1.2').

    Like :func:`ucc_integrate_term`, but instead of dropping a summed-spin case
    whose factor is in a forbidden block, it re-expresses that factor into its
    allowed block via antisymmetry (:func:`_antisym_to_allowed`), folding the swap
    sign into the coefficient. A case is dropped only when a factor is GENUINELY
    zero (a rank-2 line with mismatched spins). This is the integration that
    matches the GCC residual on real integrals; the plain filtered form is exact
    only for spin-conserving-per-line tensors."""
    ext_block = tuple(sorted(external_spins.items()))
    out = []
    for label in spin_label_cases(term, external_spins):
        factors = []
        sign = 1
        alive = True
        for f in term.factors:
            res = _antisym_to_allowed(f, label)
            if res is None:
                alive = False
                break
            s, cix = res
            sign *= s
            factors.append(SpinFactor(
                name=f.name,
                block="".join(x.spin for x in cix),
                indices=tuple(cix),
            ))
        if alive:
            out.append(SpinTerm(coeff=term.coeff * sign,
                                external_block=ext_block,
                                factors=tuple(factors)))
    return out


# ── S1.3a full-manifold UCC aggregation ───────────────────────────────
#
# Enumerate the residual's own valid external spin blocks (same conservation rule
# as any tensor), then integrate every GCC term of the manifold into each block.
# Gated by the full-manifold spin-orbital identity (the S1.2 oracle over the whole
# residual) -- no PySCF, no chemist convention. The PySCF cross-check (which also
# exercises the physicist->chemist ERI mapping) is a separate later step.


def external_blocks(residual_template) -> list[dict]:
    """The canonical UCC external spin blocks for a residual, as
    {free-index-name: spin} dicts. Enumerates the spin patterns that conserve
    spin along the residual's lines (:func:`block_exists`), then keeps one
    representative per block up to a global a<->b flip (so doubles -> aaaa, abab,
    bbbb, not also baba)."""
    n = len(residual_template.indices) // 2
    names = [i.name for i in residual_template.indices]
    seen = set()
    blocks = []
    for combo in itertools.product(SPINS, repeat=len(names)):
        label = {nm: SpinIndex(idx, s)
                 for nm, idx, s in zip(names, residual_template.indices, combo)}
        if not block_exists(residual_template, label):
            continue
        flip = tuple("b" if s == "a" else "a" for s in combo)
        key = min(combo, flip)  # canonical under global a<->b
        if key in seen:
            continue
        seen.add(key)
        blocks.append(dict(zip(names, combo)))
    return blocks


def ucc_manifold(terms, residual_template):
    """Integrate a whole GCC residual manifold into UCC blocks (S1.3a).

    Returns ``{external_block_tag: [SpinTerm]}`` where ``external_block_tag`` is
    the per-slot spin string of the residual (e.g. doubles -> "aaaa", "abab",
    "bbbb"), and the value is every surviving integrated term of every GCC term
    for that block. Carries raw GCC coefficients (S1.2); the coefficient
    correctness is what the identity / PySCF gates check."""
    names = [i.name for i in residual_template.indices]
    out: dict = {}
    for block in external_blocks(residual_template):
        tag = "".join(block[nm] for nm in names)
        acc: list = []
        for term in terms:
            acc.extend(ucc_integrate_term(term, block))
        out[tag] = acc
    return out


# ── S2.0 closed-shell block relations (alpha == beta) ─────────────────
#
# For a closed-shell RHF reference the alpha and beta spatial orbitals are
# identical, so the UCC blocks are not independent:
#
#   t1a == t1b == t1                                         (singles collapse)
#   t2aa[a,b,i,j] == t2ab[a,b,i,j] - t2ab[b,a,i,j]           (S2.0 relation)
#
# The same-spin doubles block is fully antisymmetric under swapping its two
# virtuals (or its two occupieds); the mixed block is not. The relation says the
# same-spin block is the antisymmetrized part of the mixed block. Only ONE of the
# two swaps is written here -- occupied vs virtual swap give the SAME result
# because the mixed block itself satisfies t2ab[a,b,i,j] == t2ab[b,a,j,i]; the
# S2.0 gate pins that this specific (virtual-swap, minus-sign) convention
# reproduces the sliced same-spin block numerically.


def t2aa_from_t2ab(t2ab):
    """Reconstruct the same-spin doubles block from the mixed (abab) block under
    the closed-shell alpha==beta relation (S2.0): antisymmetrize the two virtual
    slots, ``t2aa = t2ab - t2ab.transpose(virtual swap)``. Input/output are
    spatial ``[v, v, o, o]`` arrays."""
    import numpy as np

    a = np.asarray(t2ab)
    return a - a.transpose(1, 0, 2, 3)


# ── S2.2a canonicalize spin blocks to the abab representative ──────────
#
# The UCC abab residual (ucc_manifold["abab"]) carries factors in four spin
# blocks: aaaa, bbbb, abab, baba. For a closed-shell (alpha==beta) tensor a
# GLOBAL spin flip a<->b is an exact symmetry: the value at a spin-labeled slot
# tuple equals the value at the fully-flipped tuple, with the SAME spatial
# indices. So each factor collapses to its global-flip canonical block by simply
# flipping every slot's spin when that yields the smaller tag -- no index
# permutation, no coefficient change. This maps baba->abab and bbbb->aaaa,
# leaving only {aaaa, abab}. It is the pure-relabel first step of the S2.2
# collapse; S2.2b then removes the same-spin (aaaa) block via t2aa = t2ab-P.


def _flip_spins(tag: str) -> str:
    return "".join("b" if s == "a" else "a" for s in tag)


def _canonical_block(tag: str) -> tuple[str, bool]:
    """The global-a<->b-flip canonical form of a spin block tag and whether a
    flip was applied. Canonical = the lexicographically smaller of (tag, flip)."""
    flip = _flip_spins(tag)
    return (tag, False) if tag <= flip else (flip, True)


def canonicalize_spin_blocks(spinterm: SpinTerm) -> SpinTerm:
    """Rewrite a SpinTerm so every factor uses the global-flip canonical spin
    block (S2.2a): flip a<->b on any factor whose block is not canonical, keeping
    its spatial indices and the term coefficient unchanged. Under the closed-shell
    alpha==beta symmetry the flipped factor is the identical spatial quantity, so
    this is a pure relabel (baba->abab, bbbb->aaaa). Returns a new SpinTerm; the
    external block is left as-is (it is already the abab representative by
    construction of ucc_manifold)."""
    new_factors = []
    for f in spinterm.factors:
        canon, flipped = _canonical_block(f.block)
        if not flipped:
            new_factors.append(f)
            continue
        idx = tuple(SpinIndex(si.base, "b" if si.spin == "a" else "a")
                    for si in f.indices)
        new_factors.append(SpinFactor(name=f.name, block=canon, indices=idx))
    return SpinTerm(coeff=spinterm.coeff,
                    external_block=spinterm.external_block,
                    factors=tuple(new_factors))


# ── S2.2b amplitude collapse: t2aa -> t2ab - P(t2ab), t1 spins drop ────
#
# After S2.2a the canonical doubles blocks are {aaaa, abab}. The same-spin
# AMPLITUDE block t2[aaaa] is not independent RCC data -- S2.0 gives, in [v,v,o,o]
# layout,
#
#   t2aa(A,B,I,J) = t2ab(A,B,I,J) - t2ab(B,A,I,J)     (swap the two virtual slots)
#
# so a t2[aaaa] factor expands into a TWO-TERM sum of abab factors, splitting its
# host SpinTerm in two (coefficient -1 on the swapped one). Multiple same-spin t2
# factors in one term take the Cartesian product. This is the first step where
# coefficients change (one term -> two). t1[aa]/f[aa] are already the single
# spatial block (S2.2a removed their bb partner), so they are left as-is; the
# INTEGRAL same-spin block v[aaaa] is deferred to S2.2c (the 2J-K step).


def _split_t2aaaa(factor: SpinFactor) -> list[tuple[object, SpinFactor]]:
    """Expand a same-spin t2[aaaa] factor into the two abab factors of the S2.0
    relation, each as a (sign, SpinFactor) pair. Slots are [v,v,o,o]; the second
    term swaps the two virtual slots and carries -1. The emitted factors are abab
    (per-slot spins a,b,a,b) over the (unswapped / virtual-swapped) spatial
    indices."""
    A, B, I, J = factor.indices  # SpinIndex, all spin 'a'
    abab_spins = ("a", "b", "a", "b")

    def mk(order):
        idx = tuple(SpinIndex(si.base, s) for si, s in zip(order, abab_spins))
        return SpinFactor(name="t2", block="abab", indices=idx)

    return [(1, mk((A, B, I, J))), (-1, mk((B, A, I, J)))]


def collapse_amplitudes(spinterm: SpinTerm) -> list[SpinTerm]:
    """Apply the S2.0 amplitude collapse to a canonical (post-S2.2a) SpinTerm
    (S2.2b): replace every same-spin t2[aaaa] factor by the two-term
    ``t2ab - t2ab(virtual swap)`` combination, returning the resulting list of
    SpinTerms (Cartesian product over multiple such factors, signs folded into the
    coefficient). Factors that are not t2[aaaa] pass through unchanged -- t1[aa]
    and f[aa] are already the single spatial block; v[aaaa] is left for S2.2c.
    Input must already be canonicalized (:func:`canonicalize_spin_blocks`)."""
    # choices[k] is the list of (sign, factor) alternatives for factor k
    choices = []
    for f in spinterm.factors:
        if f.name == "t2" and f.block == "aaaa":
            choices.append(_split_t2aaaa(f))
        else:
            choices.append([(1, f)])

    return _product_over_choices(spinterm, choices)


# ── S2.2c integral collapse: v[aaaa] -> v[abab] - P(v[abab]) ───────────
#
# After S2.2b the only same-spin block left is the INTEGRAL v[aaaa]. Its
# closed-shell relation is the KET-slot swap (vs the virtual/bra swap for t2):
#
#   v[aaaa](p,q,r,s) = v[abab](p,q,r,s) - v[abab](p,q,s,r)
#
# i.e. antisymmetrize the two ket slots (slots 2,3). So a v[aaaa] factor splits
# into a two-term sum of abab factors, exactly like t2[aaaa] does -- same shape,
# ket swap instead of bra swap. After this every doubles factor is a single
# spatial block (t2/v abab; t1/f aa).
#
# CAVEAT on validation: unlike the t2 relation, this v relation CANNOT be gated
# by a numeric no-op on the synthetic spin-conserving-per-line fixture. ccgen's v
# conserves spin along each line, so the ket-swapped abab entry is spin-forbidden
# (zero) there and v[aaaa] would reduce to just v[abab] -- the exchange term the
# relation needs lives in separate ccgen terms, not folded into v. The numeric
# gate for this collapse is the chemist 2J-K form at S2.2d against real
# integrals; S2.2c is gated STRUCTURALLY (S22cIntegralCollapseStructureTests) +
# proven correct by the S2.2d end-to-end residual. See CCGEN_SPIN_ADAPTATION_SCOPE.md.


def _split_vaaaa(factor: SpinFactor) -> list[tuple[object, SpinFactor]]:
    """Expand a same-spin v[aaaa] factor into the two abab factors of the
    closed-shell integral relation, each a (sign, SpinFactor) pair. The second
    term swaps the two KET slots (2,3) and carries -1 -- the ket-swap analog of
    :func:`_split_t2aaaa`'s virtual (bra) swap."""
    P, Q, R, S = factor.indices  # SpinIndex, all spin 'a'
    abab_spins = ("a", "b", "a", "b")

    def mk(order):
        idx = tuple(SpinIndex(si.base, s) for si, s in zip(order, abab_spins))
        return SpinFactor(name="v", block="abab", indices=idx)

    return [(1, mk((P, Q, R, S))), (-1, mk((P, Q, S, R)))]


def collapse_integrals(spinterm: SpinTerm) -> list[SpinTerm]:
    """Apply the closed-shell integral collapse to a SpinTerm whose amplitudes are
    already collapsed (post-S2.2b): replace every same-spin v[aaaa] factor by the
    two-term ``v[abab] - v[abab](ket swap)`` combination (S2.2c). Cartesian product
    over multiple such factors, signs folded into the coefficient; all other
    factors pass through. After this the doubles residual is in a single spatial
    block per tensor."""
    choices = []
    for f in spinterm.factors:
        if f.name == "v" and f.block == "aaaa":
            choices.append(_split_vaaaa(f))
        else:
            choices.append([(1, f)])
    return _product_over_choices(spinterm, choices)


def _product_over_choices(spinterm: SpinTerm, choices) -> list[SpinTerm]:
    """Cartesian-product a per-factor list of (sign, SpinFactor) alternatives into
    SpinTerms, folding signs into the coefficient. Shared by the S2.2b amplitude
    and S2.2c integral collapses."""
    out = []
    for combo in itertools.product(*choices):
        sign = 1
        factors = []
        for s, fac in combo:
            sign *= s
            factors.append(fac)
        out.append(SpinTerm(
            coeff=spinterm.coeff * sign,
            external_block=spinterm.external_block,
            factors=tuple(factors),
        ))
    return out


# ── S2.2d-1 merge like terms ──────────────────────────────────────────
#
# After the S2.2a->b->c collapse the residual is a flat list of single-block
# spatial SpinTerms, with structural duplicates: terms that are the SAME
# contraction up to (i) the order of their factors and (ii) a relabeling of the
# summed (non-external) indices. Merging sums their coefficients into one term.
# This is where the characteristic RCC spatial coefficients (2J - K: the -2, 4,
# 1/4 ... combinations) appear -- they are the sum of the exchange/Coulomb pair
# coefficients the S2.2c v-split produced.
#
# The merge is a pure algebraic identity (independent of any tensor values), so
# it is value-preserving on ANY fixture -- unlike the integral collapse, it does
# not depend on v satisfying a block relation.


def _merge_signature(term: SpinTerm, externals: frozenset) -> tuple:
    """A canonical key identifying a SpinTerm up to factor order and summed-index
    relabeling. External (free) index names are kept verbatim; each summed name is
    replaced by a positional placeholder, minimized over all summed-name
    permutations and factor orderings (so `t2(c,d,..)v(c,d,..)` and
    `t2(d,c,..)v(d,c,..)` share a key). Two terms with the same key are the same
    spatial contraction and merge by coefficient sum."""
    summed = sorted({si.name for f in term.factors for si in f.indices}
                    - externals)
    best = None
    for perm in itertools.permutations(summed):
        relabel = {old: f"S{k}" for k, old in enumerate(perm)}
        keys = tuple(sorted(
            (f.name, f.block,
             tuple(relabel.get(si.name, si.name) for si in f.indices))
            for f in term.factors
        ))
        if best is None or keys < best:
            best = keys
    return best


def merge_terms(terms, externals) -> list[SpinTerm]:
    """Merge structurally-identical spatial SpinTerms (S2.2d-1): group by
    :func:`_merge_signature` (factor-order- and summed-relabel-invariant), sum the
    group coefficients, keep one representative per surviving (nonzero) group, and
    drop groups whose coefficients cancel to zero. ``externals`` is the set of
    free-index names (e.g. ``{"a","b","i","j"}`` for a doubles residual). Pure
    algebra -- value-preserving on any tensors."""
    externals = frozenset(externals)
    groups: dict = {}
    order: list = []
    for term in terms:
        sig = _merge_signature(term, externals)
        if sig not in groups:
            groups[sig] = [term.coeff, term]
            order.append(sig)
        else:
            groups[sig][0] = groups[sig][0] + term.coeff
    out = []
    for sig in order:
        total, rep = groups[sig]
        if total != 0:
            out.append(SpinTerm(coeff=total,
                                external_block=rep.external_block,
                                factors=rep.factors))
    return out


# ── S3.0 bridge: spatial SpinTerm -> AlgebraTerm ──────────────────────
#
# The S2 pipeline produces merged spatial RCC `SpinTerm`s (single spatial block
# per factor, with the 2J-K coefficients). The emit path
# (`emit/planck_tensor_cpp` via `lowering/restricted_closed_shell`) consumes
# `AlgebraTerm`s. This bridge drops the spin labels -- each `SpinIndex.base` is
# already the spatial occ/vir `Index` -- and rebuilds an `AlgebraTerm` with the
# same coefficient, factor tensors, and free/summed split. It is a pure
# structural transform: the spatial algebra (which factor contracts which index)
# is unchanged, only the wrapper type differs. Gated by evaluation equivalence
# (the converted term contracts to the same residual as the SpinTerm).


def spinterm_to_algebraterm(spinterm: SpinTerm, externals):
    """Convert a spatial ``SpinTerm`` to an ``AlgebraTerm`` for the emit path
    (S3.0). ``externals`` is the set of free-index names (e.g.
    ``{"a","b","i","j"}``). Each factor becomes a ``Tensor`` over the spatial base
    indices; free and summed indices are split by name (de-duplicated,
    first-appearance order). The coefficient is carried as a ``Fraction``."""
    from fractions import Fraction

    from .project import AlgebraTerm
    from .tensors import Tensor

    externals = set(externals)
    factors = tuple(
        Tensor(f.name, tuple(si.base for si in f.indices))
        for f in spinterm.factors
    )
    free: list = []
    summed: list = []
    seen_free: set = set()
    seen_summed: set = set()
    for f in spinterm.factors:
        for si in f.indices:
            if si.name in externals:
                if si.name not in seen_free:
                    seen_free.add(si.name)
                    free.append(si.base)
            elif si.name not in seen_summed:
                seen_summed.add(si.name)
                summed.append(si.base)
    return AlgebraTerm(Fraction(spinterm.coeff), factors,
                       tuple(free), tuple(summed), True)
