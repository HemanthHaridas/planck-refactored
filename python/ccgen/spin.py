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


def _permutation_parity(order) -> int:
    """Sign (+1/-1) of the permutation given by ``order`` (a list whose k-th
    entry is the source position placed at position k)."""
    seen = [False] * len(order)
    parity = 1
    for start in range(len(order)):
        if seen[start]:
            continue
        j = start
        length = 0
        while not seen[j]:
            seen[j] = True
            j = order[j]
            length += 1
        if length % 2 == 0:
            parity = -parity
    return parity


def _orientation_normalized(factor):
    """Reorient a rank-4 ``v`` factor to one canonical arrangement of its 8-fold ERI
    orbit, returning ``(factor, sign)`` (V1.1e.2).

    Non-``v`` and non-rank-4 factors pass through with sign ``+1``.

    WHY THIS EXISTS. :func:`_line_pairs` reads the interaction lines off SLOT POSITION
    (slot k pairs with slot k+n, the physicist ``<pq|rs>`` convention), so two writings
    of the SAME integral that differ by a bra<->ket exchange composed with a
    within-group swap present different line structures to the adapter and integrate
    differently. Measured on the dressed-CCSD doubles manifold:

        v(k,b,c,j) t2(a,c,i,k)  and  -1 v(j,c,k,b) t2(a,c,i,k)

    are one term (equal under ``_eri_canonical``: same key, same folded coefficient),
    yet integrated to 2 and 0. Normalizing here makes the adapter a function of the
    algebra rather than of how it was written.

    Note the bra<->ket exchange ALONE is harmless -- it maps lines ``p-r, q-s`` to
    ``r-p, s-q``, the same lines (verified over all 256 space x spin cases). It is the
    composition with a within-group swap that re-pairs the lines, which is why the fix
    must canonicalize over the full 8-fold orbit rather than just try the exchange.

    KEY CHOICE (load-bearing). The representative is the lexicographically smallest
    ``(space, name)`` arrangement -- the same rule ``_eri_normalize_factor`` uses in
    ``dressing.py``, whose group and parity helper are reused here rather than
    reimplemented. A ``(space, spin)`` key was tried first, to avoid depending on index
    names, and is WRONG: it is degenerate (several orbit members tie on it), so the
    tie-break decides the representative and two writings land on different ones --
    measured, that made all 6 surviving spin cases disagree instead of 4. Names are
    what break the tie deterministically.

    The consequence is the constraint ``_eri_canonical`` already documents (D7.2.5.2
    Fmi): this normalization is only meaningful on consistently-named indices. Within a
    single term that holds by construction -- both writings of one integral carry the
    same dummy names, only arranged differently -- which is exactly the scope this fix
    needs. It does NOT make two terms with differently-named dummies converge; that is
    ``canonicalize_term``'s job and is applied upstream.

    The returned sign is the parity of the chosen permutation. ``v`` is antisymmetric
    within each bra/ket pair, so a representative reached by an odd number of intra-pair
    swaps carries -1. The caller folds it into the term coefficient alongside the
    within-group parity, which is where the invariant actually lives: two writings of
    one integral can legitimately disagree PER FACTOR (``f2 = -f1`` here) and still
    agree PER TERM, because the term's own coefficient carries the difference.
    """
    # `getattr` rather than `.name`: this runs on every factor the adapter sees, and
    # rank-6+ callers legitimately pass lightweight index-only factor objects. Anything
    # without a name is not a `v`, so it needs no reorientation. Reordering also needs
    # `with_indices`, so require that too rather than assuming the full Tensor API.
    if getattr(factor, "name", None) != "v" or len(factor.indices) != 4:
        return factor, 1
    if not hasattr(factor, "with_indices"):
        return factor, 1

    from .optimization.dressing import _ERI_PERMUTATIONS, _perm_parity

    best = None
    for perm in _ERI_PERMUTATIONS:
        order = tuple(factor.indices[p] for p in perm)
        sig = tuple((x.space, x.name) for x in order)
        if best is None or sig < best[0]:
            best = (sig, order, _perm_parity(perm))
    return factor.with_indices(best[1]), best[2]


def _antisym_to_allowed(factor, label):
    """Map a spin-labeled factor to its allowed (spin-conserving-per-line) block
    via antisymmetry, returning ``(sign, indices)`` or ``None`` if genuinely zero.
    General rank-2n.

    Rank-4 ``v`` factors are first reoriented to one canonical member of their 8-fold
    ERI orbit (:func:`_orientation_normalized`), with that reorientation's parity folded
    into the returned sign, so the result does not depend on which exchange-related
    arrangement the caller happened to write (V1.1e.2).

    A rank-2n amplitude/integral is antisymmetric WITHIN its bra group (slots
    ``0..n-1``) and WITHIN its ket group (``n..2n-1``); the lines pair slot k with
    slot k+n. A within-group permutation reorders the spins, each transposition
    contributing sign -1. The factor maps to an allowed block (every line
    spin-conserving) iff the bra and ket spin MULTISETS match
    (``sorted(bra) == sorted(ket)``): then sorting the bra and the ket into the
    same spin order aligns the lines. The sign is the product of the two
    within-group permutation parities. If the multisets differ no permutation can
    conserve every line and the block is genuinely zero (rank-2: a single line
    that simply must match).

    On rank-4 this picks the same physical value as the bra/ket-swap enumeration
    it replaced -- when it lands on a different canonical block (abab vs baba)
    the two are related by exactly a bra-swap + ket-swap and evaluate identically;
    validated to reproduce GCC at rank-4 raw and through the full collapse+merge
    pipeline (~1e-17)."""
    factor, orient_sign = _orientation_normalized(factor)
    idx = [label[i.name] for i in factor.indices]
    n = len(idx) // 2
    bra, ket = idx[:n], idx[n:]
    bra_spins = [x.spin for x in bra]
    ket_spins = [x.spin for x in ket]
    if sorted(bra_spins) != sorted(ket_spins):
        return None
    bra_order = sorted(range(n), key=lambda k: bra_spins[k])
    ket_order = sorted(range(n), key=lambda k: ket_spins[k])
    sign = (orient_sign
            * _permutation_parity(bra_order)
            * _permutation_parity(ket_order))
    new_idx = [bra[k] for k in bra_order] + [ket[k] for k in ket_order]
    return (sign, new_idx)


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


def external_blocks(residual_template, fold_spin_flip: bool = True) -> list[dict]:
    """The external spin blocks of a residual, as {free-index-name: spin} dicts.

    Enumerates the spin patterns that conserve spin along the residual's lines
    (:func:`block_exists`), then keeps one representative per block.

    ``fold_spin_flip`` selects the equivalence used for "per block":

    - ``True`` (default, RCC): fold under the global a<->b flip, so doubles gives
      ``aaaa, abab`` -- ``bbbb`` is dropped as the flip image of ``aaaa``. Valid only
      when alpha and beta orbitals coincide (closed shell), which is exactly RCC's
      premise.
    - ``False`` (UCC): no folding. Doubles gives ``aaaa, abab, baba, bbbb`` and singles
      ``aa, bb``. In UCC alpha and beta are different orbitals, so ``bbbb`` is an
      independent amplitude, not a relabeling of ``aaaa``.

    The default is ``True`` so every existing RCC caller is unchanged. UCC block
    enumeration goes through :func:`ucc_independent_blocks`, which wraps this with
    ``fold_spin_flip=False`` and additionally folds the within-half antisymmetry
    (``baba`` -> ``abab``) that this function does not touch.
    """
    names = [i.name for i in residual_template.indices]
    seen = set()
    blocks = []
    for combo in itertools.product(SPINS, repeat=len(names)):
        label = {nm: SpinIndex(idx, s)
                 for nm, idx, s in zip(names, residual_template.indices, combo)}
        if not block_exists(residual_template, label):
            continue
        if fold_spin_flip:
            flip = tuple("b" if s == "a" else "a" for s in combo)
            key = min(combo, flip)  # canonical under global a<->b
        else:
            key = combo
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


def _split_same_spin_amplitude(factor: SpinFactor) -> list[tuple[object, SpinFactor]]:
    """Expand a same-spin all-alpha amplitude factor `t_n[a..a]` (slots
    [v*n, o*n]) into the mixed block via the pinned S4b.0 relation, as a list of
    (sign, SpinFactor) pairs.

    The mixed block has bra spins (a,..,a,b) and ket spins (a,..,a,b) -- a single
    beta on the last bra and last ket slot. The all-alpha block is reconstructed
    by BRA-ONLY antisymmetrization: one term per bra position p in 0..n-1 that
    receives the beta bra-slot (moved from position n-1), with the transposition
    sign; the KET is fixed. At n=2 this is exactly the old `_split_t2aaaa`
    (t2[aaaa] = t2[abab] - t2[abab](bra swap)); rank-6/8 fall out unchanged.
    Numerically pinned at n=2,3 to ~1e-17 (`S4bZeroCollapseRelationTests`)."""
    idx = factor.indices                      # SpinIndex, all spin 'a'
    n = len(idx) // 2
    vbra = [si.base for si in idx[:n]]         # virtual (bra) base indices
    ket = [si.base for si in idx[n:]]          # occupied (ket) base -- FIXED
    # Fixed mixed block: bra spins (a,..,a,b), ket spins (a,..,a,b) -- the single
    # beta on the last bra slot and last ket slot. Block string is those per-slot
    # spins (n=2 -> "abab"; n=3 -> "aabaab").
    spins = tuple("a" if k != n - 1 else "b" for k in range(n)) * 2
    block = "".join(spins)

    def mk(bra_order):
        bases = tuple(vbra[o] for o in bra_order) + tuple(ket)
        new_idx = tuple(SpinIndex(b, s) for b, s in zip(bases, spins))
        return SpinFactor(name=factor.name, block=block, indices=new_idx)

    # Move each virtual into the beta slot (last bra position) in turn: base order
    # places virtual `q` at slot n-1. The identity (q = n-1) leads. Sign = parity
    # of that base permutation. This is the base-permutation form of the old
    # `_split_t2aaaa` (n=2: virtual swap), generalized to rank-2n and pinned
    # numerically at n=2,3 (S4bSplitterTests).
    out = []
    for q in range(n - 1, -1, -1):
        order = [x for x in range(n) if x != q] + [q]   # q occupies slot n-1
        sign = 1
        for a in range(n):
            for b in range(a + 1, n):
                if order[a] > order[b]:
                    sign = -sign
        out.append((sign, mk(order)))
    return out


# Back-compat alias: the rank-4 entry name kept for existing callers/tests.
_split_t2aaaa = _split_same_spin_amplitude


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
        if _is_same_spin_amplitude(f):
            choices.append(_split_same_spin_amplitude(f))
        else:
            choices.append([(1, f)])

    return _product_over_choices(spinterm, choices)


def _is_same_spin_amplitude(f: SpinFactor) -> bool:
    """A rank-2n (n>=2) cluster amplitude in the all-alpha same-spin block --
    the factors the S2.0 collapse splits. t1 (rank-2) is already a single spatial
    block (`aa`) and is left alone; v/f are handled by the integral collapse and
    the single-block pass-through. Generalizes the old `t2 && aaaa` check to
    t2/t3/t4/... via the block being a nonempty all-'a' string of even length >=4."""
    return (f.name.startswith("t")
            and len(f.block) >= 4
            and len(f.block) % 2 == 0
            and set(f.block) == {"a"})


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


def independent_spin_blocks(rank):
    """The independent spin-orbital blocks of a rank-`rank` (= 2n) closed-shell
    cluster amplitude, one representative per Sz sector (R3.1.3a).

    A physical (Sz-conserving) block has equal α-count `k` in its bra and ket
    halves. Spin-flip pairs sector `k` with `n-k`; permutations exhaust the rest.
    So the independent sectors are `k = ⌈n/2⌉ … n` (the α-majority half of each
    flip pair), each written α-before-β per half. The α-count-⌈n/2⌉ sector is the
    balanced *reference*; higher-k sectors are the extra independent blocks that
    must be stored/read separately -- `aabb` alone does NOT carry a rank-8 t4
    (measured: `aaab` is not a signed-permutation combination of `aabb`, even
    from one shared spatial amplitude).

    The all-α sector (`k = n`) is EXCLUDED: `collapse_amplitudes` /
    `_split_same_spin_amplitude` already reduce the all-α block into the lower-Sz
    ones, so it never survives to the bridge. The surviving independent sectors
    are therefore `k = ⌈n/2⌉ … n-1`. Returns the balanced reference first, in
    α-before-β per-half layout: n=2→[`abab`], n=3→[`aabaab`],
    n=4→[`aabbaabb`,`aaabaaab`], n=5→[`aaabbaaabb`,`aaaabaaaab`].
    """
    n = rank // 2
    ref_k = -(-n // 2)                      # ceil(n/2)
    hi = max(ref_k, n - 1)                  # up to n-1 (all-α k=n is split away)
    blocks = []
    for k in range(ref_k, hi + 1):
        half = "a" * k + "b" * (n - k)
        blocks.append(half + half)
    return blocks


def _amplitude_block_tag(block):
    """Fold a spin-block string to its independent-sector tag (R3.1.3a): the
    α-majority half of its spin-flip pair, α-before-β per half. Blocks with the
    same tag are the same independent amplitude component (a spin-flip and/or a
    slot permutation apart); blocks with different tags are genuinely independent
    Sz sectors that need separate storage. t3 `aabaab`/`abbabb` → `aabaab`
    (one component); t4 `aabbaabb`→`aabbaabb`, `aaabaaab`/`abbbabbb`→`aaabaaab`
    (two components)."""
    n = len(block) // 2
    k = block[:n].count("a")
    if k < n - k:                           # β-majority -> flip to α-majority
        k = n - k
    half = "a" * k + "b" * (n - k)
    return half + half


def _ucc_block_tag(block: str) -> str:
    """Fold a spin-block string to its UCC storage tag (U0).

    Same α-before-β-per-half layout convention as :func:`_amplitude_block_tag`, but
    **without** the β-majority flip. That flip identifies a block with its spin-flip
    image, which is valid only when α and β orbitals coincide; in UCC they do not, so
    `bbbb` is its own stored amplitude rather than a relabeling of `aaaa`.

    Folds only the within-half antisymmetry (a slot permutation of one stored tensor):
    `baba` -> `abab`, `bbaa`(t3-style halves) -> sorted per half. Leaves the α-count of
    each half intact, so the surviving tags are the `n+1` α-count sectors per rank.
    """
    n = len(block) // 2
    return "".join(sorted(block[:n])) + "".join(sorted(block[n:]))


def ucc_independent_blocks(rank: int) -> list[str]:
    """The UCC spin blocks of a rank-``rank`` amplitude that must be STORED (U0).

    ``rank`` is the number of index slots (rank 2 = t1, rank 4 = t2, rank 6 = t3), so a
    rank-2n amplitude has ``n`` slots per half.

    Unlike the closed-shell case there is no global a<->b flip available, so all ``n+1``
    α-count sectors are independent and the all-α / all-β blocks do not fold away::

        rank 2 (t1) -> ["aa", "bb"]
        rank 4 (t2) -> ["aaaa", "abab", "bbbb"]          (PySCF t2aa / t2ab / t2bb)
        rank 6 (t3) -> ["aaaaaa", "aabaab", "abbabb", "bbbbbb"]

    Spin is conserved per line (slot k with slot k+n), so a stored block always has the
    same α-count in both halves -- which is why the answer is indexed by that one count.
    Within-half antisymmetry is folded by :func:`_ucc_block_tag`, so `baba` is not a
    separate entry.

    Deliberately NOT built on :func:`external_blocks`: that enumerates blocks of a
    residual template (and folds a<->b by default), whereas this is a property of the
    amplitude rank alone. Callers needing the residual's own blocks want
    ``external_blocks(tpl, fold_spin_flip=False)``.
    """
    if rank < 2 or rank % 2 != 0:
        raise ValueError(f"amplitude rank must be even and >= 2, got {rank}")
    n = rank // 2
    return [("a" * k + "b" * (n - k)) * 2 for k in range(n, -1, -1)]


def _canonicalize_amplitude_factor(f):
    """Reorder a cluster-amplitude factor's slots to ONE reference spin-block
    layout per rank, returning ``(sign, reordered_SpinIndices)`` (R3.1.2).

    A rank-2n amplitude is antisymmetric within its bra (first n slots) and within
    its ket (last n slots) independently. Different surviving spin blocks of the
    SAME spatial tensor (e.g. t3 in `aabaab` vs `abaaba`) are therefore signed
    permutations of one reference layout. The spin→AlgebraTerm bridge drops the
    spin label, so unless every factor is first mapped to that one reference
    layout, a factor read in a non-reference block indexes the wrong slice of the
    single spatial tensor -- the cross-target inconsistency that leaves T4≈0.

    Reference layout = each half stably sorted by spin (α before β). The sign is
    the product of the bra-half and ket-half sort parities. Numerically exact:
    the reordered block equals the reference block's slice (verified 0.0 on the
    UCCSDT t3 fixture). Non-amplitude factors (v/f) and t1 (single `aa` block)
    are returned unchanged with sign +1.

    R3.1.2 half (i): a β-majority block (e.g. t3 `abbabb`, 1α/2β per half) is not
    a permutation of the α-majority reference (`aabaab`, 2α/1β) -- it is the
    reference's SPIN-FLIP partner. A closed-shell amplitude is spin-flip
    symmetric (t[σ] = t[flip σ] index-for-index), so mapping a β-majority factor
    onto the stored reference block is a two-step slot permutation: flip α↔β,
    then sort α-before-β. Both halves flip together (a spin-balanced amplitude
    has na_bra == na_ket, so both are the same majority). The flip touches only
    the slot-ORDER used to read the single stored block; the base (spatial)
    indices keep their identities and spins as seen by the rest of the term, so
    shared/summed indices stay consistent across factors."""
    idx = f.indices
    n = len(idx) // 2
    if not (f.name.startswith("t") and len(idx) >= 4 and len(idx) % 2 == 0):
        return 1, idx

    # β-majority in the bra half => this block is the reference's spin-flip
    # partner; flip the sort key so α-before-β lands on the reference layout.
    flip = sum(1 for si in idx[:n] if si.spin == "a") * 2 < n

    def sort_half(slots):
        # stable sort by (flipped) spin (a<b); return new order + permutation parity
        def spin_key(si):
            s = si.spin
            if flip:
                s = "b" if s == "a" else "a"
            return s
        order = sorted(range(len(slots)), key=lambda k: (spin_key(slots[k]), k))
        sign = 1
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                if order[a] > order[b]:
                    sign = -sign
        return order, sign

    bra_order, s_bra = sort_half(idx[:n])
    ket_order, s_ket = sort_half(idx[n:])
    new_idx = (tuple(idx[o] for o in bra_order)
               + tuple(idx[n + o] for o in ket_order))
    return s_bra * s_ket, new_idx


def spinterm_to_algebraterm(spinterm: SpinTerm, externals):
    """Convert a spatial ``SpinTerm`` to an ``AlgebraTerm`` for the emit path
    (S3.0). ``externals`` is the set of free-index names (e.g.
    ``{"a","b","i","j"}``). Each factor becomes a ``Tensor`` over the spatial base
    indices; free and summed indices are split by name (de-duplicated,
    first-appearance order). The coefficient is carried as a ``Fraction``.

    Each amplitude factor is first canonicalized to one reference spin-block
    layout (:func:`_canonicalize_amplitude_factor`) so every reference to a given
    spatial tensor -- output or input factor -- uses the SAME layout; the factor
    permutation sign is folded into the coefficient (R3.1.2)."""
    from fractions import Fraction

    from .project import AlgebraTerm
    from .tensors import Tensor

    externals = set(externals)
    sign = 1
    canon_factors = []
    for f in spinterm.factors:
        s, new_idx = _canonicalize_amplitude_factor(f)
        sign *= s
        canon_factors.append(SpinFactor(name=f.name, block=f.block, indices=new_idx))

    # R3.1.3c: name each amplitude factor by its independent Sz sector. A rank-2n
    # amplitude with n >= 4 has more than one independent block (t4: aabbaabb +
    # aaabaaab); the reference sector keeps the bare name (`t4`), a higher sector
    # is read from its own stored tensor (`t4_aaabaaab`). The canonicalizer's
    # spin-flip already reorders the base indices + folds the sign so the read of
    # the tagged block's tensor is exact; the tag just routes it to the right
    # storage. Lower ranks (t1/t2/t3) have a single independent block, so the tag
    # equals the reference and the name is unchanged -- byte-identical there.
    def _factor_tensor_name(f):
        if not (f.name.startswith("t") and len(f.block) >= 8):
            return f.name                      # t1/t2/t3, v, f: single block
        blocks = independent_spin_blocks(len(f.block))
        tag = _amplitude_block_tag(f.block)
        if tag == blocks[0]:                   # reference sector
            return f.name
        return f"{f.name}_{tag}"

    factors = tuple(
        Tensor(_factor_tensor_name(f), tuple(si.base for si in f.indices))
        for f in canon_factors
    )
    spinterm = SpinTerm(coeff=spinterm.coeff * sign,
                        external_block=spinterm.external_block,
                        factors=tuple(canon_factors))
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
    # R3.1.2 half (ii): order free indices canonically (occupieds before
    # virtuals, then by base name) instead of per-term first-appearance. The
    # load-bearing part is the NAME-SORT WITHIN each space: first-appearance order
    # differs between terms with the same externals (a,c,b vs a,b,c), transposing
    # their residual arrays so a manifold sum over them is wrong; name-sort makes
    # every term agree. The occ-vs-vir BETWEEN-space order is chosen occ-first to
    # match the C++ runtime's amplitude/denominator/residual layout (rank_dims:
    # t_r(i1..ir, a1..ar), occupied first) so the emitted spatial kernel's result
    # buffer lines up with the runtime without a transpose. This between-space
    # choice is invariant for the Python oracle (`residual_einsum` re-splits by
    # space internally -> always [vir,occ] output) and for the P2 bridge gates
    # (`_bridge_output_layout` re-derives vir+occ), so only the C++ path cares.
    # See docs/CCGEN_CCSDTQ_MULTISECTOR.md.
    free.sort(key=lambda b: (0 if b.space == "occ" else 1, b.name))
    return AlgebraTerm(Fraction(spinterm.coeff), factors,
                       tuple(free), tuple(summed), True)


def ucc_spinterm_to_algebraterm(spinterm: SpinTerm, externals):
    """U1.1 -- the UCC sibling of :func:`spinterm_to_algebraterm`.

    Same bridge, with the two closed-shell mechanisms disabled:

    * **No slot canonicalization.** `_canonicalize_amplitude_factor` maps every
      rank >= 2 amplitude onto ONE reference layout, because the RCC bridge drops
      the spin label and every block must then index the single stored spatial
      tensor. Under UCC `t2aa`/`t2ab`/`t2bb` are separate arrays, so reordering a
      factor's slots onto another block's layout reads the wrong array. Slots are
      left exactly as the spin integration produced them.
    * **Every amplitude factor is suffixed with its UCC block tag**, via U0's
      `_ucc_block_tag` (which folds only the within-half antisymmetry, never the
      beta-majority spin flip). RCC keeps its reference sector bare (`t4`) so its
      emit stays byte-identical; UCC has no privileged reference -- `t2_aaaa` and
      `t2_bbbb` are peers -- so a bare name would be ambiguous and is never
      emitted here.

    Non-amplitude factors (`v`, `f`) keep their names: they are spin-free in the
    MO block cache and carry their spin structure in the block, not the storage.

    Free-index ordering is shared with the RCC bridge (occ-first, then by base
    name) -- that contract is about matching the C++ runtime's `rank_dims`, not
    about spin, so it applies unchanged. See docs/CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md.
    """
    from fractions import Fraction

    from .project import AlgebraTerm
    from .tensors import Tensor

    externals = set(externals)

    def _ucc_factor_tensor_name(f):
        # F2.0b: v/f carry their block too. Under RCC they are spin-free (one
        # stored tensor sliced by space), but UCC needs v_aaaa / v_abab / v_bbbb
        # -- different shapes -- and the spin is NOT recoverable downstream: on
        # doubles_abab, 51 of 82 terms have a v/f index appearing on no amplitude
        # factor, so inference from term context is a dead end. The block is
        # already on the SpinFactor; this just stops discarding it.
        #
        # Amplitudes fold to their UCC storage tag (within-half antisymmetry is a
        # slot permutation of one stored array). v/f keep the block verbatim: an
        # integral block is not an amplitude and has no such folding.
        if f.name.startswith("t"):
            return f"{f.name}_{_ucc_block_tag(f.block)}"
        return f"{f.name}_{f.block}"

    factors = tuple(
        Tensor(_ucc_factor_tensor_name(f), tuple(si.base for si in f.indices))
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
    free.sort(key=lambda b: (0 if b.space == "occ" else 1, b.name))
    return AlgebraTerm(Fraction(spinterm.coeff), factors,
                       tuple(free), tuple(summed), True)


# ── S3/R1.0: full spin-adaptation of a residual manifold to spatial AlgebraTerms ──
#
# Chains the S2 pipeline end to end, per target, on the closed-shell representative
# external block, then bridges to AlgebraTerms the emit path consumes:
#
#   ucc_manifold -> canonicalize_spin_blocks -> collapse_amplitudes
#     -> collapse_integrals -> merge_terms -> spinterm_to_algebraterm
#
# This is what replaces the relabel-only `lowering/restricted_closed_shell` in the
# planck emit path: the coefficients become the spatial 2*(direct)-(exchange)
# structure, the term count drops to the spatial count, and the emitted kernel is
# a genuine restricted (spatial) contraction rather than spin-orbital algebra bound
# to spatial storage. Gated numerically by GeneratedSpatialEnergyGate (R1.2).


def _residual_template(target: str, terms):
    """Build the residual `Tensor` template for a target from its terms' free
    indices (virtuals first, then occupieds, first-appearance order). Energy
    (rank 0) has no free indices and yields an index-less template."""
    from .tensors import Tensor

    if not terms:
        return Tensor("R", ())
    free = terms[0].free_indices
    vir = [i for i in free if i.space == "vir"]
    occ = [i for i in free if i.space == "occ"]
    return Tensor("R", tuple(vir) + tuple(occ))


def intermediate_template(spec):
    """The adaptation output template for a dressed-intermediate ``IntermediateSpec``
    (V1.0): the spec's OWN declared slot order, not virtuals-first.

    Feed to ``spin_adapt_equations(..., templates={spec.name: intermediate_template(spec)})``
    so the operator adapts on a block that respects its physical line pairing.
    `Wmbej` is the case that forces this: its `ovvo` slots are [m,b,e,j] with lines
    m-e and b-j, but the virtuals-first default reorders them to [b,e,m,j], so the
    external block is applied to factors carrying the operator's real pairing --
    `v(m,b,e,j)` takes tag `aabb`, whose m-e line has m=a/e=b, and is rejected. Every
    spin case of every term dies the same way, so the operator adapts to zero.

    `Wmbej` is the only operator the default ZEROES, but agreement elsewhere is not a
    property to rely on: whether a reorder is harmful depends on the adapter's
    relabeling, not on space homogeneity. Measured post-adaptation, the free-index
    order differs from `spec.indices` for `Fmi`, `Wmnij`, and `Wmbej` -- while `Fme`
    (mixed-space `ov`) agrees. So always pass the own-order template for an
    intermediate; do not special-case by block pattern."""
    from .tensors import Tensor

    return Tensor(spec.name, tuple(spec.indices))


def emitted_intermediate_layout(definition_terms):
    """The slot layout the EMITTER will give a `build_<op>` result, as
    ``(indices, index_space_sig)`` (V1.1b).

    Read out of the emitter's own normalization rather than re-derived:
    ``_emit_intermediate_builder`` shapes the builder from
    ``lower_term_restricted_closed_shell(definition_terms[0]).canonical_free_indices``,
    and the consumer side (``_map_factor`` -> ``_target_expr``) emits usage-site
    indices that went through the same lowering. Reusing that one normalizer is what
    keeps the two ends agreeing BY CONSTRUCTION instead of by luck -- a second sort
    that agrees today and drifts later is exactly how the spec/term desynchronization
    arose in the first place."""
    from .lowering.restricted_closed_shell import lower_term_restricted_closed_shell

    if not definition_terms:
        return (), ""
    lowered = lower_term_restricted_closed_shell(definition_terms[0], "reference")
    indices = tuple(lowered.canonical_free_indices)
    sig = "".join("o" if i.space == "occ" else "v" for i in indices)
    return indices, sig


def block_keyed_intermediate_name(name, block=None):
    """A dressed intermediate's storage name for one spin block (V1.1c).

    ``block=None`` (the RCC reference sector) keeps the bare name, so the RCC emit
    is byte-identical; a non-reference block appends its tag -- ``Wmnij`` ->
    ``Wmnij_abab``. Same ``f"{name}_{tag}"`` shape the bridge already uses for
    amplitude sectors (``t4_aaabaaab``, R3.1.3c) and that U1.1 uses for UCC
    amplitudes (``t2_aaaa``): ONE naming mechanism for amplitudes, ERIs, and
    intermediates.

    Needed because ``IntermediateSpec`` hashes/compares on
    ``(name, indices, index_space_sig)``. Under RCC each operator adapts to a single
    spec so nothing collides, but under UCC one ``Wmnij`` becomes several spin-block
    variants that would collide into one. Costs nothing on the consumer side --
    ``_map_factor`` already resolves any name in ``intermediate_names`` as a local."""
    return name if block is None else f"{name}_{block}"


def recount_intermediate_usage(specs, equations):
    """Recount each spec's ``usage_count`` / ``usage_targets`` against the ADAPTED
    residual (V1.1d), returning new specs.

    The counts on a freshly dressed spec are computed from the GCC residual in
    ``_dress_operator_equations``, and adaptation changes them -- measured on dressed
    CCSD, ``Wmbej`` goes 5 -> 10 as the adapter splits its usage sites across spin
    cases (the other four are unchanged at 1). A stale count is a trap rather than a
    live bug today (it only feeds the emitted comment), but it is the natural input to
    any materialize-once-vs-inline decision, so it must not be left describing a
    different equation set than the one being emitted."""
    from dataclasses import replace

    names = {s.name for s in specs}
    out = []
    for spec in specs:
        count = 0
        targets = []
        for manifold, terms in equations.items():
            n = sum(1 for t in terms for f in t.factors if f.name == spec.name)
            if n:
                count += n
                targets.append(manifold)
        out.append(replace(spec, usage_count=count, usage_targets=tuple(targets)))

    # Bidirectional closure: every referenced intermediate needs a spec, and every
    # spec needs a reference. A one-sided rename (the failure mode V1.1c's tagging
    # could introduce) shows up here as a dangling reference or an orphan spec.
    def _is_primitive(fname):
        # v/f/delta, plus any cluster amplitude `t<rank>` or sector-tagged
        # `t<rank>_<tag>` (R3.1.3c) -- those are runtime state, not intermediates.
        if fname in {"v", "f", "delta"}:
            return True
        stem = fname.split("_", 1)[0]
        return stem.startswith("t") and stem[1:].isdigit()

    referenced = {
        f.name
        for terms in equations.values()
        for t in terms
        for f in t.factors
        if not _is_primitive(f.name)
    }
    dangling = referenced - names
    if dangling:
        raise ValueError(
            f"recount_intermediate_usage: the adapted residual references "
            f"{sorted(dangling)} with no matching spec -- an intermediate was "
            f"renamed on one side only.")
    orphans = [s.name for s in out if s.usage_count == 0]
    if orphans:
        raise ValueError(
            f"recount_intermediate_usage: spec(s) {sorted(orphans)} are never "
            f"referenced by the adapted residual -- they would emit an unused "
            f"build_<op>, or their name drifted from the usage sites.")
    return out


def adapt_intermediate_spec(spec, adapter=None, relayout=True, block=None):
    """Spin-adapt a dressed-intermediate ``IntermediateSpec``'s definition terms
    (V1.1a) and re-derive its declared layout from them (V1.1b).

    ``adapter`` is a ``{target: [AlgebraTerm]} -> {key: [AlgebraTerm]}`` callable
    taking a ``templates`` keyword; defaults to :func:`spin_adapt_equations` (RCC).
    Pass ``ucc_adapt_equations`` for UCC -- the point of the parameter is that V5
    becomes a substitution, not a second code path.

    Adaptation runs on the spec's OWN slot order via :func:`intermediate_template`,
    so the operator's physical line pairing survives (V1.0).

    ``relayout=True`` (V1.1b) replaces ``indices``/``index_space_sig`` with
    :func:`emitted_intermediate_layout` of the adapted terms. Without it the spec
    keeps the operator's declared slot order, which disagrees with what the emitter
    builds for `Fmi`, `Wmnij`, and `Wmbej` -- and for `Wmbej` the SIGNATURE differs
    too (`ovvo` declared vs `oovv` emitted), so the metadata is not merely permuted
    but wrong about which spaces sit where.

    This corrects metadata, NOT the emitted code: the emit path already normalizes
    both the builder and the usage sites through the same lowering, so it is
    self-consistent today. The danger is downstream consumers that trust
    ``indices``/``index_space_sig`` -- dependency ordering, block-keyed identity
    (V1.1c), memory estimates. Do not "fix" the mismatch by forcing the declared
    order INTO the builder; that would create a miscompile that does not exist.

    ``block`` (V1.1c) tags the returned spec's name for a non-reference spin block,
    via :func:`block_keyed_intermediate_name`; ``None`` keeps the bare name so the
    RCC path is byte-identical.

    Recounting usage against the adapted residual is V1.1d. ``relayout=False`` keeps
    V1.1a-only behavior for tests that isolate the steps."""
    from dataclasses import replace

    fn = adapter or spin_adapt_equations
    adapted = fn({spec.name: list(spec.definition_terms)},
                 templates={spec.name: intermediate_template(spec)})
    if spec.name not in adapted:
        raise ValueError(
            f"adapt_intermediate_spec: adapter returned no manifold for "
            f"{spec.name!r} (got keys {sorted(adapted)}). A dressed intermediate "
            f"has one target; a split result means the adapter treated it as a "
            f"multi-sector residual.")
    terms = tuple(adapted[spec.name])
    name = block_keyed_intermediate_name(spec.name, block)
    if not relayout:
        return replace(spec, name=name, definition_terms=terms)

    indices, sig = emitted_intermediate_layout(terms)
    if len(indices) != len(spec.indices):
        raise ValueError(
            f"adapt_intermediate_spec: {spec.name!r} adapted to rank "
            f"{len(indices)} but was declared rank {len(spec.indices)} -- the "
            f"adapter changed the operator's external slot count, which it must "
            f"never do.")
    return replace(spec, name=name, definition_terms=terms, indices=indices,
                   index_space_sig=sig)


def _closed_shell_representative_block(template):
    """The canonical closed-shell external block: within each half (bra virtuals,
    ket occupieds) put all α slots before all β slots (α = ceil(n/2)). Each occ/vir
    residual line is spin-balanced (bra[k]==ket[k]), so the block is spin-valid, and
    it is the SAME reference layout `_canonicalize_amplitude_factor` sorts every
    amplitude factor into (α-before-β per half) -- so the residual OUTPUT and every
    input amplitude factor share one spatial layout and the spin→AlgebraTerm bridge
    is lossless (R3.1.2). n=2 -> abab, n=3 -> aabaab, n=4 -> aabbaabb. Energy (n=0)
    -> {} (scalar)."""
    names = [i.name for i in template.indices]
    n = len(names) // 2
    n_alpha = (n + 1) // 2
    half = ["a" if k < n_alpha else "b" for k in range(n)]
    spins = half * 2
    return {nm: sp for nm, sp in zip(names, spins)}


def _representative_block_for_sector(template, k_alpha):
    """External block for a target with `k_alpha` α slots per half, α-before-β --
    the residual sector matching amplitude tag `('a'*k+'b'*(n-k))*2` (R3.1.3d).
    `k_alpha = ceil(n/2)` reproduces `_closed_shell_representative_block` (the
    reference)."""
    names = [i.name for i in template.indices]
    n = len(names) // 2
    half = ["a" if j < k_alpha else "b" for j in range(n)]
    spins = half * 2
    return {nm: sp for nm, sp in zip(names, spins)}


def _ucc_block_for_tag(template, tag):
    """External block dict assigning `tag`'s spins to `template`'s slots (U1.0).

    `tag` is a UCC storage tag from :func:`ucc_independent_blocks` -- one spin per
    slot, bra half then ket half, already in the alpha-before-beta-per-half layout
    `_ucc_block_tag` folds to.
    """
    names = [i.name for i in template.indices]
    return {nm: sp for nm, sp in zip(names, tag)}


def ucc_term_spins(term) -> dict:
    """F2.2a -- the spin of every index in a UCC ``AlgebraTerm``.

    Returns ``{index_name: "a" | "b"}``. Each factor carries its spin block in
    its name (F2.0b), and the tag is **positional**: character k is slot k's
    spin, independent of that slot's space. `v_abab` appears with 13 different
    space patterns all sharing the one tag, so the space cannot be used to
    recover it -- the position can.

    This is the map the evaluator slices by. It exists because the spin is not
    otherwise recoverable downstream: on `doubles_abab`, 51 of 82 terms have a
    `v`/`f` index that appears on no amplitude factor, so inference from term
    context is a dead end for the majority of terms.

    Raises rather than resolving a disagreement. Measured across every manifold
    the generator reaches -- ccsd 328, ccsdt 2490, ccsdtq 18137 terms -- there
    are **zero** clashes, zero untagged factors and zero tag-length mismatches,
    so every branch below is unreachable today. They are asserted because this
    map is what F2.2b/F2.2c rest on, and a silent last-write-wins would surface
    as a wrong slice far from its cause.
    """
    spins: dict = {}
    for factor in term.factors:
        name = factor.name
        root, sep, tag = name.partition("_")
        if not sep or not tag:
            raise ValueError(
                f"ucc_term_spins: factor {name!r} carries no spin block. Every "
                f"UCC factor is tagged by the bridge (F2.0b); an untagged one "
                f"means an RCC term reached the UCC evaluator.")
        if not set(tag) <= {"a", "b"}:
            raise ValueError(
                f"ucc_term_spins: factor {name!r} has a non-spin tag {tag!r}.")
        if len(tag) != len(factor.indices):
            raise ValueError(
                f"ucc_term_spins: factor {name!r} has {len(factor.indices)} "
                f"slots but a {len(tag)}-character tag {tag!r}.")
        for idx, spin in zip(factor.indices, tag):
            previous = spins.setdefault(idx.name, spin)
            if previous != spin:
                raise ValueError(
                    f"ucc_term_spins: index {idx.name!r} is {previous!r} in one "
                    f"factor and {spin!r} in {name!r}. A summed index must carry "
                    f"one spin throughout a term; two means the spin integration "
                    f"produced a term that cannot be evaluated block-wise.")
    return spins


def ucc_term_index_spins(term) -> dict:
    """U3b.0 -- the spin of every index in one resolved UCC ``AlgebraTerm``.

    Returns ``{index name: 'a' | 'b'}``.

    WHY THIS IS NEEDED AT ALL. The emitter writes loop bounds from an ``Index``,
    but an Index carries only a SPACE (occ/vir), never a spin -- the U1 bridge
    drops the spin label by design, because RCC stores one spatial tensor per rank
    and has nothing to route. Under UCC the bound differs per spin (``noa`` vs
    ``nob``), so it has to come from somewhere else.

    WHERE IT COMES FROM. A resolved factor carries its block in its NAME
    (``t2_abab``, ``v_aaaa``, ``f_bb``), and slot *k* of that factor carries spin
    ``tag[k]`` -- POSITIONALLY, not grouped by space. That matters: ``t2_abab``'s
    slots are ``(vir, vir, occ, occ)``, so the tag does not read "occ half then vir
    half" here; it reads straight across the slots as they appear.

    Every index of every term is covered, because a summed index appears in at
    least one resolved factor by construction. Measured across the CCSD UCC
    manifold: 2346 assignments, zero conflicts. A conflict would mean two factors
    disagree about an index's spin, which is a slot-mapping defect (the R3.1.2
    failure mode) rather than something to paper over -- so it raises.

    Unresolved factors (a bare ``v``/``f``/``t2``, i.e. the RCC path) contribute
    nothing, so calling this on RCC terms returns an empty map rather than
    guessing.
    """
    spins: dict[str, str] = {}
    for factor in term.factors:
        tensor = factor.tensor if hasattr(factor, "tensor") else factor
        name = getattr(tensor, "name", getattr(factor, "name", ""))
        # Split on the last underscore rather than a regex: spin.py imports no `re`
        # and this needs no pattern beyond "suffix is all a/b".
        root, _, tag = name.rpartition("_")
        if not root or not tag or any(ch not in "ab" for ch in tag):
            continue
        indices = getattr(factor, "indices", ())
        if len(indices) != len(tag):
            raise ValueError(
                f"ucc_term_index_spins: factor {name!r} has {len(indices)} slots "
                f"but a {len(tag)}-slot spin tag; the slot mapping is wrong.")
        for slot, index in enumerate(indices):
            spin = tag[slot]
            previous = spins.get(index.name)
            if previous is not None and previous != spin:
                raise ValueError(
                    f"ucc_term_index_spins: index {index.name!r} is {previous!r} in "
                    f"one factor and {spin!r} in {name!r}. Two factors disagree "
                    f"about an index's spin, which is a slot-mapping defect.")
            spins[index.name] = spin
    return spins


def ucc_adapt_equations(equations, blocks=None, adapter=None):
    """U1.0 -- resolve a GCC residual manifold into UNRESTRICTED (spin-block)
    ``AlgebraTerm``s. Returns ``{f"{target}_{tag}": [AlgebraTerm]}``.

    The UCC sibling of :func:`spin_adapt_equations`, and deliberately not a
    modification of it: RCC drives one residual per *Sz sector* and runs three
    collapse steps (`canonicalize_spin_blocks`, `collapse_amplitudes`,
    `collapse_integrals`) that fold spin blocks into a single spatial tensor. UCC
    drives one residual per *stored block* and runs **none** of them, because
    `t2aa`/`t2ab`/`t2bb` are separate arrays.

    Every target is keyed by its block, including the reference: UCC has no
    privileged sector, so a bare `doubles` would be ambiguous.

    ``blocks`` overrides the per-rank block set (defaults to
    :func:`ucc_independent_blocks`). ``adapter`` overrides the term bridge and is
    parameterized rather than hard-wired so a later variant substitutes here
    instead of forking this driver -- the same shape
    `adapt_intermediate_spec(adapter=...)` already uses.
    """
    bridge = adapter or ucc_spinterm_to_algebraterm
    out: dict = {}
    for target, terms in equations.items():
        template = _residual_template(target, terms)
        n = len(template.indices) // 2
        if n == 0:
            # energy: scalar target, no external block to resolve
            block = _closed_shell_representative_block(template)
            externals = frozenset(block)
            merged = merge_terms(ucc_integrate_target(terms, block), externals)
            out[target] = [bridge(st, externals) for st in merged]
            continue
        for tag in (blocks or ucc_independent_blocks(2 * n)):
            block = _ucc_block_for_tag(template, tag)
            externals = frozenset(block)
            merged = merge_terms(ucc_integrate_target(terms, block), externals)
            adapted = [bridge(st, externals) for st in merged]
            if terms and not adapted:
                # Same guard spin_adapt_equations carries: a non-empty GCC
                # manifold that integrates to nothing is a slot-ordering or
                # block-routing bug, and it emits a kernel that compiles, runs and
                # is silently wrong. Fail loudly.
                raise ValueError(
                    f"ucc_adapt_equations: target {target}_{tag} has "
                    f"{len(terms)} GCC term(s) but resolved to ZERO terms on "
                    f"block {tag!r} (slots {[i.name for i in template.indices]})."
                )
            out[f"{target}_{tag}"] = adapted
    return out


def spin_adapt_equations(equations, templates=None):
    """Spin-adapt a whole GCC residual manifold to restricted (spatial)
    ``AlgebraTerm``s (R1.0). Returns ``{key: [AlgebraTerm]}``; `key` is the target
    for the reference Sz sector, and `target + "_" + tag` for each additional
    independent sector a rank-2n (n>=4) residual carries (R3.1.3d) -- e.g.
    ``quadruples`` (aabbaabb) and ``quadruples_aaabaaab``. Each stored amplitude
    block gets its own residual, integrated on that sector's external block. The
    bridge already names the second-sector *input* factors `t4_aaabaaab`
    (R3.1.3c), so the residual sets close on the same block vocabulary. Pure
    symbolic transform; gated numerically by R1.2 (reference) and the rank-8
    solve-path gate (both sectors).

    ``templates`` (V1.0) optionally supplies an explicit output `Tensor` template
    per target, overriding the virtuals-first :func:`_residual_template` default.
    Required for DRESSED INTERMEDIATES: the external block is assigned by slot
    POSITION and lines pair slot k with k+n, so a target whose own slot order is
    not virtuals-first (e.g. Wmbej, `ovvo`, lines m-e and b-j) is assigned a block
    that violates spin conservation on every line and integrates to ZERO. Passing
    the operator's declared index order keeps its physical line pairing intact.
    Residual targets pass ``None`` and keep the virtuals-first convention the C++
    runtime's `rank_dims` depends on (R3.1.2 half (ii), 02364db) -- this override
    exists so that contract does NOT have to move to accommodate intermediates."""
    out: dict = {}
    for target, terms in equations.items():
        template = (templates or {}).get(target) or _residual_template(target, terms)
        n = len(template.indices) // 2
        if n == 0:
            # energy: scalar, single block
            block = _closed_shell_representative_block(template)
            out[target] = _adapt_on_block(terms, block)
            continue
        ref_k = -(-n // 2)                          # ceil(n/2)
        hi = max(ref_k, n - 1)                       # sectors ceil(n/2)..n-1
        for k in range(ref_k, hi + 1):
            block = _representative_block_for_sector(template, k)
            adapted = _adapt_on_block(terms, block)
            tag = _amplitude_block_tag(("a" * k + "b" * (n - k)) * 2)
            key = target if k == ref_k else f"{target}_{tag}"
            if terms and not adapted:
                # V1.0 guard: a non-empty GCC manifold that integrates to nothing
                # is a slot-ordering/block bug, not a physical result -- it emits a
                # kernel that compiles, runs, and is silently wrong (the class of
                # the R3.1.2 bridge and B5 ERI-convention defects). Fail loudly
                # instead. See `templates` above for the dressed-intermediate fix.
                raise ValueError(
                    f"spin_adapt_equations: target {key!r} has {len(terms)} GCC "
                    f"term(s) but adapted to ZERO on block "
                    f"{''.join(block[i.name] for i in template.indices)!r} "
                    f"(slots {[i.name for i in template.indices]}). The external "
                    f"block is assigned by slot position and lines pair k with "
                    f"k+n, so this usually means the target's slot order is not "
                    f"the virtuals-first convention -- pass an explicit template "
                    f"via `templates` to preserve its own line pairing.")
            out[key] = adapted
    return out


def _adapt_on_block(terms, block):
    """The S2 spin-adaptation pipeline for one external `block`, returning the
    bridged spatial ``AlgebraTerm``s (R3.1.3d helper -- the body
    :func:`spin_adapt_equations` runs per independent sector)."""
    externals = frozenset(block)
    canon = [canonicalize_spin_blocks(st)
             for st in ucc_integrate_target(terms, block)]
    collapsed = [c for st in canon for c in collapse_amplitudes(st)]
    collapsed = [c for st in collapsed for c in collapse_integrals(st)]
    merged = merge_terms(collapsed, externals)
    return [spinterm_to_algebraterm(st, externals) for st in merged]


def ucc_integrate_target(terms, block):
    """All surviving integrated SpinTerms of `terms` for one external `block`
    (a {name: spin} dict). The per-block slice of :func:`ucc_manifold`, factored
    out so :func:`spin_adapt_equations` can drive a single representative block."""
    acc: list = []
    for term in terms:
        acc.extend(ucc_integrate_term_antisym(term, block))
    return acc
