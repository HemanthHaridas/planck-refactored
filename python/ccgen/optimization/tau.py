"""tau-recognition for coupled-cluster residual factorization (A1).

The hand-written CCSD/CCSDT tensor backends factor their residuals through
dressed operators; the first and simplest of these is the doubles pseudo-
amplitude

    tau_{ij}^{ab} = t2(a,b,i,j) + 1/2 * ( t1(a,i) t1(b,j) - t1(a,j) t1(b,i) )

ccgen's current CSE pass cannot introduce tau because tau is a *definition*
substituted globally, not a repeated leaf sub-contraction.  This module builds
up tau-recognition in small offline (codegen-inert) pieces:

  A1.0  tau spec           -- what tau *is*, as data (this file)
  A1.1  external skeleton  -- the fingerprint two terms must share to merge

Later steps (A1.2+) add the detector, exact-coefficient validation, the
term-list rewrite, and finally the emit wiring behind a flag.  Nothing here
changes generated code.

ponytail: tau is a fixed closed-form definition, so recognition is exact
pattern-match, not the heuristic subgraph-iso the Wmnij/Wabef operators (A3)
will need. Keep those out of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ..canonicalize import canonicalize_term, relabel_term_dummies
from ..indices import Index
from ..project import AlgebraTerm
from ..tensors import Tensor, t1, t2 as make_t2

# tau carries the same index layout and antisymmetry as t2: virtuals first,
# then occupieds, antisymmetric within each pair.  Mirrors tensors.t2.
TAU_NAME = "tau"


def tau(a: Index, b: Index, i: Index, j: Index) -> Tensor:
    """The tau pseudo-amplitude factor tau_{ij}^{ab} (same shape as t2)."""
    return Tensor(TAU_NAME, (a, b, i, j), antisym_groups=((0, 1), (2, 3)))


@dataclass(frozen=True)
class TauSpec:
    """The tau = t2 + 1/2 P(t1 t1) definition, as data.

    ``t1t1_coeff`` is the weight of ONE ordered t1(a,i) t1(b,j) product in the
    bare definition (1/2, with the antisymmetrizing partner carrying the
    opposite sign implicitly via P).

    ``written_t1t1_weight`` is the coefficient of the SINGLE ordered
    t1(a,i)t1(b,j) representative that the ccgen pipeline actually writes, per
    unit tau coefficient.  It is NOT 1/2: the pipeline's antisymmetrization
    collapses the two equal P permutations (against a symmetric <pq||rs>-type
    residue) into one representative, doubling the bare 1/2 P weight -- once for
    the collapse and once because the ordered product appears with unit weight
    in tau itself.  Empirically ``written_t1t1_weight = 2`` exactly, verified
    across the CCSD energy and both singles tau contractions (t1t1_member =
    2 * t2_member; see test_tau).  A1.4 reconstructs the written t1t1
    coefficient with this and checks it against the source EXACTLY.
    """

    t2_coeff: Fraction = Fraction(1)
    t1t1_coeff: Fraction = Fraction(1, 2)  # bare 1/2 P definition weight
    written_t1t1_weight: Fraction = Fraction(2)  # single written representative


TAU_SPEC = TauSpec()


def external_skeleton(term: AlgebraTerm) -> tuple[object, ...]:
    """Dummy-blind, coeff-blind fingerprint of a term's external structure.

    Two terms can only be the t2-half and t1t1-half of the same tau if they
    contract to the *same* free-index block.  The skeleton captures exactly
    that shared structure and nothing that distinguishes the two halves:

      * the free-index space pattern (which externals are occ / vir / gen),
      * the multiset of factor *names* (so a t2 term and a t1,t1 term differ
        here -- that is intentional; grouping happens on the free block, and
        this skeleton is the *bucket key* that must match on the free block
        while the name-multiset lets the grouper tell the halves apart).

    Coefficients and dummy names are excluded: the term is relabeled first so
    two structurally identical terms with different dummy letters agree.
    """
    relabeled = relabel_term_dummies(term)

    free_pattern = tuple(idx.space for idx in relabeled.free_indices)
    name_multiset = tuple(sorted(f.name for f in relabeled.factors))
    return (free_pattern, name_multiset)


def free_block_key(term: AlgebraTerm) -> tuple[str, ...]:
    """Just the free-index space pattern -- the block both tau halves share.

    Unlike ``external_skeleton`` this is name-blind, so the t2 term and the
    t1t1 term of one tau produce the *same* key.  This is the grouping key for
    A1.3; A1.1 only needs it to state the shared-block invariant.
    """
    relabeled = relabel_term_dummies(term)
    return tuple(idx.space for idx in relabeled.free_indices)


@dataclass(frozen=True)
class T1T1Half:
    """A term recognized as an isolated 1/2 P(t1 t1) half of a tau definition.

    ``externals`` are the four free indices in tau's (a, b, i, j) layout:
    virtuals first (from the two t1 upper indices), then occupieds.  ``coeff``
    is the term's coefficient (magnitude 1/2, sign either).
    """

    externals: tuple[Index, Index, Index, Index]
    coeff: Fraction


def match_t1t1_half(term: AlgebraTerm) -> T1T1Half | None:
    """A1.2 -- is ``term`` a bare 1/2 P(t1 t1) half of a tau?

    The tau *definition* introduces the isolated pseudo-amplitude half
    ``1/2 * t1(a,i) t1(b,j)`` over a doubles block's four externals.  This is
    the term the substitution recognizes -- NOT the energy contribution
    ``1/2 t1 t1 v`` (which carries a v factor and contracts to a scalar).

    Accepts iff, after canonical relabeling:

      * exactly two factors, both named ``t1``;
      * every index is free (external) -- no summed/contracted index, so the
        two t1s do not contract with each other or anything else;
      * the four indices are distinct and split 2 vir + 2 occ, so they cover a
        genuine (a,b|i,j) doubles block;
      * the coefficient magnitude is exactly 1/2.

    Returns a ``T1T1Half`` with the externals in tau's (vir, vir, occ, occ)
    order, or ``None``.  Sign is preserved for A1.4's exact validation.
    """
    term = relabel_term_dummies(term)

    if len(term.factors) != 2:
        return None
    if any(f.name != "t1" for f in term.factors):
        return None
    if term.summed_indices:
        return None

    indices: list[Index] = []
    for f in term.factors:
        # t1 is t1(vir, occ); a well-formed factor has exactly one of each.
        if len(f.indices) != 2:
            return None
        indices.extend(f.indices)

    if len(set(indices)) != 4:
        return None  # a repeated index means an internal contraction

    virs = [x for x in indices if x.space == "vir"]
    occs = [x for x in indices if x.space == "occ"]
    if len(virs) != 2 or len(occs) != 2:
        return None

    if abs(term.coeff) != TAU_SPEC.t1t1_coeff:
        return None

    externals = (virs[0], virs[1], occs[0], occs[1])
    return T1T1Half(externals=externals, coeff=term.coeff)


# ---------------------------------------------------------------------------
# A1.3 -- pairing t2-terms with their t1t1 halves via a tau-normalized residue
# ---------------------------------------------------------------------------
#
# A t2-term and a t1t1-term are the two halves of ONE tau contraction iff they
# are identical after the amplitude part is collapsed to a single ``tau``
# factor.  We build that "residue" for each candidate and pair on residue
# equality.  This works whether tau sits alone (the energy manifold's
# ``1/4 t2 v`` + ``1/2 t1t1 v``) or inside a larger term, because everything
# except the amplitude factor(s) must match exactly.
#
# The residue is coeff-blind: A1.3 only *detects* pairs.  A1.4 checks that the
# two coefficients are exactly consistent with tau = t2 + 1/2 P(t1 t1) before
# any rewrite is allowed.


def _residue_signature(term: AlgebraTerm) -> tuple[object, ...] | None:
    """tau-normalized, coeff/dummy-blind signature of a term, or None.

    Returns a hashable signature in which the amplitude factor(s) have been
    replaced by a single ``tau`` placeholder, so a matching t2-term and
    t1t1-term produce the *same* signature.  ``None`` if the term is neither a
    single-t2 nor a two-t1 doubles-block form.
    """
    t1_positions = [k for k, f in enumerate(term.factors) if f.name == "t1"]
    t2_positions = [k for k, f in enumerate(term.factors) if f.name == "t2"]

    if len(t2_positions) == 1 and not t1_positions:
        t2f = term.factors[t2_positions[0]]
        a, b, i, j = t2f.indices
        placeholder = tau(a, b, i, j)
        others = tuple(
            f for k, f in enumerate(term.factors) if k != t2_positions[0]
        )
    elif len(t1_positions) == 2 and not t2_positions:
        f1 = term.factors[t1_positions[0]]
        f2 = term.factors[t1_positions[1]]
        idx = list(f1.indices) + list(f2.indices)
        virs = [x for x in idx if x.space == "vir"]
        occs = [x for x in idx if x.space == "occ"]
        # The two t1s must together span a (vir,vir,occ,occ) block with the
        # occupieds pairing back to the same two virtuals (a<-i via f1, b<-j via
        # f2); anything else is not a tau half.
        if len(virs) != 2 or len(occs) != 2:
            return None
        placeholder = tau(f1.indices[0], f2.indices[0], f1.indices[1], f2.indices[1])
        others = tuple(
            f for k, f in enumerate(term.factors)
            if k not in (t1_positions[0], t1_positions[1])
        )
    else:
        return None

    residue = term.with_factors((placeholder,) + others)
    residue = relabel_term_dummies(residue)
    # Sort factors so factor order does not matter; indices within a factor are
    # positional and left as-is (tau/t2 antisym already canonical from source).
    factor_sig = tuple(
        (f.name, tuple((x.space, x.name) for x in f.indices))
        for f in sorted(residue.factors, key=lambda f: f.sort_key)
    )
    free_sig = tuple((x.space, x.name) for x in residue.free_indices)
    return (factor_sig, free_sig)


def _embedded_residue_signatures(
    term: AlgebraTerm,
) -> list[tuple[tuple[object, ...], str]]:
    """A3.0.a -- tau-residues for a term with arbitrary rest factors.

    Generalizes ``_residue_signature`` to the embedded case: a term may carry
    other amplitudes / integrals alongside the tau piece.  Enumerates every way
    to designate the tau block:

      * each single ``t2`` factor -> that t2 IS the tau (rest = all other
        factors), or
      * each unordered pair of ``t1`` factors that together span a
        (vir,vir,occ,occ) block -> that pair IS the tau's t1t1 half.

    Returns a list of ``(residue_signature, kind)`` where kind is ``"t2"`` or
    ``"t1t1"``.  Ambiguity (several candidates) yields several signatures; the
    exact firewall (A3.0.c) disambiguates.  Read-only; A1's single-tau
    ``_residue_signature`` is untouched.
    """
    from itertools import combinations

    results: list[tuple[tuple[object, ...], str]] = []

    def _residue_for(placeholder: Tensor, drop: set[int]) -> tuple[object, ...]:
        others = tuple(f for k, f in enumerate(term.factors) if k not in drop)
        residue = term.with_factors((placeholder,) + others)
        residue = relabel_term_dummies(residue)
        factor_sig = tuple(
            (f.name, tuple((x.space, x.name) for x in f.indices))
            for f in sorted(residue.factors, key=lambda f: f.sort_key)
        )
        free_sig = tuple((x.space, x.name) for x in residue.free_indices)
        return (factor_sig, free_sig)

    # t2-as-tau candidates
    for k, f in enumerate(term.factors):
        if f.name == "t2":
            a, b, i, j = f.indices
            results.append((_residue_for(tau(a, b, i, j), {k}), "t2"))

    # t1t1-pair-as-tau candidates
    t1_positions = [k for k, f in enumerate(term.factors) if f.name == "t1"]
    for k1, k2 in combinations(t1_positions, 2):
        f1, f2 = term.factors[k1], term.factors[k2]
        idx = list(f1.indices) + list(f2.indices)
        virs = [x for x in idx if x.space == "vir"]
        occs = [x for x in idx if x.space == "occ"]
        if len(virs) != 2 or len(occs) != 2:
            continue
        placeholder = tau(f1.indices[0], f2.indices[0], f1.indices[1], f2.indices[1])
        results.append((_residue_for(placeholder, {k1, k2}), "t1t1"))

    return results


@dataclass(frozen=True)
class EmbeddedTauMatch:
    """A3.0.b -- an embedded (t2-half, t1t1-half) pair sharing a residue.

    Like ``TauMatch`` but for terms carrying arbitrary rest factors.  ``residue``
    is the shared generalized residue signature the two halves matched on; the
    A3.0.c firewall uses it (plus the two members) to reconstruct and validate.
    """

    t2_index: int
    t1t1_index: int
    t2_coeff: Fraction
    t1t1_coeff: Fraction
    residue: tuple[object, ...]


def find_embedded_tau_matches(
    terms: Sequence[AlgebraTerm],
) -> list[EmbeddedTauMatch]:
    """A3.0.b -- detect embedded (t2-half, t1t1-half) pairs.

    Buckets every term's generalized residues (A3.0.a) by ``(residue, kind)``
    and pairs a t2-half with a t1t1-half on a shared residue.  Read-only.

    This is a LOOSE OVER-APPROXIMATION, unlike A1.3.  In the embedded case the
    generalized residue is coeff-blind and the pipeline's outer antisymmetriza-
    tion fractures coefficients, so a residue can be shared by terms that are
    NOT genuine tau halves -- empirically the reported "t1t1-halves" include
    3- and 4-t1 terms whose t1t1-pair candidate collides with a t2-half's
    residue, and the t1t1/t2 coefficient ratios come out inconsistent (2/3, 4/3,
    8/3, 3 -- not the bare case's clean 2).  A3.0.b therefore only PROPOSES
    pairs; the exact reconstruction firewall (A3.0.c) is load-bearing here and
    rejects the spurious majority.  If A3.0.c rejects (nearly) all of these,
    that is the empirical proof embedded-tau needs the full A3.2 index binding,
    not a residue heuristic.

    Conservative bucketing (unique-on-both-sides) still applies.  Bare pairs are
    a subset of what this reports, so run this OR A1's find_tau_matches, not
    both, to avoid double-collapse (A3.0.d handles that).
    """
    t2_by_residue: dict[tuple[object, ...], list[int]] = {}
    t1t1_by_residue: dict[tuple[object, ...], list[int]] = {}

    for k, term in enumerate(terms):
        for sig, kind in _embedded_residue_signatures(term):
            if kind == "t2":
                t2_by_residue.setdefault(sig, []).append(k)
            else:
                t1t1_by_residue.setdefault(sig, []).append(k)

    matches: list[EmbeddedTauMatch] = []
    for sig, t2_list in t2_by_residue.items():
        t1t1_list = t1t1_by_residue.get(sig)
        if not t1t1_list:
            continue
        if len(set(t2_list)) == 1 and len(set(t1t1_list)) == 1:
            k2, k1 = t2_list[0], t1t1_list[0]
            if k2 == k1:
                continue  # a single term is not paired with itself
            matches.append(
                EmbeddedTauMatch(
                    t2_index=k2,
                    t1t1_index=k1,
                    t2_coeff=terms[k2].coeff,
                    t1t1_coeff=terms[k1].coeff,
                    residue=sig,
                )
            )
    return matches


@dataclass(frozen=True)
class TauMatch:
    """A detected t2-half / t1t1-half pair that would collapse to one tau term.

    ``t2_index`` / ``t1t1_index`` are positions in the source term list.
    ``t2_coeff`` / ``t1t1_coeff`` are the two members' coefficients, carried
    for A1.4's exact tau = t2 + 1/2 P(t1 t1) consistency check.  A1.3 does NOT
    validate the coefficient relation and does NOT rewrite anything.
    """

    t2_index: int
    t1t1_index: int
    t2_coeff: Fraction
    t1t1_coeff: Fraction


def find_tau_matches(terms: Sequence[AlgebraTerm]) -> list[TauMatch]:
    """A1.3 -- detect (t2-term, t1t1-term) pairs sharing a tau residue.

    Read-only: returns ``TauMatch`` records, changes nothing.  A term is a
    t2-candidate if it has exactly one t2 and no t1; a t1t1-candidate if it has
    exactly two t1 (spanning a doubles block) and no t2.  Two candidates of
    opposite kind pair iff their tau-normalized residues are equal.

    Each t2-term is matched to at most one t1t1-term and vice versa (a residue
    identifies a unique tau contraction); duplicate residues on the same side
    are left unmatched for A1.4/A1.5 to handle conservatively.
    """
    t2_by_residue: dict[tuple[object, ...], list[int]] = {}
    t1t1_by_residue: dict[tuple[object, ...], list[int]] = {}

    for k, term in enumerate(terms):
        has_t2 = any(f.name == "t2" for f in term.factors)
        has_t1 = any(f.name == "t1" for f in term.factors)
        sig = _residue_signature(term)
        if sig is None:
            continue
        if has_t2 and not has_t1:
            t2_by_residue.setdefault(sig, []).append(k)
        elif has_t1 and not has_t2:
            t1t1_by_residue.setdefault(sig, []).append(k)

    matches: list[TauMatch] = []
    for sig, t2_list in t2_by_residue.items():
        t1t1_list = t1t1_by_residue.get(sig)
        if not t1t1_list:
            continue
        # Unique residue on both sides -> a clean pair. Ambiguous multiplicities
        # are deferred (not paired) to keep A1.3 conservative.
        if len(t2_list) == 1 and len(t1t1_list) == 1:
            k2, k1 = t2_list[0], t1t1_list[0]
            matches.append(
                TauMatch(
                    t2_index=k2,
                    t1t1_index=k1,
                    t2_coeff=terms[k2].coeff,
                    t1t1_coeff=terms[k1].coeff,
                )
            )
    return matches


# ---------------------------------------------------------------------------
# A1.4 -- exact-coefficient validation (the correctness firewall)
# ---------------------------------------------------------------------------
#
# A1.3 pairs on *structure*; it does not check that the two coefficients are
# consistent with the tau definition.  Before A1.5 is ever allowed to collapse
# a pair, A1.4 proves the collapse is lossless: take the t2 member, promote its
# t2 factor to tau (this IS the intended collapsed term), then symbolically
# expand tau -> t2 + 1/2 P(t1 t1) and canonicalize.  The expansion must
# reproduce the two source members *exactly* (Fraction coefficient equality),
# using the pipeline's own canonicalizer as the equality oracle -- so nothing
# about the collapse depends on assuming a fixed t1t1:t2 ratio.


def _canonical_fixed_point(term: AlgebraTerm) -> AlgebraTerm:
    """Canonicalize to a fixed point (coeff/sign included).

    ``canonicalize_term`` is not idempotent when its dummy relabeling reorders
    an antisymmetric factor's indices: the antisymmetry sign is normalized
    *before* the relabel, so a relabel-induced swap (e.g. v(a,b,i,j) ->
    v(a,b,j,i)) is left un-flipped until the next pass.  Re-applying until
    stable folds that residual sign into the coefficient.  Bounded to a few
    iterations -- it converges in two for the cases here.
    """
    prev = term
    for _ in range(4):
        cur = canonicalize_term(prev)
        if cur.factors == prev.factors and cur.coeff == prev.coeff:
            return cur
        prev = cur
    return prev


def _canonical_key(term: AlgebraTerm) -> tuple[object, ...]:
    """Structure-only canonical key (coeff excluded) for exact comparison."""
    c = _canonical_fixed_point(term)
    return (
        tuple(
            (f.name, tuple((x.space, x.name) for x in f.indices))
            for f in c.factors
        ),
        tuple((x.space, x.name) for x in c.free_indices),
        tuple((x.space, x.name) for x in c.summed_indices),
    )


def _expand_tau_term(tau_term: AlgebraTerm) -> list[AlgebraTerm]:
    """Expand a single tau factor into the two terms the pipeline would write.

    ``tau_term`` must contain exactly one ``tau`` factor.  Returns:

      * the t2 piece  ``coeff * t2_coeff * t2(a,b,i,j) * rest``, and
      * the t1t1 piece ``coeff * written_t1t1_weight * t1(a,i) t1(b,j) * rest``

    i.e. tau = t2 + (written) t1(a,i)t1(b,j), matching how ccgen records the
    single antisymmetrized t1t1 representative (see TauSpec).  Not canonicalized
    here -- the caller canonicalizes and compares.
    """
    pos = next(k for k, f in enumerate(tau_term.factors) if f.name == TAU_NAME)
    tf = tau_term.factors[pos]
    a, b, i, j = tf.indices
    others = tuple(f for k, f in enumerate(tau_term.factors) if k != pos)
    c = tau_term.coeff

    def _term(coeff: Fraction, amp_factors: tuple[Tensor, ...]) -> AlgebraTerm:
        return AlgebraTerm(
            coeff=coeff,
            factors=amp_factors + others,
            free_indices=tau_term.free_indices,
            summed_indices=tau_term.summed_indices,
            connected=tau_term.connected,
            provenance=tau_term.provenance,
        )

    return [
        _term(c * TAU_SPEC.t2_coeff, (make_t2(a, b, i, j),)),
        _term(c * TAU_SPEC.written_t1t1_weight, (t1(a, i), t1(b, j))),
    ]


def _tau_term_from_t2(t2_member: AlgebraTerm) -> AlgebraTerm:
    """The collapsed tau-term for a t2 member: swap its t2 factor for tau.

    Coefficient is unchanged (the tau-term coeff equals the t2 member's coeff
    by construction -- the t2 half of tau carries weight t2_coeff = 1).
    """
    pos = next(k for k, f in enumerate(t2_member.factors) if f.name == "t2")
    tf = t2_member.factors[pos]
    a, b, i, j = tf.indices
    others = tuple(f for k, f in enumerate(t2_member.factors) if k != pos)
    return AlgebraTerm(
        coeff=t2_member.coeff,
        factors=(tau(a, b, i, j),) + others,
        free_indices=t2_member.free_indices,
        summed_indices=t2_member.summed_indices,
        connected=t2_member.connected,
        provenance=t2_member.provenance,
    )


def validate_tau_match(
    terms: Sequence[AlgebraTerm],
    match: TauMatch,
) -> bool:
    """A1.4 -- is ``match`` an EXACT tau = t2 + 1/2 P(t1 t1) collapse?

    Promotes the t2 member's t2 factor to tau (coeff = t2 member's coeff),
    expands, canonicalizes, and merges.  The merged expansion must equal the
    two source members exactly: one canonical piece matching the t2 member
    (same structure AND coefficient) and one matching the t1t1 member.  Any
    mismatch -- wrong coefficient, extra/missing piece -- returns False, so a
    structurally-plausible but numerically-inconsistent pair is never
    collapsed.
    """
    t2_member = terms[match.t2_index]
    t1t1_member = terms[match.t1t1_index]

    # Build the intended collapsed tau-term from the t2 member.
    tau_term = _tau_term_from_t2(t2_member)

    # Merge the expansion by the coeff-blind canonical key.  merge_like_terms
    # is NOT enough here: the two antisymmetrizer t1t1 pieces are structurally
    # equal but relabel to different raw factor tuples (the P-swap introduces
    # occ indices in the opposite order), so they land in different
    # _term_signature buckets.  _canonical_key already accounts for that, so we
    # bucket on it and sum the signed coefficients.
    expanded_by_key: dict[tuple[object, ...], Fraction] = {}
    for t in _expand_tau_term(tau_term):
        ct = _canonical_fixed_point(t)  # folds antisymmetry sign into ct.coeff
        key = _canonical_key(ct)
        expanded_by_key[key] = expanded_by_key.get(key, Fraction(0)) + ct.coeff
    # Drop any piece that cancelled to zero.
    expanded_by_key = {k: c for k, c in expanded_by_key.items() if c != 0}
    want = {
        _canonical_key(t2_member): t2_member.coeff,
        _canonical_key(t1t1_member): t1t1_member.coeff,
    }
    if len(want) != 2:
        return False  # the two members canonicalize to the same structure
    return expanded_by_key == want


def _tau_terms_from_t2_by_residue(
    t2_member: AlgebraTerm,
    residue: tuple[object, ...],
) -> list[AlgebraTerm]:
    """Every tau-term for a t2-half whose t2 factor matches ``residue``.

    In the embedded case a term may hold several t2 factors; only the one whose
    designation produced ``residue`` is the tau.  Returns a tau-term per
    matching t2 position (usually one; several only under residue collision).
    """
    out: list[AlgebraTerm] = []
    for k, f in enumerate(t2_member.factors):
        if f.name != "t2":
            continue
        a, b, i, j = f.indices
        others = tuple(g for gk, g in enumerate(t2_member.factors) if gk != k)
        candidate = AlgebraTerm(
            coeff=t2_member.coeff,
            factors=(tau(a, b, i, j),) + others,
            free_indices=t2_member.free_indices,
            summed_indices=t2_member.summed_indices,
            connected=t2_member.connected,
            provenance=t2_member.provenance,
        )
        # Keep only the t2 designation that produced `residue`.
        if _tau_term_matches_residue(candidate, residue):
            out.append(candidate)
    return out


def _tau_term_matches_residue(
    tau_term: AlgebraTerm,
    residue: tuple[object, ...],
) -> bool:
    """Does this tau-term's (relabeled) factor/free signature equal ``residue``?"""
    relab = relabel_term_dummies(tau_term)
    factor_sig = tuple(
        (f.name, tuple((x.space, x.name) for x in f.indices))
        for f in sorted(relab.factors, key=lambda f: f.sort_key)
    )
    free_sig = tuple((x.space, x.name) for x in relab.free_indices)
    return (factor_sig, free_sig) == residue


def validate_embedded_tau_match(
    terms: Sequence[AlgebraTerm],
    match: EmbeddedTauMatch,
) -> bool:
    """A3.0.c -- is an embedded pair an EXACT tau collapse?

    The load-bearing firewall for the embedded case.  Reconstructs the tau-term
    from the t2-half's residue-matching t2 factor, expands tau = t2 + written
    t1t1, canonicalizes, and requires the expansion to reproduce the two paired
    members EXACTLY (structure + Fraction coefficient).  Rejects any pair whose
    coefficients do not reconstruct -- which, given A3.0.b's loose bucketing, is
    the spurious majority.

    A pair validates iff the tau-term expands to exactly two canonical pieces,
    one matching the t2-half and one the t1t1-half, with identical coefficients.
    """
    t2_member = terms[match.t2_index]
    t1t1_member = terms[match.t1t1_index]

    tau_terms = _tau_terms_from_t2_by_residue(t2_member, match.residue)
    if not tau_terms:
        return False

    want = {
        _canonical_key(t2_member): t2_member.coeff,
        _canonical_key(t1t1_member): t1t1_member.coeff,
    }
    if len(want) != 2:
        return False  # both members canonicalize to the same structure

    # Any residue-matching tau designation that reconstructs exactly validates.
    for tau_term in tau_terms:
        expanded_by_key: dict[tuple[object, ...], Fraction] = {}
        for t in _expand_tau_term(tau_term):
            ct = _canonical_fixed_point(t)
            key = _canonical_key(ct)
            expanded_by_key[key] = expanded_by_key.get(key, Fraction(0)) + ct.coeff
        expanded_by_key = {k: c for k, c in expanded_by_key.items() if c != 0}
        if expanded_by_key == want:
            return True
    return False


# ---------------------------------------------------------------------------
# A1.5 -- the tau collapse rewrite (term list -> term list)
# ---------------------------------------------------------------------------


def apply_tau(terms: Sequence[AlgebraTerm]) -> list[AlgebraTerm]:
    """A1.5 -- collapse every EXACT tau pair into a single tau term.

    Detects candidate pairs (A1.3), keeps only those that pass the exact
    coefficient firewall (A1.4), and replaces each surviving pair's two members
    with the single tau-term (t2 factor -> tau, coeff unchanged).  Every term
    not part of a validated pair is passed through untouched, in its original
    order.  A pair that fails validation is left as its two separate members --
    the rewrite never changes algebra it cannot prove lossless.

    Idempotent: after one pass no bare-t2/bare-t1t1 pair remains for the same
    residue, so a second pass finds nothing to collapse.
    """
    terms = list(terms)
    validated = [m for m in find_tau_matches(terms) if validate_tau_match(terms, m)]

    # Indices consumed by a validated collapse; the tau-term takes the t2 slot.
    replaced: dict[int, AlgebraTerm] = {}
    dropped: set[int] = set()
    for m in validated:
        replaced[m.t2_index] = _tau_term_from_t2(terms[m.t2_index])
        dropped.add(m.t1t1_index)

    out: list[AlgebraTerm] = []
    for k, term in enumerate(terms):
        if k in dropped:
            continue
        out.append(replaced.get(k, term))
    return out


# ---------------------------------------------------------------------------
# A1.6 -- offline algebraic-equivalence gate
# ---------------------------------------------------------------------------
#
# The load-bearing guarantee before A1.5 is ever wired into codegen: collapsing
# tau pairs must not change the algebra.  We prove it structurally, with no
# numeric run: expand every tau back to t2 + t1t1 and canonically merge the
# whole list, then check it equals the original list as a coefficient-keyed
# multiset.  ``apply_tau`` is algebra-preserving iff
# ``canonical_multiset(expand_all_tau(apply_tau(R))) == canonical_multiset(R)``
# for every residual R.


def expand_all_tau(terms: Sequence[AlgebraTerm]) -> list[AlgebraTerm]:
    """Expand every tau factor back to t2 + t1t1, canonicalize, and merge.

    Terms without a tau factor pass through (still canonicalized+merged so the
    output is directly comparable to a canonical_multiset of the original).
    """
    pieces: list[AlgebraTerm] = []
    for term in terms:
        if any(f.name == TAU_NAME for f in term.factors):
            pieces.extend(_expand_tau_term(term))
        else:
            pieces.append(term)
    return pieces


def canonical_multiset(terms: Sequence[AlgebraTerm]) -> dict[tuple[object, ...], Fraction]:
    """Structure -> summed-coefficient map, the order-blind identity of a sum.

    Two term lists denote the same algebraic expression iff their
    canonical_multisets are equal.  Uses the fixed-point canonical key (folds
    antisymmetry signs into coefficients) and drops any structure whose
    coefficients cancel to zero.
    """
    acc: dict[tuple[object, ...], Fraction] = {}
    for term in terms:
        ct = _canonical_fixed_point(term)
        key = _canonical_key(ct)
        acc[key] = acc.get(key, Fraction(0)) + ct.coeff
    return {k: c for k, c in acc.items() if c != 0}


def tau_rewrite_preserves_algebra(terms: Sequence[AlgebraTerm]) -> bool:
    """A1.6 -- does collapsing then re-expanding tau reproduce the original?

    Returns True iff ``apply_tau(terms)`` is algebraically identical to
    ``terms``: expand its tau factors back out and compare canonical multisets.
    This is the offline safety proof for wiring A1.5 into generation.
    """
    collapsed = apply_tau(terms)
    reexpanded = expand_all_tau(collapsed)
    return canonical_multiset(reexpanded) == canonical_multiset(terms)


# ---------------------------------------------------------------------------
# A1.7 -- tau as a materialized intermediate the existing emitter can build
# ---------------------------------------------------------------------------


def tau_intermediate_spec(usage_count: int, usage_targets: tuple[str, ...]):
    """Build the IntermediateSpec for tau so the Planck emitter materializes it.

    tau rides the same intermediate machinery as the CSE "W_*" tensors: a
    ``build_tau(...)`` function lowered from its definition terms, referenced
    once per kernel.  Definition (over the canonical (a,b,i,j) block):

        tau(a,b,i,j) = t2(a,b,i,j) + written_t1t1_weight * t1(a,i) t1(b,j)

    Imported lazily to avoid a hard dependency from the tau module onto the
    optimization.intermediates dataclass.
    """
    from .intermediates import IntermediateSpec

    from ..indices import make_vir, make_occ

    a = make_vir("a", dummy=True)
    b = make_vir("b", dummy=True)
    i = make_occ("i", dummy=True)
    j = make_occ("j", dummy=True)
    block = (a, b, i, j)

    t2_term = AlgebraTerm(
        coeff=TAU_SPEC.t2_coeff,
        factors=(make_t2(a, b, i, j),),
        free_indices=block,
        summed_indices=(),
        connected=True,
    )
    t1t1_term = AlgebraTerm(
        coeff=TAU_SPEC.written_t1t1_weight,
        factors=(t1(a, i), t1(b, j)),
        free_indices=block,
        summed_indices=(),
        connected=True,
    )
    return IntermediateSpec(
        name=TAU_NAME,
        indices=block,
        definition_terms=(t2_term, t1t1_term),
        usage_count=usage_count,
        index_space_sig="vvoo",
        usage_targets=usage_targets,
    )


def factorize_tau_equations(
    equations: dict[str, list[AlgebraTerm]],
) -> tuple[dict[str, list[AlgebraTerm]], object | None]:
    """Apply tau across all manifolds; return (new_equations, tau_spec_or_None).

    Runs ``apply_tau`` per manifold.  If any manifold collapsed a tau pair, the
    returned spec is the tau IntermediateSpec (with an aggregate usage count);
    otherwise ``None`` and the equations are returned unchanged (structurally).
    The A1.6 algebra-preservation gate must have been satisfied by construction
    (apply_tau only collapses validated pairs), so this is safe to wire in.
    """
    new_eqs: dict[str, list[AlgebraTerm]] = {}
    usage_count = 0
    usage_targets: list[str] = []
    for target, terms in equations.items():
        collapsed = apply_tau(terms)
        new_eqs[target] = collapsed
        n_tau = sum(
            1 for t in collapsed if any(f.name == TAU_NAME for f in t.factors)
        )
        if n_tau:
            usage_count += n_tau
            usage_targets.append(target)

    if usage_count == 0:
        return equations, None
    return new_eqs, tau_intermediate_spec(usage_count, tuple(usage_targets))
