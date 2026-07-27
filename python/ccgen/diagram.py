"""Canonical diagram strings for coupled-cluster term enumeration (D1).

ccgen currently enumerates *algebraic terms* by Wick contraction and removes
duplicates afterwards by canonicalization.  MRCC/CFOUR instead enumerate
*diagram topologies* in a form that is canonical by construction, so duplicates
are never generated at all (Kallay & Surjan, JCP 113, 1359 (2000); JCP 115,
2945 (2001)).  The measured waste of the term path grows with rank -- 16x at
CCSD, 78x at CCSDT -- which is what motivates the diagram representation.

This module is step D1: the encoding and its canonical form, nothing else.

  D1.0  DiagramString      -- the integer-triplet encoding, as data (this file)
  D1.1  is_wellformed      -- the validity predicate
  D1.2  canonical / key    -- canonical form and the equivalence claim
  D1.3  to_string/from_str -- compact round-trippable text form
  D1.4  ANCHORS            -- hand-derived fixtures, D2's oracle

Later steps (D2 enumeration, D3 the AlgebraTerm bridge) build on this.  Nothing
here is imported by the generator, so generated code is untouched.  See
``docs/CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md`` for the full plan.

ponytail: canonical form is a sort, not a graph-isomorphism search -- that is
the whole point of the encoding.  If a future step reaches for isomorphism
machinery here, the encoding is wrong, not the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Sequence

from .indices import Index, OCC_POOL, VIR_POOL, make_occ, make_vir

# A two-body Hamiltonian vertex carries four lines, so at most four cluster
# operators can connect to it in one diagram. Kallay-Surjan's generation
# algorithm iterates "for each l <= 4" for exactly this reason
# (arXiv:2409.06759 Fig. 3).
MAX_T_OPERATORS = 4


@dataclass(frozen=True)
class DiagramString:
    """A coupled-cluster diagram in Kallay-Surjan integer-triplet form.

    One triplet ``(mu1, mu2, mu3)`` per cluster operator T_n in the diagram:

      ``mu1``  excitation level of the operator (the *n* of T_n), >= 1
      ``mu2``  number of its lines internally contracted with the Hamiltonian
               vertex, in ``0 .. 2 * mu1`` (an operator of level mu1 carries
               mu1 particle and mu1 hole lines)
      ``mu3``  how many of those internal connections are particle lines,
               in ``0 .. mu1``

    ``bra_level`` / ``ket_level`` are the excitation levels of the projection
    manifold and of the state being projected on.

    The tuple order of ``t_ops`` is NOT normalized by the constructor -- a raw
    diagram and its canonical form must stay distinguishable, otherwise the
    D1.2 canonicalization claim is untestable.  Use :func:`canonical`.
    """

    t_ops: tuple[tuple[int, int, int], ...]
    bra_level: int
    ket_level: int


def is_wellformed(ds: DiagramString) -> bool:
    """Whether *ds* satisfies the structural bounds of the encoding.

    The bounds are stated, not guessed -- each follows from what the triplet
    counts:

      * ``mu1 >= 1``       -- T_0 is not a cluster operator.
      * ``1 <= mu2 <= 2 * mu1``  -- a level-mu1 operator has mu1 particle plus
        mu1 hole lines, so at most 2*mu1 of them can be internal; and at least
        ONE must be, or the operator is disconnected from the Hamiltonian and
        the diagram is not a connected diagram at all.
      * ``0 <= mu3 <= min(mu1, mu2)`` -- internal *particle* lines are drawn
        from the mu1 particle lines, and are a subset of the mu2 internal ones.
      * ``bra_level``, ``ket_level >= 0`` -- a manifold level, 0 being the
        reference.
      * ``len(t_ops) <= MAX_T_OPERATORS`` -- a two-body Hamiltonian vertex has
        four lines, so it can connect at most four cluster operators.

    These are the bounds of the Kallay-Surjan generation algorithm as
    reproduced in Brandejs et al., arXiv:2409.06759 Fig. 3 ("mu2: all
    l-partitions of the number of internal lines between 1 and 2*mu1", "mu3:
    ... between 0 and min(mu1, mu2)", "for each l <= 4").

    A bound that is too tight silently rejects real diagrams in D2; that shows
    up there as a missing-count failure rather than as a wrong answer, which is
    why the bounds are asserted here rather than left implicit.
    """
    if ds.bra_level < 0 or ds.ket_level < 0:
        return False
    if len(ds.t_ops) > MAX_T_OPERATORS:
        return False
    for triplet in ds.t_ops:
        if len(triplet) != 3:
            return False
        mu1, mu2, mu3 = triplet
        if mu1 < 1:
            return False
        if mu2 < 1 or mu2 > 2 * mu1:
            return False
        if mu3 < 0 or mu3 > min(mu1, mu2):
            return False
    return True


def canonical(ds: DiagramString) -> DiagramString:
    """Return the canonical form of *ds*: triplets sorted ascending.

    The cluster operators in a diagram are unordered -- ``T2`` then ``T1`` is
    the same topology as ``T1`` then ``T2`` -- so the tuple order carries no
    information and sorting removes exactly that freedom.  This is the whole
    content of the representation: equivalence becomes tuple equality.

    Idempotent by construction (sorting is), and pinned as such by the tests.
    That is worth one assertion because ``canonicalize.canonicalize_term`` on
    the term path is *not* idempotent -- it needs a fixed-point loop (see the
    tau/dressing work).  The diagram layer does not inherit that trap.
    """
    return DiagramString(
        t_ops=tuple(sorted(ds.t_ops)),
        bra_level=ds.bra_level,
        ket_level=ds.ket_level,
    )


def key(ds: DiagramString) -> tuple[object, ...]:
    """Hashable equivalence key for *ds*.

    Two diagrams are equivalent iff their keys are equal.  Use this as the
    dict/set key when enumerating (D2), so duplicates collide instead of being
    generated and discarded.
    """
    c = canonical(ds)
    return (c.bra_level, c.ket_level, c.t_ops)


def equivalent(a: DiagramString, b: DiagramString) -> bool:
    """Whether two diagrams denote the same topology."""
    return key(a) == key(b)


# ── D1.3 text form ───────────────────────────────────────────────────

def to_string(ds: DiagramString) -> str:
    """Render *ds* as ``"bra:ket|mu1,mu2,mu3;mu1,mu2,mu3"``.

    Vertices are emitted in the order held, NOT canonicalized -- so the text
    form round-trips exactly and ``to_string(canonical(x))`` is the thing to
    compare when a canonical rendering is wanted.  An empty vertex list renders
    as a trailing bar (``"2:0|"``).
    """
    body = ";".join(f"{a},{b},{c}" for a, b, c in ds.t_ops)
    return f"{ds.bra_level}:{ds.ket_level}|{body}"


def from_string(s: str) -> DiagramString:
    """Parse the :func:`to_string` form.  Raises ``ValueError`` if malformed.

    Parses shape only -- the result may still be structurally invalid, which is
    :func:`is_wellformed`'s job to report.  Keeping the two separate means a
    test fixture can be written down and *then* checked, rather than the parser
    silently deciding what is a legal diagram.
    """
    head, sep, body = s.partition("|")
    if not sep:
        raise ValueError(f"missing '|' separator in diagram string {s!r}")
    bra, sep2, ket = head.partition(":")
    if not sep2:
        raise ValueError(f"missing ':' in manifold levels of {s!r}")
    try:
        bra_level, ket_level = int(bra), int(ket)
    except ValueError as exc:
        raise ValueError(f"non-integer manifold level in {s!r}") from exc

    t_ops: list[tuple[int, int, int]] = []
    if body:
        for chunk in body.split(";"):
            parts = chunk.split(",")
            if len(parts) != 3:
                raise ValueError(
                    f"vertex {chunk!r} in {s!r} is not a 3-tuple"
                )
            try:
                t_ops.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError as exc:
                raise ValueError(
                    f"non-integer vertex component in {chunk!r}"
                ) from exc
    return DiagramString(tuple(t_ops), bra_level, ket_level)


# ── D1.4 anchor fixtures ─────────────────────────────────────────────
#
# Hand-derived diagrams, named by topology, used as D2's enumeration oracle.
#
# These are anchored to what the *existing* Wick path actually produces: each
# entry's vertex multiset corresponds to a distinct (factor-name multiset,
# summed-occ, summed-vir) class of the generated residual.  That class count is
# the countable, checkable thing D2 must reproduce -- for CCD doubles the term
# path emits 35 terms falling into 7 such classes, for CCSD doubles 123 into
# 19.  The 35-vs-7 gap is precisely the duplication the diagram representation
# is meant to remove.
#
# ponytail: anchored to observed generator output rather than transcribed from
# the papers. That makes them a *consistency* oracle (diagram path vs term
# path), not an independent correctness oracle -- which is exactly what D3's
# multiset-equality gate needs. Independent validation against Kallay-Surjan
# is a separate, and still-open, check.

ANCHORS: dict[str, DiagramString] = {
    # --- CCD ---
    # Bare ERI, no cluster operators: <ij||ab>.
    "ccd_doubles_bare_v": DiagramString((), 2, 0),
    # Particle-particle ladder: 1/2 sum_cd t2(c,d,i,j) <cd||ab>.
    "ccd_doubles_pp_ladder": DiagramString(((2, 2, 2),), 2, 0),
    # Hole-hole ladder: 1/2 sum_kl t2(a,b,k,l) <ij||kl>.
    "ccd_doubles_hh_ladder": DiagramString(((2, 2, 0),), 2, 0),
    # Ring / particle-hole: sum_kc t2(c,b,i,k) <jc||ka>.
    "ccd_doubles_ring": DiagramString(((2, 2, 1),), 2, 0),
    # Fock contractions: one internal line, particle or hole.
    "ccd_doubles_fock_particle": DiagramString(((2, 1, 1),), 2, 0),
    "ccd_doubles_fock_hole": DiagramString(((2, 1, 0),), 2, 0),
    # Quadratic t2*t2*v.
    "ccd_doubles_t2t2": DiagramString(((2, 2, 1), (2, 2, 1)), 2, 0),
    # CCD energy: 1/4 sum_ijab t2(a,b,i,j) <ij||ab>.
    "ccd_energy": DiagramString(((2, 4, 2),), 0, 0),
    # --- CCSD additions ---
    "ccsd_energy_t1t1": DiagramString(((1, 2, 1), (1, 2, 1)), 0, 0),
    "ccsd_energy_f_t1": DiagramString(((1, 2, 1),), 0, 0),
    "ccsd_singles_bare_f": DiagramString((), 1, 0),
    "ccsd_singles_t1_v": DiagramString(((1, 2, 1),), 1, 0),
    "ccsd_singles_t2_v": DiagramString(((2, 3, 2),), 1, 0),
    "ccsd_doubles_t1_v": DiagramString(((1, 1, 1),), 2, 0),
    # Corrected in D2.3: the originally hand-written ((1,2,1),(2,2,1)) is NOT a
    # real diagram -- it sends 2 particle + 2 hole internal lines into a
    # two-body vertex that has only 2 slots of each. The closure filter caught
    # it, and the term path confirms it never appears. This is a genuine t1*t2
    # doubles diagram.
    "ccsd_doubles_t1t2": DiagramString(((1, 1, 0), (2, 2, 1)), 2, 0),
}


def anchors_for(prefix: str) -> dict[str, DiagramString]:
    """The :data:`ANCHORS` whose name starts with *prefix* (e.g. ``"ccd_"``)."""
    return {k: v for k, v in ANCHORS.items() if k.startswith(prefix)}


# ── D2.0 topology observables ────────────────────────────────────────
#
# Two keys for grouping generated AlgebraTerms by topology.  They exist to give
# D2's enumeration a countable target measured from the *existing* term path.
#
# The coarse key was written first and is a genuine UNDERCOUNT: it merges
# diagrams that differ in how externals are distributed over the factors.  For
# CCD doubles the 24 quadratic t2*t2*v terms all share one coarse key, yet they
# are not one topology -- t2(a,b,k,l)*t2(c,d,i,j) (each amplitude fully
# internal on one side) and t2(a,c,i,k)*t2(b,d,j,l) (each carrying two
# externals) are different diagrams.  The fine key separates them.
#
# Both are kept: the gap between them is itself the thing being measured, and a
# future change that moves either number should have to say so out loud.


def coarse_topology_key(term) -> tuple[object, ...]:
    """Group by (factor names, #summed occ, #summed vir).

    A LOWER BOUND on the diagram count -- see the module note above.  Retained
    because it is the number the D1.4 anchors were first pinned against.
    """
    from collections import Counter

    names = tuple(sorted(f.name for f in term.factors))
    spaces = Counter(i.space for i in term.summed_indices)
    return (names, spaces["occ"], spaces["vir"])


def topology_signature(term) -> tuple[object, ...]:
    """Group by the per-factor external-index signature.

    For each factor, record ``(name, #external-occ, #external-vir)``; the
    diagram is the sorted multiset of those.  This distinguishes factors by how
    many external lines they carry, which is what the coarse key loses.

    Still an approximation -- it does not track *which* external attaches
    where, so it can in principle merge two genuinely distinct diagrams.  It is
    therefore also a lower bound, just a much tighter one.  D3's multiset gate,
    not this, is the authoritative equality check.
    """
    free = set(term.free_indices)
    sig = []
    for f in term.factors:
        ext_occ = sum(1 for i in f.indices if i in free and i.space == "occ")
        ext_vir = sum(1 for i in f.indices if i in free and i.space == "vir")
        sig.append((f.name, ext_occ, ext_vir))
    return tuple(sorted(sig))


def exact_topology_key(term) -> tuple[object, ...]:
    """Exact, label-independent topology key for a generated term.

    Unlike :func:`topology_signature` this records *which* index sits in which
    slot, so two terms share a key only if they are the same labelled graph up
    to renaming of summed indices.  Construction:

      1. Relabel summed indices canonically by first appearance (reusing
         ``canonicalize.relabel_term_dummies``, the same normalization the tau
         and dressing passes rely on).
      2. Give every index a role.  An external is ``("e", space)`` -- its
         *space* but NOT which external it is; a summed index is
         ``("s", space, canonical-name)``, so shared dummies still tie factors
         together.
      3. Emit the sorted multiset of ``(factor name, role-tuple)``.

    Sorting the factors makes the key blind to factor ORDER (a product is
    commutative) while keeping it sensitive to factor CONTENT.

    External identity is deliberately dropped.  Terms related by the residual's
    antisymmetrizer -- ``P(ij)P(ab)`` and friends -- are ONE diagram whose
    externals have been permuted, and a key that records external position
    splits them.  Measured: keying on position gives 34 classes for the 35 CCD
    doubles terms (near-injective, useless), because the 4 ring terms
    ``t2(c,b,i,k) v(j,c,k,a)`` etc. differ only in which external landed where.
    Dropping position collapses those to the single ring diagram.

    This is the tightest *useful* key available without a graph-isomorphism
    search, and it is an OVERCOUNT rather than an undercount -- the opposite
    bias to the two counting keys above.  Known limitation: when a term repeats
    a factor (two ``t2``s), exchanging the two copies gives the same diagram but
    permutes the shared dummy names, and sorting cannot identify those.
    Measured on CCD doubles this splits one quadratic diagram in two, giving 12
    classes where hand-derivation gives 11.  Canonicalizing over the
    permutations of repeated factors is exactly the isomorphism search the
    diagram representation exists to avoid, so it is deliberately NOT done here.

    Use it as an upper bound and the counting keys as lower bounds; D3's
    multiset gate remains the authority.
    """
    from .canonicalize import relabel_term_dummies

    relabeled = relabel_term_dummies(term)
    free = set(relabeled.free_indices)

    def role(idx) -> tuple[object, ...]:
        if idx in free:
            return ("e", idx.space)
        return ("s", idx.space, idx.name)

    def factor_key(f) -> tuple[object, ...]:
        roles = [role(i) for i in f.indices]
        # Sort roles WITHIN each antisymmetry group: t2(c,b,...) and
        # t2(a,c,...) write the same {summed, external} pair in two orders and
        # are the same diagram up to the sign antisymmetry already carries.
        for group in getattr(f, "antisym_groups", ()) or ():
            slots = sorted(group)
            for slot, value in zip(slots, sorted(roles[s] for s in slots)):
                roles[slot] = value
        return (f.name, tuple(roles))

    return tuple(sorted(factor_key(f) for f in relabeled.factors))


# ── D2.1 vertex enumeration ──────────────────────────────────────────

def enumerate_vertices(mu1: int) -> list[tuple[int, int, int]]:
    """All legal ``(mu1, mu2, mu3)`` triplets for one T operator of level *mu1*.

    Direct transcription of the inner two loops of the Kallay-Surjan algorithm
    (arXiv:2409.06759 Fig. 3): mu2 over ``1 .. 2*mu1``, mu3 over
    ``0 .. min(mu1, mu2)``.  Returned in ascending order.

    The count is closed-form -- ``sum_{mu2=1}^{2*mu1} (min(mu1, mu2) + 1)`` --
    which the tests check independently rather than re-deriving the loop.
    """
    if mu1 < 1:
        raise ValueError(f"cluster operator level must be >= 1, got {mu1}")
    return [
        (mu1, mu2, mu3)
        for mu2 in range(1, 2 * mu1 + 1)
        for mu3 in range(0, min(mu1, mu2) + 1)
    ]


def mu1_sum_range(bra_level: int, max_rank: int, n_ops: int) -> range:
    """Legal totals of ``sum(mu1)`` for a diagram projected on *bra_level*.

    The algorithm's outermost mu1 loop reads "all positive l-tuples with sum
    between ``k-2``, ``min(k+2, n)``" -- k the projection level, n the maximum
    excitation level of an *individual* T operator (arXiv:2409.06759 Fig. 3).

    Taken literally, ``n`` would bound the *sum*, which is wrong: for CCSD
    (k=2, n=2) it gives sum in 1..2, yet the real CCSD doubles residual has 99
    of its 123 terms at sum(mu1) = 3 or 4 (t1*t2, t1*t1*t2, t1^4, ...).  Since
    each of the ``l`` operators is individually capped at ``n``, the total is
    capped at ``l * n``.  We read the paper's ``n`` as that per-operator cap and
    apply it as ``n_ops * max_rank``.

    The +/-2 window is the reach of a two-body Hamiltonian: it changes the
    excitation level by at most two in either direction, so a diagram whose
    cluster operators sum outside that window cannot close onto the bra.  Note
    the window is a *necessary* condition on the sum, not a sufficient one --
    D2.3's closure filter is what actually enforces that the diagram closes.

    Lower limit is clamped at 1: a connected diagram has at least one T
    operator (sum 0 is the bare Hamiltonian, enumerated separately).
    """
    if n_ops < 1:
        raise ValueError(f"a diagram needs at least one T operator, got {n_ops}")
    lo = max(1, bra_level - 2)
    hi = min(bra_level + 2, n_ops * max_rank)
    return range(lo, hi + 1)


# ── D2.2 multi-vertex combination ────────────────────────────────────

def enumerate_candidates(
    ranks: Sequence[int],
    bra_level: int,
    ket_level: int = 0,
    max_operators: int = MAX_T_OPERATORS,
) -> list[DiagramString]:
    """All canonical candidate diagrams over cluster operators of *ranks*.

    Emits **only sorted vertex multisets**, via
    ``combinations_with_replacement`` over the pooled legal vertices.  That is
    the whole point of the representation: the paper's "keep the candidate if
    its integer triples are mutually sorted ascendingly" becomes a generator
    that never proposes the unsorted permutations in the first place, so
    duplicates are not produced and then discarded.

    These are *candidates*, not the final diagram set.  Each satisfies the
    per-vertex bounds (D2.1) and the sum(mu1) window, but nothing here checks
    that the diagram actually closes -- that internal lines pair up, that the
    external count matches ``bra_level``, or that the Hamiltonian vertex has
    enough lines to reach every operator.  D2.3 supplies that filter, so this
    function deliberately over-generates.

    ``ranks`` is the method's cluster ranks, e.g. ``[1, 2]`` for CCSD.
    """
    if max_operators < 0:
        raise ValueError(f"max_operators must be >= 0, got {max_operators}")
    if any(r < 1 for r in ranks):
        raise ValueError(f"cluster ranks must be >= 1, got {sorted(ranks)}")

    pool: list[tuple[int, int, int]] = []
    for rank in sorted(set(ranks)):
        pool.extend(enumerate_vertices(rank))
    pool.sort()

    max_rank = max(ranks) if ranks else 0
    out: list[DiagramString] = []
    for n_ops in range(1, max_operators + 1):
        allowed_sums = set(mu1_sum_range(bra_level, max_rank, n_ops))
        for combo in combinations_with_replacement(pool, n_ops):
            if sum(v[0] for v in combo) not in allowed_sums:
                continue
            out.append(DiagramString(tuple(combo), bra_level, ket_level))
    return out


# ── D2.3 closure filters ─────────────────────────────────────────────
#
# enumerate_candidates over-generates ~6-10x: its candidates satisfy the
# per-vertex bounds and the sum(mu1) window, but nothing yet requires the
# diagram to CLOSE.  Four independent rules do that, each its own predicate so
# a wrong rule localizes to one function and one test.
#
#   D2.3a  closes_internally  -- internal lines terminate on the H vertex
#   D2.3b  matches_manifold   -- leftover external lines match the bra
#   D2.3c  hamiltonian_type   -- the H vertex is one Planck implements
#   D2.3d  enumerate_diagrams -- the three composed
#
# ponytail: these are counting rules over the triplets, not a graph walk. The
# encoding exists precisely so closure is arithmetic; if one of these starts
# needing the line graph, that is a signal the triplets are missing a field,
# not that the predicate should grow a traversal.


def internal_line_count(ds: DiagramString) -> int:
    """Total internal lines demanded by the cluster operators (``sum(mu2)``)."""
    return sum(v[1] for v in ds.t_ops)


def internal_particle_count(ds: DiagramString) -> int:
    """Internal *particle* lines demanded by the operators (``sum(mu3)``)."""
    return sum(v[2] for v in ds.t_ops)


def closes_internally(ds: DiagramString, h_rank: int = 2) -> bool:
    """Whether the operators' internal lines can terminate on the H vertex.

    Every internal line has one end on a T operator and the other on the
    Hamiltonian vertex, so ``sum(mu2)`` must not exceed the vertex's capacity:
    ``2 * h_rank`` lines for an ``h_rank``-body operator (4 for the two-body V,
    2 for the one-body F).

    A candidate demanding more internal lines than the vertex can absorb is
    not a diagram at all.
    """
    return internal_line_count(ds) <= 2 * h_rank


def external_line_count(ds: DiagramString) -> int:
    """Lines left over to reach the bra: ``sum(2*mu1) - sum(mu2)``.

    Each level-mu1 operator contributes ``2*mu1`` lines (mu1 particle, mu1
    hole); ``mu2`` of them are internal, so the rest are external.  The
    Hamiltonian vertex can also emit external lines -- see
    :func:`matches_manifold`, which is why this is not the whole story.
    """
    return sum(2 * v[0] - v[1] for v in ds.t_ops)


def matches_manifold(ds: DiagramString, h_rank: int = 2) -> bool:
    """Whether the diagram's external lines land exactly on the bra.

    Count external lines from both ends:

      * the T operators leave ``sum(2*mu1 - mu2)`` lines free
        (:func:`external_line_count`);
      * the H vertex has ``2*h_rank`` slots, ``sum(mu2)`` of them consumed by
        internal lines, leaving ``2*h_rank - sum(mu2)`` free.

    A bra at excitation level ``k`` absorbs exactly ``2k`` lines -- k particle
    and k hole -- so the two must satisfy

        T_external + H_free == 2 * bra_level

    Derived from, and checked against, the D1.4 anchors: every one satisfies it
    at its own Hamiltonian rank (the two Fock anchors close at ``h_rank = 1``,
    the rest at 2).  The ket level is not yet a factor -- every anchor has
    ``ket_level = 0`` -- so a nonzero ket is rejected rather than guessed at.
    """
    if ds.ket_level != 0:
        return False
    h_free = 2 * h_rank - internal_line_count(ds)
    if h_free < 0:
        return False
    if external_line_count(ds) + h_free != 2 * ds.bra_level:
        return False

    # Each operator's own lines must be individually consistent: it cannot put
    # more particle lines inside than it has (mu3 <= mu1, already enforced by
    # is_wellformed) nor more hole lines (mu2 - mu3 <= mu1).
    for mu1, mu2, mu3 in ds.t_ops:
        if mu2 - mu3 > mu1:
            return False

    # Per-species vertex capacity. An h_rank-body vertex has h_rank particle
    # slots and h_rank hole slots, so the internal lines arriving from the
    # cluster operators cannot exceed either -- sum(mu3) particle lines and
    # sum(mu2 - mu3) hole lines.
    #
    # This is a CAP, not an equality. The externals' particle/hole split is
    # free (a two-body <pq||rs> can emit externals of either species, which is
    # how the pp ladder gets both its particle externals from the vertex); only
    # the TOTAL is fixed, by the check above. An earlier attempt that also
    # forced ext_p + h_free_p == bra_level per species wrongly rejected the pp
    # and hh ladders.
    internal_particles = internal_particle_count(ds)
    internal_holes = internal_line_count(ds) - internal_particles
    return internal_particles <= h_rank and internal_holes <= h_rank


# The normal-ordered Hamiltonian Planck builds is F_N + V_N: a one-body Fock
# vertex and a two-body ERI vertex. The paper's "for each Hamiltonian type
# matching excitation level" loop is this sweep.
HAMILTONIAN_RANKS = (1, 2)


def admissible_hamiltonian_ranks(ds: DiagramString) -> tuple[int, ...]:
    """Which Hamiltonian vertices *ds* can close on.

    A candidate is a diagram once *some* vertex in :data:`HAMILTONIAN_RANKS`
    both absorbs its internal lines and balances its externals.  Returning the
    set (rather than a bool) keeps the information D3 needs: the vertex rank
    determines whether the term carries ``f`` or ``v``.
    """
    return tuple(
        h for h in HAMILTONIAN_RANKS
        if closes_internally(ds, h) and matches_manifold(ds, h)
    )


def enumerate_diagrams(
    ranks: Sequence[int],
    bra_level: int,
    ket_level: int = 0,
    max_operators: int = MAX_T_OPERATORS,
) -> list[tuple[DiagramString, int]]:
    """All closed diagrams for *ranks* projected on *bra_level*.

    Returns ``(diagram, h_rank)`` pairs -- one entry per admissible Hamiltonian
    vertex, since the same cluster-operator topology closing on F and on V are
    different diagrams producing different terms.

    This is D2's output and D3's input.  Connectivity is NOT enforced here:
    a candidate whose operators all connect to the H vertex (mu2 >= 1, which
    :func:`is_wellformed` already requires) is connected through that vertex by
    construction, so the separate connectivity rule the scope anticipated turns
    out to be subsumed.  That is a claim the tests check rather than assume.
    """
    out: list[tuple[DiagramString, int]] = []
    for cand in enumerate_candidates(ranks, bra_level, ket_level, max_operators):
        for h_rank in admissible_hamiltonian_ranks(cand):
            out.append((cand, h_rank))
    return out


# ── D3.0 the weight oracle ───────────────────────────────────────────
#
# Each generated term names the diagram it came from, so a diagram's weight can
# be read off the term path by summing the coefficients of its terms.
#
# The individual coefficients are RAGGED -- the CCD ring-ring diagram's ten
# terms carry 1/32, 1/16 and 3/32 -- because the Wick path writes the P(ij)P(ab)
# antisymmetrizer out as unequally-weighted representatives.  So a diagram maps
# to one weight, not to n terms, and D3's gate must be per-diagram rather than
# the term-by-term multiset equality the scope document proposed: matching
# term-by-term would mean reproducing an arbitrary splitting.
#
# WHAT D3.0 ESTABLISHED, AND ITS LIMIT.  For CCD the signed sum is exactly the
# textbook diagram weight (hh +1/2, pp +1/2, ring -1, Fock +/-1, quads 1/4,
# 1/2, -1/2, -1/2). Beyond CCD it is NOT:
#
#   * 12-20 diagrams per manifold sum to exactly ZERO, because the two halves
#     of a P(ij) antisymmetrizer land in the same diagram with opposite signs
#     and annihilate. Concretely, ccsd/doubles diagram (((1,1,1),),2) holds
#     -1*t1(c,i)v(j,c,a,b) and +1*t1(c,j)v(i,c,a,b) -- one diagram, written as
#     its two antisymmetric halves.
#   * denominators reach 36 (ccsdt), so the signed sum is not a clean weight.
#
# Summing |coeff| removes every zero but leaves denominator 8, i.e. it sums
# over the P expansion without dividing by it. Neither variant is the weight.
# The missing piece is the antisymmetrizer multiplicity: a correct oracle must
# fold the P(..) halves back together before summing, which needs to know how
# many permutations the diagram's externals admit -- exactly the information
# D3.1's line graph carries and the triplets alone do not.
#
# Both variants are provided, honestly labelled. `diagram_weights` is usable as
# a gate ONLY for CCD; `diagram_weight_magnitudes` is zero-free everywhere and
# is the better starting point for D3.3, but is not itself the weight.


def term_diagram_id(term) -> tuple[tuple[tuple[int, int, int], ...], int]:
    """The ``(t_ops, h_rank)`` diagram identity of a generated term.

    A rank-n cluster factor contributes ``(n, #summed, #summed-and-virtual)``:
    its summed indices are exactly the lines running to the Hamiltonian vertex,
    and the virtual ones among them are the particle lines.  An ``f`` factor
    means the vertex is one-body.

    Returns the same shape :func:`enumerate_diagrams` emits, so the two are
    directly comparable.  A term with no cluster factors (the bare Hamiltonian)
    yields an empty ``t_ops``.
    """
    free = set(term.free_indices)
    triples = []
    for f in term.factors:
        if not f.name.startswith("t"):
            continue
        summed = [x for x in f.indices if x not in free]
        triples.append((
            int(f.name[1:]),
            len(summed),
            sum(1 for x in summed if x.space == "vir"),
        ))
    h_rank = 1 if any(f.name == "f" for f in term.factors) else 2
    return (tuple(sorted(triples)), h_rank)


def diagram_weights(terms) -> dict[tuple[tuple[tuple[int, int, int], ...], int], object]:
    """Signed sum of term coefficients per diagram.

    For CCD this IS the textbook diagram weight.  Beyond CCD it is not -- the
    two halves of a P(ij) antisymmetrizer share a diagram id and cancel, so
    many entries are exactly zero.  See the module note above; use only where
    the zero-free property has been checked.
    """
    from collections import defaultdict
    from fractions import Fraction

    out: dict[object, Fraction] = defaultdict(Fraction)
    for term in terms:
        out[term_diagram_id(term)] += term.coeff
    return dict(out)


def diagram_weight_magnitudes(terms) -> dict[object, object]:
    """Sum of ``|coeff|`` per diagram -- zero-free on every manifold measured.

    Immune to the P-halves cancellation that makes :func:`diagram_weights` go
    to zero, so it is a usable per-diagram fingerprint.  It is NOT the diagram
    weight: it sums over the antisymmetrizer expansion without dividing by it
    (denominators reach 8 where the weight's are <= 4).  D3.3 has to supply
    that divisor.
    """
    from collections import defaultdict
    from fractions import Fraction

    out: dict[object, Fraction] = defaultdict(Fraction)
    for term in terms:
        out[term_diagram_id(term)] += abs(term.coeff)
    return dict(out)


# ── AR2.1 diagram sign: (-1)^(h+l) via directed loop trace ────────────
#
# Crawford & Schaefer III, Rev. Comput. Chem. 14 (2000), p.84:
#   sign = (-1)^(h + l),  h = hole lines,  l = loops,
#   "a loop is a route along a series of DIRECTED lines that either returns to
#    its beginning or begins at one external line and ends at another."
# `directed_loops` implements that directed pass-through trace and is VALIDATED
# against Crawford's worked examples (energy p.84/p.87 l=2; open singles p.91
# Eq.[180] LEFT l=2 h=2 +, RIGHT l=2 h=3 -). Both l and h are pure functions of
# the diagram topology, so `diagram_sign` is a well-defined diagram invariant.
#
# NOTE: this is Crawford's CANONICAL-arrangement sign. It equals the term path's
# per-diagram sign only up to a rep-external-labeling / P(ij..)-orbit convention
# (the AR2.3 delta); reconciling that so an emitted diagram term carries the
# right sign end-to-end is AR2.3/D4, not AR2.1. AR2.1 owes the source-validated
# sign function, which is what this is. Consumes a diagram REPRESENTATIVE
# (an AlgebraTerm from `diagram_representative`), not a DiagramString.


def _trace_directed_loops(rep):
    """Shared oriented-loop trace. Returns ``(nloops, open_pairs)`` where
    ``open_pairs`` is the list of ``(entry_ext_name, exit_ext_name)`` for each
    OPEN loop (one that starts and ends at an external line).

    Directed pass-through: each summed line is a directed factor->factor edge
    (particle vir-index in enumeration order; hole occ-index REVERSED); external
    lines are half-edges to an ("EXT", name) terminus; at each factor an incoming
    edge is pass-through-paired with an outgoing edge, and loops are the resulting
    cycles. Stdlib only. `directed_loops` (Crawford l count, AR2.1) and
    `open_loop_external_pairing` (AR2.3(i).1b crossing) both consume this."""
    from collections import defaultdict

    sset = set(rep.summed_indices)
    occ_by: dict[str, list] = defaultdict(list)
    for k, f in enumerate(rep.factors):
        for i in f.indices:
            occ_by[i.name].append((k, i.space, i in sset))
    edges = []  # (src, dst, species); src/dst is a factor int or ("EXT", name)
    for name, eps in occ_by.items():
        if len(eps) == 2:
            (k1, sp, _), (k2, _, _) = eps
            edges.append((k1, k2, sp) if sp == "vir" else (k2, k1, sp))
        else:
            (k, sp, _) = eps[0]
            ext = ("EXT", name)
            edges.append((k, ext, sp) if sp == "vir" else (ext, k, sp))
    inc: dict = defaultdict(list)
    out: dict = defaultdict(list)
    for ei, (s, d, _sp) in enumerate(edges):
        out[s].append(ei)
        inc[d].append(ei)
    nxt: dict = {}
    facs = {k for e in edges for k in e[:2] if not isinstance(k, tuple)}
    for k in facs:
        for a, b in zip(inc[k], out[k]):
            nxt[a] = b
    seen: set = set()
    nloops = 0
    open_pairs = []
    for ei in range(len(edges)):  # open loops from EXT sources first
        if ei in seen or not isinstance(edges[ei][0], tuple):
            continue
        nloops += 1
        entry = edges[ei][0][1]  # EXT name we started at
        cur, last = ei, ei
        while cur is not None and cur not in seen:
            seen.add(cur)
            last = cur
            cur = nxt.get(cur)
        end = edges[last][1]
        open_pairs.append((entry, end[1] if isinstance(end, tuple) else None))
    for ei in range(len(edges)):  # then closed cycles
        if ei in seen or ei not in nxt:
            continue
        nloops += 1
        cur = ei
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = nxt.get(cur)
    return nloops, open_pairs


def directed_loops(rep) -> int:
    """Number of oriented loops in a diagram representative's contraction.

    Each open EXT->...->EXT path counts as one loop, plus the closed cycles.
    Validated against Crawford's worked (l) values -- see module note."""
    return _trace_directed_loops(rep)[0]


def diagram_hole_lines(rep) -> int:
    """Number of hole lines h = distinct occupied index names (internal +
    external), per Crawford's h count (p.91 worked examples)."""
    return len({i.name for f in rep.factors for i in f.indices if i.space == "occ"})


def diagram_sign(rep) -> int:
    """Crawford's diagram sign ``(-1)^(h + l)`` for a diagram representative.

    Validated against Crawford's worked examples (see module note). A pure
    topology invariant. This is the canonical-arrangement sign; see the note on
    the AR2.3 convention delta before using it to set an emitted term's sign.
    """
    return (-1) ** (diagram_hole_lines(rep) + directed_loops(rep))


# ── AR2.3(i).1b crossing parity (the structural sign-correction lead) ──
#
# `diagram_sign` (Crawford `(-1)^(h+l)`) matches the PySCF-solved sign on only
# 19/30 CCSD-doubles diagrams; the +/-1 delta is a fixed CONVENTION correction
# (assembled-rep external labeling + P-orbit vs Crawford's canonical arrangement).
# It is NOT a scalar count -- every loop/hole-count variant scores <=21/30. The
# missing structural quantity is the open loops' EXTERNAL ENDPOINT PAIRING, which
# `directed_loops` collapses: whether the oriented loops pair the externals as
# (i<->a, j<->b) [identity, +1] or (i<->b, j<->a) [crossed, -1]. This CROSSING
# PARITY alone reproduces the correction 23/30 -- the best structural predictor
# found, and the .1b lead. The residual 7 (AR2.3(i).1b.2) are the open piece.


def open_loop_external_pairing(rep):
    """The list of ``(entry_external, exit_external)`` name pairs for the open
    directed loops (AR2.3(i).1b.0). Pure topology; the endpoint structure
    `directed_loops` discards."""
    return _trace_directed_loops(rep)[1]


def crossing_parity(rep) -> int:
    """+1 if the open loops pair the doubles externals as the identity
    (occ i -> vir a, j -> b), -1 if crossed (i -> b, j -> a) (AR2.3(i).1b.1).

    Reproduces the .1a stored sign correction 23/30 -- the structural lead for a
    solve-free sign rule; the residual mismatches are AR2.3(i).1b.2. Defined for
    the doubles manifold (two occ + two vir externals). Superseded by the
    rank-general :func:`external_pairing_parity` (B1), to which it is identical
    on the doubles manifold."""
    d = dict(open_loop_external_pairing(rep))
    return 1 if d.get("i") == "a" else -1


def _permutation_parity(items) -> int:
    """Sign (+/-1) of the permutation that sorts *items* into ascending order.
    Counts inversions -- O(n^2), fine for the <=4 externals per species here."""
    seq = list(items)
    inv = sum(
        1
        for a in range(len(seq))
        for b in range(a + 1, len(seq))
        if seq[a] > seq[b]
    )
    return -1 if inv % 2 else 1


def external_pairing_parity(rep) -> int:
    """Rank-general crossing parity (B1): the sign of the permutation the open
    directed loops impose between the occupied and virtual externals.

    Each open loop pairs one occ external with one vir external
    (:func:`open_loop_external_pairing`). Sort the pairs by their occ name; the
    parity of the resulting vir-name sequence (vs sorted vir order) is the sign.
    For the doubles manifold this is exactly :func:`crossing_parity`
    ((i->a,j->b) identity = +1, (i->b,j->a) one swap = -1); for triples/higher it
    is the parity of the full k-element external permutation, which the
    doubles-hardcoded `crossing_parity` could not read."""
    pairs = sorted(open_loop_external_pairing(rep), key=lambda p: p[0])
    return _permutation_parity([vir for _occ, vir in pairs])


def structural_sign(rep, h_rank: int) -> int:
    """A solve-free emit sign for a diagram, from topology alone. Reproduces the
    PySCF-solved sign for ALL 30 CCSD-doubles diagrams (AR2.3(i).1b complete).

    Rule:
      ``sign = crossing_parity · (-1)^l · (-1 if the Fock line contracts a hole)``

    - ``crossing_parity · (-1)^l`` gives the ERI-vertex sign exactly (26/26 on
      ``h_rank == 2``; .1b.2a).
    - the one-body Fock vertex adds a ``-1`` iff its contracted (internal) line
      is a hole/occ line -- a hole-line direction convention on the one-body
      vertex (.1b.2b, `(-1 if fock-occ)` = 4/4 on the ``h_rank == 1`` diagrams;
      `(-1 always)` = 3/4 and `(-1 if fock-vir)` = 0/4 both fail, so the species
      dependence is real, not a flat flip).

    Uses the rank-general :func:`external_pairing_parity` (B1) -- identical to
    `crossing_parity` on doubles, and defined for triples/higher (the doubles
    verification is unchanged; higher-rank correctness is gated on the AR3 CCSDT
    oracle, B0)."""
    sign = external_pairing_parity(rep) * (-1) ** directed_loops(rep)
    if h_rank == 1:
        summed = set(rep.summed_indices)
        fock_is_hole = any(
            i in summed and i.space == "occ"
            for f in rep.factors
            if f.name == "f"
            for i in f.indices
        )
        if fock_is_hole:
            sign = -sign
    return sign


# ── AR2.3(i).0 PySCF-derived signed-weight oracle ─────────────────────
#
# The committed `tests/ccsd_diagram_weights.json` is the AR2 oracle. Its weights
# come from `solve_diagram_weights_vs_pyscf` (a full-rank least-squares solve
# against the PySCF doubles residual, span ~1e-15) and its diagram-id set comes
# from `enumerate_diagrams` (PySCF-free), so it is PySCF-provenance-only -- never
# from the term-path generator. `dump_ccsd_weight_table` keeps the SIGN on the
# numerator, so this is the ground-truth SIGNED weight per diagram, the oracle
# AR2.3(i) reconciles `diagram_sign` against. Consuming it needs no PySCF (the
# freshness-vs-PySCF pin lives in test_reference_vs_pyscf.py).


def pyscf_signed_weights() -> dict:
    """The committed PySCF-derived signed weight per CCSD-doubles diagram.

    Returns ``{(t_ops, h_rank): Fraction}`` (plus ``"bare": Fraction``). The
    sign is the ground truth AR2.3(i)'s canonical-external relabel must make
    ``diagram_sign`` reproduce."""
    import ast
    import json
    from fractions import Fraction
    from pathlib import Path

    path = Path(__file__).parent / "tests" / "ccsd_diagram_weights.json"
    table = json.load(open(path))
    out = {}
    for key, (num, den) in table.items():
        did = key if key == "bare" else ast.literal_eval(key)
        out[did] = Fraction(num, den)
    return out


# ── AR2.2a equivalent-line pairs (magnitude 1/2 per pair) ─────────────
#
# Crawford & Schaefer III, p.85: "two lines beginning at the same [interaction
# line] and ending at the same interaction line" are EQUIVALENT; each such pair
# contributes a prefactor of 1/2 to |weight|. Operationally, on the diagram
# representative: two summed lines are equivalent iff they connect the SAME pair
# of factor endpoints with the SAME species. Validated against known pair counts:
# pp/hh ladders 1, ring 0 (mixed species), f2-energy 2, t1*t1*v 0 (distinct t1
# starts). This is the equivalent-LINE part of the magnitude only; the
# equivalent-VERTEX 1/n_v! (AR2.2b) and the amplitude-normalization convention
# (AR2.2c) are the remaining AR2.2 pieces.


def equivalent_line_pairs(rep) -> int:
    """Number of equivalent-line pairs in a diagram representative (Crawford
    p.85). Each pair supplies one factor of 1/2 to ``|weight|``."""
    from collections import Counter, defaultdict

    summed = set(rep.summed_indices)
    endpoints: dict[str, list] = defaultdict(list)
    for k, f in enumerate(rep.factors):
        for i in f.indices:
            if i in summed:
                endpoints[i.name].append((k, i.space))
    conn: Counter = Counter()
    for _name, eps in endpoints.items():
        if len(eps) == 2:
            (k1, sp), (k2, _) = eps
            conn[(frozenset([k1, k2]), sp)] += 1
    return sum(c // 2 for c in conn.values())


# ── AR2.2b equivalent-vertex factor (magnitude 1/n_v! per group) ──────
#
# Crawford & Schaefer III, p.87: "equivalent" vertices are IDENTICAL operators
# "connected to the same interaction line in exactly the same manner (each by a
# hole line and a particle line)"; n such vertices contribute 1/n!. The load-
# bearing qualifier is "the same manner" -- a naive 1/n! over ALL identical-rank
# operators OVER-fires (it produced the messy residuals 3/6/2 in the AR2.2
# scoping). The fix: group identical operators by their internal CONNECTION
# SIGNATURE (the multiset of (species, partner-factor) over their summed legs);
# apply 1/n! only within a same-signature group.
#
# This detector is validated to leave a CLEAN DYADIC residual {1,2,4} against the
# weight table (the messy 3/6 are gone) -- that residual is the separate
# amplitude-normalization factor (AR2.2c), not a vertex-factor error.


def _connection_signature(rep, k: int):
    """How factor ``k`` connects internally: its name + the sorted multiset of
    (species, partner-factor-name) over its summed legs."""
    summed = set(rep.summed_indices)
    f = rep.factors[k]
    sig = []
    for i in f.indices:
        if i not in summed:
            continue
        for k2, f2 in enumerate(rep.factors):
            if k2 != k and any(j.name == i.name for j in f2.indices):
                sig.append((i.space, f2.name))
                break
    return (f.name, tuple(sorted(sig)))


def equivalent_vertex_factor(rep):
    """The ``∏ 1/n_v!`` equivalent-vertex prefactor of ``|weight|`` (Crawford
    p.87), returned as a ``Fraction``.

    Groups the cluster-amplitude factors by (name, connection signature) and
    multiplies 1/n! for each group of n>=2 identical, same-manner operators.
    """
    import math
    from collections import Counter
    from fractions import Fraction

    groups = Counter(
        _connection_signature(rep, k)
        for k, f in enumerate(rep.factors)
        if f.name.startswith("t")
    )
    factor = Fraction(1)
    for count in groups.values():
        if count > 1:
            factor *= Fraction(1, math.factorial(count))
    return factor


# ── AR2.2c external-pair (amplitude/vertex normalization) factor ──────
#
# ccgen stores bare ANTISYMMETRIC amplitudes and integrals -- no leading
# 1/(k!) on a k-fold antisymmetric block. The diagrammatic weight rule assumes
# the physicist normalization where each antisymmetric block carries that
# prefactor, so every same-species EXTERNAL line pair the P(..) antisymmetrizer
# will regenerate is double-counted by exactly 2. Each such pair therefore
# contributes a 1/2 to |weight| that the equivalent-line (AR2.2a) and
# equivalent-vertex (AR2.2b) factors do NOT capture -- those count INTERNAL
# (summed) equivalences; this counts EXTERNAL ones, on the amplitudes and on the
# Hamiltonian vertex alike.
#
# Pure function of the triplets + bra_level (no rep assembly needed). For an
# amplitude (mu1, mu2, mu3): particle-external = mu1 - mu3, hole-external =
# mu1 - (mu2 - mu3); the vertex carries whatever externals the amplitudes don't
# (bra_level of each species total). Same-species pairs are floor-divided by 2.
#
# Derived + verified 30/30 against ccsd_diagram_weights.json (the 6 rows the
# amplitude-only count missed are exactly the ones whose external pair sits on
# the vertex).
#
# M1.0/M1.1 (AR3): the factor splits into an AMPLITUDE part and a VERTEX part.
# The amplitude part is the amplitude-normalization factor `prod (1/n_ext!)` over
# each amplitude's same-species external legs -- NOT the pair-count `(1/2)^(k//2)`.
# On doubles the two coincide exactly (verified 30/30: for T2, k_ext in {0,1,2}
# and `(1/2)^(k//2) == 1/k!`), but they DIVERGE at T3 (`1/3! = 1/6` vs
# `(1/2)^(3//2) = 1/2`), so the factorial form is the correct generalization and
# the pair form was only an artifact of n<=2. The vertex part stays a pair count.


def _amplitude_norm_factor(ds: DiagramString):
    """Amplitude external-normalization ``prod_amp prod_species (1/n_ext!)``
    (M1.1). For an amplitude ``(mu1,mu2,mu3)``: particle-external ``= mu1-mu3``,
    hole-external ``= mu1-(mu2-mu3)``. Identical to the old pair-count amplitude
    part on doubles; the correct ``1/n!`` at T3+. Returns a ``Fraction``."""
    import math
    from fractions import Fraction

    f = Fraction(1)
    for m1, m2, m3 in ds.t_ops:
        pe = m1 - m3
        he = m1 - (m2 - m3)
        f *= Fraction(1, math.factorial(pe)) * Fraction(1, math.factorial(he))
    return f


def _vertex_pair_factor(ds: DiagramString) -> int:
    """The vertex external same-species pair count ``2^p`` (the part of the old
    `external_pair_factor` that sits on the Hamiltonian vertex, unchanged)."""
    bra = ds.bra_level
    amp_pe = sum(m1 - m3 for m1, m2, m3 in ds.t_ops)
    amp_he = sum(m1 - (m2 - m3) for m1, m2, m3 in ds.t_ops)
    return 2 ** ((bra - amp_pe) // 2 + (bra - amp_he) // 2)


def external_pair_factor(ds: DiagramString) -> int:
    """DEPRECATED (kept for the AR2.2c doubles gate): the ``2^p`` amplitude+vertex
    pair count. Superseded on the magnitude path by ``_amplitude_norm_factor`` x
    ``_vertex_pair_factor`` (M1), which is identical on doubles and correct at
    T3+. ``p`` = same-species external-line pairs, per amplitude and on the
    vertex, each a 1/2 to ``|weight|``."""
    amp_pairs = sum(
        (m1 - m3) // 2 + (m1 - (m2 - m3)) // 2 for m1, m2, m3 in ds.t_ops
    )
    return 2 ** amp_pairs * _vertex_pair_factor(ds)


def diagram_magnitude(ds: DiagramString, h_rank: int):
    """``|weight|`` for a diagram.

    The structural factors:
      * equivalent-VERTEX ``∏ 1/n_v!`` (AR2.2b)
      * equivalent-LINE ``(1/2)^(pairs)`` (AR2.2a, internal/summed)
      * amplitude normalization ``∏ 1/n_ext!`` (AR2.2c / M1, per amplitude)
      * vertex external pairs ``(1/2)^(pairs)`` (AR2.2c, on the vertex)

    Reproduces every CCSD-doubles magnitude in ``ccsd_diagram_weights.json``
    (30/30). At T3+ the amplitude factor is non-dyadic (``1/3!`` etc.); that
    higher-rank value is validated only by the AR3 energy/FCI gate, not a
    per-diagram oracle. Sign is separate (``structural_sign``)."""
    from fractions import Fraction

    rep = diagram_representative(ds, h_rank)
    return (
        equivalent_vertex_factor(rep)
        * _amplitude_norm_factor(ds)
        / Fraction(2 ** equivalent_line_pairs(rep) * _vertex_pair_factor(ds))
    )


# ── AR2.3(i).1b.3 solve-free signed weight ────────────────────────────
#
# `diagram_signed_weight` now uses the STRUCTURAL sign (`structural_sign`,
# AR2.3(i).1b) x the structural magnitude (`diagram_magnitude`, AR2.2), so the
# full signed weight is derived from topology alone -- no PySCF solve, no stored
# table, no term-path generator. Both halves are pinned 30/30 against the PySCF
# signed-weight oracle. This is the AR2.3(i) payoff: the weight generalizes past
# any rank with a PySCF solve to build a table against.
#
# `sign_correction` (the .1a stored +/-1) is retained as a DIAGNOSTIC only -- it
# is what `structural_sign` had to reproduce, and comparing the two is the
# regression that would catch a convention drift. It is no longer on the
# signed-weight path.


def sign_correction(ds: DiagramString, h_rank: int) -> int:
    """DIAGNOSTIC (AR2.3(i).1a): the +/-1 that turns Crawford's ``diagram_sign``
    into the PySCF-solved sign, looked up from ``pyscf_signed_weights``. Superseded
    on the signed-weight path by the solve-free ``structural_sign`` (.1b); kept as
    the oracle ``structural_sign`` is regression-checked against. Raises
    ``KeyError`` for a diagram absent from the table."""
    w = pyscf_signed_weights()[(ds.t_ops, h_rank)]
    rep = diagram_representative(ds, h_rank)
    return (1 if w > 0 else -1) * diagram_sign(rep)


def diagram_signed_weight(ds: DiagramString, h_rank: int):
    """The full signed diagram weight ``sign * |weight|`` as a ``Fraction``,
    SOLVE-FREE (AR2.3(i).1b.3 + M1.2).

    ``sign = (-1)^bra_level * structural_sign`` (topology-derived, .1b) and
    ``|weight| = diagram_magnitude`` (structural, AR2.2 + M1). The
    ``(-1)^bra_level`` is a per-MANIFOLD sign convention (M1.2): it was invisible
    on doubles (``(-1)^2 = +1``, a no-op) and only surfaced when the
    diagram-weighted residual was validated against ccgen at singles (bra=1) and
    triples (bra=3), where the whole manifold's weight is flipped. Verified: with
    this factor, ``diagram_signed_weight * orbit(rep)`` reproduces the ccgen
    residual per-diagram across ccsd/ccsdt singles+doubles+triples (140 diagrams),
    and the diagram-built CCSDT residual reaches the FCI energy. Still 30/30 on
    the PySCF doubles table (the factor is +1 there)."""
    rep = diagram_representative(ds, h_rank)
    manifold_sign = (-1) ** ds.bra_level
    return manifold_sign * structural_sign(rep, h_rank) * diagram_magnitude(ds, h_rank)


# ── D3.1 line graph ──────────────────────────────────────────────────
#
# The triplets say HOW MANY particle/hole lines each operator sends internally
# vs externally; the line graph records the endpoints, which is what D3.3 needs
# to fold the P(..) antisymmetrizer.
#
# Endpoints are labelled nodes:
#   ("t", i)   the i-th cluster operator (order of ds.t_ops)
#   "H"        the Hamiltonian vertex
#   "bra"      the projection determinant
#
# A line is (species, endpoint_a, endpoint_b) with species "p" or "h". Internal
# lines run ("t",i) -> "H"; external lines run ("t",i) -> "bra" and, for the H
# vertex's own free slots, "H" -> "bra".
#
# The graph is a pure function of (t_ops, h_rank): the counts fully determine
# how many lines of each species/kind exist. It carries NO index identities --
# that is D3.2's job -- only the multiplicity structure D3.0 found missing.


@dataclass(frozen=True)
class LineGraph:
    """Edge-list form of a diagram: particle/hole lines between endpoints.

    ``lines`` is a tuple of ``(species, endpoint_a, endpoint_b)``.  The counts
    that matter to D3.3 are read back via the properties.
    """

    lines: tuple[tuple[str, object, object], ...]
    bra_level: int
    h_rank: int

    @property
    def external_particles(self) -> int:
        return sum(1 for sp, a, b in self.lines if sp == "p" and "bra" in (a, b))

    @property
    def external_holes(self) -> int:
        return sum(1 for sp, a, b in self.lines if sp == "h" and "bra" in (a, b))


def build_line_graph(ds: DiagramString, h_rank: int) -> LineGraph:
    """Construct the D3.1 line graph for diagram *ds* closing on an *h_rank*
    vertex.

    Each level-mu1 operator carries mu1 particle and mu1 hole lines.  Of those,
    mu3 particle and (mu2 - mu3) hole lines are internal (to the H vertex); the
    remaining (mu1 - mu3) particle and (mu1 - (mu2 - mu3)) hole lines are
    external (to the bra).  The H vertex's own unused slots then emit whatever
    externals the operators did not supply, so the total external count matches
    the bra -- the same balance :func:`matches_manifold` enforces.

    Requires ``h_rank in admissible_hamiltonian_ranks(ds)``; a non-closing
    diagram has no consistent graph and raises.
    """
    if h_rank not in admissible_hamiltonian_ranks(ds):
        raise ValueError(
            f"{to_string(ds)} does not close on an h_rank={h_rank} vertex"
        )

    lines: list[tuple[str, object, object]] = []
    for i, (mu1, mu2, mu3) in enumerate(ds.t_ops):
        node = ("t", i)
        int_p, int_h = mu3, mu2 - mu3
        ext_p, ext_h = mu1 - mu3, mu1 - int_h
        lines += [("p", node, "H")] * int_p
        lines += [("h", node, "H")] * int_h
        lines += [("p", node, "bra")] * ext_p
        lines += [("h", node, "bra")] * ext_h

    # The H vertex fills the bra's remaining slots from its own free lines.
    h_ext_p = ds.bra_level - sum(v[0] - v[2] for v in ds.t_ops)
    h_ext_h = ds.bra_level - sum(v[0] - (v[1] - v[2]) for v in ds.t_ops)
    lines += [("p", "H", "bra")] * h_ext_p
    lines += [("h", "H", "bra")] * h_ext_h

    return LineGraph(tuple(lines), ds.bra_level, h_rank)


# ── D3.2a index-pool allocation ──────────────────────────────────────
#
# A diagram's lines become indices: external lines carry the bra's own indices
# (i,j,.. / a,b,..), internal lines carry fresh summed dummies. The one hard
# constraint is the apply_deltas bug documented in Open Work: a dummy must NEVER
# share a name with an external, or downstream collapses it onto that external.
# The allocator enforces this by drawing dummies from strictly BEYOND the
# externals in each pool.


@dataclass(frozen=True)
class IndexPools:
    """Externals and fresh dummies for one diagram, guaranteed disjoint.

    ``ext_occ`` / ``ext_vir`` are the bra indices (``bra_level`` of each);
    ``dummy_occ`` / ``dummy_vir`` are fresh summed indices that share no name
    with any external.
    """

    ext_occ: tuple[Index, ...]
    ext_vir: tuple[Index, ...]
    dummy_occ: tuple[Index, ...]
    dummy_vir: tuple[Index, ...]

    @property
    def all_names(self) -> set[str]:
        return {
            i.name
            for grp in (self.ext_occ, self.ext_vir, self.dummy_occ, self.dummy_vir)
            for i in grp
        }


def allocate_indices(
    bra_level: int, n_dummy_occ: int, n_dummy_vir: int
) -> IndexPools:
    """Allocate externals and non-colliding dummies for a diagram.

    Externals take the first ``bra_level`` names of each pool; dummies take the
    next ``n_dummy_*`` names, so a dummy can never coincide with an external.
    That disjointness is the guard against the ``apply_deltas`` collapse bug
    (Open Work): a summed index sharing a name with an external gets silently
    rewritten onto it.

    Pools are extended on demand, so high-rank diagrams (many dummies) just draw
    further down the alphabet.
    """
    from .indices import extend_pool

    need_occ = bra_level + n_dummy_occ
    need_vir = bra_level + n_dummy_vir
    extend_pool(OCC_POOL, need_occ)
    extend_pool(VIR_POOL, need_vir)

    ext_occ = tuple(make_occ(OCC_POOL[k], dummy=False) for k in range(bra_level))
    ext_vir = tuple(make_vir(VIR_POOL[k], dummy=False) for k in range(bra_level))
    dummy_occ = tuple(
        make_occ(OCC_POOL[bra_level + k], dummy=True) for k in range(n_dummy_occ)
    )
    dummy_vir = tuple(
        make_vir(VIR_POOL[bra_level + k], dummy=True) for k in range(n_dummy_vir)
    )
    return IndexPools(ext_occ, ext_vir, dummy_occ, dummy_vir)


# ── D3.2b factor assembly ────────────────────────────────────────────
#
# Turning a diagram into an AlgebraTerm means placing shared dummies (internal
# lines) and bra indices (external lines) into the factor slots so the
# CONTRACTION matches ccgen -- not just the per-factor line counts (attempt 1
# got the counts right and the contraction wrong) nor the residual norm (attempt
# 2 matched the norm with a different tensor). Both wrong attempts were caught by
# the residual evaluator; see the scope doc.
#
# LADDER CASE (landed here). A diagram is "ladder-shaped" when every operator's
# internal lines fill a whole antisymmetric pair -- for each (mu1,mu2,mu3), the
# internal particle count mu3 is 0 or mu1 AND the internal hole count (mu2-mu3)
# is 0 or mu1. For these the vertex convention
#   v bra = vir-internal ++ occ-external,  v ket = vir-external ++ occ-internal
# reproduces ccgen EXACTLY (pp ladder t2(c,d,i,j)v(c,d,a,b) to 0.0). About half
# of all CCD/CCSD diagrams are ladder-shaped. The mixed case (one operator with
# an internal vir AND an internal occ that pair with different vertex slots, e.g.
# the ring) is NOT handled here -- diagram_representative raises for it, so this
# function never silently returns a wrong contraction.


def is_ladder_diagram(ds: DiagramString) -> bool:
    """Whether every operator's internal lines fill whole antisymmetric pairs.

    True when, for each operator, the internal particle lines are 0 or all mu1,
    and likewise the internal hole lines.  These are exactly the diagrams the
    ladder assembly reproduces exactly; the mixed case needs the edge matching
    (D3.2b-iii).
    """
    for mu1, mu2, mu3 in ds.t_ops:
        int_p, int_h = mu3, mu2 - mu3
        if int_p not in (0, mu1) or int_h not in (0, mu1):
            return False
    return True


def _assemble_ladder(ds: DiagramString, h_rank: int, pools: IndexPools):
    """Factors for a ladder-shaped diagram (see :func:`is_ladder_diagram`).

    Cluster slots are (vir x n, occ x n) with internal lines first in each block.
    The vertex takes bra = vir-internal ++ occ-external, ket = vir-external ++
    occ-internal -- the convention proved exact on the ladders.
    """
    from .tensors import tn, f as fock, v as eri

    occ_d = list(pools.dummy_occ)
    vir_d = list(pools.dummy_vir)
    occ_e = list(pools.ext_occ)
    vir_e = list(pools.ext_vir)

    internal_vir: list[Index] = []
    internal_occ: list[Index] = []

    factors = []
    for mu1, mu2, mu3 in ds.t_ops:
        int_p, int_h = mu3, mu2 - mu3
        vslots = [vir_d.pop(0) for _ in range(int_p)]
        internal_vir += vslots[:]
        vslots += [vir_e.pop(0) for _ in range(mu1 - int_p)]
        oslots = [occ_d.pop(0) for _ in range(int_h)]
        internal_occ += oslots[:]
        oslots += [occ_e.pop(0) for _ in range(mu1 - int_h)]
        factors.append(tn(mu1, tuple(vslots), tuple(oslots)))

    if h_rank == 1:
        # Fock f(p,q): read off ccgen, bra p = vir-external OR occ-internal,
        # ket q = vir-internal OR occ-external -- the OPPOSITE species-to-side
        # mapping from the ERI vertex. A ladder Fock vertex has exactly one line
        # of each side (it is one-body), so exactly one of these lists is
        # nonempty per slot.
        bra = list(vir_e) + internal_occ
        ket = internal_vir + list(occ_e)
        factors.append(fock(bra[0], ket[0]))
    else:
        # ERI <pq||rs>: bra = vir-internal ++ occ-external, ket = vir-external
        # ++ occ-internal.
        bra = internal_vir + list(occ_e)
        ket = list(vir_e) + internal_occ
        factors.append(eri(bra[0], bra[1], ket[0], ket[1]))

    free = tuple(pools.ext_vir) + tuple(pools.ext_occ)
    summed = tuple(internal_vir) + tuple(internal_occ)
    return tuple(factors), free, summed


def _assembler_handles(ds: DiagramString) -> bool:
    """Whether the assembler produces the correct topology for *ds*.

    Now **all** diagrams: the `_assemble_ladder` slot-fill convention (read off
    CCD/CCSD ladders) was verified proportional to ccgen for every diagram
    shape — ladder, single-operator mixed, and multi-operator mixed — once the
    residual gate is evaluated at adequate dimensions (`no, nv >= 4`; at the
    old `no=3, nv=4` a mixed/T3 residual partially vanishes and a *correct*
    assembly reads as non-proportional, which is what produced the three
    reverted D3.2b "mixed assembly is unbuilt" attempts).

    Out-of-sample coverage: ladders (ccd/ccsd/ccsdt), single-op mixed (ccsd 6 +
    ccsdt 17), multi-op mixed (ccsd 12 + ccsdt 3+15+40) — **0 failures**. The
    only remaining precondition is diagram closure, which
    :func:`diagram_representative` checks separately via
    :func:`admissible_hamiltonian_ranks`.

    Retained as a named predicate (rather than inlining `True`) so the boundary
    has one documented home if a future diagram shape is found the convention
    does not cover.
    """
    return True


def diagram_representative(ds: DiagramString, h_rank: int):
    """One ``AlgebraTerm`` for diagram *ds* (unit coefficient, no P orbit yet).

    Handles the cases :func:`_assembler_handles` covers (ladder-shaped and
    single-operator diagrams); raises ``NotImplementedError`` for the
    multi-operator mixed case, which needs the edge-matching assembly
    (D3.2b-iii).  The coefficient and antisymmetrizer orbit are D3.2c.
    """
    from fractions import Fraction
    from .project import AlgebraTerm

    if h_rank not in admissible_hamiltonian_ranks(ds):
        raise ValueError(
            f"{to_string(ds)} does not close on an h_rank={h_rank} vertex"
        )
    if not _assembler_handles(ds):
        raise NotImplementedError(
            f"{to_string(ds)} is multi-operator mixed; edge-matching assembly "
            "(D3.2b-iii) is not implemented"
        )
    n_int_vir = sum(v[2] for v in ds.t_ops)
    n_int_occ = sum(v[1] - v[2] for v in ds.t_ops)
    pools = allocate_indices(ds.bra_level, n_int_occ, n_int_vir)
    factors, free, summed = _assemble_ladder(ds, h_rank, pools)
    return AlgebraTerm(
        coeff=Fraction(1),
        factors=factors,
        free_indices=free,
        summed_indices=summed,
        connected=True,
        provenance=("diagram", to_string(ds), h_rank),
    )
