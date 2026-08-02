"""Seeded dressed-operator hypotheses for CCSD residual factorization (A2).

A1 introduced the tau pseudo-amplitude.  A2 adds the rest of the Stanton-Gauss
dressed-operator family that the hand-written CCSD tensor backend uses:

    Fae, Fmi, Fme          (one-particle dressed Fock blocks)
    Wmnij, Wabef, Wmbej    (two-particle dressed intermediates)

These operators are NOT discovered from scratch (that is A2b / A3's job); they
are *seeded* -- transcribed verbatim from the spin-orbital CCSD equations
(Stanton, Gauss, Watts, Bartlett, JCP 94, 4334 (1991), Eqs. 3-8), which are
exactly the convention ccgen's spin-orbital residual uses.  A2 builds and
validates these hypotheses offline; it does not rewrite any equation.

ponytail: seeded, not discovered. The general subgraph-isomorphism rewrite of
embedded operators is A3 -- keep that out of this module. A2 only says "here is
what a Wmnij looks like, and here is proof the definition is self-consistent".

Reference (spin-orbital, antisymmetrized <pq||rs> = v, amplitudes t1/t2, and
the A1 pseudo-amplitudes tau, tau_tilde):

    Fae   = f_ae - 1/2 f_me t1_ma + t1_mf <ma||fe> - 1/2 tau~_mnaf <mn||ef>
    Fmi   = f_mi + 1/2 f_me t1_ie + t1_ne <mn||ie> + 1/2 tau~_inef <mn||ef>
    Fme   = f_me + t1_nf <mn||ef>
    Wmnij = <mn||ij> + P(ij) t1_je <mn||ie> + 1/4 tau_ijef <mn||ef>
    Wabef = <ab||ef> - P(ab) t1_mb <am||ef> + 1/4 tau_mnab <mn||ef>
    Wmbej = <mb||ej> + t1_jf <mb||ef> - t1_nb <mn||ej>
                     - ( 1/2 t2_jnfb + t1_jf t1_nb ) <mn||ef>

(The diagonal Fock pieces f_ae/f_mi are the orbital-energy seeds; in the ccgen
algebra they are the ``f`` tensor's vv/oo blocks.)
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ..canonicalize import canonicalize_term
from ..indices import Index, make_occ, make_vir
from ..project import AlgebraTerm
from ..tensors import Tensor, f, t1, t2, v, reindex_tensors
from .tau import (TAU_NAME, TAU_CONTRACTED_NAME, TAU_SPEC, tau, tau_contracted,
                  _canonical_key, _canonical_fixed_point)

TAU_TILDE_NAME = "tau_tilde"


# ---------------------------------------------------------------------------
# D7.1.0 -- fragment line-graph data model
# ---------------------------------------------------------------------------
#
# D7 recognizes a dressed operator as a SUBGRAPH of a residual diagram's line
# graph.  For that, each operator (more precisely, each of its definition terms)
# must be expressed as a line-graph *fragment*: the same edge-list form as
# diagram.LineGraph, but OPEN -- its block indices are dangling "ports" that the
# match will wire to the rest of the residual, in place of LineGraph's "bra".
#
# Line format is deliberately identical to diagram.LineGraph.lines so a subgraph
# match (D7.2) runs on one homogeneous representation:
#
#     (species, endpoint_a, endpoint_b)      species in {"p", "h"}
#
# Endpoints:
#   ("factor", k)  -- the k-th factor of the definition term (v / t1 / tau / f)
#   ("port",   s)  -- a dangling block port; s is the slot position in the
#                     operator's block tuple (the index that connects outward)
#
# A line with two factor endpoints is INTERNAL (a summed index shared by two
# factors); a line with one port endpoint is DANGLING (a block index). Occupied
# indices are hole lines ("h"), virtual indices are particle lines ("p") -- the
# same species convention as the diagram engine.
#
# D7.1.0 is the data model only; the encoders that populate it are D7.1.1
# (single factor) and D7.1.2 (whole definition term).


@dataclass(frozen=True)
class FragmentLineGraph:
    """Open edge-list form of one dressed-operator definition term.

    ``lines`` is a tuple of ``(species, endpoint_a, endpoint_b)`` -- the same
    shape as :class:`ccgen.diagram.LineGraph`, so D7.2's subgraph match is a
    homogeneous graph match.  ``n_factors`` is the number of factor nodes
    (``("factor", 0..n_factors-1)``); ``n_ports`` is the operator's block size
    (``("port", 0..n_ports-1)``).  A line touching a ``("port", _)`` endpoint is
    dangling (a block index that wires to the rest of the residual); a line
    between two ``("factor", _)`` endpoints is internal (a summed index).

    ``factor_names[k]`` is the tensor species of factor node k (``v`` / ``t1`` /
    ``t2`` / ``tau`` / ``f`` / ...).  Load-bearing for D7.2: line topology alone
    does NOT distinguish ``t2*v`` from ``t1*t1*v`` (found in D7.1.4 -- they can
    wire identically), so a subgraph match must agree on factor species, not just
    wiring.
    """

    lines: tuple[tuple[str, object, object], ...]
    n_factors: int
    n_ports: int
    factor_names: tuple[str, ...] = ()

    def _is_port(self, e) -> bool:
        return isinstance(e, tuple) and len(e) == 2 and e[0] == "port"

    @property
    def internal_lines(self) -> tuple[tuple[str, object, object], ...]:
        """Lines between two factor nodes (summed indices)."""
        return tuple(l for l in self.lines
                     if not self._is_port(l[1]) and not self._is_port(l[2]))

    @property
    def dangling_lines(self) -> tuple[tuple[str, object, object], ...]:
        """Lines with a port endpoint (block indices)."""
        return tuple(l for l in self.lines
                     if self._is_port(l[1]) or self._is_port(l[2]))

    @property
    def port_species(self) -> dict[int, str]:
        """Slot -> species ("p"/"h") for each block port, read off the dangling
        lines.  The match must connect a residual line of the same species."""
        out: dict[int, str] = {}
        for sp, a, b in self.dangling_lines:
            port = a if self._is_port(a) else b
            out[port[1]] = sp
        return out


@dataclass(frozen=True)
class OperatorFragments:
    """A dressed operator encoded as line-graph fragments (D7.1.3 output).

    One ``(coeff, FragmentLineGraph)`` per defining term; ``name`` / ``block`` /
    ``uses`` carried from the :class:`DressedOperator` so the recognizer knows
    what it matched and the build-order dependencies.  Populated by
    :func:`operator_fragments` (D7.1.3); this dataclass is the D7.1.0 container.
    """

    name: str
    block: tuple[Index, ...]
    fragments: tuple[tuple[Fraction, FragmentLineGraph], ...]
    uses: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# D7.1.1 -- single-factor fragment encoder
# ---------------------------------------------------------------------------
#
# One factor's contribution to the fragment: one line per index.  A block index
# becomes a dangling line to its ("port", slot); a summed index becomes a
# half-line to a ("stub", name) endpoint that D7.1.2 later fuses with the
# matching stub on the partner factor (same summed name).  Species: occ -> "h"
# (hole), vir -> "p" (particle) -- the diagram engine's convention.
#
# The ("stub", name) endpoint is intentionally the raw INDEX NAME so the D7.1.2
# assembler can join two factors' stubs by name; it does not survive into the
# final FragmentLineGraph (which has only ("factor",_) and ("port",_) ends).


def _index_species(idx: Index) -> str:
    """Line species of an index: occupied -> hole "h", virtual -> particle "p"."""
    return "h" if idx.space == "occ" else "p"


def factor_to_fragment(factor, node, block):
    """D7.1.1: the lines contributed by a single factor at ``node`` = ("factor",
    k).  ``block`` is the operator's block tuple (index order fixes port slots).

    Returns a tuple of ``(species, node, other_end)`` lines -- one per factor
    index.  ``other_end`` is ``("port", slot)`` if the index is a block index
    (``slot`` = its position in ``block``), else ``("stub", index_name)`` for a
    summed index awaiting the D7.1.2 join.  A factor index that is neither in
    the block nor a genuine summed dummy would be a malformed definition; the
    caller (D7.1.2) supplies only well-formed definition terms."""
    block_slot = {idx.name: s for s, idx in enumerate(block)}
    lines = []
    for idx in factor.indices:
        sp = _index_species(idx)
        if idx.name in block_slot:
            lines.append((sp, node, ("port", block_slot[idx.name])))
        else:
            lines.append((sp, node, ("stub", idx.name)))
    return tuple(lines)


# ---------------------------------------------------------------------------
# D7.1.2 -- definition-term fragment assembler
# ---------------------------------------------------------------------------
#
# Compose the single-factor fragments of one definition term into a closed
# FragmentLineGraph, JOINING the ("stub", name) half-lines that share a summed
# index.  A summed index appears on exactly two factors (a contraction line), so
# its two stubs fuse into one internal line ("factor", a) <-> ("factor", b).
# tau is treated as one ATOMIC factor node (its own contraction to t2/t1t1 is
# D7.3's expansion, not D7.1's).


def term_to_fragment(term, block) -> FragmentLineGraph:
    """D7.1.2: assemble one definition term into a FragmentLineGraph.

    Each factor contributes its lines (D7.1.1); block indices land on ports, and
    each summed index -- which appears on exactly two factors -- fuses its two
    ("stub", name) half-lines into one internal factor<->factor line.  Raises if
    a summed name does not appear on exactly two factor endpoints (a malformed
    definition term: an uncontracted or over-contracted dummy)."""
    port_lines = []
    # stub_ends[name] collects (species, factor_node) for each half-line on `name`
    stub_ends: dict[str, list[tuple[str, object]]] = {}
    for k, factor in enumerate(term.factors):
        node = ("factor", k)
        for sp, nd, other in factor_to_fragment(factor, node, block):
            if other[0] == "port":
                port_lines.append((sp, nd, other))
            else:                                   # ("stub", name)
                stub_ends.setdefault(other[1], []).append((sp, nd))

    internal_lines = []
    for name, ends in stub_ends.items():
        if len(ends) != 2:
            raise ValueError(
                f"summed index {name!r} appears on {len(ends)} factor endpoints, "
                f"expected exactly 2 (a contraction line)")
        (sp_a, node_a), (sp_b, node_b) = ends
        if sp_a != sp_b:
            raise ValueError(
                f"summed index {name!r} joins mismatched species {sp_a}/{sp_b}")
        internal_lines.append((sp_a, node_a, node_b))

    return FragmentLineGraph(
        lines=tuple(internal_lines) + tuple(port_lines),
        n_factors=len(term.factors),
        n_ports=len(block),
        factor_names=tuple(f.name for f in term.factors),
    )


# ---------------------------------------------------------------------------
# D7.2.0 -- residual-term fragment encoding (the match substrate)
# ---------------------------------------------------------------------------
#
# The residual side of the D7.2 match is an AlgebraTerm too, so it encodes with
# the SAME machinery: its FREE (external) indices play the operator-block role
# (they become ports), and its summed indices become internal lines.  A summed
# index in a CC residual always appears on exactly two factors (a contraction
# is one edge -- verified across the CCSD singles+doubles manifold), so the
# term_to_fragment 2-endpoint invariant holds without special-casing.


def residual_term_to_fragment(term) -> FragmentLineGraph:
    """D7.2.0: encode a residual AlgebraTerm as a FragmentLineGraph, with its
    free indices as the ports.  This is the substrate a D7.2 subgraph match runs
    against; an operator fragment (D7.1.3) is recognized as a sub-fragment of it."""
    return term_to_fragment(term, term.free_indices)


# ---------------------------------------------------------------------------
# D7.2.2a -- factor-subset enumeration + species prefilter
# ---------------------------------------------------------------------------
#
# A match places the operator's n factor nodes onto n of the residual term's
# factors.  The cheap prefilter that bounds the search: only subsets whose
# factor-NAME multiset equals the operator's can possibly match (D7.1.4 -- a
# t2*v operator can only sit on a {t2,v} residual pair, never a {t1,v} one).
# The induced-subfragment + isomorphism test (D7.2.2b/c) then filters the
# survivors by wiring.


def candidate_factor_subsets(op_frag: FragmentLineGraph, term) -> list[tuple[int, ...]]:
    """D7.2.2a: the residual-factor index subsets whose factor-name multiset
    equals ``op_frag``'s.  Each is a sorted tuple of positions into
    ``term.factors``.  A cheap necessary condition for a match -- the wiring
    tests (D7.2.2b/c) run only on these survivors."""
    import itertools
    from collections import Counter

    want = Counter(op_frag.factor_names)
    n = len(op_frag.factor_names)
    out = []
    for combo in itertools.combinations(range(len(term.factors)), n):
        if Counter(term.factors[k].name for k in combo) == want:
            out.append(combo)
    return out


# ---------------------------------------------------------------------------
# D7.2.2b -- induced sub-fragment of a residual factor subset
# ---------------------------------------------------------------------------
#
# The sub-fragment a candidate subset induces: an index shared by TWO factors
# WITHIN the subset is an internal line; an index that also touches a factor
# OUTSIDE the subset, or is a residual external, is a dangling port (it connects
# the operator to the rest).  This is where the "extra shared line" case is made
# explicit -- if the subset shares MORE contraction lines than the operator has
# internal lines, the induced fragment carries them and D7.2.2c will reject the
# match (the l-line in t2(c,d,j,l) v(c,d,k,l), which shares c,d AND l).
#
# Factor nodes are renumbered 0..n-1 by position within the subset, so the
# induced FragmentLineGraph is directly comparable to an operator fragment.


def _induce(term, subset: tuple[int, ...]):
    """Core of D7.2.2b: return ``(FragmentLineGraph, slot_to_index_name)``.  The
    second value maps each port slot to the residual index name that dangles
    there -- needed so a match can report WHICH residual index each operator port
    bound to (D7.2.3 cross-term consistency)."""
    from collections import Counter

    local = {k: pos for pos, k in enumerate(subset)}   # residual pos -> local node
    within = Counter()
    for k in subset:
        for idx in term.factors[k].indices:
            within[idx.name] += 1

    internal_lines = []
    port_lines = []
    port_slot: dict[str, int] = {}
    seen_internal: set[str] = set()
    for k in subset:
        node = ("factor", local[k])
        for idx in term.factors[k].indices:
            sp = "h" if idx.space == "occ" else "p"
            if within[idx.name] == 2:
                if idx.name in seen_internal:
                    continue
                seen_internal.add(idx.name)
                others = [local[j] for j in subset
                          if j != k and any(i.name == idx.name
                                            for i in term.factors[j].indices)]
                internal_lines.append((sp, node, ("factor", others[0])))
            else:
                slot = port_slot.setdefault(idx.name, len(port_slot))
                port_lines.append((sp, node, ("port", slot)))

    frag = FragmentLineGraph(
        lines=tuple(internal_lines) + tuple(port_lines),
        n_factors=len(subset),
        n_ports=len(port_slot),
        factor_names=tuple(term.factors[k].name for k in subset),
    )
    slot_to_index = {slot: name for name, slot in port_slot.items()}
    return frag, slot_to_index


def induced_subfragment(term, subset: tuple[int, ...]) -> FragmentLineGraph:
    """D7.2.2b: the FragmentLineGraph induced on ``subset`` (positions into
    ``term.factors``).  Within-subset shared indices -> internal lines; every
    other index of a subset factor (shared with an outside factor, or external)
    -> a port.  Ports are numbered by first appearance so distinct outward
    indices get distinct slots."""
    return _induce(term, subset)[0]


# ---------------------------------------------------------------------------
# D7.2.2c -- fragment isomorphism test (the core)
# ---------------------------------------------------------------------------
#
# Does the operator fragment occur as EXACTLY the induced sub-fragment?  A match
# is a node bijection sigma (op factor -> induced factor) that agrees on factor
# species and carries the op's internal lines onto the induced internal lines
# AND the op's ports onto the induced ports, both as species-matched bijections.
# "Exactly" is load-bearing: the internal-line MULTISETS must be equal, so an
# induced fragment with an extra contraction line (the l-line) cannot match --
# it has an internal line the operator does not.  Bounded backtracking over the
# <=4 factor nodes.


def _line_multiset(lines, node_map=None):
    """Canonical multiset of internal lines under an optional node relabel.  Each
    internal line -> (species, frozenset of its two mapped factor-node ids)."""
    from collections import Counter
    out = []
    for sp, a, b in lines:
        na = node_map[a] if node_map else a
        nb = node_map[b] if node_map else b
        out.append((sp, frozenset((na, nb))))
    return Counter(out)


def fragments_match(op_frag: FragmentLineGraph, induced: FragmentLineGraph):
    """D7.2.2c: is ``op_frag`` isomorphic to ``induced`` (an exact induced-
    sub-fragment match)?  Returns a binding dict or None.

    The binding maps op ports to induced port slots:
    ``{"nodes": {op_node: induced_node}, "ports": {op_slot: induced_slot}}``.
    None if no species-consistent bijection carries the op's internal lines and
    ports onto the induced ones exactly (equal internal-line multisets -- an
    extra induced line rules the match out)."""
    from collections import Counter as _C
    import itertools

    n = op_frag.n_factors
    if (induced.n_factors != n or induced.n_ports != op_frag.n_ports
            or _C(op_frag.factor_names) != _C(induced.factor_names)):
        return None

    op_int = op_frag.internal_lines
    ind_int = induced.internal_lines
    if len(op_int) != len(ind_int):
        return None                                # different line count -> no

    op_names = op_frag.factor_names
    ind_names = induced.factor_names

    # candidate node maps: permutations that preserve factor species
    for perm in itertools.permutations(range(n)):
        node_map = {("factor", k): ("factor", perm[k]) for k in range(n)}
        if any(op_names[k] != ind_names[perm[k]] for k in range(n)):
            continue
        # internal lines must map exactly (species + endpoint set multiset)
        if _line_multiset(op_int, node_map) != _line_multiset(ind_int):
            continue
        # ports: match op port slots to induced port slots by (species, the
        # species of the factor node each attaches to)
        pm = _match_ports(op_frag, induced, node_map)
        if pm is None:
            continue
        return {"nodes": {("factor", k): ("factor", perm[k]) for k in range(n)},
                "ports": pm}
    return None


def _match_ports(op_frag, induced, node_map):
    """Bijection op-port-slot -> induced-port-slot consistent with species and
    the mapped factor node each dangling line attaches to.  None if impossible."""
    import itertools

    def port_desc(frag, mapped=None):
        # slot -> (species, mapped-factor-node) for each dangling line
        d = {}
        for sp, a, b in frag.dangling_lines:
            port = a if frag._is_port(a) else b
            node = b if frag._is_port(a) else a
            nn = mapped[node] if mapped else node
            d.setdefault(port[1], []).append((sp, nn))
        return d

    op_d = port_desc(op_frag, node_map)
    ind_d = port_desc(induced)
    op_slots = sorted(op_d)
    ind_slots = sorted(ind_d)
    if len(op_slots) != len(ind_slots):
        return None
    # each port here is a single dangling line (one index) -> desc list len 1
    for perm in itertools.permutations(ind_slots):
        ok = True
        for os_, is_ in zip(op_slots, perm):
            if sorted(op_d[os_]) != sorted(ind_d[is_]):
                ok = False
                break
        if ok:
            return {os_: is_ for os_, is_ in zip(op_slots, perm)}
    return None


def _all_port_bindings(op_frag, induced, node_map):
    """Like :func:`_match_ports` but yields EVERY valid op-slot -> induced-slot
    bijection, not just the first.  A symmetric fragment (e.g. bare v, all-hole
    ports, no internal lines) admits several -- each is a distinct block
    orientation the hypothesis enumeration (D7.2.3c-0) must try."""
    import itertools

    def port_desc(frag, mapped=None):
        d = {}
        for sp, a, b in frag.dangling_lines:
            port = a if frag._is_port(a) else b
            node = b if frag._is_port(a) else a
            nn = mapped[node] if mapped else node
            d.setdefault(port[1], []).append((sp, nn))
        return d

    op_d = port_desc(op_frag, node_map)
    ind_d = port_desc(induced)
    op_slots = sorted(op_d)
    ind_slots = sorted(ind_d)
    if len(op_slots) != len(ind_slots):
        return
    for perm in itertools.permutations(ind_slots):
        if all(sorted(op_d[os_]) == sorted(ind_d[is_])
               for os_, is_ in zip(op_slots, perm)):
            yield {os_: is_ for os_, is_ in zip(op_slots, perm)}


# ---------------------------------------------------------------------------
# D7.2.2d -- match driver
# ---------------------------------------------------------------------------


def match_fragment(op_frag: FragmentLineGraph, term) -> list[dict]:
    """D7.2.2d: every occurrence of ``op_frag`` in residual ``term``.  Composes
    the prefilter (D7.2.2a), the induced sub-fragment (D7.2.2b), and the exact
    isomorphism test (D7.2.2c).

    Each occurrence is a dict:
        ``subset``      -- residual factor positions the fragment sits on
        ``nodes``       -- op factor-node -> induced (subset-local) factor-node
        ``port_index``  -- op port slot -> the RESIDUAL index NAME it bound to
                           (so D7.2.3 can check the same operator block binds
                           consistently across all its defining-term fragments)."""
    out = []
    for subset in candidate_factor_subsets(op_frag, term):
        induced, slot_to_index = _induce(term, subset)
        binding = fragments_match(op_frag, induced)
        if binding is None:
            continue
        port_index = {op_slot: slot_to_index[ind_slot]
                      for op_slot, ind_slot in binding["ports"].items()}
        out.append({
            "subset": subset,
            "nodes": binding["nodes"],
            "port_index": port_index,
        })
    return out


# ---------------------------------------------------------------------------
# D7.2.3a -- per-fragment occurrence collection
# ---------------------------------------------------------------------------


def collect_fragment_occurrences(op: "DressedOperator", terms) -> list[dict]:
    """D7.2.3a: fan ``match_fragment`` out over every residual term for each of
    the operator's tau-expanded defining fragments.

    Returns a flat list of occurrence dicts, one per (fragment, term, subset)
    hit, each carrying enough to group into whole-operator instances (D7.2.3c)
    and later rewrite (D7.3):
        ``frag_id``    -- index into the tau-expanded fragment list
        ``op_coeff``   -- the fragment's coefficient in the operator definition
        ``term_id``    -- index into ``terms``
        ``term_coeff`` -- the residual term's coefficient
        ``subset``     -- residual factor positions the fragment sits on
        ``port_index`` -- op port slot -> residual index name."""
    frags = tau_expanded_operator_fragments(op).fragments
    out = []
    for term_id, term in enumerate(terms):
        for frag_id, (op_coeff, fr) in enumerate(frags):
            for m in match_fragment(fr, term):
                out.append({
                    "frag_id": frag_id,
                    "op_coeff": op_coeff,
                    "term_id": term_id,
                    "term_coeff": term.coeff,
                    "subset": m["subset"],
                    "port_index": m["port_index"],
                })
    return out


# ---------------------------------------------------------------------------
# D7.2.3b -- hypothesize W*rest from an anchor fragment match
# ---------------------------------------------------------------------------
#
# The 5 defining fragments of one operator instance scatter across residual
# terms with different bindings (D7.2.3a finding), so they cannot be grouped
# structurally.  Instead HYPOTHESIZE: from ONE anchor fragment match, build the
# dressed term  (term_coeff / anchor_op_coeff) * W(bound block) * rest,  where
# `rest` is the residual factors outside the anchor subset and the block indices
# come from the anchor's port_index.  expand_dressed_term then regenerates all
# the operator's raw pieces (D7.2.3c verifies they are in the residual).
#
# The coefficient divides out the anchor fragment's own coefficient in the
# operator definition: if the anchor is Wmnij's bare-v piece (op_coeff 1) sitting
# in a residual term with coeff 1/2, the operator prefactor is 1/2.


def hypothesize_operator_term(op: "DressedOperator", occurrence: dict, term):
    """D7.2.3b: build the dressed AlgebraTerm ``c * W(block) * rest`` implied by
    one anchor fragment ``occurrence`` (from :func:`collect_fragment_occurrences`)
    in residual ``term``.

    ``c = term.coeff / occurrence["op_coeff"]`` -- the residual coefficient with
    the anchor fragment's operator-internal coefficient divided out.  The W factor
    carries the residual indices its block bound to (via ``port_index``); ``rest``
    is the residual factors outside the anchor subset.  Returns the AlgebraTerm to
    feed :func:`expand_dressed_term`."""
    name_to_index = {}
    for fac in term.factors:
        for idx in fac.indices:
            name_to_index[idx.name] = idx
    block_indices = tuple(name_to_index[occurrence["port_index"][s]]
                          for s in range(len(op.block)))
    antisym = op.antisym_groups   # derived from the block, not hardcoded
    w_factor = Tensor(op.name, block_indices,
                      antisym_groups=antisym if antisym else None)
    rest = tuple(f for k, f in enumerate(term.factors)
                 if k not in set(occurrence["subset"]))
    coeff = term.coeff / occurrence["op_coeff"]
    return AlgebraTerm(
        coeff=coeff,
        factors=(w_factor,) + rest,
        free_indices=term.free_indices,
        summed_indices=term.summed_indices,
        connected=term.connected,
    )


# ---------------------------------------------------------------------------
# D7.2.3c-0 -- hypothesis enumeration (block orientation x rest interpretation)
# ---------------------------------------------------------------------------
#
# A single anchor UNDERDETERMINES the hypothesis (found while scoping D7.2.3c):
#   (1) block orientation -- a symmetric fragment (bare v: all-hole ports) admits
#       several port bindings; match_fragment returns only one, but the CORRECT
#       Wmnij orientation binds (m,n)->summed and (i,j)->external, a DIFFERENT
#       orientation.  Enumerate all valid orientations.
#   (2) rest interpretation -- the true rest is often a DRESSED tau (Wmnij*tau),
#       not a raw t2; the raw residual carries tau only as t2 + t1t1.  So offer
#       both rest=t2 and rest=tau for a single-t2 rest.
# D7.2.3c-1 then verifies each candidate; the correct one passes, the rest fail.


def enumerate_hypotheses(op: "DressedOperator", occurrence: dict, term):
    """D7.2.3c-0: candidate ``W*rest`` dressed terms for one anchor occurrence,
    over {block orientations} x {rest as-is, rest-as-tau}.

    Yields AlgebraTerms.  Orientation comes from every valid port binding of the
    anchor fragment on its subset (not just the one match_fragment returned); the
    tau-rest variant replaces a single raw ``t2`` rest factor with ``tau`` (the
    residual carries tau expanded, so the true rest may be dressed)."""
    frags = tau_expanded_operator_fragments(op).fragments
    op_frag = frags[occurrence["frag_id"]][1]
    op_coeff = occurrence["op_coeff"]
    subset = occurrence["subset"]
    induced, slot_to_index = _induce(term, subset)
    # node map: the isomorphism that placed the fragment (recompute)
    binding = fragments_match(op_frag, induced)
    if binding is None:
        return
    name_to_index = {}
    for fac in term.factors:
        for idx in fac.indices:
            name_to_index[idx.name] = idx
    rest = tuple(f for k, f in enumerate(term.factors) if k not in set(subset))
    coeff = term.coeff / op_coeff
    n = len(op.block)
    antisym = op.antisym_groups or None   # derived from the block, not hardcoded
    asym_block = _block_is_asymmetric(op)

    for port_map in _all_port_bindings(op_frag, induced, binding["nodes"]):
        port_index = {os_: slot_to_index[is_] for os_, is_ in port_map.items()}
        block_indices = tuple(name_to_index[port_index[s]] for s in range(n))
        w = Tensor(op.name, block_indices, antisym_groups=antisym)
        # For an asymmetric-block operator (Wmbej, ovvo) the block reorientation
        # this binding applies carries the bare-v antisymmetry sign, which the
        # bare coeff omits (D7.2.5.3 S1).  Fold it in -- but ONLY for asymmetric
        # blocks: for oooo/vvvv every genuine orientation is bare-v sign +1, and
        # signing there would rescue the spurious same-space-swap orientations the
        # unsigned filter correctly rejects (verified: would double Wmnij/Wabef).
        sign = _binding_sign(block_indices) if asym_block else 1
        w_indices = {idx.name for idx in w.indices}
        summed = {idx.name for idx in term.summed_indices}
        for rest_variant in _rest_variants(rest, w_indices, summed):
            yield AlgebraTerm(
                coeff=coeff * sign, factors=(w,) + rest_variant,
                free_indices=term.free_indices,
                summed_indices=term.summed_indices, connected=term.connected)


def _block_is_asymmetric(op) -> bool:
    """True if the operator's rank-4 block has a mixed-space (asymmetric) bra or
    ket pair (e.g. Wmbej ovvo).  Such a block has no bare-v-sign-+1 canonical
    orientation, so its genuine bindings need the reorientation sign folded in;
    oooo/vvvv blocks never do (D7.2.5.3 S1)."""
    ss = op.space_sig()
    return len(ss) == 4 and (ss[0] != ss[1] or ss[2] != ss[3])


def _binding_sign(block_indices) -> int:
    """The bare-v antisymmetry sign of this block orientation -- the parity the
    bra/ket intra-pair antisymmetry of the operator's defining ERI picks up when
    reoriented to `block_indices`.  Computed self-calibrating as the canonical
    signed coefficient of a bare v over these indices."""
    from ccgen.tensors import v as _vfac
    vt = AlgebraTerm(coeff=Fraction(1), factors=(_vfac(*block_indices),),
                     free_indices=tuple(block_indices), summed_indices=(),
                     connected=True)
    return _eri_canonical(vt)[1]


def _rest_variants(rest, op_indices=None, summed=None):
    """The rest as-is, plus a tau-dressed variant if it is a single ``t2``
    factor (the true rest of an operator collapse is often the pseudo-amplitude
    tau, which the raw residual only ever carries expanded).

    The tau variant is a ``tau_c`` (half written-t1t1 weight) when the t2's bra
    (virtual) pair is SUMMED and lies INSIDE the operator's block -- i.e. the
    pair is antisymmetrically contracted into the operator's own v, which then
    supplies the P(t1t1) partner the standard doubled representative would
    double-count (Wabef).  Otherwise a plain ``tau`` (weight 2, e.g. Wmnij whose
    tau bra pair is the external doubles indices).  Threading the choice here --
    rather than inspecting the expanded term -- is load-bearing: after operator
    expansion a rest-tau and the operator's own definition-tau coexist in one
    term and are otherwise indistinguishable."""
    yield rest
    if len(rest) == 1 and rest[0].name == "t2":
        idx = rest[0].indices
        bra = {idx[0].name, idx[1].name}
        contracted = (op_indices is not None and summed is not None
                      and bra <= op_indices and bra <= summed)
        yield (tau_contracted(*idx),) if contracted else (tau(*idx),)


# ---------------------------------------------------------------------------
# D7.2.3c-1 -- sound containment verify
# ---------------------------------------------------------------------------
#
# A hypothesis W*rest is CONSISTENT with the residual iff every primitive its
# expansion produces is present in the residual, with the same sign and a
# magnitude no larger than the residual's for that primitive.  This is a SOUND
# NECESSARY filter, not an exactness check: a primitive shared by several
# operator instances carries only PART of the residual coefficient in any one
# hypothesis (measured: 2 of Wmnij*tau's 10 keys are half the residual's), so
# requiring equality would wrongly reject the correct hypothesis.  The exact
# arbiter is the whole-equation verify_dressed_equation at D7.3; here we only
# reject hypotheses that produce a primitive ABSENT from the residual or of the
# WRONG SIGN (a false anchor / wrong orientation), which the correct one never
# does.


def hypothesis_is_consistent(hyp, residual_terms, operators=None) -> bool:
    """D7.2.3c-1: is the dressed term ``hyp`` consistent with ``residual_terms``?

    Expands ``hyp`` to primitives and requires every ERI-canonical key to be
    present in the residual with the same sign and ``|hyp_coeff| <= |raw_coeff|``.
    True for the correct ``W*rest``, False for a wrong orientation / rest (a key
    absent from the residual or of the wrong sign)."""
    from fractions import Fraction
    from .dressed_equation import expand_dressed_term, raw_multiset
    ops = operators or {hyp.factors[0].name: _seeded_by_name(hyp.factors[0].name)}
    raw = raw_multiset(residual_terms)
    acc: dict[tuple, Fraction] = {}
    for prim in expand_dressed_term(hyp, ops):
        key, coeff = _eri_canonical(prim)
        acc[key] = acc.get(key, Fraction(0)) + coeff
    for key, coeff in acc.items():
        if coeff == 0:
            continue
        rc = raw.get(key)
        if rc is None:                              # primitive not in residual
            return False
        if (rc > 0) != (coeff > 0):                 # wrong sign
            return False
        if abs(coeff) > abs(rc):                    # over-covers a shared key
            return False
    return True


def _seeded_by_name(name: str) -> "DressedOperator":
    """The seeded operator with this name (helper for the default operator
    table of a single-operator hypothesis)."""
    for op in seeded_operators():
        if op.name == name:
            return op
    raise KeyError(f"no seeded operator named {name!r}")


# ---------------------------------------------------------------------------
# D7.2.3d -- find_operator_occurrences driver
# ---------------------------------------------------------------------------
#
# Enumerate anchors -> enumerate_hypotheses -> hypothesis_is_consistent, then
# dedup.  A partial hypothesis (Wmnij*t2 or Wmnij*t1t1) covers a SUBSET of the
# primitives the complete one (Wmnij*tau) covers -- measured tau_cover =
# t2_cover | t1t1_cover.  So keep only MAXIMAL-cover hypotheses: those whose
# primitive-cover is not a subset of another consistent hypothesis's.  That
# selects the complete dressing (rest = tau) with no arbitrary preference rule.


def _hypothesis_cover(hyp, op) -> frozenset:
    """The ERI-canonical primitive keys ``hyp`` accounts for, CLOSED under the
    residual's external-pair antisymmetry (D7.2.5.2 W3).

    A single written t1t1 representative covers one residual term but not its
    antisym partner (Wabef's term 28 = the i<->j swap of term 27), which would
    otherwise resurface as a spurious standalone occurrence.  A dressed
    occurrence physically REPLACES both partners, so the cover includes both: for
    each expanded primitive we also add the keys reached by exchanging the
    hypothesis's free (external) pairs -- (i,j) occ and (a,b) vir, the pairs R2
    is antisymmetric in -- throughout the whole primitive."""
    from .dressed_equation import expand_dressed_term
    swaps = _external_antisym_pairs(hyp)
    ks = set()
    for prim in expand_dressed_term(hyp, {op.name: op}):
        for variant in _with_pair_swaps(prim, swaps):
            key, coeff = _eri_canonical(variant)
            if coeff != 0:
                ks.add(key)
    return frozenset(ks)


def _external_antisym_pairs(hyp):
    """The hypothesis's free (external) indices grouped by space, taken
    pairwise -- the pairs the residual is antisymmetric in (doubles: (i,j) occ,
    (a,b) vir).  Exchanging such a pair throughout a primitive reaches the
    antisym-partner term the single written t1t1 representative omitted."""
    by_space: dict[str, list[Index]] = {}
    for idx in hyp.free_indices:
        by_space.setdefault(idx.space, []).append(idx)
    pairs = []
    for idxs in by_space.values():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                pairs.append((idxs[x], idxs[y]))
    return tuple(pairs)


def _with_pair_swaps(prim, swaps):
    """Yield ``prim`` and every variant with one of ``swaps`` (an index pair)
    exchanged throughout all factors."""
    yield prim
    for x, y in swaps:
        ren = {x: y, y: x}
        new_factors = tuple(
            f.with_indices(tuple(ren.get(idx, idx) for idx in f.indices))
            for f in prim.factors
        )
        yield prim.with_factors(new_factors)


def _antisym_sort_factor(factor: Tensor) -> tuple[Tensor, int]:
    """Reorder ``factor``'s indices to sorted order WITHIN each antisym group,
    returning ``(factor, sign)`` with sign = parity of the reordering.

    Folds an antisym factor's own symmetry to a slot-canonical form so two
    orientations related by intra-group swaps (e.g. Wmnij(k,l,i,j) vs the even
    double-swap Wmnij(l,k,j,i)) reach the same index order.  Non-antisym /
    rank-<2 factors are returned unchanged, sign +1."""
    groups = factor.antisym_groups
    if not groups:
        return factor, 1
    order = list(range(len(factor.indices)))  # slot -> source position
    sign = 1
    for group in groups:
        pos = list(group)
        want = sorted(pos, key=lambda p: (factor.indices[p].space,
                                          factor.indices[p].name))
        # write sorted sources back into this group's own slots (keeps groups
        # in place); parity = inversions of the sorted-source sequence
        sign *= _perm_parity(tuple(want))
        for slot, src in zip(pos, want):
            order[slot] = src
    new_idx = tuple(factor.indices[order[s]] for s in range(len(order)))
    return factor.with_indices(new_idx), sign


def _dressed_canonical_key(term: AlgebraTerm) -> tuple:
    """Antisym-canonical key for a dressed ``W*rest`` term.

    Sorts each antisym factor's indices within its groups (folding the sign),
    normalizes free-index listing order, then dummy-relabels to a fixed point.
    Two occurrences that are the SAME operator instance written in
    antisym-equivalent orientations map to the same key (D7.2.5 gap 2 dedup)."""
    new_factors = []
    for f in term.factors:
        nf, _ = _antisym_sort_factor(f)
        new_factors.append(nf)
    folded = term.with_factors(tuple(new_factors))
    return _canonical_key(_canonical_fixed_point(_free_order_normalized(folded)))


def find_operator_occurrences(op: "DressedOperator", terms) -> list[dict]:
    """D7.2.3d: the verified occurrences of ``op`` in ``terms``.

    Enumerates every anchor's hypotheses (D7.2.3c-0), keeps the consistent ones
    (D7.2.3c-1), and dedups to MAXIMAL primitive covers -- discarding a partial
    hypothesis whose cover is contained in a fuller one (so the complete
    ``W*tau`` wins over ``W*t2`` / ``W*t1t1``).  Antisym-equivalent orientations
    of the same instance (D7.2.5 gap 2, exposed by the v-parity sign fold) are
    folded via :func:`_dressed_canonical_key` so each instance is one
    occurrence.  Each returned occurrence is a dict
    ``{"term": AlgebraTerm, "cover": frozenset}`` -- the dressed ``W*rest`` term
    for D7.3 to rewrite, plus the residual primitives it accounts for."""
    consistent = []
    for anchor in collect_fragment_occurrences(op, terms):
        for hyp in enumerate_hypotheses(op, anchor, terms[anchor["term_id"]]):
            if hypothesis_is_consistent(hyp, terms):
                consistent.append((hyp, _hypothesis_cover(hyp, op)))

    # keep only maximal covers (drop any strictly contained in another)
    covers = [c for _, c in consistent]
    occurrences = []
    seen: set = set()
    for hyp, cover in consistent:
        if any(cover < other for other in covers):   # strictly contained
            continue
        key = _dressed_canonical_key(hyp)
        if key in seen:
            continue
        seen.add(key)
        occurrences.append({"term": hyp, "cover": cover})
    return occurrences


# ---------------------------------------------------------------------------
# D7.3.0b -- P-branch consolidation
# ---------------------------------------------------------------------------
#
# An operator's multiple occurrences in a residual are the branches of a single
# antisymmetrized dressed term: e.g. Fae's `+Fae(b,c)t2(a,c,ij)` and
# `-Fae(a,c)t2(b,c,ij)` are `P(ab)[Fae(b,c)t2(a,c,ij)]`, and Wmbej's four are
# `P(ij)P(ab)[...]`.  Treating them as independent occurrences double-counts the
# primitives they share (the non-antisymmetrized common part), the same-operator
# half of the D7.3.0 over-count.  Consolidation folds the branches back into one
# `{base term, antisymmetrizer pairs}` so downstream assembly counts the shared
# part once.


def consolidate_p_branches(op: "DressedOperator", occurrences) -> list[dict]:
    """Group ``occurrences`` of ``op`` into antisymmetrized dressed terms.

    Two occurrences belong to the same P-group iff one is a signed external-pair
    (i<->j / a<->b) image of the other -- checked on the dressed-canonical key.
    Each returned group is ``{"base": AlgebraTerm, "antisym_pairs": (...),
    "branch_signs": {pair-subset: sign}}``: the base branch, the external pairs it
    is antisymmetrized over (the pairs whose swap maps the base onto ANOTHER
    branch), and the sign each pair-subset image carries (== the branch coeff /
    base coeff).  A lone occurrence yields a group with no antisym pairs."""
    from itertools import chain, combinations

    def powerset(s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    remaining = list(occurrences)
    groups: list[dict] = []
    while remaining:
        base_occ = remaining.pop(0)
        base = base_occ["term"]
        base_key = _dressed_canonical_key(base)
        all_pairs = _external_antisym_pairs(base)
        # map every P-image key of base -> the antisym sign of that image
        image_sign = {}
        for subset in powerset(all_pairs):
            img, sgn = _apply_external_swaps(base, subset)
            image_sign.setdefault(_dressed_canonical_key(img), (subset, sgn))
        # collect branches (base + any remaining occ that is a P-image)
        members = [base_occ]
        active_pairs = set()
        still = []
        for occ in remaining:
            k = _dressed_canonical_key(occ["term"])
            if k in image_sign and k != base_key:
                subset, sgn = image_sign[k]
                # verify the branch's coeff matches the antisym sign
                if occ["term"].coeff == base.coeff * sgn:
                    members.append(occ)
                    active_pairs.update(subset)
                    continue
            still.append(occ)
        remaining = still
        groups.append({
            "base": base,
            "antisym_pairs": tuple(sorted(active_pairs,
                                          key=lambda p: (p[0].space, p[0].name))),
            "branches": len(members),
            "cover": frozenset().union(*(m["cover"] for m in members)),
        })
    return groups


def _operator_unit_expansion(op, terms):
    """The summed ERI-canonical primitive contribution of ALL of ``op``'s
    occurrences in ``terms``, at their recognized coefficients (scale 1)."""
    from .dressed_equation import expand_dressed_term
    acc: dict[tuple, Fraction] = {}
    for occ in find_operator_occurrences(op, terms):
        for prim in expand_dressed_term(occ["term"], {op.name: op}):
            key, coeff = _eri_canonical(prim)
            if coeff:
                acc[key] = acc.get(key, Fraction(0)) + coeff
    return acc


def reconcile_operator_scales(operators, terms):
    """D7.3.0c-1: per-operator coefficient scales that remove the CROSS-operator
    over-count from nesting (Fme nested in Fae/Fmi).

    Operators that nest inside another (Fme, via the enclosing operator's
    ``-1/2 f*t1`` = Fme-correction) are over-counted when recognized standalone:
    the enclosing operator already accounts for part of the shared primitive.  The
    correct scale for a nested operator is the COMPLEMENT -- what the residual
    needs beyond what the outer operators (at their already-fixed scales) supply.

    Processes operators in dependency order (roots at scale 1, then inners), and
    for each inner operator solves ``scale = (raw - already_accounted) / own`` on
    every key it touches.  Self-calibrating (the 1/2 for Fme is derived, not
    hardcoded); requires a single consistent scale across the operator's keys
    (raises if inconsistent -- that would mean the nesting model is wrong for it).
    Returns ``{op_name: Fraction}``.  Only 0c-1 (Fme nesting); the residual
    {Wabef,Wmnij} tau-overlap (0c-2) and Fmi tau-tilde tail (0d) are separate."""
    from .dressed_equation import raw_multiset
    raw = raw_multiset(terms)
    units = {op.name: _operator_unit_expansion(op, terms) for op in operators}
    order = _operator_dependency_order(operators)  # roots first, inners last
    scale = {op.name: Fraction(1) for op in operators}
    for name in order:
        if _is_nesting_root(name, operators):
            continue  # roots stay at 1
        needed: set[Fraction] = set()
        for key, own in units[name].items():
            if own == 0:
                continue
            accounted = sum(scale[o] * units[o].get(key, Fraction(0))
                            for o in units if o != name)
            needed.add((raw.get(key, Fraction(0)) - accounted) / own)
        if len(needed) > 1:
            raise ValueError(
                f"operator {name} has no single consistent nesting scale: {needed}"
            )
        if needed:
            scale[name] = next(iter(needed))
    return scale


def _operator_dependency_order(operators):
    """Operator names outer(root)->inner, so an inner operator's scale is solved
    after the operators that enclose it are fixed.  Fme is inner to Fae/Fmi (they
    carry the -1/2 f*t1 Fme-correction); everything else is a root here."""
    roots = [op.name for op in operators if _is_nesting_root(op.name, operators)]
    inners = [op.name for op in operators
              if not _is_nesting_root(op.name, operators)]
    return roots + inners


def _is_nesting_root(name, operators):
    """True unless ``name`` is nested inside another operator's definition as a
    same-block correction (currently only Fme, whose bare f(occ,vir) block appears
    as a `-1/2 f*t1`-style correction term inside Fae and Fmi)."""
    by_name = {op.name: op for op in operators}
    target = by_name.get(name)
    if target is None or target.rank != 2:
        return True  # only rank-2 Fock ops can nest as an f-correction
    tgt_bare = target.definition_terms[0]
    if not (len(tgt_bare.factors) == 1 and tgt_bare.factors[0].name == "f"):
        return True
    tgt_spaces = tuple(i.space for i in tgt_bare.factors[0].indices)
    for op in operators:
        if op.name == name:
            continue
        for t in op.definition_terms:
            fs = [f for f in t.factors if f.name == "f" and len(t.factors) > 1]
            if any(tuple(i.space for i in f.indices) == tgt_spaces for f in fs):
                return False  # name's bare block appears as a correction elsewhere
    return True


def _operator_tau_role(op, terms):
    """Which pseudo-amplitude an operator's occurrences carry: 'tau' (external,
    written weight 2), 'tau_c' (contracted, weight 1), or None.  Structural --
    read from the recognized occurrence rest factors, not the operator name."""
    for occ in find_operator_occurrences(op, terms):
        for f in occ["term"].factors[1:]:
            if f.name in (TAU_NAME, TAU_CONTRACTED_NAME):
                return f.name
    return None


def tau_overlap_corrections(operators, terms, scales=None):
    """D7.3.0c-2: per-primitive corrections for the τ/τ_c cross-operator overlap.

    A primitive shared between a τ-operator (Wmnij, external τ, written weight 2)
    and a τ_c-operator (Wabef, contracted, weight 1) is over-counted when both
    land their t1t1 half on it: the raw residual writes that shared primitive
    ONCE, but both dressed terms produce it.  On such a key the τ-operator's
    contribution is exactly DOUBLE the τ_c-operator's (the weight-2 vs weight-1
    t1t1 half), and ``raw == τ-operator's contribution`` -- i.e. the external-τ
    operator OWNS the shared primitive and the τ_c operator's duplicate must be
    removed.  Genuinely-additive shared keys (the τ-t2 pieces, where the two
    contributions are EQUAL, ratio 1) are left untouched.

    Returns ``{key: delta}`` to ADD to the assembled recon (delta is negative --
    it subtracts the redundant τ_c contribution).  Derived, not raw-peeking: the
    fired keys are exactly those with τ/τ_c ratio == 2, and the subtracted amount
    is the τ_c operator's own contribution."""
    scales = scales or {op.name: Fraction(1) for op in operators}
    tau_expand: dict[tuple, Fraction] = {}
    tauc_expand: dict[tuple, Fraction] = {}
    for op in operators:
        role = _operator_tau_role(op, terms)
        if role is None:
            continue
        target = tau_expand if role == TAU_NAME else tauc_expand
        s = scales.get(op.name, Fraction(1))
        for key, coeff in _operator_unit_expansion(op, terms).items():
            target[key] = target.get(key, Fraction(0)) + coeff * s

    corrections: dict[tuple, Fraction] = {}
    for key, tau_c in tauc_expand.items():
        tau_v = tau_expand.get(key, Fraction(0))
        if tau_c == 0 or tau_v == 0:
            continue
        # fire only on the doubling (t1t1-half) overlap, ratio exactly 2
        if tau_v == 2 * tau_c:
            corrections[key] = -tau_c   # remove the redundant tau_c duplicate
    return corrections


def _apply_external_swaps(term, pairs):
    """Return (term with each (x,y) in ``pairs`` exchanged throughout, sign),
    where sign = (-1)**len(pairs) -- each external-pair swap is a P antisymmetry."""
    ren = {}
    for x, y in pairs:
        ren[x] = y
        ren[y] = x
    new_factors = tuple(
        f.with_indices(tuple(ren.get(i, i) for i in f.indices))
        for f in term.factors
    )
    swapped = AlgebraTerm(
        coeff=term.coeff, factors=new_factors,
        free_indices=term.free_indices, summed_indices=term.summed_indices,
        connected=term.connected,
    )
    return swapped, (-1) ** len(pairs)


# ---------------------------------------------------------------------------
# D7.1.3 -- operator fragment set
# ---------------------------------------------------------------------------


def fragment_signature(fr: FragmentLineGraph) -> tuple:
    """A match-relevant canonical signature of a fragment: the factor-species
    multiset, the internal-line species multiset (summed contractions), and the
    per-slot (port, species, factor-species-at-the-other-end) wiring.  Two
    fragments with the same signature are indistinguishable to a D7.2 subgraph
    match; D7.1.4 requires distinct definition terms to have distinct signatures
    (no false-collision).  Factor-node identities are abstracted to their species
    so the signature is invariant under factor relabeling but sensitive to WHAT
    each node is -- the property line topology alone lacked (D7.1.4 finding)."""
    names = fr.factor_names
    factor_species = tuple(sorted(names))
    internal = tuple(sorted(sp for sp, _, _ in fr.internal_lines))
    # port wiring: (slot, species, species-of-the-factor the port attaches to)
    port_wiring = []
    for sp, a, b in fr.dangling_lines:
        port = a if fr._is_port(a) else b
        node = b if fr._is_port(a) else a
        node_species = names[node[1]] if node[0] == "factor" and node[1] < len(names) else "?"
        port_wiring.append((port[1], sp, node_species))
    return (factor_species, internal, tuple(sorted(port_wiring)))


def operator_fragments(op: "DressedOperator") -> OperatorFragments:
    """D7.1.3: encode a whole dressed operator as line-graph fragments -- one
    ``(coeff, FragmentLineGraph)`` per defining term.  This is the D7.1
    deliverable: the representation D7.2's subgraph match consumes.  Carries the
    operator's name / block / uses so the recognizer knows what it matched."""
    frags = tuple((term.coeff, term_to_fragment(term, op.block))
                  for term in op.definition_terms)
    return OperatorFragments(name=op.name, block=op.block,
                             fragments=frags, uses=op.uses)


def tau_tilde(a: Index, b: Index, i: Index, j: Index) -> Tensor:
    """The tau~ pseudo-amplitude tau~_{ij}^{ab} = t2 + 1/2 P(t1 t1).

    Same index layout / antisymmetry as t2 and tau; differs from tau only in
    the t1t1 weight (1/2 * 1/2 vs tau's 1/2), which the definition machinery
    accounts for.  Used by the F-operators.
    """
    return Tensor(TAU_TILDE_NAME, (a, b, i, j), antisym_groups=((0, 1), (2, 3)))


@dataclass(frozen=True)
class DressedOperator:
    """A seeded dressed operator: its name, index block, and definition.

    ``block`` is the canonical free-index tuple of the operator (e.g. the four
    occupied indices of Wmnij), in the order the operator's factor carries.
    ``definition_terms`` are the AlgebraTerms that sum to the operator over that
    block -- transcribed from the spin-orbital CCSD equations.  ``uses`` names
    the pseudo-amplitudes the definition references (``tau`` / ``tau_tilde``),
    so a caller knows the build-order dependencies.
    """

    name: str
    block: tuple[Index, ...]
    definition_terms: tuple[AlgebraTerm, ...]
    uses: frozenset[str] = frozenset()

    @property
    def rank(self) -> int:
        return len(self.block)

    @property
    def antisym_groups(self) -> tuple[tuple[int, int], ...]:
        """Index-slot antisymmetry of the operator's own factor, derived from
        its block (NOT hardcoded per call site).

        A dressed W operator is antisymmetric WITHIN its bra pair and WITHIN its
        ket pair (like tau: Wmnij(m,n,i,j) = -Wmnij(n,m,i,j)) -- NOT under the
        ERI bra<->ket exchange, which is a symmetry of the integral, not of a
        dressed operator.  A pair is antisymmetric only when BOTH its slots are
        the same index space: Wmnij(oooo)/Wabef(vvvv) get ((0,1),(2,3)), but
        Wmbej(ovvo) has mixed pairs and gets () -- stamping (0,1)/(2,3) on it
        would be a FALSE symmetry.  Rank-2 operators (Fme/Fae/Fmi) get ().
        """
        if self.rank != 4:
            return ()
        spaces = [idx.space for idx in self.block]
        groups = []
        for a, b in ((0, 1), (2, 3)):
            if spaces[a] == spaces[b]:
                groups.append((a, b))
        return tuple(groups)

    def space_sig(self) -> str:
        return "".join(
            "o" if idx.space == "occ" else "v" if idx.space == "vir" else "g"
            for idx in self.block
        )


# ---------------------------------------------------------------------------
# A2.0 -- the seeded operator family, as data
# ---------------------------------------------------------------------------
#
# Canonical block indices per operator.  Dummy (summed) indices inside a
# definition term use fresh letters; the block indices are the operator's own
# externals.


def _term(coeff, factors, free, summed) -> AlgebraTerm:
    return AlgebraTerm(
        coeff=Fraction(coeff),
        factors=tuple(factors),
        free_indices=tuple(free),
        summed_indices=tuple(summed),
        connected=True,
    )


def _build_fme() -> DressedOperator:
    m, e = make_occ("m"), make_vir("e")
    n, ff = make_occ("n", dummy=True), make_vir("f", dummy=True)
    block = (m, e)
    return DressedOperator(
        name="Fme",
        block=block,
        definition_terms=(
            _term(1, (f(m, e),), block, ()),
            _term(1, (t1(ff, n), v(m, n, e, ff)), block, (n, ff)),
        ),
    )


def _build_fae() -> DressedOperator:
    a, e = make_vir("a"), make_vir("e")
    m, n = make_occ("m", dummy=True), make_occ("n", dummy=True)
    ff = make_vir("f", dummy=True)
    block = (a, e)
    return DressedOperator(
        name="Fae",
        block=block,
        uses=frozenset({TAU_TILDE_NAME}),
        definition_terms=(
            _term(1, (f(a, e),), block, ()),
            _term(Fraction(-1, 2), (f(m, e), t1(a, m)), block, (m,)),
            _term(1, (t1(ff, m), v(m, a, ff, e)), block, (m, ff)),
            _term(Fraction(-1, 2), (tau_tilde(a, ff, m, n), v(m, n, e, ff)),
                  block, (m, n, ff)),
        ),
    )


def _build_fmi() -> DressedOperator:
    m, i = make_occ("m"), make_occ("i")
    n = make_occ("n", dummy=True)
    e, ff = make_vir("e", dummy=True), make_vir("f", dummy=True)
    block = (m, i)
    return DressedOperator(
        name="Fmi",
        block=block,
        uses=frozenset({TAU_TILDE_NAME}),
        definition_terms=(
            _term(1, (f(m, i),), block, ()),
            _term(Fraction(1, 2), (f(m, e), t1(e, i)), block, (e,)),
            _term(1, (t1(e, n), v(m, n, i, e)), block, (n, e)),
            _term(Fraction(1, 2), (tau_tilde(e, ff, i, n), v(m, n, e, ff)),
                  block, (n, e, ff)),
        ),
    )


def _build_wmnij() -> DressedOperator:
    m, n, i, j = make_occ("m"), make_occ("n"), make_occ("i"), make_occ("j")
    e, ff = make_vir("e", dummy=True), make_vir("f", dummy=True)
    block = (m, n, i, j)
    return DressedOperator(
        name="Wmnij",
        block=block,
        uses=frozenset({TAU_NAME}),
        definition_terms=(
            _term(1, (v(m, n, i, j),), block, ()),
            # P(ij) t1_je <mn||ie>
            _term(1, (t1(e, j), v(m, n, i, e)), block, (e,)),
            _term(-1, (t1(e, i), v(m, n, j, e)), block, (e,)),
            _term(Fraction(1, 4), (tau(e, ff, i, j), v(m, n, e, ff)),
                  block, (e, ff)),
        ),
    )


def _build_wabef() -> DressedOperator:
    a, b, e, ff = make_vir("a"), make_vir("b"), make_vir("e"), make_vir("f")
    m, n = make_occ("m", dummy=True), make_occ("n", dummy=True)
    block = (a, b, e, ff)
    return DressedOperator(
        name="Wabef",
        block=block,
        uses=frozenset({TAU_NAME}),
        definition_terms=(
            _term(1, (v(a, b, e, ff),), block, ()),
            # -P(ab) t1_mb <am||ef>
            _term(-1, (t1(b, m), v(a, m, e, ff)), block, (m,)),
            _term(1, (t1(a, m), v(b, m, e, ff)), block, (m,)),
            _term(Fraction(1, 4), (tau(a, b, m, n), v(m, n, e, ff)),
                  block, (m, n)),
        ),
    )


def _build_wmbej() -> DressedOperator:
    m, b, e, j = make_occ("m"), make_vir("b"), make_vir("e"), make_occ("j")
    n = make_occ("n", dummy=True)
    ff = make_vir("f", dummy=True)
    block = (m, b, e, j)
    return DressedOperator(
        name="Wmbej",
        block=block,
        definition_terms=(
            _term(1, (v(m, b, e, j),), block, ()),
            _term(1, (t1(ff, j), v(m, b, e, ff)), block, (ff,)),
            _term(-1, (t1(b, n), v(m, n, e, j)), block, (n,)),
            _term(Fraction(-1, 2), (t2(ff, b, j, n), v(m, n, e, ff)),
                  block, (n, ff)),
            _term(-1, (t1(ff, j), t1(b, n), v(m, n, e, ff)),
                  block, (n, ff)),
        ),
    )


def seeded_operators() -> list[DressedOperator]:
    """The full seeded CCSD dressed-operator family (A2.0)."""
    return [
        _build_fme(),
        _build_fae(),
        _build_fmi(),
        _build_wmnij(),
        _build_wabef(),
        _build_wmbej(),
    ]


def operator_to_intermediate_spec(op: DressedOperator, canonical_fock: bool = False):
    """D7.3.1: bridge a recognized ``DressedOperator`` to the ``IntermediateSpec``
    the emit pipeline (``emit_planck_translation_unit(intermediates=...)``)
    materializes into a ``build_<name>`` function.

    Generalizes ``tau.tau_intermediate_spec`` (proven end-to-end for tau) to the
    seeded W/F family.  The mapping is direct -- ``op.definition_terms`` are
    already the right ``AlgebraTerm`` shape, ``op.block`` -> ``indices``,
    ``op.space_sig()`` -> ``index_space_sig`` (same convention tau uses, e.g.
    "vvoo") -- so this carries no algebra of its own; correctness is the
    faithfulness gate (the spec expands to the same primitives as the operator).

    ``usage_count`` / ``usage_targets`` are left at their defaults here (0 / ())
    -- they are per-residual annotation, filled by the usage pass (D7.3.1d) from
    the P-branch-consolidated occurrences.

    ``canonical_fock=True`` drops Brillouin-zero ``f_ov``/``f_vo`` definition
    terms (Planck always feeds a canonical Fock, so those are runtime-inert; see
    ``generate._drops_under_canonical_fock``).  Under it, Fme collapses to its
    ``t1*oovv`` piece and Fae/Fmi lose their ``f_ov*t1`` corrections, while
    diagonal-``f`` and tau/tau_tilde terms survive.
    """
    from .intermediates import IntermediateSpec
    from ..generate import _drops_under_canonical_fock

    terms = op.definition_terms
    if canonical_fock:
        terms = tuple(t for t in terms if not _drops_under_canonical_fock(t))

    return IntermediateSpec(
        name=op.name,
        indices=tuple(op.block),
        definition_terms=terms,
        usage_count=0,
        index_space_sig=op.space_sig(),
        usage_targets=(),
    )


# ---------------------------------------------------------------------------
# A2.1 -- definition self-consistency gate
# ---------------------------------------------------------------------------


def operator_definition_is_consistent(op: DressedOperator) -> bool:
    """A2.1 -- is the operator's transcribed definition self-consistent?

    Catches a mis-transcribed formula before it is ever matched against a
    residual (the operator analog of A1.4).  Requires:

      * every definition term carries exactly the operator's free block (same
        index set, so the terms genuinely define a tensor over that block),
      * every term canonicalizes without collapsing to zero,
      * no two DISTINCT-looking terms canonicalize to the same structure with
        coefficients that cancel to zero (a sign/coefficient transcription slip
        that would silently drop a defining contribution).

    Returns True iff the definition passes.
    """
    if not op.definition_terms:
        return False

    block_set = frozenset(op.block)
    acc: dict[tuple[object, ...], Fraction] = {}
    for term in op.definition_terms:
        if frozenset(term.free_indices) != block_set:
            return False
        ct = _canonical_fixed_point(term)
        if ct.coeff == 0:
            return False
        key = _canonical_key(ct)
        acc[key] = acc.get(key, Fraction(0)) + ct.coeff

    # Every distinct defining structure must survive with a non-zero net
    # coefficient; a cancellation to zero means a transcription error dropped a
    # real contribution.
    return all(c != 0 for c in acc.values())


# ---------------------------------------------------------------------------
# A2.2 -- operator footprint (the index A2.3 / A3 match residual terms against)
# ---------------------------------------------------------------------------
#
# When operator W sits inside a residual term ``c * W(block) * rest``, expanding
# W yields ``c * defn_term_k * rest`` for each defining term.  A residual term
# is "a piece of W" if, ignoring its contraction with rest, it has the SAME
# non-block factor structure and the SAME wiring of W's block indices as some
# defn_term_k.  The footprint captures exactly that, per defining term, in a
# form independent of which concrete rest / dummy letters the residual used.
#
# Representation: the canonical key of the defining term with the operator's
# block indices rewritten to positional slot tokens ($0, $1, ...).  Two
# structures share a footprint entry iff their non-block factors AND their
# block-slot wiring agree -- rest and dummy names are already normalized away by
# the canonical key.


def _slotize_key(
    key: tuple[object, ...],
    block: tuple[Index, ...],
) -> tuple[object, ...]:
    """Rewrite block index NAMES in a canonical key to positional slot tokens.

    ``key`` is a ``_canonical_key`` result: nested tuples of (space, name)
    pairs.  Block index names become ``"$k"`` for the k-th block slot; every
    other name is left as-is (it is a canonicalized dummy, already normalized).
    Space is preserved so a slot cannot match the wrong index space.
    """
    slot_of = {idx.name: f"${k}" for k, idx in enumerate(block)}

    def _rewrite(node):
        if isinstance(node, tuple):
            if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], str):
                space, name = node
                return (space, slot_of.get(name, name))
            return tuple(_rewrite(x) for x in node)
        return node

    return _rewrite(key)


@dataclass(frozen=True)
class OperatorFootprint:
    """Per-definition-term footprint of one operator (A2.2).

    ``entries`` maps each slotized definition-term key to the coefficient that
    defining term carries.  A2.4 uses the coefficients to verify a matched
    occurrence is COMPLETE (all defining pieces present with the right weights);
    A2.3 uses the keys to find candidate pieces.
    """

    name: str
    block: tuple[Index, ...]
    entries: dict[tuple[object, ...], Fraction]


def operator_footprint(op: DressedOperator) -> OperatorFootprint:
    """A2.2 -- build the slotized footprint of an operator's definition."""
    entries: dict[tuple[object, ...], Fraction] = {}
    for term in op.definition_terms:
        ct = _canonical_fixed_point(term)
        key = _slotize_key(_canonical_key(ct), op.block)
        entries[key] = entries.get(key, Fraction(0)) + ct.coeff
    entries = {k: c for k, c in entries.items() if c != 0}
    return OperatorFootprint(name=op.name, block=op.block, entries=entries)


def footprints_are_distinct(operators: Sequence[DressedOperator]) -> bool:
    """Do the seeded operators have non-overlapping footprint entry sets?

    Two operators sharing a footprint entry would make an ambiguous match in
    A2.3.  The bare-integral seed term of each operator (its <block>||<block>
    ERI) is the natural discriminator; this checks the whole entry sets are
    pairwise disjoint.
    """
    seen: dict[tuple[object, ...], str] = {}
    for op in operators:
        fp = operator_footprint(op)
        for key in fp.entries:
            if key in seen and seen[key] != op.name:
                return False
            seen[key] = op.name
    return True


# ---------------------------------------------------------------------------
# A2.3 -- occurrence detection (read-only, conservative over-approximation)
# ---------------------------------------------------------------------------
#
# Scan a residual term list and report which terms could be a piece of an
# operator's definition.  Detection is deliberately a SOUND OVER-APPROXIMATION:
# a term is a candidate piece of defining term d if its factor multiset contains
# d's operator-piece factors with matching (name, space-pattern).  It never
# misses a real piece, but may over-report; A2.4 validates each candidate group
# exactly (all defining pieces present, coefficients consistent) before anything
# is collapsible.  Keeping A2.3 conservative isolates the exact/expensive check
# to A2.4 and the isomorphism rewrite to A3.


def _factor_shape(factor: Tensor) -> tuple[str, tuple[str, ...]]:
    """(name, space-pattern) of a factor -- e.g. ('v', ('o','o','v','v'))."""
    return (
        factor.name,
        tuple(
            "o" if x.space == "occ" else "v" if x.space == "vir" else "g"
            for x in factor.indices
        ),
    )


def _entry_factor_shapes(key: tuple[object, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Factor shapes of a footprint entry key ((name, ((space,slot),...)),...)."""
    shapes = []
    for name, indices in key[0]:
        pattern = tuple(space for space, _slot in indices)
        pattern = tuple("o" if s == "occ" else "v" if s == "vir" else "g" for s in pattern)
        shapes.append((name, pattern))
    return tuple(sorted(shapes))


def _term_factor_shapes(term: AlgebraTerm) -> list[tuple[str, tuple[str, ...]]]:
    return sorted(_factor_shape(f) for f in term.factors)


def _multiset_contains(
    haystack: Sequence[tuple[str, tuple[str, ...]]],
    needle: Sequence[tuple[str, tuple[str, ...]]],
) -> bool:
    """Is ``needle`` a sub-multiset of ``haystack``?"""
    from collections import Counter

    h = Counter(haystack)
    n = Counter(needle)
    return all(h.get(shape, 0) >= count for shape, count in n.items())


@dataclass(frozen=True)
class OperatorPieceMatch:
    """A residual term reported as a candidate piece of one operator entry.

    ``term_index`` is the position in the scanned residual list; ``entry_key``
    is the matched footprint entry; ``entry_coeff`` is that defining piece's
    coefficient (for A2.4's completeness check).
    """

    operator: str
    term_index: int
    entry_key: tuple[object, ...]
    entry_coeff: Fraction


def find_operator_pieces(
    terms: Sequence[AlgebraTerm],
    op: DressedOperator,
) -> list[OperatorPieceMatch]:
    """A2.3 -- report residual terms that could be pieces of ``op``.

    For each residual term and each footprint entry, report a candidate match
    when the term's factor shapes contain the entry's operator-piece factor
    shapes.  Read-only, sound over-approximation (see module note).
    """
    fp = operator_footprint(op)
    entry_shapes = {key: _entry_factor_shapes(key) for key in fp.entries}

    matches: list[OperatorPieceMatch] = []
    for k, term in enumerate(terms):
        term_shapes = _term_factor_shapes(term)
        for key, coeff in fp.entries.items():
            if _multiset_contains(term_shapes, entry_shapes[key]):
                matches.append(
                    OperatorPieceMatch(
                        operator=op.name,
                        term_index=k,
                        entry_key=key,
                        entry_coeff=coeff,
                    )
                )
    return matches


# ---------------------------------------------------------------------------
# A2.4 -- completeness classification (the hand-off seam to A3)
# ---------------------------------------------------------------------------
#
# Group A2.3's candidate pieces into occurrences and classify each as COMPLETE
# (every defining footprint entry of the operator has at least one matched
# piece in the group) or PARTIAL (some defining entry is unmatched, so the
# group can never assemble the full operator).
#
# Scope boundary, stated honestly: A2.4 classifies at the granularity A2.3
# provides -- factor SHAPES, not index bindings.  "COMPLETE" here means
# "coverage-complete: worth A3's exact binding + coefficient check", NOT
# "proven-collapsible".  The exact firewall (bind block slots, reconstruct
# c * operator * rest, verify coefficients) requires subgraph-isomorphism
# binding and lives in A3.  A2.4 is sound in the safe direction: PARTIAL is a
# definitive reject (a missing defining piece can never be recovered), so A3
# only ever sees coverage-complete candidates.


def _rest_signature(term: AlgebraTerm, entry_shapes: Sequence) -> tuple[object, ...]:
    """Signature of a term's ``rest`` after removing one operator-piece worth of
    factors (by shape).  Groups pieces of the same occurrence together.

    Removes, for each operator-piece factor shape, one matching factor from the
    term; the remaining factors' shapes plus the term's free-index space pattern
    form the rest signature.  Shape-level (no binding) -- consistent with A2.3.
    """
    from collections import Counter

    remaining = list(term.factors)
    for shape in entry_shapes:
        for k, fac in enumerate(remaining):
            if _factor_shape(fac) == shape:
                del remaining[k]
                break
    rest_shapes = tuple(sorted(_factor_shape(f) for f in remaining))
    free_pattern = tuple(
        "o" if x.space == "occ" else "v" if x.space == "vir" else "g"
        for x in term.free_indices
    )
    return (rest_shapes, free_pattern)


@dataclass(frozen=True)
class OperatorOccurrence:
    """A group of candidate pieces classified for collapsibility (A2.4).

    ``matched_entries`` are the operator footprint entries covered by pieces in
    this group; ``covered`` is True iff that set is the operator's FULL entry
    set (coverage-complete).  ``term_indices`` are the residual positions in the
    group.  See the A2.4 module note: covered means "worth A3's exact check",
    not "proven collapsible".
    """

    operator: str
    rest_signature: tuple[object, ...]
    term_indices: tuple[int, ...]
    matched_entries: frozenset
    covered: bool


def classify_operator_occurrences(
    terms: Sequence[AlgebraTerm],
    op: DressedOperator,
) -> list[OperatorOccurrence]:
    """A2.4 -- group candidate pieces by rest and classify coverage.

    Returns one OperatorOccurrence per distinct rest signature.  ``covered`` is
    True iff the group's matched footprint entries equal the operator's full
    entry set.  Read-only.
    """
    fp = operator_footprint(op)
    full_entries = frozenset(fp.entries)
    entry_shapes = {key: _entry_factor_shapes(key) for key in fp.entries}
    matches = find_operator_pieces(terms, op)

    groups: dict[tuple[object, ...], dict] = {}
    for m in matches:
        rest = _rest_signature(terms[m.term_index], entry_shapes[m.entry_key])
        g = groups.setdefault(
            rest, {"indices": set(), "entries": set()}
        )
        g["indices"].add(m.term_index)
        g["entries"].add(m.entry_key)

    occurrences: list[OperatorOccurrence] = []
    for rest, g in groups.items():
        entries = frozenset(g["entries"])
        occurrences.append(
            OperatorOccurrence(
                operator=op.name,
                rest_signature=rest,
                term_indices=tuple(sorted(g["indices"])),
                matched_entries=entries,
                covered=(entries == full_entries),
            )
        )
    return occurrences


# ---------------------------------------------------------------------------
# A3.1 -- single definition-term binding (subgraph match, one entry)
# ---------------------------------------------------------------------------
#
# Bind one operator definition term against a residual term: find every way to
# identify a subset of the residual's factors with the definition's factors,
# unifying indices.  A definition term's FREE indices are the operator's block
# (exported as the slot binding); its SUMMED indices are internal dummies
# (bound consistently but not exported).  Definition terms have 1-2 factors, so
# the backtracking search is tiny.


# Full permutational symmetry of the antisymmetrized ERI <pq||rs>: the two
# intra-pair antisymmetries AND bra<->ket exchange (pq <-> rs).  This is the
# 8-fold group the emitter's _ERI_SYMMETRY_PERMUTATIONS also uses -- crucially
# it includes the (2,3,0,1) exchange, which reconciles the textbook operator
# definitions' ERI arrangement (<oo||ov>) with the pipeline's raw-residual
# arrangement (<ov||oo>).  Without the exchange, binding could never match the
# W-operator t1-pieces against the real residual (the A3.2 wall).
_ERI_PERMUTATIONS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _perm_parity(perm: tuple[int, ...]) -> int:
    """Sign (+1/-1) of a permutation given as the image tuple perm[i] = source
    position feeding output slot i.  Counts inversions."""
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return -1 if inv & 1 else 1


def _eri_normalize_factor(factor: Tensor) -> tuple[Tensor, int]:
    """Reorder a v factor's indices to a canonical arrangement under 8-fold ERI
    symmetry, returning ``(reordered_factor, sign)``.  Non-v factors are
    returned unchanged with sign +1.

    Picks the permutation whose resulting (space, index-name) sequence is
    lexicographically smallest, so any two exchange-related v arrangements map
    to the SAME order.  The indices themselves are unchanged (just reordered),
    so the downstream dummy relabel still converges two equivalent terms.

    ``v`` is antisymmetric in each bra/ket pair, so a permutation reached by an
    ODD number of intra-pair swaps carries a -1; the returned ``sign`` is the
    parity of the chosen permutation, which the caller folds into the term
    coefficient.  Discarding it (the pre-D7.2.5 behavior) compared v factors
    reachable only by an odd swap with the WRONG sign, silently rejecting
    correct Fae/Wabef hypotheses.
    """
    if factor.name != "v" or len(factor.indices) != 4:
        return factor, 1
    best = None
    for perm in _ERI_PERMUTATIONS:
        order = tuple(factor.indices[p] for p in perm)
        sig = tuple((x.space, x.name) for x in order)
        if best is None or sig < best[0]:
            best = (sig, order, _perm_parity(perm))
    return factor.with_indices(best[1]), best[2]


def _eri_normalize_term(term: AlgebraTerm) -> AlgebraTerm:
    """Normalize every v factor in a term to its canonical ERI arrangement,
    folding each reordering's antisymmetry parity into the coefficient."""
    new_factors = []
    sign = 1
    for f in term.factors:
        nf, s = _eri_normalize_factor(f)
        new_factors.append(nf)
        sign *= s
    out = term.with_factors(tuple(new_factors))
    return out if sign == 1 else out.scaled(sign)


def _free_order_normalized(term: AlgebraTerm) -> AlgebraTerm:
    """Relabel a term's FREE indices to canonical names by sorted order.

    Two terms with the same free-index *set* but listed in a different order
    (e.g. R2 externals as (a,b,i,j) vs the generated residual's (i,j,a,b))
    otherwise get different canonical keys, because the key records free indices
    positionally.  Renames each free index to a fixed reserved token by its
    sorted (space, name) rank, consistently across the factors, so listing order
    no longer matters.  Dummies are untouched (the downstream relabel handles
    them).
    """
    ordered = sorted(term.free_indices, key=lambda x: (x.space, x.name))
    ren: dict[Index, Index] = {}
    for rank, idx in enumerate(ordered):
        ren[idx] = Index(f"$free{rank}", idx.space, False)
    if not ren:
        return term
    new_factors = reindex_tensors(term.factors, ren)
    new_free = tuple(ren[i] for i in ordered)
    return AlgebraTerm(
        coeff=term.coeff, factors=new_factors, free_indices=new_free,
        summed_indices=term.summed_indices, connected=term.connected,
    )


def _eri_canonical(term: AlgebraTerm) -> tuple[tuple, Fraction]:
    """(ERI-canonical key, signed coefficient) for a term.

    Folds v's bra<->ket exchange symmetry (which _canonical_key alone does not,
    since ccgen's v carries only intra-pair antisymmetry) so the pipeline's raw
    residual and the dressed reconstruction -- which write the same integral in
    exchange-related arrangements -- compare equal.  Also normalizes free-index
    listing order so two terms with the same externals in a different order
    compare equal.  Any antisymmetry sign from reordering v is folded into the
    returned coefficient via the fixed point.

    Ordering is load-bearing (D7.2.5.2 Fmi): the bra<->ket normalization
    (_eri_normalize_factor) picks the lexicographically smallest (space, name)
    arrangement, so it MUST run on CANONICAL index names -- otherwise two terms
    that are the same integral but carry differently-named dummies (Fmi's
    t1(e,n)v(m,n,i,e) vs the residual's t1(b,k)v(i,b,j,k)) normalize their v to
    DIFFERENT orientations and never fold.  So: relabel dummies to a fixed point
    FIRST, then fold bra<->ket, then settle the fixed point again to absorb any
    normalization-induced sign/relabel.
    """
    settled = _canonical_fixed_point(_free_order_normalized(term))
    folded = _eri_normalize_term(settled)
    cf = _canonical_fixed_point(folded)
    return _canonical_key(cf), cf.coeff


def _eri_canonical_key(term: AlgebraTerm) -> tuple:
    return _eri_canonical(term)[0]


def _antisym_permutations(factor: Tensor):
    """Yield index-order permutations of a factor allowed by its symmetry.

    For ``v`` (the antisymmetrized ERI) this is the FULL 8-fold ERI symmetry
    group -- intra-pair antisymmetry plus bra<->ket exchange -- because two v
    factors that differ by any of those permutations are the same integral (up
    to a sign, which structural binding ignores).  For every other tensor the
    symmetry is just its ``antisym_groups`` (free permutation within each
    group).  Always includes the identity; yields tuples of indices in permuted
    order, de-duplicated.
    """
    from itertools import permutations, product

    n = len(factor.indices)

    if factor.name == "v" and n == 4:
        seen = set()
        for perm in _ERI_PERMUTATIONS:
            order = tuple(factor.indices[p] for p in perm)
            if order not in seen:
                seen.add(order)
                yield order
        return

    group_perms = []
    for group in factor.antisym_groups:
        group_perms.append([
            dict(zip(group, perm)) for perm in permutations(group)
        ])

    if not group_perms:
        yield tuple(factor.indices)
        return

    for combo in product(*group_perms):
        pos_map = {}
        for mapping in combo:
            pos_map.update(mapping)
        order = [
            factor.indices[pos_map.get(p, p)] for p in range(n)
        ]
        yield tuple(order)


def _bind_factor(
    defn_factor: Tensor,
    res_factor: Tensor,
    binding: dict[Index, Index],
) -> list[dict[Index, Index]]:
    """Extend ``binding`` by matching ``defn_factor`` to ``res_factor``.

    Antisymmetry-aware: tries every index order of the residual factor allowed
    by its antisymmetry groups (the definition and residual carry the same
    antisym structure, so permuting one side suffices).  Returns the list of
    successfully extended bindings -- possibly several (distinct permutations),
    empty on no match.  A definition index already bound must bind consistently;
    spaces must agree.
    """
    if defn_factor.name != res_factor.name:
        return []
    if len(defn_factor.indices) != len(res_factor.indices):
        return []

    outs: list[dict[Index, Index]] = []
    seen: set[tuple] = set()
    for res_order in _antisym_permutations(res_factor):
        out = dict(binding)
        ok = True
        for di, ri in zip(defn_factor.indices, res_order):
            if di.space != ri.space:
                ok = False
                break
            if di in out:
                if out[di] != ri:
                    ok = False
                    break
            else:
                out[di] = ri
        if not ok:
            continue
        key = tuple(sorted((k.name, v.name) for k, v in out.items()))
        if key not in seen:
            seen.add(key)
            outs.append(out)
    return outs


def bind_definition_term(
    defn_term: AlgebraTerm,
    res_term: AlgebraTerm,
) -> list[dict[Index, Index]]:
    """A3.1 -- all block bindings embedding ``defn_term`` into ``res_term``.

    Returns a list of ``{block_index -> residual_index}`` maps, one per distinct
    embedding of the definition's factors into a factor-subset of the residual
    term.  Only the block (definition free) indices are exported; internal
    dummies are unified during the search but dropped from the result.

    An empty list means the definition term does not embed in the residual term
    (this is the common case -- most residual terms are not a piece of a given
    operator).  Multiple maps mean an ambiguous embedding, which A3.2 must
    reconcile across the whole occurrence.
    """
    block = set(defn_term.free_indices)
    defn_factors = list(defn_term.factors)
    res_factors = list(res_term.factors)

    results: list[dict[Index, Index]] = []
    seen: set[tuple] = set()

    def _search(di: int, used: frozenset[int], binding: dict[Index, Index]) -> None:
        if di == len(defn_factors):
            block_binding = {k: v for k, v in binding.items() if k in block}
            key = tuple(sorted((k.name, v.name) for k, v in block_binding.items()))
            if key not in seen:
                seen.add(key)
                results.append(block_binding)
            return
        dfac = defn_factors[di]
        for ri, rfac in enumerate(res_factors):
            if ri in used:
                continue
            for extended in _bind_factor(dfac, rfac, binding):
                _search(di + 1, used | {ri}, extended)

    _search(0, frozenset(), {})
    return results


# ---------------------------------------------------------------------------
# A3.2 -- global occurrence binding (unify per-term bindings)
# ---------------------------------------------------------------------------
#
# Given an operator and a set of residual terms that might form ONE instance of
# it, find a single block->index map under which EVERY definition term of the
# operator binds to some residual term in the set.  This is the isomorphism
# core: A3.1 gives per-(defn,residual) bindings; A3.2 finds one global binding
# consistent across all definition terms at once (or proves none exists).
#
# Search strategy: seed candidate global bindings from the most constrained
# definition term (fewest admitted bindings across the residual set -- usually
# the bare ERI), then verify each remaining definition term binds to some
# residual under that exact binding.


def _bindings_for_defn(
    defn_term: AlgebraTerm,
    res_terms: Sequence[AlgebraTerm],
) -> list[tuple[int, dict[Index, Index]]]:
    """All (residual_index, block_binding) pairs for one definition term."""
    out: list[tuple[int, dict[Index, Index]]] = []
    for ri, rt in enumerate(res_terms):
        for b in bind_definition_term(defn_term, rt):
            out.append((ri, b))
    return out


def _binding_key(binding: dict[Index, Index]) -> tuple:
    return tuple(sorted((k.name, v.name) for k, v in binding.items()))


@dataclass(frozen=True)
class OccurrenceBinding:
    """A global block-binding under which an operator instance is present.

    ``binding`` maps each block index name to the residual index name.
    ``coverage`` maps each definition-term position to the residual-term index
    (within the given res_terms list) it binds to under this global binding --
    the fragments that would collapse into this operator instance.
    """

    operator: str
    binding: tuple[tuple[str, str], ...]
    coverage: tuple[tuple[int, int], ...]  # (defn_term_index, res_term_index)


def bind_occurrence(
    op: DressedOperator,
    res_terms: Sequence[AlgebraTerm],
) -> list[OccurrenceBinding]:
    """A3.2 -- global bindings covering all of ``op``'s definition terms.

    Returns every distinct block-binding under which each definition term of
    ``op`` binds to at least one residual term in ``res_terms``.  Empty if no
    such global binding exists (the operator instance is not assemblable from
    this fragment set).  Read-only.
    """
    defn_terms = list(op.definition_terms)
    if not defn_terms:
        return []

    # Per-definition admitted bindings; the seed is the one with the fewest.
    per_defn = [_bindings_for_defn(dt, res_terms) for dt in defn_terms]
    if any(not opts for opts in per_defn):
        return []  # some definition term has no residual match at all

    seed_idx = min(range(len(defn_terms)), key=lambda k: len(per_defn[k]))

    results: list[OccurrenceBinding] = []
    seen: set[tuple] = set()

    for seed_ri, seed_binding in per_defn[seed_idx]:
        key = _binding_key(seed_binding)
        if key in seen:
            continue
        # Verify every definition term binds to some residual under THIS binding.
        coverage: list[tuple[int, int]] = []
        ok = True
        for di in range(len(defn_terms)):
            match_ri = next(
                (ri for ri, b in per_defn[di] if _binding_key(b) == key),
                None,
            )
            if match_ri is None:
                ok = False
                break
            coverage.append((di, match_ri))
        if not ok:
            continue
        seen.add(key)
        results.append(
            OccurrenceBinding(
                operator=op.name,
                binding=key,
                coverage=tuple(coverage),
            )
        )
    return results


# ---------------------------------------------------------------------------
# A3.2 (completion) -- tau-expanded operator variant
# ---------------------------------------------------------------------------
#
# The raw residual carries no literal `tau` factor (A3.0.c proved tau is not
# collapsible there by residue-pairing).  So an operator whose definition
# references tau/tau_tilde -- Wmnij, Wabef, Fae, Fmi -- cannot bind its tau
# piece against raw doubles directly.  The fix is to bind against a variant
# operator whose tau/tau_tilde factors are pre-expanded into their
# t2 + written*t1t1 pieces, matching the raw residual's un-collapsed form.  No
# tau pre-collapse is then needed, and A3.0's dead embedded-tau path is bypassed
# entirely.


def _expand_pseudo_amplitude_in_term(
    term: AlgebraTerm,
) -> list[AlgebraTerm]:
    """Expand one tau / tau_tilde factor in a term into its t2 + t1t1 pieces.

    tau      = t2 + 2  * t1(a,i) t1(b,j)   (written single-rep weight)
    tau_tilde = t2 + 1 * t1(a,i) t1(b,j)   (half the t1t1 weight)

    Returns the (one or two) expansion terms; a term with no pseudo-amplitude
    is returned unchanged.  Only the first pseudo-amplitude factor is expanded;
    call repeatedly (fixed point) if a term carries several.
    """
    tilde_weight = TAU_SPEC.written_t1t1_weight / 2  # tau_tilde carries half
    # tau_c (contracted tau) carries HALF the standard written t1t1 weight: when
    # tau's bra pair is summed and antisym-contracted into the dressed operator's
    # own v (Wabef's tau(c,d,i,j) into v(c,d,a,b)), the v's antisymmetry already
    # supplies the second P(t1t1) permutation, so the standard doubled
    # representative over-counts.  This weight rides on the FACTOR NAME (threaded
    # from _rest_variants), NOT inspected from the term -- because a rest-tau_c and
    # an operator-definition tau (which keeps weight 2) coexist in the same term
    # after operator expansion, so no local term inspection can tell them apart.
    # See tau.py::TAU_CONTRACTED_NAME and D7.2.5.2 V0.4.
    weights = {
        TAU_NAME: TAU_SPEC.written_t1t1_weight,
        TAU_TILDE_NAME: tilde_weight,
        TAU_CONTRACTED_NAME: TAU_SPEC.written_t1t1_weight / 2,
    }

    pos = next(
        (k for k, f in enumerate(term.factors) if f.name in weights), None
    )
    if pos is None:
        return [term]

    pf = term.factors[pos]
    a, b, i, j = pf.indices
    others = tuple(f for k, f in enumerate(term.factors) if k != pos)
    w = weights[pf.name]

    def _mk(coeff, amp):
        return AlgebraTerm(
            coeff=coeff, factors=amp + others,
            free_indices=term.free_indices, summed_indices=term.summed_indices,
            connected=term.connected,
        )

    return [
        _mk(term.coeff, (t2(a, b, i, j),)),
        _mk(term.coeff * w, (t1(a, i), t1(b, j))),
    ]


def tau_expanded_operator(op: DressedOperator) -> DressedOperator:
    """Return a variant of ``op`` with every tau/tau_tilde piece expanded.

    Binding against this variant matches the raw (un-collapsed) residual, since
    the residual never carries a literal tau factor.  The variant keeps the same
    name and block; only the definition terms change.
    """
    new_terms: list[AlgebraTerm] = []
    for term in op.definition_terms:
        frontier = [term]
        # fixed point: expand until no pseudo-amplitude factor remains
        changed = True
        while changed:
            changed = False
            nxt: list[AlgebraTerm] = []
            for t in frontier:
                expanded = _expand_pseudo_amplitude_in_term(t)
                if len(expanded) != 1 or expanded[0] is not t:
                    changed = changed or (len(expanded) != 1)
                nxt.extend(expanded)
            frontier = nxt
        new_terms.extend(frontier)
    return DressedOperator(
        name=op.name, block=op.block,
        definition_terms=tuple(new_terms), uses=frozenset(),
    )


# ---------------------------------------------------------------------------
# D7.2.1 -- tau-expanded operator fragments (the raw-tensor pattern set)
# ---------------------------------------------------------------------------
#
# The raw residual carries NO literal tau/tau_tilde (a doubles term is t2*v or
# t1*t1*v, never tau*v), so the operator patterns a D7.2 match looks for must be
# in raw tensors.  D7.2.1 = operator_fragments o tau_expanded_operator: expand
# every pseudo-amplitude into t2 + t1t1 (fixed point), then encode the resulting
# raw-tensor definition terms as fragments.


def tau_expanded_operator_fragments(op: "DressedOperator") -> OperatorFragments:
    """D7.2.1: the operator's defining terms, tau/tau_tilde expanded to raw
    tensors, encoded as line-graph fragments.  This is the pattern set a D7.2.2
    match searches for in the raw residual (which never carries a literal tau).
    ``uses`` comes back empty -- the expanded form references no pseudo-amplitude."""
    return operator_fragments(tau_expanded_operator(op))


# ---------------------------------------------------------------------------
# A3.3 -- exact-coefficient firewall for a bound occurrence
# ---------------------------------------------------------------------------
#
# A3.2 gives a structural binding; it is coeff-blind, so a binding can be
# structurally valid yet numerically wrong (a covered fragment carries a
# coefficient inconsistent with a single instance scalar c).  A3.3 is the
# firewall: given the operator, the residual, and a global binding, reconstruct
# the operator instance c * op(bound_block) * rest and require that expanding it
# reproduces EXACTLY the residual fragments it claims (Fraction coefficients,
# with fracturing handled by summing residual fragments per canonical key).
#
# This is A1.4 / A1.6 generalized to a bound operator instance.  A "verified"
# occurrence is genuinely collapsible; a rejected one is left for A3.4 to skip.


def _apply_binding_to_term(
    defn_term: AlgebraTerm,
    binding_map: dict[str, str],
    rest_factors: tuple[Tensor, ...],
    rest_summed: tuple[Index, ...],
    rest_free: tuple[Index, ...],
    index_by_name: dict[str, Index],
) -> AlgebraTerm:
    """Instantiate one (tau-expanded) definition term under a block binding.

    Renames the definition's block indices to the residual indices named by
    ``binding_map`` (internal dummies kept, but disambiguated from rest names),
    appends ``rest_factors``, and builds the full residual-space term.
    """
    # Rename block indices via the binding; leave internal dummies as-is but
    # ensure they do not collide with rest/free names (prefix with '__').
    ren: dict[Index, Index] = {}
    block_targets = set(binding_map.values())
    for idx in defn_term.free_indices:  # block indices
        tgt_name = binding_map[idx.name]
        ren[idx] = index_by_name.get(tgt_name, Index(tgt_name, idx.space, True))
    for idx in defn_term.summed_indices:  # internal dummies
        safe = Index("__" + idx.name, idx.space, True)
        ren[idx] = safe

    from ..tensors import reindex_tensors
    new_amp = reindex_tensors(defn_term.factors, ren)
    factors = tuple(new_amp) + rest_factors
    internal = tuple(ren[i] for i in defn_term.summed_indices)
    return AlgebraTerm(
        coeff=defn_term.coeff,
        factors=factors,
        free_indices=rest_free,
        summed_indices=tuple(sorted(set(rest_summed) | set(internal),
                                    key=lambda x: (x.space, x.name))),
        connected=True,
    )


def verify_occurrence(
    op: DressedOperator,
    res_terms: Sequence[AlgebraTerm],
    ob: OccurrenceBinding,
) -> bool:
    """A3.3 -- is a bound occurrence an EXACT operator collapse?

    Derives the instance scalar c and the rest factors from the SEED fragment
    (the bare-ERI definition term's covered residual term), then reconstructs
    c * op * rest for the tau-expanded operator, expands, and checks the result
    reproduces the claimed residual fragments exactly (per-canonical-key
    coefficient sums match).  Returns False on any coefficient mismatch.

    Uses the tau-expanded operator so the reconstruction is in the same
    (un-collapsed) basis as the raw residual.

    SCOPE (coupling with A3.4): this checks only the fragments the binding
    CLAIMS (one per definition term via ob.coverage).  In the raw residual each
    definition-term contribution is fractured across several index-permuted
    fragments that must SUM to c * defn_coeff; A3.3 alone does not gather that
    fracture (that assignment is A3.4).  So verify_occurrence returns True for a
    complete, unfractured occurrence (e.g. one already assembled, or a synthetic
    instance) and False for a single-fragment-per-term slice of a fractured raw
    residual.  It is the exact coefficient check; A3.4 supplies the complete
    fragment set it checks against.
    """
    expected = expected_instance_fragments(op, res_terms, ob)
    if expected is None:
        return False

    # Actual: sum the claimed residual fragments per ERI-canonical key.
    claimed_indices = {ri for _di, ri in ob.coverage}
    actual: dict[tuple, Fraction] = {}
    for ri in claimed_indices:
        key, coeff = _eri_canonical(res_terms[ri])
        actual[key] = actual.get(key, Fraction(0)) + coeff
    actual = {k: v for k, v in actual.items() if v != 0}

    return expected == actual


def expected_instance_fragments(
    op: DressedOperator,
    res_terms: Sequence[AlgebraTerm],
    ob: OccurrenceBinding,
) -> dict[tuple, Fraction] | None:
    """The exact fragment contribution of one bound operator instance.

    Returns ``{canonical_key -> coefficient}`` for ``c * op * rest`` (tau-
    expanded), where c and rest are derived from the seed fragment.  This is the
    fracture spec A3.4 gathers against: every residual fragment whose canonical
    key appears here, summed, must equal these coefficients for the instance to
    be exact.  Returns None if the seed fragment cannot yield a clean c / rest.
    """
    op = tau_expanded_operator(op)
    binding_map = dict(ob.binding)

    seed_di, seed_ri = ob.coverage[0]
    seed_defn = op.definition_terms[seed_di]
    seed_res = res_terms[seed_ri]

    if len(seed_defn.factors) != 1:
        return None
    dfac = seed_defn.factors[0]
    rest_factors = None
    for k, rf in enumerate(seed_res.factors):
        if rf.name == dfac.name and len(rf.indices) == len(dfac.indices):
            rest_factors = tuple(g for gk, g in enumerate(seed_res.factors) if gk != k)
            break
    if rest_factors is None:
        return None

    if seed_defn.coeff == 0:
        return None
    c = seed_res.coeff / seed_defn.coeff

    index_by_name: dict[str, Index] = {}
    for t in res_terms:
        for idx in list(t.free_indices) + list(t.summed_indices):
            index_by_name[idx.name] = idx

    expected: dict[tuple, Fraction] = {}
    for dt in op.definition_terms:
        inst = _apply_binding_to_term(
            dt, binding_map, rest_factors,
            seed_res.summed_indices, seed_res.free_indices, index_by_name,
        )
        key, coeff = _eri_canonical(inst.scaled(c))
        expected[key] = expected.get(key, Fraction(0)) + coeff
    return {k: v for k, v in expected.items() if v != 0}
