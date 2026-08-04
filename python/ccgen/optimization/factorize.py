"""F0/F1 — contraction-path cost model + term inventory for the
higher-rank CC factorization thread (docs/CCGEN_HIGHER_OPERATOR_REUSE.md).

F0: inventory the multi-factor residual terms of a manifold with the peak
    exponent of evaluating each as ONE n-ary contraction (the baseline to beat).
F1: contraction_tree_cost / best_contraction_tree — search binary association
    orders for the tree whose peak pairwise-step exponent is lowest.

Cost is an (n_occ, n_vir) exponent pair: the number of distinct occ/vir indices
touched by a single contraction step (its loop nest). For a pairwise step over
operands A,B the touched indices are (A ∪ B) — output free indices plus the
shared summed indices — which is exactly the loop-nest depth of that GEMM-like
step. Peak = the max over steps. This is real path cost, NOT
IntermediateSpec.estimated_build_flops (an element count).

ponytail: only occ/vir tracked; a 'gen' index (never appears in CC residuals)
would raise NotImplementedError rather than be silently miscounted.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..project import AlgebraTerm
from ..tensors import Tensor


@dataclass(frozen=True, order=True)
class Cost:
    """Peak loop-nest exponent of a contraction step, split by space."""

    n_occ: int
    n_vir: int

    @property
    def total(self) -> int:
        return self.n_occ + self.n_vir

    def __repr__(self) -> str:
        parts = []
        if self.n_occ:
            parts.append(f"o^{self.n_occ}")
        if self.n_vir:
            parts.append(f"v^{self.n_vir}")
        return "".join(parts) or "o^0"

    def flops(self, n_occ: int = 10, n_vir: int = 50) -> int:
        """Symbolic flop magnitude o^a·v^b of one step at the given sizes.
        v>o by design, so this is scaling-dominated — an o^3v^5 step swamps an
        o^3v^3 one, which the total-degree metric alone hides."""
        return (n_occ ** self.n_occ) * (n_vir ** self.n_vir)


def _cost_of_indices(indices) -> Cost:
    n_occ = sum(1 for i in indices if i.space == "occ")
    n_vir = sum(1 for i in indices if i.space == "vir")
    if n_occ + n_vir != len(indices):
        raise NotImplementedError("only occ/vir index spaces are supported")
    return Cost(n_occ, n_vir)


def _max_cost(a: Cost, b: Cost) -> Cost:
    """Peak of two steps: compare by total exponent, tie-break on vir."""
    return a if (a.total, a.n_vir) >= (b.total, b.n_vir) else b


def nary_cost(term: AlgebraTerm) -> Cost:
    """Peak exponent of evaluating the term as a single n-ary contraction:
    every free AND summed index is a loop, all at once."""
    return _cost_of_indices(set(term.free_indices) | set(term.summed_indices))


# ── F1: binary contraction tree search ─────────────────────────────


def _pairwise_step_indices(a_idx, b_idx, keep):
    """Indices touched when contracting operands with index sets a_idx, b_idx.
    `keep` = indices needed later (free indices of the whole term + indices
    used by not-yet-contracted operands). The step loops over the union of
    both operands; its output keeps only the `keep` ones."""
    touched = a_idx | b_idx
    output = touched & keep
    return touched, output


@dataclass(frozen=True)
class Node:
    """A node in a binary contraction tree.

    Leaf: ``tensor`` set, ``children``/``summed`` empty.
    Internal: two ``children``, ``summed`` = indices consumed at this step,
    ``block`` = output free indices (what survives upward)."""

    block: frozenset       # index set this node exposes to its parent
    tensor: Tensor | None = None
    children: tuple = ()    # (Node, Node) for internal nodes
    summed: frozenset = frozenset()

    @property
    def is_leaf(self) -> bool:
        return self.tensor is not None

    def build_cost(self) -> Cost:
        """Cost of the contraction step that builds this node (block+summed).
        A leaf costs nothing — it is a stored tensor."""
        if self.is_leaf:
            return Cost(0, 0)
        return _cost_of_indices(self.block | self.summed)


def _tree_max_build_flops(node: Node) -> int:
    """Largest single-intermediate build flop cost anywhere in the tree.
    Used as the tie-break: among equal-peak trees, prefer the one whose most-
    expensive intermediate is BIGGEST, so the manifold consolidates onto shared
    expensive operators (the ones worth materializing — see value_operators)
    rather than splitting them across association orders by iteration luck."""
    if node.is_leaf:
        return 0
    here = node.build_cost().flops()
    return max([here] + [_tree_max_build_flops(c) for c in node.children])


def _tree_signature(node: Node) -> tuple:
    """Canonical, factor-order-independent fingerprint of a whole tree: the
    sorted build-cost + factor-name signature of every intermediate. This is
    the FINAL tie-break — the max-build-flops key alone still leaves genuine
    ties (several equal-peak, equal-max-flops trees), and without a total order
    those resolve by `combinations` iteration order, i.e. factor input order.
    A canonical signature makes the chosen tree a deterministic function of the
    term. (Verified: removing this reintroduces order-dependence.)"""
    sigs = []

    def walk(n):
        if n.is_leaf:
            return
        bc = n.build_cost()
        sigs.append((bc.n_occ, bc.n_vir,
                     tuple(sorted(f.name for f in _leaf_tensors(n)))))
        for c in n.children:
            walk(c)

    walk(node)
    return tuple(sorted(sigs))


def _best_tree(operands, free_needed):
    """operands: list of Node (each exposes `.block`).
    free_needed: indices that must survive to the end (term free indices).
    Returns (peak_cost, root_Node). Exhaustive over pairings — CC terms have
    <=5 factors so this is tiny.

    Selection key: (peak exponent asc, then max-intermediate-build-flops DESC)
    — minimize cost first, then break ties toward consolidating expensive
    shared operators (deterministic; no iteration-order dependence).
    ponytail: plain recursion, memo not needed at <=5 factors."""
    if len(operands) == 1:
        return Cost(0, 0), operands[0]

    best_key = None
    best_cost = None
    best_node = None
    for i, j in combinations(range(len(operands)), 2):
        a, b = operands[i], operands[j]
        rest = [operands[k] for k in range(len(operands)) if k not in (i, j)]
        # indices needed after this step: term-free ones, plus anything the
        # remaining operands still reference.
        keep = set(free_needed)
        for r in rest:
            keep |= r.block
        touched, output = _pairwise_step_indices(a.block, b.block, keep)
        step_cost = _cost_of_indices(touched)
        merged = Node(
            block=frozenset(output),
            children=(a, b),
            summed=frozenset(touched - output),
        )
        sub_cost, sub_root = _best_tree(rest + [merged], free_needed)
        peak = _max_cost(step_cost, sub_cost)
        # primary: minimize peak (total, then vir). secondary: maximize the
        # biggest intermediate's build flops (negate so smaller wins) —
        # consolidates onto expensive shared operators. tertiary: canonical
        # tree signature — a total order that kills residual factor-order
        # dependence (load-bearing for determinism, verified).
        key = (
            peak.total, peak.n_vir,
            -_tree_max_build_flops(sub_root),
            _tree_signature(sub_root),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_cost = peak
            best_node = sub_root
    return best_cost, best_node


def best_contraction_tree(term: AlgebraTerm) -> Cost:
    """Peak pairwise-step exponent of the best binary contraction tree."""
    return best_contraction_tree_full(term)[0]


def best_contraction_tree_full(term: AlgebraTerm):
    """(peak_cost, root_Node) of the best binary contraction tree.
    For a 1-factor term the root is the lone leaf and cost is the n-ary cost."""
    operands = [Node(block=frozenset(f.indices), tensor=f) for f in term.factors]
    free_needed = frozenset(term.free_indices)
    if len(operands) <= 1:
        return nary_cost(term), operands[0] if operands else None
    return _best_tree(operands, free_needed)


def contraction_tree_cost(term: AlgebraTerm):
    """Return (nary_cost, best_binary_tree_cost) for a term."""
    return nary_cost(term), best_contraction_tree(term)


# ── F2.0: tree nodes → AlgebraTerm ─────────────────────────────────


def _leaf_tensors(node: Node) -> tuple[Tensor, ...]:
    """All leaf Tensor factors under a node."""
    if node.is_leaf:
        return (node.tensor,)
    out = ()
    for c in node.children:
        out += _leaf_tensors(c)
    return out


def node_to_term(node: Node) -> AlgebraTerm:
    """The AlgebraTerm this internal node computes: its subtree's leaf factors,
    with `summed` = indices consumed AT this step and `free` = its output block.
    Coefficient is 1 — F2 keying (`_eri_canonical`) is structure-only.
    ponytail: a leaf has no contraction of its own, so it is not a term."""
    if node.is_leaf:
        raise ValueError("a leaf node is a bare tensor, not a contraction term")
    from fractions import Fraction

    from ..indices import canonical_index_order
    factors = _leaf_tensors(node)
    free = tuple(canonical_index_order(list(node.block)))
    summed = tuple(canonical_index_order(list(node.summed)))
    return AlgebraTerm(
        coeff=Fraction(1),
        factors=factors,
        free_indices=free,
        summed_indices=summed,
        connected=True,
    )


def internal_nodes(node: Node) -> list[Node]:
    """Every internal (contraction) node in a tree, leaves excluded."""
    if node is None or node.is_leaf:
        return []
    out = [node]
    for c in node.children:
        out += internal_nodes(c)
    return out


def tree_terms(term: AlgebraTerm) -> list[AlgebraTerm]:
    """The AlgebraTerm of every internal node of `term`'s best contraction tree
    (the intermediates F2.1+ will key and match)."""
    _, root = best_contraction_tree_full(term)
    return [node_to_term(n) for n in internal_nodes(root)]


def tree_preserves_term(term: AlgebraTerm) -> bool:
    """F3 exact gate (associativity bookkeeping): does the best contraction tree
    evaluate to the raw term? Tensor contraction is associative, so this is a
    bookkeeping check, NOT a numeric one — it holds iff:
      1. every raw factor appears exactly once as a tree leaf, and
      2. every raw summed index is consumed at exactly one tree node.
    When both hold the tree computes the raw n-ary contraction exactly, with the
    term coefficient (carried whole, never split across nodes) untouched."""
    from collections import Counter

    if len(term.factors) < 2:
        return True
    _, root = best_contraction_tree_full(term)
    if Counter(_leaf_tensors(root)) != Counter(term.factors):
        return False
    consumed = Counter()
    for n in internal_nodes(root):
        consumed.update(n.summed)
    return consumed == Counter(term.summed_indices)


# ── F2.1: canonical node key ───────────────────────────────────────


def node_key(node_or_term) -> tuple:
    """Canonical key of an intermediate, structure-only (dummy-name and
    v-orientation independent). Accepts a Node or its AlgebraTerm.

    Delegates to dressing's `_eri_canonical`, which folds v's 8-fold ERI
    symmetry and normalizes free-index order — so two nodes that are the same
    contraction up to external relabeling / v arrangement collapse to one key.
    This is what makes an operator reused across a term family show up once."""
    from .dressing import _eri_canonical_key
    term = node_or_term if isinstance(node_or_term, AlgebraTerm) else node_to_term(node_or_term)
    return _eri_canonical_key(term)


def block_signature(term) -> str:
    """Space signature of a node/operator block: 'o'/'v' per free index,
    sorted (o before v) so it is orientation-independent."""
    free = term.free_indices if isinstance(term, AlgebraTerm) else term.block
    occ = sum(1 for i in free if i.space == "occ")
    vir = len(free) - occ
    return "o" * occ + "v" * vir


# ── F2.2: seeded-operator fingerprints ─────────────────────────────


@dataclass(frozen=True)
class OperatorFingerprint:
    """One recognizable shape of a seeded operator: the canonical key of one of
    its definition terms, tagged with the owning operator and its block sig.

    A tree node is a SINGLE sub-contraction, so it can only match one
    definition TERM of an operator, not the operator's full sum — hence the
    fingerprint is per-definition-term."""

    op_name: str
    op_block_sig: str          # 'oooo', 'vvvv', ... (sorted)
    term_factors: tuple        # sorted factor names of this definition term
    key: tuple                 # _eri_canonical key of the definition term


def seeded_fingerprints():
    """All definition-term fingerprints of the six CCSD seeded operators.
    F2.3 matches node keys against these (with a block-sig prefilter)."""
    from .dressing import seeded_operators, _eri_canonical_key

    out = []
    for op in seeded_operators():
        sig = "".join(sorted(op.space_sig()))
        for dt in op.definition_terms:
            out.append(OperatorFingerprint(
                op_name=op.name,
                op_block_sig=sig,
                term_factors=tuple(sorted(f.name for f in dt.factors)),
                key=_eri_canonical_key(dt),
            ))
    return out


# ── F2.3: match or derive ──────────────────────────────────────────


@dataclass(frozen=True)
class Reuse:
    """A tree node identified as a re-use of a seeded CCSD operator."""

    op_name: str


@dataclass(frozen=True)
class Derived:
    """A tree node that matches no seeded operator: a newly-derived one, carried
    as the IntermediateSpec the emit pipeline can materialize."""

    spec: "object"          # IntermediateSpec (imported lazily)

    @property
    def name(self) -> str:
        return self.spec.name


def _derived_name(node_term: AlgebraTerm) -> str:
    """Stable name for a derived operator from its factor set + block sig,
    e.g. W_t3v_ooov. Factor names are SORTED so the name is invariant under
    factor input order (f·t3 and t3·f are one operator, W_ft3v)."""
    facs = "".join(sorted(f.name for f in node_term.factors if f.name != "v"))
    return f"W_{facs}v_{block_signature(node_term)}"


def identify_node(node_or_term, fingerprints=None, usage_count: int = 1):
    """Classify one intermediate: Reuse(op) if its key matches a seeded
    operator definition term (block-sig prefilter first), else Derived(spec).

    ponytail: exact key match only. A 're-blocking' (CCSD operator on a
    permuted/extended block) is a real but boundary case (F2.3 honest ceiling);
    exact match is sound — it never calls a genuinely-new object 'reuse'."""
    from .intermediates import IntermediateSpec

    term = node_or_term if isinstance(node_or_term, AlgebraTerm) else node_to_term(node_or_term)
    if fingerprints is None:
        fingerprints = seeded_fingerprints()

    sig = block_signature(term)
    key = node_key(term)
    for fp in fingerprints:
        if fp.op_block_sig == sig and fp.key == key:
            return Reuse(op_name=fp.op_name)

    spec = IntermediateSpec(
        name=_derived_name(term),
        indices=term.free_indices,
        definition_terms=(term,),
        usage_count=usage_count,
        index_space_sig=sig,
    )
    return Derived(spec=spec)


def identify_tree(term: AlgebraTerm, fingerprints=None):
    """Classify every internal node of `term`'s best contraction tree.
    Returns [(node_term, Reuse|Derived), ...]. The unit F3 sweeps."""
    if fingerprints is None:
        fingerprints = seeded_fingerprints()
    return [
        (nt, identify_node(nt, fingerprints))
        for nt in tree_terms(term)
    ]


# ── E0.0: emittable (non-root) derived operators ───────────────────


def emittable_operators(term: AlgebraTerm, fingerprints=None):
    """The derived operators a term's tree exposes that are safe to emit as
    `build_W` intermediates — i.e. every internal node EXCEPT the root.

    The root node's contraction IS the residual term itself (its leaves are the
    whole term); naming it would collapse the term to a single operator instead
    of factoring it. Only the strictly-internal nodes are reusable operators.
    Returns [(node_term, Reuse|Derived), ...] with the root dropped.

    ponytail: `internal_nodes` yields the root first (pre-order), so dropping
    index 0 is the whole fix — no tree re-walk."""
    if fingerprints is None:
        fingerprints = seeded_fingerprints()
    _, root = best_contraction_tree_full(term)
    nodes = internal_nodes(root)
    non_root = nodes[1:]  # drop the root; its contraction is the residual
    return [
        (node_to_term(n), identify_node(node_to_term(n), fingerprints))
        for n in non_root
    ]


# ── E0.1: hierarchical substitution (rewrite a term through its tree) ──


def _node_operator_name(node: Node, fingerprints) -> str:
    """The operator name a non-leaf child references (Reuse op or Derived name)."""
    r = identify_node(node_to_term(node), fingerprints)
    return r.op_name if isinstance(r, Reuse) else r.name


def rewrite_term_factorized(
    term: AlgebraTerm, fingerprints=None, derived_only: bool = True
) -> AlgebraTerm:
    """Rewrite a term into its ROOT contraction step: the root's children as
    factors, each internal child replaced by a reference to the operator it
    builds, each leaf child left as its bare tensor.

    Result: `coeff · <root-child factors> summed over <root.summed>`, e.g.
    `t2·t3·v`  ->  `t2 · W_t3v_ooov`. The referenced operators are emitted
    separately (E0.0's `emittable_operators`); this is the term that consumes
    them. Innermost-first is automatic — a child's own sub-structure lives in
    ITS build_W, so the root step only names its immediate children.

    A single-step term (≤1 internal node, e.g. `t4·v`) is returned unchanged:
    there is nothing to factor out.
    """
    from fractions import Fraction

    from ..indices import canonical_index_order

    if fingerprints is None:
        fingerprints = seeded_fingerprints()
    _, root = best_contraction_tree_full(term)
    if root is None or root.is_leaf:
        return term
    # a 2-factor term is one pairwise step: no inner operator to hoist.
    if all(c.is_leaf for c in root.children):
        return term

    new_factors = []
    extra_summed = set()  # summed indices pulled back in from un-hoisted children
    hoisted_any = False
    for child in root.children:
        if child.is_leaf:
            new_factors.append(child.tensor)
            continue
        r = identify_node(node_to_term(child), fingerprints)
        if derived_only and isinstance(r, Reuse):
            # Do NOT hoist a CCSD (Reuse) child — its dressing is D7.3's job, and
            # its definition needs tau/tau_tilde builders the factorizer does not
            # own. Inline its leaf tensors and re-absorb its consumed indices.
            new_factors.extend(_leaf_tensors(child))
            extra_summed |= set(child.summed)
        else:
            name = r.op_name if isinstance(r, Reuse) else r.spec.name
            block = tuple(canonical_index_order(list(child.block)))
            new_factors.append(Tensor(name, block))
            hoisted_any = True

    if not hoisted_any:
        return term  # nothing Derived to hoist -> leave the term as-is

    return AlgebraTerm(
        coeff=term.coeff,
        factors=tuple(new_factors),
        free_indices=term.free_indices,
        summed_indices=tuple(
            canonical_index_order(list(set(root.summed) | extra_summed))
        ),
        connected=term.connected,
        provenance=term.provenance,
    )


# ── E0.2: dedup operators across a manifold ────────────────────────


def _complete_definition_summation(defn: AlgebraTerm) -> AlgebraTerm:
    """Return `defn` with summed_indices = every index used by its factors that
    is NOT a free (block) index. `node_to_term` records only the top tree step's
    summed index; a standalone builder needs the FULL internal summation so no
    contraction loop variable is left undeclared."""
    from ..indices import canonical_index_order
    free = set(defn.free_indices)
    used = {i for f in defn.factors for i in f.indices}
    summed = canonical_index_order(list(used - free))
    return AlgebraTerm(
        coeff=defn.coeff,
        factors=defn.factors,
        free_indices=defn.free_indices,
        summed_indices=tuple(summed),
        connected=defn.connected,
        provenance=defn.provenance,
    )


def manifold_operators(terms, fingerprints=None, include_reuse=True):
    """The distinct derived operators to emit for a manifold, one per operator
    NAME, with `usage_count` = number of reference sites across all terms.

    Dedup is by name, NOT by canonical node key: the name (factor set + block
    signature) is the operator's identity — a single `build_W` indexed at each
    site with that site's externals — whereas the canonical key over-splits an
    operator into its external-relabeling instances (all one `build_W`). Measured
    on CCSDT triples: 447 reference sites collapse to 24 distinct operators (the
    key would give 74, over-fine).

    Returns [IntermediateSpec, ...], each with a representative definition term
    and the summed usage_count.

    ``include_reuse`` (default True) also emits specs for the CCSD operators the
    rewrite references (Fme/Fae/Fmi/Wmnij/Wabef/Wmbej), bridged from
    ``seeded_operators()`` via ``operator_to_intermediate_spec``. The emitter
    needs a spec for every factor a rewritten term names — a `Reuse` child (e.g.
    `t1·v` recognized as Fme) appears as an `Fme(...)` factor, so its builder
    must be materialized. Set False to emit only the newly-derived operators.
    """
    from .intermediates import IntermediateSpec
    from .dressing import seeded_operators, operator_to_intermediate_spec

    if fingerprints is None:
        fingerprints = seeded_fingerprints()
    derived: dict[str, list] = {}
    reuse_counts: dict[str, int] = {}
    for t in terms:
        for node_term, r in emittable_operators(t, fingerprints):
            if isinstance(r, Derived):
                derived.setdefault(r.spec.name, []).append(r.spec)
            else:  # Reuse
                reuse_counts[r.op_name] = reuse_counts.get(r.op_name, 0) + 1
    out = []
    for name, specs in derived.items():
        rep = specs[0]  # any instance defines the operator (all same shape)
        # A multi-factor operator (e.g. W_t2t2v) is itself a multi-step
        # contraction; node_to_term recorded only the TOP step's summed index,
        # but a STANDALONE build_W must declare EVERY internal contraction index
        # as a loop. Recompute the full summation = (all indices used by the
        # def-term's factors) - (the operator's block/free), else the emitted
        # builder references undeclared loop vars.
        rep_def = _complete_definition_summation(rep.definition_terms[0])
        out.append(IntermediateSpec(
            name=name,
            indices=rep.indices,
            definition_terms=(rep_def,),
            usage_count=len(specs),
            index_space_sig=rep.index_space_sig,
        ))
    if include_reuse and reuse_counts:
        by_op = {o.name: o for o in seeded_operators()}
        for op_name, count in reuse_counts.items():
            spec = operator_to_intermediate_spec(by_op[op_name], canonical_fock=True)
            # carry the measured reference count
            out.append(IntermediateSpec(
                name=spec.name,
                indices=spec.indices,
                definition_terms=spec.definition_terms,
                usage_count=count,
                index_space_sig=spec.index_space_sig,
            ))
    return out


# ── E0.3: emit a factorized translation unit ───────────────────────


def emit_factorized_translation_unit(method: str, engine: str = "diagram",
                                     canonical_fock: bool = True):
    """E0.3: emit a Planck C++ TU whose kernels reference the factorizer's
    derived operators, with a `build_W` for each.

    Pipeline: generate the residual, rewrite every term through its contraction
    tree (`rewrite_term_factorized`, derived-only — CCSD dressing stays D7.3's
    job), collect the manifold's derived operators (`manifold_operators`,
    `include_reuse=False`), and hand both to `emit_planck_translation_unit`.

    Returns the TU string. The emitted kernels name the derived operators as
    local factors, which the emitter materializes once per kernel via the
    build_W functions."""
    from ..generate import generate_cc_equations
    from ..emit.planck_tensor_cpp import emit_planck_translation_unit

    eqs = generate_cc_equations(method, engine=engine, canonical_fock=canonical_fock)
    rewritten = {
        m: [rewrite_term_factorized(t) for t in terms]
        for m, terms in eqs.items()
    }
    substitutable = [
        t for m, terms in eqs.items()
        if m not in ("energy", "reference")
        for t in terms
    ]
    ops = manifold_operators(substitutable, include_reuse=False)
    return emit_planck_translation_unit(method, rewritten, intermediates=ops)


# ── F4: savings-weighted operator valuation ────────────────────────


@dataclass(frozen=True)
class OperatorValue:
    """One operator's value over a manifold: how often it recurs and the flop
    cost of building it once. Savings = the recomputation avoided by
    materializing it instead of rebuilding per term."""

    name: str
    kind: str               # "reuse" | "derived"
    uses: int
    build_step: Cost        # n-ary cost of building the operator once
    n_occ: int = 10
    n_vir: int = 50

    @property
    def build_flops(self) -> int:
        return self.build_step.flops(self.n_occ, self.n_vir)

    @property
    def savings(self) -> int:
        """(uses - 1) × build cost: the per-term rebuild avoided by caching.
        This is the F4 lever — an operator reused 15× that avoids an o^3v^5
        step outranks one reused 75× saving only o^2v^2, which a raw reuse
        count gets exactly backwards."""
        return max(0, self.uses - 1) * self.build_flops


def value_operators(terms, fingerprints=None, n_occ=10, n_vir=50):
    """Rank the operators a manifold's contraction trees expose by SAVINGS,
    not frequency. Returns [OperatorValue, ...] sorted savings-descending."""
    from collections import defaultdict

    if fingerprints is None:
        fingerprints = seeded_fingerprints()
    uses = defaultdict(int)
    kind = {}
    build = {}
    for t in terms:
        for nt, r in identify_tree(t, fingerprints):
            name = r.op_name if isinstance(r, Reuse) else r.name
            uses[name] += 1
            kind[name] = "reuse" if isinstance(r, Reuse) else "derived"
            build[name] = nary_cost(nt)  # cost of building this node once
    vals = [
        OperatorValue(
            name=name, kind=kind[name], uses=uses[name],
            build_step=build[name], n_occ=n_occ, n_vir=n_vir,
        )
        for name in uses
    ]
    vals.sort(key=lambda v: v.savings, reverse=True)
    return vals


# ── F5.3: cross-rank recursion summary ─────────────────────────────


def derived_operators(terms, fingerprints=None) -> set:
    """Set of newly-derived operator names a manifold's trees expose."""
    if fingerprints is None:
        fingerprints = seeded_fingerprints()
    return {
        r.name
        for t in terms
        for _, r in identify_tree(t, fingerprints)
        if isinstance(r, Derived)
    }


def reused_operators(terms, fingerprints=None) -> set:
    """Set of seeded CCSD operator names a manifold's trees reuse."""
    if fingerprints is None:
        fingerprints = seeded_fingerprints()
    return {
        r.op_name
        for t in terms
        for _, r in identify_tree(t, fingerprints)
        if isinstance(r, Reuse)
    }


def recursion_summary(lower_terms, higher_terms):
    """F5.3: is the derived operator set cumulative across rank? Given a lower-
    rank manifold and a higher one, report the shared vs new derived operators.
    A full-containment result (`lower ⊆ higher`) means operators derived at the
    lower rank are reused verbatim at the higher one — the recursive-reuse
    verdict as a measured dict."""
    lo = derived_operators(lower_terms)
    hi = derived_operators(higher_terms)
    return {
        "lower_derived": len(lo),
        "higher_derived": len(hi),
        "shared": len(lo & hi),
        "lower_only": sorted(lo - hi),
        "higher_only": sorted(hi - lo),
        "cumulative": lo <= hi,
    }


# ── F0: term inventory ─────────────────────────────────────────────


@dataclass(frozen=True)
class TermEntry:
    factors: tuple[str, ...]       # tensor names, e.g. ('t2','t3','v')
    n_factors: int
    nary: Cost
    best: Cost

    @property
    def factors_over_bare(self) -> bool:
        return (self.best.total, self.best.n_vir) < (self.nary.total, self.nary.n_vir)


def inventory(terms) -> list[TermEntry]:
    """Multi-factor terms only (n_factors >= 2) with their n-ary + best cost."""
    out = []
    for t in terms:
        if len(t.factors) < 2:
            continue
        nary, best = contraction_tree_cost(t)
        out.append(TermEntry(
            factors=tuple(f.name for f in t.factors),
            n_factors=len(t.factors),
            nary=nary,
            best=best,
        ))
    return out


# ── self-check (F1 de-risk gate) ───────────────────────────────────


def _demo():
    """Printout of the CCSDT-triples inventory. The F1 de-risk gate lives in
    tests/test_factorize.py, not here."""
    from collections import Counter

    from ..generate import generate_cc_equations

    eqs = generate_cc_equations("ccsdt", engine="diagram", canonical_fock=True)
    triples = eqs["triples"]
    inv = inventory(triples)
    print(f"CCSDT triples: {len(triples)} terms, {len(inv)} multi-factor")
    wins = Counter(e.factors for e in inv if e.factors_over_bare)
    print("shapes that factor below n-ary (count by factor set):")
    for shape, c in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"  {c:3d}  {'·'.join(shape)}")

    print("\ntop operators by SAVINGS (uses-1 × build flops), not frequency:")
    print(f"  {'savings':>10} {'uses':>4} {'build':>8}  operator")
    for v in value_operators(triples)[:12]:
        print(f"  {v.savings:10.2e} {v.uses:4d} {str(v.build_step):>8}  "
              f"[{v.kind}] {v.name}")


if __name__ == "__main__":
    _demo()
