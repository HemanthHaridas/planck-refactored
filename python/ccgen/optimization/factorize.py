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


def build_cost(term: AlgebraTerm) -> Cost:
    """Cost of actually BUILDING this operator once — its contraction TREE
    cost, not its n-ary cost. A two-factor operator is one step, so the two
    coincide; a multi-factor one (W_t2t2v, W_t1t2t2v) is emitted as a nested
    contraction (`emit_intermediate_steps`), and pricing it as a single flat
    loop nest overstates it badly — measured 500x on W_t2t2v_oooovv
    (625G n-ary vs 1.25G tree)."""
    if len(term.factors) < 2:
        return nary_cost(term)
    return best_contraction_tree_full(term)[0]


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
    with `free` = its output block and `summed` = EVERY index its factors use
    that is not free. Coefficient is 1 — F2 keying (`_eri_canonical`) is
    structure-only.

    `node.summed` alone is only the indices consumed at THIS step; a subtree's
    inner contractions are summed at descendant nodes. Since the factors here
    are the whole subtree's leaves, using `node.summed` leaves those inner
    indices bound to nothing — a spec whose definition references an index that
    is neither a slot nor a declared loop, and whose emitted `build_W` has no
    loop for it (D4; measured 20/52 derived ccsd-doubles specs).
    ponytail: a leaf has no contraction of its own, so it is not a term."""
    if node.is_leaf:
        raise ValueError("a leaf node is a bare tensor, not a contraction term")
    from fractions import Fraction

    from ..indices import canonical_index_order
    factors = _leaf_tensors(node)
    free = tuple(canonical_index_order(list(node.block)))
    used = {i for f in factors for i in f.indices}
    summed = tuple(canonical_index_order(list(used - set(free))))
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


def _contraction_shape(node_term: AlgebraTerm) -> tuple:
    """D6: the operator's contraction shape — what makes two call sites the SAME
    operator rather than merely the same-looking one.

    Each factor slot becomes either its OUTPUT SLOT POSITION (an int, as a str)
    or a canonically-renumbered internal summed index (``S0``, ``S1``, …). Factor
    order is preserved.

    All three properties are load-bearing, each measured on GCC ccsd doubles
    (41 rewritten terms, value-checked against the source via `residual_einsum`):

    - **slot POSITION, not F/S** — marking a slot merely "free" conflates
      `t1(c,j) v(i,c,k,a)` with `t1(c,i) v(j,c,k,a)`: same shape, `i`/`j`
      swapped between the two factors. 21 -> 13 disagreements.
    - **positions, not names** — `(i,j,k,a)` and `(i,j,k,b)` are one operator
      called at two sites; keying on names would split them pointlessly.
      13 -> 6.
    - **same-tensor copies kept DISTINCT** — with two factors of the same tensor
      (`t1 … t1`), collapsing them loses which copy carries which slot:
      `t1(d,j) t1(c,i)` vs `t1(d,i) t1(c,j)`. 6 -> 0.

    The per-factor entries are sorted AFTER slot substitution, so the key is a
    function of the contraction and not of factor INPUT order (`f·t3` and `t3·f`
    are one operator) — while still separating same-name copies, because by then
    they carry different slot tuples. Sorting the raw factors instead would
    destroy exactly the distinction the third property needs.

    Cost, recorded because it is real and matters to the reuse case: on spatial
    ccsd doubles this splits 14 names into 61 keys, dropping sites-at-a-reused-
    operator from 76 to 31. That is the price of correctness here, not a
    regression — the coarser key was binding definitions that do not compute the
    term at their call site.
    """
    pos = {idx: str(k) for k, idx in enumerate(node_term.free_indices)}

    def entries(renamed):
        out = []
        for f in node_term.factors:
            out.append((f.name, tuple(
                pos[x] if x in pos else renamed[x] for x in f.indices)))
        return tuple(sorted(out))

    # Summed indices carry no identity of their own, so they must be renumbered
    # canonically -- but "first appearance while walking factors" depends on
    # factor INPUT order, which is exactly what this key must not see. Choose the
    # numbering that minimizes the sorted entry list: a well-defined function of
    # the contraction alone. The summed count is tiny (<= 4 here), so the
    # permutation sweep is cheap.
    import itertools

    summed = [x for x in node_term.summed_indices if x not in pos]
    best = None
    for perm in itertools.permutations(range(len(summed))):
        cand = entries({idx: f"S{perm[k]}" for k, idx in enumerate(summed)})
        if best is None or cand < best:
            best = cand
    return best if best is not None else entries({})


def _derived_name(node_term: AlgebraTerm) -> str:
    """Stable name for a derived operator from its factor set + block sig,
    e.g. W_t3v_ooov. Factor names are SORTED so the name is invariant under
    factor input order (f·t3 and t3·f are one operator, W_ft3v).

    NOTE: the name alone does NOT identify an operator — two call sites can
    share it and need different definitions. `manifold_operators` dedups on
    (name, `_contraction_shape`) and disambiguates collisions with a suffix.
    """
    facs = "".join(sorted(f.name for f in node_term.factors if f.name != "v"))
    base = f"W_{facs}v_{block_signature(node_term)}"
    return f"{base}_{_shape_tag(node_term)}"


def _shape_tag(node_term: AlgebraTerm) -> str:
    """Short stable discriminator for an operator's contraction shape.

    Folded into the NAME rather than kept beside it so a name can never denote
    two definitions anywhere downstream — the emitter keys `build_W` functions
    by name, so a collision there silently emits one builder for two different
    contractions. Deriving it from the shape keeps it deterministic across runs
    (unlike `hash()`, which is salted per process).

    NOT canonicalized over transpose-equivalence — see O4 in
    `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`. Merging those names requires the
    REWRITE to permute each call site's indices into the shared operator's slot
    order; doing it here alone reintroduces the D6 defect (measured: 11 GCC terms
    stop reproducing their source).
    """
    import hashlib

    raw = repr(_contraction_shape(node_term)).encode()
    return hashlib.blake2s(raw, digest_size=2).hexdigest()


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
    term: AlgebraTerm, fingerprints=None, derived_only: bool = True,
    keep_operators=None, merge_plan_map=None,
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
        name = r.op_name if isinstance(r, Reuse) else r.spec.name
        # Hoist a child ONLY if it is a Derived operator we are keeping:
        #  - Reuse (CCSD) children are always inlined (dressing is D7.3's job,
        #    their definitions need tau/tau_tilde builders the factorizer lacks);
        #  - a Derived child not in `keep_operators` (E1 budget) is inlined too.
        # Under merging, `keep_operators` holds REPRESENTATIVE names — the
        # merged-away members no longer exist as specs — so the budget check
        # must ask about the representative, not this member. Testing `name`
        # rejects every merged call site before its permutation is applied,
        # which silently un-does O4.2 (found by O4.5: no read order changed in
        # the emitted TU).
        kept_name = (merge_plan_map or {}).get(name, (name, None))[0]
        hoist = (not isinstance(r, Reuse)) and (
            keep_operators is None or kept_name in keep_operators
        )
        if hoist:
            block = tuple(canonical_index_order(list(child.block)))
            if merge_plan_map is not None:
                # O4.2: read the operator in its class REPRESENTATIVE's slot
                # order. `perm[k] = j` means slot k here is slot j of the rep,
                # so the index at position k moves to position j.
                #
                # Landing this BEFORE the names merge (O4.3) is deliberate: it
                # exercises the new index order while every operator still owns
                # its own array, so a wrong permutation surfaces as a value
                # failure that cannot be blamed on sharing. Doing both at once
                # was tried and reverted — 11 GCC terms broke and the value gate
                # could not say which half was at fault.
                _rep, perm = merge_plan_map.get(
                    name, (name, tuple(range(len(block)))))
                if perm != tuple(range(len(block))):
                    reordered = [None] * len(block)
                    for k, j in enumerate(perm):
                        reordered[j] = block[k]
                    block = tuple(reordered)
                # ...and under the REPRESENTATIVE's name: it is the only spec
                # emitted, so referencing the member would call a `build_W`
                # that does not exist.
                name = kept_name
            new_factors.append(Tensor(name, block))
            hoisted_any = True
        else:
            # Inline the child's leaves and re-absorb its FULL internal summation
            # (all subtree-consumed indices = used - block), not just its top
            # step's — a deep child otherwise leaves loop vars undeclared.
            leaves = _leaf_tensors(child)
            new_factors.extend(leaves)
            used = {i for f in leaves for i in f.indices}
            extra_summed |= (used - set(child.block))

    if not hoisted_any:
        return term  # nothing hoisted -> leave the term as-is

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


def manifold_operators(terms, fingerprints=None, include_reuse=True,
                       merge_transposes=False, spatial=True):
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

    ``merge_transposes`` (O4.3) collapses operators that are one contraction up
    to a permutation of their own slots, keeping the class representative and
    summing the members' usage counts. Off by default: a merged operator is only
    correct if every CALL SITE reads it in the representative's slot order, so a
    caller that sets this MUST also pass the returned plan to
    ``rewrite_term_factorized(..., merge_plan_map=...)``. Use
    ``manifold_operators_with_plan`` to get both together rather than wiring the
    two halves by hand — landing them separately is what broke 11 GCC terms.
    """
    return _manifold_operators_impl(
        terms, fingerprints, include_reuse, merge_transposes, spatial)[0]


def manifold_operators_with_plan(terms, fingerprints=None, include_reuse=True,
                                 spatial=True):
    """O4.3: the merged operator set AND the call-site plan that makes it valid.

    Returns ``(specs, plan)``. The two are only correct together — emitting the
    merged specs while call sites still read in their own slot order is exactly
    the reverted first attempt (11 GCC doubles terms stopped reproducing their
    source). Handing them back as a pair is what stops them being separated.
    """
    return _manifold_operators_impl(
        terms, fingerprints, include_reuse, True, spatial)


def _manifold_operators_impl(terms, fingerprints, include_reuse,
                             merge_transposes, spatial):
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
        rep_def = rep.definition_terms[0]
        out.append(IntermediateSpec(
            name=name,
            indices=rep.indices,
            definition_terms=(rep_def,),
            usage_count=len(specs),
            index_space_sig=rep.index_space_sig,
        ))
    plan = None
    if merge_transposes:
        from .operator_identity import merge_plan

        plan = merge_plan(out, spatial=spatial)
        by_rep: dict[str, list] = {}
        for spec in out:
            by_rep.setdefault(plan[spec.name][0], []).append(spec)
        merged = []
        for rep_name, members in by_rep.items():
            rep = next(m for m in members if m.name == rep_name)
            # one array per class; every member's call sites reference it
            merged.append(IntermediateSpec(
                name=rep.name,
                indices=rep.indices,
                definition_terms=rep.definition_terms,
                usage_count=sum(m.usage_count for m in members),
                index_space_sig=rep.index_space_sig,
            ))
        out = merged

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
    return out, plan


# ── E1: savings-budgeted operator selection ────────────────────────


def operator_savings(spec, n_occ: int = 10, n_vir: int = 50) -> int:
    """Savings of materializing an emittable operator once vs rebuilding it at
    every reference site: (usage_count - 1) × build_flops, where build_flops is
    the scaling-dominated flop magnitude of the operator's own contraction."""
    build = build_cost(spec.definition_terms[0]).flops(n_occ, n_vir)
    return max(0, spec.usage_count - 1) * build


# ── M0: footprint + density inventory (memory/locality investigation) ──


def operator_bytes(spec, n_occ: int = 30, n_vir: int = 100,
                   dtype_bytes: int = 8) -> int:
    """Storage footprint of the materialized operator TENSOR (its block), at the
    given orbital-space sizes. This is what selection currently ignores (B1).

    Defaults O=30/V=100 to match `IntermediateSpec.estimated_bytes`; pass
    explicit sizes to sweep. Note the factorizer's `operator_savings` defaults to
    O=10/V=50 (`Cost.flops`), so mixing the two at their defaults compares
    inconsistent scales — pass matching sizes to both when ranking by density."""
    n = 1
    for idx in spec.indices:
        n *= n_occ if idx.space == "occ" else n_vir
    return n * dtype_bytes


def operator_density(spec, n_occ: int = 30, n_vir: int = 100) -> float:
    """FLOP savings per byte of storage — the joint metric the flops-only
    selection ignores. Savings and bytes computed at the SAME sizes."""
    b = operator_bytes(spec, n_occ, n_vir)
    return operator_savings(spec, n_occ, n_vir) / max(1, b)


@dataclass(frozen=True)
class FootprintEntry:
    name: str
    rank: int
    sig: str
    uses: int
    savings: int
    bytes: int
    density: float


def footprint_inventory(specs, n_occ: int = 30, n_vir: int = 100):
    """M0: per-operator savings + footprint + density, at fixed O/V. The
    reproducible form of the baseline B1/B2 tables. Sorted by savings desc."""
    out = [
        FootprintEntry(
            name=s.name,
            rank=len(s.indices),
            sig=s.index_space_sig,
            uses=s.usage_count,
            savings=operator_savings(s, n_occ, n_vir),
            bytes=operator_bytes(s, n_occ, n_vir),
            density=operator_density(s, n_occ, n_vir),
        )
        for s in specs
    ]
    out.sort(key=lambda e: e.savings, reverse=True)
    return out


def select_operators_by_savings(specs, top_k=None, savings_fraction=None,
                                max_operator_bytes=None, n_occ=30, n_vir=100):
    """Rank emittable operator specs by savings and keep the worthwhile ones.

    - `top_k`: keep the k highest-savings operators.
    - `savings_fraction`: keep operators until the cumulative savings reach this
      fraction (0..1) of the total (e.g. 0.99 keeps ~99% of the win).
    Pass at most one; with neither, all specs are kept (sorted). The measured
    concentration is extreme — on CCSDT the top 5 of 24 operators carry >98%,
    so a small budget inlines the long tail for ~free.

    - `max_operator_bytes` (M1 feasibility guard): drop any operator whose
      materialized tensor exceeds this footprint at (`n_occ`, `n_vir`) BEFORE
      ranking, so it is inlined (via the E1 keep-set) rather than emitted as an
      un-storable `build_W`. The 64.8 GB rank-6 / 194,400 GB rank-8 operators
      have huge savings; without this guard the savings ranking would materialize
      them regardless of whether they fit.

    Returns (kept_specs, kept_names) with kept_specs sorted savings-descending.
    """
    if max_operator_bytes is not None:
        specs = [s for s in specs
                 if operator_bytes(s, n_occ, n_vir) <= max_operator_bytes]
    ranked = sorted(specs, key=operator_savings, reverse=True)
    if top_k is not None:
        kept = ranked[:top_k]
    elif savings_fraction is not None:
        total = sum(operator_savings(s) for s in ranked) or 1
        kept, cum = [], 0
        for s in ranked:
            kept.append(s)
            cum += operator_savings(s)
            if cum / total >= savings_fraction:
                break
    else:
        kept = ranked
    return kept, frozenset(s.name for s in kept)


# ── M2.0: total-memory-budget greedy selection ─────────────────────


def select_under_memory_budget(specs, total_bytes, key="savings",
                               n_occ=30, n_vir=100):
    """Greedily fill a TOTAL memory budget with operators, in `key` order.

    Unlike M1 (a per-operator cap), this bounds the SUM of materialized operator
    footprints: `Σ bytes(kept) ≤ total_bytes`. `key` is "savings" (flops-greedy)
    or "density" (savings/byte-greedy) — the two rankings M2 compares. An
    operator that would overflow the remaining budget is skipped (inlined via the
    E1 keep-set), and later smaller operators may still fit.

    This is the greedy baseline M2.1's exact knapsack is measured against.
    Measured divergence between the two keys: ~0 on CCSDT (operators cluster by
    footprint), 23% of budgets / up to 16.7% savings on CCSDTQ.

    Returns (kept_specs, kept_names), kept sorted by the chosen key descending.
    """
    if key == "savings":
        order = sorted(specs, key=lambda s: operator_savings(s, n_occ, n_vir),
                       reverse=True)
    elif key == "density":
        order = sorted(specs, key=lambda s: operator_density(s, n_occ, n_vir),
                       reverse=True)
    else:
        raise ValueError(f"key must be 'savings' or 'density', got {key!r}")

    kept, used = [], 0
    for s in order:
        b = operator_bytes(s, n_occ, n_vir)
        if used + b <= total_bytes:
            kept.append(s)
            used += b
    return kept, frozenset(s.name for s in kept)


# ── M2.1: best-of-both-greedy joint selection ──────────────────────


def select_best_of_both(specs, total_bytes, n_occ=30, n_vir=100):
    """Joint FLOP/memory selection under a total footprint budget: run both
    greedy keys (savings, density) and return the set with higher total savings.

    This IS the joint objective — measured against a correct exact 0/1 knapsack
    (branch-and-bound), best-of-both-greedy is within 0.002% of optimal on CCSDTQ
    across a dense budget sweep (exact beat it in 5/273 budgets, all ≤ 2e-5). So
    no exact solver is warranted; the win over the flops-only baseline comes from
    also considering the density ranking where the two diverge (23% of CCSDTQ
    budgets, up to 16.7%). Branch-and-bound lives only in the test oracle.

    Returns (kept_specs, kept_names) for the winning key.
    """
    ks, nks = select_under_memory_budget(specs, total_bytes, "savings",
                                         n_occ, n_vir)
    kd, nkd = select_under_memory_budget(specs, total_bytes, "density",
                                         n_occ, n_vir)
    sv = sum(operator_savings(s, n_occ, n_vir) for s in ks)
    dv = sum(operator_savings(s, n_occ, n_vir) for s in kd)
    return (ks, nks) if sv >= dv else (kd, nkd)


# ── E0.3: emit a factorized translation unit ───────────────────────


def emit_factorized_translation_unit(method: str, engine: str = "diagram",
                                     canonical_fock: bool = True,
                                     top_k=None, savings_fraction=None,
                                     max_operator_bytes=None,
                                     memory_budget_bytes=None,
                                     factor_builder_bodies=False,
                                     merge_transposes=False, spatial=True,
                                     n_occ=30, n_vir=100):
    """E0.3 + E1 + M1 + M2: emit a Planck C++ TU whose kernels reference the
    factorizer's derived operators, with a `build_W` for each KEPT operator.

    Pipeline: generate the residual, collect the manifold's derived operators
    (`manifold_operators`, `include_reuse=False`), select which to materialize,
    rewrite every term hoisting only the kept operators (the rest stay inline,
    along with the CCSD/Reuse children which are always inlined — D7.3's job), and
    hand the rewritten equations + kept specs to `emit_planck_translation_unit`.

    Selection precedence:
    - `memory_budget_bytes` (M2): joint FLOP/memory selection under a TOTAL
      footprint budget via `select_best_of_both` (best of the savings- and
      density-greedy fills). Takes precedence over the E1/M1 knobs.
    - else `top_k` / `savings_fraction` (E1) under the optional `max_operator_bytes`
      per-operator guard (M1).

    Non-selected operators inline via the E1 keep-set path. Returns the TU string.

    Generates its own equations, so it can only emit the GCC manifold this call
    produces. To factorize an ALREADY-ADAPTED manifold (spin-adapted or UCC),
    call `emit_factorized_from_equations` directly — that is the entry the
    production wiring needs (W1 in `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`).
    """
    from ..generate import generate_cc_equations

    return emit_factorized_from_equations(
        method,
        generate_cc_equations(method, engine=engine,
                              canonical_fock=canonical_fock),
        top_k=top_k, savings_fraction=savings_fraction,
        max_operator_bytes=max_operator_bytes,
        memory_budget_bytes=memory_budget_bytes,
        factor_builder_bodies=factor_builder_bodies,
        merge_transposes=merge_transposes, spatial=spatial,
        n_occ=n_occ, n_vir=n_vir)


def emit_factorized_from_equations(method: str, eqs, *,
                                   top_k=None, savings_fraction=None,
                                   max_operator_bytes=None,
                                   memory_budget_bytes=None,
                                   factor_builder_bodies=False,
                                   merge_transposes=False, spatial=True,
                                   n_occ=30, n_vir=100):
    """W1: the factorize-and-emit pipeline, over equations the caller supplies.

    Same body as `emit_factorized_translation_unit`, which is now a thin wrapper
    that generates and delegates — split so an already-adapted manifold can be
    factorized. `spin_adapt_equations(...)` and `ucc_adapt_equations(...)` output
    both work: the factorizer keys on contraction structure and does not care how
    a factor is named. Measured on adapted input, 31 merged operators on spatial
    `ccsd` doubles and 86 on UCC `doubles_abab`.

    `spatial` selects the symmetry table the transpose-equivalence uses; it is
    about the ERI's symmetries, NOT about whether `eqs` is spin-adapted. Leave it
    True unless the manifold is GCC-only — it is the smaller, always-sound set.
    """
    from ..emit.planck_tensor_cpp import emit_planck_translation_unit

    rewritten, kept = factorize_equations(
        eqs, top_k=top_k, savings_fraction=savings_fraction,
        max_operator_bytes=max_operator_bytes,
        memory_budget_bytes=memory_budget_bytes,
        merge_transposes=merge_transposes, spatial=spatial,
        n_occ=n_occ, n_vir=n_vir)
    return emit_planck_translation_unit(
        method, rewritten, intermediates=kept,
        factor_builder_bodies=factor_builder_bodies)


def factorize_equations(eqs, *, top_k=None, savings_fraction=None,
                        max_operator_bytes=None, memory_budget_bytes=None,
                        merge_transposes=False, spatial=True,
                        n_occ=30, n_vir=100):
    """W3.2: derive operators from `eqs` and rewrite it against them.

    The factorize half of `emit_factorized_from_equations`, split out so a
    caller that owns its own emit path can use the derivation route without a
    second emit call site. Returns `(rewritten_eqs, kept_specs)` -- exactly the
    `(eqs, intermediates)` pair `print_cpp_planck` already threads to the
    emitter for the recognition route, so `derived` slots into the same shared
    path rather than forking it.
    """
    substitutable = [
        t for m, terms in eqs.items()
        if m not in ("energy", "reference")
        for t in terms
    ]
    if merge_transposes:
        # O4.5: merged specs and the call-site plan come from ONE call, so the
        # emitter cannot end up with one without the other.
        all_ops, plan = manifold_operators_with_plan(
            substitutable, include_reuse=False, spatial=spatial)
    else:
        all_ops, plan = manifold_operators(substitutable, include_reuse=False), None
    if memory_budget_bytes is not None:
        kept, keep_names = select_best_of_both(
            all_ops, memory_budget_bytes, n_occ=n_occ, n_vir=n_vir)
    else:
        kept, keep_names = select_operators_by_savings(
            all_ops, top_k=top_k, savings_fraction=savings_fraction,
            max_operator_bytes=max_operator_bytes, n_occ=n_occ, n_vir=n_vir)
    rewritten = {
        m: [rewrite_term_factorized(t, keep_operators=keep_names,
                                    merge_plan_map=plan) for t in terms]
        for m, terms in eqs.items()
    }
    return rewritten, kept


# ── M3.0: factor an operator's OWN builder body ────────────────────


def factored_builder_steps(spec, scratch_prefix="X", stride_order=False):
    """M3.0: decompose an operator's definition term into the sequence of
    pairwise contraction steps its best tree gives, instead of one flat n-ary
    nest. Returns [(lhs_name, AlgebraTerm), ...] in build order (inner scratch
    first, `result` last); the final step's lhs is "result".

    ``stride_order`` (M3.2): order each step's summed loops so the min-stride
    index is innermost. Pure reorder — no factor/coeff/free change — so exact.

    A scratch step's term references its child steps by scratch name (a
    `Tensor(scratch_name, block)` factor), so `emit_planck_term` with those
    names in scope emits nested contractions writing to scratch tensors. A
    single-step (≤2-factor) definition returns one ("result", def) step — no
    change from the flat emit.

    The builder body is emitted flat (one loop nest over the definition's full
    summation); this factors it. Measured: cuts 3/8 top CCSDT builders from
    over-cost to their tree cost (e.g. W_t1t2v_oooovv o⁵v³→o⁵v²), and the largest scratch is
    ~0.3× the operator's own footprint — a FLOP win at no peak-memory cost.
    """
    from fractions import Fraction

    from ..indices import canonical_index_order

    defn = spec.definition_terms[0]
    _, root = best_contraction_tree_full(defn)
    if root is None or root.is_leaf or all(c.is_leaf for c in root.children):
        return [("result", defn)]

    steps = []
    counter = [0]

    def emit_node(node, is_root):
        # returns the factor (Tensor) that references this node's output
        if node.is_leaf:
            return node.tensor
        # recurse children first (inner scratch built before this step)
        child_factors = [emit_node(c, False) for c in node.children]
        block = tuple(canonical_index_order(list(node.block)))
        if is_root:
            lhs = "result"
        else:
            counter[0] += 1
            lhs = f"{scratch_prefix}{counter[0]}"
        step_term = AlgebraTerm(
            coeff=defn.coeff if is_root else Fraction(1),
            factors=tuple(child_factors),
            free_indices=block,
            summed_indices=tuple(canonical_index_order(list(node.summed))),
            connected=True,
        )
        if stride_order:
            step_term = AlgebraTerm(
                coeff=step_term.coeff,
                factors=step_term.factors,
                free_indices=step_term.free_indices,
                summed_indices=stride_ordered_summed(step_term),
                connected=True,
            )
        steps.append((lhs, step_term))
        return Tensor(lhs, block)

    emit_node(root, True)
    return steps


# ── M3.1: static stride metric for the emitted builder loop ────────


def _factor_access_stride(factor, inner_index):
    """How a factor's accessor strides against the innermost loop index:
      0  -> unit stride  (inner index is the factor's LAST axis)
      k>0 -> strided     (inner index is k axes from the last; bigger = worse)
      None -> invariant  (factor does not depend on the inner index — hoistable)
    """
    axes = [i for i, idx in enumerate(factor.indices) if idx == inner_index]
    if not axes:
        return None
    last = len(factor.indices) - 1
    return min(last - a for a in axes)  # closest occurrence to the last axis


def step_stride_penalty(term, inner_index, n_occ=30, n_vir=100):
    """Penalty for ONE contraction step's innermost-loop access pattern: sum over
    the step's factors of (distance of the inner index from each factor's last
    axis), i.e. how far from unit-stride each read is. 0 = every factor reads the
    inner index unit-stride; invariant factors contribute 0 (hoistable).

    Weighted by the step's loop volume (∏ index sizes) so a badly-strided large
    step outweighs a small one — a strided read in an o⁵v² loop matters more than
    in an o²v² loop."""
    vol = 1
    for idx in list(term.free_indices) + list(term.summed_indices):
        vol *= n_occ if idx.space == "occ" else n_vir
    dist = 0
    for f in term.factors:
        s = _factor_access_stride(f, inner_index)
        if s is not None:
            dist += s
    return dist * vol


def builder_stride_score(spec, n_occ=30, n_vir=100, reorder=False):
    """M3.1 (and M3.2 when reorder=True): aggregate stride penalty of a builder's
    emitted loops, over its factored steps. Innermost loop = the step's last
    summed index. With `reorder=False` (baseline) this is the emitter's current
    order; with `reorder=True` the summed loops are reordered so the min-penalty
    index is innermost (M3.2). Lower is better; 0 = every factor reads the
    innermost index unit-stride in every step."""
    total = 0
    for _lhs, term in factored_builder_steps(spec):
        if not term.summed_indices:
            continue  # no inner loop
        inner = stride_inner_index(term) if reorder \
            else term.summed_indices[-1]
        total += step_stride_penalty(term, inner, n_occ, n_vir)
    return total


# ── M3.2: shape the summed-loop order for stride ───────────────────


def stride_inner_index(term, n_occ=30, n_vir=100):
    """The summed index that should be INNERMOST to minimize the step's stride
    penalty — the one closest to unit-stride across the step's factors. Ties
    resolve to the canonical order for determinism."""
    from ..indices import canonical_index_order
    ordered = canonical_index_order(list(term.summed_indices))
    if not ordered:
        return None
    return min(ordered, key=lambda s: (step_stride_penalty(term, s, n_occ, n_vir),
                                       s.space, s.name))


def stride_ordered_summed(term, n_occ=30, n_vir=100):
    """The step's summed indices reordered for stride: the min-penalty index
    LAST (innermost loop). Outer summed loops keep the canonical order — only the
    innermost placement affects the metric, and pinning the rest keeps the emit
    deterministic. Pure reorder: no factor/coeff/free change, so algebra-exact."""
    from ..indices import canonical_index_order
    ordered = canonical_index_order(list(term.summed_indices))
    if len(ordered) < 2:
        return tuple(ordered)
    inner = stride_inner_index(term, n_occ, n_vir)
    rest = [s for s in ordered if s != inner]
    return tuple(rest + [inner])


# ── F4: savings-weighted operator valuation ────────────────────────


@dataclass(frozen=True)
class OperatorValue:
    """One operator's value over a manifold: how often it recurs and the flop
    cost of building it once. Savings = the recomputation avoided by
    materializing it instead of rebuilding per term."""

    name: str
    kind: str               # "reuse" | "derived"
    uses: int
    build_step: Cost        # tree cost of building the operator once
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
        # emittable_operators, NOT identify_tree: the latter includes the ROOT
        # node, whose contraction IS the residual term. A root is never
        # materializable as a build_W (naming it collapses the term to one
        # operator instead of factoring it), so ranking it is meaningless.
        for nt, r in emittable_operators(t, fingerprints):
            name = r.op_name if isinstance(r, Reuse) else r.name
            uses[name] += 1
            kind[name] = "reuse" if isinstance(r, Reuse) else "derived"
            build[name] = build_cost(nt)  # tree cost of building this node once
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

    # M0: footprint + density inventory — the memory/locality baseline. Savings
    # and density are memory-blind vs memory-aware views of the SAME operators.
    ops = manifold_operators(triples, include_reuse=False)
    inv_fp = footprint_inventory(ops, n_occ=30, n_vir=100)
    print("\nM0 footprint inventory (O=30,V=100) — savings vs bytes vs density:")
    print(f"  {'savings':>10} {'GB':>8} {'flops/byte':>11}  operator")
    for e in inv_fp[:8]:
        print(f"  {e.savings:10.2e} {e.bytes/1e9:8.2f} {e.density:11.2e}  {e.name}")
    top_s = inv_fp[0].name
    top_d = max(inv_fp, key=lambda e: e.density).name
    print(f"  savings-top={top_s}  density-top={top_d}  "
          f"(disagree: {top_s != top_d})")


if __name__ == "__main__":
    _demo()
