# Factorizing higher-rank CC contractions: the rank-locality of derived operators

**Status: investigation complete.** The engine, all gates, and the results below
are landed in `python/ccgen/optimization/factorize.py` and
`python/ccgen/tests/test_factorize.py` (33 tests). This document states the
structural theorem the investigation produced, then answers the research
question it was opened to settle.

---

## The research question

D7 dressing works by *recognizing* six curated CCSD intermediates (Fae/Fmi/Fme,
Wmnij/Wabef/Wmbej) as subgraphs and folding them out. Extending that to
CCSDT/CCSDTQ, the naive plan is to hand-seed each rank's new intermediates. This
investigation asked whether a sharper, mechanical route works instead:

> **Re-associate (factorize) each residual contraction into a binary tree that
> minimizes the FLOP exponent. Does the factoring both reduce cost AND *derive*
> the reused sub-contractions — so that the intermediate a good factorization
> step produces either IS a known operator (reuse) or is a new operator the
> factorization has just derived?**

The concrete sub-question that motivated it: **"are the higher-rank operators
combinations of the CCSD six, or genuinely new — and does that structure repeat
across rank?"** Recognition (D7) needs a curated operator set up front;
factorization would *derive* the set as the shared sub-contractions a
cost-minimizing tree exposes.

**Answer, in one line:** yes on all counts — factorization both cuts the FLOP
exponent and derives the operators, and the derived set obeys a precise
rank-locality structure (the theorem below). The rest of this document is that
result and its evidence.

---

## The rank-locality theorem

This is a theorem about **the F3 optimization model**, not about coupled-cluster
theory as such. The model is: exhaustive enumeration of binary contraction trees
(per term, ≤5 factors), over **diagram-generated, canonical-Fock** residuals,
with the minimum-peak-exponent tree selection. Within that model the statement
is exact; nothing here claims universality beyond it.

Notation. Write `Rₙ` for the rank-`n` residual manifold (e.g. the CCSDT triples,
`n=3`), `Tₙ` for the highest-rank cluster amplitude its terms carry, and `V·Tₘ`
for a derived operator whose *definition* contracts the two-electron integral `V`
with amplitude `Tₘ`. A term is "`Tₙ`-bearing" if `Tₙ` is one of its factors.

The investigation set out to formalize this conjecture:

> *In `Rₙ`, contraction trees containing `Tₙ` generate and reuse a rank-local
> family of `V·Tₙ`-derived operators. Lower-rank derived operators `V·Tₘ` (m<n)
> are reused only in those terms of `Rₙ` that do NOT involve `Tₙ`.*

Its first sentence is **true**; its second is **false**. The two clauses
conflate what an operator is *built from* (its definition's amplitude content)
with which *terms* reuse it (whether a term's tree contains that operator as a
node). Separating them gives:

> **Theorem (rank locality, within the F3 model).**
> 1. **(Rank-local generation.)** Every derived operator whose definition
>    contains `Tₙ` is generated *only* in `Tₙ`-bearing terms of `Rₙ`.
> 2. **(Compositional separation.)** No operator whose definition contains `Tₙ`
>    appears as a node in a `Tₙ`-free term. (This is the *true* content the
>    conjecture's second clause was reaching for — stated on operator
>    *composition*, not on the reused operator's rank.)
> 3. **(Lower operators are NOT confined — the main result.)** Lower-rank
>    operators `V·Tₘ` (m<n) *are* reused in `Tₙ`-bearing terms, via a `(Tₘ·V)`-
>    first association whose intermediate then contracts *against* `Tₙ`. The
>    conjecture's "only in `Tₙ`-free terms" is thereby refuted.
>
> **Observed cumulative property (CCSDT → CCSDTQ).**
> 4. For the ranks investigated, every operator derived at the lower rank is
>    reused verbatim at the higher one (both across methods and in the higher
>    method's lower manifolds). Each investigated rank contributes only a new
>    `V·Tₙ` family while preserving all previously derived operators.

### Proofs

**Parts 1 and 2 are structural** — they follow from the tree representation
alone, independent of the particular coupled-cluster equations. An operator is
generated in / reused by a term only as an internal node of that term's
contraction tree, whose leaves are exactly the term's factors. If the operator's
definition contains `Tₙ`, its node's subtree contains a `Tₙ` leaf, which the term
must therefore supply. Contrapositive: a `Tₙ`-free term can neither generate nor
carry a `V·Tₙ` operator. No measurement is needed for the claim; the enumeration
is **implementation verification**. *Verified:* CCSDT (`n=3`) exposes 10 distinct
`V·t3` operators, **0** appearing in any `t3`-free triples term; CCSDTQ (`n=4`)
exposes 10 distinct `V·t4` operators, **0** in any `t4`-free quadruples term. ∎

**Part 3 — lower operators are not confined (the scientific contribution).**
The conjecture claimed `V·Tₘ` (m<n) is reused *only* in `Tₙ`-free terms. This is
false, and the reason is the crux of the whole result: **operator composition,
operator reuse, and excitation rank are three distinct concepts.** A `Tₙ`-bearing
*term* can reuse a *lower*-rank operator, because association order can route the
term through a low-rank intermediate before touching `Tₙ`. Counterexample class,
directly enumerated: in `Rₙ` terms of shape `Tₘ·Tₙ·V` whose contracted indices
are shared between `Tₘ` and `V` (not `Tₙ` and `V`), the minimum-peak tree
contracts `(Tₘ·V)` first, producing a `V·Tₘ` node, which then contracts against
`Tₙ`. Concrete CCSDT witness: `¼ t2(d,e,i,j) t3(a,b,c,k,l,m) v(d,e,l,m)` — a
`t3`-bearing term — factors through a `W_t2v_oooo` node (a `V·t2` operator).
*Measured:* lower-rank `V·Tₘ` operators appear in `Tₙ`-bearing terms **36** times
in CCSDT triples and **64** times in CCSDTQ quadruples — nonzero, so the clause
is refuted. What Part 2 excludes from `Tₙ`-free terms is not a *rank*, it is only
operators *built from `Tₙ` itself*. ∎

**Part 4 — observed cumulative property (CCSDT → CCSDTQ), not deduced.** This is
an observation over two ranks, not a proof for arbitrary excitation rank. Let
`D(R)` be the set of derived-operator keys a manifold `R` exposes
(`recursion_summary`). *Measured:* `D`(CCSDT triples) has 35 operators;
`D`(CCSDTQ triples) has 38; the overlap is **35** with **0** CCSDT-only — i.e.
`D`(CCSDT triples) ⊆ `D`(CCSDTQ triples), the 3 extras all `t4`-bearing.
Independently, the CCSDT-derived `V·t3` operators (`W_t3v_ooovvv`, `W_t3v_oooovv`,
…) recur inside CCSDTQ's *own* quadruples manifold, from its non-`t4` terms
(`t3·t3·v` etc.). These observations suggest a recursive hierarchy in which each
excitation rank extends the derived-operator basis by introducing only its own
`V·Tₙ` family — but establishing it for all ranks would need an independent
proof, out of scope here.

```
  CCSD           Fae  Fmi  Fme  Wmnij  Wabef  Wmbej      (6 seeded)
                 │
                 ▼   reused verbatim
  CCSDT          (all CCSD operators)  +  V·T3 family     (+35 derived)
                 │
                 ▼   reused verbatim
  CCSDTQ         (all previous)        +  V·T4 family     (+10 t4-bearing)
```

**Corollary (recursive intermediate library).** Because the derived-operator
basis is cumulative across the investigated hierarchy, an implementation need
only extend the intermediate library by the newly introduced `V·Tₙ` family at
each excitation rank; previously generated `build_W*` kernels are reused
unchanged. A materialize-once scheme is therefore rank-recursive *for the
investigated ranks* by construction, not per-method.

*Gates* (in `RankLocalityTheoremTests`, parametrized over CCSDT `n=3` and CCSDTQ
`n=4`): `test_part1_and_2_Vtn_ops_only_in_Tn_terms` (Parts 1 & 2 — 0 `V·Tₙ`
appearances in any `Tₙ`-free term); `test_part3_lower_ops_do_appear_in_Tn_terms`
(Part 3 — the nonzero `36`/`64` lower-rank reuse in `Tₙ`-bearing terms). Part 4
(observed) is gated in `CCSDTQTests` by `test_ccsdt_operators_reused_in_ccsdtq_triples`
and `test_recursion_summary_is_cumulative`.

---

## Answering the research question

### 1. Does factorization cut the FLOP exponent? Yes.

Association order changes the peak contraction exponent. Measured, the
minimum-peak tree drops the exponent on every multi-factor `Tₙ·V`-family term,
and the win grows with rank:

| term | n-ary peak | best-tree peak |
|---|---|---|
| `t2·t3·v` (CCSDT) | `o⁵v⁵` (deg 10) | `o⁴v³` (deg 7) |
| `t2·t4·v` (CCSDTQ) | `o⁶v⁶` (deg 12) | `o⁵v⁴` (deg 9) |
| `t1·t4·v` | `o⁶v⁵` | `o⁶v⁴` |
| `t4·v`, `f·t4` | — | best == n-ary (single step, correctly not factored) |

The FLOP win and the intermediate are the *same act of factoring*: the tree step
that lowers the exponent is exactly the step that materializes a reusable
sub-contraction.

### 2. Does factorization derive the operators? Yes.

Running the tree search + node classification over the CCSDT triples exposes 40
distinct operators. **5 of the 6 CCSD operators are reused** (Fae/Fme/Fmi/Wmbej/
Wmnij; Wabef's pure-`vvvv` block never arises in a triples node), and the rest
are **newly derived** — the `t3·v` family (`W_t3v_ooov`, `W_t3v_ovvv`,
`W_t3v_ooovvv`, …) plus the mixed `t1/t2`-dressing operators. No hand-seeding: the
curated set for the rank falls out of the cost-minimizing trees. The exact gate
`tree_preserves_term` confirms all 399 triples (and all 2672 quadruples) trees
reproduce their raw term, so the derivation changes cost, not the answer.

### 3. Are the higher operators combinations of the CCSD six? Partly — and the theorem says exactly how.

This was the original sub-question, and the theorem is its precise answer:
- Every rank reuses **all** lower-rank operators (the CCSD six, then the CCSDT
  `V·t3` family, …) — Parts 1, 2, 4.
- Each rank adds **one genuinely new family**, `V·Tₙ`, that no lower rank
  contains — the `t3·v` operators at rank 3, the `t4·v` operators at rank 4.
- The reuse is not confined by term rank: a high-rank term routinely reuses a
  low-rank operator (Part 3).

So "combinations of the CCSD six" is *true for the reused part and false for the
new part* — the higher ranks are CCSD-six-plus-a-new-family, cumulatively.

### 4. What is worth materializing? The expensive derived operators, not the frequent ones.

A raw reuse count is the wrong metric: an operator reused 15× that avoids an
`o³v⁵` build outranks one reused 75× saving only `o²v²`. Weighting by
`savings = (uses − 1) × build_flops` (scaling-dominated `o^a·v^b`, v>o) inverts
the naive ranking:

| rank | top operator by savings | build | uses | savings | best CCSD reuse |
|---|---|---|---|---|---|
| CCSDT | `W_t3v_ooovvv` (derived) | `o³v⁵` | 15 | ~4.4e12 | Wmbej `o³v³` far below |
| CCSDTQ | `W_t4v_oooovvvv` (derived) | `o⁴v⁶` | 28 | ~4.2e15 | Wmbej `o³v³` at ~3.9e9 — 6 orders below |

The operators worth caching are exactly the expensive `Tₙ·V` intermediates the
investigation set out to derive; the cheap high-frequency CCSD reuses are
near-worthless to materialize. (`value_operators` returns the ranked list.)

### 5. Does the structure repeat across rank? Yes — cumulatively (Part 4).

All 35 CCSDT-triples derived operators reappear in CCSDTQ's triples (zero
CCSDT-only), and recur inside CCSDTQ's quadruples. CCSDTQ adds only its 10
`t4`-bearing operators. The derived-operator basis is cumulative for the two
ranks measured; the corollary is the implementation consequence — one library,
extended by `V·Tₙ` per rank, kernels reused unchanged.

---

## The engine (how the results were produced)

`python/ccgen/optimization/factorize.py`. A contraction-path cost model + binary
tree search + node-level operator identification. Rank-agnostic: the same
functions run on CCSDT and CCSDTQ.

**Cost model.** `Cost(n_occ, n_vir)` is a step's loop-nest exponent (the distinct
occ/vir indices it touches). `nary_cost(term)` is the term as one blob;
`best_contraction_tree(term)` searches all binary associations (≤5 factors ⇒
exhaustive) for the minimum-peak tree. `Cost.flops(o,v) = o^a·v^b` gives the
scaling-dominated magnitude used for savings. This is real path cost — NOT
`IntermediateSpec.estimated_build_flops`, which is an element count.

**Tree → operator.** `best_contraction_tree_full` returns the actual `Node` tree.
`node_to_term(node)` lowers each internal node to an `AlgebraTerm` (its subtree's
leaf factors, indices consumed at that step, output block). `node_key` gives it a
canonical, factor-order-independent key via dressing's `_eri_canonical`.

**Classification.** `seeded_fingerprints()` keys the six CCSD operators'
definition terms; `identify_node` returns `Reuse(op)` on an exact keyed match
(block-signature prefiltered) or `Derived(IntermediateSpec)` otherwise, with a
sorted, order-invariant derived name. `identify_tree` classifies every node;
`value_operators` ranks by savings; `recursion_summary` reports the cross-rank
containment.

**Determinism (load-bearing).** 41% of triples terms admit more than one
minimum-peak association. The selection key is a total order —
`(peak.total, peak.n_vir, −max_intermediate_build_flops, canonical_tree_signature)`
— which, with sorted derived names, makes the operator multiset a deterministic
function of the terms (was 47 wobbling names under factor shuffle, now 40 stable).

**Exactness.** `tree_preserves_term(t)`: every raw factor is one tree leaf and
every summed index is consumed at one node. Associativity then guarantees the
tree equals the raw n-ary contraction, coefficient untouched. All 399 triples +
2672 quadruples pass.

---

## Honest ceilings and remaining work

- **The theorem is model-local.** Parts 1–3 are exact within the F3 optimization
  model (exhaustive ≤5-factor trees, canonical-Fock diagram residuals). They are
  not claims about coupled-cluster theory in general, nor about heuristic
  contraction-path optimizers.
- **Part 4 is two-rank evidence, not a proof.** Cumulativity is measured for
  CCSDT → CCSDTQ. Establishing it for arbitrary rank would need an independent
  proof.
- **Cross-term scheduling is out of scope (the real compounding win).** The
  per-term trees identify the shared intermediates; scheduling their builds once
  across the whole manifold — where the largest savings compound — is NP-hard in
  general and not attempted here. `value_operators` is the input to it: it says
  which operators are worth the shared build.
- **Emit is not wired.** The derived operators are `IntermediateSpec`s; feeding
  them into the actual generated kernels (the D7.3 emit path) is a separate
  integration, scoped below.

## Scoping the emit integration (cost vs savings)

The emitter's `--include-intermediates` uses `detect_intermediates` (verbatim CSE),
NOT the factorizer. The two speak the same type (`IntermediateSpec`) but disagree
on what an intermediate *is*: CSE hoists recurring leaves verbatim; the factorizer
derives operators from cost-minimizing trees and keys them canonically. The
measured gap (CCSDTQ, diagram, canonical Fock, doubles+triples+quadruples):

| | count | basis |
|---|---|---|
| `detect_intermediates` (CSE, threshold 5) | **212** builders | verbatim recurring sub-contractions |
| `value_operators` (factorizer) | **80** derived operators | canonical, topology-merged |

Canonical keying alone collapses ~132 CSE duplicates (the topology-distinct
`W_oovvvv_2..N` families that are the same integral in a different ERI slot).
**And the savings are extremely concentrated: the top 10 derived operators carry
90.8% of the total savings; the remaining 70 carry ~9%.** So the integration is
not all-or-nothing — a small ranked, budgeted emit captures nearly all the win.

That reshapes the work from "replace the CSE pass" into a cost-bounded add-on:

- **E0 — bridge derived specs to the emit path (~M).** Route the factorizer's
  `Derived(...).spec` `IntermediateSpec`s into `rewrite_equations` +
  `emit_planck_translation_unit`, as an alternative to `detect_intermediates`.
  A factorized spec is ALREADY a valid single-def-term `IntermediateSpec` (factors
  + free + summed indices), so the *type* fits and `_try_substitute` consumes it —
  the blocker is two structural mismatches, both measured:
  1. **Root node is not an operator.** `identify_tree` returns a spec for every
     internal node INCLUDING the root, whose def-term factors are the whole term.
     `_try_substitute` is greedy-first, so the root spec matches first and collapses
     the entire term to one `W_...` reference — a rename, not a factorization
     (measured: a `t2·t3·v` term collapsed to `W_t2t3v_ooovvv` instead of
     `t2·W_t3v_ooov`). The root contraction IS the residual; it must never emit a
     `build_W`.
  2. **Substitution must be hierarchical, innermost-first.** The factorizer's specs
     are nested (a node and its sub-nodes); `_try_substitute` is flat. The correct
     factored term is `t2 · W_t3v_ooov` (root step over the leaf `t2` and the inner
     `(t3·v)` operator), which requires substituting the child operator, then
     leaving the root as a plain contraction against it.

  E0 sub-steps:
  - **E0.0 — non-root operator set (~S). LANDED.** `emittable_operators(term)`
    drops the root node (its contraction is the residual) and returns the
    internal-but-non-root nodes classified as `Reuse`/`Derived`. `internal_nodes`
    yields the root first, so dropping index 0 is the whole fix. *Gate (in
    `CostModelTests`):* `test_emittable_drops_root_operator` (a `t2·t3·v` term
    yields `{W_t3v_ooov}`, not the root `W_t2t3v_ooovvv`) and
    `test_no_emittable_operator_equals_its_term` (manifold-wide: no emitted
    operator has its source term's factor multiset — no leaked root).
  - **E0.1 — hierarchical substitution (~M, the real work). LANDED.**
    `rewrite_term_factorized(term)` walks the `Node` tree (cleaner than patching
    the flat `_try_substitute`) and emits the ROOT contraction step: the root's
    children as factors, each internal child replaced by a reference
    `Tensor(op_name, child.block)`, each leaf child left bare. Innermost-first is
    automatic — a child's sub-structure lives in ITS `build_W`, so the root step
    only names its immediate children. `t2·t3·v` → `½ t2(a,b,i,l)
    W_t3v_ooov(j,k,l,c)` (inner `m,d,e` moved into the operator, only `l` survives,
    coeff preserved). *Gate (in `CostModelTests`):* `test_rewrite_factors_t2t3v`,
    `test_rewrite_single_step_term_unchanged` (2-factor terms pass through), and
    `test_rewrite_is_exact_over_manifold` — re-expanding each factored term (root
    leaves + operator definition leaves) reproduces the original factor multiset
    across all 399 triples, 0 failures.
  - **E0.2 — dedup specs across the manifold (~S). LANDED.**
    `manifold_operators(terms)` collects one `IntermediateSpec` per operator
    NAME, `usage_count` = reference-site count. Dedup is by name, NOT by
    canonical `node_key`: the name (factor set + block signature) is the operator
    identity — one `build_W` indexed at each site with that site's externals —
    whereas the key over-splits an operator into its external-relabeling
    instances (measured on CCSDT triples: 447 sites → 24 operators by name, but 74
    by key; verified all instances of a name share ONE index-space shape, so the
    key split was spurious). Emittable count excludes roots (E0.0) and Reuse
    (CCSD) ops, so it is smaller than `value_operators`' 80: **CCSDT → 24
    operators / 499 sites; CCSDTQ → 43 / 4155.** *Gate (in `CostModelTests`):*
    `test_manifold_operators_deduped_by_name` (24, distinct, no CCSD),
    `test_manifold_operator_usage_counts_reference_sites`,
    `test_manifold_operator_indices_match_signature`.
  - **E0.3 — emit + compile (~S given E0.0–E0.2). LANDED.**
    `emit_factorized_translation_unit(method)` runs the full pipeline: rewrite
    every term (E0.1, derived-only), collect operators (E0.2,
    `include_reuse=False`), hand both to `emit_planck_translation_unit`. Two
    boundary bugs surfaced and were fixed:
    1. **CCSD-operator factors don't lower.** A `Reuse` child (e.g. `t1·v` = Fme)
       hoisted to an `Fme(...)` factor hits `NotImplementedError` — its definition
       needs `tau`/`tau_tilde` builders the factorizer doesn't own (that is D7.3's
       `dress_operators`). Fix: `rewrite_term_factorized(derived_only=True)`
       inlines `Reuse` children (re-absorbing their consumed indices) and hoists
       only `Derived` operators — the factorizer's unique contribution. CCSD
       dressing stays D7.3's job; the two paths are complementary.
    2. **Multi-factor operators under-declared their summation.** `node_to_term`
       records only the top tree step's summed index, but a standalone `build_W`
       for e.g. `W_t2t2v_oooovv` (itself a multi-step contraction) must loop over
       EVERY internal contraction index, else the emitted builder references
       undeclared vars (compile error `use of undeclared identifier 'm'`). Fix:
       `_complete_definition_summation` recomputes summed = (all factor indices) −
       (block) when building the standalone spec.
    *Gate (in `CostModelTests`):* `test_factorized_tu_is_wellformed` (balanced TU,
    24 `build_W`, no leaked CCSD factor) and `test_factorized_tu_compiles` — the
    factorized CCSDT TU passes `c++ -std=c++23 -fsyntax-only` against the real CC
    headers (same pattern as the tau compile gate; skipped without a compiler /
    Eigen fetch). The numeric energy-equivalence run is **E2** (needs a compiled
    binary); E0.1's `test_rewrite_is_exact_over_manifold` already proves the
    algebra is unchanged structurally.
- **E1 — savings-budgeted selection (~S, given E0).** Emit only the top-`k`
  operators by `value_operators` savings (or until a cumulative-savings /
  builder-count budget is hit), inline the rest. The 90.8%-in-top-10 curve is the
  knob: default to ~top-15 (>93% of savings) and a CLI/CMake
  `--intermediate-savings-budget`. *Gate:* emitted-builder count drops from 212
  toward the budget with measured savings retained ≥ the target fraction.
- **E2 — cross-engine equivalence + energy gate (~S).** The factorized-intermediate
  TU must produce the same CC energy as the plain (`detect_intermediates`) TU and
  the hand-written solver. Reuses the exact gate `tree_preserves_term` (already
  proves the algebra) plus a compiled-energy comparison. *Gate:* CCSDTQ energy on
  a 4-electron system matches FCI/hand-solver to ~1e-10 with factorized emit on.

**What it is NOT.** Still not cross-term scheduling (materialize-once across the
whole manifold, NP-hard) — E0–E2 emit per-term trees with shared *named* operators,
which the compiler/runtime can hoist, but optimal shared-build ordering stays out
of scope. The savings ranking (`value_operators`) is what makes E1 a bounded
decision rather than a search.

**Priority read.** E0 is the only real work (~M, the rewrite-representation
bridge). Given the 90.8%-in-top-10 concentration, even a partial E0 that only
handles the ≤10 highest-savings operators — all of the same `W_t4v*`/`W_t3t3v*`
`oooovvvv`/`ooooovvv` shape — would capture most of the FLOP win. The 70-operator
tail is where E0's generality cost lives for ~9% of the benefit, so E1's budget
is the natural stopping point.

## What this reuses

- `_eri_canonical` / `expand_dressed_term` — canonical keys + expansion.
- `seeded_operators()` / `DressedOperator` — the six CCSD fingerprints.
- `IntermediateSpec` — a derived operator is one, ready for the emit path.
- Diagram engine + `canonical_fock=True` — the residual source (see the
  `cc_canonical_fock_only` invariant).

See `CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` (Decision 4) for the recognition/
assembly machinery this builds on.
