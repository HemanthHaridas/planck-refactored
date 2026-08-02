# Factorize higher-rank CC contractions to cut FLOPs and expose the operators

**The thread.** D7 dressing works by *recognizing* six curated CCSD intermediates
(Fae/Fmi/Fme, Wmnij/Wabef/Wmbej) as subgraphs and folding them out. Going to
CCSDT/CCSDTQ, the naive plan is to hand-seed each rank's new intermediates. This
document scopes a sharper, more general approach that comes at the same
intermediates from the other direction:

> **Re-associate (factorize) each residual contraction into a binary tree that
> minimizes the FLOP exponent. The factoring both reduces cost AND *identifies*
> the reused sub-contractions — the intermediate that a good factorization step
> produces often IS a known operator, and where it is not, it is a new operator
> the factorization has just derived.**

Recognition (D7) and factorization are two routes to the same dressed form.
Recognition needs a curated operator set up front; **factorization derives the
operators** as the shared sub-contractions a cost-minimizing binary tree exposes.
That makes the earlier "are higher operators combinations of the CCSD six?"
question a *consequence*: run the factorizer, then check which produced
sub-contractions match seeded operators (reuse) vs are new (the derived
curated-set for that rank).

**Why the lever is real (measured).** Association order changes the peak
contraction exponent. A `t2·t3·v` triples term written as one n-ary contraction
is ~`o⁵v⁵`; factoring `(v·t3)` into an intermediate first, then contracting with
`t2`, drops the peak step to ~`o³v⁴`. The FLOP win and the intermediate are the
same act of factoring — that is the whole point of this thread.

This is a **research investigation**, not a committed feature. A "these terms
don't factor below the bare cost" answer is a valid result.

---

## What there is to factor (the grounding probe)

The CCSDT **doubles** manifold is already dressed exactly by the CCSD six
(measured: 0 mismatches — its `t3`-bearing terms stay bare, everything else
reuses Wmnij/Wabef/Wmbej/F). The factoring target is the **triples** manifold: of
its 399 terms, **120 are `t3`-bearing** — multi-factor contractions no CCSD
operator covers, and the terms whose association order matters most. Their shapes:

| count | shape | ERI core blocks (the would-be operator) |
|---|---|---|
| 39 | `t2·t3·v` | vvvoo, vovoo, … |
| 36 | `t1·t3·v` | |
| 24 | `t1·t1·t3·v` | vvvov, vovov, … |
| 15 | `t3·v`   | voooo, vvvvv |
|  6 | `f·t3`   | — |

The suggestive part: every `t3`-bearing term is `t3` contracted with a
**dressing built from the same primitives the CCSD operators are made of**
(`t1·v`, `t2·v`, `t1·t1·v`, `f`). A CCSDT triples W-operator's *definition* is a
sum of exactly those pieces. So the structural question is concrete: **is the
factor `t3` contracts against, in the triples residual, (a signed sum of) the
existing CCSD operators re-blocked, or a genuinely new object?**

Three a-priori outcomes for what factoring exposes, each with precedent:
- **Reused operator.** `t2·t3·v` factors as `(v·t3)`-first or `(v·t2)`-first, and
  the intermediate is a re-blocked CCSD operator — `t3·Wmbej`-shaped, `t1·t3·v` as
  `t3·Fme`/`Fae`-shaped. (Literature: CCSDT does reuse `Wmbej`, `Fae`, `Fmi` in
  the triples equation.)
- **Newly-derived operator.** The best-factoring intermediate has no CCSD analog
  (e.g. a `vvvoo`-block object) — factorization *derives* it, and it becomes a
  new seeded operator for this rank.
- **No useful factoring.** The term admits no association below its n-ary cost —
  emit it bare. (The honest ceiling.)

The literature leans toward "mostly reused + a few new" — exactly what the
factorizer measures. Both wins land regardless: FLOPs drop, and the derived
operators are the curated set for the rank, obtained mechanically.

---

## What "factorize" must mean (make it falsifiable)

Vague "rearrange" is untestable. Pin it to a cost model + the existing exact gate.

A residual term `T = c · f₁·f₂·…·fₖ` (a single n-ary contraction over its summed
indices) is **factorizable** iff there is a **binary contraction tree** over its
factors — a sequence of pairwise contractions, each producing a named
intermediate — such that:
1. **exact:** the tree evaluates to `T` (index-for-index; guaranteed by
   associativity of tensor contraction, so this is a bookkeeping check, not a
   numeric one), and
2. **cheaper:** the tree's **peak pairwise-step exponent** is strictly below the
   n-ary exponent (the FLOP win — the D7.4-style honesty check, now on a real
   contraction-path cost, not element count).

A tree is **operator-identifying** iff one of its intermediate nodes, canonicalized,
matches a seeded operator's `IntermediateSpec` (reuse) — or, failing a match, is
recorded as a *newly derived* operator (its definition = that sub-contraction).

The current `estimated_build_flops` counts an intermediate's **element count**
(∏ index sizes), NOT the pairwise-step cost that determines scaling — so a real
**contraction-path cost model** is the one genuinely new piece (see F1). Exactness
(1) reuses `assemble_dressed_equation` + `dressed_multiset == raw`.

A term that admits no cost-lowering tree is irreducible-for-flops (emit it bare);
a tree whose best intermediate matches a seeded operator is the "reuse" case that
the earlier reuse-question was really asking about — now answered as a byproduct.

---

## Scope — small verifiable steps

A factorization investigation. The engine is a contraction-path cost model + a
binary-tree search; each step returns a concrete measurement.

### F0 — enumerate the target terms (~S)
Per method/manifold, list the multi-factor residual terms with their factor +
index-space structure — the candidates for factoring (the probe above is the
CCSDT-triples start: 120 `t3`-bearing terms). Output: the term inventory + the
n-ary peak exponent of each (the baseline to beat). *Gate:* the inventory matches
the raw residual; the n-ary exponents are computed correctly on a hand-checked
case.

### F1 — contraction-path cost model (~M, the one new piece)
A function `contraction_tree_cost(factors, summed_indices)` giving the peak
pairwise-step exponent of a given binary tree, and `best_contraction_tree(term)`
searching association orders for the minimum-peak tree (small factor counts →
exhaustive over pairings; the CCSD/CCSDT terms have ≤5 factors). NOT the existing
`estimated_build_flops` (element count) — this is real path cost. *Gate:* on the
`t2·t3·v` example, it finds the `(v·t3)`-first tree (`o³v⁴`) and reports it below
the n-ary `o⁵v⁵`; on a term with no cost-lowering association it returns the
n-ary cost unchanged.

### F2 — operator identification from the tree (~S given F1)
For each intermediate node the best tree produces, canonicalize it and test
against `seeded_operators()` (reuse `_eri_canonical` + the containment logic in
`hypothesis_is_consistent`): match → reuse an existing operator; no match →
record it as a newly-derived operator (its `IntermediateSpec` = that node's
sub-contraction). *Gate:* on the `t2·t3·v` tree, the `(v·t3)` node is identified —
either as a re-blocked known operator or recorded as new — and its
`IntermediateSpec` round-trips (expands back to the node exactly).

### F3 — factor the whole CCSDT triples manifold (~S given F1/F2)
Run F1+F2 over the 120 `t3`-bearing terms. Output: per term, its best tree +
FLOP win + the operators it identifies (reused vs newly-derived). Tabulate the
distinct derived operators — this IS the CCSDT curated set, obtained by
factorization rather than by hand. *Gate:* substituting the factored trees keeps
`assemble_dressed_equation` exact on CCSDT doubles+triples (the associativity
bookkeeping check), and every term's tree peak-exponent ≤ its n-ary.

### F4 — reuse verdict (answers the original reuse question) (~S)
Of the operators F3 derived, how many match the CCSD six (or re-blockings of
them) vs are genuinely new? The ratio settles "are higher operators combinations
of CCSD operators?" — now as a measured byproduct of factorization, not a
separate search. *Gate:* the classification is complete; reused vs new counts
reported per operator.

### F5 — generalize to CCSDTQ (~M, only if F3/F4 are promising)
Repeat F0–F4 for `t4`. Sub-question: does factoring CCSDTQ reuse CCSD **and**
CCSDT-derived operators (recursive reuse), making the derivation rank-recursive?

---

## Sequencing & risk
- **F1 is the one ~M piece** — the contraction-path cost model + tree search.
  De-risk it on the `t2·t3·v` example (known `o⁵v⁵ → o³v⁴`) before trusting F3's
  sweep. The search is bounded (≤5 factors ⇒ few pairings), so no heuristic
  needed at CCSDT scale.
- F2–F4 are measurement over F1 + the existing exact gates; each returns a
  concrete table, so the investigation degrades gracefully (partial answers are
  still answers).
- **Cheapest first step: F0 + F1's de-risk** — the term inventory with n-ary
  exponents, and the cost model proving `t2·t3·v` factors from `o⁵v⁵` to `o³v⁴`.
  That validates the FLOP lever and the engine before the 120-term sweep.

## Honest ceiling (the NP-hardness caveat, sharpened)
Optimal contraction-path selection over a whole equation is NP-hard in general.
This investigation is bounded: per-term trees with ≤5 factors are exhaustively
searchable, and the goal is not the global optimum but (a) each term below its
n-ary cost and (b) identifying the shared intermediates. Cross-term intermediate
sharing (the same node reused across terms — where the real win compounds) is the
harder follow-on; F3's derived-operator table is the input to it, but scheduling
the shared builds optimally is out of scope here.

## What this reuses (no new machinery except F1's cost model)
- `expand_dressed_term` / `_eri_canonical` — expansion + canonical keys.
- `hypothesis_is_consistent` — the sound containment filter (adapt to "is this
  combination contained in the target").
- `assemble_dressed_equation` + `dressed_multiset == raw` — the exact gate.
- `intermediate_dependencies` / the emit path — unchanged; a derived operator is
  still an `IntermediateSpec` whose definition happens to reference other
  operators.

See `CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` (Decision 4) for the recognition/
assembly machinery this builds on, and the `cc_canonical_fock_only` invariant
(all of this runs against the canonical-Fock residual).
