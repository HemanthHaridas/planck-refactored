# Can factorizing CC residual contractions derive the reused intermediates?

**The one question.** Going from CCSD to CCSDT/CCSDTQ, the naive plan is to
hand-seed each rank's new dressed intermediates (`Wmnij`, `Wabef`, the triples
W-operators, …). This investigation asked whether a mechanical route replaces
the hand-seeding:

> **If you re-associate each residual contraction into the binary tree that
> minimizes the FLOP exponent, does the factoring both cut the cost AND *derive*
> the reused intermediates — so each tree node is either a known operator (reuse)
> or a new operator the factorization has just produced — and does that operator
> structure repeat across excitation rank?**

**The answer is yes**, and its precise form is a rank-locality theorem. The rest
of this document states that answer, gives the evidence, and lists what is built.
Everything is landed in `python/ccgen/optimization/factorize.py`, gated by
`python/ccgen/tests/test_factorize.py` (47 tests).

> **Those 47 gates are STRUCTURAL ONLY, and the distinction is load-bearing.**
> `tree_preserves_term` checks that each factor is one leaf and each summed index is
> consumed once; `test_budgeted_rewrite_is_exact` checks the rewrite re-expands to the
> same factor **`Counter`**. A `Counter` of factor names is blind to index order by
> construction, so neither can see a rewrite that names the right tensors in the wrong
> slots.
>
> Measured 2026-08-22: evaluating the rewritten manifold numerically, **23 of 66 GCC
> `ccsd` doubles terms do not reproduce their source term** (‖diff‖/‖R‖ = 3.73e-01);
> 46 of 113 on the spatial manifold. Two contributing shapes are demonstrated —
> `_derived_name` builds a name from sorted factor names + output block signature, so
> it is order-blind, and an i↔j exchange pair collapses onto one operator — but
> neither predicate covers all the disagreements, so the mechanism is not yet fully
> characterized. See `docs/CCGEN_TWO_DRESSING_ROUTES.md` (D0).
>
> The rank-locality answer below is about which operators *appear*, and is unaffected.
> What is unproven is that **hoisting them preserves the residual's value**.

---

## The answer, precisely: the rank-locality theorem

Scope: a theorem about **the optimization model**, not coupled-cluster theory —
exhaustive enumeration of per-term binary contraction trees (≤5 factors), over
**diagram-generated, canonical-Fock** residuals, minimum-peak-exponent tree
selection. Within that model it is exact.

Notation. `Rₙ` = the rank-`n` residual manifold (CCSDT triples: `n=3`); `Tₙ` =
the highest-rank amplitude its terms carry; `V·Tₘ` = a derived operator whose
*definition* contracts the ERI `V` with amplitude `Tₘ`. A term is "`Tₙ`-bearing"
if `Tₙ` is one of its factors.

> **Theorem (rank locality).**
> 1. **Rank-local generation.** Every derived operator whose definition contains
>    `Tₙ` is generated *only* in `Tₙ`-bearing terms of `Rₙ`.
> 2. **Compositional separation.** No operator whose definition contains `Tₙ`
>    appears as a node in a `Tₙ`-free term.
> 3. **Lower operators are NOT confined (the main result).** Lower-rank operators
>    `V·Tₘ` (m<n) *are* reused in `Tₙ`-bearing terms — via a `(Tₘ·V)`-first
>    association whose intermediate then contracts *against* `Tₙ`. So operator
>    composition, operator reuse, and excitation rank are three distinct concepts.
>
> **Observed cumulative property (CCSDT → CCSDTQ).**
> 4. For the ranks measured, every operator derived at the lower rank is reused
>    verbatim at the higher one. Each rank adds only its own `V·Tₙ` family.

```
  CCSD    Fae Fmi Fme Wmnij Wabef Wmbej          (6 seeded)
          │ reused verbatim
  CCSDT   (all CCSD)  +  V·T3 family              (+35 derived)
          │ reused verbatim
  CCSDTQ  (all previous)  +  V·T4 family          (+10 t4-bearing)
```

**Why this IS the answer to the question.** The question had three clauses;
the theorem answers each. *Does factoring derive the operators?* — yes, every
tree node is a reuse or a new `IntermediateSpec` (Parts 1–3). *Are higher
operators combinations of the CCSD six?* — the reused part yes, the new `V·Tₙ`
family no; each rank is CCSD-plus-a-new-family (Parts 1–2, 4). *Does the structure
repeat across rank?* — yes, cumulatively (Part 4). The corollary: the
`build_W*` kernels are shared across the method hierarchy — a materialize-once
scheme extends the library by one `V·Tₙ` family per rank, reusing all prior
kernels unchanged.

### Proofs

**Parts 1–2 are structural** (independent of the CC equations): an operator is
generated in / reused by a term only as an internal node of that term's tree,
whose leaves are the term's factors; a definition containing `Tₙ` needs a `Tₙ`
leaf, which the term must supply. The enumeration is verification: CCSDT has 10
`V·t3` operators, **0** in any `t3`-free term; CCSDTQ 10 `V·t4`, **0** in any
`t4`-free term.

**Part 3** (the scientific content) is a refutation-by-witness. A `Tₙ`-bearing
term reuses a *lower*-rank operator when association routes it through a low-rank
intermediate first — e.g. `¼ t2(d,e,i,j) t3(a,b,c,k,l,m) v(d,e,l,m)` (a
`t3`-term) factors `(t2·v)`-first through a `W_t2v_oooo` node. Enumerated: **36**
such reuses in CCSDT triples, **64** in CCSDTQ quadruples — nonzero, so
"reused only in `Tₙ`-free terms" is false. What Part 2 excludes is not a *rank*,
only operators *built from `Tₙ` itself*.

**Part 4** is measured, not deduced. `D`(CCSDT triples) = 35 operators ⊆
`D`(CCSDTQ triples) = 38 (0 CCSDT-only; the 3 extras all `t4`-bearing), and the
`V·t3` operators recur inside CCSDTQ's quadruples. Cumulativity for arbitrary
rank would need an independent proof.

*Gates:* `test_part1_and_2_Vtn_ops_only_in_Tn_terms`,
`test_part3_lower_ops_do_appear_in_Tn_terms`,
`test_ccsdt_operators_reused_in_ccsdtq_triples`,
`test_recursion_summary_is_cumulative`.

---

## The evidence behind the answer

**The FLOP lever is real and grows with rank.** The minimum-peak tree lowers the
exponent on every multi-factor `Tₙ·V` term:

| term | n-ary | best tree |
|---|---|---|
| `t2·t3·v` (CCSDT) | `o⁵v⁵` | `o⁴v³` |
| `t2·t4·v` (CCSDTQ) | `o⁶v⁶` | `o⁵v⁴` |

The step that lowers the exponent is the step that materializes the operator —
one act. All 399 CCSDT-triples and 2672 CCSDTQ-quadruples trees are exact
(`tree_preserves_term`: every factor one leaf, every summed index consumed once;
associativity then gives numeric equality).

**The operators are derived, not seeded.** Over CCSDT triples: 40 distinct
operators, **5 of the 6 CCSD reused** (Wabef's pure-`vvvv` block never arises),
the rest new — the `t3·v` family plus mixed `t1/t2` dressings.

**Value tracks cost, not frequency.** Weighting by
`savings = (uses−1)·build_flops` (scaling-dominated `o^a·v^b`) inverts the naive
ranking: the derived `W_t4v_oooovvvv` (`o⁴v⁶`, 28 uses) tops CCSDTQ at ~4.2e15,
while the best CCSD reuse (Wmbej, `o³v³`, 32 uses) sits ~6 orders below. The
operators worth materializing are the expensive derived ones; the frequent CCSD
reuses are near-worthless to cache.

---

## What is built

The engine (`factorize.py`) is a contraction-path cost model + exhaustive binary
tree search + node-level operator identification, rank-agnostic across CCSDT and
CCSDTQ. Determinism is load-bearing (41% of terms have ties): the selection key
is a total order `(peak.total, peak.n_vir, −max_build_flops, tree_signature)`
which, with sorted derived names, makes the operator set a function of the terms.

**Emit bridge (E0–E1).** `emit_factorized_translation_unit(method, top_k=…,
savings_fraction=…)` produces a Planck C++ TU whose kernels reference the derived
operators, with a `build_W` per kept operator. The factorizer emits only its
*new* `V·Tₙ` operators; CCSD (`Reuse`) children are inlined, since their dressing
(the `tau`/`tau_tilde` builders) is D7.3's job — the two paths are complementary.
The factorized CCSDT TU compiles against the real CC headers. Two boundary bugs
were found and fixed via the compile gate: CCSD-operator factors don't lower
(inline them), and a multi-step operator's `build_W` must declare its *full*
internal summation, not just its top tree step's.

**Savings budget (E1).** `select_operators_by_savings(specs, top_k=|
savings_fraction=)` keeps only the worthwhile operators — the concentration is
extreme (CCSDT top 5 of 24 = 98.8% of savings), so a small budget inlines the
long tail: builder count goes 24 → 6 at `savings_fraction=0.99`, → 5 at
`top_k=5`, all compiling, all algebra-exact. (Fewer builders *raises* line count
— inlined operators expand into the kernels — so lines are not the metric; FLOP
savings are, and they are retained per the budget.)

---

## What is NOT answered here

- **The theorem is model-local** (exhaustive ≤5-factor trees, canonical-Fock
  diagram residuals) and Part 4 is two-rank evidence, not an all-rank proof.
- **Numeric energy equivalence (E2) is not run** — the factorized TU compiles
  (`-fsyntax-only`) but has not been built into a binary and checked against
  FCI / the hand-written solver. E0.1's manifold-wide re-expansion proves the
  algebra is unchanged structurally; the runtime energy match is the open step.
- **Cross-term scheduling is out of scope** — materializing shared operators once
  across the whole manifold (where the largest savings compound) is NP-hard in
  general; `value_operators` is its input, not its solution.

## What this reuses

`_eri_canonical` / `expand_dressed_term` (canonical keys), `seeded_operators()` /
`DressedOperator` (the CCSD fingerprints), `IntermediateSpec` +
`emit_planck_translation_unit` (a derived operator is one, emitted as a
`build_W`), and the diagram engine + `canonical_fock=True` (the residual source;
see the `cc_canonical_fock_only` invariant). See
`CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` (Decision 4) for the recognition/assembly
machinery this builds on.

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`, which are canonical.
