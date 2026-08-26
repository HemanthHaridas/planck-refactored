# Why did ccgen's dressed-operator route fail when CFOUR and MRCC ship one?

**Answer: ccgen has two dressing routes, and production was wired to the weaker one.**

The route that was retired — matching six hand-seeded spin-orbital fingerprints — is genuinely
broken and stays retired. The other route, deriving operators from each term's own contraction
tree, was built a week later, already recognizes five of the same six operators *on spatial
terms*, and turns out to be worth 2-7x. It was never wired to production, and nothing recorded a
decision not to.

So the question's premise — that ccgen tried what CFOUR/MRCC do and it did not work — is wrong.
ccgen built it, gated it structurally, and left it disconnected.

What follows is the evidence, the defect that made the derivation route unusable until it was
fixed, and what is genuinely still unknown.

---

## The two routes

| route | operators from | where | status |
|---|---|---|---|
| **recognition** | 6 hand-seeded spin-orbital fingerprints | `dressing.py`, `dressed_equation.py` | **retired** — 52 % short on Be/STO-3G, five failed fix attempts |
| **derivation** | each term's own contraction tree | `factorize.py` | works, value-gated, **not wired to production** |

They share no machinery. `dressing.py` has zero references to `derived_operators` or
`contraction_tree`; `factorize.py` imports `seeded_operators` only to avoid re-deriving what is
already named.

### The recognition route was first, and the derivation route was deferred by its own author

| file | created |
|---|---|
| `dressing.py`, `dressed_equation.py` | 2026-07-27 |
| `factorize.py` | 2026-08-04 |

`factorize.py`'s commit (`f68f7e2`) says it "derives the dressed operators as the reused
sub-contractions those trees expose", reports "5 of 6 CCSD reused", and ships
`emit_factorized_translation_unit()` producing a TU that compiles. In the same message:
**"CCSD dressing stays D7.3's job"** — D7.3 being the recognition route.

`emit_factorized_translation_unit` has **no production caller** to this day. The generator script
exposes only `--dress-operators`, which routes to recognition. The derivation route's emit bridge
was built and left unconnected on the strength of that one deferral.

This is the same shape as the rank-3 defect, where the kernel was correct and the harness around
it was not.

## What the derivation route actually does

Measured on the current tree via `emittable_operators`:

| basis | reuse sites | derived sites | seeded operators recognized |
|---|---|---|---|
| GCC | 36 | 54 | Fme, Fmi, Fae, Wmnij, Wmbej |
| **spatial** | **61** | 87 | the same five |
| UCC | **0** | 370 | **none** — see the UCC section |

It recognizes five of the six Stanton-Gauss operators **on spatial terms**, deriving them from
contraction topology, so it never needed a spatial fingerprint set.

That contradicts the retirement's stated mechanism. `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`
argues a spatial `Wmbej` "is not a relabeled GCC `Wmbej`; it is several different operators", and
concludes "deriving it is research, not porting". The antisymmetry claim is right — spatial has
13 `v` index-space patterns against 9 in GCC — but the conclusion does not follow: deduped, the
two bases give 18 and 19 operators, and the derivation never needed the antisymmetry.

**The retirement's decision still stands. Its stated reason does not.** Both routes failed value
preservation, for different reasons, and the derivation route failed on GCC where there is no
spin adaptation to blame.

## The defect that made the derivation route unusable

Run with the gate's fixture at `savings_fraction=1.0`, 23 of 66 GCC `ccsd` doubles terms did not
reproduce their source (‖diff‖/‖R‖ = 3.73e-01). **The GCC number is the important one** — a value
check failing on the known-good basis is measuring the probe, not the route.

Two mechanisms, both now fixed.

**Incomplete summed lists.** `node_to_term` recorded only `node.summed` — the indices consumed at
*that* tree step — while its factors are the whole subtree's leaves, so inner contraction indices
were bound to nothing. `doubles[45]`:

```
parent term : t1(a,k) t1(b,l) t1(c,j) v(i,c,k,l)     free i,j,a,b   summed k,l,c
child node  : t1(c,j) t1(a,k) v(i,c,k,l)             free i,j,l,a   summed c
                            ^^^         ^^^
                            k appears TWICE and is in NEITHER list
```

The emitted `build_W` had no loop over `k`. Fixed by completing the summation (`used − free`) at
`node_to_term`, the single upstream source for `identify_node` / `node_key` / `block_signature` /
`_derived_name`. Measured 20/52 → 0/50 malformed specs.

**One name, several contractions.** `_derived_name` built names from sorted factor names plus a
block signature, discarding slot order. `W_t2v_ooov(i,j,k,a)` denoted both `t2(a,c,j,l) v(i,c,k,l)`
and `t2(c,d,i,j) v(c,d,k,a)` — different contractions, one `build_W`. Fixed by folding the
contraction shape into the name. Three properties of that shape key were each isolated by a
failing case:

| property | without it |
|---|---|
| slot **position**, not merely free/summed | 21 → 13 disagreements |
| positions, not index **names** | 13 → 6 |
| same-tensor copies kept distinct | 6 → **0** |

### Why it was never caught

The route had **no numeric validation at all**. `tree_preserves_term` checks leaf and index
bookkeeping; `test_budgeted_rewrite_is_exact` compares a factor `Counter`, blind to index order
by construction. `tree_preserves_term(doubles[45])` returns `True` on a malformed node, because
it asks the question at term level and the defect lives at node level.

That is the recurring failure in this work: **a structural gate standing in for a value gate**,
which is also what let the 52 % recognition defect survive five fix attempts that each passed
their gate and made the energy worse.

## What it is worth

Operator sharing — materialize once against rebuild at every reference site, with terms already
tree-factored so this isolates dressing from ordinary binary factorization:

| manifold | before merging | **after** | retirement's estimate |
|---|---|---|---|
| GCC `ccsd` doubles | 1.97x | **2.58x** | 1.20-1.50x "actual" |
| spatial `ccsd` doubles | 1.21x | **2.38x** | 1.9-2.8x "expected" |
| `ccsdt` doubles | 2.00x | **2.04x** | — |
| `ccsdt` triples | 3.26x | **7.11x** | — |

**The payoff grows with rank.** The retirement measured only `ccsd`, observed the saving shrinks
as `n_vir/n_occ` grows, and concluded it "pays least in the production regime". By rank it pays
*most* there, and the production target is rank 3+.

The merging referred to above is a separate finding with its own answer
(`CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`): the shape key that fixed correctness also over-split
operators that are one contraction up to a transpose. Merging them back is exact, costs nothing
at the call site (operators are read by index inside the loop nest; no `W` is ever copied), and
roughly doubles the spatial payoff.

**Every figure here is a FLOP model** (`operator_savings` / `build_cost`). It does not price the
emitter's contraction order, which `CCGEN_KERNEL_SCALING_SCOPE.md` measured as a scaling defect
(21.8x → 50.1x, no plateau) that no current cost model predicts. These are ratios between
comparable configurations, not predicted wall-clock.

## The route is now gated

`test_factorize_value_preservation` — the instrument the route never had:

| | terms rewritten | disagreements |
|---|---|---|
| GCC `ccsd` singles+doubles | 34 | **0** |
| spatial `ccsd` singles+doubles | 25 | **0** |
| `ccsdt` triples | 345 | **0** |
| `ccsdtq` quadruples | 2536 | **0** |

Both `canonical_fock` settings, each basis on its matching fixture. Rank 4 is asserted separately
because this codebase has twice shown rank 3 does not predict it.

**Fixture trap, found while gating this.** `random_tensors` antisymmetrizes `t2` and `v` — correct
for `<pq||rs>`, wrong for spatial, where neither is antisymmetric. Running a spatial check on it
lets a result pass for a property of the fixture. Use `ucc_closed_shell_tensors(...)[1]` for
spatial. Audited repo-wide: three other files evaluate spin-adapted terms through
`random_tensors`, and all three are safe because they are A-vs-B comparisons of two writings of
the same equation set — both sides see identical tensors, so an unphysical fixture cancels. The
rule is that a fixture must match the basis when a check asserts a property *of the tensors* or
compares against an independent oracle.

## What this means for production

**Wire the derivation route.** It is value-gated at ranks 2-4, it merges end to end into the
emitted C++ (27 → 19 builders on `ccsd`, 254 → 69 at rank 4), and it is worth 2-7x. It is already
the route `CCGEN_HIGHER_OPERATOR_REUSE.md` builds on. This is a wiring task, scoped in
**`docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`** (W1-W5).

The gap is narrower than it sounds and wider than one flag: there are **two emitters** sharing
exactly one parameter, and `emit_factorized_translation_unit` generates its own equations
internally, so it cannot be handed a spin-adapted manifold. The machinery underneath has no such
limit — it produces 31 merged operators on spatial input and 86 on UCC — so the fix is a
signature change, not new algebra.

**Leave recognition retired.** Nothing in this work touched it. It remains 52 % short on Be with
five failed fix attempts behind it, and its seven `expectedFailure` gates stay as the tripwire —
an unexpected pass still means someone fixed the composition.

Do not read this document as un-retiring dressing. What is shown is that the *derivation* route
works; what is not shown is that the *recognition* route is fixable.

## What is still unknown

**How CFOUR and MRCC actually factorize their intermediates.** This is the original question and
it is **not answered here** — it needs literature the repo does not contain. What the repo does
carry is a citation about *enumeration*, not factorization: MRCC/CFOUR enumerate diagram
topologies canonical by construction rather than deduplicating algebraic terms (Kállay & Surján,
JCP 113, 1359 (2000); JCP 115, 2945 (2001), cited in `ccgen/diagram.py`). Whether their
intermediates are hand-derived per method or fall out of contraction-order optimization is the
open half.

It matters less than it did. The answer above stands on ccgen's own measurements, and the
practical conclusion — wire the route that works — does not depend on what the other codes do.
If someone does establish it, the useful form is a per-code classification:
**fixed-operator-recognition** or **structure-derived**. If both are structure-derived, that is
independent evidence ccgen wired the wrong one of its own two routes.

**Whether the operators are correct for UCC.** Recognition finds **zero** operators on UCC terms,
because UCC factors are spin-tagged (`v_abab`, `t2_aaaa`) while the fingerprints key on bare
`('t1','v')`. Making the match tag-blind recovers 152 reuse sites and is **unsound**: it collapses
12 distinct spin-tagged contractions onto one `Wmbej`, and `t1_aa·v_aaaa` and `t1_aa·v_abab` are
different arrays under UHF. Keyed properly the five names become 30 operator instances. Scoped as
O6 in `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`. **Do not attempt tag-blind matching** — it is
measured, it looks like it works, and it is the 52 % defect one level down.

**Six selection-model gates need re-deriving.** They assert properties of the savings
distribution, which the shape key changed. They are not correctness failures — the value gate is
0/0 throughout — but three of the four distinct kinds need their *claim* restated rather than
their constant loosened. Detail in `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` (O4.6).

## Key code locations

| what | where |
|---|---|
| derivation route | `manifold_operators`, `emittable_operators`, `identify_node`, `python/ccgen/optimization/factorize.py` |
| its emit bridge (no production caller) | `emit_factorized_translation_unit`, same file |
| the shape key | `_contraction_shape`, `_shape_tag`, same file |
| transpose-equivalence and merging | `python/ccgen/optimization/operator_identity.py` |
| **the value gate** | `python/ccgen/tests/test_factorize_value_preservation.py` |
| recognition route (retired) | `seeded_operators()`, `dressing.py`; `dressed_equation.py` |
| the bridge between the routes | `operator_to_intermediate_spec` (`dressing.py`), `seeded_fingerprints` (`factorize.py`) |
| basis-matched fixtures | `random_tensors` (GCC) / `ucc_closed_shell_tensors(...)[1]` (spatial), `residual_eval.py` |
| the retirement this questions | `docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` |
| operator granularity, UCC carry-over | `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` |
| unmodelled cost | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
