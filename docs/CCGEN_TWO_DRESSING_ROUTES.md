# ccgen's Two Dressing Routes

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Why did ccgen's dressed-operator route fail when CFOUR and MRCC ship one?**

## Short answer

ccgen has two dressing routes, and production was wired to the weaker one. The route that was retired — matching six hand-seeded spin-orbital fingerprints — is genuinely broken and stays retired. The other route, deriving operators from each term's own contraction tree, was built a week later, already recognizes five of the same six operators *on spatial terms*, and turns out to be worth 2-7x. It was never wired to production, and nothing recorded a decision not to.

So the question's premise — that ccgen tried what CFOUR/MRCC do and it did not work — is wrong. ccgen built it, gated it structurally, and left it disconnected.

## Where the logic lives

- `python/ccgen/optimization/factorize.py` — the derivation route: `manifold_operators`, `emittable_operators`, `identify_node`, `node_to_term`; the shape key `_contraction_shape`, `_shape_tag`; the emit bridge `emit_factorized_translation_unit` (no production caller)
- `python/ccgen/optimization/operator_identity.py` — transpose-equivalence and merging
- `python/ccgen/tests/test_factorize_value_preservation.py` — the value gate
- `python/ccgen/dressing.py`, `python/ccgen/dressed_equation.py` — the recognition route (retired): `seeded_operators()`, `operator_to_intermediate_spec`
- `python/ccgen/tests/residual_eval.py` — basis-matched fixtures: `random_tensors` (GCC) / `ucc_closed_shell_tensors(...)[1]` (spatial)
- `docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` — the retirement this document questions
- `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` — operator granularity, UCC carry-over
- `docs/CCGEN_KERNEL_SCALING_SCOPE.md` — the unmodelled cost this document's FLOP figures do not capture
- `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md` — the production wiring of the derivation route

## The two routes

| route | operators from | where | status |
|---|---|---|---|
| **recognition** | 6 hand-seeded spin-orbital fingerprints | `dressing.py`, `dressed_equation.py` | **retired** — 52% short on Be/STO-3G, five failed fix attempts |
| **derivation** | each term's own contraction tree | `factorize.py` | works, value-gated, was not wired to production at the time this was found |

They share no machinery. `dressing.py` has zero references to `derived_operators` or `contraction_tree`; `factorize.py` imports `seeded_operators` only to avoid re-deriving what is already named.

The recognition route was first (`dressing.py`, `dressed_equation.py`, 2026-07-27), and the derivation route (`factorize.py`, 2026-08-04) was deferred by its own author: its commit (`f68f7e2`) says it "derives the dressed operators as the reused sub-contractions those trees expose", reports "5 of 6 CCSD reused", and ships `emit_factorized_translation_unit()` producing a TU that compiles — but the same commit message says "CCSD dressing stays D7.3's job" (D7.3 being the recognition route). `emit_factorized_translation_unit` had no production caller to this day at the time of this investigation. The generator script exposed only `--dress-operators`, which routes to recognition. The derivation route's emit bridge was built and left unconnected on the strength of that one deferral — the same shape as the rank-3 kernel/solver defect, where the kernel was correct and the harness around it was not.

## What invariants matter

### 1. A structural gate is not a substitute for a value gate

The derivation route had no numeric validation at all when this investigation began. `tree_preserves_term` checks leaf and index bookkeeping; `test_budgeted_rewrite_is_exact` compares a factor `Counter`, blind to index order by construction. `tree_preserves_term(doubles[45])` returns `True` on a malformed node, because it asks the question at term level and the defect lived at node level.

Design rule:

- A gate comparing structural properties (leaf sets, factor counts) cannot substitute for a gate comparing the actual numeric value the rewritten term produces. This is the recurring failure across this codebase's dressing work — it is also what let the 52% recognition defect survive five fix attempts that each passed their gate and made the energy worse.

### 2. A fixture must match the basis it is checking

`random_tensors` antisymmetrizes `t2` and `v` — correct for `<pq||rs>`, wrong for spatial, where neither is antisymmetric. Running a spatial check on it lets a result pass for a property of the fixture rather than the code. Audited repo-wide: three other files evaluate spin-adapted terms through `random_tensors`, and all three are safe because they are A-vs-B comparisons of two writings of the same equation set — both sides see identical tensors, so an unphysical fixture cancels.

Design rule:

- Use `ucc_closed_shell_tensors(...)[1]` for spatial checks, not `random_tensors`. The rule generalizes: a fixture must match the basis whenever a check asserts a property *of the tensors themselves* or compares against an independent oracle; it is safe to reuse an unphysical fixture only for an A-vs-B comparison where both sides see identical tensors.

### 3. Do not attempt tag-blind matching to extend an operator scheme to a new spin structure

Recognition finds zero operators on UCC terms, because UCC factors are spin-tagged (`v_abab`, `t2_aaaa`) while the fingerprints key on bare `('t1','v')`. Making the match tag-blind recovers 152 reuse sites and is unsound: it collapses 12 distinct spin-tagged contractions onto one `Wmbej`, and `t1_aa·v_aaaa` and `t1_aa·v_abab` are different arrays under UHF. Keyed properly the five names become 30 operator instances.

Design rule:

- Do not attempt tag-blind matching to bridge a fingerprint-based scheme across a new spin structure. It is measured, it looks like it works, and it is the same 52% defect one level down. Scoped as O6 in `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`.

## What was found

1. **The retirement's stated mechanism does not hold, though its decision does.** `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` argues a spatial `Wmbej` "is not a relabeled GCC `Wmbej`; it is several different operators", and concludes "deriving it is research, not porting". The antisymmetry claim is right — spatial has 13 `v` index-space patterns against 9 in GCC — but the conclusion does not follow: deduped, the two bases give 18 and 19 operators, and the derivation never needed the antisymmetry. Measured via `emittable_operators`:

   | basis | reuse sites | derived sites | seeded operators recognized |
   |---|---|---|---|
   | GCC | 36 | 54 | Fme, Fmi, Fae, Wmnij, Wmbej |
   | **spatial** | **61** | 87 | the same five |
   | UCC | **0** | 370 | **none** — see invariant 3 |

   The retirement's decision still stands — both routes failed value preservation, for different reasons, and the derivation route failed on GCC where there is no spin adaptation to blame — but its stated reason does not.

2. **Two mechanisms made the derivation route unusable before this investigation, both now fixed.** Run with the gate's fixture at `savings_fraction=1.0`, 23 of 66 GCC `ccsd` doubles terms did not reproduce their source (‖diff‖/‖R‖ = 3.73e-01). The GCC number is the important one — a value check failing on the known-good basis is measuring the probe, not the route.

   - **Incomplete summed lists.** `node_to_term` recorded only `node.summed` — the indices consumed at *that* tree step — while its factors are the whole subtree's leaves, so inner contraction indices were bound to nothing. Example, `doubles[45]`:

     ```
     parent term : t1(a,k) t1(b,l) t1(c,j) v(i,c,k,l)     free i,j,a,b   summed k,l,c
     child node  : t1(c,j) t1(a,k) v(i,c,k,l)             free i,j,l,a   summed c
                                 ^^^         ^^^
                                 k appears TWICE and is in NEITHER list
     ```

     The emitted `build_W` had no loop over `k`. Fixed by completing the summation (`used − free`) at `node_to_term`, the single upstream source for `identify_node` / `node_key` / `block_signature` / `_derived_name`. Measured 20/52 → 0/50 malformed specs.

   - **One name, several contractions.** `_derived_name` built names from sorted factor names plus a block signature, discarding slot order. `W_t2v_ooov(i,j,k,a)` denoted both `t2(a,c,j,l) v(i,c,k,l)` and `t2(c,d,i,j) v(c,d,k,a)` — different contractions, one `build_W`. Fixed by folding the contraction shape into the name. Three properties of that shape key were each isolated by a failing case:

     | property | without it |
     |---|---|
     | slot **position**, not merely free/summed | 21 → 13 disagreements |
     | positions, not index **names** | 13 → 6 |
     | same-tensor copies kept distinct | 6 → **0** |

## What was measured

Operator sharing — materialize once against rebuild at every reference site, with terms already tree-factored so this isolates dressing from ordinary binary factorization:

| manifold | before merging | after merging | retirement's estimate |
|---|---|---|---|
| GCC `ccsd` doubles | 1.97x | **2.58x** | 1.20-1.50x "actual" |
| spatial `ccsd` doubles | 1.21x | **2.38x** | 1.9-2.8x "expected" |
| `ccsdt` doubles | 2.00x | **2.04x** | — |
| `ccsdt` triples | 3.26x | **7.11x** | — |

The payoff grows with rank. The retirement measured only `ccsd`, observed the saving shrinks as `n_vir/n_occ` grows, and concluded it "pays least in the production regime". By rank it pays *most* there, and the production target is rank 3+.

The merging referred to above is a separate finding with its own answer (`docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`): the shape key that fixed correctness also over-split operators that are one contraction up to a transpose. Merging them back is exact, costs nothing at the call site (operators are read by index inside the loop nest; no `W` is ever copied), and roughly doubles the spatial payoff.

Every figure here is a FLOP model (`operator_savings` / `build_cost`). It does not price the emitter's contraction order, which `docs/CCGEN_KERNEL_SCALING_SCOPE.md` measured as a scaling defect (21.8x → 50.1x, no plateau) that no current cost model predicts. These are ratios between comparable configurations, not predicted wall-clock.

## Validation strategy that should remain in place

- `test_factorize_value_preservation` — the instrument the route never had:

  | | terms rewritten | disagreements |
  |---|---|---|
  | GCC `ccsd` singles+doubles | 34 | **0** |
  | spatial `ccsd` singles+doubles | 25 | **0** |
  | `ccsdt` triples | 345 | **0** |
  | `ccsdtq` quadruples | 2536 | **0** |

  Both `canonical_fock` settings, each basis on its matching fixture. Rank 4 is asserted separately because this codebase has twice shown rank 3 does not predict it.
- The seven `expectedFailure` gates on the retired recognition route, kept as a tripwire — an unexpected pass would mean someone fixed the composition.

## Related but separate outcome: production wiring

**Wire the derivation route** was the recommendation from this investigation. It is value-gated at ranks 2-4, it merges end to end into the emitted C++ (27 → 19 builders on `ccsd`, 254 → 69 at rank 4), and it is worth 2-7x. It is already the route `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` builds on.

This has since happened: `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md` — originally scope, now itself an answer — records that the route was wired, went red on its first end-to-end energy comparison, and the cause was an invalid ERI symmetry table on the lowering path (the same class of mistake as the 52% recognition defect recorded above). Fixed; CH4 and LiH both match the undressed baseline. The gap turned out to be narrower than it sounded and wider than one flag: there were two emitters sharing exactly one parameter, and production called the other one. W1-W2 landed first (the pipeline takes equations rather than generating them, byte-identical over six configurations; the spatial TU compiles against the real CC headers both merged and un-merged — 59 → 31 builders), leaving W3 as a project-direction decision rather than a technical step.

UCC is a separate matter: the factorizer handles UCC input fine (86 merged operators on `doubles_abab`), but the emitter rejects the spin-blocked manifold names, and the dressed-operator story there is unsound until O6 (see invariant 3 above).

Leave recognition retired. Nothing in this work touched it. It remains 52% short on Be with five failed fix attempts behind it. Do not read this document as un-retiring dressing — what is shown is that the *derivation* route works; what is not shown is that the *recognition* route is fixable.

## Remaining architecture concern

- **How CFOUR and MRCC actually factorize their intermediates** is the original question and it is not answered here — it needs literature the repo does not contain. What the repo does carry is a citation about *enumeration*, not factorization: MRCC/CFOUR enumerate diagram topologies canonical by construction rather than deduplicating algebraic terms (Kállay & Surján, JCP 113, 1359 (2000); JCP 115, 2945 (2001), cited in `ccgen/diagram.py`). Whether their intermediates are hand-derived per method or fall out of contraction-order optimization is the open half. It matters less than it did — the answer above stands on ccgen's own measurements, and the practical conclusion (wire the route that works) does not depend on what the other codes do. If someone does establish it, the useful form is a per-code classification: fixed-operator-recognition or structure-derived. If both are structure-derived, that is independent evidence ccgen wired the wrong one of its own two routes.
- **Whether the operators are correct for UCC** — see invariant 3 above; scoped as O6 in `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`.
- **Six selection-model gates need re-deriving.** They assert properties of the savings distribution, which the shape key changed. They are not correctness failures — the value gate is 0/0 throughout — but three of the four distinct kinds need their claim restated rather than their constant loosened. Detail in `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` (O4.6).
