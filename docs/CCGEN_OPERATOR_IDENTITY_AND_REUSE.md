# ccgen Derived Operator Identity and Merging

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**When are two derived contractions the SAME operator?**

Coupled to `docs/CCGEN_TWO_DRESSING_ROUTES.md`, which owns the value-preservation question
(D0–D8) and whose D6 fix created this one.

## Short answer

The factorizer must answer one question at every tree node: *have I built this operator already?*
Answer too coarsely and it emits one `build_W` for two different contractions — the D6 defect,
23/66 GCC terms wrong. Answer too finely and it stores the same array several times under
different names, which is correct but destroys the reuse the factorizer exists to create. D6
moved the answer from far-too-coarse to measurably too fine. **O1–O5 are complete**: two
contractions are the same operator exactly when they are transpose-equivalent under a
sign-preserving permutation, decided symbolically (not by comparing arrays), and this now merges
end to end in the emitted C++ (27→19 builders on `ccsd`, 254→69 at rank 4), value-gated at 0/2536
disagreements on quadruples. The merge ratio **grows with rank** (1.4x → 2.1x → 3.7x). **O6 (UCC
carry-over) is measured and open**: naive tag-blind matching is unsound.

## Where the logic lives

- `python/ccgen/optimization/factorize.py` — `_contraction_shape`, `_shape_tag` (the shape key D6
  introduced), `_derived_name`/`identify_node` (where names are assigned), `manifold_operators`
  (`rep = specs[0]`, where operators are deduped), `operator_savings`/`build_cost`/`OperatorValue`
  (the savings model), `select_operators_by_savings` (threshold that drops split operators)
- `python/ccgen/optimization/operator_identity.py` — `tensor_symmetries`, `symbolic_transpose`
  (the equivalence predicate, O1/O2), `canonical_shape` (O4), `merge_plan` (O4.1)
- `python/ccgen/tests/test_operator_identity.py` — its gate
- `python/ccgen/tests/residual_eval.py` — basis-matched fixtures (`random_tensors` for GCC,
  `ucc_closed_shell_tensors(...)[1]` for spatial)
- `python/ccgen/tests/test_factorize_value_preservation.py` — **the referee**
- `docs/CCGEN_TWO_DRESSING_ROUTES.md` — the coupled scope (D6 landed; D8 blocked on O3, now
  unblocked)
- `docs/CCGEN_KERNEL_SCALING_SCOPE.md` — cost-model caveat

## What invariants matter

### 1. Transpose-equivalence is decided symbolically, and only within sign-preserving symmetries

`symbolic_transpose(sp1, sp2, spatial=)` decides equivalence on the **shape key**, not by
comparing arrays. Held against a numeric oracle (materialize both, try every axis permutation) at
two fixtures (`no=3,nv=4` and `no=4,nv=3`, inverting the asymmetry) × three seeds:

| | pairs | agree | false merges | misses |
|---|---|---|---|---|
| **GCC** (`canonical_fock` True and False) | 23 | **23** | **0** | **0** |
| **SPATIAL** (both) | 229 / 230 | **229 / 230** | **0** | **0** |

Three sources of freedom are required, each found by a failing case rather than by design:

1. **slot permutation** — the transpose itself.
2. **summed-label permutation** — `_contraction_shape` numbers internal indices canonically
   *per shape*, so the numbering does not align across two shapes. Without this,
   `W_t1t1v_ooov_b656`/`_719d` (identical slots, `v`'s summed labels swapped) is missed.
   GCC went 20/23 → 23/23.
3. **the ERI's own SIGN-PRESERVING symmetries** — see below.

**The sign restriction is the load-bearing part.** Folding all of `_ERI_PERMUTATIONS` produced
**two false merges on GCC** (`W_t2v_vv`, `W_t1t1v_oo`): the predicate claimed equivalence for
arrays differing by a **sign**. Four of the eight members are odd and hold only up to
`<qp|rs> = -<pq|rs>`. This is the *same blind spot* that let the 52% dressed-energy defect pass
every symbolic check — already documented on `_ERI_PERMUTATIONS_SPATIAL` in `dressing.py`, and
reached independently here by a probe that had it wrong first. Restricting to the even members
gives 8 merges instead of 10; `SignPreservationTests` pins the reason so a future edit reaching
for the full orbit fails with the cause named.

Design rule:

- Only fold **sign-preserving** symmetries into an equivalence predicate for physical tensors.
  Using all 8 ERI permutations produces false merges — the same blind spot as the 52% dressing
  defect. Verified falsifiable: removing the parity filter makes the predicate find 10 merges and
  turns the suite red (3 failures).
- Exactness is only meaningful with a discrimination check: a predicate that merged *everything*
  would also report zero misses. `PredicateStillDiscriminatesTests` pins that merges are a strict
  subset of same-family pairs and that cross-family merges are 0.

### 2. An oracle's fixture must match the basis it's checking

Spatial reached exactness in three stages, and the middle one is the transferable lesson: **the
first thing this found was a defect in its own oracle, not in the code under test.**

`random_tensors` builds **antisymmetrized** `t2` and `v` — the GCC fixture. Spatial `t2` and
`<pq|rs>` have no such antisymmetry, so on spatial terms the oracle was reporting pairs equal when
they were equal only *for that fixture*. Re-run on `ucc_closed_shell_tensors`'s `spatial` bundle
(`t2[abij] == t2[baji]`, `<pq|rs> == <qp|sr>`, neither antisymmetric):

| spatial oracle | pairs | agree | false merges | misses |
|---|---|---|---|---|
| old (spin-orbital fixture) | 229 / 230 | 181 | 0 | 48 / 49 |
| **new (spatial fixture)** | 229 / 230 | **211** | **0** | **18 / 19** |

**The error was in the ORACLE, not the predicate.** ~30 of the 48 "misses" were oracle false
positives. O1's soundness claim survives unchanged — still zero false merges — and its
completeness on spatial was *understated*.

An audit across the repo found `random_tensors` used by nine files; three besides the two fixed
here evaluate spin-adapted terms through it (`test_dress_per_operator`, `test_emit_flag_matrix`,
`test_residual_symmetry`). **None is compromised**, because those three are **A-vs-B comparisons
of two writings of the same equation set**, not checks of a physical property — both sides see
identical tensors, so an unphysical fixture cancels.

Design rule:

- A fixture must match the basis when a check asserts a property *of the tensors*, or compares a
  value against an independent oracle. It may be unphysical when both sides of a comparison
  contract the same arrays (an A-vs-B rewrite comparison). Know which kind of gate you are writing.

### 3. Per-tensor symmetry tables do not generalize between tensors

`v_variants`' hardcoded `v` case became `tensor_symmetries(spatial) -> {tensor name:
sign-preserving permutations}`, with `t2` added:

```
"v":  parity-+1 members of the ERI symmetry group   (from dressing.py, not re-derived)
"t2": ((0,1,2,3), (1,0,3,2))                        # t2[abij] = t2[baji]
```

**`v`'s pattern does NOT transfer to `t2`.** Spatial `t2` is symmetric under the *simultaneous*
pair swap only; the single-pair swaps are not symmetries of it at all (they are antisymmetries of
the spin-orbital `t2`). Adding one produces false merges — verified, not assumed. A factor whose
name is absent from the table is treated as opaque, which is the safe default: an unmodelled
symmetry costs a missed merge, never a false one.

Design rule:

- Never assume one tensor's symmetry group applies to another. Add each tensor's symmetries to
  `tensor_symmetries` individually and verify both directions: removing an entry should reopen
  known misses, and adding an unsound entry should create false merges. (Checked here: removing
  the `t2` entry reopens 2 test failures; adding the unsound single-pair swap creates 2 failures.)

### 4. A merge must permute call sites, not just merge names

Naming operators by a canonical key alone does not work, and this was measured, not predicted.
Folding `canonical_shape` into `_shape_tag` merges the names and produces the intended counts and
histograms, but also **reintroduces the D6 defect**: 11 GCC doubles terms stop reproducing their
source. The cause: `rewrite_term_factorized` builds each call site using **the site's own**
canonical index order, with no knowledge of which slot order the shared operator is stored in.
Merge the names without permuting the sites and the array is read as if it were the other member
of its class.

The fix pairs a canonical merge plan (`merge_plan(specs, spatial)` returning
`{name: (representative_name, permutation)}`) with call-site permutation
(`rewrite_term_factorized(..., merge_plan_map=plan)`), and hands both back together from
`manifold_operators_with_plan` so they cannot be separated again — emitting merged specs while
call sites read in their own slot order is exactly the reverted first attempt.

Design rule:

- Never merge operator names without also permuting every call site that reads the merged array.
  Emit the name and the permutation plan as one paired return value so a caller cannot apply one
  without the other.
- The call-site transpose costs nothing at runtime: every operator is read by index inside the
  loop nest (verified: zero `W` arrays are ever copied or transposed into a temporary in the
  emitted C++), so `operator_savings` pricing the transpose at zero is correct, not optimistic.

### 5. Two defects were invisible to every algebra-level gate and found only by reading emitted text

Wiring the merge into the actual emitter exposed two bugs that every symbolic/value gate had
missed, because both leave the symbolic rewrite evaluable:

1. **The budget check rejected every merged call site.** `keep_operators` holds *representative*
   names after merging, but the hoist test asked `name in keep_operators` using the *member's*
   name. Every merged site failed the check and was inlined, silently undoing the merge. Fixed by
   testing the representative.
2. **Call sites referenced builders that do not exist.** With the permutation applied but the name
   left alone, a site emitted `W_member(...)` while only `build_W_rep` was generated — a dangling
   reference in the generated source. Fixed by emitting the representative's name too.

Design rule:

- A symbolic/algebra-level gate cannot substitute for reading the actual emitted output. Add a
  gate that inspects the generated C++ text directly (builder count, no merged-away name
  surviving, each builder defined exactly once) whenever a transform changes what gets emitted.

### 6. Rank 3 does not predict rank 4 for this mechanism either

The merge holds at every rank and **improves** with rank:

| manifold | operators | non-identity perms | savings | disagreements |
|---|---|---|---|---|
| `ccsd` doubles | 27 → **19** (1.4x) | 8 | 1.02x | **0 / 45** |
| `ccsdt` triples | 80 → **39** (2.1x) | 41 | 1.03x | **0 / 345** |
| `ccsdtq` quadruples | 254 → **69** (3.7x) | 184 | 1.20x | **0 / 2536** |

The merge ratio strengthens with rank — 1.4x → 2.1x → 3.7x — because higher-rank manifolds
generate far more transpose-equivalent operators, so the over-splitting D6 introduced was worst
exactly where the factorizer matters most.

Design rule:

- Assert rank-4 behaviour separately from rank 3; do not assume a rank-3 result transfers. This
  codebase has twice shown rank 3 does not predict rank 4 (the tensor-accessor fix left rank 4
  unchanged while giving rank 3 a 206x speedup; the rank-3 solver defect did not generalize).

## What was found

**The over-splitting cost, and how much of it was real.** D6 folded the contraction shape into
the operator name (`_contraction_shape` + a 2-byte digest, `W_t2v_ooov_a049`), which was necessary
— the value gate `test_factorize_value_preservation` goes 21/41 → 0/41 (GCC) and 39/63 → 0/63
(spatial) because of it. The cost, GCC `ccsd` doubles at `canonical_fock=True`:

| | before D6 | after D6 |
|---|---|---|
| distinct operators | 12 | **27** |
| terms actually rewritten | 39 | **30** |
| operators with `usage_count > 1` | 12 / 12 | **17 / 27** |
| usage histogram | `[11,6,6,6,5,4,2,2,…]` | `[6,2,2,2,2,2,2,2,…]` |
| modeled total savings (`operator_savings`, 30/100) | 4.96e+12 | 7.48e+10 |

That last row is **not** a like-for-like comparison: the 4.96e+12 baseline was computed with
`nary_cost`, which overstates multi-factor operators (500x on one measured case, dominated here by
`W_t1t2v_ooov` alone contributing 4.05e+12 of the 4.96e+12). Priced identically on both sides
(`nary_cost` throughout), the split costs `1.00x -> 0.51x`, and merging recovers it to `0.85x`.
Under the corrected `build_cost` pricing — the honest number to carry forward:

| | GCC | SPATIAL |
|---|---|---|
| D6 split | 7.48e+10 | 6.81e+10 |
| **+ merged** | **9.28e+10** (1.24x) | **2.24e+11** (3.29x) |
| operators | 27 -> 19 | 59 -> 31 |

Spatial gains far more, which is the case that matters for the dressed route.

**Grouping the split operators by pre-D6 base name and asking which pairs are the same
contraction up to a permutation of slots:** 8/23 same-family pairs on GCC, 38/229 on spatial.
Concretely, two `W_t2v_ooov` variants with identical slots `(i,j,k,a)`:

```
t2(a,c,j,l) v(i,c,k,l)
t2(a,c,i,l) v(j,c,k,l)     <- the same contraction with i <-> j
```

D6 gives these different names because their slot *positions* differ; one stored array serves both
if the call site reads it transposed `(1,0,2,3)`.

**Symbolic merging by union-find recovers about half the split**, on the GCC `ccsd` doubles
manifold:

| | operators | usage histogram |
|---|---|---|
| after D6 | 27 | `[6,2,2,2,2,2,2,2]` |
| **+ symbolic merge** | **19** | `[6,4,4,4,4,3,2,2]` |

**The over-splitting objection to the dressing-route decision (D8) is resolved.** The savings
figure D8 should use is the merged, `build_cost`-priced one — not 7.48e+10 (un-merged), and not
4.96e+12 (mis-priced). D8 is unblocked. One caveat to carry: all of the above is
`operator_savings`, a FLOP model. It does not price the emitter's actual contraction order, which
`docs/CCGEN_KERNEL_SCALING_SCOPE.md` measured as a scaling defect no current cost model predicts.
Treat these as ratios between comparable configurations, not as absolute performance.

**Which of the six pre-existing failing `test_factorize` selection gates merging fixes: one
partly, none fully.**

| gate | un-merged | merged | verdict |
|---|---|---|---|
| `savings_concentration` (wants >0.98) | 0.6535 | **0.8599** | much better, still fails |
| `ccsdt_keys_barely_diverge` (wants <0.01) | 0.2097 | 0.1845 | barely moves — **premise genuinely falsified** |
| `joint_beats_flops_only_baseline` | exact tie | **still exact tie** | unrelated to the split |
| `optimized_beats_baseline_all_axes` | exact tie | still exact tie | same |

`savings_concentration` and `ccsdt_keys_barely_diverge` encode measured findings about the
operator distribution that the split changed and that must be re-derived, not re-pinned. The two
exact ties are unrelated to the split: both selection strategies return identical totals at 850 GB
regardless, because the budget is large enough to make the comparison vacuous — fix the budget,
not the threshold. Separately, these six failures were once reported as pre-existing; they are
not — a clean worktree at `8e4bb0c` runs `test_factorize` 72/72 green. The "pre-existing" reading
came from a baseline whose `unittest discover` had a bad `-t` root, so `test_factorize` never ran.
They are genuine regressions from the D6 split, in the selection model only; the value gate is 0/0
throughout.

## What was built

- **The predicate** (`symbolic_transpose`, `tensor_symmetries`) — O1/O2, exact on both GCC and
  spatial bases as detailed in Invariants 1–3.
- **`canonical_shape(spec, spatial)`** — the orbit representative of a shape under slot
  permutation, summed relabeling and the factors' own symmetries, paired with the slot-space
  pattern (two shapes can canonicalize alike while their slots differ occ/vir, and one array
  cannot serve both). Partitions operators **exactly** as the pairwise predicate does, at one
  canonicalization per operator instead of one comparison per pair: GCC 27 → 19 classes, spatial
  59 → 31. Gated by `CanonicalShapeTests`.
- **`merge_plan(specs, spatial)`** — returns `{name: (representative_name, permutation)}` for
  every input spec (a representative maps to itself under the identity). The representative is
  the lexicographically smallest name in the class, making the plan deterministic and independent
  of input order. Verified inert by default: operator sets across all seven manifolds (212
  operators) are byte-identical to the pre-merge snapshot when the merge is not requested. Gated
  by `MergePlanTests` (totality, partition match, self-mapped representatives, determinism, and
  that the permutations are not all the identity — 8 GCC / 19 spatial).
- **Call-site permutation** (`rewrite_term_factorized(..., merge_plan_map=plan)`) — orders each
  hoisted child's indices into its class representative's slot order, proven value-preserving:
  0/30 disagreements on GCC doubles, 0/21 on spatial doubles once permutation is applied. Gated by
  `PermutedCallSiteTests`.
- **`manifold_operators_with_plan(terms, spatial=)`** — returns merged specs and the call-site plan
  as one paired return value, implemented as an opt-in `merge_transposes` flag (not by folding
  `canonical_shape` into `_shape_tag` everywhere, which would reach callers whose call sites do not
  permute). Default path verified byte-identical across all seven manifolds. Gated by
  `MergedOperatorsTests`.
- **The merged-path invariant**: `test_one_emitted_name_one_canonical_shape_when_merged` — one
  name per *canonical* shape (an emitted name covering several *raw* shapes is the intended
  merge behaviour; covering more than one *canonical* shape is the D6 defect returning). The
  original un-merged invariant (`test_one_name_one_contraction_shape`) is kept unchanged since it
  still describes the default path.
- **Emitted-C++ sharing** — the merge reaches the generated builders: 27 → 19 builders on `ccsd`,
  264 → 76 on CCSDTQ. Gated by `test_merged_emit_shares_builders_and_permutes_reads`.
- **Every step is falsifiable by construction.** Sabotage tests confirm: merging without
  permuting fails 5 tests; merging into the wrong representative fails 2; renaming without
  permuting fails 5; permuting without renaming fails 3.

## Validation strategy that should remain in place

- `python/ccgen/tests/test_factorize_value_preservation.py` — the referee for every candidate key
  change, run on GCC *and* spatial, at `canonical_fock` True *and* False
- `python/ccgen/tests/test_operator_identity.py` — `SignPreservationTests`,
  `PredicateStillDiscriminatesTests`, `CanonicalShapeTests`, `MergePlanTests`,
  `PermutedCallSiteTests`, `MergedOperatorsTests`, `HigherRankMergeTests`
- `test_merged_emit_shares_builders_and_permutes_reads` — checks the actual emitted C++ text, not
  just the symbolic rewrite
- Never fix a `test_factorize` selection-model failure by re-pinning a constant without first
  deciding whether the underlying claim needs restating (a genuine distributional finding) or the
  test fixture needs fixing (a vacuous budget)

## What NOT to do

- **Do not loosen the equivalence key without the value gate.** The whole reason this question
  exists is that the coarse pre-D6 key was wrong for 23/66 terms and no gate caught it for the
  factorizer's entire life. Every candidate key must run against
  `test_factorize_value_preservation` on GCC *and* spatial, at `canonical_fock` True *and* False.
- **Do not revert D6.** The split is correct; only its granularity was in question. Reverting
  restores a route that does not compute its own equations.
- **Do not treat `operator_savings` as ground truth.** It is a model — element-count-free but
  still a model, and it does not price the call-site transpose or the emitter's actual contraction
  order. `docs/CCGEN_KERNEL_SCALING_SCOPE.md` measured the generated-vs-hand gap as a *scaling*
  defect that no current cost model predicts.
- **Do not attempt tag-blind operator matching for UCC.** See Remaining architecture concern
  below — it is measured and unsound.

## Remaining architecture concern: O6, UCC carry-over

Asked because the retirement answer predicts it would work: *"For UCC, the mechanism predicts it
would work — UCC keeps per-spin-block tensors rather than folding to one spatial tensor, so each
block stays close to the spin-orbital form where recognition is correct. Untested."*

**Measured on the current tree, `ucc_adapt_equations(generate_cc_equations("ccsd"))`:**

| | reuse sites | derived sites | seeded operators recognized |
|---|---|---|---|
| GCC | 36 | 54 | 5 |
| spatial | 61 | 87 | 5 |
| **UCC** | **0** | **370** | **none** |

The derivation route runs fine (370 operators). **Recognition is completely dead**, and the cause
is nomenclature, not structure: UCC factors are spin-tagged — `v_abab`, `t2_aaaa`, `f_aa` — while
`seeded_fingerprints()` keys on bare `('t1','v')`. Stripping the tag gives exactly `f, t1, t2, v`,
so the fingerprints never match anything.

**The obvious fix is UNSOUND, and it is this project's recurring failure shape.** Making the match
tag-blind does recover recognition — 152 reuse sites, the same five operators (Fme 47, Fmi 24,
Fae 18, Wmnij 6, Wmbej 57). It is also wrong, for the reason D6 was wrong:

| operator | distinct spin-tagged contractions collapsed onto ONE name |
|---|---|
| Wmbej | **12** |
| Fmi | 6 |
| Fme / Fae / Wmnij | 4 each |

`t1_aa·v_aaaa` and `t1_aa·v_abab` are **different arrays** under UHF. Binding them to one
`build_W` is the D6 defect at spin-block level — and the 52% dressed-energy defect was the same
shape one level up (spin-orbital fingerprints matched against spatial terms). Keyed on
`(operator, spin tags)` instead, the five names become **30 distinct operator instances** — the
honest count, each seeded operator needing its own per-block set, exactly as the UCC work found
for ERIs (24 arrays: 7 `aaaa` / 10 `abab` / 7 `bbbb`) and denominators.

What this means:

- The retirement's prediction is half right. Each UCC block *is* close to the spin-orbital form,
  so recognition is structurally applicable — but not with one shared fingerprint set. It needs a
  spin-resolved one, the same "one vocabulary per layer instead of one object" cost
  `CCGEN_UNRESTRICTED_CC.md` records for every other UCC layer.
- O1–O4 transfer without change. They operate on contraction shape and are indifferent to factor
  naming; a spin-tagged factor is just a factor. The `tensor_symmetries` table (O2.1) is where
  UCC's per-block relations would go — and `v_abab` does **not** have `v_aaaa`'s symmetries, so
  entries must be keyed per tagged name, not per base name.
- Do not attempt tag-blind matching. It is measured above, it looks like it works, and it is wrong.

A second, smaller symptom is already marked RED for O6:
`test_the_mixed_block_needs_more_arrays_than_the_same_spin_ones` is `expectedFailure` as of
2026-08-26: `abab` correctly emits 10 ERI arrays, but both same-spin tags emit **7** rather than
6. The extra is `ovvo`, which for a same-spin block folds into `ovov` under the particle swap (a
symmetry there and not for `abab`). Not a correctness defect — the redundant array holds the right
values — but it is the same spin-blocked ERI question, and fixing it before O6 means guessing at
the fold rule. Pre-existing at `b82fc69`. An unexpected pass means the fold landed.

When O6 is started, it needs a spin-resolved fingerprint set recognizing the seeded operators per
block, with **zero** cases of one emitted name covering two spin-tagged contractions — the
`test_one_name_one_contraction_shape` invariant extended to spin tags. No UCC numeric gate for the
factorizer exists yet; `test_factorize_value_preservation` would need a UCC evaluator
(`ucc_residual_einsum` + `ucc_closed_shell_tensors`'s block bundle) before any of this is trusted.

## Related but separate outcome: why this is coupled to, not merged into, the dressing scope

The dressing scope (`docs/CCGEN_TWO_DRESSING_ROUTES.md`) asks *does the derivation route compute
the right numbers* (D0–D7) and *is it worth wiring to production* (D8). This document asks *how
finely should operators be distinguished* — a question that only became askable once D6 answered
the first, and whose answer directly sets D8's input. D8 could not be run until this was settled:
re-costing the retirement against a 66×-degraded savings figure would resolve against reinstating
the dressed path for a reason that is an artifact of over-splitting, not a property of dressing.
That is the specific wrong conclusion this doc exists to prevent.
