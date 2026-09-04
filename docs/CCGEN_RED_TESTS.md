# ccgen Nine Standing Red Tests

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Why were nine ccgen Python tests red, and what did fixing them find?**

## Short answer

Nine tests in the ccgen Python suite had been failing — through the whole `merge_transposes` investigation and well before it. All nine are now green or skipped, and not one was a live product defect. Every one was a test whose claim the code had legitimately moved past, or an optional dependency reporting its absence as a failure. The suite went `862 passed, 10 failed` to **876 passed, 82 skipped, 0 failed** (the tenth failure was caused by the merge work and fixed with it; the nine here were red on a clean `HEAD`, verified in a `git worktree` rather than inferred).

The most useful finding is not the fix count: two of the nine were misdiagnosed in the scope written for this work, by someone who had already read the failures — one filed as a distributional claim turned out to be an engine mismatch, and a measurement asserting a property "never" held came from too coarse a scan and was simply wrong. A standing red is not noise — it is a place nobody has looked recently, and the stated reasons rot faster than the code does. Nine of them train a team to read red as normal, which is how a real regression gets waved through.

## Where the logic lives

- `python/ccgen/tests/test_factorize.py` (`CostModelTests`, `CCSDTQTests`) — the six selection gates
- `python/ccgen/optimization/factorize.py` — `_derived_name`, `select_under_memory_budget`, `select_best_of_both`
- `python/ccgen/tests/test_factorize_value_preservation.py` — `FactorOrderValueTests`, the shuffle value probe
- `python/ccgen/tests/test_reference_vs_pyscf.py` — `_HAVE_PYSCF`, the pyscf guard both borrowers need
- `python/ccgen/lowering/restricted_closed_shell.py` — the lowering C pins
- `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md` — why the antisymmetric ERI form is wrong here

## The three causes

| # | tests | cause | broke in |
|---|---|---|---|
| A | 6 in `test_factorize.py` | an operator split changed the savings distribution they assert | `7bdfdaf1` |
| B | 2 in `test_iterate_amps_fixed_point.py` | pyscf absent, and the skip guard could not fire | environment |
| C | 1 in `test_optimizations.py` | asserted the antisymmetry defect W4.3 had fixed | `04a5ac2b` |

## What invariants matter

### 1. A fixture must not silently validate the wrong relation

`test_planck_term_uses_lowered_eri_block_and_phase` (cause C) demanded that `v(a,i,j,b)` lower to `-mo_blocks.ovov(i, a, j, b)`. The emitter produces `+mo_blocks.ovvo(i, a, b, j)`. The emitter was right. Checked against a fixture carrying only the symmetries a real spatial ERI has (`<pq|rs> = <qp|sr> = <rs|pq>`, deliberately *not* antisymmetry):

```
max| v(a,i,j,b) - ( -ovov(i,a,j,b) ) | = 8.77e-01     <- what the test asserted
max| v(a,i,j,b) - ( +ovvo(i,a,b,j) ) | = 0.00e+00     <- what the emitter emits
```

`-<ic|ka>` for `<ic|ak>` is the antisymmetry relation: true for the antisymmetrized `<pq||rs>`, false for the spatial blocks these kernels index. That is precisely the defect `04a5ac2b` fixed, where 41 of 288 emitted builders read the wrong block with a bogus sign (see `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`). The fix corrected the lowering and left this gate behind still pinning the pre-fix phase — a test that would have gone *green* if the defect returned.

Design rule:

- Do not use `random_tensors` in a numeric gate here. It antisymmetrizes `v`, under which the invalid ERI relations are *true*. That fixture is why the 41/288 defect passed every symbolic check, and this gate is what it left behind.
- Any gate pinning a magic string derived from a symmetry relation should carry a docstring recording which relation and why — the absence of one here is how the bogus phase survived a lowering rewrite.

### 2. A fixture can be too general as well as too narrow

The merge work separately saw a random `t2` make valid merges look broken, because real amplitudes satisfy `t2(a,b,i,j) == t2(b,a,j,i)`. Withholding a symmetry the physical object *has* manufactures failures — the inverse of the antisymmetric-fixture trap above.

Design rule:

- Check the fixture's symmetries against the object being modeled before concluding a defect exists.

### 3. A test that cannot fire its own skip guard fails as if the feature broke

`test_lih_is_a_fixed_point_with_live_triples` and `test_be_triples_are_inert` (cause B) each wrapped their call in `except ImportError: self.skipTest(...)`. The handler could never fire. `test_reference_vs_pyscf` imports pyscf under a `try/except` that leaves `gto` unbound and protects itself with a module-level `skipUnless(_HAVE_PYSCF)`. These two borrow its helpers *without* inheriting that decorator, so they reach `gto.M(...)` and raise `NameError`, which `except ImportError` does not catch.

Design rule:

- A borrowed test helper must inherit the same skip guard as its source, not attempt to reconstruct the condition with a narrower exception type.
- Verify a skip guard in both directions: that it skips when the dependency is absent, and that it runs when forced true. A guard that always skips is worth nothing.

### 4. A property gate must be restated from a measurement, not retuned to pass

All six `test_factorize.py` gates (cause A) broke in `7bdfdaf1`, which fixed the factorizer's value preservation and, as part of that, folded contraction shape into `_derived_name` — correctly splitting operators 26 to 83 at rank 3. Splitting redistributes the same savings across more, smaller entries, and every one of the six asserts a property of that distribution. The commit that caused this already recorded the debt:

> Six selection-model gates still fail; they assert properties of the savings distribution that the split genuinely changed, and need their claims restated rather than their constants moved.

One of the six is different in kind. `test_operator_set_invariant_under_factor_order` asserts a real correctness property — that the derived operator multiset is a function of the terms, not of factor input order — and shuffling factor order produces genuinely different decompositions, not one contraction misnamed:

```
base:  t2(a,e,j,k) v(d,e,l,m) t1(b,l) t1(d,i)
shuf:  t2(a,e,i,j) v(d,e,l,m) t1(c,m) t1(d,k)
```

The value gate ran only on the unshuffled manifold, so whether a shuffled-order decomposition still reproduces its source terms was unmeasured — and that is exactly what must hold if factor order can steer the factorizer. It holds: **0 disagreements** across 4 seeds, GCC doubles+triples and spatial singles+doubles, using the value harness's own `_disagreements` oracle (each rewritten term compared against *its own* source, so shuffled input is self-referencing and needs no second oracle). Non-vacuity asserted — the shuffle demonstrably moves 16–22 operator names per seed — and mutation-verified: a sign flip in `rewrite_term_factorized` fails all 16 subtests. Landed as `FactorOrderValueTests`.

So the factorizer reaches different but equally valid trees. Factor order changes the decomposition, not the value, and all six were test debt.

Design rule:

- Do not move constants to make a property gate pass. A retuned constant under the old docstring pins today's behaviour for no stated reason — a change-detector, which is what cause C had degenerated into.
- Do not delete a red gate. `7bdfdaf1` kept these deliberately; they measure a selection model that still ships.

## What was fixed

1. **Cause C** — `test_planck_term_uses_lowered_eri_block_and_phase` now asserts `ovvo(i, a, b, j)`, with the measurement in the docstring. Two additions beyond the minimum fix: a counter-assertion that `ovov(i, a, j, b)` is absent (checking the right block appears does not check the wrong one is gone, and "the wrong one is gone" is the property W4.3 established), and a second test executing the numeric claim on a fixture deliberately not antisymmetrized (under an antisymmetric `v` the rejected relation becomes *true* and the gate would pass vacuously — the same blind spot that let the 41/288 defect through every symbolic check). Mutation-verified: restoring `ANTISYMMETRIZED_ERI_SYMMETRIES` in `lowering/restricted_closed_shell.py` reproduces the old `-ovov(i, a, j, b)` exactly and turns the gate red — it fails for the defect it exists to catch, not merely for a changed string.
2. **Cause B** — Fixed with a class-level `skipUnless(T._HAVE_PYSCF, ...)` on both tests, reusing `test_reference_vs_pyscf`'s flag rather than deriving a second one so the two cannot disagree about what "have pyscf" means. The dead `except ImportError` handlers are deleted rather than left beside a working guard, where they would invite the next reader to trust them. Verified in both directions: SKIPPED without pyscf, and running when the flag is forced true. `test_spatial_residual_vs_pyscf` borrows the same helpers with no `_HAVE_PYSCF` mention and was checked too — it is already correct by a different route, doing its own `from pyscf import ...` inside the test body so a genuine `ImportError` is raised and caught.
3. **Cause A** — each of the six gates restated against measured numbers:

| gate | restated as | measured |
|---|---|---|
| `savings_concentration` | a **fraction** of the set, not a fixed top-5 | top ⅛ > 85 %, top ¼ > 92 %, bottom half < 5 % |
| `ccsdt_keys_barely_diverge` | the **median** budget shows no divergence | median 0.0000; 15 of 133 budgets ≥ 1 % |
| `joint_beats_flops_only_baseline` | **searches** the range for the divergence regime | peak +5.77 % at 3200 GB |
| `optimized_beats_baseline_all_axes` | same search; stride axis pinned separately | stride holds at every budget tested |
| `operator_set_invariant_under_factor_order` | the **reference count**, and savings drift < 1 % | 963 refs exactly, every seed |
| `emit_memory_budget_selects_best_of_both` | unchanged — it was never distributional | 24 = 24 |

`emit_memory_budget_selects_best_of_both` was misfiled — it is a plain test bug: the emitter helper defaults `engine="diagram"` while the comparison side called `generate_cc_equations` without it, taking the `"wick"` default — a different equation set, hence 26 selected operators against 24 emitted. The test carries a comment warning about this exact trap for `canonical_fock`, one argument over.

The "never clears 5 %" claim behind `joint_beats_flops_only_baseline` was also wrong. A coarse scan said the joint selector could no longer beat flops-only by the >5 % asserted. A 100 GB sweep finds 37 divergent budgets, peaking at +5.77 % at 3200 GB — which clears it. The regime moved; it did not shrink below the claim.

Three judgement calls behind the restatements:

- The +5.77 % peak is a knife edge — +1.58 % at 3150 GB, +4.95 % at 3250 — so pinning 3200 would swap one brittle constant for another and re-break on the next legitimate change to operator identity. Both CCSDTQ gates therefore search the range and assert a qualifying regime exists, which is what the claim actually says.
- The stride axis was measured independent of the budget (holds at 850 and 3200 GB alike), so it stays pinned at a fixed budget rather than riding the search. Binding it to the searched budget would hide which axis regressed.
- The invariance gate asserts what is invariant, measured rather than assumed: total operator **references** are exactly 963 under every shuffle, while distinct names (484 -> 485-488), operator count (80 -> 81-84) and total savings (within 0.17 %) all move. Every term is factored to the same depth regardless of input order — the same work is hoisted, only the grouping differs.

Each restatement is mutation-verified against the defect it claims to guard:

| mutation | caught by |
|---|---|
| reverse the density key ordering | `keys_barely_diverge` |
| flatten `operator_savings` to a constant | `savings_concentration` |
| make the joint selector ignore its density arm | both CCSDTQ gates |
| make hoisting depth depend on factor order | `operator_set_invariant` |

## Validation strategy that should remain in place

- The full ccgen Python suite (`876 passed, 82 skipped, 0 failed`) run to completion, not sampled
- `FactorOrderValueTests` as the standing check that factor-order shuffles never change the reproduced value, only the decomposition
- The mutation table above kept alongside each restated gate so a future editor can re-run it after any change to `factorize.py` or `restricted_closed_shell.py`
- Confirming failures in a clean `git worktree` rather than inferring cause from memory or a stale scope document

## What is still open

- Whether the factorizer should be made order-invariant by a canonical tie-break over equal-cost candidates. The value probe removed *correctness* as a reason to hurry; build reproducibility is still a legitimate argument for it. The invariance gate therefore also fails loudly if shuffling ever stops changing the operator set — that would mean this question had been answered, and the stronger multiset-equality assertion should return. Its docstring says so.
- Whether CI has pyscf at all. If it does not, the two tests fixed under cause B have never run anywhere, and the fixture they protect — LiH, chosen because Be's `t3` is inert and cannot validate a rank-3 kernel — is unguarded. The skip makes that visible instead of drowning it in a red count.
