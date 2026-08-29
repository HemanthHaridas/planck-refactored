# Scope: the nine red ccgen tests

**Scope for in-flight work. Not started.** Written 2026-08-29 after the
`merge_transposes` work found the suite at `862 passed, 10 failed`. One of those
ten was caused by that work and is fixed; **the nine below were red before it and
are red on a clean `HEAD`** (verified in a `git worktree` at HEAD, not inferred).

**None is a live product defect.** Every one is a *test* that outlived its
premise, or an environment gap. That is the finding, and it is also the risk: a
suite with nine standing failures trains everyone to read red as normal, which is
how a real regression gets waved through. The fix is to make each one green or
explicitly skipped, not to keep explaining them.

## The three causes

| # | tests | cause | commit that broke them |
|---|---|---|---|
| A | 6 in `test_factorize.py` | operator split changed the savings distribution | `7bdfdaf1` (2026-08-26) |
| B | 2 in `test_iterate_amps_fixed_point.py` | pyscf absent; the skip guard does not fire | environment |
| C | 1 in `test_optimizations.py` | **asserts the antisymmetry defect W4.3 fixed** | `04a5ac2b` |

Ordered below by cost, cheapest first. **C is the one to do first regardless** —
it is a stale gate pinning a known-wrong phase, so it is actively misleading.

---

## C — one test asserts a bug that was fixed (~XS, do this first)

`test_planck_term_uses_lowered_eri_block_and_phase` asserts that `v(a,i,j,b)`
lowers to `-mo_blocks.ovov(i, a, j, b)`. The emitter now produces
`+mo_blocks.ovvo(i, a, b, j)`.

**The emitter is right and the test is wrong.** Checked numerically against a
fixture carrying only the symmetries a real spatial ERI has
(`<pq|rs> = <qp|sr> = <rs|pq>`, and NOT antisymmetry):

```
max| v(a,i,j,b) - ( -ovov(i,a,j,b) ) | = 8.77e-01     <- what the test asserts
max| v(a,i,j,b) - ( +ovvo(i,a,b,j) ) | = 0.00e+00     <- what the emitter emits
```

`-<ic|ka>` for `<ic|ak>` is the **antisymmetry** relation. It holds for
antisymmetrized `<pq||rs>` and is **false for the spatial blocks these kernels
index** — exactly the defect `04a5ac2b` fixed, where 41 of 288 emitted builders
read the wrong block with a bogus sign (see
`CCGEN_WIRING_THE_DERIVATION_ROUTE.md`). The fix corrected the lowering and left
this gate behind, still pinning the pre-fix phase.

**The work:** update the assertion to `+mo_blocks.ovvo(i, a, b, j)` and put the
numeric justification in the test body, so it reads as a claim about spatial ERI
symmetry rather than a magic string. **Add the counter-assertion too** — that the
antisymmetric form is NOT emitted — since that is the property W4.3 established
and nothing else in this file pins it.

*Verify:* green, and re-introducing the antisymmetry relation in
`lowering/restricted_closed_shell.py` turns it red.

**Do not** "fix" this by making the emitter match the test.

---

## B — two tests cannot skip when pyscf is missing (~XS)

`test_lih_is_a_fixed_point_with_live_triples` and `test_be_triples_are_inert`
both intend to skip without pyscf — each wraps its call in
`except ImportError: self.skipTest(...)`. **The handler never fires.**

`test_reference_vs_pyscf.py` imports pyscf under a `try/except ImportError` that
sets `_HAVE_PYSCF = False` and leaves **`gto` unbound**; the module then guards
itself with a module-level `skipUnless(_HAVE_PYSCF)`.
`test_iterate_amps_fixed_point.py` borrows that module's helpers **without
inheriting the decorator**, so it reaches `gto.M(...)` and raises
`NameError: name 'gto' is not defined` — which `except ImportError` does not
catch.

**The work:** apply the same `skipUnless(_HAVE_PYSCF)` guard that
`test_reference_vs_pyscf` already uses, importing that flag rather than
re-deriving it. One decorator, two tests.

*Verify:* the two report SKIPPED (not passed, not failed) in an environment
without pyscf, and still run where it is installed.

**Not in scope:** installing pyscf in the default dev environment. These are
cross-code validation gates and are correctly optional; the defect is that a
missing optional dependency reports as a failure. **Whether CI has pyscf is worth
knowing** — if it does not, these two have never run anywhere, and the fixture
they protect (LiH, because Be's t3 is inert and cannot validate a rank-3 kernel)
is unguarded.

---

## A — six selection-model gates outlived their premise (~M, the real work)

All six broke in `7bdfdaf1`, whose own commit message names them:

> Six selection-model gates still fail; they assert properties of the savings
> distribution that the split genuinely changed, and need their claims restated
> rather than their constants moved.

That commit fixed the factorizer's value preservation. Part of the fix folded the
contraction **shape** into `_derived_name`, because one name had denoted several
different contractions. That correctly **split** operators — 26 → 83 at rank 3 —
and splitting one operator into several redistributes savings across more, smaller
entries. Every one of the six asserts a property of that distribution.

**Measured, so the restatement starts from numbers rather than a story:**

| test | asserts | measured now |
|---|---|---|
| `test_savings_concentration` | top 5 carry > 98 % of savings | **0.656** (0.863 even merged) |
| `test_ccsdt_keys_barely_diverge` | savings- and density-greedy agree < 1 % | **0.210** |
| `test_joint_beats_flops_only_baseline` | joint beats flops-only by > 5 % at 850 GB | **+0.00 %** at 850 GB |
| `test_optimized_beats_baseline_all_axes` | same, all axes | same |
| `test_emit_memory_budget_selects_best_of_both` | joint > baseline savings | **equal** |
| `test_operator_set_invariant_under_factor_order` | see below — **different in kind** | 8-10 names differ |

**The divergence regime moved rather than vanished**, which is why these need
restating and not deleting. Scanning budgets on `ccsdtq` (264 operators):

```
 100-1200 GB   +0.00 %
     1800 GB   +0.51 %
     2500 GB   +1.09 %
     4000 GB   +4.62 %   <- peak
     6000 GB   +0.00 %
```

So the joint selector still wins, but **never by the > 5 % the gate demands**, and
the 850 GB the test hardcodes is now flat. Moving the constant to 4000 GB would
make it pass at +4.62 % and still fail the `> 0.05` assertion — the *claim* has to
change, which is what "restated rather than their constants moved" means.

**The work for five of the six:** re-derive each property against the current
operator set, then assert the property with a margin, not the historical number.
Where a property is simply no longer true (concentration at 98 %), say so in the
docstring and assert what IS true — a gate that pins today's measurement with no
stated reason is a change-detector, not a property gate.

### `test_operator_set_invariant_under_factor_order` is different in kind — do it separately

The other five are distributional claims. This one asserts a **genuine
invariant**: the derived operator multiset must be a function of the terms, not of
factor input order. That is a correctness property, and it is currently false.

Measured on rank-3 triples, shuffling factor order across 4 seeds:

- **The operator COUNT is invariant** — 963 both ways, every seed.
- But **8-10 operators differ by name**, and inspecting them, they are **genuinely
  different contractions**, not the same one misnamed:

```
base:  t2(a,e,j,k) v(d,e,l,m) t1(b,l) t1(d,i)
shuf:  t2(a,e,i,j) v(d,e,l,m) t1(c,m) t1(d,k)
```

So factor order steers the factorizer to a **different, equally valid
decomposition**. The invariant as written is not merely unmet — it may be the
wrong invariant, since nothing requires one canonical tree.

**Decide which before writing code:**

1. **The multiset genuinely should be order-invariant** → the tie-break in tree
   selection is incomplete and needs a canonical order over equal-cost
   candidates. Real work, real payoff: build reproducibility.
2. **Only the RESULT need be order-invariant** → assert what actually matters —
   operator count, total savings, and *value preservation* — and drop the
   name-multiset equality.

**Option 2 exposes the gap that most concerns me here.** The value gate
(`test_factorize_value_preservation`) runs on the **unshuffled** manifold only. So
whether a shuffled-order decomposition still reproduces its source terms is
**untested**, at rank 3 and rank 4 alike. If factor order can steer the
factorizer, value preservation under shuffling is exactly the property that must
hold — and it is the one nobody has checked. **Measure that first**: it is cheap
(the value harness exists), and a red result there would reclassify this whole
item from "stale gate" to "live defect".

*Verify:* whichever option, the gate must fail if a real order-dependence is
introduced — mutation-test it, as `test_merged_call_sites` was.

---

## Sequencing

1. **C** (~XS) — stale gate asserting a fixed bug; actively misleading.
2. **B** (~XS) — one decorator; also answers whether CI has ever run these.
3. **A's value-preservation-under-shuffle probe** (~S) — the one measurement that
   could turn this from test debt into a defect. Do it before the restatements.
4. **A's five distributional gates** (~M) — restate claims against measured
   numbers.
5. **A's invariance gate** (~M) — after the probe decides which invariant is the
   right one.

**Acceptance for the whole item:** `python -m pytest ccgen/tests/ -q` reports zero
failures, with skips only where an optional dependency is genuinely absent. That
is the point — not the individual green ticks, but that the next real regression
is visible instead of buried in nine standing failures.

## Traps

- **Do not move constants to make these pass.** Five of the six encode a
  *property*; a re-tuned constant with the old docstring is a gate that pins
  today's behaviour for no stated reason and will mislead the next reader exactly
  as C does now.
- **Do not delete a gate because it is red.** `7bdfdaf1` kept these deliberately;
  they are measurements of a selection model that still ships.
- **Do not use `random_tensors` in any new numeric gate here** — it
  antisymmetrizes `v`, under which the invalid ERI relations are true. That
  fixture is why the 41/288 defect passed every symbolic check, and C is the gate
  that defect left behind.
- **A fixture can also be too general.** The merge work hit the inverse trap: a
  random `t2` made valid merges look broken because real amplitudes satisfy
  `t2(a,b,i,j) == t2(b,a,j,i)`. If the shuffle probe goes red, check the fixture's
  symmetries before concluding a defect.

## Key code locations

| what | where |
|---|---|
| the six selection gates | `python/ccgen/tests/test_factorize.py` (`CostModelTests`, `CCSDTQTests`) |
| the split that changed the distribution | `_derived_name`, `python/ccgen/optimization/factorize.py` |
| the selection model under test | `select_under_memory_budget`, `select_best_of_both`, same file |
| the value gate that does not cover shuffling | `python/ccgen/tests/test_factorize_value_preservation.py` |
| the pyscf guard to reuse | `_HAVE_PYSCF`, `python/ccgen/tests/test_reference_vs_pyscf.py:27-32` |
| the corrected lowering C should assert | `python/ccgen/lowering/restricted_closed_shell.py` |
| why the antisymmetric form is wrong here | `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
