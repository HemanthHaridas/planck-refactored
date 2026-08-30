# Why were nine ccgen tests red, and what did fixing them find?

**DONE (2026-08-29). All nine are green or skipped; the suite is clean.** Written
after the `merge_transposes` work found the suite at `862 passed, 10 failed`. One
of those ten was caused by that work and was fixed with it; **the nine here were
red before it and on a clean `HEAD`** (verified in a `git worktree`, not
inferred).

**None was a live product defect** — as suspected, though not for the reason
assumed in every case. Seven were tests that outlived their premise, one was a
missing optional dependency, and one (`emit_memory_budget_selects_best_of_both`)
turned out to be a plain test bug this document had misfiled as distributional:
its two sides generated equations with **different engines**. A's blocking
question — test debt or live defect? — was settled first by the value probe:
**test debt.**

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

**Status: all DONE.**

Ordered below by cost, cheapest first. **C is the one to do first regardless** —
it is a stale gate pinning a known-wrong phase, so it is actively misleading.

---

## C — **DONE (2026-08-29).** One test asserted a bug that was fixed

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

**What landed.** The assertion is now `mo_blocks.ovvo(i, a, b, j)`, with the
`-ovov` measurement above written into the docstring so the phase reads as a claim
about spatial ERI symmetry rather than a magic string — the absence of any
docstring is how the stale assertion survived a lowering rewrite. Two additions
beyond the minimum:

- **A counter-assertion**, `assertNotIn("ovov(i, a, j, b)")`. That is the property
  W4.3 established and nothing else in the file pinned it; without it the gate
  only checks that the right block appears, not that the wrong one is gone.
- **A second test**, `test_spatial_eri_lacks_the_antisymmetry_the_lowering_must_not_use`,
  which executes the numeric claim instead of quoting it. Its fixture carries only
  the real spatial symmetries and is deliberately **not** antisymmetrized — under
  an antisymmetric `v` the relation it rejects becomes true and it would pass
  vacuously, which is the same trap that let the 41/288 defect through every
  symbolic check.

**Mutation-verified:** swapping `_ERI_SYMMETRY_PERMUTATIONS` back to
`ANTISYMMETRIZED_ERI_SYMMETRIES` in `lowering/restricted_closed_shell.py`
reproduces exactly the old `-mo_blocks.ovov(i, a, j, b)` and turns the gate red.
So it fails for the defect it exists to catch, not merely for a changed string.

---

## B — **DONE (2026-08-29).** Two tests could not skip when pyscf is missing

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

**What landed.** A class-level
`@unittest.skipUnless(T._HAVE_PYSCF, ...)`, reusing `test_reference_vs_pyscf`'s
flag rather than re-deriving one so the two cannot disagree about what "have
pyscf" means. The two per-test `except ImportError` handlers are **deleted**: they
never fired, and leaving dead handlers beside a working guard invites the next
reader to trust them.

*Verified both directions*, because a guard that always skips is worth nothing:

- Without pyscf, both report **SKIPPED** — not passed, not failed.
- With `_HAVE_PYSCF` forced true, both **run** and reach pyscf. So the decorator
  gates on the right condition rather than disabling the tests.

**Checked for the same pattern elsewhere.** `test_spatial_residual_vs_pyscf.py`
also borrows those helpers with no `_HAVE_PYSCF` mention, but it is already
correct for a different reason — it does its own `from pyscf import ...` **inside
the test body**, so a genuine `ImportError` is raised and caught. Left alone.

**Not in scope:** installing pyscf in the default dev environment. These are
cross-code validation gates and are correctly optional; the defect was that a
missing optional dependency reported as a failure. **Whether CI has pyscf is still
worth knowing** — if it does not, these two have never run anywhere, and the
fixture they protect (LiH, because Be's t3 is inert and cannot validate a rank-3
kernel) is unguarded. The skip now makes that visible instead of drowning it in a
red count.

---

## A — **DONE (2026-08-29).** Six gates, and one was not what this scope claimed

All six broke in `7bdfdaf1`, whose own commit message named them:

> Six selection-model gates still fail; they assert properties of the savings
> distribution that the split genuinely changed, and need their claims restated
> rather than their constants moved.

That commit fixed value preservation, and part of the fix folded contraction
**shape** into `_derived_name` — correctly splitting operators 26 → 83 at rank 3,
which redistributes savings across more, smaller entries.

**One correction to this document.** It classified all six as distributional.
`test_emit_memory_budget_selects_best_of_both` is **not** — it is a plain test
bug. Its emitter helper defaults `engine="diagram"` while its comparison side
called `generate_cc_equations` without it, picking up the `"wick"` default: a
different equation set, hence 26 selected operators against 24 emitted. The test
even carries a comment warning about exactly this trap for `canonical_fock`, one
argument over. Fixed by passing the engine.

**And one correction to this document's measurements.** It stated the joint
selector "never" beats flops-only by the >5 % asserted. That was from a coarse
scan. A 100 GB sweep finds **37 divergent budgets, peaking at +5.77 % at 3200 GB**
— which *does* clear the bar the gate demands. The regime moved; it did not
shrink below the claim.

**What each gate now asserts, and why:**

| gate | restated as | measured |
|---|---|---|
| `savings_concentration` | a **fraction** of the set, not a fixed top-5 | top ⅛ > 85 %, top ¼ > 92 %, bottom half < 5 % |
| `ccsdt_keys_barely_diverge` | the **median** budget shows no divergence; exceptions stay a minority | median 0.0000, 15 of 133 budgets ≥ 1 % |
| `joint_beats_flops_only_baseline` | **searches** the range for the divergence regime | peak +5.77 % at 3200 GB |
| `optimized_beats_baseline_all_axes` | same search; stride axis pinned separately | stride holds at every budget tested |
| `operator_set_invariant_under_factor_order` | **reference count** is invariant; savings drift < 1 % | 963 refs exactly, every seed |
| `emit_memory_budget_selects_best_of_both` | unchanged — it was an engine mismatch | 24 = 24 |

Three judgement calls worth recording:

- **The +5.77 % peak is a knife edge** — +1.58 % at 3150 GB, +4.95 % at 3250 — so
  pinning 3200 would swap one brittle constant for another and re-break on the
  next legitimate change to operator identity. Both CCSDTQ gates therefore
  **search** the range and assert that a qualifying regime exists, which is what
  M2.3 actually claims.
- **The stride axis (B3) was measured independent of the budget** (holds at both
  850 and 3200 GB), so it stays pinned at a fixed budget rather than riding the
  search. Binding it to the searched budget would hide which axis regressed.
- **The invariance gate asserts what is invariant**, measured rather than assumed:
  total operator **references** are exactly 963 under every shuffle, while
  distinct names (484 → 485-488), operator count (80 → 81-84) and total savings
  (within 0.17 %) all move. Every term is factored to the same depth regardless of
  input order — the same work is hoisted, only the grouping differs. It also
  fails loudly if shuffling ever *stops* changing the set, since that would mean
  the open question below had been answered and multiset equality should return.

**Mutation-verified, each against the defect it claims to guard:**

| mutation | gate that caught it |
|---|---|
| reverse the density key ordering | `keys_barely_diverge` |
| flatten `operator_savings` to a constant | `savings_concentration` |
| make the joint selector ignore its density arm | both CCSDTQ gates |
| make hoisting depth depend on factor order | `operator_set_invariant` |

**Still open, deliberately:** whether the factorizer *should* be made
order-invariant by a canonical tie-break over equal-cost candidates. That is a
build-reproducibility argument, not a correctness one — the value probe settled
correctness — and it is recorded in the gate's own docstring so whoever does it
knows to restore the stronger assertion.

## Sequencing

1. ~~**C**~~ — **DONE.** Stale gate asserting a fixed bug; was actively misleading.
2. ~~**B**~~ — **DONE.** One decorator. Whether CI has pyscf is still unanswered.
3. ~~**A's value-preservation-under-shuffle probe**~~ — **DONE.** 0 disagreements
   across 4 seeds on both bases; A is test debt, not a defect.
4. ~~**A's five distributional gates**~~ — **DONE.** Four restated against
   measured numbers; the fifth was an engine mismatch, not a distributional claim.
5. ~~**A's invariance gate**~~ — **DONE.** Asserts the reference count, which is
   exactly invariant, and fails loudly if shuffling ever stops mattering.

**Acceptance, met:** `python -m pytest ccgen/tests/ -q` reports zero failures,
with skips only where an optional dependency is genuinely absent. That was the
point — not the individual green ticks, but that the next real regression is
visible instead of buried in nine standing failures.

**The lesson worth carrying.** Two of the nine were misdiagnosed in this very
document before anyone looked closely: one filed as a distributional claim was an
engine mismatch, and the "never clears 5 %" measurement came from too coarse a
scan and was wrong. A standing red is not just noise — it is a place where nobody
has looked recently, and the reasons drift as much as the code.

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
