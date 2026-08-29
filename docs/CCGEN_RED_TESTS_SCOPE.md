# Scope: the nine red ccgen tests

**Scope for in-flight work. C and B are DONE (2026-08-29); A remains.** Written
after the `merge_transposes` work found the suite at `862 passed, 10 failed`. One
of those ten was caused by that work and is fixed; **the nine below were red
before it and are red on a clean `HEAD`** (verified in a `git worktree` at HEAD,
not inferred). C and B account for three of the nine and are now green/skipped,
leaving **A's six**. A's blocking question — is it test debt or a live defect? —
is answered: **test debt.** The value probe is green (see below).

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

**Status: C DONE, B DONE, A open.**

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

### The value probe — **DONE (2026-08-29). It holds: 0 disagreements.**

This was the measurement that could have reclassified the whole item from "stale
gate" to "live defect", and it was sequenced first for that reason. The value gate
(`test_factorize_value_preservation`) ran on the **unshuffled** manifold only, so
whether a shuffled-order decomposition still reproduces its source terms was
untested — and that is exactly the property that must hold if factor order can
steer the factorizer.

Reusing that file's own `_disagreements` oracle (each rewritten term compared
against **its own** source via `residual_einsum`, so shuffled input is
self-referencing and needs no new oracle):

| basis | manifold | unshuffled | 4 shuffled seeds |
|---|---|---|---|
| GCC | doubles | 0 / 32 | **0** / 30-32 |
| GCC | triples | 0 / 324 | **0** / 316-322 |
| spatial | singles | 0 / 4 | **0** / 4 |
| spatial | doubles | 0 / 21 | **0** / 18-22 |

**Non-vacuity asserted, not assumed** — the shuffle must actually change the
decomposition or this compares a tree against itself: measured **16-22 operator
names differ per seed** while the operator count stays at exactly 963, and
316-322 terms are rewritten and checked each seed. **Mutation-verified**: flipping
the sign of the rewritten term's coefficient in `rewrite_term_factorized` fails
all 16 subtests, so the gate detects value defects rather than merely running.

**Conclusion: the factorizer reaches different but equally valid trees.** Factor
order changes the decomposition, not the value. So the invariance gate asserts
something **stronger than correctness requires**, and A stays test debt — no live
defect. Landed as `FactorOrderValueTests` in
`python/ccgen/tests/test_factorize_value_preservation.py`, covering GCC
doubles+triples and spatial singles+doubles (spatial stops at `ccsd` because the
spatial fixture carries no `t3`, the same limit
`test_spatial_rewrite_is_value_preserving` works under).

**What this does NOT settle:** which invariant the gate SHOULD assert. Option 1
(canonical tie-break, so the multiset really is order-invariant) is still a
legitimate choice on build-reproducibility grounds, and this probe does not argue
against it — it only removes correctness as the reason to hurry.

*Verify:* whichever option, the gate must fail if a real order-dependence is
introduced — mutation-test it, as `test_merged_call_sites` was.

---

## Sequencing

1. ~~**C**~~ — **DONE.** Stale gate asserting a fixed bug; was actively misleading.
2. ~~**B**~~ — **DONE.** One decorator. Whether CI has pyscf is still unanswered.
3. ~~**A's value-preservation-under-shuffle probe**~~ — **DONE.** 0 disagreements
   across 4 seeds on both bases; A is test debt, not a defect.
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
