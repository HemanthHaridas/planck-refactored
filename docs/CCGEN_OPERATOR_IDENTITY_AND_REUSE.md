# When are two derived contractions the SAME operator?

**Research scope. Not started.** Coupled to
`docs/CCGEN_TWO_DRESSING_ROUTES.md`, which owns the value-preservation question
(D0–D8) and whose D6 fix created this one.

The factorizer must answer one question at every tree node: *have I built this operator already?*
Answer too coarsely and it emits one `build_W` for two different contractions — the D6 defect,
23/66 GCC terms wrong. Answer too finely and it stores the same array several times under
different names, which is correct but destroys the reuse the factorizer exists to create.

D6 moved the answer from far-too-coarse to **measurably too fine**. This doc is the question of
where the right line is.

---

## What D6 did, and what it cost

D6 folded the contraction shape into the operator name (`_contraction_shape` + a 2-byte digest,
`W_t2v_ooov_a049`). That was necessary and is not in question here — the value gate
`test_factorize_value_preservation` goes 21/41 → 0/41 (GCC) and 39/63 → 0/63 (spatial) because
of it, and holds under `canonical_fock` both ways.

The cost, GCC `ccsd` doubles at `canonical_fock=True` (the emitter's default):

| | before D6 | after D6 |
|---|---|---|
| distinct operators | 12 | **27** |
| terms actually rewritten | 39 | **30** |
| operators with `usage_count > 1` | 12 / 12 | **17 / 27** |
| usage histogram | `[11,6,6,6,5,4,2,2,…]` | `[6,2,2,2,2,2,2,2,…]` |
| modeled total savings (`operator_savings`, 30/100) | 4.96e+12 | 7.48e+10 |

**That last row is NOT a like-for-like comparison, and O3 shows most of it is a pricing fix.**
The 4.96e+12 was computed with `nary_cost`, which D4 replaced because it overstates multi-factor
operators (500x on one measured case). Priced the same way on both sides, the split costs
`1.00x -> 0.51x`, and **merging recovers it to 0.85x** (O3). The row is kept because it is the
figure this doc was opened on; read O3 before quoting it.

Two real losses remain, and they compound:

1. **Sharing collapses.** Operators that served 11 or 6 call sites now serve 2 — merging brings
   this back to `[6,4,4,4,4,3,…]`.
2. **Nine terms stop being factored at all.** Once split, most operators fall below the savings
   threshold, so `select_operators_by_savings` declines to materialize them and the terms inline.

Rank 3 splits less violently (24 → 84 operators), so the effect is not uniform in rank and the
`ccsd`-doubles number should not be extrapolated.

## The over-splitting is real, and now fully characterized

Grouping the 27 split operators back by their pre-D6 base name and asking which pairs are the
same contraction up to a permutation of the operator's slots:

```
8 / 23 same-family pairs on GCC       38 / 229 on spatial
```

Concretely, two `W_t2v_ooov` variants with **identical slots** `(i,j,k,a)`:

```
t2(a,c,j,l) v(i,c,k,l)
t2(a,c,i,l) v(j,c,k,l)     <- the same contraction with i <-> j
```

D6 gives these different names because their slot *positions* differ. One stored array serves
both if the call site reads it transposed `(1,0,2,3)`.

This is decided **symbolically**, on the shape key, not by comparing arrays — O1 built the
predicate and O2 completed it, and both are exact against a numeric oracle at two fixtures and
three seeds. Merging by union-find:

| | operators | usage histogram |
|---|---|---|
| after D6 | 27 | `[6,2,2,2,2,2,2,2]` |
| **+ symbolic merge** | **19** | `[6,4,4,4,4,3,2,2]` |

About **half** the split is surplus and recoverable without touching correctness. Whether that
recovers the *savings* proportionally is the open question — see O3, which is what D8 needs.

## What to establish

### O1 — LANDED (2026-08-25): transpose-equivalence is symbolic, and exact

`python/ccgen/optimization/operator_identity.py` + `tests/test_operator_identity.py`.

`symbolic_transpose(sp1, sp2, spatial=)` decides the question on the **shape key**, not by
comparing arrays. Held against a numeric oracle (materialize both, try every axis permutation)
at **two fixtures** (`no=3,nv=4` and `no=4,nv=3`, inverting the asymmetry) × **three seeds**:

| | pairs | agree | false merges | misses |
|---|---|---|---|---|
| **GCC** (`canonical_fock` True and False) | 23 | **23** | **0** | **0** |
| **SPATIAL** (both) | 229 / 230 | **229 / 230** | **0** | **0** |

Each basis runs on the fixture that matches it — spin-orbital `random_tensors` for GCC, the
`spatial` bundle of `ucc_closed_shell_tensors` for spatial.

Spatial reached exactness in three stages, and the middle one is the cautionary part:

| stage | spatial misses |
|---|---|
| as first written (oracle on the spin-orbital fixture) | 48 / 49 |
| **O2.0** — oracle moved to the spatial fixture | 18 / 19 |
| **O2.2** — predicate models `t2[abij] = t2[baji]` | **0** |

~30 of the original 48 were **oracle false positives**, not predicate gaps — pairs equal only
because `random_tensors` antisymmetrizes `t2`.

**Exact on GCC. Sound but incomplete on spatial** — every disagreement is a miss, never a false
merge, which is the direction that matters: a false merge binds one array to two different
contractions, the D6 defect again.


Three sources of freedom are required, each found by a failing case rather than by design:

1. **slot permutation** — the transpose itself.
2. **summed-label permutation** — `_contraction_shape` numbers internal indices canonically
   *per shape*, so the numbering does not align across two shapes. Without this,
   `W_t1t1v_ooov_b656`/`_719d` (identical slots, `v`'s summed labels swapped) is missed.
   GCC went 20/23 → 23/23.
3. **the ERI's own SIGN-PRESERVING symmetries** — see below.

#### The sign restriction, which is the load-bearing part

Folding all of `_ERI_PERMUTATIONS` produced **two false merges on GCC** (`W_t2v_vv`,
`W_t1t1v_oo`): the predicate claimed equivalence for arrays differing by a **sign**. Four of the
eight members are odd and hold only up to `<qp|rs> = -<pq|rs>`.

This is the *same blind spot* that let the 52 % dressed-energy defect pass every symbolic check
— already documented on `_ERI_PERMUTATIONS_SPATIAL` in `dressing.py`, and reached independently
here by a probe that had it wrong first. Restricting to the even members gives 8 merges instead
of 10; `SignPreservationTests` pins the reason so a future edit reaching for the full orbit
fails with the cause named. **Verified falsifiable:** removing the parity filter makes the
predicate find 10 merges and turns the suite red (3 failures).

#### Exactness is only meaningful with a discrimination check

A predicate that merged *everything* would also report zero misses. Pinned by
`PredicateStillDiscriminatesTests`: merges are a strict subset of same-family pairs (GCC 8/23,
spatial 38/229), and **cross-family merges are 0** — binding unrelated contractions to one array
is the failure this whole line of work exists to prevent.

### O2 — COMPLETE (2026-08-25): the spatial gap is closed

Scoped as five steps; O2.0–O2.2 landed and closed O2.3/O2.4 with them. The predicate is now
exact on both bases. The record below keeps the order the work actually happened in, because the
middle step is the transferable lesson: **the first thing this found was a defect in its own
oracle, not in the code under test.**

#### O2.0 — DONE (2026-08-25): the fixture was wrong; fixing it IMPROVED the result

`random_tensors` builds **antisymmetrized** `t2` and `v` — the GCC fixture. Spatial `t2` and
`<pq|rs>` have no such antisymmetry, so on spatial terms the oracle was reporting pairs equal
when they were equal only *for that fixture*.

Re-run on `ucc_closed_shell_tensors`' `spatial` bundle (`t2[abij] == t2[baji]`,
`<pq|rs> == <qp|sr>`, neither antisymmetric):

| spatial oracle | pairs | agree | false merges | misses |
|---|---|---|---|---|
| old (spin-orbital fixture) | 229 / 230 | 181 | 0 | 48 / 49 |
| **new (spatial fixture)** | 229 / 230 | **211** | **0** | **18 / 19** |

**The error was in the ORACLE, not the predicate.** ~30 of the 48 "misses" were oracle false
positives. O1's soundness claim survives unchanged — still zero false merges — and its
completeness on spatial was *understated*.

`test_factorize_value_preservation`'s spatial gate also re-run on the correct fixture: still
**0 disagreements** (singles 0/4, doubles 0/21, both `canonical_fock`). Confirmed falsifiable
there — reverting the D6 shape tag gives 41/61 on the spatial fixture.

Both test files now select the fixture by basis (`_tensors(..., spatial=)`), and the value gate
is parameterized over `canonical_fock` as well, closing a loose end open since D5.

#### O2.0b — the same fixture question, audited across the repo

`random_tensors` is used by nine files. Only three besides the two fixed here evaluate
**spin-adapted** terms through it: `test_dress_per_operator`, `test_emit_flag_matrix`,
`test_residual_symmetry`.

**None is compromised**, and the reason is worth recording because it bounds the blast radius:
those three are **A-vs-B comparisons of two writings of the same equation set**, not checks of a
physical property. Both sides see identical tensors, so an unphysical fixture cancels. Verified
rather than argued — `test_adapted_dressed_matches_adapted_raw` re-run on the spatial fixture
gives rel ~2.5e-15 (vs ~1.3e-15 on the spin-orbital one); the whole of
`test_dress_per_operator` passes on both.

The gates that *do* break under substitution are the ones that must:
`test_v_carries_all_three_symmetries` and `test_gcc_expansion_matches_raw` assert **GCC**
properties and correctly require the GCC fixture.

**Rule.** A fixture must match the basis when a check asserts a property *of the tensors*, or
compares a value against an independent oracle. It may be unphysical when both sides of a
comparison contract the same arrays. The two gates fixed here were the first kind wearing the
second kind's clothes.

#### O2.1 + O2.2 — LANDED (2026-08-25): a per-tensor symmetry table, with `t2` in it

One change, because they are the same edit: `v_variants`' hardcoded `v` case became
`tensor_symmetries(spatial) -> {tensor name: sign-preserving permutations}`, and `t2` was added
to it.

```
"v":  parity-+1 members of the ERI symmetry group   (from dressing.py, not re-derived)
"t2": ((0,1,2,3), (1,0,3,2))                        # t2[abij] = t2[baji]
```

**Result: spatial misses 18/19 → 0, false merges still 0.** Exact on both bases, at both
`canonical_fock` settings. Better than the advance estimate — that 48 → 30 was measured on the
superseded fixture, and once O2.0 removed the oracle's false positives, `t2` symmetry closed the
entire residue rather than half of it.

**`v`'s pattern does NOT transfer to `t2`, and the table says so.** Spatial `t2` is symmetric
under the *simultaneous* pair swap only; the single-pair swaps are not symmetries of it at all
(they are antisymmetries of the spin-orbital `t2`). Adding one produces false merges — verified,
not assumed. A factor whose name is absent from the table is treated as opaque, which is the
safe default: an unmodelled symmetry costs a missed merge, never a false one.

*Falsifiability, checked both directions:* removing the `t2` entry reopens the misses (2 test
failures); adding the unsound single-pair swap creates false merges (2 test failures).

#### O2.3 / O2.4 — CLOSED by O2.2

O2.3 was "classify what remains" and O2.4 was "decide whether to stop". Nothing remains: the
predicate is exact on both bases. If a future manifold (rank 3, rank 4 — see O5) reopens a gap,
the classification question returns with it, and the table is the place to extend.

### O3 — ANSWERED (2026-08-25): the transpose is free, and most of the "66x loss" was a PRICING FIX

Two questions: what merging is worth, and what the call-site transpose costs. The second one
reframes the first.

#### The call-site transpose costs nothing

Inspected in the emitted C++ (`emit_factorized_translation_unit`, 15 call sites):

```
acc += amplitudes.t1(k, a) * W_t2v_ooov_d4a0(i, j, k, b);
```

Every operator is read **by index, inside the loop nest**. No `W` is ever copied or transposed
into a temporary — grepped, zero occurrences. So merging two transpose-equivalent operators
changes a call site from

```
W_..._a049(i,j,k,b)   and   W_..._d4a0(i,j,k,b)     <- two arrays, today
W_..._a049(i,j,k,b)   and   W_..._a049(j,i,k,b)     <- one array, indices swapped
```

That pair is real: `a049` = `t2(a,c,j,l) v(i,c,k,l)`, `d4a0` = `t2(a,c,i,l) v(j,c,k,l)`,
permutation `(1,0,2,3)`, and they are already emitted at adjacent call sites. The merge costs an
index reorder in generated source and **zero data movement** — so `operator_savings` pricing the
transpose at zero is, for this emitter, correct rather than optimistic.

#### The 66x was ~97% a pricing correction, not a real loss

The pre-D6 baseline of **4.96e+12** was computed with `nary_cost`. D4 replaced that with
`build_cost` because `nary_cost` prices a multi-factor operator as one flat loop nest and
overstates it badly (measured 500x on `W_t2v_oooovv`). The pre-D6 figure is dominated by exactly
such an operator — `W_t1t2v_ooov`, `nary` = 8.1e+11, contributing 4.05e+12 of the 4.96e+12.

Like-for-like on GCC `ccsd` doubles, both priced with `nary_cost` as the baseline was:

| | operators | savings (nary) | vs baseline |
|---|---|---|---|
| pre-D6 | 12 | 4.96e+12 | 1.00x |
| D6 split | 27 | 2.54e+12 | 0.51x |
| **+ merged** | **19** | **4.21e+12** | **0.85x** |

**Merging recovers the split loss almost entirely: 0.51x -> 0.85x.** The residual 15% is the
genuine cost of distinguishing operators that really are different.

Under the corrected `build_cost` pricing — the honest number, and the one to carry forward:

| | GCC | SPATIAL |
|---|---|---|
| D6 split | 7.48e+10 | 6.81e+10 |
| **+ merged** | **9.28e+10** (1.24x) | **2.24e+11** (3.29x) |
| operators | 27 -> 19 | 59 -> 31 |

Spatial gains far more, which is the case that matters for the dressed route.

#### What this means for D8

**The over-splitting objection to D8 is resolved.** The savings figure D8 should use is the
merged, `build_cost`-priced one — not 7.48e+10 (un-merged), and not 4.96e+12 (mis-priced).
D8 is unblocked.

One caveat to carry: all of the above is `operator_savings`, a FLOP model. It does not price the
emitter's actual contraction order, which `docs/CCGEN_KERNEL_SCALING_SCOPE.md` measured as a
scaling defect no current cost model predicts. Treat these as ratios between comparable
configurations, not as absolute performance.

*Not done:* the merge is measured but **not implemented** — `manifold_operators` still emits 27
distinct operators on GCC. O4 built the canonical key that would do it and found that naming
alone is not enough: the rewrite must permute each call site too, or the D6 defect returns.

### O4 — COMPLETE (2026-08-25): operators merge, end to end, in the emitted C++

All six steps landed. The canonical key exists, call sites permute, names merge, the invariant is
gated for both paths, and the merge reaches the emitted C++ — 27 → 19 builders on `ccsd`,
264 → 76 on CCSDTQ.

**Naming operators by the key does not work on its own**, which is the finding the ladder was
built around and the reason it has six steps rather than one.

#### What landed

`canonical_shape(spec, spatial)` — the orbit representative of a shape under slot permutation,
summed relabeling and the factors' own symmetries, paired with the slot-space pattern (two
shapes can canonicalize alike while their slots differ occ/vir, and one array cannot serve both).

It partitions operators **exactly** as the pairwise predicate does, at one canonicalization per
operator instead of one comparison per pair: GCC 27 → 19 classes, spatial 59 → 31 — the same
numbers O3's union-find produced. Gated by `CanonicalShapeTests`, which checks the partition in
both directions (no co-classified pair is inequivalent, no equivalent pair is split).

#### Why naming alone is wrong — measured, not predicted

Folding `canonical_shape` into `_shape_tag` merges the names and produces exactly the intended
counts (19 / 31) and histograms (`[6,4,4,4,4,3,…]`). It also **reintroduces the D6 defect**: 11
GCC doubles terms stop reproducing their source, and the value gate fails 8/8.

The reason is the half O3 measured and this step forgot to act on. A merged operator is one array
serving several call sites *with permuted indices*. But `rewrite_term_factorized` builds the call
site as

```python
block = tuple(canonical_index_order(list(child.block)))   # the SITE's own order
new_factors.append(Tensor(name, block))
```

— the site's canonical index order, with no knowledge of which slot order the shared operator is
stored in. Merge the names without permuting the sites and the array is read as if it were the
other member of its class.

**Reverted.** `_shape_tag` is back to the un-canonicalized shape; `canonical_shape` stays in
`operator_identity.py` with its gate, ready for the wiring.

#### What remains, in six verifiable steps

The naming half is built and reverted; what is missing is the call-site permutation. Each step
below is independently checkable, and **every one of them keeps
`test_factorize_value_preservation` at 0/0 on both bases and both `canonical_fock` settings** —
that gate, not the operator count, is the acceptance criterion throughout. A step that improves
the count and reddens the gate has failed.

The surface is narrower than "a contract change" suggested. Measured:

| end | where | reads |
|---|---|---|
| call site | `_access_indices` → one caller, `planck_tensor_cpp.py:539` | the factor's `indices` |
| builder | `planck_tensor_cpp.py:1079` | `spec.indices` |

Two independent points, so the two halves can be verified separately — which is what makes the
ladder possible.

##### O4.1 — LANDED (2026-08-25): the merge plan exists and is inert

`merge_plan(specs, spatial)` in `operator_identity.py` returns
`{name: (representative_name, permutation)}` for **every** input spec — a representative maps to
itself under the identity, so no caller special-cases it. `permutation[k] = j` means slot k of
this operator is slot j of the representative.

The representative is the lexicographically smallest name in the class, which makes the plan
deterministic and independent of input order.

| | operators | classes | non-identity permutations |
|---|---|---|---|
| GCC doubles | 27 | **19** | **8** |
| spatial doubles | 59 | **31** | **19** |

The class counts match `canonical_shape`'s partition and O3's union-find exactly.

**Inertness verified, which is the whole point of this step.** Operator sets across all seven
manifolds (`gcc`/`spatial` `ccsd` + `ccsdt`, 212 operators) are **byte-identical** to the
pre-O4.1 snapshot — same names, same usage counts. Value gate 0/0.

*Gates:* `MergePlanTests` — totality, partition match, self-mapped representatives, determinism
under input reordering, and **that the permutations are not all the identity** (8 GCC / 19
spatial). That last one guards against O4.2/O4.3 passing vacuously: if every permutation were
the identity, merging would be a rename and their gates would prove nothing.

*Falsifiability checked:* making the representative choice order-dependent fails 2 tests; making
the plan skip representatives fails 4.

##### O4.2 — LANDED (2026-08-25): call sites permute; the merge is proven safe

`rewrite_term_factorized(..., merge_plan_map=plan)` orders each hoisted child's indices into its
class representative's slot order. Names stay un-merged, so every operator still owns its array —
which is the point: a wrong permutation surfaces as a value failure that **cannot** be blamed on
sharing.

**The decisive measurement.** With call sites permuted and each one reading its
*representative's* array rather than its own:

| | disagreements |
|---|---|
| GCC doubles | **0 / 30** |
| spatial doubles | **0 / 21** |

That is O4.3's proof: the merge is value-preserving once the permutation is in place. 9 GCC
terms change under the plan; operator counts are unchanged (27 / 59), and the default path
(no `merge_plan_map`) is byte-identical across all seven manifolds.

*Gates:* `PermutedCallSiteTests` — permuted sites reproduce their terms, plus two anti-vacuity
assertions.

**The anti-vacuity assertion earned its place immediately.** The first version used the
`savings_fraction=1.0` keep set; on spatial that admits 13 operators, almost none of the 19 with
a non-identity permutation, so **zero call sites were permuted** and the spatial half was
proving nothing. The gate failed on its own `permuted > 0` check rather than passing green.
Fixed by hoisting everything (`keep_operators=None`), which exercises 17 permuted operators.

**Known limit, recorded rather than papered over.** Every permutation `merge_plan` produces on
`ccsd` is a single two-element swap and therefore **self-inverse** (8 GCC, 19 spatial, measured).
Applying `perm` backwards is a no-op on this data, so the gate cannot detect inversion. It does
catch dropping the permutation (3 failures) and applying a wrong one (2 errors) — both verified
by sabotage. A manifold with a 3-cycle would close the gap; rank 3+ is where to look.

##### O4.3 — LANDED (2026-08-25): the names merge, and the terms still evaluate

`manifold_operators_with_plan(terms, spatial=)` returns the merged specs **and** the call-site
plan as a pair. That pairing is the design, not convenience: emitting merged specs while call
sites read in their own slot order is exactly the reverted first attempt, and handing the two
halves back together is what stops them being separated again.

| | operators | usage histogram | disagreements |
|---|---|---|---|
| GCC doubles | 27 → **19** | `[6,4,4,4,4,3,2,2]` | **0 / 39** |
| spatial doubles | 59 → **31** | — | **0 / 64** |

Zero disagreements **with only the 19/31 representatives existing** — the merge is live, not
simulated.

Implemented as an opt-in `merge_transposes` flag rather than by folding `canonical_shape` into
`_shape_tag`. The tag route merges everywhere at once, including callers whose call sites do not
permute; the flag keeps the unsafe combination unreachable. Default path verified byte-identical
across all seven manifolds (212 operators).

*Gates:* `MergedOperatorsTests` — merged representatives reproduce every term, the merge actually
removes operators (anti-vacuity: 27 > 19), and the default path still returns 27.

*Falsifiability, both failure modes:* merging **without** permuting — the reverted attempt —
fails 5 tests; merging into the **wrong** representative fails 2.

Full suite after: 95 tests, the same 6 selection-model failures as before O4.3. No new breakage.

##### O4.4 — LANDED (2026-08-25): the invariant ADDED for the merged path, not restated

The step was scoped as "`test_one_name_one_contraction_shape` will fail for the right reason
once merging works; restate it." **It did not fail** — because O4.3 made merging opt-in, that
gate exercises the default un-merged path and is still correct there.

So the real gap was not a wrong assertion but a **missing one**: nothing covered the merged path
at all. Measured under merging:

| | emitted names | covering >1 RAW shape | covering >1 CANONICAL shape |
|---|---|---|---|
| GCC doubles | 19 | **8** | **0** |
| spatial doubles | 31 | **21** | **0** |

An emitted name covering several *raw* shapes **is** the merge — that is the intended behaviour,
and asserting against it would forbid the feature. What still guards correctness is one name per
*canonical* shape: two contractions that are not transpose-equivalent must never share a
`build_W`.

`test_one_emitted_name_one_canonical_shape_when_merged` asserts exactly that, alongside an
anti-vacuity check that some name really does cover multiple raw shapes — without it the gate
would silently degrade into a re-run of the un-merged one whenever merging stopped working.

The original gate is **kept unchanged**: it still describes the default path, which is what most
callers use.

*Falsifiability:* dropping the slot-space guard from `canonical_shape` (which over-merges
operators whose slots differ occ/vir) fails 6 tests; making the canonical key ignore slot spaces
entirely fails 1.

Full suite: 96 tests, same 6 selection-model failures. No new breakage.

##### O4.5 — LANDED (2026-08-25): the emitted C++ shares builders — after two real bugs

Scoped as "check the output". It was not: the emitter never requested merging, so the TU
contained **zero** merged operators. Wiring it up exposed two defects in O4.3's plumbing that
every algebra-level gate had missed.

**Bug 1 — the budget check rejected every merged call site.** `keep_operators` holds
*representative* names after merging (the members no longer exist as specs), but the hoist test
asked `name in keep_operators` using the *member's* name. Every merged site failed the check and
was inlined, silently un-doing O4.2. Fixed by testing the representative.

**Bug 2 — call sites referenced builders that do not exist.** With the permutation applied but
the name left alone, a site emitted `W_member(...)` while only `build_W_rep` was generated — a
dangling reference in the generated source. Fixed by emitting the representative's name too.

Neither is visible from the algebra: both leave the symbolic rewrite evaluable. **Only reading
the emitted text finds them**, which is the entire justification for this step.

Result on `ccsd`:

| | builders emitted |
|---|---|
| default | **27** |
| `merge_transposes=True` | **19** |

with 8 operators read in a different index order than they would be un-merged, zero merged-away
names surviving, and every builder defined exactly once.

*Gate:* `test_merged_emit_shares_builders_and_permutes_reads` — builder count drops, no
merged-away name survives, each builder defined once, and **some operator's read order changes**
versus the un-merged emission. That last assertion is compared against the real un-merged output
rather than a guessed canonicality rule; a first attempt at inferring "canonical order" was
wrong and failed on correct code.

*Falsifiability, all three failure modes:* merging without permuting → 5 failures; renaming
without permuting → 5; permuting without renaming → 3. The first of these was **not** caught by
the gate as first written — it passed — and the assertion above was added because of it.

**The O4.2 gate needed updating, and its anti-vacuity check is why that was noticed.** Once call
sites emit the representative's name, `plan[f.name]` is always the identity, so the gate's
permutation counter read zero and it failed on `permuted > 0` rather than passing green. Counting
now compares against the un-permuted rewrite.

Full suite: 97 tests, same 6 selection-model failures. No new breakage.

##### O4.6 — LANDED (2026-08-25): measured on the real merge; D8 has its number

O3's figures came from a simulated union-find merge. Re-taken on the implemented one, they
**reproduce exactly** — the estimate was sound.

| | operators | savings (`build_cost`) | vs un-merged |
|---|---|---|---|
| GCC doubles | 27 → **19** | 7.48e+10 → **9.28e+10** | 1.24x |
| spatial doubles | 59 → **31** | 6.81e+10 → **2.24e+11** | **3.29x** |
| CCSDT triples | 83 → **42** | — | — |
| CCSDTQ (d+t+q) | 264 → **76** | — | — |

Like-for-like against the pre-D6 baseline (both priced with `nary_cost`, as that 4.96e+12 was):

| | operators | savings | vs pre-D6 |
|---|---|---|---|
| pre-D6 | 12 | 4.96e+12 | 1.00x |
| D6 split | 27 | 2.54e+12 | 0.51x |
| **+ merged** | **19** | **4.21e+12** | **0.85x** |

**The over-splitting objection to D8 is closed.** The correct-and-merged configuration costs
~15% of the pre-D6 modeled savings, not 66x — and the pre-D6 figure was itself mis-priced. The
number D8 should carry is the merged `build_cost` one: **GCC 9.28e+10, spatial 2.24e+11**, the
spatial case being what the dressed route cares about.

#### Which of the six failing gates merging fixes: one partly, none fully

Predicted that merging might resolve some by moving the distribution back. Measured:

| gate | un-merged | merged | verdict |
|---|---|---|---|
| `savings_concentration` (wants >0.98) | 0.6535 | **0.8599** | much better, still fails |
| `ccsdt_keys_barely_diverge` (wants <0.01) | 0.2097 | 0.1845 | barely moves — **premise genuinely falsified** |
| `joint_beats_flops_only_baseline` | exact tie | **still exact tie** | unrelated to the split |
| `optimized_beats_baseline_all_axes` | exact tie | still exact tie | same |

So the re-derivation is still owed, and it should be done against the **merged** set. Two
distinct kinds:

- `savings_concentration` and `ccsdt_keys_barely_diverge` encode measured *findings* about the
  operator distribution. The split changed the distribution, so the findings must be re-taken and
  the prose updated — `ccsdt_keys_barely_diverge`'s claim that "operators cluster by footprint on
  CCSDT" is simply no longer true at 42 operators.
- The two exact ties are **not** about the split at all: both selection strategies return
  identical totals at 850 GB, un-merged and merged alike. The budget is large enough to make the
  comparison vacuous, which the gate's own docstring hints at ("at a budget in the divergence
  regime"). Fix the budget, not the threshold.

`emit_memory_budget_selects_best_of_both` is a third kind again — the emitter and the test build
their operator sets differently (all manifolds vs triples-only, and `canonical_fock` defaults).

**Do not re-pin any of these to their new values without deciding which kind each is.** Three of
the four need their claim restated, not their constant loosened.

### O5 — ANSWERED (2026-08-25): the merge holds at every rank, and IMPROVES with rank

Measured at `canonical_fock=True` (what the emitter uses), merge and value-check both:

| manifold | operators | non-identity perms | savings | disagreements |
|---|---|---|---|---|
| `ccsd` doubles | 27 → **19** (1.4x) | 8 | 1.02x | **0 / 45** |
| `ccsdt` triples | 80 → **39** (2.1x) | 41 | 1.03x | **0 / 345** |
| `ccsdtq` quadruples | 254 → **69** (3.7x) | 184 | 1.20x | **0 / 2536** |

**The merge ratio strengthens with rank** — 1.4x → 2.1x → 3.7x. Higher-rank manifolds generate
far more transpose-equivalent operators, so the over-splitting D6 introduced was worst exactly
where the factorizer matters most, and the fix pays best there.

**Rank 4 is the strongest single result in this doc**: 2536 rewritten quadruples terms, zero
disagreements, nothing unevaluable. It is asserted separately from rank 3 because this codebase
has twice shown rank 3 does not predict rank 4 — the tensor-accessor fix left rank 4 completely
unchanged while giving rank 3 a 206x speedup, and the rank-3 solver defect did not generalize.
Rank 4 also uses different tensor types and a different code path.

*Gate:* `HigherRankMergeTests`, ~26 s. The rank-4 case is slow and deliberately kept.

*Config trap, hit while writing the gate:* `generate_cc_equations` defaults to
`canonical_fock=False` and gives (83, 42) at rank-3 triples; the emitter's `True` gives (80, 39).
Two different equation sets. The gate pins the emitter's.

### O6 — does any of this carry over to UCC? (~M, measured; NOT started)

Asked because the retirement answer predicts it would: *"For UCC, the mechanism predicts it would
work — UCC keeps per-spin-block tensors rather than folding to one spatial tensor, so each block
stays close to the spin-orbital form where recognition is correct. Untested."*

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

#### The obvious fix is UNSOUND, and it is this project's recurring failure shape

Making the match tag-blind does recover recognition — 152 reuse sites, the same five operators
(Fme 47, Fmi 24, Fae 18, Wmnij 6, Wmbej 57). It is also wrong, for the reason D6 was wrong:

| operator | distinct spin-tagged contractions collapsed onto ONE name |
|---|---|
| Wmbej | **12** |
| Fmi | 6 |
| Fme / Fae / Wmnij | 4 each |

`t1_aa·v_aaaa` and `t1_aa·v_abab` are **different arrays** under UHF. Binding them to one
`build_W` is the D6 defect at spin-block level — and the 52 % dressed-energy defect was the same
shape one level up (spin-orbital fingerprints matched against spatial terms).

Keyed on `(operator, spin tags)` instead, the five names become **30 distinct operator
instances**. That is the honest count: each seeded operator needs its own per-block set, exactly
as the UCC work found for ERIs (24 arrays: 7 `aaaa` / 10 `abab` / 7 `bbbb`) and denominators.

#### What this means

- **The retirement's prediction is half right.** Each UCC block *is* close to the spin-orbital
  form, so recognition is structurally applicable — but not with one shared fingerprint set. It
  needs a spin-resolved one, which is the same "one vocabulary per layer instead of one object"
  cost `CCGEN_UNRESTRICTED_CC.md` records for every other UCC layer.
- **O1–O4 transfer without change.** They operate on contraction shape and are indifferent to
  factor naming; a spin-tagged factor is just a factor. The `tensor_symmetries` table (O2.1) is
  where UCC's per-block relations would go — and note `v_abab` does **not** have `v_aaaa`'s
  symmetries, so entries must be keyed per tagged name, not per base name.
- **Do not attempt tag-blind matching.** It is measured above, it looks like it works, and it is
  wrong. Recorded here so the next person measures the collapse instead of rediscovering it from
  a wrong energy.

**A second, smaller symptom is already marked RED for O6.**
`test_the_mixed_block_needs_more_arrays_than_the_same_spin_ones` is
`expectedFailure` as of 2026-08-26: `abab` correctly emits 10 ERI arrays, but both same-spin tags
emit **7** rather than 6. The extra is `ovvo`, which for a same-spin block folds into `ovov`
under the particle swap (a symmetry there and not for `abab`, which is why `abab` legitimately
carries both). Not a correctness defect — the redundant array holds the right values — but it is
the same spin-blocked ERI question, and fixing it before O6 means guessing at the fold rule.
Pre-existing at `b82fc69`. An unexpected pass means the fold landed.

*Verify, when started:* a spin-resolved fingerprint set recognizing the seeded operators per
block, with **zero** cases of one emitted name covering two spin-tagged contractions — the
`test_one_name_one_contraction_shape` invariant extended to spin tags. No UCC numeric gate for
the factorizer exists yet; `test_factorize_value_preservation` would need a UCC evaluator
(`ucc_residual_einsum` + `ucc_closed_shell_tensors`' block bundle) before any of this is trusted.

## What NOT to do

- **Do not loosen the key without the value gate.** The whole reason this question exists is
  that the coarse key was wrong for 23/66 terms and no gate caught it for the factorizer's
  entire life. `test_factorize_value_preservation` exists now; every candidate key runs against
  it, on GCC *and* spatial, at `canonical_fock` True *and* False.
- **Do not revert D6.** The split is correct; only its granularity is in question. Reverting
  restores a route that does not compute its own equations.
- **Do not treat `operator_savings` as ground truth.** It is a model — element-count-free but
  still a model, and it does not price the call-site transpose (O3) or the emitter's actual
  contraction order. `docs/CCGEN_KERNEL_SCALING_SCOPE.md` measured the generated-vs-hand gap as
  a *scaling* defect that no current cost model predicts.
- **Do not fix the failing `test_factorize` selection gates by re-pinning constants.** They
  assert properties of the savings distribution, which this work will move again. Settle the key
  (O4) first, then re-derive them once.

  Correction on the record: these were reported here as *pre-existing*. They are not — a clean
  worktree at `8e4bb0c` runs `test_factorize` **72/72 green**. The earlier "pre-existing" reading
  came from a baseline whose `unittest discover` had a bad `-t` root, so `test_factorize` never
  ran and contributed no failures. `test_savings_concentration` measures **0.9882 at baseline**
  (passes `>0.98`) and **0.6535 here** — the D6 split, 26 → 83 CCSDT operators, spreading savings
  across more of them. They are regressions from this work, in the selection model only; the
  value gate is 0/0 throughout.

## Why this is coupled to the dressing scope, not merged into it

The dressing scope asks *does the derivation route compute the right numbers* (D0–D7) and
*is it worth wiring to production* (D8). This asks *how finely should operators be
distinguished* — a question that only became askable once D6 answered the first, and whose
answer directly sets D8's input.

**D8 cannot be run until this is settled.** Re-costing the retirement against a 66×-degraded
savings figure would resolve against reinstating the dressed path for a reason that is an
artifact of over-splitting, not a property of dressing. That is the specific wrong conclusion
this doc exists to prevent.

## Key code locations

| what | where |
|---|---|
| the shape key D6 introduced | `_contraction_shape`, `_shape_tag`, `python/ccgen/optimization/factorize.py` |
| where names are assigned | `_derived_name`, `identify_node`, same file |
| where operators are deduped | `manifold_operators` (`rep = specs[0]`), same file |
| the savings model | `operator_savings`, `build_cost`, `OperatorValue`, same file |
| threshold that drops split operators | `select_operators_by_savings`, same file |
| the equivalence predicate (O1/O2) | `tensor_symmetries`, `symbolic_transpose`, `python/ccgen/optimization/operator_identity.py` |
| its gate | `python/ccgen/tests/test_operator_identity.py` |
| basis-matched fixtures | `random_tensors` (GCC) / `ucc_closed_shell_tensors(...)[1]` (spatial), `python/ccgen/tests/residual_eval.py` |
| **the referee** | `python/ccgen/tests/test_factorize_value_preservation.py` |
| the coupled scope | `docs/CCGEN_TWO_DRESSING_ROUTES.md` (D6 landed; D8 blocked on O3) |
| cost-model caveat | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
