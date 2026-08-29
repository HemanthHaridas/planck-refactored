# How does a derived-operator CC kernel reach production, and what was wrong with it?

*Successor to the W1-W5 scope of the same name, rewritten as an answer once W4.3
went green (the scope's exemption expires when the work lands). Opened by
`CCGEN_TWO_DRESSING_ROUTES.md`, which established that the derivation route was
value-gated, worth 2-7x, and had no production caller — and recommended wiring
it. This is what that took.*

ccgen has two routes that produce dressed CC operators. **Recognition** matches
hand-seeded Stanton-Gauss fingerprints; it is retired, 52 % short on Be.
**Derivation** builds operators from each term's own contraction tree; it was
value-gated, worth 2-7x, and for months had **no production caller** — deferred
in its own commit and never revisited.

It has one now. This answers what wiring it took, the correctness defect that
wiring exposed, and how a defect of that size survived every gate the project
had.

## The wiring

**One dressing axis with a value, not a second boolean.**

```
--dressing {none,recognized,derived}        default: none
--dress-operators                           deprecated alias -> recognized
PLANCK_CC_DRESSING={recognized,derived}     the CMake side
```

The alternative — a `--derive-operators` flag beside `--dress-operators` — was
rejected on evidence in the tree, not taste. `print_cpp_planck` already carries
16 branches, `dress_operators` interacts at three separate points
(`generate.py:1052/1064/1152`), and a comment at `generate.py:1060` records that
a second emit call site had already forced UCC to be wired twice. A fourth
dressing-ish axis makes every pairwise combination a question with no meaningful
answer.

**The blocker was a seam, not an algorithm.** `emit_factorized_from_equations`
called the emitter itself, so it could not feed `print_cpp_planck`'s single
downstream emit. Splitting out `factorize_equations(eqs, ...) -> (rewritten_eqs,
kept_specs)` — exactly the `(eqs, intermediates)` pair the recognition route
already threads — made `derived` a branch rather than a fork. The old entry
became a thin delegate, byte-identical before and after.

**The two routes run at different points, and that is structural.** `recognized`
dresses BEFORE spin-adaptation, because its hand-seeded specs declare GCC layouts
that `adapt_intermediate_spec` must then transform. `derived` factorizes AFTER,
because it derives operators FROM whatever manifold reaches it, so its specs are
already in the adapted layout. Putting `derived` early would declare one layout
and build another.

## The defect it exposed

With the route wired, the first end-to-end energy comparison went red:

| system | undressed | derivation-dressed | Δ |
|---|---|---|---|
| CH4/STO-3G | −39.8058445098 | −39.8058606381 | **−1.61e-05** |
| LiH/STO-3G | −7.8823242576 | −7.8823350582 | **−1.08e-05** |

161x the tolerance, on two independent systems, **both converging cleanly**
(`rms(res)` 8.7e-11). A converged-but-wrong answer — the same signature as the
`SPIN_ADAPT` defect and the 52 % recognition defect before it.

**Root cause: two ERI symmetry tables, one of them invalid.**

A spatial physicist `<pq|rs>` over real orbitals has exactly four index
symmetries, all `+1`. The four single-swap relations `<qp|rs> = -<pq|rs>` and
`<pq|sr> = -<pq|rs>` hold **only** for the antisymmetrized `<pq||rs>`.

`lowering/restricted_closed_shell.py` carried the full 8-fold group. Its phase
reaches the emitted C++ directly — `_map_eri_tensor` returns
`LoweredTensorFactor.phase` without re-deriving it — so the four invalid
relations became wrong ERI reads with a bogus sign in **41 of 288** emitted
operator builders:

```
spec    : t1(c,j) v(i,c,a,k)               <ic|ak> = ovvo(i,c,a,k)
emitted : acc += -t1({j,c}) * mo_blocks.ovov(i, c, k, a);
fixed   : acc +=  t1({j,c}) * mo_blocks.ovvo(i, c, a, k);
```

`ovov` and `ovvo` differ by 3.9e-01 on the fixture; relative errors reached 8.8 —
larger than the quantity being computed.

**The fix:** `SPATIAL_ERI_SYMMETRIES` and `ANTISYMMETRIZED_ERI_SYMMETRIES` now
live once, in `ccgen/tensors.py` (a leaf both consumers import; `emit` imports
`lowering`, so the shared home cannot be either of them). Both bind to the
spatial set.

| system | after the fix | Δ vs undressed |
|---|---|---|
| CH4 | −39.8058445096 | **2e-10** |
| LiH | −7.8823242576 | **exact, ten digits** |

CH4 also converges in 15 steps against 26 — the wrong fixed point took longer to
reach. The fix additionally repairs the retired `recognized` route, which shared
the table and had no builder-level gate.

## Why every existing gate missed it

**The value gate never emits C++.** `grep -c "emit_planck\|print_cpp" on
test_factorize_value_preservation.py` returns **0**. It validates Python objects
— and the rewrite, the specs, the operator reuse and the per-term algebra were
each *exact* as objects while the rendered C++ computed a different tensor.

Two further reasons it could not have caught this:

- **It compares terms individually and skips single-step terms.** On the
  manifolds the kernel runs, it covers 4/36 singles, **27/142 doubles**, 589/806
  triples. The skip is correct — a single-step term has no operator to hoist —
  but "value-preserving" covers under a fifth of the doubles manifold.
- **Its fixture antisymmetrizes `v`.** `random_tensors` produces an
  antisymmetric ERI, under which the invalid relation is **TRUE**. Measured:
  0/288 builders disagree on an antisymmetrized fixture, 41/288 on a spatial one.
  Any gate reusing that fixture passes while the defect is fully present.

That last point is the sharpest lesson here: **a fixture with more symmetry than
the real object cannot see a defect that abuses symmetry.**

## How it was found: five eliminations

Each step removed a layer, and the order was cheapest-first. Two of them refuted
hypotheses that looked decisive.

| step | result |
|---|---|
| **D1** algebra | rewritten manifold SUM vs unrewritten: spatial doubles **exactly 0.0**, triples 1.4e-15. Clean. |
| **D2** per-term emit | one term rebuilt from its emitted loop: **3.6e-16**. Clean. The `canonical_fock` term-count gap (148 vs 142) is a red herring — `max\|f_ov\| = 7.8e-17`. |
| **D3** operator reuse | 616 rewritten terms through the shared-operator path: worst **2.5e-16**. Clean. |
| **D4** emitted text | **interpreting the emitted C++ in Python reproduced the disagreement** (5.06e-05 vs C++ 5.99e-05). Defect is in the emitter's rendering. |
| **D5** the table | one patched constant: **41/288 -> 0/288**. |

**Two operator censuses looked decisive and were each refuted by direct numeric
test.** First: the defect appeared to track operators over more than one distinct
amplitude kind (singles 0, doubles 15, triples 91). Filtering all 106 of them out
changed rank 2 by **nothing**. Second: it appeared to track operators read
through more than one index binding — a **perfect** correlation across three
manifolds (singles 0 and correct; doubles 10 and wrong; triples 175 and wrong).
Direct evaluation of all 616 terms refuted it too.

Both were real correlations and neither was causal. Treat a third census
correlation with suspicion.

## What now guards it

| gate | pins |
|---|---|
| `test_emitted_builder_matches_spec.py` | every `build_W_*` computes its own spec, **by evaluating the emitted C++ text**; ships a vacuity control asserting the fixture is spatial and NOT antisymmetric |
| `test_eri_symmetry_tables.py` | the relations verified on a real tensor; the odd ones verified FALSE on a spatial integral; no module redefines a signed table (matched on **shape**, so renaming does not evade); both consumers bind the shared object; `dressing.py`'s unsigned sets agree |
| `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g` | the generated route end to end, both requiring `PLANCK_CC_SPIN_ADAPT` |

**Why tests and not comments.** Two warning comments already existed —
`planck_tensor_cpp.py` ("Do NOT re-add the -1 perms") and `dressing.py`,
recording that folding all 8 caused a **52 % energy defect** that "passed every
symbolic check". Neither prevented a third module from carrying the bad set.
Comments document an invariant; only a test enforces one.

Writing the guard then found a **third** copy — `dressing.py`'s own
`_ERI_PERMUTATIONS` pair. Those are sign-free permutation *sets* used for
canonicalization, with parity computed separately, so they are deliberately not
merged: different shape, and merging would be a false unification. What they must
share is which permutations belong to which basis, and that is gated.

## Facts worth carrying

- **`choose_determinant_backstop` binds the hand-written tensor path only.**
  `PLANCK_RCCSDT_BACKEND=tensor` does not bypass it (it is called inside
  `run_tensor_rccsdt` off reference size), but `optimized` does — it routes
  through `rccgen.cpp` to the arbitrary-order harness, which never consults it.
  So the `nso > 16 || ndet > 10000` limit recorded across several ccgen scopes
  does **not** constrain generated-route test cases. LiH/STO-3G (nso=12,
  ndet=495) exercises the generated route in 5 s.
- **The generated undressed kernel reproduces the hand-written baseline** to
  3e-10 on CH4, which is what makes a dressing disagreement attributable.
- The six failing selection-model gates (`test_savings_concentration` and five
  others) are pre-existing and unrelated; baselined on a clean worktree, before
  and after.

## What it is worth: the first wall-clock numbers

Everything previously claimed for this route (2.0x-7.1x) was a FLOP model.
Measured, same input, same binary configuration apart from `--dressing`:

| system | no/nv | undressed | derivation-dressed | speedup |
|---|---|---|---|---|
| LiH/STO-3G | 4/8 | 5.12 s | **1.64 s** | **3.12x** |
| CH4/STO-3G | 5/4 | 104.56 s | **28.94 s** | **3.61x** |

Medians of 3 and 2 runs; spreads 0.03-0.10 s (LiH) and 0.3-0.4 s (CH4).
**Energies identical to all printed digits on both**, and CH4 takes 15 steps
either way — so this is per-iteration work, not fewer iterations.

Both land inside the modelled range, which is worth stating plainly because
`CCGEN_KERNEL_SCALING_SCOPE.md` gave good reason to expect otherwise: it measured
the generated-vs-hand-written gap as a *scaling* defect that no cost model
predicts. The model survived contact here; two points is not enough to say it
survives generally, and the ratio does grow between them (3.12 -> 3.61).

## One emitter

`emit_factorized_translation_unit` is deleted (-45 lines). It had **no production
caller** — 25 references, all tests — so the "two emitters" problem was already
one emitter plus dead weight by the time W3.3 came due.

The generate-then-emit convenience now lives in `test_factorize.py`, its only
consumer. Inlining `generate_cc_equations` at all 25 call sites would have been a
net *positive* diff to remove 13 lines, which is not what "delete the second
emitter" was for.

`print_cpp_planck` gained exactly one parameter (`dressing`) and none of the
factorizer's seven selection knobs — the condition W3 set for doing the merge
after W4/W5 rather than before.

## What remains

- **`merge_transposes` is not threaded** on the production path, so `derived`
  emits the un-merged 59 builders on spatial `ccsd` rather than 31. Scoped in
  `docs/CCGEN_MERGE_TRANSPOSES.md` — which also corrects a reading this
  document invited: the 1.4x -> 2.1x -> 3.7x figures are an **operator count**
  reduction, while the modelled FLOP saving is 1.02x-1.20x. The likely win is
  compile time, not speed, and that should be measured before wiring.
- **The scaling ladder has not been re-run under dressing.**
  `CCGEN_KERNEL_SCALING_SCOPE.md` attributes the generated-vs-hand gap to H3
  (n-ary contraction order) and recommends consuming
  `_optimal_contraction_order`. Dressing addresses the same hypothesis by a
  different mechanism, so the two fixes may overlap; its six-point ladder should
  be re-run with `--dressing derived` before the emitter change is attempted.
- **UCC** is out of scope: the emitter rejects spin-blocked manifold names
  (`Unknown manifold 'singles_aa'`), and recognition finds zero operators there.
  Needs O6 in `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` first.
- The six failing selection-model gates are unrelated and pre-existing.

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
