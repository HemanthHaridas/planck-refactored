# V1.1e.2.5 — the doubles=14 residue is a comparison artifact, not a defect

> **LANDED — design history.** Status lives in [`CCGEN_DRESSED_KERNEL_COMPLETION.md`](CCGEN_DRESSED_KERNEL_COMPLETION.md); read that
> first. This document is kept for the reasoning behind specific choices (including the
> wrong turns), not as a statement of current state.


Scopes **e.2.5**, the residue left after the orientation fix. Probing resolved it before
scoping, and the answer inverts the premise: **the adapted dressed residual is already
correct.** The 14 symbolic "mismatches" are an artifact of comparing *written forms*
where two forms differ by a `v` symmetry the comparison folds but the term-by-term
multiset does not.

So e.2.5 is not a fix. It is: replace the wrong gate with a right one, and record the
trap so it is not re-litigated.

**Everything below is measured.** The investigation also found a real defect — in a test
fixture, not in ccgen — which is the load-bearing finding here.

**Status: e.2.5.0 and e.2.5.1 landed.** The fixture is fixed and the numeric gate passes
at ~1e-14 on all three manifolds, so V1.1e's requirement is demonstrably met. e.2.5.2
(retire the `{"doubles": 14}` assertion) and e.2.5.3 (mark V1.1e closed) remain — both
gate hygiene, no algebra involved.

---

## The resolution

With a symmetry-correct `v`, the adapted dressed and adapted raw residuals agree to
**machine precision on every manifold**:

| no, nv, seed | energy | singles | doubles |
|---|---|---|---|
| 2, 3, 0 | 0.00e+00 | 2.66e-15 | 1.42e-14 |
| 3, 4, 11 | 0.00e+00 | 1.07e-14 | 3.20e-14 |
| 4, 5, 3 | 0.00e+00 | 1.78e-14 | 6.44e-14 |

The GCC pre-adaptation comparison is equally clean (singles 8.9e-16, doubles 8.0e-15).

**V1.1e's actual requirement — that the dressed spatial equation equals the raw spatial
equation — is therefore already met.** The `{"doubles": 14}` count is a property of the
symbolic multiset comparison, not of the algebra.

---

## The real defect found: `residual_eval.random_tensors` violates `<pq||rs> = <rs||pq>`

This is the finding worth keeping, and it is a live hazard for every numeric gate built
on that fixture.

`ccgen/tests/residual_eval.py::random_tensors` builds `v` with intra-pair antisymmetry
only:

```
bra antisym  max|v + v(1,0,2,3)| = 0.0     ✓
ket antisym  max|v + v(0,1,3,2)| = 0.0     ✓
bra<->ket    max|v - v(2,3,0,1)| = 2.35    ✗   should be 0
```

Real antisymmetrized integrals **do** satisfy the exchange. Checked against pyscf on
H2/STO-3G (`<pq||rs>` built as `(pr|qs) − (ps|qr)`):

```
bra antisym  3.03e-16      ket antisym  0.0      bra<->ket  2.22e-16
```

`_ERI_PERMUTATIONS` (`dressing.py`) includes `(2,3,0,1)` precisely because that symmetry
is real — its comment records it as load-bearing for reconciling the textbook operator
definitions' `<oo||ov>` arrangement with the residual's `<ov||oo>` (the "A3.2 wall").

**Consequence: any numeric test using `random_tensors` cannot distinguish two forms that
are equal only under the bra↔ket exchange, and will report spurious differences.** That
is exactly what happened here — with the raw fixture the "GCC expansion vs GCC raw"
comparison showed a 170% relative difference on a pair the symbolic fold (correctly)
calls identical.

Repair is four lines, applied to `v` after construction:

```python
V = 0.5 * (V + V.transpose(2, 3, 0, 1))     # impose bra<->ket
V = 0.5 * (V - V.transpose(1, 0, 2, 3))     # restore bra antisym
V = 0.5 * (V - V.transpose(0, 1, 3, 2))     # restore ket antisym
V = 0.5 * (V + V.transpose(2, 3, 0, 1))     # re-impose (the antisym steps perturb it)
```

The re-imposition on line 4 is required — verified, not decorative: the two antisym
projections do not commute with the exchange projection, so a single pass leaves a
residual.

---

## What this says about the earlier diagnosis

Two of my own earlier conclusions were wrong and are corrected here rather than left in
the history:

1. **"The residue is the collapse's Cartesian product over multiple collapsible
   factors."** The factor-count signature (`t1t1v`, `t2t2v`, `t1t1t1t1v`) was real but
   incidental — those are simply the terms with the most written forms, hence the most
   opportunities for two sides to pick different ones. Traced through
   `collapse_amplitudes` / `_product_over_choices`, the collapse is *not* implicated: on
   the minimal reproducer both sides sum to the same coefficient (1) after the full
   pipeline.
2. **"e.2.1 was necessary but not sufficient."** Half right. It was necessary and it *is*
   sufficient for the algebra; what remained was never an algebra defect. The
   orientation fix stands on its own merits (it fixed a measured latent defect and shrank
   the emit by merging duplicate terms), but it was not "incomplete".

The `{"doubles": 14}` assertion did its job — it stopped me concluding the fix had closed
V1.1e — but it then pointed at a defect that did not exist. **The lesson for the gate
design: a symbolic term-by-term multiset comparison is the wrong instrument when the two
sides are free to choose among symmetry-equivalent written forms.** A numeric gate on
symmetry-correct tensors is the right one.

---

## Steps

### e.2.5.0 — fix `random_tensors` — **LANDED**

`ccgen/tests/residual_eval.py` now imposes the bra↔ket exchange on `v`, and `f = fᵀ`
(real Fock matrices are symmetric; the fixture's was not).

The re-imposition after the antisymmetry projections is **required, not defensive**: the
two intra-pair projections do not commute with the exchange projection, so a single pass
leaves a residual. Verified — all three `v` residuals and the `f` residual are exactly
`0.0` across the three test sizes.

*Risk that did not materialize:* the scope warned a gate might be tuned to the broken
fixture. None was. Every consumer (`test_diagram`, `test_regressions`,
`test_reference_vs_pyscf`, `test_spin`) passes unchanged, as does the full 696-test
suite. No test depended on the missing symmetry.

### e.2.5.1 — numeric gate for the adapted dressed equation — **LANDED**

`ccgen/tests/test_residual_symmetry.py` (4 tests) pins both halves:

- **`FixtureSymmetryTests`** — all three `v` symmetries plus `f` symmetry, over three
  `(no, nv, seed)` triples, so the fixture cannot silently regress.
- **`AdaptedDressedNumericTests`** — the V1.1e gate. `test_gcc_expansion_matches_raw` is
  the precondition (the dressed assembly is exact before adaptation);
  `test_adapted_dressed_matches_adapted_raw` is the requirement itself.

Measured, relative to the raw manifold's scale:

| no, nv, seed | energy | singles | doubles |
|---|---|---|---|
| 2, 3, 0 | 0.00e+00 | 2.66e-15 | 1.42e-14 |
| 3, 4, 11 | 0.00e+00 | 1.07e-14 | 3.20e-14 |
| 4, 5, 3 | 0.00e+00 | 1.78e-14 | 6.44e-14 |

Three triples rather than one, so a single lucky fixture cannot pass it.

### e.2.5.2 — retire the `{"doubles": 14}` assertion, deliberately (~S)

`AdaptedExpansionOrderTests.test_expansion_order_is_pinned` asserts the exact symbolic
count. Now that the numeric gate exists and passes:

- Keep the **relative** claim: expand-then-adapt is closer than adapt-then-verify. That
  is the ordering rationale, and it survives.
- Replace the exact `14` with a comment recording that the symbolic count is *not* a
  correctness measure and pointing at the numeric gate, so nobody "fixes" 14 → 0 again.
- Do **not** simply delete the test. The ordering choice still needs pinning.

*Gate:* the ordering test still fails if the orders are swapped.

### e.2.5.3 — close V1.1e (~S)

With e.2.5.1 green, V1.1e's requirement is met: the dressed spatial equation reproduces
the raw spatial equation. Mark V1.1e satisfied, note that the passing gate is numeric
rather than symbolic and why, and proceed to **e.3** (per-operator localization) —
which should now be recast as *numeric* per-operator localization for the same reason.

---

## Sequencing

```
e.2.5.0 (fix random_tensors)   LANDED   ← value independent of V1.1e
   └→ e.2.5.1 (numeric gate)   LANDED   ← V1.1e's requirement, passing at ~1e-14
        └→ e.2.5.2 (retire the 14, ~S)   NEXT
             └→ e.2.5.3 (close V1.1e, ~S)
                  │
                  ▼
             e.3 (numeric per-operator localization) → V1.1f → V1.2
```

All ~S. There is no ~M here, because there is no defect to fix in ccgen.

## Test status after e.2.5.0/e.2.5.1

Run through `tests/pyscf/.venv/bin/python` (pyscf 2.13.0) — the default interpreter
skips every numeric gate:

| suite | result |
|---|---|
| full `ccgen/tests` discover | **696 tests OK** (2 skipped, 4 expected failures) |
| `test_diagram` + `test_regressions` | 138 OK (1 skipped) |
| `test_reference_vs_pyscf` + `test_spin` | 125 OK (1 skipped, 1 expected failure) |
| `test_residual_symmetry` (new) | 4 OK |

The 4 expected failures are pre-existing and documented elsewhere
(`test_ccsd_spatial_energy_raw_is_wrong`, `test_parallel_generation_matches_serial`, and
two in `test_tau`) — none are related to this work.

---

## What NOT to do

- **Do not "fix" the 14.** There is nothing wrong. Changing the adapter or the collapse
  to drive a symbolic count to zero would be changing correct code to satisfy a wrong
  measurement — and the numeric gate would then have to be broken to keep it passing.
- **Do not delete the ordering test** with the `14`. The ordering rationale is still
  real; only the exact count is meaningless.
- **Do not weaken `random_tensors` back** if a test fails after e.2.5.0. A test that only
  passes on a `v` violating a symmetry real integrals have is the finding.
- **Do not trust `random_tensors`-based numeric comparisons of two written forms** until
  e.2.5.0 lands. That is what produced the false 170% GCC difference here.
- **Do not run the numeric gates in the default interpreter.** They skip (pyscf not
  importable); use `tests/pyscf/.venv/bin/python`. Carried over from e.2.2 and it
  matters again here.

---

## Honest summary

V1.1e's algebra was correct from e.2.1 onward, and arguably before it for this particular
comparison. Of the four steps in this document, three are gate hygiene and one
(`e.2.5.0`) fixes a genuine, previously-unnoticed defect in a shared test fixture that
undermines any numeric comparison of symmetry-equivalent forms. That fixture bug is the
substantive output of e.2.5.

---

See `CCGEN_V11E2_ORIENTATION_INVARIANCE_SCOPE.md` (e.2.0–e.2.4, and the e.2.5 stub this
supersedes), `CCGEN_V11_SPEC_ADAPTATION_SCOPE.md` (V1.1e's requirement and V1.1f/V1.2
after it), and `dressing.py::_ERI_PERMUTATIONS` (the symmetry the fixture is missing).
