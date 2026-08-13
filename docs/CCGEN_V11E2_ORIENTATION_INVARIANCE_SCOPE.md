# V1.1e.2 — making `ucc_integrate_term_antisym` orientation-invariant

> **LANDED — design history.** Status lives in [`CCGEN_DRESSED_KERNEL_COMPLETION.md`](CCGEN_DRESSED_KERNEL_COMPLETION.md); read that
> first. This document is kept for the reasoning behind specific choices (including the
> wrong turns), not as a statement of current state.


Scopes **V1.1e.2**: make the spin adapter's output depend only on *what the algebra
is*, not on *how it is written*. This is route (b) from
`CCGEN_V11_SPEC_ADAPTATION_SCOPE.md` — the principled fix, chosen over normalizing
`v` at the dress/adapt boundary so the codebase does not accumulate a per-caller
patch for a defect that lives in one function.

**Fully root-caused with a minimal reproducer.** Everything below is measured against
the current tree.

---

## The defect, exactly

Two writings of the same integral, taken verbatim from the dressed-CCSD doubles
manifold (the expansion side vs the raw side of the same ERI-canonical key):

```
expansion:      v(k,b,c,j) t2(a,c,i,k)
raw:         -1 v(j,c,k,b) t2(a,c,i,k)
```

These are the same term. `v(j,c,k,b)` is `v(k,b,c,j)` bra↔ket-exchanged
(`(2,3,0,1)` → `(c,j,k,b)`) and then bra-swapped to `(j,c,k,b)`; for antisymmetric
`v` the swap costs −1, which is exactly the `−1` the raw side carries. GCC agrees:
under `_eri_canonical` the two manifolds match with **0 mismatches**.

The adapter does not agree. Integrating the two writings on the closed-shell
representative block gives coefficient sums **2 vs 0**.

### Why

`_line_pairs` pairs slot `k` with slot `k+n` — the physicist `<pq|rs>` convention,
where `p–r` and `q–s` are the interaction lines. Read positionally:

```
v(k,b,c,j)  lines:  k–c ,  b–j      (occ–vir, occ–vir)
v(j,c,k,b)  lines:  j–k ,  c–b      (occ–occ, vir–vir)
```

So the two writings present **different line structures** to the adapter. Per spin
case (all 16, externals `k,b,c,j`):

| spins | `v(k,b,c,j)` | `v(j,c,k,b)` |
|---|---|---|
| `ka ba ca ja` | `+1 aaaa` | `+1 aaaa` |
| `ka bb ca jb` | `+1 abab` | **`−1 abab`** |
| `ka bb cb ja` | `−1 abab` | **`+1 abab`** |
| `kb ba ca jb` | `−1 abab` | **`+1 abab`** |
| `kb ba cb ja` | `+1 abab` | **`−1 abab`** |
| `kb bb cb jb` | `+1 bbbb` | `+1 bbbb` |
| (10 others) | `None` | `None` |

**The same cases survive, with the same blocks — but the four mixed-spin cases carry
opposite signs.** Combined with the raw side's `−1` term coefficient the two are
consistent; the adapter loses that because `_antisym_to_allowed` re-derives its sign
from the *written* slot order via `_permutation_parity`, treating the two writings as
independent inputs rather than one integral.

### What is NOT the cause

Ruled out by measurement, so the fix does not go looking there:

- **Not the bra↔ket exchange alone.** A 256-case sweep (all occ/vir slot patterns ×
  all spin labels) comparing a `v` factor against its `(2,3,0,1)` exchange found
  **0 divergences**: exchange maps lines `p–r, q–s` to `r–p, s–q`, the same lines, so
  it never changes a verdict. The defect needs exchange **composed with a within-group
  swap**, which is what re-pairs the lines.
- **Not `merge_terms`, and not the collapse.** Adaptation is additive over term
  partitions (0 mismatches splitting doubles in half — gated by V1.1e.1).
- **Not the dressed operators.** The GCC dressed assembly is exact (0 on every
  manifold). This is a pre-existing adapter property that dressing merely *exposes*,
  because dressing is the first thing to feed the adapter a differently-written form
  of an equation it already handles.

**Consequence worth stating plainly:** this defect is not introduced by V1. It is
latent in `ucc_integrate_term_antisym` today, and any future caller that writes its
`v` factors differently from the diagram generator will hit it. That is the argument
for route (b).

---

## The invariant to establish

> For any two `AlgebraTerm`s that are equal as algebra — i.e. equal under
> `_eri_canonical`, the 8-fold ERI symmetry fold — `ucc_integrate_term_antisym` must
> produce spin-integrated results with equal total coefficient per (block, spatial
> content).

Note what this does *not* demand: identical `SpinTerm` lists. Two writings may
legitimately produce different intermediate representatives; what must match is the
integrated contribution, since that is what reaches `merge_terms` and the emitted
kernel.

---

## Steps

### e.2.0 — pin the reproducer as a failing test (~S)

Encode the measured pair as a unit test *before* touching the adapter:

```
v(k,b,c,j) t2(a,c,i,k)      integrates to  2
-1 v(j,c,k,b) t2(a,c,i,k)   integrates to  0     ← must become equal
```

Plus the 16-case sign table above, asserting the four mixed cases currently disagree.

*Gate:* the test fails on today's tree for the stated reason (sign divergence, not
survival divergence). Mark it `expectedFailure` only long enough for e.2.1–e.2.3 to
land in separate commits; a permanently-xfailed reproducer is how a known defect
becomes folklore.

**Why first:** it converts "doubles = 14" into a two-term assertion that a fix can be
checked against in milliseconds, and it pins the *mechanism* (signs, not survival) so
a fix that accidentally changes which cases survive is caught.

### e.2.1 — canonicalize `v` orientation *inside* `_antisym_to_allowed` — **LANDED**

`_orientation_normalized` reorients every rank-4 `v` to one canonical member of its
8-fold ERI orbit before the lines are read, folding that reorientation's parity into
the returned sign. Reuses `_ERI_PERMUTATIONS` / `_perm_parity` from `dressing.py`.
Reproducer holds: both writings integrate to 0 (were 2 and 0).

**The wrong turn, recorded because the scope recommended it.** The direction note below
preferred a name-independent key (`(space, spin)`) over relying on index names. That was
tried first and **is wrong**: the key is degenerate — several orbit members tie on it —
so the tie-break decides the representative and the two writings land on *different*
ones. Measured, it made all 6 surviving spin cases disagree instead of 4, i.e. strictly
worse than no fix. Names are what break the tie deterministically, so the landed fix
uses the same lexicographic `(space, name)` rule `_eri_normalize_factor` uses. Within a
single term that is sound by construction: both writings of one integral carry the same
dummy names, only arranged differently. The documented constraint (this normalization is
only meaningful on consistently-named indices) is therefore *accepted*, not circumvented.

**Per-factor verdicts still differ, and that is correct.** `v(j,c,k,b) = −v(k,b,c,j)` as
a factor; the raw term's own `−1` compensates. The invariant is per-**term**. What did
change per factor is that the flip is now **uniform** across all 6 surviving cases (the
orbit parity) rather than the pre-fix inconsistent 4-of-6 — the signature of a clean
global reorientation, and the gate asserts that uniformity.

**Unexpected side effect, verified benign.** The spatial emit *shrank*: 73260 → 65431
bytes (doubles 120 → 113 terms). With the adapted residual multisets identical
before/after (0 mismatched keys on every manifold), this is the normalization merging
orientation-duplicate terms — the same answer in fewer terms.

### e.2.1 — original scoping (retained for the rationale)

Make the function normalize each `v` factor's orientation before reading its lines,
folding the antisymmetry sign of that normalization into its returned sign. The
existing `_eri_normalize_factor` in `dressing.py` already does exactly this
(lexicographically-smallest `(space, name)` arrangement over `_ERI_PERMUTATIONS`,
returning the parity) — **reuse it; do not write a second ERI normalizer.**

Two constraints, both already documented in the tree and both load-bearing:

1. **Normalization must run on canonical index names.** `_eri_canonical`'s docstring
   records this as a hard-won ordering (`D7.2.5.2 Fmi`): the fold picks the
   lexicographically smallest arrangement, so two terms that are the same integral but
   carry differently-named dummies normalize to *different* orientations and never
   fold. Inside the adapter the indices are the caller's, not canonical — so either
   relabel first, or key the normalization on something name-independent (slot spaces
   + which slots share a line). **This is the crux of e.2.1 and the most likely place
   to get it wrong.**
2. **The sign must compose correctly with the existing within-group parity.**
   `_antisym_to_allowed` already returns
   `_permutation_parity(bra_order) * _permutation_parity(ket_order)`. The
   normalization parity multiplies that. Getting the composition wrong trades 14
   mismatches for wrong signs — strictly worse, because wrong signs are silent
   whereas a mismatch count is visible.

*Gate:* e.2.0's reproducer passes; the 16-case table agrees between the two writings.

**Direction note:** a name-independent orientation key (constraint 1's second option)
is preferable to relabeling inside the adapter. Relabeling is stateful and would
duplicate `canonicalize.py`'s job in a hot path; the line structure is what the
adapter actually reads, so keying on it is both cheaper and more honest about the
invariant.

### e.2.2 — re-gate the validated adapter — **LANDED**

**The numeric gates were silently skipping.** In the default interpreter every pyscf
gate in `test_spin` reports `skipped 'pyscf not importable'` — so the "93 tests OK" of
earlier steps never exercised S1/S2/S4 or the FCI-limit fixtures. pyscf 2.13.0 lives in
`tests/pyscf/.venv`; run through it the gates execute.

**Record this: validate the adapter with
`tests/pyscf/.venv/bin/python -W ignore -m unittest ...`, not the default interpreter.**
A green default-interpreter run is not evidence for anything in this area.

Results, with the pre-change baseline captured by stashing so the comparison is real:

| gate | baseline | with fix |
|---|---|---|
| `test_spin` (pyscf) | 93 OK | 93 OK |
| adapted residual multiset (energy/singles/doubles) | — | 0 mismatched keys |
| spatial emit | 73260 | 65431 (fewer terms, same multiset) |
| `test_spin_orientation` + `test_spin` + `test_dress_adapt` + `test_dressed_equation` + `test_dressing` | — | 267 OK |

### e.2.5 — the residue is NOT a defect at all — **RESOLVED, see `CCGEN_V11E25_RESIDUE_SCOPE.md`**

**This section's diagnosis below is superseded.** It was written before the residue was
probed numerically, and its leading hypothesis (the closed-shell collapse's Cartesian
product over multiple collapsible factors) is **wrong**. Retained because the ruling-out
list is still valid and because the wrong turn is worth recording.

What probing found: with a **symmetry-correct** `v`, the adapted dressed and adapted raw
residuals agree to **~1e-14 on every manifold**, across three `(no, nv, seed)` triples.
V1.1e's requirement was already met. The 14 symbolic "mismatches" are an artifact of
comparing *written forms* — a term-by-term multiset comparison cannot see that two sides
picked different, symmetry-equivalent writings of the same algebra.

The real defect was in the **test fixture**: `residual_eval.random_tensors` built `v`
with intra-pair antisymmetry only, violating `<pq||rs> = <rs||pq>` (residual 2.35, where
real integrals give ~1e-16 — checked against pyscf on H2/STO-3G). Since
`_ERI_PERMUTATIONS` folds by that symmetry, any numeric comparison of two exchange-related
writings reported a spurious difference. Fixed in e.2.5.0; gate added in e.2.5.1.

Also corrected: the collapse is **not** implicated — on the minimal reproducer both sides
sum to the same coefficient after the full pipeline. The factor-count signature
(`t1t1v`, `t2t2v`, `t1t1t1t1v`) was real but incidental: those terms simply have the most
written forms, hence the most opportunities to differ in choice.

### e.2.5 — original diagnosis (SUPERSEDED, retained for the ruled-out list)

**e.2.1 was necessary but not sufficient.** The dressed-vs-raw adapted doubles residual
is still **14 mismatches, completely unchanged** by the orientation fix, and still
carries the repeated-same-name-factor signature:

```
4 x (t1, t1, t1, v)      2 x (t1, t1, v)
4 x (t2, t2, v)          2 x (t1, t1, t1, t1, v)
2 x (t1, t1, t2, v)      onlyD 7, onlyR 6
```

So the earlier draft's collapse-commutation hypothesis — which this document
demoted to "the symptom, not the cause" when the orientation mechanism was found —
is **back in scope as a genuinely separate defect.** Orientation sensitivity was real,
measurable, and is now fixed; it simply was not what produced the 14.

What is now known, and narrows e.2.5 considerably:

- Not orientation (fixed; residue unchanged).
- Not additivity (gated in e.1).
- Not the dressed assembly (GCC exact, 0 on every manifold since e.0).
- Not the adapter's spatial semantics (multisets identical before/after e.2.1).

That leaves the closed-shell collapse's Cartesian product over **multiple collapsible
factors** (`collapse_amplitudes` / `collapse_integrals` / `_product_over_choices`) as
the remaining candidate, which is exactly what the factor signature points at: a term
with *k* collapsible factors expands into 2^k spatial terms, and a dressed term hides
some of its collapsible factors inside `W`/`τ` — so the two sides carry different *k*
for the same algebra.

Do **not** restate that as established. It is the leading hypothesis with the other
four ruled out; e.2.5 starts by finding a minimal two-writing reproducer that differs
in collapsible-factor count, the way e.2.0 did for orientation.

### e.2.2 — original scoping (retained)

`ucc_integrate_term_antisym` is load-bearing for the whole spin-adaptation stack, so
its own validation must be re-run, not just the dressing tests:

- `test_spin` in full — the S1/S2/S4 numeric gates, including the rank-6 and rank-8
  FCI-limit fixtures.
- The spatial-emit regressions (`--spin-adapt`), byte-compared. A spatial kernel whose
  emitted text moves means the adapter changed answers on the *existing* path, which
  would be a regression, not a fix.
- The **Be CCSDTQ == FCI** acceptance. This is the strongest gate in the effort and it
  runs through this exact function.

*Gate:* all unchanged. If the spatial emit moves, stop — the fix changed the validated
path and needs to be understood before proceeding.

### e.2.3 — update the V1.1e.1 residue assertion — **no change needed yet**

`AdaptedExpansionOrderTests.test_expansion_order_is_pinned` asserts `{"doubles": 14}`
deliberately. e.2.1 left the count at exactly 14, so the assertion still holds
unchanged — and it did its job: it is *why* we know the orientation fix did not close
V1.1e, rather than assuming it had.

This is the branch the original scoping anticipated: "if it lands somewhere between 0
and 14, that is a second, distinct defect: record it with its own reproducer rather
than relaxing the assertion." It landed at 14, i.e. the residue is entirely a distinct
defect — recorded as **e.2.5**. Update this assertion when e.2.5 lands, not before.

Keep the rejected adapt-then-verify contrast in that test. Its numbers will also move;
re-measure rather than delete, since the ordering rationale survives the fix.

### e.2.4 — no double normalization — **SATISFIED**

Exactly one `v`-orientation fold ships, and it is inside the adapter
(`_orientation_normalized`, called from `_antisym_to_allowed`). No boundary pre-pass was
added: e.1 pinned the expansion order but introduced no normalization, and e.2.1 did not
need one. So every caller of the adapter gets orientation invariance for free, which was
the whole point of route (b) over route (a).

The reused pieces (`_ERI_PERMUTATIONS`, `_perm_parity`) are *imported* from
`dressing.py`, not copied — so the 8-fold group and its parity have one definition in
the tree. `_eri_normalize_factor` remains a separate function because it operates on
`Tensor` for the canonical-key path; it shares the group and the lexicographic rule with
`_orientation_normalized` but not the call site. If a third consumer appears, extract the
shared representative-picking step rather than adding a third copy.

---

## Sequencing

```
e.2.0 (failing reproducer)              LANDED
   └→ e.2.1 (normalize in the adapter)  LANDED  ← orientation invariance
        ├→ e.2.2 (re-gate via the pyscf venv)  LANDED  (93 OK, multisets identical)
        ├→ e.2.3 (residue assertion)    no change — still 14, which is the finding
        ├→ e.2.4 (no double normalization)  satisfied: one fold, in the adapter
        └→ e.2.5 (the 14 are a comparison artifact, not a defect)  RESOLVED
             │      fixture fixed + numeric gate; own doc
             ▼
        e.3 (numeric per-operator localization)  LANDED → V1.1f (~S, NEXT) → V1.2
```

**Honest status.** e.2.1 fixed a real, measured, latent defect in
`ucc_integrate_term_antisym` — one that would have bitten any future caller writing its
`v` factors differently, which is the argument that made route (b) right.

It did not close V1.1e *as measured by the symbolic count*, and I initially read that as
a second defect. It was not: e.2.5 showed the algebra was already correct and the
**symbolic count was the wrong instrument**. So e.2.1 was both necessary and sufficient
for the algebra.

The deliberately-exact residue assertion still earned its keep — it stopped me declaring
victory on e.2.1 — but it then pointed at a defect that did not exist, which is the cost
of pinning a proxy rather than the property. e.2.5.2 replaces it accordingly.

---

## What this reuses

| Reused | From |
|---|---|
| The 8-fold ERI symmetry group and its parity | `_ERI_PERMUTATIONS`, `_perm_parity` (`dressing.py`) |
| `v` orientation normalization + sign | `_eri_normalize_factor` — reuse, do not reimplement |
| The ordering-must-run-on-canonical-names constraint | `_eri_canonical`'s docstring (`D7.2.5.2 Fmi`) |
| Within-group antisymmetry mapping | `_antisym_to_allowed`'s existing bra/ket multiset logic |
| Additivity precondition | V1.1e.1's gate |
| Adapter validation | `test_spin` S1/S2/S4, Be CCSDTQ == FCI |

**Net new:** an orientation-normalization step inside `_antisym_to_allowed` and its
sign composition. No new symmetry group, no new normalizer, no per-caller patch.

---

## What NOT to do

- **Do not normalize at the dress/adapt boundary instead.** That was route (a). It
  fixes the one caller that currently hurts and leaves the adapter write-order
  sensitive for the next one — the spaghetti outcome this route rejects.
- **Do not reimplement the ERI fold.** `_eri_normalize_factor` exists, carries the
  parity correctly, and its sign handling was itself a bug fix (pre-D7.2.5 discarded
  the sign and silently rejected correct `Fae`/`Wabef` hypotheses). A second copy will
  drift.
- **Do not change `_line_pairs`.** The slot-`k`/slot-`k+n` pairing *is* the physicist
  convention and the C++ runtime's layout contract (`02364db`). The writing gets
  normalized to fit it, not the reverse.
- **Do not relax the `{"doubles": 14}` assertion to make progress visible.** It exists
  to force a deliberate update.
- **Do not accept "the comparison agrees" as the gate.** Folding with
  `_eri_canonical` at compare time makes the *verifier* agree while the *adapted
  output* still depends on writing — and the adapted output is what ships.
- **Do not skip e.2.2.** This function is upstream of the Be CCSDTQ == FCI gate. A fix
  that quietly changes the existing spatial path is a regression wearing a fix's
  clothes.

---

## Risk

The fix touches validated, load-bearing code — deliberately, since that is where the
defect is. Two specific failure modes, both silent:

1. **Wrong sign composition** (e.2.1 constraint 2). Mitigated by e.2.0's 16-case sign
   table, which pins signs per spin case rather than only totals.
2. **Name-dependent normalization** (e.2.1 constraint 1). Mitigated by the direction
   note (key on line structure, not names) and by e.2.2's requirement that the
   existing spatial emit be byte-identical — a name-dependent fold would move it.

If e.2.1 cannot be made name-independent, **stop and reconsider route (a)** rather
than relabeling inside the adapter. Route (a) is the worse design but a known
quantity; a stateful relabel in the adapter's hot path is a third thing, worse than
both.

---

See `CCGEN_V11_SPEC_ADAPTATION_SCOPE.md` (V1.1e.0/e.1 landed, e.3 next),
`CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md` (V1.0's sibling slot-ordering defect — same
class, same root: the adapter reading slot position rather than structure), and
`CCGEN_SPIN_ADAPTATION_SCOPE.md` (the S1/S2/S4 gates e.2.2 must preserve).
