# V1.1e.2 — making `ucc_integrate_term_antisym` orientation-invariant

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

### e.2.1 — canonicalize `v` orientation *inside* `_antisym_to_allowed` (~M, the fix)

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

### e.2.2 — re-gate the validated adapter (~S, non-negotiable)

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

### e.2.3 — update the V1.1e.1 residue assertion (~S)

`AdaptedExpansionOrderTests.test_expansion_order_is_pinned` asserts
`{"doubles": 14}` deliberately, so a partial fix cannot pass silently. Update it to
the post-fix count — **0 if e.2.1 is complete.** If it lands somewhere between 0 and
14, that is a second, distinct defect: record it with its own reproducer rather than
relaxing the assertion.

Keep the rejected adapt-then-verify contrast in that test. Its numbers will also move;
re-measure rather than delete, since the ordering rationale survives the fix.

### e.2.4 — retire the boundary workaround, if any exists (~S)

None exists yet — e.1 pinned the order but added no `v` normalization at the boundary.
This step is a checkpoint: if e.2.1 turns out to need a boundary pre-pass after all,
**say so and delete the in-adapter half** rather than shipping both. Two overlapping
normalizations is precisely the spaghetti this route was chosen to avoid.

---

## Sequencing

```
e.2.0 (failing reproducer, ~S)
   └→ e.2.1 (normalize inside _antisym_to_allowed, ~M)   ← the fix
        ├→ e.2.2 (re-gate test_spin + spatial emit + Be CCSDTQ==FCI, ~S)
        ├→ e.2.3 (update the pinned residue to 0, ~S)
        └→ e.2.4 (checkpoint: no double normalization, ~S)
             │
             ▼
        e.3 (per-operator localization) → V1.1f → V1.2
```

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
