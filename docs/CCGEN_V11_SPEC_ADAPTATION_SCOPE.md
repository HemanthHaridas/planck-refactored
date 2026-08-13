# V1.1 — adapting the dressed intermediate specs, in small verifiable steps

> **LANDED — design history.** Status lives in [`CCGEN_DRESSED_KERNEL_COMPLETION.md`](CCGEN_DRESSED_KERNEL_COMPLETION.md); read that
> first. This document is kept for the reasoning behind specific choices (including the
> wrong turns), not as a statement of current state.


Scopes **V1.1** of `CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md`: route each dressed
operator's `IntermediateSpec` through the spin adapter so the emitted
`build_<op>` computes a **spatial** operator that the adapted residual can index.

V1.1 is the only ~M step in V1. V1.0 (the slot-ordering contract) is **landed**,
which is what makes V1.1's gate reachable at all.

**Measured against the current tree.** Four probes below changed the shape of this
step relative to how the parent scope predicted it — including one prediction of
mine that was wrong, and one defect that turns out to be narrower than feared.

---

## Measured findings

### Finding A — only **three** operators need adapting, not six

`_dress_operator_equations` recognizes all six CCSD operators, but the assembled
dressed CCSD residual references only:

```
emitted specs : tau, tau_c, Wmnij, Wabef, Wmbej
referenced    : tau, tau_c, Wmnij, Wabef, Wmbej
```

`Fme` / `Fae` / `Fmi` recognize but **are not referenced** — under canonical Fock
their contributions fold elsewhere (`Fme` collapses to its `t1*oovv` piece; `Fae`
/ `Fmi` lose their `f_ov*t1` corrections), and the spec list is built from what
the residual actually references. So V1.1's surface is **three W operators plus
two τ pseudo-amplitudes**, not six operators.

This shrinks the work and sharpens the gates. It also means `Wmbej` — the hardest
operator, and the one V1.0 had to fix — is unavoidably in scope.

### Finding B — the spec/term index-order mismatch is real, and I had it wrong twice

Adapted free-index order versus `spec.indices`, on each operator's own slot order
(post-V1.0):

| operator | sig | `spec.indices` | adapted `free_indices` | agree |
|---|---|---|---|---|
| Fme | ov | `[m,e]` | `[m,e]` | yes |
| Fae | vv | `[a,e]` | `[a,e]` | yes |
| Fmi | oo | `[m,i]` | `[i,m]` | **no** |
| Wmnij | oooo | `[m,n,i,j]` | `[i,j,m,n]` | **no** |
| Wabef | vvvv | `[a,b,e,f]` | `[a,b,e,f]` | yes |
| Wmbej | ovvo | `[m,b,e,j]` | `[j,m,b,e]` | **no** |

The parent scope predicted the two mixed-space operators (`Fme`, `Wmbej`) would be
the mismatched ones. **That was wrong.** The actual set is `Fmi`, `Wmnij`,
`Wmbej` — and `Fme`, the other mixed-space operator, *agrees*. The mismatch tracks
the adapter's relabeling, not space homogeneity. Of the three, only `Wmnij` and
`Wmbej` are referenced (Finding A), so **two** specs actually need reordering.

### Finding C — the GCC dressed emit is self-consistent *despite* the mismatch

This is the finding that de-risks V1.1, and it corrects a worry the parent scope
raised.

`_emit_intermediate_builder` does **not** shape the builder from `spec.indices`:

```python
lowered_definition_terms = tuple(lower_term_restricted_closed_shell(t, "reference")
                                for t in spec.definition_terms)
builder_indices = (lowered_definition_terms[0].canonical_free_indices
                   if lowered_definition_terms else spec.indices)
```

`canonical_free_indices` comes from `_stable_spatial_indices`, which sorts
occ-before-vir with a stable tiebreak. The **consumer** side (`_map_factor` →
`_target_expr`) emits the factor's indices at the usage site, and those went
through the same lowering. So both ends normalize identically. Verified in the
emitted C++ for the three referenced operators:

```
Wmnij  alloc=(no,no,no,no)  uses: Wmnij(k, l, i, j)
Wabef  alloc=(nv,nv,nv,nv)  uses: Wabef(a, b, c, d)
Wmbej  alloc=(no,no,nv,nv)  uses: Wmbej(j, i, a, b), Wmbej(k, i, a, c)
```

`Wmbej`'s builder allocates `(o,o,v,v)` — the lowered order, not its `ovvo` spec
order — and every usage site indexes `(o,o,v,v)`. Consistent.

**So `spec.indices` is metadata, not the emitted layout.** The mismatch is a
latent trap rather than a live miscompile: the emit path routes around it because
lowering is applied on both sides. That means V1.1's job is narrower than "fix a
broken layout" — it is *keep the two sides normalizing identically after
adaptation*, and make the metadata honest so nothing downstream (V1.4's dependency
order, V5's block-keying, any future consumer that trusts `index_space_sig`) is
misled.

**Do not "fix" this by forcing `spec.indices` into the builder.** That would
*create* the miscompile that currently does not exist.

### Finding D — the adapter multiplies terms, so builders get bigger

Adapted term counts on each operator's own order (post-V1.0):

```
Fme    1 -> 2      Wmnij  4 -> 4
Fae    3 -> 8      Wabef  4 -> 4
Fmi    3 -> 5      Wmbej  5 -> 8
```

`Wmbej` 5→8 and `Fae` 3→8 (~1.6–2.7×) — consistent with Decision 5's measured
1.8× on the CCSD doubles residual. For the three referenced operators the growth
is modest (`Wmnij`/`Wabef` unchanged, `Wmbej` 5→8), so the compile-time risk V3
must measure is small at CCSD. Worth re-measuring at CCSDT, where the operator set
and each builder both grow.

---

## What V1.1 must produce

```
_dress_operator_equations  →  (dressed_eqs, specs)
        │                            │
        │                            └─→ adapt each spec's definition_terms
        │                                on intermediate_template(spec)   [V1.0 hook]
        │                                re-derive indices + index_space_sig
        │                                block-key the name
        ▼                            ▼
   adapt residual  ────────────→  emit(equations=adapted, intermediates=adapted_specs)
```

Both sides must agree on **one** vocabulary of intermediate names and **one**
normalization of slot order.

---

## Steps

### V1.1a — adapt a spec's `definition_terms` (~S)

Add `adapt_intermediate_spec(spec, adapter)` returning a new `IntermediateSpec`
whose `definition_terms` are the adapter's output, using V1.0's
`intermediate_template(spec)` so the operator's own line pairing is preserved.

Keep `name`, `usage_count`, `usage_targets` for now (V1.1c renames; V1.1d
recounts). Carry `memory_layout` / `blocking_hint` / `allocation_strategy`
through unchanged.

*Gate:* each of the three referenced operators (`Wmnij`, `Wabef`, `Wmbej`) yields
a spec with non-empty `definition_terms`, at the counts in Finding D (4, 4, 8).
The V1.0 zero-guard already makes a silent vanish impossible, so this gate is
about the counts, not the emptiness.

### V1.1b — re-derive `indices` and `index_space_sig` from the adapted terms (~S)

Fix Finding B's metadata dishonesty. Derive both from the adapted terms' free
indices under **the same normalization the emitter uses** — i.e.
`_stable_spatial_indices`, reached via `lower_term_restricted_closed_shell`, not a
hand-rolled sort. Reusing the emitter's own normalizer is what keeps Finding C's
self-consistency intact by construction rather than by luck.

*Gate (the load-bearing one for this sub-step):* for every adapted spec, the
declared `indices` equal the builder's actual `builder_indices` — assert against
`lower_term_restricted_closed_shell(adapted_terms[0]).canonical_free_indices`,
which is literally what `_emit_intermediate_builder` computes. This is the
assertion whose absence let Finding B hide.

*Gate:* `index_space_sig` matches the space pattern of the re-derived `indices`
(e.g. `Wmbej` becomes `oovv`, not `ovvo`, if lowering reorders it — and the sig
must say so).

### V1.1c — block-key the spec identity (~S)

`IntermediateSpec.__hash__` / `__eq__` key on `(name, indices, index_space_sig)`.
Under RCC's reference sector each operator yields one adapted spec, so nothing
collides today. Under UCC one `Wmnij` becomes several spin-block variants and they
**would** collide.

Fold the block tag into the name — `Wmnij_abab` — using the **same** naming path
U1.1 uses for `t2_aaaa` and R3.1.3c already uses for `t4_aaabaaab`. One naming
mechanism for amplitudes, ERIs, and intermediates.

Build it now even though RCC needs one variant: `_map_factor` already resolves any
name in `intermediate_names` as a local, so a tagged name costs nothing on the
consumer side, and retrofitting during V5 means touching both sides again.

*Gate:* two distinct blocks of the same operator produce two specs that are
unequal and hash differently; the RCC reference sector's tag is either absent or
stable, so the RCC emit is byte-identical to V1.1b's output. That byte-identity
assertion is what keeps V1.1c from being a behavior change.

### V1.1d — recount usage against the adapted residual (~S)

`usage_count` / `usage_targets` are computed from the **GCC** residual in
`_dress_operator_equations`. After adaptation the counts change (Finding D:
adaptation splits terms). `usage_count` feeds the emitted comment today, but it is
also the natural input to any future materialize-once-vs-inline decision, so a
stale count is a trap.

Recount by scanning the adapted residual for each adapted spec name.

*Gate:* every referenced spec has `usage_count > 0` against the adapted residual,
and every name referenced by the adapted residual has a spec (no dangling
reference, no orphan spec). This bidirectional closure is cheap and catches a
rename that only landed on one side — the exact failure mode V1.1c could
introduce.

### V1.1e — the faithfulness gate (**PASSING**, e.0–e.3 all landed)

The parent scope's V1.1 gate. **Substituting an adapted spec's definition back into
the adapted dressed residual must reproduce the adapted raw residual.**

**Status: met.** Adapted dressed == adapted raw to ~1e-14 relative on energy/singles/
doubles across three `(no, nv, seed)` triples, per-operator and combined
(`test_residual_symmetry.py`, `test_dress_per_operator.py`).

**One correction to how this section is written.** The original phrasing — "term-by-term
after canonicalization" — is the instrument e.2.5 disproved. A symbolic term multiset
cannot distinguish *different algebra* from *the same algebra written in a
symmetry-equivalent form*, which is why it reported `{"doubles": 14}` while the algebra
was exact. **The gate is numeric residual equality on symmetry-correct tensors.** Treat
any symbolic count in this section as a diagnostic, never as the pass criterion.

`dressed_equation.py`'s `verify_dressed_equation(dressed, raw, operators)` is the
reuse surface — its `operators` dict is fully substitutable, so an adapted-definition
operator table drops straight in. **Reuse it; do not write a second verifier.**

#### Measured state

| configuration | singles | doubles |
|---|---|---|
| GCC baseline (no adaptation) | **0** | **0** |
| adapt-then-verify (adapted operator table) | 13 | 61 |
| expand-in-GCC-then-adapt | **0** | 14 |

The GCC baseline is now clean on every manifold: its one singles mismatch was a real
lost coefficient in `assemble_dressed_equation` (an all-or-nothing bare/dressed
partition dropping a partially-covered key), **fixed and gated** by
`PartialCoverageRemainderTests`. That fix also took expand-then-adapt singles 2 → 0.
So the whole residue is now **one number: doubles = 14**.

#### Root cause (measured, not hypothesized)

An earlier draft of this scope blamed the closed-shell collapse's Cartesian product
over repeated collapsible factors. **That was the symptom, not the cause.** Probing
established:

1. **In GCC, the dressed expansion and the raw residual are identical** — 0 mismatches
   under `_eri_canonical`. So the dressed assembly is exact, and the 14 appear *only*
   after adaptation.
2. **Adaptation is additive** over term partitions: `adapt(A+B) == adapt(A)+adapt(B)`,
   0 mismatches on a split-in-half doubles manifold. So the 14 are **not** a linearity
   or merge failure.
3. **The two manifolds are equal only under the ERI bra↔ket fold.** Comparing them
   with the plain `canonicalize_term` instead of `_eri_canonical`:

   ```
   WITHOUT the v bra<->ket fold : 72 mismatched keys
   WITH the fold (_eri_canonical):  0 mismatched keys
   ```

   The expansion writes 72 terms where raw writes 64; the extra 8 (all at 4 summed
   indices) are bra↔ket-flipped orientations of terms raw writes the other way, plus
   operator-internal dummies (`__Wmnij_e`, `__Wabef_m`).

**So: `_eri_canonical` folds a symmetry of the integral that `ucc_integrate_term_antisym`
does not.** The adapter assigns spin blocks by reading each `v` factor's *written*
slot orientation (`resolve_block` / `block_exists` key on slot position). Two writings
that are bra↔ket-equivalent — hence identical in GCC, where the fold is applied —
adapt to *different* spin-block sets. Adaptation is therefore a function of how the
algebra is **written**, not only of what it **is**.

Concrete instance found: one canonical key carries total ½ on both sides, written as
**3 terms** in the expansion (`½ t1t1t1t1v` + `¼ …` − `¼ …`, with `__Wmnij_e` /
`__Wabef_m` dummies) versus **1 term** in raw. Adapted in isolation each side gives 1
— they only diverge in the full manifold, where the differing orientations interact
with the spin enumeration.

This is the same class as V1.0's defect (the adapter reading slot position rather than
physical structure) and the same class as the R3.1.2 bridge bug. It is **not** a
dressed-operator problem, which is why it did not show up in D7.

#### Sub-steps

- **V1.1e.0 — clean GCC baseline. LANDED.** Was the `assemble_dressed_equation`
  partial-coverage defect; fixed, gated, and it took singles to 0 in both the GCC and
  the expand-then-adapt configurations.

- **V1.1e.1 — pin the expansion order. LANDED.** `expand_then_adapt` and
  `verify_adapted_dressed_equation` (`dressed_equation.py`) implement the chosen
  order; `AdaptedExpansionOrderTests` gates **both** configurations, so the choice
  cannot be silently revisited. The verifier returns per-manifold diffs, so a failure
  names `doubles` rather than "the equation".

  Two mechanical reasons for the order are asserted, not just the counts: expansion is
  what introduces the operator-internal dummies (`__Wmnij_e`, `__Wabef_m`), so doing it
  in GCC keeps them out of the adapter (a test walks the adapted output and fails if
  any operator factor survives to it); and an adapted operator table adapts the same
  operator once per definition and again per usage site, applying the orientation
  sensitivity twice, inconsistently.

  Also gated: **adaptation is additive** over term partitions. That is the precondition
  making any expansion order meaningful, and it is what establishes the residue as a
  write-order sensitivity rather than a `merge_terms` failure.

  The gate asserts the **exact** count (`{"doubles": 14}`) rather than "is exact", so
  e.2 landing will require updating it deliberately.

- **V1.1e.2 — make `ucc_integrate_term_antisym` orientation-invariant (~M).**
  **Route (b) chosen. Scoped in detail in
  `CCGEN_V11E2_ORIENTATION_INVARIANCE_SCOPE.md` (e.2.0–e.2.4).**

  The two routes were: (a) normalize `v` orientation as a pre-pass at the dress/adapt
  boundary, or (b) fix the adapter itself. This document originally recommended (a) on
  cost grounds — ~S change, validated adapter untouched. **Overruled deliberately:**
  (a) fixes the one caller that currently hurts and leaves the adapter write-order
  sensitive for the next one, accumulating a per-caller patch for a defect that lives
  in one function.

  Root-caused since that recommendation, which strengthens the case for (b): **the
  defect is not introduced by V1.** It is latent in `ucc_integrate_term_antisym`
  today, and any caller writing its `v` factors differently from the diagram generator
  hits it. Minimal reproducer, verbatim from the doubles manifold:

  ```
  expansion:      v(k,b,c,j) t2(a,c,i,k)   integrates to 2
  raw:         -1 v(j,c,k,b) t2(a,c,i,k)   integrates to 0
  ```

  Same term (exchange + bra-swap, and the raw `−1` is exactly that swap's sign; GCC
  matches at 0 mismatches). But `_line_pairs` reads slot `k` with `k+n`, so the two
  writings present different line structures — `k–c, b–j` versus `j–k, c–b`. All 16
  spin cases survive identically in both; the four mixed-spin cases carry **opposite
  signs**, because `_antisym_to_allowed` re-derives its sign from written slot order.

  Also measured and ruled out, so the fix does not go looking there: the bra↔ket
  exchange **alone** never diverges (0 in a 256-case sweep — it maps lines `p–r, q–s`
  to `r–p, s–q`, the same lines). The defect needs exchange *composed with* a
  within-group swap.

  **LANDED, and it did not close V1.1e.** `_orientation_normalized` fixed the measured
  defect — the reproducer now integrates to 0 on both writings — and `test_spin` is
  93/93 with the adapted residual multisets identical before/after (0 mismatched keys on
  every manifold). The spatial emit *shrank* 73260 → 65431 bytes, which is the
  normalization merging orientation-duplicate terms: same answer, fewer terms.

  **The doubles residue stayed at exactly 14 — and that turned out not to be a defect.**
  e.2.5 (`CCGEN_V11E25_RESIDUE_SCOPE.md`) resolved it: with a **symmetry-correct** `v` the
  adapted dressed and adapted raw residuals agree to **~1e-14 on every manifold**, across
  three `(no, nv, seed)` triples. **V1.1e's requirement is met.**

  The 14 are an artifact of the *comparison*: a term-by-term symbolic multiset cannot see
  that two sides chose different, symmetry-equivalent writings of the same algebra. The
  real defect was in `residual_eval.random_tensors`, which built `v` violating
  `<pq||rs> = <rs||pq>` (residual 2.35 vs ~1e-16 for real integrals, checked against
  pyscf) — so it silently defeated any numeric comparison of exchange-related writings.
  Fixed, with all four symmetries now gated.

  So orientation sensitivity was a real latent defect worth fixing on its own terms, and
  e.2.1 was both necessary *and* sufficient for the algebra.

  Two process notes worth carrying forward:

  - **The numeric gates were silently skipping.** `test_spin`'s pyscf gates report
    `skipped 'pyscf not importable'` in the default interpreter, so earlier "93 OK" runs
    never exercised S1/S2/S4 or the FCI-limit fixtures. Validate via
    `tests/pyscf/.venv/bin/python` (pyscf 2.13.0). A green default-interpreter run is
    not evidence here.
  - **The deliberately-exact `{"doubles": 14}` assertion earned its keep** — it stopped a
    premature "fixed" call on e.2.1 — **but it pinned a proxy, not the property.** A
    symbolic term-by-term multiset is the wrong instrument whenever both sides may pick
    among symmetry-equivalent written forms; it reports differences that are not there.
    Gate the *numeric* residual on symmetry-correct tensors instead, and treat symbolic
    counts as diagnostics only. e.2.5.2 makes that change.
  - **A numeric gate is only as good as its fixture's symmetry.** `random_tensors` was
    missing `<pq||rs> = <rs||pq>` for years without any test noticing, because nothing
    compared two exchange-related writings until dressing did. When adding a numeric gate,
    assert the fixture's invariants too — not just the result.

- **V1.1e.3 — per-operator localization — LANDED.** `_dress_operator_equations` gained an
  `operators` parameter (`None` = full seeded family, so callers are byte-unchanged), and
  `test_dress_per_operator.py` runs the numeric comparison one operator at a time.
  Measured vs adapted raw: `Wmnij` 7.11e-14, `Wabef` 5.33e-14, `Wmbej` 1.28e-13 (doubles).

  **The gate is numeric, not the scoped "0 mismatches".** That phrasing meant the symbolic
  term multiset, which e.2.5 disproved as an instrument — it sat at 14 while the algebra
  was exact. Residual values on symmetry-correct tensors replace it.

  Two guards so it cannot pass for the wrong reason: an assertion that each operator is
  genuinely *referenced* (otherwise "nothing dressed" trivially equals raw and the gate is
  vacuous — the same failure mode U0's original gate had), and a verified teeth check —
  corrupting one `Wmnij` definition coefficient moves doubles 7.11e-14 → 3.32e+01.

  **Measured aside now pinned:** under `canonical_fock=True` — the only mode Planck feeds
  CC — *no F operator is referenced at all*, alone or with the family. Their `f_ov` terms
  are Brillouin-zero and drop, so the dressed equation references exactly
  `{Wmnij, Wabef, Wmbej, tau, tau_c}`. Recorded as intended behavior so the three inert
  operators don't read as a gap.

*Gate:* per-operator and combined numeric agreement ≤1e-12 relative on every manifold,
over two `(no, nv, seed)` triples. **Passing.**

**Do not weaken the gate to make it pass.** A tolerance on term counts, or excluding
the repeated-factor keys, would discard exactly the terms most likely to be wrong —
and per the root cause above, those keys are where the orientation sensitivity lives.

**Do not "fix" this by comparing with `_eri_canonical` and calling it done.** The fold
makes the *comparison* agree while the *adapted output* still depends on writing —
which is what actually ships into the kernel. The gate must constrain the adapted
manifold, not just the verifier's view of it.

### V1.1f — index-space validity (the CSE trap) — **LANDED**

`validate_intermediate_spec` / `validate_intermediate_specs` in
`optimization/intermediates.py`, gated by `test_intermediate_validity.py` (17 tests).

Checks: sig matches slot spaces **positionally**, rank agrees, no repeated slot, every
definition term carries the spec's own free indices **in order**, slots buildable against
the reference partition, plus two list-level checks (duplicate names; forward/self
reference, since the emitter materializes builders in list order).

Two ordering details a weaker check would miss, and the reason both are positional: the sig
comparison is not a character multiset (`oovv` vs `vvoo` has the same characters in the
wrong order and would emit transposed dimensions), and definition-term free indices are
compared in order, not as a set (a permuted term writes a transpose into the buffer).

**All five dressed specs validate clean** (`tau`, `tau_c`, `Wmnij`, `Wabef`, `Wmbej`), so
this is a guard for whatever rides `IntermediateSpec` next, not a fix — as the scope
predicted ("*should* be immune … 'should' is what an assertion is for").

**New measurement, and a correction to my own first reading of it.** The validator was run
against both CSE paths: `detect_intermediates` yields 7 specs on the raw GCC equations and
16 on the spin-adapted ones — **all 23 clean**. My first conclusion from the GCC-only run
("so the mislabeling is specific to the spin-adapted path") was wrong; checking that path
too shows it clean as well.

What this actually establishes: index-space validity is **not** the live blocker for
`include_intermediates`. It does **not** establish CSE is correct — this validator checks
metadata self-consistency, not whether an intermediate computes the right value. Settling
that needs a numeric gate (CSE-rewritten residual vs un-rewritten), per the e.2.5 lesson.
The documented compile-time half of `e0f3849` (~1544 `build_W_*`, ~28 min at `-O3`) stands
regardless.

Nine of the 17 tests are negative cases, one per check — a validator that never fires is
worthless. Two anti-vacuity guards on the gate itself (the five expected spec names must be
present; `tau`/`tau_c` must precede their consumers).

*Gate:* zero problems on the real dressed spec list, and each check demonstrated to fire on
a spec broken in exactly that one way. **Passing.**

---

## Sequencing

```
V1.1a (adapt terms)               LANDED
   └→ V1.1b (re-derive layout)    LANDED  ← the assertion that would have caught Finding B
        └→ V1.1c (block-key)      LANDED  ← byte-identical on RCC
             └→ V1.1d (recount)   LANDED  ← bidirectional closure
                  └→ V1.1e (faithfulness)   PASSING (~1e-14, numeric)
                       ├─ e.0 clean GCC baseline        LANDED (was a real defect)
                       ├─ e.1 pin expansion order       LANDED
                       ├─ e.2 adapter orientation-invariance  LANDED (route b)
                       ├─ e.2.5 the 14 were a comparison artifact  RESOLVED
                       │      fixture fixed; numeric gate passes ~1e-14
                       └─ e.3 numeric per-operator localization  LANDED
                              W-operators localized; F inert under canonical Fock
                       └→ V1.1f (index-space validity)   LANDED
                            │      dressed specs clean; CSE-on-GCC also clean
                            ▼
                       V1.2 (restructure print_cpp_planck)   NEXT
```

**V1.1e is closed.** Of its four sub-steps, two fixed real defects (e.0's
`assemble_dressed_equation` bug, e.2's latent adapter orientation sensitivity), one
resolved a phantom (e.2.5 — the residue was a comparison artifact, and the actual bug was
in a shared test fixture), and one is pure localization (e.3). The recurring lesson, now
applied in three places: **gate on numeric residual values, not symbolic term counts.**

**All of V1.1 (a–f) is landed.** Emit is byte-identical throughout — currently
37216 / 65431 / 27960 — since none of it is wired into `print_cpp_planck` yet; that is V1.2.
(The spin-adapt figure moved 73260 → 65431 at e.2.1, when `v`-orientation normalization
merged orientation-duplicate terms: same algebra, fewer terms.)

V1.1e was where the real question lived, and it was **~M, not ~S** as re-estimated. Final
accounting of its four sub-steps: e.0 fixed a genuine lost coefficient in
`assemble_dressed_equation`; e.2 fixed a latent, pre-existing `v` bra↔ket orientation
sensitivity in the adapter; e.2.5 resolved the remaining `doubles = 14` as **not a defect
at all** — a comparison artifact, with the real bug in a shared test fixture; e.3 localized
the gate per operator. It was correctly sequenced last: none of that residue would have been
interpretable without a–d correct.

---

## What this reuses

| Reused | From |
|---|---|
| Own-slot-order adaptation | `intermediate_template` + `templates=` (V1.0, landed) |
| Zero-adaptation guard | V1.0's `ValueError` — makes a silent vanish impossible |
| The emitter's slot normalization | `_stable_spatial_indices` via `lower_term_restricted_closed_shell` — reuse, don't re-sort |
| Expansion / verification machinery | `dressed_equation.py` (proved GCC assembly exact, 0 mismatches) |
| Block-keyed naming | R3.1.3c (`t4_aaabaaab`), U1.1 (`t2_aaaa`) |
| Consumer-side name resolution | `_map_factor`'s `intermediate_names` set — already generic |

**Net new:** `adapt_intermediate_spec` and the six gates. No new verifier, no new
naming scheme, no new normalizer.

---

## What NOT to do

- **Do not force `spec.indices` into the builder** to "fix" Finding B. The emit
  path currently normalizes both sides via lowering and is self-consistent
  (Finding C); overriding it with the spec order would create a miscompile that
  does not exist today.
- **Do not hand-roll the slot normalization.** Use the emitter's own
  (`_stable_spatial_indices`). A second sort that agrees today and drifts later is
  precisely how Finding B happened.
- **Do not write a second expansion verifier for V1.1e.** `dressed_equation.py`'s
  is the one that established exactness in GCC.
- **Do not scope `Fme`/`Fae`/`Fmi` into V1.1.** They are recognized but
  unreferenced under canonical Fock (Finding A). Adapting them is dead code — and
  if a later method *does* reference them, V1.1a–f apply unchanged.
- **Do not defer block-keying (V1.1c).** Cheap now; touches both sides later.
- **Do not let V1.1c change the RCC emit.** Its gate is byte-identity against
  V1.1b's output.
- **Do not enable CSE `include_intermediates` alongside this.** Off under
  `--spin-adapt` already; mixing it in confounds V1.1e/V1.1f.

  V1.1f measured both CSE paths, and **neither shows the mislabeling**:
  `detect_intermediates` yields 7 specs on the raw GCC equations and 16 on the
  spin-adapted ones, and **all 23 pass `validate_intermediate_specs`**.

  So index-space validity is *not* the live blocker for `include_intermediates`, and the
  remaining reason to keep it off under `--spin-adapt` is the other half of `e0f3849`:
  **compile time** (~1544 `build_W_*` functions, ~28 min at `-O3`). Anyone re-enabling it
  should treat the correctness concern as *unreproduced by this check* — which is not the
  same as absent, since `validate_intermediate_specs` checks metadata self-consistency, not
  whether the intermediate computes the right value. A numeric gate (evaluate the
  CSE-rewritten residual against the un-rewritten one, per the e.2.5 lesson) is what would
  actually settle it.

---

## Why V1.1 was ~M (retrospective — all six sub-steps landed)

Six sub-steps: a–d ~S and mechanical, f ~S, and e carrying all the risk at ~M.

**The estimate was right about where the risk was and wrong about what it was.** V1.1e was
upgraded to ~M on the belief that the residue pointed at a collapse-commutation question
(repeated collapsible factors). **Probing disproved that** — the collapse is not implicated
(both sides sum identically through the full pipeline on the minimal reproducer), and the
repeated-factor signature was incidental: those terms simply have the most written forms,
hence the most opportunities for two sides to choose different ones. The ~M was nonetheless
warranted, just spent differently: on one real adapter defect (e.2), one phantom (e.2.5),
and the fixture bug underneath it.

Findings that moved the estimate, revisited:

- **Finding A** (only three operators in scope) — **confirmed and now pinned.** Under
  canonical Fock no F operator is referenced at all; e.3 asserts it.
- **Finding C** (emit path already self-consistent) — confirmed; no layout rewrite needed.
- **Finding B** (dishonest metadata, nothing checking it) — addressed by V1.1f, which found
  the current specs clean. So the exposure was real but unrealized.
- **"V1.1e is harder than running the existing verifier"** — true, but not for the
  anticipated reason. The existing verifier was the *wrong instrument*, not an insufficient
  one.

**Status:** all of a–f landed and gated, and V1.1e's numeric gate validates the composition
end-to-end at ~1e-14. The earlier warning here — "do not treat a–d landing as evidence the
composition is correct" — has been discharged by e.2.5/e.3 rather than merely asserted away.

The one caveat that remains: none of V1.1 is wired into `print_cpp_planck` yet (emit is
byte-identical), so this validates the *algebra and metadata*, not the emitted kernel. That
is V1.2.

---

See `CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md` (V1.0 landed, V1.2–V1.4 next),
`CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE.md` (V2–V6, which V1 unblocks), and
`CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U1.1, whose naming path V1.1c shares).
