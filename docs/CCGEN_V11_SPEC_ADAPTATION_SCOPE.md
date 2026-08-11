# V1.1 — adapting the dressed intermediate specs, in small verifiable steps

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

### V1.1e — the faithfulness gate (~M, **upgraded from ~S: probed, not yet passing**)

The parent scope's V1.1 gate. **Substituting an adapted spec's definition back into
the adapted dressed residual must reproduce the adapted raw residual**, term-by-term
after canonicalization.

`dressed_equation.py`'s `verify_dressed_equation(dressed, raw, operators)` is the
reuse surface — its `operators` dict is fully substitutable, so an adapted-definition
operator table drops straight in. **Reuse it; do not write a second verifier.**

**Probed against the tree. It does not pass yet, and the ordering matters:**

| configuration | singles | doubles |
|---|---|---|
| GCC baseline (no adaptation) | 1 | **0** |
| adapt-then-verify (adapted operator table) | 13 | 61 |
| expand-in-GCC-then-adapt | 2 | 14 |

Three things this establishes:

1. **The GCC doubles baseline is clean (0), so the verifier is a valid zero-reference
   there.** The GCC *singles* baseline is already 1 before any adaptation — a
   pre-existing canonicalization artifact (`$free0`/`$free1` tokens from
   `_free_order_normalized` landing in a summed slot), **not** caused by this work.
   So singles must be gated against its own baseline of 1, or that artifact fixed
   first; a naive "expect 0" gate on singles would fail for reasons unrelated to V1.1.
2. **Expansion order is load-bearing.** Expanding the dressed manifold in GCC and
   adapting the expansion is dramatically closer (61→14, 13→2) than adapting the
   operator definitions and the residual separately and then expanding. That is
   consistent with Decision 5 (`GCC → dress → adapt`) and says the gate should
   expand **before** adapting, not after.
3. **The residue is not a coefficient nudge.** The adapt-then-verify doubles diff has
   30 keys only-in-dressed and 12 only-in-raw — structurally different terms. Even in
   the better ordering, 7 of the 14 are only-in-dressed.

**The signature of the remaining 14.** Every one involves *repeated same-name
factors* — `t1·t1·v`, `t2·t2·v`, `t1·t1·t1·v`, `t1·t1·t1·t1·v`, `t1·t1·t2·v`. That is
the fingerprint of the closed-shell collapse's **Cartesian product over multiple
splittable factors** (`collapse_amplitudes` / `collapse_integrals` /
`_product_over_choices`): a term with two collapsible factors expands into 2×2 spatial
terms, and the count depends on how many collapsible factors a term carries. A dressed
term and its raw counterpart carry *different numbers* of them — the dressed one hides
some inside `W`/`τ` — so collapsing before vs after expansion is not commutative.

**So the real V1.1e question is not "does the verifier agree" but "does
expand ∘ collapse == collapse ∘ expand for terms with repeated collapsible
factors".** That is an algebra question about the RCC collapse, not about spec
plumbing, and it is why this step is ~M rather than ~S.

Sub-steps:

- **V1.1e.0 — fix or baseline the `$free` singles artifact (~S).** Decide whether
  it is a genuine `_free_order_normalized` bug (a free index colliding with a
  summed one) or benign. Until then, gate singles against its measured baseline
  rather than 0, and say so explicitly in the test.
- **V1.1e.1 — pin the expansion order (~S).** Gate expand-in-GCC-then-adapt, which
  is both the closer configuration and the Decision-5-consistent one. Record the
  adapt-then-verify numbers as the rejected alternative so the ordering is not
  silently revisited.
- **V1.1e.2 — resolve the repeated-collapsible-factor commutation (~M, the real
  work).** Establish whether the 14 are (a) a genuine non-commutation that the
  gate must account for by expanding first *always*, (b) a τ/τ_c double-count where
  a collapsed `t1t1` inside `τ` is also collapsed outside it, or (c) a bug in the
  dressed assembly that only adaptation exposes. Option (b) is the most likely
  given `tau_overlap_corrections` already exists to handle exactly this class of
  overlap in GCC — the corrections may need re-deriving post-adaptation.
- **V1.1e.3 — per-operator localization (~S given the above).** Once the manifold
  gate passes, run it per operator (`Wmnij`, `Wabef`, `Wmbej`) so a future
  regression names one operator.

*Gate:* 0 mismatches on doubles; singles at 0 or its justified baseline. Per-operator
and combined.

**Do not weaken the gate to make it pass.** A tolerance on term counts, or excluding
the repeated-factor keys, would discard exactly the terms most likely to be wrong.

### V1.1f — index-space validity (the CSE trap) (~S)

Assert every adapted spec's slot spaces match its `index_space_sig` and its dims
match the reference partition. This is the assertion `--spin-adapt` never got for
CSE intermediates, and the reason `include_intermediates` is force-disabled there
(`e0f3849`).

Dressed operators *should* be immune — their indices come from a recognized
physical operator with a declared block, not a syntactic pattern match — but the
two ride the same `IntermediateSpec` machinery, and "should" is what an assertion
is for.

*Gate:* per adapted spec, `len(indices) == rank`, spaces match the sig
character-for-character, and no index appears twice.

---

## Sequencing

```
V1.1a (adapt terms)               LANDED
   └→ V1.1b (re-derive layout)    LANDED  ← the assertion that would have caught Finding B
        └→ V1.1c (block-key)      LANDED  ← byte-identical on RCC
             └→ V1.1d (recount)   LANDED  ← bidirectional closure
                  └→ V1.1e (faithfulness)   PROBED, NOT PASSING  ← ~M, the real work
                       └→ V1.1f (index-space validity)
                            │
                            ▼
                       V1.2 (restructure print_cpp_planck)
```

a→b→c→d were each ~S and mechanical, and are landed: emit byte-identical throughout
(37216 / 73260 / 27561), since none of them is wired into `print_cpp_planck` yet —
that is V1.2.

V1.1e is where the real algebra question lives and it is **~M, not ~S**: probing
showed the naive gate fails 61/13 and the better ordering still fails 14/2, with the
residue localized to terms carrying repeated collapsible factors. It is last because
it needs a–d correct to be interpretable, and it should be treated as the step that
may reveal something about the RCC collapse rather than about spec plumbing.

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

---

## Why V1.1 is ~M

Six sub-steps. Four (a–d) were ~S, mechanical, and are landed. V1.1f is ~S. V1.1e
carries all the risk and has been **upgraded to ~M after probing** — the faithfulness
gate does not pass in any configuration yet, and the residue points at a
collapse-commutation question (repeated collapsible factors), not at spec plumbing.

The measurements moved the step *down* from the parent scope's estimate in two ways —
only three operators are in scope (Finding A), and the emit path is already
self-consistent so no layout rewrite is needed (Finding C) — and up in two: the
metadata was dishonest for three referenced operators with nothing checking it
(Finding B), and V1.1e is harder than "run the existing verifier".

**Honest status:** V1.1a–d are landed and gated, but they are *plumbing*. Nothing is
validated end-to-end until V1.1e passes, and V1.1e is where a real error in the
dressed-adapted algebra would live. Do not treat a–d landing as evidence the
composition is correct.

---

See `CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md` (V1.0 landed, V1.2–V1.4 next),
`CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE.md` (V2–V6, which V1 unblocks), and
`CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U1.1, whose naming path V1.1c shares).
