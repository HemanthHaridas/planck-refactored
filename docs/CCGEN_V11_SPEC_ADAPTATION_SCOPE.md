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

### V1.1e — the faithfulness gate (~S, the one that matters)

The parent scope's V1.1 gate, now reachable. Per operator, one at a time so a
failure localizes:

**Substituting an adapted spec's definition back into the adapted dressed residual
must reproduce the adapted raw residual**, term-by-term after canonicalization.

This is V2.0's whole-manifold gate restricted to a single operator. Run it
per-operator first (`Wmnij`, then `Wabef`, then `Wmbej` — cheapest and most
structured first, the 5→8 one last), then once with all three.

`dressed_equation.py` already has the expansion/verification machinery that proved
the GCC assembly exact with 0 mismatches; V1.1e is that same check with adaptation
applied to both sides. **Reuse it — do not write a second verifier.**

*Gate:* 0 mismatches per operator and 0 for the combined manifold.

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
V1.1a (adapt terms)
   └→ V1.1b (re-derive indices/sig)   ← the assertion that would have caught Finding B
        └→ V1.1c (block-key names)    ← byte-identical on RCC
             └→ V1.1d (recount usage) ← bidirectional closure
                  └→ V1.1e (faithfulness, per-operator)   ← the gate that matters
                       └→ V1.1f (index-space validity)
                            │
                            ▼
                       V1.2 (restructure print_cpp_planck)
```

a→b→c→d are each ~S and mechanical. V1.1e is where a real algebra error would
surface; it is last because it needs a/b/c/d to be right to be interpretable.

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

Six sub-steps, five of them ~S and mechanical, one (V1.1e) carrying the real risk.
The measurements moved it *down* from the parent scope's estimate in two ways —
only three operators are in scope (Finding A), and the emit path is already
self-consistent so no layout rewrite is needed (Finding C) — and up in one: the
metadata is dishonest for two referenced operators and nothing currently checks it
(Finding B).

---

See `CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md` (V1.0 landed, V1.2–V1.4 next),
`CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE.md` (V2–V6, which V1 unblocks), and
`CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U1.1, whose naming path V1.1c shares).
