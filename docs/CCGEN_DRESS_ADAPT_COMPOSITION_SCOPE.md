# V1 — composing dress ∘ adapt, in small verifiable steps

Scopes **V1** of `CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE.md`: make
`print_cpp_planck(dress_operators=True, spin_adapt=True)` emit a dressed
**spatial** translation unit, implementing Decision 5's
`GCC → dress → adapt → dressed spatial RCC/UCC`.

V1 is the shared critical path — V2/V3/V4 (RCC numeric) and V5/V6 (UCC) are all
blocked on it, and it is the only ~M item in the dressed-validation ladder.

**This scope is measured, not predicted.** Two defects were reproduced against
the current tree while writing it (§ Measured findings). They change the shape of
the work: V1 is not a plumbing change.

---

## Measured findings (reproduced, current tree)

Running the six seeded operators' `definition_terms` through
`spin_adapt_equations` — which is exactly what V1 must do — gives:

```
Fme      sig=ov     tpl=[e:vir, m:occ]                -> 2 terms
Fae      sig=vv     tpl=[a:vir, e:vir]                -> 8 terms
Fmi      sig=oo     tpl=[m:occ, i:occ]                -> 5 terms
Wmnij    sig=oooo   tpl=[m:occ, n:occ, i:occ, j:occ]  -> 4 terms
Wabef    sig=vvvv   tpl=[a:vir, b:vir, e:vir, f:vir]  -> 4 terms
Wmbej    sig=ovvo   tpl=[b:vir, e:vir, m:occ, j:occ]  -> 0 terms   ← DEFECT
```

### Finding 1 — `Wmbej` adapts to **zero terms**, silently

`_residual_template` (`spin.py:895`) builds the free-index template as
**virtuals first, then occupieds**. For `Wmbej` that reorders the operator's own
slot order `[m, b, e, j]` (`ovvo`) into `[b, e, m, j]`. The adapter then pairs
slot `k` with slot `k+n` — i.e. `b↔m` and `e↔j` — but `Wmbej`'s physical lines
are `m↔e` and `b↔j`. `_closed_shell_representative_block` assigns
`{b:a, e:b, m:a, j:b}`, which violates spin conservation on **every** physical
line, so `block_exists` rejects every term. Result: 0 survivors, no exception.

The operator itself is fine. Enumerating all 16 blocks over `[m,b,e,j]` in the
operator's own order:

```
aaaa -> 6    abab -> 6    abba -> 5
baab -> 5    baba -> 6    bbbb -> 6
```

`abab` (respecting `m↔e`, `b↔j`) yields 6 survivors. **So this is a
slot-ordering-contract bug at the dress/adapt boundary, not a missing algebra.**

This is precisely the failure class the validation scope predicted for V1.1 — and
the same class as the R3.1.2 bridge bug and the B5 ERI-convention bug: a wrong
slot binding yields a **silently zero or wrong** result, never a crash. It is now
measured rather than hypothesized.

### Finding 2 — the spec's `indices` desynchronize from the adapted terms

`operator_to_intermediate_spec` sets `indices` from `op.block` in the operator's
own order (`Wmbej` → `[m, b, e, j]`; `Fme` → `[m, e]`). `_residual_template`
independently derives virtuals-first order (`[b, e, m, j]`; `[e, m]`). After
adaptation the terms' free-index order is the template's, but the spec still
carries the operator's — and the spec is what emits `build_<op>`'s signature and
loop nest.

So even for the five operators that *do* adapt, `Fme` and `Wmbej` have a spec
whose declared index order disagrees with the residual's usage. `Wmnij` /
`Wabef` / `Fae` / `Fmi` happen to agree because their blocks are already
space-homogeneous (`oooo`, `vvvv`, `vv`, `oo`) — **the two mixed-space operators
are exactly the two that disagree.** That is a coincidence of the seeded set, not
a property to rely on.

**Consequence for sequencing:** V1.0 (the ordering contract) must land before
V1.1 (adapting the specs), because V1.1's faithfulness gate cannot pass while the
`Wmbej` manifold is empty.

---

## What V1 must produce

```
generate_cc_equations(engine="diagram", canonical_fock=True)
  → _dress_operator_equations          # dressed GCC residual + intermediate specs
  → <adapter>                          # spin_adapt_equations OR ucc_adapt_equations
       ├─ residual manifolds           (dressed terms, adapted)
       └─ intermediate definition_terms (adapted, block-keyed)
  → emit_planck_translation_unit(..., intermediates=..., spin_adapted=True,
                                 force_arbitrary=...)
```

Today `print_cpp_planck` returns early in the `dress_operators` branch
(`generate.py:1008-1018`), reaching none of `spin_adapt`, `force_arbitrary`, or
`spin_adapted=`.

---

## Steps

Ordered so each is independently checkable and the two measured defects are fixed
before anything is composed.

### V1.0 — pin the slot-ordering contract at the dress/adapt boundary (~S)

Fix Finding 1. The adapter pairs slot `k` with slot `k+n`; a dressed operator's
`indices` must be presented in an order where that pairing matches its physical
lines.

Two candidate fixes — **prefer (a)**:

- **(a) Adapt each operator on its own block order**, bypassing
  `_residual_template` for intermediates. `_representative_block_for_sector`
  already takes a template, so the change is to build that template from
  `spec.indices` (the operator's declared order) rather than re-deriving it
  virtuals-first. Keeps `_residual_template` untouched, so the residual path is
  byte-identical.
- **(b) Permute each operator's slots to virtuals-first and carry the sign.**
  Mirrors what `_canonicalize_amplitude_factor` does for amplitudes. Strictly more
  work and introduces a sign that must then be threaded into the emitted builder —
  more places to be wrong, for no gain.

**Do not "fix" `_residual_template`.** It encodes the occ-first/virtuals-first
convention that the C++ runtime's `rank_dims` depends on — pinned by `02364db`
(R3.1.2 half (ii)). Changing it would move the residual layout contract to fix an
intermediate-layout bug.

*Gate:* every seeded operator adapts to a **non-empty** term list, and the
operator's physical line pairing is preserved (assert `block_exists` accepts the
chosen representative block for each). Specifically: `Wmbej` yields 6 survivors,
not 0. This is a seconds-long pure-Python test and it is the single highest-value
assertion in V1 — it is the one that is currently failing.

*Also assert:* a regression guard that no operator silently adapts to zero terms.
An empty manifold must be an **error**, not a valid result. Silence is what made
this defect invisible.

### V1.1 — adapt the intermediate specs (~M, the core step)

Fix Finding 2 and deliver the actual composition for intermediates.

`definition_terms` are ordinary `AlgebraTerm`s over `t1`/`t2`/`v`/`f`/`tau`
(verified: see the dump in Findings) — the same shape `spin_adapt_equations`
consumes. **So the specs can be adapted by the same adapter, not a bespoke one.**
That is the finding that keeps V1.1 at ~M instead of ~L.

Sub-parts:

- **V1.1a — adapt `definition_terms`.** Route each spec's terms through the
  selected adapter on the block from V1.0. Rebuild the spec with the adapted terms.
- **V1.1b — re-derive `indices` and `index_space_sig` from the adapted result**,
  so the spec's declared layout is the adapted layout (fixing Finding 2's
  desynchronization). The emitted `build_<op>` signature and the residual's
  factor usage must agree by construction, not by coincidence.
- **V1.1c — block-key the spec identity.** `IntermediateSpec.__hash__` / `__eq__`
  key on `(name, indices, index_space_sig)`. Under RCC one operator yields one
  adapted spec, but Decision 5 notes adaptation can split an operator into
  block-variants, and **under UCC it certainly does** (one `Wmnij` per surviving
  spin block). So the spec name must carry the block tag — `Wmnij_abab` — reusing
  the **same** naming path U1.1 uses for `t2_aaaa`. One naming mechanism for
  amplitudes, ERIs, and intermediates; do not add a second.

  Build this block-keyed **now**, even though RCC's reference sector needs only
  one variant. Retrofitting it during V5 is the expensive path, and the UCC scope
  already commits U1.1 to the same mechanism.

*Gate (faithfulness, the load-bearing one):* for each operator, the adapted spec
expands to the same primitives as adapting the operator's expansion —
`adapt(definition_terms)` is consistent with how the adapted residual indexes it.
Concretely: substituting the adapted spec's definition back into the adapted
dressed residual reproduces the adapted **raw** residual, term-by-term after
canonicalization. This is V2.0's gate restricted to one operator at a time, which
is how a failure localizes to a single operator instead of the whole manifold.

*Gate (spaces):* every adapted spec's slot spaces match its `index_space_sig`,
and its dims match the reference partition. This is the CSE-trap assertion
(V2.1) applied per-spec — the thing `--spin-adapt` disabling `include_intermediates`
never got.

### V1.2 — restructure `print_cpp_planck` (~S, given V1.0/V1.1)

Remove the early return. Make the dressed manifold flow into whichever adapter is
selected, and thread through the two flags the early return currently skips:

- `spin_adapt` → `spin_adapt_equations` (RCC)
- `force_arbitrary` → the emitter, so a dressed TU can target
  `ArbitraryOrderRCCAmplitudes`

**Build the adapter as a parameter, not a branch.** The dressed manifold should
flow into "the selected adapter", so V5 (UCC) is a one-line switch rather than a
second composition path. This is the same constraint the UCC scope's U6 section
records from the other side.

Keep `dress_operators` and `factorize_tau` mutually exclusive (already the case —
dressing supersedes tau collapse). Keep CSE `include_intermediates` **off** under
dressing for now: it is off under `--spin-adapt` already (`e0f3849`), and mixing
an unvalidated CSE pass into the composition would confound V2's gates. Revisit
only after V4 passes.

*Gate:* `dress_operators=True, spin_adapt=False` emits a TU byte-identical to
today's (no regression to the GCC dressed path); `dress_operators=False` is
byte-identical to today's undressed path (the default build must not move).

### V1.3 — `force_arbitrary` composition (~S)

The validation scope flagged this as settle-before-V1, and it is: the generated
production path **is** the arbitrary-order runtime (`run_rccsdtq` drives it, and
the Be CCSDTQ == FCI gate reached it via `--spin-adapt`). A dressed TU that cannot
target `ArbitraryOrderRCCAmplitudes` has no runtime to execute in, so V4.2 would
have nothing to run.

Mostly covered by V1.2's flag threading; called out separately because it needs
its own check — the emitter's arbitrary-order path must resolve dressed
intermediate factors as locals, the same way `_map_factor` /
`_emit_intermediate_builder` were extended to take an `intermediate_names` set for
the GCC dressed emit.

*Gate:* `dress_operators=True, spin_adapt=True, force_arbitrary=True` emits a TU
whose kernel signature matches the arbitrary-order runtime's, and whose dressed
`build_<op>` functions are present and dependency-ordered.

### V1.4 — dependency order re-derived post-adaptation (~S)

`_dress_operator_equations` returns `pseudo_specs + op_specs` — a two-level order
valid because every operator's dependencies are pseudo (τ/τ_c). After adaptation
that reasoning still holds *if* adaptation does not introduce cross-operator
references. It should not (adaptation is a per-term rewrite). But with V1.1c's
block-keyed variants there are more specs, and the order must be a topological
sort of the **adapted** DAG, not the GCC one reused.

*Gate:* the emitted builder order is a valid topological sort of the adapted
spec DAG — every referenced intermediate is defined before first use. Assert on
the emitted TU, not on the spec list, so it catches an emit-layer reordering too.

---

## Sequencing

```
V1.0 (ordering contract, ~S) ──→ V1.1 (adapt specs, ~M) ──→ V1.2 (restructure, ~S)
       │                                                          │
       └─ fixes the measured Wmbej-zero defect                    ├─→ V1.3 (force_arbitrary, ~S)
                                                                  └─→ V1.4 (dep order, ~S)
                                                                            │
                                                                            ▼
                                                                     V2 (symbolic gates)
```

V1.0 first and alone — it is ~S, it fixes a measured defect, and V1.1's gate
cannot pass while `Wmbej` is empty. V1.1 is the only ~M step. V1.2–V1.4 are
mechanical once the algebra is right.

---

## What this reuses

| Reused | From |
|---|---|
| Dressed GCC manifold + specs, exact vs canonical raw (0 mismatches) | `_dress_operator_equations`, `assemble_dressed_equation` (D7.3.0) |
| The adapter itself — `definition_terms` are ordinary `AlgebraTerm`s | `spin_adapt_equations` / `ucc_adapt_equations` (U1) |
| Block-keyed factor naming | U1.1 (`t2_aaaa`) and the R3.1.3c precedent (`t4_aaabaaab`) |
| Emitter resolving intermediates as locals | `_map_factor` / `_emit_intermediate_builder` `intermediate_names` set |
| Canonical-Fock premise (`f_ov = 0` by construction) | `cc_canonical_fock_only` invariant |

**Net new:** the ordering contract (V1.0) and spec adaptation with block-keyed
identity (V1.1). V1.2–V1.4 are wiring.

---

## What NOT to do

- **Do not change `_residual_template`.** It encodes the occ-first residual layout
  the C++ runtime's `rank_dims` depends on (`02364db`). Fix the intermediate side.
- **Do not let an operator adapt to zero terms silently.** That is the measured
  defect; make it an error (V1.0's guard). An empty dressed operator produces a
  kernel that compiles, runs, and is wrong.
- **Do not write a bespoke adapter for `definition_terms`.** They are ordinary
  `AlgebraTerm`s — verified. A second adapter is a second thing to keep exact, and
  the bridge took two commits (`ef42800`, `cfe302a`) to get right once.
- **Do not defer block-keying to V5.** Under UCC one operator becomes several
  block-variants; `IntermediateSpec` identity keys on `(name, indices, sig)`, so
  without a tag in the name the variants collide. Cheap now, a retrofit later.
- **Do not enable CSE `include_intermediates` under dressing yet.** It is disabled
  under `--spin-adapt` for unvalidated-layout and compile-time reasons; mixing it
  in confounds V2's gates.
- **Do not skip V1.3.** Without `force_arbitrary`, V4.2 has no runtime.
- **Do not re-derive the enumeration, τ/τ_c weights, or the four structural fixes.**
  The GCC dressed work is correct and hard-won; V1 changes only what happens
  *after* `_dress_operator_equations` returns.

---

## Why V1 is ~M and not ~S

The validation scope estimated ~M on the grounds that
`_dress_operator_equations` returns `(dressed_eqs, specs)` with only the first
half having a path through adaptation. The measurements confirm it and add
specificity: the boundary has a **slot-ordering contract mismatch** that zeroes
one of six operators, and a **spec/template index desynchronization** affecting
exactly the two mixed-space operators. Both are invisible without an assertion.
V1.1 is genuinely new code; the rest is wiring around it.

---

See `CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE.md` (V2–V6, which V1 unblocks),
`CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` (Decision 5 and the D7.3 work V1
composes), and `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U1/U1.1 and the U6
constraints V1.1c and V1.2 honor).
