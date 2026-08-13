# V1.2 — wire the dressed+adapted composition into `print_cpp_planck`

Scopes **V1.2** in small verifiable steps. V1.1 (a–f) validated the *algebra and metadata*;
nothing is reachable from the public entry point yet. V1.2 is the wiring that makes it
reachable.

**Probed before scoping. Every piece already works when called by hand** — so this is
genuinely wiring, not new machinery, and the estimate (~S) holds. What follows is grounded
in those measurements.

---

## Status: V1.2 LANDED (V1.2.0–V1.2.5)

All six steps landed; `V1.2.3` needed no work beyond the single exit path. Emitted sizes
match the hand-probed values exactly: `dress+spin_adapt` 45520 B, `+force_arbitrary`
46041 B, `dress_operators` alone unchanged at 27960 B.

**The six pinned baselines held byte-for-byte** across the refactor, so the only behavior
that moved is the intended reachability. Full suite: **748 tests OK**, 4 expected failures —
the same 4 as before, so no pre-existing test changed status.

**V1.2.4 was promoted ahead of schedule, because a guard caught the hazard rather than
inspection.** The assertion added in V1.2.1 (dressed and CSE/tau intermediates must not both
populate the channel) fired immediately on `dress+factorize_tau`: V1.2.1 had *activated* tau
under dressing, exactly as this scope predicted it would. Now an explicit `ValueError`.

**V1.2.2's miscompile was real, not hypothetical.** Before the fix, `dress+spin_adapt`
emitted GCC specs beside a spin-adapted residual — three of five layouts disagreeing
(`tau` `vvoo` vs `oovv`, `tau_c` likewise, `Wmbej` `ovvo` vs `oovv`). The residual referenced
spatially-adapted `Wmbej` while `build_Wmbej` built the GCC slot order.

---

## What the probe established

Called manually on `ccsd` (diagram engine, canonical Fock):

| composition | result |
|---|---|
| `spin_adapt_equations(dressed)` | works — energy 4 / singles 23 / doubles 61 terms |
| `adapt_intermediate_spec` on all 5 specs | works — `tau`, `tau_c`, `Wmnij`, `Wabef`, `Wmbej` |
| `validate_intermediate_specs(adapted)` | **clean** (V1.1f on the adapted specs) |
| emit with `spin_adapted=True` | **OK**, 45520 bytes, all five `build_<op>` present |
| emit with `force_arbitrary=True` too | **OK**, 46041 bytes, arbitrary-runtime symbols present |
| numeric: adapt(expand(dressed)) vs adapt(raw) | energy 0.00e+00, singles 4.3e-14, doubles 1.1e-13 |

Two things worth carrying into the steps:

**The early return is behaviourally equivalent to the general path at default flags.**
Verified: `emit_planck_translation_unit(method, d, intermediates=ints or None)` is
byte-identical to the same call with `force_arbitrary=False, spin_adapted=False` passed
explicitly (27960 bytes both ways). So removing the early return cannot move the
`dress_operators=True, spin_adapt=False` output — the byte-identity gate is satisfiable by
construction, not by luck.

**Adaptation changes the specs' declared layout, and that is the point.** Measured
before → after: `tau` `vvoo`→`oovv`, `tau_c` `vvoo`→`oovv`, `Wmbej` `ovvo`→`oovv`
(`Wmnij` `oooo` and `Wabef` `vvvv` unchanged in signature but reordered slots). This is
exactly what V1.1b's `relayout` exists for, and it means **V1.2 must emit the adapted specs,
never the GCC ones** — mixing them would declare one layout and build another.

---

## Steps

### V1.2.0 — pin today's two byte-identities as a test (~S, do first)

Before touching the function, pin what must not move:

- `dress_operators=False` (every existing flag combination in use: bare, `spin_adapt=True`,
  `factorize_tau=True`, `force_arbitrary=True`) — the default build must not move.
- `dress_operators=True, spin_adapt=False` — today's GCC dressed path.

Current values for reference: 37216 (bare), 65431 (`spin_adapt`), 27960 (`dress_operators`).

*Gate:* the test passes on the current tree, i.e. it records the status quo.

**Why first:** V1.2 is a refactor of a function with four interacting flags. A
byte-identity net *before* the change is the only way to know a "wiring" edit did not
quietly alter an existing path. Cheap, and it makes every later step's diff interpretable.

### V1.2.1 — remove the early return, keep behavior (~S, the mechanical core)

Replace the `if dress_operators:` early return with a flag that steers the existing single
exit path:

- dressing needs `engine="diagram"` + `canonical_fock=True`, so keep that override (it
  already pops conflicting caller kwargs rather than erroring — preserve that).
- `_dress_operator_equations` produces `(eqs, intermediates)`; let both flow into the
  *existing* `intermediates` variable and the *existing* `emit_planck_translation_unit`
  call at the bottom, rather than a second call site.

*Gate:* V1.2.0's byte-identities still hold — both of them. Nothing else changes yet.

**Constraint:** exactly one `emit_planck_translation_unit` call in the function when this
lands. If the diff leaves two, the composition has been forked and V5 (UCC) will have to be
wired twice — the thing the parent scope's "adapter as a parameter, not a branch" warning is
about.

### V1.2.2 — adapt the dressed specs when `spin_adapt` is on (~S, the real composition)

With the early return gone, `dress_operators=True, spin_adapt=True` currently spin-adapts
the *residual* but leaves the *specs* in GCC form — a real miscompile, since the residual
would reference spatially-adapted `Wmbej` while `build_Wmbej` builds the GCC layout.

So: when both flags are on, map each spec through `adapt_intermediate_spec` (V1.1a/b) before
handing it to the emitter. Measured to work on all five, with the layouts changing as
tabulated above.

*Gate:*
- `validate_intermediate_specs` (V1.1f) returns clean on the adapted spec list — wire this
  as an assertion in the emit path, not just a test, since it is the check that catches a
  declared-vs-built layout mismatch.
- All five `build_<op>` functions present in the emitted TU.
- The emitted TU compiles against the real CC headers (the same in-test compile check the
  τ work used).

**Do not skip the assertion.** V1.1f found the current specs clean, so it is a guard rather
than a fix — but this step is precisely where a layout mismatch would be introduced, and it
is the one place the guard is load-bearing rather than precautionary.

### V1.2.3 — thread `force_arbitrary` (~S; this is V1.3's substance)

Pass `force_arbitrary` through on the dressed path. Measured to work (46041 bytes,
arbitrary-runtime symbols present), so this is a parameter, not a feature.

*Gate:* `dress_operators=True, spin_adapt=True, force_arbitrary=True` emits a TU whose
kernel signature matches the arbitrary-order runtime's and whose `build_<op>` functions are
present and dependency-ordered (`tau`/`tau_c` before their consumers — V1.1f already
asserts that ordering property, so reuse it).

**This collapses most of V1.3.** The parent scope calls V1.3 out separately because the
arbitrary-order emit path must resolve dressed intermediate factors as locals. Probing shows
it already does. Keep V1.3 as a checkpoint for the *runtime-side* question (does a dressed
TU actually link and run against `ArbitraryOrderRCCAmplitudes`), which is a different
question from whether it emits.

### V1.2.4 — keep the mutual exclusions honest (~S)

Three exclusions, two already enforced, one to add:

- `dress_operators` × `factorize_tau` — dressing supersedes tau collapse. Currently
  "already the case" only because the early return made `factorize_tau` unreachable under
  dressing. **Removing the early return makes it reachable**, so this needs an explicit
  guard now. This is the one real hazard the refactor introduces.
- `dress_operators` × `include_intermediates` — keep CSE off under dressing, mirroring
  `--spin-adapt`. Note V1.1f measured CSE specs clean on *both* the GCC and spin-adapted
  paths (23/23), so the remaining reason is compile time (~1544 `build_W_*`, ~28 min at
  `-O3`) plus the absence of a numeric gate — not a known index defect.
- `spin_adapt` × `include_intermediates` — already forced off; unchanged.

*Gate:* each excluded combination either raises a clear error or documents which flag wins,
and a test asserts it. Silent precedence is what made the `factorize_tau` exclusion look
"already handled" when it was only unreachable.

### V1.2.5 — numeric gate on the composed path (~S)

The byte-identity gates prove no regression; they prove nothing about the *new* combination.
Add the e.2.5-style numeric check for `dress_operators=True, spin_adapt=True`: expand the
dressed manifold, adapt it, and compare residual values against adapted raw on
symmetry-correct tensors.

Measured already: energy 0.00e+00, singles 4.3e-14, doubles 1.1e-13.

*Gate:* ≤1e-12 relative on every manifold, at two `(no, nv, seed)` triples.

**Not a symbolic term-count comparison** — V1.1e spent five sub-steps establishing that a
term multiset cannot distinguish different algebra from a symmetry-equivalent rewriting.

---

## Sequencing

```
V1.2.0 (pin byte-identities)          LANDED  ← net before the refactor; earned its keep
   └→ V1.2.1 (remove early return)    LANDED  ← one emit call site; 6 baselines held
        └→ V1.2.2 (adapt the specs)   LANDED  ← fixed a REAL layout miscompile
             ├→ V1.2.3 (force_arbitrary)  LANDED (no work needed — single exit path)
             ├→ V1.2.4 (mutual exclusions) LANDED (promoted: a guard tripped it)
             └→ V1.2.5 (numeric gate)      LANDED (≤1e-12, two triples)
                  │
                  ▼
             V1.3 (compile: LANDED / link+run: OPEN) → V1.4 (dep order) LANDED → V2+
```

**V1.3's scoping was wrong about the blocker.** It was written as the runtime link-and-run
checkpoint, on the assumption that emitting already worked. Emitting did not work — the
blocker was a one-binding emitter gap. That half is fixed; link-and-run is still open and is
now the whole of what V1.3 has left.

**V1.4 landed alongside it**, because its gate ("assert on the emitted TU, not the spec
list") only becomes meaningful once cross-builder references exist in the text. Measured
order across all three dressed combinations: `tau`, `tau_c`, `Wmnij`, `Wabef`, `Wmbej`, zero
forward references.

Both predicted risks materialized and both were caught mechanically rather than by review:
V1.2.2's declared-vs-built layout mismatch (three of five specs) and V1.2.4's
`factorize_tau` activation. The V1.2.0 net is what made the first distinguishable from
ordinary churn, and the V1.2.1 assertion is what surfaced the second.

**V1.3 is now narrower than the parent scope describes.** Its emit-side substance was
absorbed by V1.2.3 — the arbitrary-order path already resolves dressed factors as locals.
What remains is the runtime question: does a dressed TU *link and run* against
`ArbitraryOrderRCCAmplitudes`? Emitting does not answer that.

---

## What this reuses

| Reused | From |
|---|---|
| `adapt_intermediate_spec` (adapt + relayout) | V1.1a/V1.1b — already takes `adapter=` for V5 |
| `validate_intermediate_specs` | V1.1f — as a wired assertion, not just a test |
| Dependency-order assertion (`tau` before consumers) | V1.1f's `test_dependency_order_*` |
| Numeric residual comparison + symmetry-correct fixture | e.2.5 / `residual_eval` |
| `spin_adapt_equations`, `_dress_operator_equations` | landed; called as-is |
| Single emit exit path with `force_arbitrary` / `spin_adapted` | already in the function |

**Net new:** a flag-steered path in one function, one wired assertion, and the gates. No new
emitter, no new adapter, no second composition path.

---

## What NOT to do

- **Do not leave two `emit_planck_translation_unit` call sites.** That is a forked
  composition; V5 (UCC) would need wiring twice. One exit.
- **Do not emit the GCC specs under `spin_adapt`.** Adaptation changes the declared layout
  (`tau` `vvoo`→`oovv`, `Wmbej` `ovvo`→`oovv` — measured), so the residual and the builders
  would disagree. This is the miscompile V1.2.2 exists to prevent.
- **Do not hard-code `spin_adapt_equations` as the adapter.** `adapt_intermediate_spec`
  already takes `adapter=`; keep the call site parameterized so V5 is a substitution.
- **Do not assume `factorize_tau` is still excluded** because it was before. The early return
  made it unreachable; removing it makes it reachable. Add the guard.
- **Do not enable CSE `include_intermediates` here** to "test the plumbing". Compile time and
  the missing numeric gate both stand; V1.1f's clean result is about metadata only.
- **Do not gate the new combination on byte-identity.** There is no prior output to be
  identical to — that is what V1.2.5 is for.
- **Do not run the numeric gates in the default interpreter.** They skip silently (pyscf not
  importable). Use `tests/pyscf/.venv/bin/python`.

---

## Honest status

V1.2 is landed: the composition is reachable from `print_cpp_planck`, and both ways the
wiring could go wrong are now guarded (spec layout via the wired V1.1f assertion, flag
interactions via explicit exclusions).

**The compile gate has since been delivered, and it failed.** V1.2.2 listed "the emitted TU
compiles against the real CC headers"; running it showed the dressed TU had *never* been
valid C++ — `build_Wmnij`/`build_Wabef` referenced `tau(...)` with no `tau` in scope, because
`sibling_names` made the factor render as a bare identifier without anything declaring it.
Pre-existing (fails identically at V1.2's parent); V1.2 made the path reachable, which
exposed it.

Fixed under V1.3 (`955ea33`): `_emit_intermediate_builder` now binds referenced siblings the
same way `_emit_kernel` already did. All three dressed TUs compile — dress-only 28122 B,
`+spin_adapt` 45682 B, `+force_arbitrary` 46203 B. The dressed baseline moved
27960 → 28122; the four undressed baselines are byte-identical.

**Still not validated: link-and-run.** The TU compiles (`-fsyntax-only`), but nothing has
linked or executed it. That is the remainder of V1.3's original intent.

---

See `CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md` (V1.2–V1.4 originals),
`CCGEN_V11_SPEC_ADAPTATION_SCOPE.md` (V1.1a–f, all landed — the machinery V1.2 wires), and
`CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U6, the other side of the adapter-as-parameter
constraint).
