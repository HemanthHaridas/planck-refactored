# Validating the dressed generated CC kernels (RCC, then UCC)

Scopes one capability: **numerically validate the generated CC kernels emitted
with dressed intermediates** — closing D7.3.5 — for RCC, and extend the same
validation ladder to UCC.

Today every numeric gate on the generated kernels runs on the **raw** (undressed)
residual. This document says exactly where the boundary is, why the existing
gates cannot be reused unchanged, and the smallest ladder that closes it.

Grounded in the current tree. Nothing here is landed.

---

## The boundary, stated precisely

The gap is sharper than "dressed is unvalidated". Two facts from the tree:

**Fact 1 — the dressed TU is not compiled into any binary.** `CMakeLists.txt`
exposes `PLANCK_CC_MAXORDER`, `PLANCK_CC_SPIN_ADAPT`,
`PLANCK_CC_ARBITRARY_LOWER_RANKS`, and `PLANCK_CC_ENGINE`. There is **no**
`--dress-operators` in `_ccgen_planck_kernel_args` and no `PLANCK_CC_DRESS`
option. So the dressed emit is reachable only by invoking `ccgen` by hand; no
Planck binary has ever executed a dressed kernel. This is what D7.3.5 records as
"a C++ build-integration step".

**Fact 2 — dressed and spin-adapted are mutually exclusive *in code*.**
`print_cpp_planck` (`generate.py:1008-1018`):

```python
if dress_operators:
    eqs = generate_cc_equations(method, engine="diagram", canonical_fock=True, ...)
    eqs, intermediates = _dress_operator_equations(eqs)
    return emit_planck_translation_unit(method, eqs, intermediates=intermediates or None)

eqs = generate_cc_equations(method, **kwargs)
if spin_adapt:
    ...
```

The dressed branch **returns early**. It never reaches `spin_adapt`, never
reaches `force_arbitrary`, and never passes `spin_adapted=` to the emitter.

This is the load-bearing consequence: **Decision 5's pipeline
(`GCC → dress → adapt → dressed spatial RCC/UCC`) is documented but not
implemented.** Dressing today emits a *spin-orbital* dressed TU. The validated
production path (`--spin-adapt`, which the Be CCSDTQ == FCI gate used) emits a
*spatial undressed* TU. There is no configuration that produces a dressed
spatial kernel.

So the deliverable is not only "add a numeric gate". It is:

1. compose dress ∘ adapt at all (**V1**), then
2. gate the composition symbolically (**V2**), then
3. gate it numerically end-to-end in C++ (**V3–V4**),
4. and only then extend to UCC (**V5–V6**).

Skipping to a numeric gate is not possible — there is nothing to compile.

---

## What is already validated (do not re-derive)

The equation-level dressed work is genuinely done, in GCC, and is strong:

- **Recognition** — all six CCSD operators (`Fme`/`Fae`/`Fmi`/`Wmnij`/`Wabef`/
  `Wmbej`) recognize as diagram subgraphs (D7.2 complete, `test_dressing.py`).
- **Assembly exactness** — `assemble_dressed_equation` re-expands to the
  canonical-Fock raw residual with **0 mismatches** (D7.3.0). The partition is on
  the expansion footprint, not the occurrence cover; the τ/τ_c overlap
  corrections are in; the four earlier "real" mismatches dissolved in canonical
  mode (2 `f_ov`, 2 `f_ov`-entangled τ̃ artifacts).
- **The canonical-Fock premise is an invariant, not an assumption.** Every Planck
  CC kernel gets a canonical Fock by construction (`f_ov = 0` identically,
  `f_oo`/`f_vv` diagonal) because all CC paths build from a converged RHF/UHF SCF
  via `build_{rhf,uhf}_reference`. So dressing against the canonical-Fock residual
  is exact *for Planck*, and no general-Fock oracle is needed.
- **Emit path** — `print_cpp_planck(dress_operators=True)` / `--dress-operators`
  emits `build_<op>` functions dependency-ordered (τ/τ_c before the W/F that
  reference them); default-off is byte-identical.

**So V2's symbolic gate is cheap**, because exactness-vs-raw already holds in
GCC. What is unproven is that exactness *survives spin adaptation*.

---

## Why the existing numeric gates do not transfer

| Existing gate | Why it does not cover dressing |
|---|---|
| Be CCSDTQ == FCI (`0970e21`, `ce03048`, gap 6.4e-11) | ran with `--spin-adapt`, which cannot coexist with `--dress-operators` (Fact 2). The strongest gate in the effort, and it has never seen a dressed kernel. |
| Diagram-engine FCI-limit gates (`test_spin.py`) | raw GCC / spatial-adapted residuals; no dressed manifold. |
| `test_dressing.py` / `test_dressed_equation.py` | symbolic only, and **GCC only** — they assert re-expansion against the raw *spin-orbital* residual. Silent about the adapted case. |
| `test_generated_source_compiles` (tau A1) | compiles a TU against real CC headers. Reusable as a *template* for V2's compile check, but proves nothing numerically. |
| CSE-intermediate gates | CSE is explicitly **disabled** under `--spin-adapt` (`e0f3849`) for both correctness and compile time. See the trap below. |

### The trap: dressed intermediates are not CSE intermediates

`--spin-adapt` force-disables `include_intermediates` because CSE "mislabels
indices (occ/vir size mismatch)" on spatial spin-adapted terms and has no numeric
gate there. Dressed operators ride the **same** `IntermediateSpec` /
`build_<name>` machinery (`operator_to_intermediate_spec`).

So the obvious question is whether dressing inherits CSE's spatial-layout bug.
It should not — a dressed operator's indices come from a *recognized physical
operator* with a declared block (`oooo`, `vvvv`, `ovvo`), not from a syntactic
pattern match over leaf sub-contractions. But "should not" is exactly the kind of
claim that needs an assertion rather than a paragraph, because the failure mode is
identical: a wrong occ/vir label produces a wrong-shaped intermediate and a
silently wrong residual. **V2 must assert dressed intermediate specs carry
correctly-spaced indices after adaptation** — and must not be satisfied by
CSE-style shape agreement alone.

---

## Scope (RCC)

### V1 — compose dress ∘ adapt (~M, the real work)

Make `print_cpp_planck(dress_operators=True, spin_adapt=True)` produce a dressed
**spatial** TU. Restructure the early return so the dressed manifold flows into
the `spin_adapt` branch instead of bypassing it:

```
generate_cc_equations(engine="diagram", canonical_fock=True)
  → _dress_operator_equations         # dressed GCC + intermediate specs
  → spin_adapt_equations              # per Decision 5
  → emit_planck_translation_unit(..., intermediates=..., spin_adapted=True)
```

Three things this must get right, all flagged by Decision 5 or by the adapter's
own contracts:

- **V1.0 — dressed factors must survive `ucc_integrate_term_antisym`.** The
  adapter is name-agnostic (`_line_pairs` / `block_exists` key on rank + slot
  structure, not factor name), which is why Decision 5 claims the FLOP win
  transfers for free. But Decision 5 also flags the caveat: *a dressed operator
  may carry different symmetry than a bare ERI*. `Wmnij` (`oooo`) and `Wabef`
  (`vvvv`) are symmetric-block; `Wmbej` (`ovvo`) is the asymmetric one whose
  binding sign is applied **gated on block asymmetry** (`_block_is_asymmetric`) —
  precisely the case where a bare-`v` antisymmetry assumption could be wrong.
  Each dressed factor needs the same GCC-slice-vs-antisym-integration check `t2`
  and `v` got when they first flowed through.
- **V1.1 — the intermediate specs must be adapted too.** Adaptation rewrites the
  residual's factor blocks, so a spec describing `Wmnij` in GCC form describes the
  wrong tensor after adaptation. The specs are what emit `build_<op>`, so if they
  are passed through unadapted the builder computes a spin-orbital operator that
  the spatial residual then indexes — the same class of error as the R3.1.2 bridge
  bug (a factor indexing the wrong slice, residual ≈ 0) and the B5 physicist-ERI
  convention bug. **This is the single most likely place V1 goes wrong.**
- **V1.2 — dependency order must hold post-adaptation.** τ/τ_c before the W/F
  referencing them. Adaptation may split one operator into block-variants; the
  topological order must be re-derived after adaptation, not reused from GCC.

*Gate:* structural — the composed path emits a TU; every residual factor name
resolves to either an amplitude, an ERI block, or a declared intermediate; no
factor carries an unresolved block; dependency order is a valid topological sort.

**Why V1 is ~M and not ~S:** it is not a plumbing change. `_dress_operator_equations`
returns `(dressed_eqs, ordered_intermediates)`, and only the first half currently
has a path through adaptation. V1.1 is new code.

### V2 — symbolic gate on the composed path (~S given V1)

The cheap, high-value gate, and it runs in seconds with no PySCF and no C++.

- **V2.0 — exactness survives adaptation.** In GCC, dressed re-expands to raw with
  0 mismatches. Assert the *adapted* dressed manifold re-expands to the *adapted*
  raw manifold: `adapt(expand(dress(G))) == adapt(G)` term-by-term after
  canonicalization. Since `expand ∘ dress == id` already holds in GCC and
  adaptation is a linear per-term rewrite, this should hold — and if it does not,
  it localizes the bug to V1.0/V1.1 before any compile.
- **V2.1 — index spaces are right (the CSE trap).** Every dressed
  `IntermediateSpec` surviving adaptation has occ/vir slot labels matching its
  declared block, and its dims match the reference partition. This is the
  assertion the CSE path lacks; it is the whole reason `--spin-adapt` disables
  intermediates.
- **V2.2 — the TU compiles** against real CC headers. Reuse the tau A1
  `test_generated_source_compiles` harness directly.
- **V2.3 — FLOP scaling actually improved.** Dressing that emits correct algebra
  at unchanged scaling is cosmetic — the same failure mode the CSE pass had (it
  removed duplicate leaves but left term structure unfactored). Assert the dressed
  spatial residual's cost **exponent** drops versus undressed. `estimated_build_flops`
  is element-count, not path cost, so this needs the contraction-path cost model
  already scoped in `CCGEN_HIGHER_OPERATOR_REUSE.md` (F-steps) — **reuse it, do
  not write a second one**. If that model is not landed, V2.3 degrades to a term-
  count-and-max-tensor-rank proxy and the real assertion waits.

### V3 — build integration (~S)

- `PLANCK_CC_DRESS` CMake option, **default OFF**, appending `--dress-operators`
  to `_ccgen_planck_kernel_args` — same shape as `PLANCK_CC_SPIN_ADAPT`.
- Default build stays byte-identical.
- **Measure the registry compile time immediately.** This is the demonstrated
  wall, not FLOPs: `-O3` on the intermediate-heavy registry took ~40 min
  (`a690014` dropped it to `-O1`), 256-term chunking followed (`c48a253`,
  `079a9e9`), and `--spin-adapt` disabled CSE partly because ~1544 `build_W_*`
  functions cost ~28 min. Dressed emits *far fewer* builders than CSE (six
  operators + τ/τ_c, versus ~1544), so the expectation is that dressing
  **improves** compile time — but that is a prediction to measure, and it is the
  number that decides the reachable rank.

*Gate:* `-DPLANCK_CC_DRESS=ON -DPLANCK_CC_SPIN_ADAPT=ON` configures, generates,
and compiles; record wall-clock registry compile versus the undressed spatial
build.

### V4 — the numeric gate: dressed spatial CCSDTQ == FCI (~S given V3)

Re-run the strongest existing gate with dressing on: **Be CCSDTQ == FCI**
(-14.4036550465, closed-shell reference gap 6.4e-11). Same system, same
tolerance, `PLANCK_CC_DRESS=ON`. This is D7.3.5 closed.

Ladder it, cheapest first, so a failure localizes:

1. **V4.0 — dressed CCSD energy** vs the hand-written `run_rccsd` on water/STO-3G.
   Smallest rank, in-tree oracle, and dressing is *defined* at CCSD (all six
   operators are CCSD operators). If V4.0 fails, V4.1 is uninformative.
2. **V4.1 — dressed CCSDT** vs the hand-written tensor-backend RCCSDT.
3. **V4.2 — dressed CCSDTQ == FCI** on Be. The real gate.

**Add the fixed-point probe as the debugging tool, not as a gate.** The B5 defect
(physicist-vs-chemist ERI binding *and* invalid antisym permutations — two coupled
bugs) was found only by injecting FCI-correct oracle amplitudes into live C++
state and evaluating residuals once, expecting all zeros. If V4.2 fails, that
probe is how you localize it; a failing energy alone will not distinguish a wrong
`Wabef` from a wrong τ_c coefficient. The probe already exists as uncommitted
debug (`fixture_probe.{cpp,h}`) — **commit it** as a test-only harness before
starting V4, so it is available when needed rather than reconstructed.

*Gate:* V4.2 passes at the same 1e-7 atol the ccsdtq FCI acceptance uses
(`491f485`).

---

## Scope (UCC) — extending the same ladder

This slots into `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` after its U1 (the
no-collapse adapt entry). Decision 5's pipeline is `GCC → dress → adapt`, and
**UCC is an adaptation** — so dressing composes with UCC through exactly the
V1 restructure, with no second dressing implementation.

### V5 — compose dress ∘ ucc_adapt (~S given V1 + U1)

Route the dressed GCC manifold through `ucc_adapt_equations` instead of
`spin_adapt_equations`. If V1 was built as "dressed manifold flows into
*whichever* adapter is selected" rather than "flows into `spin_adapt_equations`",
V5 is a one-line switch. **Build V1 that way.**

Two UCC-specific risks, both amplifications of V1's:

- **V5.0 — one dressed operator becomes several block-variants.** Decision 5
  measures adaptation splitting CCSD doubles 68 GCC → 124 RCC terms (~1.8×) and
  notes one `Wmnij` becomes several block-variants. UCC splits *further* (no a↔b
  collapse), so `Wmnij` yields an `oooo` variant per surviving spin block. Each
  needs its own `IntermediateSpec` and its own builder — V1.1's spec-adaptation
  must be block-keyed, not operator-keyed. **Design V1.1 block-keyed from the
  start**; retrofitting it is the expensive path.
- **V5.1 — `Wmbej`'s asymmetric-block sign under UCC.** The `ovvo` binding sign
  is applied gated on `_block_is_asymmetric` and is the one operator whose sign
  depends on bare-`v` antisymmetry. Under UCC the spin blocks of an `ovvo`
  operator are genuinely distinct tensors, so the gating predicate must key on the
  *spin-resolved* block, not the space pattern alone. Highest-risk single item in
  the UCC dressing work.

### V6 — the UCC numeric ladder (~S given V5 + U5)

Mirror V4 exactly, on UCC oracles:

1. **V6.0 — dressed UCC rank 2** vs the hand-written UCCSD energy on a radical
   cation. Same in-tree oracle U5 uses for its cheapest gate.
2. **V6.1 — dressed UCC residuals at PySCF amps.** Evaluate the dressed
   `doubles_aaaa`/`abab`/`bbbb` residuals at PySCF UCCSD's converged
   `t1a/t1b/t2aa/t2ab/t2bb` against `uccsd.update_amps`. This is UCC's one
   **direct** rank-4 oracle (RCC's is transitive), it costs minutes, and it is the
   only gate that isolates dressed-UCC *algebra* from the C++ runtime. Non-optional.
3. **V6.2 — dressed open-shell UCCSDTQ == FCI** on Li or BeH. The UCC analog of
   V4.2, and U5's own headline gate with dressing on.

*Gate:* V6.2 passes at the same tolerance as V4.2.

---

## Sequencing

```
V1 (compose)  →  V2 (symbolic, seconds)  →  V3 (build)  →  V4 (numeric RCC)
                                                             │
                                    U0, U1 ─────────────────┼→ V5 → V6 (numeric UCC)
```

V2 gates V1 before anything compiles. V4.0 (dressed CCSD vs hand-written) gates
the C++ integration before V4.2 (FCI) is attempted. V5 needs both V1 and U1, so
if UCC lands first, V5 is still blocked on V1 — **V1 is the shared critical
path and should land first regardless of which method is the target.**

---

## What this reuses

| Reused | From |
|---|---|
| Recognition of all six operators; assembly exact vs canonical raw | D7.2 / D7.3.0 (`dressing.py`, `dressed_equation.py`) |
| Canonical-Fock premise (`f_ov = 0` by construction) | `cc_canonical_fock_only` invariant — no general-Fock oracle needed |
| Adapters, name-agnostic per Decision 5 | `spin_adapt_equations`, `ucc_adapt_equations` (U1) |
| Emit + dependency-ordered builders | `operator_to_intermediate_spec`, `_emit_intermediate_builder` |
| Default-off build switch pattern | `PLANCK_CC_SPIN_ADAPT` |
| Compile harness | tau A1 `test_generated_source_compiles` |
| Headline numeric gate | Be CCSDTQ == FCI (`0970e21` / `ce03048`) |
| Failure-localization probe | `fixture_probe.{cpp,h}` (uncommitted — commit before V4) |
| Contraction-path cost model for V2.3 | `CCGEN_HIGHER_OPERATOR_REUSE.md` F-steps — do not write a second one |

**Net new:** the dress ∘ adapt composition (V1) with block-keyed spec adaptation
(V1.1), the symbolic post-adaptation exactness + index-space gates (V2), a CMake
switch (V3). Everything else is re-running existing gates in a new configuration.

---

## What NOT to do

- **Do not write a numeric gate first.** There is no dressed spatial TU to
  compile (Facts 1 and 2). V1 is unavoidable.
- **Do not dress the RCC/UCC terms directly.** Decision 5 rules this out on
  measured grounds: recognition runs on `diagram_representative` /
  `build_line_graph`, which post-adaptation `SpinTerm`s do not have, and the RCC
  surface is ~1.8× larger. Dress in GCC, adapt after.
- **Do not pass GCC intermediate specs through adaptation unchanged.** V1.1. This
  is the R3.1.2 / B5 failure class — a factor indexing the wrong slice yields a
  residual ≈ 0 or a plausible-but-wrong energy, not a crash.
- **Do not assume dressed intermediates inherit CSE's spatial-layout bug — or
  that they don't.** Assert it (V2.1). The two ride the same `IntermediateSpec`
  machinery and CSE is disabled under `--spin-adapt` for exactly this reason.
- **Do not accept a dressed kernel that passes V4 but not V2.3.** Correct algebra
  at unchanged scaling is the cosmetic outcome CSE already delivered; dressing's
  entire justification is the FLOP exponent.
- **Do not enable `PLANCK_CC_DRESS` by default until V4.2 passes**, and not before
  the registry compile time is measured at the target rank.
- **Do not re-derive the enumeration or the τ/τ_c weights.** The doc is explicit:
  the verified enumeration, assembly, and structural fixes (the four listed
  coefficient/ordering fixes, including the `tau_c` weight-1-vs-2 distinction and
  the `_eri_canonical` relabel-then-fold ordering) are correct and hard-won.

---

## Open question worth settling before V1

**Does `force_arbitrary` compose with dressing?** The dressed branch returns
before `force_arbitrary` is consulted, so a dressed TU cannot currently target
`ArbitraryOrderRCCAmplitudes`. The generated production path *is* the arbitrary-
order runtime (that is what `run_rccsdtq` drives, and what the FCI gate exercised
via `--spin-adapt`). So V1's restructure must thread `force_arbitrary` through
as well, or V4.2 has no runtime to execute in.

Treat this as part of V1, not a follow-up — it is the same early-return defect,
and discovering it at V3 wastes the build integration.

---

See `CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` (Decisions 3–5 and the D7.3.5 gate
this closes), `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (U0/U1, which V5 extends),
`CCGEN_SPIN_ADAPTATION_SCOPE.md` (the adapters), and
`CCGEN_HIGHER_OPERATOR_REUSE.md` (the cost model V2.3 needs).
