# How does the derivation route reach production?

**Scope for in-flight work.** Opened by `docs/CCGEN_TWO_DRESSING_ROUTES.md`, which established
that ccgen's *derivation* route (operators from each term's contraction tree) is value-preserving
at ranks 2-4, worth 2.0x-7.1x, and **has no production caller** — deferred in its own commit
(`f68f7e2`, "CCSD dressing stays D7.3's job") and never revisited.

This doc scopes connecting it. It is not a research question: the algebra is gated, the merge is
implemented, and the emit bridge exists. What is missing is that **there are two emitters** and
production uses the other one.

Retire this file when W1-W5 land; the answer belongs in `CCGEN_TWO_DRESSING_ROUTES.md`.

---

## The actual gap

Two parallel emitters that share exactly one parameter (`method`):

| | production | factorizer |
|---|---|---|
| entry | `print_cpp_planck` (`generate.py`) | `emit_factorized_translation_unit` (`factorize.py`) |
| called by | `generate_planck_cc_kernels.py` | **nothing** |
| dressing | `dress_operators=True` → the **retired** recognition route | derivation, merged |
| spin adaptation | `spin_adapt=`, `ucc=` | **none** |
| intermediates | `include_intermediates=`, three memory-budget knobs | `top_k` / `savings_fraction` / `memory_budget_bytes` |

```
production-only: dress_operators, factorize_tau, force_arbitrary, include_intermediates,
                 intermediate_{memory,peak_memory}_budget_bytes, intermediate_threshold,
                 spin_adapt, ucc
factorizer-only: canonical_fock, engine, factor_builder_bodies, max_operator_bytes,
                 memory_budget_bytes, merge_transposes, n_occ, n_vir, savings_fraction,
                 spatial, top_k
```

### The structural blocker, and the good news

`emit_factorized_translation_unit` calls `generate_cc_equations(method, ...)` **internally**
(`factorize.py:976`). It cannot be handed an already-adapted manifold, so it can never emit
spatial or UCC kernels — which is most of what production emits.

**The machinery underneath has no such limit.** Measured on adapted input:

| input | merged operators | terms rewritten |
|---|---|---|
| spatial `ccsd` doubles | 31 | 64 |
| UCC `ccsd` doubles_abab | 86 | 71 |

So the fix is to make the factorizer *accept* equations rather than generate them. That is a
signature change, not new algebra.

## Steps

Each step keeps `test_factorize_value_preservation` at 0/0 on both bases and both
`canonical_fock` settings. That gate, not the operator count or the TU size, is the acceptance
criterion throughout — a step that improves the numbers and reddens it has failed.

### W1 — make the factorizer take equations, not a method name (~S)

Split `emit_factorized_translation_unit` so the pipeline body accepts an `eqs` dict, with the
current signature kept as a thin wrapper that generates and delegates. No behaviour change.

*Verify:* the existing TU is **byte-identical** for `ccsd` and `ccsdt` at every current call
(the tests in `test_factorize.py` that emit TUs), and the wrapper's output is unchanged.

### W2 — the factorizer path on spatial input, end to end (~M)

Feed `spin_adapt_equations(...)` output through W1's entry and emit.

Two things to check that the algebra-level gates do not:

- the emitted TU **compiles** against the real CC headers (the existing compile gates in
  `test_factorize.py` show the pattern; they skip without Eigen);
- the spatial ERI symmetry contract still holds — `_ERI_SYMMETRY_PERMUTATIONS` in
  `planck_tensor_cpp.py` allows only the four parity-`+1` relations, and O1 found that folding
  the other four produces silent sign errors.

*Verify:* value gate 0/0; TU compiles; builder count reported against the un-merged baseline.

### W3 — decide what production calls, and remove the fork (~M, the real decision)

Today `--dress-operators` routes to recognition. Options, in the order I would try them:

1. **New flag** (`--derive-operators`), leaving `--dress-operators` on the retired route. Lowest
   risk, but keeps two dressing paths alive indefinitely and invites someone to pick the broken
   one.
2. **Repoint `--dress-operators`** at the derivation route and delete the recognition emit path.
   Honest, and matches the D8 recommendation — but it changes the meaning of an existing flag,
   so anything reproducing an old run silently gets different kernels.
3. **Merge the emitters**: `print_cpp_planck` gains the factorizer's selection knobs and calls
   W1's entry. Most work, and the only option that ends with one emitter.

**This is a judgement call about the project's direction, not a technical one — do not pick it
unilaterally.** My recommendation is 1 then 3: ship behind a new flag, prove it on the regression
suite, then merge and delete. What must not happen is 1 with no follow-through, which is exactly
how the route ended up disconnected the first time.

*Verify:* a stated decision with its reason, and if not option 3, a recorded owner for the
follow-up.

### W4 — the regression suite (~M)

The CC regression cases are the only end-to-end check that generated kernels compute the right
energies. Run the affected ones with derivation-emitted kernels.

*Verify:* energies match the current kernels to the tolerance each case already asserts. Any
disagreement is a **correctness** finding and outranks everything else in this doc — the value
gate is symbolic and a kernel can still be wrong in ways it cannot see (that is the entire
lesson of the rank-3 defect, where the kernel was correct and the harness was not; and of the
52 % dressed defect, where five gates passed).

### W5 — cost, measured rather than modelled (~M)

Everything claimed for this route so far is `operator_savings` / `build_cost`, a FLOP model.
`CCGEN_KERNEL_SCALING_SCOPE.md` measured the generated-vs-hand gap as a **scaling** defect
(21.8x → 50.1x, no plateau) that no current cost model predicts.

*Verify:* wall-clock for a derivation-emitted kernel against the current one on a case that
actually reaches the tensor path — note `choose_determinant_backstop`
(`tensor_backend.cpp:243`) routes `nso <= 16 && ndet <= 10000` to the determinant backstop, which
never calls the generated kernel, so most small cases yield **no timing at all**. `ch4_rccsdt_sto3g`
is the known-good rank-3 point.

Expect the modelled 2-7x not to survive intact. That is worth knowing either way; it is the
first wall-clock number this route will ever have had.

## What NOT to do

- **Do not un-retire the recognition route.** Nothing here touches it. It is 52 % short on Be
  with five failed fix attempts behind it, and its seven `expectedFailure` gates stay as the
  tripwire.
- **Do not trust the value gate as a correctness gate for kernels.** It evaluates the symbolic
  rewrite; it never compiles or runs anything. W2's compile check and W4's energies are separate
  instruments for a reason.
- **Do not wire UCC in this pass.** The machinery runs on UCC input (86 merged operators), but
  recognition finds **zero** operators there and the obvious tag-blind fix is measured and
  unsound — it collapses 12 spin-tagged contractions onto one `Wmbej`. That is O6 in
  `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` and needs its own numeric gate first.
- **Do not re-pin the six failing selection-model gates as part of this.** They need their claims
  restated, not their constants moved, and the merge changes the distribution again. Separate
  work, tracked in `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` (O4.6).

## Key code locations

| what | where |
|---|---|
| factorizer emitter (no caller) | `emit_factorized_translation_unit`, `python/ccgen/optimization/factorize.py:948` |
| its internal generate call — the blocker | same file, `:976` |
| production emitter | `print_cpp_planck`, `python/ccgen/generate.py:1023` |
| the CLI that chooses | `python/generate_planck_cc_kernels.py` (`--dress-operators`, `:114-147`) |
| merged operators + call-site plan | `manifold_operators_with_plan`, `factorize.py` |
| **the value gate** | `python/ccgen/tests/test_factorize_value_preservation.py` |
| spatial ERI symmetry contract | `_ERI_SYMMETRY_PERMUTATIONS`, `python/ccgen/emit/planck_tensor_cpp.py` |
| why this route, and what it is worth | `docs/CCGEN_TWO_DRESSING_ROUTES.md` |
| operator granularity and the merge | `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` |
| unmodelled cost | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
