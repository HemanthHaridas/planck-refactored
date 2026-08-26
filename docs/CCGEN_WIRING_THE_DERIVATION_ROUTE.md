# How does the derivation route reach production?

**Scope for in-flight work. W1-W2 landed; W3-W5 open.** Opened by
`docs/CCGEN_TWO_DRESSING_ROUTES.md`, which established that ccgen's *derivation* route (operators
from each term's contraction tree) is value-preserving at ranks 2-4, worth 2.0x-7.1x, and **has no
production caller** — deferred in its own commit (`f68f7e2`, "CCSD dressing stays D7.3's job") and
never revisited.

It is not a research question: the algebra is gated, the merge is implemented, and the emit bridge
exists. What is missing is that **there are two emitters** and production uses the other one.

**Where it stands.** The technical blocker is gone — the pipeline now accepts adapted equations
(W1) and the spatial TU compiles merged and un-merged, 59 → 31 builders (W2). What remains is
**W3, a decision about the project's direction rather than a technical step**, and the two
verification steps behind it: W4 (regression energies, the one that can still find a correctness
defect) and W5 (wall-clock, which this route has never had).

Retire this file when W1-W5 land; the answer belongs in `CCGEN_TWO_DRESSING_ROUTES.md`.

---

## The actual gap

Two parallel emitters that share exactly one parameter (`method`):

| | production | factorizer |
|---|---|---|
| entry | `print_cpp_planck` (`generate.py`) | `emit_factorized_translation_unit` (`factorize.py`) |
| called by | `generate_planck_cc_kernels.py` | **nothing** (this is W3) |
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

**Removed by W1.** `emit_factorized_translation_unit` used to call
`generate_cc_equations(method, ...)` internally, so it could not be handed an already-adapted
manifold and could never emit spatial or UCC kernels — which is most of what production emits.
It is now a thin wrapper over `emit_factorized_from_equations`, which takes the equations.

The machinery underneath never had that limit, which is why W1 was a signature change rather than
new algebra. Measured on adapted input:

| input | merged operators | terms rewritten |
|---|---|---|
| spatial `ccsd` doubles | 31 | 64 |
| UCC `ccsd` doubles_abab | 86 | 71 |

Spatial now emits and compiles end to end. **UCC still does not**: the emitter rejects the
spin-blocked manifold names (`ValueError: Unknown manifold 'singles_aa'`). That is emitter work
downstream of O6 and deliberately out of scope here — see *What NOT to do*.

## Steps

Each step keeps `test_factorize_value_preservation` at 0/0 on both bases and both
`canonical_fock` settings. That gate, not the operator count or the TU size, is the acceptance
criterion throughout — a step that improves the numbers and reddens it has failed.

### W1 — LANDED (2026-08-25): the pipeline takes equations; the blocker is gone

`emit_factorized_from_equations(method, eqs, **selection)` is the pipeline body;
`emit_factorized_translation_unit` is now a thin wrapper that generates and delegates.

**Byte-identical**, verified by sha256 over six configurations before and after the split
(`ccsd`, `ccsd top_k=8`, `ccsd merge_transposes`, `ccsdt`, `ccsdt top_k=5`,
`ccsdt memory_budget`). No behaviour change.

**The blocker is lifted.** Handed `spin_adapt_equations(...)` output the new entry emits a
spatial TU: 82,793 chars, **31 builders**, braces balanced, each builder defined exactly once.
Previously impossible — the wrapper called `generate_cc_equations` internally and could only ever
emit GCC.

**UCC still fails, and deliberately so:** `ValueError: Unknown manifold 'singles_aa'`. The
emitter's manifold vocabulary does not include the spin-blocked names. That is emitter work
downstream of O6 (recognition finds zero operators on spin-tagged factors, and the tag-blind fix
is unsound), so it is out of scope here rather than patched.

*Gates:* `test_emit_from_equations_is_byte_identical` (wrapper == direct, four configurations)
and `test_emit_from_equations_accepts_a_spin_adapted_manifold`.

*Falsifiability:* dropping a kwarg from the wrapper fails 1 test; making the pipeline ignore the
caller's `eqs` fails 5.

Full suite after: 101 tests, the same 6 selection-model failures. No new breakage.

### W2 — LANDED (2026-08-26): the spatial TU compiles, merged and un-merged

Both halves answered, and both **actually run** here rather than skipping — Eigen and a C++23
compiler are present in this tree.

**It compiles.** `-fsyntax-only` against the real CC headers, exit 0 both ways:

| | builders | compiles |
|---|---|---|
| spatial, un-merged | 59 | yes |
| spatial, **merged** | **31** | yes |

Merging nearly halves the generated builder count and the TU still type-checks — so the merged
call sites really do resolve against the representative's `build_W`, not just in the algebra.
This is the check neither the value gate (symbolic, never compiles) nor W1's emit check (counted
builders and braces) could make.

**The ERI parity contract holds.** `_ERI_SYMMETRY_PERMUTATIONS` is exactly the four parity-`+1`
relations, each carrying an explicit `+1` sign, and the set is identical to
`_ERI_PERMUTATIONS_SPATIAL`. Worth re-checking rather than assuming, because the merge now runs
inside the emit path: this is the contract whose violation let a 52 % energy defect pass every
symbolic check, and the same one that produced two false operator merges in O1.

*Gates:* `test_spatial_factorized_tu_compiles` (both variants, skips without a toolchain) and
`test_emitter_folds_only_sign_preserving_eri_symmetries`.

*Falsifiability:* adding a `-1` permutation to the emitter's set fails the parity gate.

Full suite: 126 tests, the same 6 selection-model failures. No new breakage.

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
| factorizer pipeline (W1) — takes equations | `emit_factorized_from_equations`, `python/ccgen/optimization/factorize.py:993` |
| its generating wrapper, still no caller | `emit_factorized_translation_unit`, same file `:948` |
| ~~internal generate call — the blocker~~ | removed by W1; the wrapper now delegates |
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
