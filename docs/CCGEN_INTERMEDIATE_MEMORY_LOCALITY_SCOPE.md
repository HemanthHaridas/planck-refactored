# Do the emitted CC intermediates jointly optimize memory and cache locality?

**The one question.** The factorizer's emit path (E0–E1, see
`CCGEN_HIGHER_OPERATOR_REUSE.md`) selects and materializes dressed intermediates
by **FLOP savings alone**. It never consults the operator's memory footprint or
the cache behavior of its build loop:

> **Can the intermediate selection and emission be made to jointly optimize FLOP
> savings, memory footprint, and cache locality — instead of FLOPs alone — and
> what does that change about which operators get materialized and how their
> `build_W` loops are written?**

This is a **research investigation**, not a committed feature. A "the flops-only
selection is already near the memory/locality optimum" answer is a valid result.
The baseline below shows it is not.

---

## Baseline: the current implementation, measured

The current emit path (`factorize.py`: `select_operators_by_savings` →
`rewrite_term_factorized` → `emit_planck_translation_unit`) has three properties,
all measured on the diagram/canonical-Fock residuals.

### B1 — selection ignores memory entirely

E1 ranks by `operator_savings = (uses−1)·build_flops`. Grepping `factorize.py`
for `byte|memory|layout|cache|locality|blocking` matches exactly one line — the
word "re-blocking" in a comment, not a single line of memory logic. The
machinery to do so already exists on `IntermediateSpec` and is simply unused by
the factorizer: `estimated_bytes`, `selection_density` (= `saved_flops / bytes`),
`memory_layout`, `blocking_hint`.

**The flops-only and memory-aware rankings disagree completely** (CCSDT triples,
O=30, V=100):

| rank | by SAVINGS (E1 today) | footprint | by SAVINGS/BYTE |
|---|---|---|---|
| 1 | `W_t2t2v_oooovv` | 64.8 GB | `W_t1t2v_ooov` (0.02 GB) |
| 2 | `W_t1t3v_oooovv` | 64.8 GB | `W_t1t1t1v_ooov` (0.02 GB) |
| 3 | `W_t1t1t2v_oooovv` | 64.8 GB | `W_t3v_ooov` (0.02 GB) |

E1's top-5 are all 64.8 GB rank-6 operators; the memory-aware top-5 are rank-4
operators storing **~3000× less** for a competitive flops/byte. Neither ranking
alone is right — the flops-only picks maximize FLOP savings but blow any real
memory budget; the density picks fit but leave FLOP savings on the table.

### B2 — the highest-savings operators are unmaterializable at scale

The rank-8 operators the factorizer derives (`W_*_ooooovvv`, o⁵v³) are **194,400
GB each** at O=30/V=100 — they cannot be stored, period. E1 would still select
them if their savings ranked high (they recur 48–120×). The current path has no
guard that an operator's tensor fits before hoisting it; it trusts that big-flops
⇒ worth-storing, which is false for high-rank blocks.

### B3 — every build_W is naive row-major, no locality shaping

The emitter defaults `memory_layout="row_major"` and sets no `blocking_hint`. The
factorizer never overrides either. So each `build_W` is whatever loop nest
`node_to_term` produces in index-declaration order — the contraction index may be
the innermost loop or the outermost, with no attention to stride. Concretely, a
`build_W` like `W_t2t2v_oooovv` accumulates over one occupied index in the inner
loop while striding a rank-6 output; the access pattern is unshaped. There is no
tiling, no layout choice matched to the contraction, no reuse of a warm cache
line across the `usage_count` reference sites.

**Baseline summary:** the emit path is FLOP-greedy and memory-blind. It (B1)
selects operators without a byte cost, (B2) has no footprint feasibility guard,
and (B3) emits unshaped row-major loops. The `IntermediateSpec` layer already
carries the fields needed to fix all three; nothing in `factorize.py` uses them.

---

## What "jointly optimize" must mean (make it falsifiable)

Pin it to a cost model + a gate, exactly as the FLOP investigation did.

An intermediate emission is **memory-locality-optimizing** iff the pipeline:
1. **selects** operators by a joint objective that trades FLOP savings against
   storage (e.g. `selection_density` under a memory budget), not FLOPs alone —
   and the selection changes measurably vs the baseline;
2. **guards feasibility** — never materializes an operator whose tensor exceeds a
   footprint budget, inlining it instead (E1's inline path already exists);
3. **shapes locality** — sets `memory_layout` / `blocking_hint` per operator so
   the emitted `build_W` loop has a defensible access pattern (contraction index
   inner, output written with unit stride where possible), verified by a stride
   metric on the emitted loop, not by eye;

while remaining **algebra-exact** (the E0.1 re-expansion gate: any operator not
materialized is inlined and the term still re-expands to the original).

---

## Scope — small verifiable steps

The engine is a byte/stride cost model layered onto the existing savings model;
each step returns a concrete measurement against the baseline above.

### M0 — footprint + density inventory (~S). LANDED.
`operator_bytes(spec, n_occ, n_vir)`, `operator_density(spec, …)`, and
`footprint_inventory(specs, …)` in `factorize.py` tabulate every emittable
operator's savings, footprint, and flops/byte at parametrized O/V (not the
hardcoded 30/100 in `IntermediateSpec.estimated_bytes` — so the inventory can
sweep). Reproduces the baseline exactly: CCSDT rank-6 `oooovv` = **64.8 GB**,
CCSDTQ rank-8 `ooooovvv` = **194,400 GB**; savings-top (`W_t2t2v_oooovv`, 64.8
GB) ≠ density-top (`W_t1t2v_ooov`, 0.02 GB — 3000× smaller, highest density).
*Gate (in `CostModelTests`):* `test_footprint_reproduces_baseline` (rank-6 =
64.8 GB), `test_savings_and_density_rankings_disagree` (B1: different top
operators, savings-winner has more bytes), `test_operator_bytes_scales_with_sizes`
(footprint is size-parametrized). One inconsistency surfaced and is documented:
`operator_savings` defaults to `Cost.flops()`'s O=10/V=50 while `operator_bytes`
defaults to O=30/V=100 — `operator_density` passes MATCHED sizes to both, and the
ranking disagreement holds at matched sizes.

### M1 — feasibility guard (~S). LANDED.
`select_operators_by_savings(max_operator_bytes=, n_occ=, n_vir=)` filters out any
operator whose materialized tensor exceeds the budget BEFORE ranking, so it is
inlined via the existing E1 keep-set path rather than emitted as an un-storable
`build_W`; `emit_factorized_translation_unit(max_operator_bytes=)` threads it.
Measured (CCSDT, O=30/V=100): a 1 GB budget drops the 7 over-budget operators
(all 64.8 GB rank-6) from 24 → 17 kept (largest kept 0.80 GB); a 0.1 GB budget
keeps 12 (largest 0.07 GB). The guarded TU compiles and the guarded rewrite
re-expands exactly (0/399 failures — inlining the big operators preserves the
algebra). *Gate (in `CostModelTests`):* `test_footprint_guard_drops_over_budget`
(nothing over budget survives), `test_footprint_guard_is_exact`,
`test_footprint_guarded_tu_compiles`. Answers B2.

### M2 — joint selection objective (~M, the real modeling piece)

Replace the per-operator M1 guard + flops-only rank with a **total-memory
budget**: maximize FLOP savings subject to `Σ bytes(kept) ≤ B` — a 0/1 knapsack
on `(savings, bytes)`. M1 answered "does each operator fit"; M2 answers "given a
fixed total memory, which SET maximizes savings."

**Measured relevance first (grounds the whole step).** How much M2 can beat the
existing greedy is not assumed — it is measured, and the answer is
rank-dependent:
- **CCSDT (24 ops): almost nothing.** flops-greedy and density-greedy diverge at
  exactly ONE budget point across a full sweep, by Δ=2e11 / 8.8e16 = **0.0002%**.
  The operators cluster hard by footprint (many at 64.8 GB, a few tiny), so the
  budget almost always fits a whole cluster or not — no room to trade. For CCSDT,
  "the flops-only selection is already at the memory optimum" is the honest
  result.
- **CCSDTQ (43 ops): real.** Footprints span 7e-6 GB → 194,000 GB across **14
  tiers**, so the budget frequently forces a genuine trade: greedy rankings
  diverge in **66/286 budget points (23%)**, max relative savings difference
  **16.7%**. This is where a joint objective earns its place.

So M2 is a CCSDTQ-and-up feature; on CCSDT it should measurably match the baseline
(a correctness check, not a win).

Sub-steps:

- **M2.0 — total-budget feasibility + greedy baseline (~S). LANDED.**
  `select_under_memory_budget(specs, total_bytes, key="savings"|"density")`
  greedily fills a TOTAL budget (`Σ bytes ≤ total_bytes`) in `key` order — the two
  rankings M2.1 is measured against. Reproduces the measured divergence exactly:
  CCSDTQ **66/286 budgets divergent, max 16.7%** savings gap between the keys;
  CCSDT < 1% (operators cluster by footprint). *Gate:* `test_total_budget_respected`
  (Σ bytes ≤ budget, both keys), `test_ccsdt_keys_barely_diverge` (< 1% gap — flops
  greedy already near the memory optimum on CCSDT), `test_ccsdtq_keys_diverge_materially`
  (> 10% of budgets divergent, worst-case > 10% — where M2.1 earns its place).
- **M2.1 — is an exact solver needed? MEASURED: NO.** Before building
  branch-and-bound, the question "does an exact 0/1 knapsack beat greedy" was
  measured directly with a correct exact solver (branch-and-bound with a
  fractional-relaxation bound — **NOT** an integer-GB-weight DP, which zeros the
  3000×-smaller high-density operators; verified that DP undercounts 3× on
  CCSDT). Result across a dense CCSDTQ budget sweep (273 budgets): the exact
  optimum beats **best-of-both-greedy** (`max(savings-greedy, density-greedy)`) in
  only **5/273 budgets, by ≤ 0.002%** — noise. The individual keys diverge (M2.0's
  66/286), but at every such budget the exact optimum equals the *better* of the
  two greedies. **So the deliverable is not a knapsack — it is
  `select_best_of_both(specs, total_bytes)` = run both M2.0 greedies, take the
  higher-savings set.** Branch-and-bound survives only as the *test oracle*.
  *Gate:* `select_best_of_both` is within ≤ 0.01% of the branch-and-bound optimum
  across the CCSDTQ sweep, and ≥ each individual greedy; on CCSDT it equals the
  flops-greedy baseline.
- **M2.2 — wire into emit (~S given M2.1).** `emit_factorized_translation_unit`
  takes `memory_budget_bytes=` (total) selecting via `select_best_of_both`,
  falling back to the M1 per-operator guard / E1 top-k when not given.
  Non-selected operators inline (E1 path). *Gate:* the budgeted TU compiles,
  re-expands exactly (E0.1), and its materialized `Σ bytes ≤ budget`.
- **M2.3 — measured verdict vs baseline (~S).** On a CCSDTQ budget in the
  divergence regime, report FLOP savings retained by `select_best_of_both` vs
  flops-greedy alone (the B1 selection), and the memory each uses. *Gate:* the
  joint selection retains ≥ flops-greedy's savings at ≤ its memory, and differs
  from the flops-only B1 pick — B1 answered with a number.

**M2 verdict (measured, was the honest-ceiling case).** The exact knapsack is
**not worth building**: best-of-both-greedy is optimal to within 0.002% on
CCSDTQ. M2's real content is (a) the total-budget framing (M2.0), (b) running
*both* rankings and taking the max (M2.1), and (c) that this differs from the
flops-only baseline where the keys diverge (23% of CCSDTQ budgets). The
"greedy is enough" outcome the scope anticipated is the one that landed.

### M3 — locality shaping of the emitted loop (~M)
For each materialized operator, choose `memory_layout` + `blocking_hint` from the
contraction structure (contraction index → inner loop, tile the reused
dimensions), and thread them through `_emit_intermediate_builder`. *Gate:* a
stride/access-pattern metric on the emitted `build_W` improves vs the baseline
row-major loop; the TU still compiles; energy-equivalence unchanged (structural).
Answers B3.

### M4 — measured joint verdict (~S, only if M1–M3 land)
Report, on a fixed budget, the baseline vs joint-optimized emit: operators
materialized, total bytes, FLOP savings retained, and the loop-stride metric. The
verdict is whichever the numbers show — "joint optimization retains X% of FLOP
savings at Y% of the baseline memory with Z better stride," or "the flops-only
selection was already near-optimal."

---

## Honest ceiling

- **The cost model is symbolic.** Bytes and stride are computed from index-space
  sizes, not measured on hardware; the gate is a model improvement, not a wall-
  clock one. A real cache-miss measurement needs the compiled binary (out of
  scope, same boundary as the FLOP investigation's E2 energy run).
- **Optimal joint selection is a knapsack** — NP-hard in general, but the operator
  set is small (24 CCSDT, 43 CCSDTQ) so exhaustive/greedy is tractable at these
  ranks; global cross-operator layout co-optimization (shared tiles across
  operators) is the harder follow-on, out of scope here.
- **Locality shaping is per-operator, not cross-kernel.** Matching an operator's
  layout to how every consuming kernel reads it (so the warm cache line survives
  across reference sites) is the deeper win and is not attempted at M3.

## What this reuses

- `IntermediateSpec.estimated_bytes` / `selection_density` / `memory_layout` /
  `blocking_hint` / `with_layout_hints` — the memory fields already present and
  currently unused by the factorizer.
- `select_operators_by_savings` / `operator_savings` / `rewrite_term_factorized`
  (with `keep_operators`) — the E1 selection + inline path M1/M2 extend.
- `emit_planck_translation_unit` / `_emit_intermediate_builder` — the emit path
  M3 threads layout hints through.
- The E0.1 re-expansion exactness gate — every step must stay algebra-exact.

Baseline and machinery: `CCGEN_HIGHER_OPERATOR_REUSE.md` (the FLOP investigation
this layers memory onto) and the `cc_canonical_fock_only` invariant.
