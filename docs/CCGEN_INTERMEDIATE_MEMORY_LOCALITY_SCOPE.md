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

  **Rank-5 stress test (cc5 / CCSDTQP) — reinforces the verdict.** On the 59-op
  cc5 set the *exact oracle itself does not scale*: with footprints spanning
  7e-6 GB → 6e8 GB (11 orders), the fractional bound goes loose at large budgets
  and branch-and-bound degenerates — it ran 8+ min with no result before being
  killed. The fractional-LP upper bound is also uninformative here (worst
  best-of-both/LP gap ≈ 1.0, a pure artifact of taking a fraction of a 6e8 GB
  operator that integral greedy correctly cannot). Checking the real question
  directly — does density-greedy ever beat savings-greedy on cc5 — the answer at
  log-spaced budgets is **0.00%** (best-of-both ≡ flops-greedy there, like CCSDT).
  So greedy is not failing; the scary LP gap was purely the relaxation artifact.
  Net: neither exact bound is tractable-or-informative at rank 5, and greedy is
  provably fine where it can be checked. The CCSDTQ within-0.002%-of-exact result
  settles "greedy is enough"; rank 5 gives no structural reason to differ.
- **M2.2 — wire into emit (~S given M2.1). LANDED.**
  `emit_factorized_translation_unit(memory_budget_bytes=B)` selects via
  `select_best_of_both` (M2), taking precedence over the M1 per-operator guard /
  E1 top-k; those remain the path when a total budget is not given. Non-selected
  operators inline (E1 path). Verified: a 1 GB CCSDT budget emits exactly the
  best-of-both set (15 builders), Σ footprint ≤ budget, and the TU compiles.
  *Gate (in `CostModelTests`):* `test_emit_memory_budget_selects_best_of_both`
  (emitted builders == best-of-both, Σ bytes ≤ budget), `test_emit_memory_budget_compiles`.
  Scaling note (measured while validating): `manifold_operators` at cc5 (20,375
  quintuples terms) takes ~38 s and the residual ~90 s to generate — the emit
  selection is cheap, the per-term tree search is the cost at high rank.
- **M2.3 — measured verdict vs baseline (~S). LANDED — B1 answered with a
  number.** At a CCSDTQ budget in the divergence regime the joint selection
  (`select_best_of_both`) beats the flops-only baseline (B1, savings-greedy)
  **both ways at once**: at **850 GB it retains +5.68% more FLOP savings using
  LESS memory (691 GB vs 850 GB)** — 26 smaller operators instead of 15 big
  ones. That is the memory-blind→memory-aware inversion (B1) quantified
  end-to-end: the density ranking finds a strictly better set that the
  flops-only ranking cannot see. (The 16.7% figure from M2.0 is the gap *between
  the two keys*; the joint-vs-flops-only gain is up to 5.68%, since best-of-both
  takes the max and only improves where density wins.) *Gate (in `CCSDTQTests`):*
  `test_joint_beats_flops_only_baseline` (more savings, ≤ memory, > 5%, different
  pick at 850 GB).

**M2 verdict (measured, was the honest-ceiling case).** The exact knapsack is
**not worth building**: best-of-both-greedy is optimal to within 0.002% on
CCSDTQ. M2's real content is (a) the total-budget framing (M2.0), (b) running
*both* rankings and taking the max (M2.1), (c) wiring it into emit (M2.2), and
(d) that this beats the flops-only baseline by up to **5.68% savings at less
memory** where the keys diverge (M2.3). The "greedy is enough" outcome the scope
anticipated is the one that landed — but "run both keys" is a genuine,
measured win over the flops-only baseline, not a no-op.

### M3 — shape the emitted builder loop (~M)

**A deeper problem than B3 surfaced when reading the emitted `build_W`.** The
operator builders are emitted as ONE flat n-ary loop nest, not as a binary
contraction tree — `_complete_definition_summation` (E0.3) declares every
internal contraction index as a single fused nest. Concretely
`build_W_t2t2v_oooovv` emits

```
for i,j,k,l,b,c:  acc=0; for m,d,e: acc += t2(i,j,b,d)*t2(k,m,c,e)*oovv(l,m,d,e)
```

i.e. an `o⁵v⁴` triple-summed body — when the operator's OWN best contraction tree
is `o⁵v²`. Measured: **3 of the top-8 CCSDT builders are emitted above their
factored cost** (`W_t1t2v_oooovv` `o⁵v³`→`o⁵v²`, a ×V waste *inside* the operator
meant to save FLOPs). So M3 is two layers: **first factor the builder body, then
shape its locality** — you cannot tile a loop nest that should not exist. B3
(row-major, unshaped access) is the second layer.

Sub-steps:

- **M3.0 — builder-body factorization (~M, the load-bearing layer). LANDED.**
  `factored_builder_steps(spec)` decomposes an operator's definition into the
  pairwise contraction steps its best tree gives (inner scratch tensors + final
  assembly); `emit_factorized_translation_unit(factor_builder_bodies=True)` (via
  `_emit_intermediate_builder(factor_body=)`) emits them as scratch-step loops
  instead of one flat n-ary nest. Measured on CCSDT: **10 of 24 builders improve**
  — `W_t2t2v_oooovv` total-degree o⁹→o⁷, `W_t1t1t1v_ooov` o⁷→o⁵, `W_t1t2v_oooovv`
  o⁵v³→o⁵v². Scratch is ~0.3× the operator's own footprint, so it is a FLOP win at
  no peak-memory cost. Default off → byte-identical flat emit. *Gate (in
  `CostModelTests`):* `test_builder_steps_cut_flat_cost` (≥8 improve, none worse),
  `test_builder_steps_are_exact` (leaves + summed-index consumption preserved),
  `test_factored_builder_tu_compiles` (scratch tensors declared/typed, compiles).
- **M3.1 — stride metric (~S). LANDED.** `builder_stride_score(spec)` (via
  `step_stride_penalty` / `_factor_access_stride`) scores a builder's
  innermost-loop access: for each factor, the distance of the innermost loop
  index from that factor's LAST axis (0 = unit stride, k>0 = strided, absent =
  loop-invariant/hoistable), summed over factors and weighted by loop volume.
  Innermost loop = the step's last summed index, the emitter's current
  (alphabetical) order. **The metric proves M3.2 has a free lever**: e.g. the
  `W_t2t2v_oooovv` X1 step (`t2·v` summed over `m,e`) scores **0** with `m`
  innermost (both factors read `m` last) but **1.08e11** with the emitter's
  current `e` — reordering the summed loops alone (no algebra change) removes the
  penalty. *Gate (in `CostModelTests`):* `test_stride_metric_ranks_unit_below_strided`
  (unit-stride fixture scores 0, strided scores higher),
  `test_builder_stride_score_is_baseline` (nonzero baseline + a step that
  benefits from inner-index reorder exists).
- **M3.2 — loop-order shaping (~M, answers B3). LANDED.** `stride_ordered_summed`
  / `stride_inner_index` reorder each step's summed loops so the min-stride index
  is innermost; `factored_builder_steps(stride_order=True)` applies it and the
  emitter rides it on `factor_builder_bodies=True`. Measured on CCSDT: the
  aggregate stride penalty drops **55% (3.4e14 → 1.5e14)**, 9/24 builders improve
  — from pure loop reordering, no algebra change, no cost. `builder_stride_score(
  reorder=True)` scores the shaped version. *Gate (in `CostModelTests`):*
  `test_stride_reorder_reduces_penalty` (> 30% aggregate cut, never worse),
  `test_stride_reorder_is_exact` (only the summed-index ORDER changes — same set,
  factors, coeff, free — so the sum is unchanged), `test_stride_ordered_builder_tu_compiles`.
  Default off → byte-identical. **B3 answered.** (Not attempted: `memory_layout`/
  `blocking_hint` tiling — the loop-order lever alone captures the measurable
  static-stride win; tiling is a compiled-binary concern, the E2 boundary.)

**Honest ceiling for M3.** The stride metric is a static model of the access
pattern, not a measured cache-miss rate (that needs the compiled binary — same
boundary as E2). M3.0 is the real FLOP win (factoring the builder body); M3.1/M3.2
are the locality layer B3 named, and their value is bounded by what a static
stride model can show — a model improvement, not a wall-clock one.

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
