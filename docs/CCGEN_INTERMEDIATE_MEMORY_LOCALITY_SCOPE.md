# Do the emitted CC intermediates jointly optimize memory and cache locality?

**The one question.** The factorized-intermediate emit path (E0–E1, see
`CCGEN_HIGHER_OPERATOR_REUSE.md`) selects and materializes dressed intermediates
by **FLOP savings alone** — never their memory footprint or the cache behavior of
their build loops. Does that leave a joint FLOP / memory / locality optimum on
the table, and can the emit path reach it?

**Answer: it did leave one, and now it can reach it.** The baseline was
FLOP-greedy and memory-blind on three measurable axes (B1–B3 below); each is now
fixed and gated. At a fixed CCSDTQ budget the optimized emit beats the baseline
on FLOP savings **and** memory **and** loop stride, simultaneously:

| at an 850 GB budget (CCSDTQ, O=30/V=100) | baseline (FLOP-only) | M1–M3 optimized |
|---|---|---|
| operators materialized | 15 | 26 (smaller ones) |
| FLOP savings retained | 1.40e18 | **1.48e18 (+5.68%)** |
| total memory used | 850 GB | **691 GB (−19%)** |
| builder loop stride penalty | 1.5e16 | **2.3e14 (−98%)** |

More savings, less memory, better locality — the three were not in tension once
the selection stopped ignoring two of them. Everything is landed in
`python/ccgen/optimization/factorize.py` + the emitter, gated by
`python/ccgen/tests/test_factorize.py` (70 tests); the verdict itself is
`test_optimized_beats_baseline_all_axes`.

---

## The three baseline defects, and their fixes

The baseline emit path had three memory-blind behaviors, each measured, each now
answered. (Sizes O=30/V=100 throughout; footprint = the materialized tensor's
block in doubles.)

### B1 — selection ignored memory → **fixed by M1/M2**

E1 ranked operators by `savings = (uses−1)·build_flops`; nothing read bytes. The
FLOP-only and savings-per-byte rankings pick **different** top operators — the
FLOP-only winner `W_t2t2v_oooovv` is 64.8 GB, the density winner `W_t1t2v_ooov`
is 0.02 GB (3000× smaller) for a higher flops/byte.

- **M1** (`select_operators_by_savings(max_operator_bytes=)`): a per-operator
  footprint guard — an over-budget operator is inlined, never emitted.
- **M2** (`select_best_of_both(total_bytes)`): joint selection under a *total*
  memory budget = run both the savings- and density-greedy fills, take the
  higher-savings set. Measured against an exact 0/1 knapsack (branch-and-bound,
  not an integer-GB-weight DP — that zeros the small operators), best-of-both is
  within **0.002%** of optimal on CCSDTQ across a dense sweep, so no exact solver
  is warranted. At 850 GB it beats the FLOP-only baseline by **+5.68% savings
  using less memory** — the memory-blind→memory-aware inversion, quantified.

### B2 — no footprint feasibility guard → **fixed by M1**

The highest-savings operators are unmaterializable at scale: the rank-8
`W_*_ooooovvv` intermediates are **194,400 GB** each at CCSDTQ (5.8e8 GB at cc5),
yet the FLOP-only ranking would still select them. M1's guard drops them; the
guarded rewrite re-expands exactly (inlining the big operators preserves the
algebra), and the guarded TU compiles.

### B3 — builder loops unshaped → **fixed by M3**

Two problems surfaced from reading the emitted `build_W`. First, the builder
bodies were emitted as one **flat n-ary loop nest**, so an operator meant to
*save* FLOPs was itself computed above its factored cost — `W_t2t2v_oooovv` at
total degree o⁹ when it factors to o⁷.

- **M3.0** (`factored_builder_steps`): emit each builder as its own contraction
  tree (scratch-step pairwise contractions). **10/24 CCSDT builders drop to their
  factored cost** at ~0.3× scratch memory — a FLOP win at no peak-memory cost.
- **M3.1** (`builder_stride_score`): a static metric — for each factor, the
  distance of the innermost loop index from that factor's last (unit-stride) axis,
  volume-weighted. It proved a free lever exists: the emitter ordered summed loops
  alphabetically, so a step scoring 0 under one inner index scored 1.08e11 under
  the emitter's choice.
- **M3.2** (`stride_ordered_summed`): reorder each step's summed loops so the
  min-stride index is innermost. **Aggregate stride penalty −55%** on CCSDT, from
  pure loop reordering — the reorder only permutes summed indices (same set,
  factors, coeff, free), so the sum is provably unchanged.

---

## What is built

All behind flags on `emit_factorized_translation_unit`, default off →
byte-identical to the FLOP-only emit:

- `memory_budget_bytes=B` — M1/M2 joint selection under a total footprint budget
  (`select_best_of_both`), the B1/B2 fix.
- `factor_builder_bodies=True` — M3.0 factored builder bodies + M3.2 stride-ordered
  loops, the B3 fix.

Supporting API: `operator_bytes` / `operator_density` / `footprint_inventory`
(M0, the reproducible baseline), `operator_savings` (E1), `builder_stride_score`
(M3.1, `reorder=` switches baseline vs shaped).

Every step is **algebra-exact** by construction and gated: inlining an
over-budget or non-kept operator re-expands to the original term (M1/M2), and
both builder-body factorization and loop reordering only rearrange a provably
order-independent sum (M3).

---

## What is NOT answered here

- **The cost model is symbolic, not measured.** Bytes, FLOP degree, and the
  stride metric are computed from index-space sizes — the gates are model
  improvements, not wall-clock ones. A real cache-miss rate or runtime needs the
  compiled binary (the same E2 boundary as the FLOP investigation's numeric
  energy run).
- **No memory-layout tiling.** M3.2 shapes loop *order*; `memory_layout` /
  `blocking_hint` tiling is not attempted — the loop-order lever captures the
  measurable static-stride win, and tiling's payoff is a compiled-binary concern.
- **Greedy, not exact, selection — by measurement.** The exact knapsack is within
  0.002% of best-of-both on CCSDTQ and does not even terminate at cc5 (footprints
  span 11 orders); "run both greedy keys" is the whole selection.
- **Cross-operator sharing is out of scope.** Each operator's footprint and loops
  are shaped independently; sharing scratch or tiles across operators (where the
  largest wins compound) is the harder follow-on.

## What this reuses

`IntermediateSpec.estimated_bytes` / `selection_density` / `memory_layout` /
`blocking_hint` (the memory fields that existed but the factorizer never read),
the E1 selection + inline path (`select_operators_by_savings`,
`rewrite_term_factorized`), the tree search (`best_contraction_tree_full`, reused
one level down for builder bodies), and `emit_planck_translation_unit`. See
`CCGEN_HIGHER_OPERATOR_REUSE.md` for the FLOP investigation this layers memory
onto, and the `cc_canonical_fock_only` invariant.
