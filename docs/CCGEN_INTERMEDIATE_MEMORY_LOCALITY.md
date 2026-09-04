# ccgen Intermediate Selection: FLOP, Memory, and Cache Locality

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**Does the emitted CC intermediate selection jointly optimize memory and cache locality, or does its FLOP-only ranking leave savings on the table?**

## Short answer

It left savings on the table, and now it can reach the joint optimum. The
factorized-intermediate emit path (E0–E1, see `CCGEN_HIGHER_OPERATOR_REUSE.md`)
originally selected and materialized dressed intermediates by FLOP savings
alone — never their memory footprint or the cache behavior of their build
loops. Three memory-blind defects (B1–B3) were found, each is now fixed
(M1–M3) and gated. At a fixed CCSDTQ budget the optimized emit beats the
baseline on FLOP savings **and** memory **and** loop stride, simultaneously:

| at an 850 GB budget (CCSDTQ, O=30/V=100) | baseline (FLOP-only) | M1–M3 optimized |
|---|---|---|
| operators materialized | 15 | 26 (smaller ones) |
| FLOP savings retained | 1.40e18 | **1.48e18 (+5.68%)** |
| total memory used | 850 GB | **691 GB (−19%)** |
| builder loop stride penalty | 1.5e16 | **2.3e14 (−98%)** |

More savings, less memory, better locality — the three were not in tension
once the selection stopped ignoring two of them.

## Where the logic lives

- `python/ccgen/optimization/factorize.py` — selection and factorization logic
- `python/ccgen/emit/planck_tensor_cpp.py` — `emit_factorized_translation_unit`
- `python/ccgen/tests/test_factorize.py` — 70 tests, including the verdict test `test_optimized_beats_baseline_all_axes`

## What invariants matter

### 1. Selection must weigh memory, not just FLOP savings

E1 originally ranked operators by `savings = (uses−1)·build_flops`; nothing
read bytes. The FLOP-only and savings-per-byte rankings pick **different**
top operators — the FLOP-only winner `W_t2t2v_oooovv` is 64.8 GB, the
density winner `W_t1t2v_ooov` is 0.02 GB (3000× smaller) for a higher
flops/byte.

Design rule:

- Any new selection path must go through `select_best_of_both(total_bytes)`
  (M2) — a joint selection under a total memory budget that runs both the
  savings- and density-greedy fills and takes the higher-savings set —
  rather than reintroducing a FLOP-only ranking. Measured against an exact
  0/1 knapsack (branch-and-bound, not an integer-GB-weight DP, which zeros
  the small operators), best-of-both is within 0.002% of optimal on
  CCSDTQ across a dense sweep, so no exact solver is warranted.

### 2. Every candidate operator needs a footprint feasibility guard

The highest-savings operators can be unmaterializable at scale: the rank-8
`W_*_ooooovvv` intermediates are 194,400 GB each at CCSDTQ (5.8e8 GB at
cc5), yet a FLOP-only ranking would still select them.

Design rule:

- `select_operators_by_savings(max_operator_bytes=)` (M1) must gate every
  candidate before selection — an over-budget operator is inlined, never
  emitted. The guarded rewrite must re-expand exactly (inlining the big
  operators preserves the algebra); verify this on any new operator family
  before trusting the guarded TU compiles at scale.

### 3. Builder loops must be shaped to their factored cost, not left as a flat n-ary nest

Two problems surfaced from reading the emitted `build_W` bodies. First, the
builder bodies were emitted as one flat n-ary loop nest, so an operator
meant to *save* FLOPs was itself computed above its factored cost —
`W_t2t2v_oooovv` at total degree o⁹ when it factors to o⁷. Second, the
emitter ordered summed loops alphabetically rather than by stride, so a
step scoring 0 stride penalty under one inner index scored 1.08e11 under
the emitter's alphabetical choice.

Design rule:

- New builder emission must go through `factored_builder_steps` (M3.0,
  emitting each builder as its own contraction tree of scratch-step
  pairwise contractions) and `stride_ordered_summed` (M3.2, reordering each
  step's summed loops so the min-stride index is innermost). Both are
  provably safe to apply — factoring only changes the FLOP path, not the
  algebra, and reordering only permutes summed indices (same set, factors,
  coefficient, free indices), so the sum is provably unchanged. Measured:
  10/24 CCSDT builders drop to their factored cost at ~0.3× scratch memory
  (a FLOP win at no peak-memory cost), and the aggregate stride penalty
  drops −55% on CCSDT from pure loop reordering.

## What was built

All behind flags on `emit_factorized_translation_unit`, default off →
byte-identical to the FLOP-only emit:

1. `memory_budget_bytes=B` — M1/M2 joint selection under a total footprint
   budget (`select_best_of_both`), the B1/B2 fix.
2. `factor_builder_bodies=True` — M3.0 factored builder bodies + M3.2
   stride-ordered loops, the B3 fix.

Supporting API: `operator_bytes` / `operator_density` / `footprint_inventory`
(M0, the reproducible baseline), `operator_savings` (E1),
`builder_stride_score` (M3.1, `reorder=` switches baseline vs shaped).

Every step is algebra-exact by construction and gated: inlining an
over-budget or non-kept operator re-expands to the original term (M1/M2),
and both builder-body factorization and loop reordering only rearrange a
provably order-independent sum (M3).

## Validation strategy that should remain in place

- `test_factorize.py`'s 70 tests, in particular
  `test_optimized_beats_baseline_all_axes` — the joint FLOP/memory/stride
  verdict test.
- Re-expansion exactness checks on the M1/M2 inlining path whenever a new
  operator family is added.
- The order-independence proof for M3.2's loop reordering should be
  re-verified (not just re-asserted) if the reorder logic changes, since
  its safety rests specifically on permuting only summed indices.

## What this does not answer

- **The cost model is symbolic, not measured.** Bytes, FLOP degree, and the
  stride metric are computed from index-space sizes — the gates are model
  improvements, not wall-clock ones. A real cache-miss rate or runtime
  needs the compiled binary (the same E2 boundary as the FLOP investigation's
  numeric energy run).
- **No memory-layout tiling.** M3.2 shapes loop *order*; `memory_layout` /
  `blocking_hint` tiling is not attempted — the loop-order lever captures
  the measurable static-stride win, and tiling's payoff is a
  compiled-binary concern.
- **Greedy, not exact, selection — by measurement.** The exact knapsack is
  within 0.002% of best-of-both on CCSDTQ and does not even terminate at
  cc5 (footprints span 11 orders); "run both greedy keys" is the whole
  selection.
- **Cross-operator sharing is out of scope.** Each operator's footprint and
  loops are shaped independently; sharing scratch or tiles across operators
  (where the largest wins compound) is the harder follow-on.

## Remaining architecture concern

Cross-operator scratch/tile sharing (noted above) is the one identified
follow-on with real potential payoff, since the largest operators' savings
would compound. Not started; no scope has been written for it.

## What this reuses

`IntermediateSpec.estimated_bytes` / `selection_density` / `memory_layout` /
`blocking_hint` (the memory fields that existed but the factorizer never
read), the E1 selection + inline path (`select_operators_by_savings`,
`rewrite_term_factorized`), the tree search (`best_contraction_tree_full`,
reused one level down for builder bodies), and `emit_planck_translation_unit`.
See `CCGEN_HIGHER_OPERATOR_REUSE.md` for the FLOP investigation this layers
memory onto, and `cc_canonical_fock_only` for the canonical-Fock invariant
this pipeline depends on.
