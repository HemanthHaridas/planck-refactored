# ccgen Dressed-Operator Recognition Cost

Canonical status now lives in:

- `vault/Status/Completion.md`

This file answers one architecture question:

**What determines the time `_dress_operator_equations` takes, and why is it now linear-ish in manifold size rather than quadratic?**

## Short answer

Recognition cost is governed by `n_hypotheses × n_terms`, and the second factor used to be paid *per hypothesis* because the residual's ERI-canonical multiset — an invariant of the whole search — was rebuilt on every consistency check. Hoisting it out is the whole difference between rank 4 being unusable and rank 4 taking a minute.

## Where the logic lives

- `python/ccgen/optimization/dressing.py` — `hypothesis_is_consistent`, `find_operator_occurrences`
- `python/ccgen/tests/test_dressing_scaling.py` — the gate

## What invariants matter

### 1. A fixed-for-the-search value must be computed once, not once per hypothesis

`assemble_dressed_equation` does three things per manifold: reconcile operator scales, compute tau-overlap corrections, and find operator occurrences. The third dominates, and inside it the loop is:

```
for anchor in collect_fragment_occurrences(op, terms):      # candidate sites
    for hyp in enumerate_hypotheses(op, anchor, ...):       # candidate W*rest forms
        if hypothesis_is_consistent(hyp, terms):            # <- the expensive test
```

`hypothesis_is_consistent` expands a hypothesis to primitives and checks every ERI-canonical key against the raw residual, requiring matching sign and `|hyp_coeff| ≤ |raw_coeff|`. To do that it needs `raw_multiset(terms)` — the residual's key→coefficient map. That map is an invariant of the search: `terms` is the whole manifold and never changes across hypotheses. It was nevertheless recomputed inside every call: measured **7,461** calls on `ccsdt` triples, at 0.036 s per rebuild, ≈ **270 s of a ~300 s manifold** spent recomputing a fixed value.

So the cost was the product of two quantities that both grow with rank — hypotheses and terms — where only the first is intrinsic.

Design rule:

- When a loop repeatedly recomputes a value that does not depend on the loop variable, hoist it out and pass it in, rather than caching it behind a dict — a parameter states the invariant directly; a cache keyed on large term tuples hides it behind a structure whose lifetime spans every manifold and method in the process.

### 2. A quadratic signature shows up as time growing faster than the input, not as any one function being slow

~5.5× the terms cost ~22× the time before the fix — the quadratic signature. Afterwards the call count is flat (19/21/19, independent of term count), and that flatness is the actual property; the timing follows from it.

Two explanations that looked right and were not, both worth recording because both are plausible enough to be proposed again:

- `_eri_canonical` is not the hotspot. It shows 3,027,528 calls and **864 s cumulative** in the triples profile — the largest number in the listing. That is its callees' time. Memoizing it achieves a 97.9 % hit rate (192,490 hits / 4,093 misses) and buys only **6 %** (15.2 → 14.3 s). Cumulative profile time is not a fix target.
- There is no second hot function. The self-time ranking is diffuse: `relabel_term_dummies` 5.9 s, `reindex_tensors` 3.0 s, `canonicalize_tensor` 2.9 s, then a long tail of 45 M-call primitives (`Index.__hash__`, `dict.get`, `_antisym_slot_key`) — 228 M calls for 45 s with nothing individually slow.

Design rule:

- When no inner function is slow in a profile, but the overall cost still scales worse than linearly with input size, look for a redundant *outer* loop recomputing something fixed — not for a hotter leaf function. Diffuseness in the self-time ranking is itself the tell.
- Do not trust cumulative profile time as evidence of where to optimize; it reflects callees, not the function's own cost.

### 3. A regression gate on a scaling defect should pin a call count, not a wall-clock threshold

`test_dressing_scaling` asserts that `raw_multiset` is called a bounded number of times, not once per hypothesis, and that the count does not scale with manifold size.

Design rule:

- A wall-clock threshold is flaky on shared machines and diagnostically useless when it fails. Pin a deterministic call count instead — it names the defect directly if it returns.

### 4. Removing one quadratic factor does not prove the search is no longer quadratic in any dimension

One quadratic factor was removed, not proven absent. The hypothesis count itself (7,461 at rank-3 triples) still grows with rank, and pruning it would be a correctness-affecting change — it can alter *which* operators are recognized, unlike this hoist.

Design rule:

- Do not conflate "the redundant recomputation is fixed" with "the algorithm's scaling in every dimension is understood." Rank 4 at 61.6 s means the hypothesis-count question is not pressing yet; a rank-5 attempt is where it would resurface.

## What was measured

Per manifold, `ccsdt`, diagram engine, canonical Fock:

| manifold | terms | before | after | `raw_multiset` calls |
|---|---|---|---|---|
| singles | 12 | 0.1 s | 0.1 s | 139 → 19 |
| doubles | 73 | 4.3 s | 2.0 s | 2452 → 21 |
| triples | 399 | 94.7 s | 6.9 s | ~7461 → 19 |

End to end: rank-3 dressing 293.7 s → **9.1 s**; rank 4 went from **abandoned after >25 minutes** to **61.6 s**. That last number is why rank-4 dressing exists as an option at all.

Generation is not part of this story. The diagram engine produces all four `ccsdtq` manifolds (3172 terms) in 3.5 s. Every second of the old cost was recognition.

## What was fixed

1. `raw_multiset` is computed once in `find_operator_occurrences` and threaded into `hypothesis_is_consistent` as an optional `raw=` argument (`None` preserves standalone behaviour for one-off callers), rather than recomputed once per hypothesis.
2. Confirmed the fix is behavior-neutral: `raw_multiset` is pure, so hoisting cannot change results, and output is byte-identical.
3. Ruled out two plausible alternative fixes by measurement rather than by argument: memoizing `_eri_canonical` (6 % gain) and searching for a second hot leaf function (none found — the self-time ranking is diffuse, which is itself informative).

## Validation strategy that should remain in place

- `python/ccgen/tests/test_dressing_scaling.py` — asserts `raw_multiset` is called a bounded, size-independent number of times

## Remaining architecture concern

The hypothesis count itself (7,461 at rank-3 triples) still grows with rank and has not been pruned; doing so would be a correctness-affecting change, since it can alter which operators are recognized. Rank 4 at 61.6 s makes this non-pressing today, but a rank-5 attempt is where the question would resurface.
