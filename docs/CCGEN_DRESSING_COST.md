# Why does dressed-operator recognition cost what it costs?

Answers one question: **what determines the time `_dress_operator_equations` takes, and why is it
now linear-ish in manifold size rather than quadratic?**

Short answer: recognition cost is governed by `n_hypotheses × n_terms`, and the second factor used
to be paid *per hypothesis* because the residual's ERI-canonical multiset — an invariant of the whole
search — was rebuilt on every consistency check. Hoisting it out is the whole difference between
rank 4 being unusable and rank 4 taking a minute.

---

## The cost model

`assemble_dressed_equation` does three things per manifold: reconcile operator scales, compute
tau-overlap corrections, and find operator occurrences. The third dominates, and inside it the loop
is:

```
for anchor in collect_fragment_occurrences(op, terms):      # candidate sites
    for hyp in enumerate_hypotheses(op, anchor, ...):       # candidate W*rest forms
        if hypothesis_is_consistent(hyp, terms):            # <- the expensive test
```

`hypothesis_is_consistent` expands a hypothesis to primitives and checks every ERI-canonical key
against the raw residual, requiring matching sign and `|hyp_coeff| ≤ |raw_coeff|`. To do that it
needs `raw_multiset(terms)` — the residual's key→coefficient map.

**That map is an invariant of the search.** `terms` is the whole manifold and never changes across
hypotheses. It was nevertheless recomputed inside every call: measured **7,461** calls on `ccsdt`
triples, at 0.036 s per rebuild, ≈ **270 s of a ~300 s manifold** spent recomputing a fixed value.

So the cost was the product of two quantities that both grow with rank — hypotheses and terms —
where only the first is intrinsic.

## What the numbers looked like

Per manifold, `ccsdt`, diagram engine, canonical Fock:

| manifold | terms | before | after | `raw_multiset` calls |
|---|---|---|---|---|
| singles | 12 | 0.1 s | 0.1 s | 139 → 19 |
| doubles | 73 | 4.3 s | 2.0 s | 2452 → 21 |
| triples | 399 | 94.7 s | 6.9 s | ~7461 → 19 |

~5.5× the terms cost ~22× the time before — the quadratic signature. Afterwards the call count is
**flat** (19/21/19, independent of term count), and that flatness is the actual property; the timing
follows from it.

End to end: rank-3 dressing 293.7 s → **9.1 s**; rank 4 went from **abandoned after >25 minutes** to
**61.6 s**. That last number is why rank-4 dressing exists as an option at all.

**Generation is not part of this story.** The diagram engine produces all four `ccsdtq` manifolds
(3172 terms) in 3.5 s. Every second of the old cost was recognition.

## The fix, and why it is a parameter rather than a cache

`raw_multiset` is computed once in `find_operator_occurrences` and threaded into
`hypothesis_is_consistent` as an optional `raw=` argument (`None` preserves standalone behaviour for
one-off callers).

Memoizing `raw_multiset` measures the same 38× and is one line — it is how the ceiling was
established. It is not what shipped, because the redundancy is a **structural fact about the
search**: the residual is fixed for the search's duration, and a parameter says so. A cache instead
hides that behind a dict keyed on large term tuples whose lifetime spans every manifold and method in
the process.

`raw_multiset` is pure, so hoisting cannot change results — which is exactly why the gate asserts it
rather than assuming it. Output is byte-identical.

## What the regression gate pins, and why it is a call count

`test_dressing_scaling` asserts that `raw_multiset` is called a **bounded** number of times, not once
per hypothesis, and that the count does not scale with manifold size.

A wall-clock threshold would be flaky on shared machines and diagnostically useless. The call count is
deterministic and names this defect directly if it returns.

## Two explanations that look right and are not

Both were tested. Recorded because both are plausible enough to be proposed again.

**`_eri_canonical` is not the hotspot.** It shows 3,027,528 calls and **864 s cumulative** in the
triples profile — the largest number in the listing. That is its callees' time. Memoizing it achieves
a 97.9 % hit rate (192,490 hits / 4,093 misses) and buys **6 %** (15.2 → 14.3 s). *Cumulative profile
time is not a fix target.*

**There is no second hot function.** The self-time ranking is diffuse: `relabel_term_dummies` 5.9 s,
`reindex_tensors` 3.0 s, `canonicalize_tensor` 2.9 s, then a long tail of 45 M-call primitives
(`Index.__hash__`, `dict.get`, `_antisym_slot_key`) — 228 M calls for 45 s with nothing individually
slow. That diffuseness is the *tell*: when no inner function is slow, the cost is structural, and the
thing to look for is a redundant outer loop.

## What is still true about scaling

One quadratic factor was removed, not proven absent. The hypothesis count itself (7,461 at rank-3
triples) still grows with rank, and pruning it would be a correctness-affecting change — it can alter
*which* operators are recognized, unlike this hoist. Rank 4 at 61.6 s means that question is not
pressing; a rank-5 attempt would be where it resurfaces.

---

Status of this work lives in `vault/Status/Completion.md`. Implementation:
`python/ccgen/optimization/dressing.py` (`hypothesis_is_consistent`, `find_operator_occurrences`),
gate: `python/ccgen/tests/test_dressing_scaling.py`.
