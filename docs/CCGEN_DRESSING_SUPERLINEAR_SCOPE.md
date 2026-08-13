# Why dressed-operator recognition scales super-linearly — investigation

Scopes a research investigation into the super-linear cost of `_dress_operator_equations`,
the blocker that forced V1.3's anchor down from rank 4 to rank 3.

**The investigation is largely complete before scoping, and the root cause is confirmed by a
38× measured speedup.** What remains is landing it safely, plus the second-order questions the
fix exposes. Everything below is measured on the current tree.

---

## The measured scaling

`assemble_dressed_equation`, one manifold at a time, `ccsdt` (diagram engine, canonical Fock):

| manifold | terms | time |
|---|---|---|
| singles | 12 | 0.1 s |
| doubles | 73 | 4.3 s |
| triples | 399 | 94.7 s |

~5.5× the terms costs ~22× the time — consistent with quadratic. Extended to `ccsdtq`:
doubles 74 → 16.5 s, triples 412 → 307.6 s, quadruples 2672 → **abandoned after >25 min**.

Generation is not implicated: the diagram engine produces all four `ccsdtq` manifolds
(3172 terms) in **3.5 s**.

---

## Root cause: an invariant rebuilt per hypothesis

`hypothesis_is_consistent` (`dressing.py:720`) opens with:

```python
raw = raw_multiset(residual_terms)
```

`residual_terms` is the **whole manifold**, and it does not change across hypothesis checks.
The `cProfile` run on `ccsdt` triples recorded **7,461** calls to `hypothesis_is_consistent`,
so `raw_multiset` is rebuilt 7,461 times over the same input.

Cost per rebuild, measured: 0.006 s (doubles, 73 terms), 0.036 s (triples, 399 terms). So
`7461 × 0.036 ≈ 270 s` of the ~300 s triples time is spent recomputing a value that never
changes. **The product `n_hypotheses × n_terms` is the quadratic term**, and both factors grow
with rank.

### Confirmed by fixing it

Memoizing `raw_multiset` on its term list:

| | triples |
|---|---|
| baseline | 278.7 s |
| memoized | **7.4 s** |

**38× on the manifold that dominates the run.** This is the root cause, not a contributing
factor.

---

## What the investigation ruled out, and one wrong turn worth recording

Two hypotheses were tested and are **not** the cause. Recorded because both are plausible
enough to be re-proposed:

**`_eri_canonical` is not the hotspot, despite appearances.** It shows 3,027,528 calls and
864 s *cumulative* in the triples profile — the largest number in the listing. I read that as
self-cost and predicted a large win from memoizing it. Measured: 97.9 % value-level hit rate
(192,490 hits / 4,093 misses) yet only **15.2 s → 14.3 s (6 %)**. The 864 s is its callees'
time; the function itself is cheap and already effectively deduplicated by the caller's access
pattern. **Cumulative time in a profile is not a fix target** — this cost an hour of chasing
the wrong function.

**There is no second hotspot to find.** The self-time (`tottime`) ranking on doubles is
diffuse: `relabel_term_dummies` 5.9 s, `reindex_tensors` 3.0 s, `canonicalize_tensor` 2.9 s,
then a long tail of 45 M-call primitives (`Index.__hash__`, `dict.get`, `_antisym_slot_key`).
228 M calls for 45 s — nothing individually slow. That diffuseness is *why* the real cause was
structural (a redundant outer loop) rather than a slow inner function.

---

## Steps

### D0 — pin the scaling and the redundancy as tests (~S, do first)

Two assertions, before any change:

- `raw_multiset` is called **once per manifold**, not once per hypothesis. Assert by counting
  calls during one `assemble_dressed_equation` — the count must not scale with the hypothesis
  count.
- A timing-independent proxy for the scaling: `canonicalize_term` (or `raw_multiset`) call
  count grows ~linearly, not quadratically, in term count across singles/doubles/triples.

**Why a call-count assertion rather than a wall-clock one:** a timing gate is flaky on shared
CI and gives no diagnosis when it fails. A call count is deterministic and names the defect
directly if it regresses.

*Gate:* both fail on the current tree for the stated reason.

### D1 — hoist the invariant (~S, the fix)

`raw_multiset(residual_terms)` is loop-invariant across the hypothesis search. Two shapes:

- **(a) Thread it through.** Compute once in `find_operator_occurrences` /
  `assemble_dressed_equation` and pass it into `hypothesis_is_consistent` as a parameter.
  Explicit, no cache lifetime, no memory growth.
- **(b) Memoize `raw_multiset`.** One decorator, zero call-site changes — this is what the
  38× measurement used.

**Recommendation: (a).** The redundancy is a *structural* fact about the search — the residual
is fixed for the whole search — and threading it says so. Memoization hides that behind a cache
whose key is a term list (large tuples as dict keys), and its lifetime spans manifolds and
methods, so it silently retains every manifold's multiset for the process's life. (b) is the
right emergency fix and the right way to *measure* the ceiling; (a) is the right thing to ship.

*Gate:* D0's call-count assertion passes; triples time drops from ~280 s to <15 s; **the
dressed equation is byte-identical** — `raw_multiset` is pure, so hoisting cannot change results,
and that must be asserted rather than assumed.

### D2 — re-measure the scaling law after the fix (~S)

The fix removes one quadratic factor; it does not prove the result is linear. Re-run the
singles/doubles/triples ladder and fit. Then attempt `ccsdtq` quadruples (2672 terms) — the case
that was abandoned — with a hard timeout.

*Gate:* a recorded table, and an explicit statement of whether rank 4 is now viable as a
build-time step. This is the question V1.3 actually needs answered.

**Do not assume 38× at rank 3 implies rank 4 is solved.** If the remaining scaling is still
super-linear, 2672 terms may still be minutes-to-hours. Measure, don't extrapolate.

### D3 — decide whether the hypothesis count itself needs work (~S to scope, ~M+ if needed)

`n_hypotheses × n_terms` had two factors. D1 removes the per-hypothesis rebuild of the
`n_terms` factor. If D2 shows the remainder is still super-linear, the next question is the
7,461 hypotheses themselves: how does that count grow with rank, and is the search enumerating
candidates it could prune cheaply?

*Gate:* a measured `hypotheses vs rank` table. **Scope-only** — do not start pruning the search
until D2 says it is necessary. Pruning risks changing *which* operators are recognized, which is
a correctness change, unlike D1.

### D4 — fold the result back into V1.3 (~S)

If D2 makes rank 4 viable, revisit V1.3's anchor decision (currently rank 3, chosen *because*
of this cost) and the "recognition-performance fix is out of scope" note in
`CCGEN_V13_LINK_AND_RUN_SCOPE.md`.

*Gate:* that document reflects the new measurement rather than the old constraint.

---

## Sequencing

```
D0 (pin redundancy + scaling proxy)   ~S   ← before any change
   └→ D1 (hoist the invariant)        ~S   ← the fix; byte-identical required
        └→ D2 (re-measure; try rank 4) ~S  ← answers V1.3's real question
             ├→ D3 (hypothesis count) scope-only, ~M+ if D2 demands it
             └→ D4 (update V1.3)      ~S
```

---

## What this reuses

| Reused | From |
|---|---|
| `raw_multiset` | `dressed_equation.py` — unchanged, just called once |
| Byte-identity gating discipline | V1.2.0's flag-matrix net |
| Call-count-over-wall-clock gating | new here, but the same "deterministic gate" principle as V1.1f |
| The rank ladder (singles/doubles/triples/quadruples) | this investigation's own measurements |

**Net new:** one hoisted parameter and the gates. No algorithm change, no new data structure.

---

## What NOT to do

- **Do not memoize `_eri_canonical` expecting a win.** Measured: 97.9 % hit rate, 6 % faster.
  Its large profile number is cumulative, not self.
- **Do not chase the diffuse `canonicalize_term` internals.** 228 M calls at 45 s with no single
  hotspot; the win is structural, not micro-optimization. `relabel_term_dummies` at 5.9 s self
  is the largest single item and is still only ~13 % of the manifold's time.
- **Do not ship the memoization (option b) as the fix** without weighing its process-lifetime
  cache. It is the right instrument for measuring the ceiling — that is how the 38× was
  established — and the wrong thing to leave in.
- **Do not gate on wall-clock time.** Flaky on shared machines and diagnostically useless;
  assert the call count.
- **Do not skip the byte-identity check on D1.** The hoist *should* be behaviour-preserving
  because `raw_multiset` is pure. "Should" is what an assertion is for — the same reasoning that
  made V1.1f worth landing.
- **Do not start D3 before D2.** Pruning the hypothesis search can change which operators are
  recognized; D1 cannot. Keep the safe fix separate from the risky one.

---

## Honest status

The hard part — finding the cause — is done, with a 38× confirmation. D1 is a small, safe,
byte-identical change. The genuinely open question is **D2: whether removing this one factor
makes rank 4 viable**, and that cannot be answered by extrapolation from rank 3. D3 exists
because it might not.

---

See `CCGEN_V13_LINK_AND_RUN_SCOPE.md` (the rank-3 anchor decision this cost forced, and the
"out of scope" note D4 would revisit) and `dressing.py:720` / `:861` / `:1097` (the three
functions involved).
