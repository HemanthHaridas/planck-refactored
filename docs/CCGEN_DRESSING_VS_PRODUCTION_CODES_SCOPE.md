# Why does the dressed-operator route work in CFOUR and MRCC but fail in ccgen?

**Research scope. Not started.** Opened because the premise of ccgen's retirement decision is in
tension with the field: CFOUR and MRCC both ship dressed intermediates as their **only** production
route, at high rank, for years. So "dressing and spin adaptation do not compose" cannot be true as a
general statement about coupled cluster — which means ccgen's retirement rests on something narrower
than it currently claims, and the doc should say what.

This is a **research** question, not a defect report. The retirement itself may well stand; what is
unclear is *why it had to*.

## What ccgen measured

Recorded in `docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md`, and re-verified while opening this scope:

| path | operator-free terms with changed coefficients | compensated? |
|---|---|---|
| GCC (`dress` only) | 2 | **yes** — valid rewrite |
| `adapt → dress` | 8 | **no** — ‖diff‖ = 8.06e+02 vs ‖R‖ = 983.79 |
| `dress → adapt` (production) | — | **no** — Be/STO-3G CCSDTQ 52 % short |

End-to-end: dressed `E_corr` = −0.0247182895 against an exact −0.0517746319.

And the structural fact underneath it, **re-measured on the current tree** (not taken from the doc):

```
GCC      v index-space patterns:  9   [oooo ooov oovv ovoo ovov ovvv vvoo vvov vvvv]
SPATIAL  v index-space patterns: 13   [... ovvo voov vovo oovo vooo vovv ...]
```

`ovvo`, `ovov`, `voov`, `vovo` are **one object** in GCC — antisymmetry `<pq||rs>` relates them —
and **four distinct objects** in the spatial basis, which has only the four `+1` symmetries of
`<pq|rs>`.

## The first hypothesis, and why it is WRONG

The obvious explanation — *ccgen's operators are hand-seeded in the spin-orbital basis and it has no
way to derive spatial ones* — was written into the first draft of this scope and is **refuted**.
`seeded_operators()` does return exactly six hardcoded Stanton–Gauss spin-orbital fingerprints, but
that is only the **recognition** route. ccgen also has a **derivation** route, and it is
basis-agnostic. Measured on the current tree:

```
GCC      ccsd doubles -> derived_operators() -> 20 operators
SPATIAL  ccsd doubles -> derived_operators() -> 20 operators
identical sets: True        (W_ft2v_oovv, W_t1t1t2v_oovv, W_t1t2v_ooov, ...)
```

`factorize.py` derives operators from the **contraction structure of the terms it is given**, and
does not care which basis they are in. So "ccgen cannot derive spatial operators" is false — it
derives the same 20 either way, and the capability was demonstrated for GCC.

**This kills the tidy version of the answer**, and it is recorded because it is the explanation a
reader will reach for first.

## The real question, restated

ccgen has **two separate dressing routes** that do not share machinery:

| route | operators from | where |
|---|---|---|
| **recognition** (retired, the 52 % failure) | 6 hand-seeded spin-orbital fingerprints | `dressing.py`, `dressed_equation.py` |
| **derivation** (works, incl. on spatial terms) | the terms' own contraction trees | `factorize.py` |

`grep` confirms the split: `dressing.py` has **zero** references to `derived_operators` or
`contraction_tree`; `factorize.py` imports `seeded_operators` only to *avoid re-deriving* what is
already named.

So the question is no longer "can ccgen derive spatial operators" (it can) but:

**Why was the RECOGNITION route the one wired to production, when the DERIVATION route is the one
that survives spin adaptation — and is what CFOUR/MRCC's factorization actually resembles?**

That reframes the retirement. What was retired is *recognition against hand-seeded spin-orbital
fingerprints*. The doc generalizes that into "dressing and spin adaptation do not compose", which
the derivation route contradicts on the same input.

## What to establish

### D0 — ANSWERED: the derivation route does NOT preserve value either, on GCC or spatial

Run with the existing gate's fixture and evaluator, `savings_fraction=1.0` (every derived operator
materialized):

| manifold | doubles terms | rewritten | **disagree** | ‖diff‖ / ‖R‖ |
|---|---|---|---|---|
| **GCC** | 66 | 39 | **23** | 3.73e-01 |
| **spatial** | 113 | 61 | **46** | 4.32e-01 |

**The GCC number is the important one.** The derivation route is claimed to work there, and it is
the control this probe needs: a value check that fails on the known-good case is measuring the
probe, not the route. It fails on GCC — so this is not a spin-adaptation phenomenon at all, and the
scope's framing ("derivation survives adaptation where recognition does not") is **refuted**.

**Both routes fail value preservation, for different reasons**, which collapses the tidy
"wire the other route" answer:

| route | failure |
|---|---|
| recognition | spin-orbital fingerprints matched against spatial terms — ‖diff‖ = 8.06e+02 |
| derivation | rewrite does not reproduce the source term — ‖diff‖ = 2.06e+02 **on GCC** |

#### What is established about the mechanism, and what is not

Traced by hand on `doubles[17]` / `doubles[18]`, an **i↔j exchange pair** (coefficients −1 and +1):

```
t17: t1(b,k) t1(c,i) v(j,c,k,a)
t18: t1(b,k) t1(c,j) v(i,c,k,a)
both rewrite to:  t1(b,k) · W_t1v_ooov(i,j,k,a)      <- identical expression
```

`_derived_name` (`factorize.py:373`) builds the name from **sorted factor names + output block
signature** and discards slot order. Verified consequence: each name stores exactly one definition
(0 names with multiple definitions), so a second contraction sharing a name is evaluated with the
first one's definition.

A second, related shape on `doubles[19]`: the spec's inner `v` has index-space pattern `vvoo` where
the source term's has `ovov`. The name encodes the **output** signature (`ooov`), not the inner
contraction's pattern, so specs and terms with different inner topology share a name.

**But neither shape explains all of it, and this is stated rather than smoothed over:**

| test | GCC | spatial |
|---|---|---|
| disagreeing terms | 23 | 46 |
| ...in a name-collision group (>1 source term → same expression) | 4 | 40 |
| ...whose inner `v` space-pattern differs from the spec's | 11 | 13 |

Neither predicate covers the disagreements, and they are not nested. **19 of 23 GCC disagreements
are not collisions**, and 12 of 23 have a matching `v` pattern. So "the operator name is
order-blind" is a real defect with two demonstrated instances, **not** a complete characterization.
Recording it as such: the next step is per-term diffing of the remaining cases, not another
plausible-sounding hypothesis — this investigation has already discarded three.

#### Why this was never caught

The derivation route has **no numeric validation at all**. Its gates check structure only:

- `tree_preserves_term` — each factor is one leaf, each summed index consumed once
- `test_budgeted_rewrite_is_exact` — the rewrite re-expands to the same factor **`Counter`**

A `Counter` of factor names is blind to index order by construction, so it cannot see any of the
above. This is the same defect shape the rest of this session kept finding: a structural gate
standing in for a value gate, and the value gate never written.

`docs/CCGEN_HIGHER_OPERATOR_REUSE.md` describes the factorizer as landed and gated — true
structurally, and that doc should say the numeric gate is absent.

#### The mechanism, narrowed by one decisive measurement

Term `doubles[44]` fits neither earlier predicate (no collision, no `v`-pattern mismatch), and
evaluating it **both ways** isolates the cause:

```
||orig - manifold-representative spec|| = 4.64            <- wrong
||orig - per-term identify_node spec  || = 1.58e-13       <- exact
```

Same operator name, same call site, two definitions:

| source | slots | definition |
|---|---|---|
| `manifold_operators` (one representative per name) | `[b, c]` | `t2(b,d,k,l) · v(c,d,k,l)` |
| `identify_node` on this term's own tree node | `[a, d]` | `t2(a,c,k,l) · v(c,d,k,l)` |

They are alpha-variants of the same contraction, but **not interchangeable at an arbitrary call
site**: binding the representative positionally to this term's indices does not reproduce it.

**So the defect is not "the name is order-blind" as such.** It is that a *single manifold-level
representative is reused for every call site of that name*, while the rewrite was produced from
each term's own tree node. The per-term spec is exact — the factorizer's tree really does
reproduce its term, which is what `tree_preserves_term` asserts and why the structural gate passes.
What is unproven is the step the structural gate never checks: that **one shared definition can
serve all call sites of a name**.

That reframes the earlier two shapes as *symptoms* of this, not independent mechanisms.

**Not yet established:** whether per-term specs are exact across the whole manifold. The obvious
next probe (evaluate every term against its own `identify_node` spec) needs the definition's
internal indices alpha-renamed per call — the same capture the manifold probe needed — and the
current probe raises `KeyError` on the nested case. That is a probe gap, not a finding, and it is
recorded here rather than guessed at.

#### Consequence for the retirement

The retirement decision is **unaffected** — if anything it is better supported, since the alternative
route it did not consider also fails. What needs correcting is the *reasoning*:
"dressing and spin adaptation do not compose" is still not demonstrated, because the second route
fails identically on GCC, where there is no adaptation to blame.

### D1 — what CFOUR and MRCC actually do (~M, literature)

Still worth answering, but its role has changed: it is no longer the blocking question, it is the
**calibration**.

- MRCC is string-based and automated at arbitrary rank. Does its factorization resemble ccgen's
  *derivation* route (operators falling out of contraction-order optimization) rather than a fixed
  operator list?
- CFOUR's CCSD intermediates are the classic Stanton–Gauss set — **in which basis are its
  production RHF kernels' intermediates defined**, and are they hand-derived per method?

*Verify:* a written answer per code with a citation, classifying each as
**fixed-operator-recognition** or **structure-derived**. If both are structure-derived, that is
independent evidence that ccgen wired the wrong one of its own two routes.

### D2 — why was recognition the production route? (~S, archaeology)

`grep` shows the two routes never shared machinery. `factorize.py` imports `seeded_operators` only
to avoid re-deriving already-named operators, and `dressing.py` has no reference to the derivation
machinery at all.

Read the history: was recognition wired first and derivation added later for a different purpose
(the `HIGHER_OPERATOR_REUSE` work), leaving the production path pointed at the older route by
inertia rather than by decision?

*Verify:* a one-paragraph answer with commit references. If it was inertia, that is the actual root
cause of the 52 %, and it is a wiring defect rather than a theory result — the same shape as the
rank-3 defect, where the kernel was correct and the harness around it was not.

### D3 — re-examine the cost case (~S, only if D0 is positive)

The retirement was justified on payoff as well as correctness: ~1.2–1.5× measured on spin-orbital,
~1.9–2.8× bounded spatial. Both numbers assumed the operators had to be *derived as research*. If
D0 shows the derivation route already produces working spatial operators, the cost side collapses to
"wire the other route".

The benefit side may also have been understated: `CCGEN_KERNEL_SCALING_SCOPE.md` has since measured
the generated-vs-hand gap as a **scaling** defect (21.8× → 50.1×, no plateau) attributed to
contraction order — which is exactly what a factorizer controls.

*Verify:* payoff re-estimated with `contraction_tree_cost` against the *scaling* baseline, and an
explicit statement of whether the retirement still holds.

## What NOT to do

- **Do not reopen the dressed route on the strength of this scope alone.** It is retired by a
  recorded decision with five failed fix attempts behind it. D0 is a literature question; changing
  the decision needs D1's numeric gate, not an argument.
- **Do not confuse "CFOUR does it" with "ccgen can".** CFOUR's intermediates are hand-derived and
  hand-coded for one method; ccgen generates at arbitrary rank. A spatial operator set that must be
  hand-derived per rank is not obviously a win for a *generator*, and that asymmetry is part of the
  answer.
- **Do not treat the 52 % number as evidence about CFOUR/MRCC.** It measures ccgen's specific
  composition, on ccgen's spin-orbital seeds.
- The four `test_dressed_*` gates and three `test_intermediate_layout_agreement` gates are
  `expectedFailure` as of this work — they encode the *current* defect. If D1 ever lands, they flip
  to unexpected-pass, which is the intended signal.

## Key code locations

| what | where |
|---|---|
| the six hand-seeded spin-orbital operators | `seeded_operators()`, `python/ccgen/optimization/dressing.py` |
| recognition + assembly | `dressing.py`, `dressed_equation.py` |
| the retirement answer this questions | `docs/CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` |
| the scaling defect that may change the cost case | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |
| xfail gates that would flip on success | `test_dressed_numeric_oracle.py`, `test_dressed_spatial_equivalence.py`, `test_intermediate_layout_agreement.py` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
