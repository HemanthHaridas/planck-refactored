# Scope — the residual ~2% CCSD raw-generation weight error

Companion to `CCGEN_NAME_OVERLOAD_BUG_HANDOFF.md`. That handoff fixed the
canonicalization bugs (false-zero T1.2b, non-idempotence T1.2c) and validated
the reference against PySCF. This doc scopes the **one remaining defect**: a
small (~2%) error in ccgen's CCSD doubles residual that survives every
canonicalization fix and is therefore in the **raw term generation** (BCH +
Wick projection), not canonicalization.

## Current state (post T1.2b/c)

- **CCD doubles: exact** — matches the PySCF-validated reference to 0.0.
- **CCSD doubles: maxdiff 1.53, norm ratio 1.02** vs the reference (canonical
  Fock, random amplitudes). Antisymmetric. The error is confined to
  **t1-containing** doubles terms (`t1·t2·v`, `t1·t1·t2·v`).
- Energy manifolds: correct and unchanged throughout.
- Reference (`gccsd_reference.py`): PySCF-exact (`test_reference_vs_pyscf.py`).

## What is ruled out (do not re-investigate)

Each was tested against the PySCF-validated gate and left the 1.53 unchanged:

- **Merge key / bucketing** — sums coefficient-faithfully per bucket (proven by
  a by-object faithful-sum == merged comparison, maxdiff 0.0).
- **Canonicalization** — both the `(space,name)` false-zero (T1.2b) and the
  non-idempotence split (T1.2c) are fixed; neither moved the 1.53.
- **The reference** — PySCF-exact on H2 (0.0) and LiH (2e-5).
- **CCD** — entirely correct, so the bug is specifically about **t1**.

## The evidence pointing at raw generation

Grouping the final `t1·t2·v` terms by structural signature (which factor holds
which externals) shows the residual error as **P-partner weight anomalies** —
structures related by P(ij) or P(ab) that should carry equal-magnitude
coefficients but do not. After T1.2c the two-t1 signatures are mostly
symmetric, but some vanish where they should not, e.g.:

```
csum= 0   (t1(i), t1(j), t2(a,b), v)     <- should be nonzero
csum=-1   (t1(a), t1(i), t2(b,j), v)     }  P-partners, OK
csum=+1   (t1(a), t1(j), t2(b,i), v)     }
```

A structure summing to 0 that the reference needs nonzero (or vice versa) is a
**generation-time** weight, set during BCH commutator expansion + Wick
contraction, before canonicalization runs. That is the locus.

## Why isolating it is subtle (a warning for the next attempt)

The naive isolation — "compute what the 1-t1 group *should* be as
`Rref − Rgood`" — **does not cleanly separate 1-t1 from 2-t1**: each group's
target is defined by subtracting the other, so both show the same 6.74 maxdiff
and neither is pinned. The groups are coupled through the shared reference.

The right isolation is **structure-by-structure against the reference expanded
into the same primitive basis**: expand the reference's tau-intermediate form
(`build_tau`, `Wmnij`, `Wabef`, `Wmbej`, the explicit `ovvv`/`ooov` singles
pieces) into individual `t1·t2·v` primitive contractions, and diff ccgen's
per-structure coefficient against the reference's per-structure coefficient.
`optimization/dressed_equation.py` already has machinery to expand the dressed
operators into primitives (`expand_dressed_term`) — reuse it to get the
reference's primitive `t1·t2·v` weights, then diff.

## Work breakdown

### W0 — primitive-level reference weights (prerequisite, ~M) — DONE
`python/ccgen/tests/w0_primitive_weights.py`. `PRIMITIVE_T1_TERMS` is the
reference r2's entire t1-part flattened into ~60 primitive einsum contractions,
each `(coeff, subscripts, operand_keys)`. `verify()` sums them and matches the
reference's t1-part (isolated by differencing `gccsd_doubles_residual` at t1 vs
t1=0) to 1e-14. Two transcription slips the numeric check caught: wrong external
on the tau_tilde-in-Fae/Fmi pieces, and a **missing `t1·t1·t2·v` cross-term** in
both W-ladders (tau's t1t1 part × the intermediate's `t2·⟨oovv⟩` part) — the
same structure class that turns out to be the ccgen bug. Gated by
`test_w0_table_reproduces_reference_t1_part`.

### W1 — diff ccgen's per-structure weights against W0 (~S) — DONE
`python/ccgen/tests/w1_weight_diff.py`. Converts each W0 einsum entry to an
`AlgebraTerm` (the only new code) and reduces **both** sides through
`dressing._eri_canonical` for a shared structure key. Two-layer verdict:
- **Grouping cross-check:** 48 == 48 structures, **0 missing / 0 spurious** —
  reference and ccgen structures map onto the identical set, so any difference
  is a real weight error (the doc's stop-condition: a clean small set, not a
  spread). Gated by `test_w1_grouping_is_sound`.
- **Numeric per-structure verdict** (`numeric_diff`, authoritative — the
  `_eri_canonical` *coefficient* sign is unreliable across the reference-einsum
  vs raw-ccgen conventions, flagging the CORRECT `t1·v` singles with a spurious
  ±2; the numeric magnitude is immune): **exactly 6 mis-weighted structures,
  all class `t1·t1·t2·v`, each exactly 2× the reference.** Sharper than the doc
  guessed — `t1·t2·v` is numerically **correct**; the bug is confined to the
  `t1·t1·t2·v` ladder cross-terms. Gated by
  `test_w1_numeric_diff_is_the_known_t1t2_bug` (classes == {t1*t1*t2*v}, count
  == 6); flip to `assert not numeric_diff()` when W2 lands.

### W2 — trace the wrong weights to their BCH/Wick origin (~L, the core) — LOCALIZED
`python/ccgen/tests/w2_bch_wick_origin.py` pins the full trace (5 gated
findings):
1. The over-count is **exactly 2×** on every mis-weighted structure — a clean
   symmetry-factor error, not a sign flip or missing/extra diagram.
2. The 2× is already in the **raw projected terms**, before canonicalize/merge
   (`test_2x_is_in_raw_projection_not_downstream`) — so it is BCH or Wick, not a
   downstream pass.
3. It originates **entirely at BCH level n=3** (`[[[V,T],T],T]`, the T1·T1·T2·V
   product) — `test_2x_originates_at_bch_level_3`.
4. It is **NOT the BCH commutator form**: the direct multinomial
   `exp(-T) H exp(T)` gives the same 2×, and even a single explicit `H·T1·T1·T2`
   product carrying the correct `3/3!` prefactor is still 2×
   (`test_2x_survives_explicit_single_ordering_product`). So the over-count is
   the **Wick contraction count of two identical T1 operators** — the
   automorphism swapping the two identical T1 blocks (block ids 2,3 in
   `_assign_block_ids`) is counted as a distinct pairing.
5. It is **NOT arithmetically detectable**: the correct `t1·t1·v` class and the
   buggy `t1·t1·t2·v` class share the same coefficient magnitudes (±1, ±1/2),
   and `t1·t1·t1·v` / `t1·t1·t1·t1·v` (also repeated-T1) are numerically
   **correct** — so a blanket "divide repeated-T1 terms by 2!" or a
   coefficient-keyed fix corrupts the correct classes
   (`test_coefficient_magnitude_is_not_a_reliable_bug_signal`).

**Root cause traced to the BCH connected-form extraction** (pinned by
`test_missing_factor_is_the_identical_T1_automorphism_1_over_2`): the missing
factor is exactly `1/2!`, the automorphism of the two identical T1 operators.
Shown pre-projection on the *same* `t1·t1·t2·v` operator content — the direct
multinomial `exp(-T) H exp(T)` connected form is a clean set of `+1/96`
OpTerms, but the nested-commutator **BCH level-3 term `/3!` carries mixed
`±1/96, ±1/48`**: the `AB−BA` expansion leaves reordering pieces that should
cancel down to the `1/2!`-reduced connected form but survive, doubling the
projected weight.

**Ruled out — the naive automorphism-division fix** (pinned by
`test_naive_graph_automorphism_correction_is_INSUFFICIENT`): dividing each term
by the automorphism group order of identical amplitude factors that share a
connection signature makes the residual **worse** (maxdiff ~4.0 vs reference),
because it assigns factor 2 to the **already-correct** `t1·t1·v` (and 6, 24 to
`t1·t1·t1·v`, `t1⁴·v`). The pipeline **already bakes the correct symmetry
factor into every repeated-operator class except `t1·t1·t2·v`** — so the fix is
NOT a blanket automorphism division. It must find why that *one* class is
under-corrected while the other repeated-T1 classes are handled right.

### W2b — the remaining fix, scoped

**Goal.** Make `generate_cc_equations` weight the 6 `t1·t1·t2·v` structures at
`1/2!` of their current value — and *only* those — so W1's `numeric_diff()` goes
empty, without disturbing the 5 already-correct t1 classes or the non-t1
residual, and with CCSDT re-checked.

**Two candidate fix directions, and what's already known about each.**

*Direction A — a correct per-term symmetry factor.* The two ruled-out attempts
(coefficient-keyed, connection-signature-automorphism) failed because they
divide classes that are already right. But those are the *naive* forms. The
open question A owes: **is there a per-term predicate, computed from the
contraction graph, that is true for the 6 `t1·t1·t2·v` structures and false for
every already-correct term?**

**A IS DEAD — the probe was run and the best candidate is a FALSE
discriminator (2026-07-23).** The most promising invariant — "an identical
cluster amplitude is *vertex-absorbed* (both its indices summed into one `v`)
while its identical partner is not" — was implemented
(`C(n, n_absorbed)` division) and it *did* drive W1's doubles `numeric_diff()`
to 0 and the `test_gccsd_gate` KNOWN_BUG maxdiff to 0. **But it silently
corrupted the SINGLES.** A new PySCF gate (`test_ccgen_singles_matches_pyscf`
in `test_reference_vs_pyscf.py`, built during this probe) shows ccgen's CCSD
singles are **already exact** (maxdiff 0 vs PySCF `gccsd.update_amps`), and the
candidate fix pushed them to 4e-8 wrong. Root reason the predicate is false:
singles `t1·t1·v` and `t1·t1·t1·v` exhibit the **identical** vertex-absorption
asymmetry (`[False, True]`) as the buggy doubles `t1·t1·t2·v`, yet are correct
— so "vertex-absorption asymmetry" does not separate buggy from correct. The
full extended suite passed *with the broken fix in place* because it had **no
singles residual gate** — only the new PySCF singles gate caught it. The fix,
the `canonicalize.py` pass, and the W1/gate test flips were all reverted; the
bug is documented-not-fixed again.

**Consequence:** no per-term contraction-graph invariant tested (coefficient
magnitude, identical-operator count, connection signature, vertex-absorption
asymmetry) separates the buggy structures from the correct ones. The mis-count
is **not term-local** — it depends on context the projected term does not
carry. This commits the work to **Direction B**.
- *Guardrail now in place:* the PySCF singles gate is the missing oracle — any
  future fix MUST keep it at ~0 (and a doubles-manifold analogue already
  exists). CCSD singles being exact also means the bug is doubles-manifold-only
  at CCSD, another constraint B's fix must respect.

### Direction B — the remaining fix, scoped (2026-07-23)

**Precise locus (measured).** The bug is a **2× over-count inside Wick's
contraction enumeration** for the `H·T1·T1·T2` connected term. Verified:
- The product `H·T1·T1·T2` yields **one** combined OpTerm (not duplicate
  OpTerms), and `wick_contract` enumerates **288 connected contractions** from
  it, merging to 13 canonical `t1·t1·t2·v` terms. So the 2× is in the
  enumeration→merge, not in operator-product construction.
- The two `T1` occupy **structurally identical blocks** (ids 2, 3 in
  `_assign_block_ids`); Wick treats them as distinguishable.
- The enumeration routes through the **`_wickaccel` C extension**
  (`_iter_wick_pairings` → `_wick_pairings_cached` / `analyze_signature`), so a
  fix there must keep serial == accel.

**Two locality fixes RULED OUT by direct experiment (do not re-attempt):**
1. *Per-term symmetry factor* (Direction A) — false discriminator, breaks
   singles. See above.
2. *Per-contraction automorphism factor* — dividing each Wick contraction by
   the order of the identical-block permutation group that fixes **its own**
   pairing edge set was implemented and measured: it makes doubles **worse**
   (28 bad structures, corrupts the already-correct `t1·t1·v` / `t1·t1·t1·v`).
   So the over-count is **not** a property of any single contraction — it is in
   the *global relationship* between the enumerated contraction set and the true
   diagram weight. This is the finding that makes B research-grade: no local
   (per-term, per-pairing, per-contraction) correction exists.

Also do **not** swap BCH for the multinomial `exp(-T)H exp(T)`: the full
multinomial connected projection measures maxdiff **4.28** vs the reference
(worse than BCH's 1.53).

**The two viable B routes (a real fork, pick deliberately):**

- **B1 — canonical-by-construction via the diagram front-end.** The landed
  diagram enumerator (`ccgen.diagram.enumerate_diagrams`, D2.3) returns the
  **correct 30 diagrams** for CCSD doubles. **Progress (2026-07-23):**
  - *Diagram ASSEMBLY (D3.2) is DONE and validated against PySCF.* The
    "mixed diagrams unbuilt" wall was a small-dims measurement artifact; the
    assembler runs on every diagram shape. Validated by a **full-rank solve of
    the assembled diagram basis against the PySCF doubles residual**
    (LiH/STO-3G: 31 diagrams, rank 31, span-residual ~5e-15) — the assembly
    reproduces PySCF exactly, and the per-diagram weights come out clean
    **±1/2^k**, seed-independent. **The two buggy diagrams get PySCF weight
    exactly 1/2** — the correct half the term path over-counts to 1.0.
    (`test_diagram_basis_spans_the_pyscf_doubles_residual` et al.)
  - *The table reproduces the residual TENSOR but is NOT a drop-in term weight*
    — MEASURED (2026-07-23): the committed table (`ccsd_diagram_weights.json`) is
    molecule-independent (table + assembled reps reproduce the PySCF residual to
    1e-16 on an unseen seed), BUT **D4-via-table does not cleanly work** —
    turning `weight × orbit(rep)` into emittable `AlgebraTerm`s hits the D3.2b
    ragged-split wall (orbit-expansion cancels/overcounts; the `t1·t1·t2·v` bug
    is SUB-diagram so per-diagram rescale can't reach it). See the diagram-scope
    doc's "D4-via-table — ATTEMPTED" note. No D4 was wired (arbitrary order
    untouched).
  - *The weight FORMULA* (Crawford diagrammatic rule: sign `(-1)^(h+l)` DONE +
    validated via AR2.0; magnitude `(1/2)^(equiv pairs)·∏1/n_v!` and the
    sign-convention reconciliation still open) is only needed for **true
    arbitrary rank** (no per-method table oracle), not the fix. See the
    diagram-scope doc's "AR2 best path — MEASURED DECISION".
  **B1 == D4-via-table (ships the fix, ~S) ; the formula (AR2.2/2.3) is the
  separable arbitrary-rank follow-up.** Also delivers the 78× scaling win. See
  `CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md`.
- **B2 — a per-diagram weight correction on the term path. PROTOTYPED
  (2026-07-23) — the over-count IS a clean per-diagram factor, but it is NOT
  derivable from the diagram; only measurable against the reference. So B2 as a
  standalone fix is not viable, but it produced the sharpest characterization
  yet (below).**

  *What the prototype found (all measured):*
  - Grouping the generated doubles by `term_diagram_id` and diffing each
    diagram's numeric contribution against the W0 reference: the over-count is
    an **exactly-uniform per-diagram factor of 2.0 on precisely 2 of the 19
    t1-class diagram-ids** — `(((1,1,0),(1,2,1),(2,1,1)),2)` and
    `(((1,1,1),(1,2,1),(2,1,0)),2)`; every other diagram is 1.0. So *within a
    fixed manifold* the correction is clean and per-diagram (unlike per-term /
    per-contraction, which were ragged and wrong).
  - **But the factor is not a function of the diagram.** Every structural
    invariant of the diagram was tried and each FAILS to separate the 2 buggy
    doubles diagrams from correct ones:
    - The buggy diagrams both carry a `(1,2,1)` operator (a T1 with both lines
      internal to the vertex — "vertex-absorbed") next to a distinct-triplet T1.
      **But the same `(1,2,1)`+partner-T1 pattern appears in 3 SINGLES
      diagrams, which are exact** (PySCF, maxdiff 0). So the triplet pattern is
      manifold-dependent, not a diagram property.
    - The **line-graph automorphism** (`build_line_graph` + permuting
      same-triplet operator nodes) is the *inverse* of what's needed: it is 2
      for the CORRECT two-identical-T1 diagrams (`(1,1,0),(1,1,0)` …) and **1
      for both buggy diagrams** (their operators have distinct triplets, no
      automorphism). Singles has no automorphisms at all.
  - *Conclusion:* the factor-2 is real and per-diagram-within-doubles, but it
    depends on the **projector rank (manifold)**, not on any invariant the
    diagram or its line graph carries. It is measurable against W0 but not
    derivable — so B2 cannot rescale from `term_diagram_id` alone without
    smuggling in the reference it is meant to reproduce.

  *Prototype verdict:* B2-standalone is **dead**. The surviving, load-bearing
  fact for B1: within a manifold the true weight IS a clean per-diagram number,
  which is exactly the D3.3 per-diagram-weight target — so the diagram front-end
  (B1) remains the principled route, and it must derive the manifold-dependent
  multiplicity that B2 proved is not a diagram invariant. This sharpens D3.3:
  the missing divisor is a function of (diagram, bra_level), not of the diagram
  alone.

- *Risk:* B1 is high/research-grade and XL; it touches the diagram front-end
  (D3.2's open assembly) and must supply the (diagram, bra_level) multiplicity.

**Recommended order:** A (per-term), the per-contraction automorphism, and B2
(per-diagram rescale) are all prototyped and dead. **The only surviving route is
B1 — finish D3.2** and supply the (diagram, bra_level) multiplicity the B2
prototype isolated. There is no shortcut on the term path left to try.

**Gates in place for Direction B (the singles gate is new, built by A's probe):**
- W0 `verify()` — reference ground truth (must stay 1e-14).
- W1 `numeric_diff()` — must go from 6 structures to **0**; and
  `test_w1_grouping_is_sound` (48==48) must still hold.
- **`test_ccgen_singles_matches_pyscf`** (new, PySCF-exact) — the singles must
  STAY at ~0. This is the gate A's broken fix would have failed; any B fix that
  touches singles-shaped terms is checked here. **The load-bearing lesson from
  A: a doubles-only residual gate is insufficient — validate every manifold the
  change can touch.**
- W2 `test_naive_*` / `test_coefficient_*` — guardrails against re-attempting
  ruled-out arithmetic/graph-local fixes.
- `test_gccsd_gate.py` `@expectedFailure` ×2 → flip to hard assertions (W3).
- Full extended ccgen suite green, **serial == accel**, and CCSDT re-checked
  (doubles *and* singles — CCSD singles are exact, so B must not regress them;
  CCSDT `t1·t1·t3·v` / higher repeated-operator classes are the generality
  test).

**Effort:** B1 ~XL (finish D3.2, also buys the scaling win) — the only route
left. A, the per-contraction automorphism, and B2 are all eliminated.
**The bug is confined to correctness of the generated CCSD/CCSDT equations; the
generated kernels are not compiled into any binary, so nothing downstream is
wrong today** — this is a prerequisite for trusting the generated path, not a
live miscompute.

### W3 — validate + finalize (~S)
Whole-residual gate → 0 (flip `test_ccgen_matches_reference_KNOWN_BUG` and
`test_t1t2v_terms_hit_their_target_T1_GATE` to hard assertions); CCSDT doubles
antisymmetric and matching (same bug, more instances); then **update the count
pins** (`test_optimizations`, `test_tau`, `test_diagram` — currently
`@expectedFailure`) to the final correct values, since generation is now
trustworthy.

## Risk

- **W2b is the research-grade step** — the BCH/Wick layer, a harder subsystem
  than the canonicalization already fixed. W2 resolved the *diagnosis* (it is a
  symmetry-factor / connected-form under-count, not name-overload); W2b is the
  correction, and Direction B in particular is L–XL and touches accelerated
  code.
- **Stop condition (now RESOLVED in favor of a patch):** W1's diff came out a
  **clean 6-structure set, all one class, all exactly 2×** — the best case the
  stop-condition hoped for, not a spread-across-many with no pattern. So a
  targeted fix (W2b) is the right path, NOT abandoning ccgen for the diagram
  front-end. The diagram path remains the *scaling* fix, independent of this.

## Cross-note — W2b and the diagram front-end share one kernel

The `1/n_k!` connected-form / identical-operator-automorphism reduction W2b-B
must implement is the **same combinatorial object** the diagram front-end
(`CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md`, D3.2c) is blocked on — the diagram
representation gets "each topology once, correctly weighted" by construction,
which is exactly the property the term path lacks here. Consequences:
- Solving W2b-B in the term path produces the automorphism-counting logic D3.2c
  needs; solving D3.2c produces a generator that doesn't have this bug at all.
- But they do **not** substitute for each other for free: W2b fixes correctness
  in the *shipping* term path today; D3.2 additionally owes its edge-matching
  assembly and a rebuilt residual gate (its old gate measured a fabricated
  antisymmetric target — see that doc's RECHECK). Fixing W2b advances D3.2's
  combinatorial half and cleans its target, but does not finish it.
- **Recommendation:** do W2b in the term path (smaller, tightly gated by
  W0/W1/W2, fixes correctness now). Treat whatever automorphism logic it yields
  as the seed for D3.2c if/when the scaling win is pursued.

## Assets carried forward

- **The gate** (`test_gccsd_gate.py` + `gccsd_reference.py`, PySCF-validated) —
  the authoritative pass/fail; W2 is graded on it.
- **CCD as a passing anchor** — any W2 change must keep CCD at 0.0.
- **`test_reference_vs_pyscf.py`** — pins the reference's correctness.
- The full failed-approach history in
  `CCGEN_DIAGRAM_REPRESENTATION_SCOPE.md` and the mechanism writeup in
  `CCGEN_NAME_OVERLOAD_BUG_HANDOFF.md`.
