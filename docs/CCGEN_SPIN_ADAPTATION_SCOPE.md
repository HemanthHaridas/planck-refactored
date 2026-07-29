# ccgen Spin-Adaptation Layer (GCC → RCC / UCC)

Scope + status for mapping ccgen's spin-orbital (GCC) equations to spatial
restricted (RCC) and unrestricted (UCC) form. Companion:
`CCGEN_GENERATION_AND_VALIDATION.md` (how the GCC equations are generated and
validated).

This file answers one architecture question:

**How does a spin-adaptation layer turn ccgen's spin-orbital CC equations into
restricted and unrestricted spatial-orbital equations, and where does it sit?**

## Where it sits

ccgen derives everything in **spin-orbital (GCC)** form — `indices.Index` has a
space (occ/vir/gen) and no spin. Spin adaptation is a new stage that consumes and
produces `AlgebraTerm`s, so generation (both the wick and diagram engines),
canonicalization, lowering, and the emitters are untouched:

```
generate (GCC AlgebraTerms)  →  [spin adaptation]  →  spatial AlgebraTerms (RCC/UCC)  →  lowering → emit
```

It is engine-agnostic by construction: it operates on the `AlgebraTerm`s either
engine produces. The existing `lowering/restricted_closed_shell.py` does NOT do
this — it only re-lays-out spin-orbital terms into spatial blocks; it explicitly
does not spin-integrate.

## Why a separate layer, not a spin field on Index

`Index`'s identity `(name, space, is_dummy)` is baked into every canonicalize /
wick / diagram hash and equality. Adding a spin field there would perturb the
validated GCC path. So the spin layer is **isolated**: it wraps a spatial `Index`
in a lightweight `SpinIndex` (spatial base + spin ∈ {a,b}) and works on the terms
generation already produces. The GCC path is not modified.

## The physics

Each spin-orbital index `p` = (spatial `p̄`, spin `σ`). A GCC term is a sum over
spin-orbital indices; adaptation performs the spin summation. Within one term a
repeated index NAME is one physical line, so it carries one spin (a contracted
line preserves spin); summed indices are summed over both spins.

- **UCC** keeps α/β distinct: each GCC term expands into the spin blocks its
  externals allow (a doubles term → `aa`, `bb`, `ab`), with the spin-integrated
  coefficient. Amplitudes/integrals become spin-blocked. Mechanical.
- **RCC** = UCC + the closed-shell constraint α ≡ β: collapse the spin blocks to
  the minimal spatial set, combining coefficients (the `2J − K` pattern). The
  coefficient algebra here is the genuinely hard part.

## The validation advantage

Unlike the GCCSDT case (PySCF ships no `gccsdt`), the adapted targets **have**
per-residual oracles: `pyscf.cc.rccsd.update_amps`, `uccsd.update_amps`, and at
higher rank `rccsdt` / `uccsdt`. So adapted equations are validated directly
against PySCF residuals, not only by an FCI limit — a stronger gate than the
diagram work had.

## Steps

- **S0 — index model + single-term spin labeling. LANDED.** `ccgen/spin.py`:
  `SpinIndex` (spatial `Index` + spin) and `spin_label_cases(term,
  external_spins)` — label every index of one GCC term consistently along shared
  lines, enumerate the summed-index spin cases (`2^(#distinct summed names)`).
  Structural, not yet coefficient-integrated. Gated by `tests/test_spin.py` on
  the pp-ladder `t2·v` doubles term (case count, external-block spins,
  shared-line consistency, exhaustive summed enumeration).

- **S1.0/S1.1 — block model + one-factor resolution. LANDED.** `ccgen/spin.py`:
  the UCC block existence rule is **spin conservation per line** — for a rank-2n
  tensor (ccgen orders n virtual then n occupied slots) the lines pair slot k
  with slot k+n, and a block is nonzero iff `spin(k) == spin(k+n)` for all k. One
  rule, no per-tensor table (covers t1→a,b; t2→aa,bb,ab; f→a,b; v→aaaa/bbbb/
  abab/baba — the physicist `⟨pq||rs⟩` lines pair p-r, q-s). `block_exists` and
  `resolve_block(factor, label) -> (tag, exists)`. Gated by
  `tests/test_spin.py::BlockModelTests` incl. the S0+S1.1 integration on the
  pp-ladder: of the 4 summed-spin cases exactly 1 survives (the block-existence
  *filter*, the heart of S1.2).

- **S1.2 — single-term integration. LANDED.** `ucc_integrate_term(term,
  external_spins)` enumerates the summed-spin cases (S0), keeps only those where
  every factor is a valid block (S1.1), and emits one `SpinTerm` per survivor
  carrying the **raw GCC coefficient** (`SpinFactor`/`SpinTerm` in `spin.py`).
  Gated by the **spin-orbital identity** (no PySCF): on a spin-STRUCTURED
  spin-orbital tensor (forbidden blocks zeroed, as physical CC tensors are), the
  chosen external block of the GCC term equals the sum of its surviving
  integrated terms on the matching block slices — maxdiff 0 on the pp-ladder
  into both the `abab` and `aaaa` blocks. This validates the block filter *and*
  that the raw GCC coefficient is already correct (the block combinatorics
  needed no extra factor for this term). `tests/test_spin.py::UccIntegrateTermTests`.

- **S1.3a — full-manifold aggregation. LANDED.** `external_blocks(residual_tpl)`
  enumerates the canonical UCC external blocks (spin conservation on the
  residual's own lines, one rep per global a↔b flip → doubles `{aaaa, abab}`,
  singles `{aa}`), and `ucc_manifold(terms, residual_tpl)` integrates every GCC
  term into each block, returning `{block_tag: [SpinTerm]}`. *Gate (no PySCF):*
  structural (aggregation = per-term union) + the full-manifold spin-orbital
  identity on the `t2·v` subset — all six terms summed reproduce the sliced GCC
  `aaaa` and `abab` blocks to ~1e-16, exercising the **multi-term aggregation +
  multi-survivor summation** the single pp-ladder did not.
  `tests/test_spin.py::UccManifoldTests`.

- **S1.3b — full-manifold identity (general evaluator). LANDED.** The general
  `SpinTerm` evaluator handles every factor kind (t1, t2, f, v), so the
  full-manifold spin-orbital identity holds for the **complete CCD and CCSD
  residuals** — singles `aa` and doubles `aaaa`/`abab` reproduce the sliced GCC
  blocks to ~1e-14 (`tests/test_spin.py::UccFullManifoldTests`). This validates
  the UCC spin-integration MECHANISM end to end (raw GCC coefficients are already
  correct — no block-combinatoric factor needed). `external_blocks` is also
  confirmed to match PySCF's UCC block set (singles t1a; doubles t2aa+t2ab, with
  the b-blocks by a↔b flip).

  **Decision — no separate PySCF `uccsd.update_amps` numeric gate.** It is
  redundant for the mechanism: ccgen's GCC residual is already PySCF-`gccsd`-
  validated to 1e-16, and the spin-orbital identity proves `ucc_manifold ==
  GCC-sliced`, so `ucc_manifold == PySCF-uccsd` transitively. The one unique
  thing a direct comparison adds — the physicist(ccgen)→chemist(PySCF) ERI
  convention — is an EMIT concern and is settled at S3, not in the integration.

### S2 — RCC closed-shell reduction (α ≡ β)

**The physics (confirmed).** For a closed-shell RHF reference, α and β spatial
orbitals are identical, so `t1a = t1b = t1` and the UCC t2 blocks collapse: RCC
stores a **single** `t2 [o,o,v,v]` (the mixed `abab` block), with the same-spin
block recovered by antisymmetry `t2aa = t2ab − P(t2ab)`. RCC's single residual
equals the closed-shell GCC/UCC residual's `abab` block (confirmed: RHF ≡ GHF
closed-shell energy to 6e-14). PySCF `rccsd.update_amps` is the oracle. The
characteristic output is `2J − K` coefficient combinations.

**Why S2 is the hard step.** S1 (UCC) was mechanical — spin-label, filter by
block existence, keep the GCC coefficient (which came out right unchanged). S2 is
**not** mechanical: imposing `t2aa = t2ab − P(t2ab)` substitutes one block's
amplitude by a combination of another's, which *changes coefficients* — terms
merge and the `2J − K` structure appears. That coefficient collapse is the
genuine derivation content.

Sub-steps (de-risk numerically **before** any symbolic collapse):

- **S2.0 — closed-shell block relations, numeric. LANDED.** `t2aa_from_t2ab` in
  `ccgen/spin.py`: the pinned swap+sign `t2aa = t2ab − t2ab.transpose(virtual
  swap)` (in `[v,v,o,o]` layout), plus `t1a = t1b`. *Gate* (`S2ClosedShell-
  RelationTests` + PySCF `S2PyscfTests.test_s20_t2aa_from_uccsd_blocks`):
  reproduce PySCF UCCSD's own `t2aa` from its `t2ab` on water/STO-3G (a
  robustly closed-shell UHF≡RHF reference — equally-spaced H4 breaks spin
  symmetry and was rejected as a fixture), maxdiff ~1e-8. Settles the single
  most error-prone spot — the `P(t2ab)` swap convention — before any equation
  work. The occupied-swap and virtual-swap forms are numerically equal because
  `t2ab[a,b,i,j] == t2ab[b,a,j,i]`; the virtual swap is the one written.

- **S2.1 — RCC residual as a numeric block identity. LANDED.**
  `S2AbabSubstitutionTests`: the UCC `abab`-block residual (S1's `ucc_manifold`)
  evaluated with same-spin t2 slices reconstructed ONLY via `t2aa_from_t2ab`
  (i.e. the RCC model stores a single mixed block) reproduces the directly-sliced
  GCC `abab` residual on closed-shell tensors, for the full CCD and CCSD doubles
  manifolds (~1e-11). Closed-shell tensors are built by a spatial-seed lift
  (`_closed_shell_tensors`) so the relations hold by construction; a sign-mutation
  of `t2aa_from_t2ab` makes the gate fail (verified — not a tautology). **Proves
  the "abab + substitution" model is right before the symbolic collapse** — if it
  fails, the model is wrong and no symbolic work saves it. The independent numeric
  anchor is PySCF UCCSD (S2.0); a direct `rccsd.update_amps` end-to-end compare
  arrives with S2.3 once the collapsed equations exist to iterate.

- **S2.2 — symbolic collapse (~L, the core).** Produce the RCC spatial equation:
  take the UCC `abab` `SpinTerm`s (the S1 `ucc_manifold["abab"]`) and rewrite
  every factor into a single spatial representative per tensor, then merge. The
  `abab` residual carries factors across four spin blocks (`aaaa`, `bbbb`,
  `abab`, `baba`) — S2.2 funnels all of them into one spatial `t2` and one
  spatial `v`, which is where coefficients change and `2J − K` appears.

  The rewrite has three genuinely different pieces (amplitude, integral, merge),
  each error-prone in a different way, so they are gated **separately before**
  the end-to-end compare. Everything is a `SpinTerm → SpinTerm` rewrite on the
  existing structures (no new equation engine); each sub-step keeps the S2.1
  numeric identity (evaluate rewritten terms on closed-shell tensors, require
  bit-for-bit the same `abab` residual as before the rewrite) as an always-on
  invariant, and the final one adds `rccsd.update_amps`.

  - **S2.2a — canonicalize spin blocks to the global-flip rep. LANDED.**
    `canonicalize_spin_blocks` in `ccgen/spin.py`: for a closed-shell
    (α≡β) tensor a GLOBAL a↔b spin flip is an exact symmetry — the value at a
    slot-spin tuple equals the value at the fully-flipped tuple with the SAME
    spatial indices. So each factor collapses to its canonical block by flipping
    every slot's spin when that gives the smaller tag (`_canonical_block`): no
    index permutation, no coefficient change. Maps `baba→abab`, `bbbb→aaaa`,
    `bb→aa`, leaving only canonical blocks (`{aaaa, abab}` for doubles, `{aa}`
    for singles factors). *Gate* (`S22aCanonicalizeBlocksTests`, ccd+ccsd): the
    rewrite is a numeric NO-OP on the `abab` residual (S2.1 harness, maxdiff
    ~1e-16) **plus** a factor-level tag/spin consistency invariant — the numeric
    no-op alone can't catch a relabel that leaves the SpinIndex spins unflipped
    (`_eval_spinterm` reads spins, not the tag), but S2.2b/c read the tag, so a
    mismatch would silently corrupt the collapse; the consistency check catches
    it (mutation-verified). Isolates the most mechanical, most-likely-to-have-a-
    transpose-bug piece before coefficients start moving.

  - **S2.2b — amplitude collapse `t2aaaa → t2ab − P(t2ab)`. LANDED.**
    `collapse_amplitudes` + `_split_t2aaaa` in `ccgen/spin.py`: on a
    canonicalized (post-S2.2a) `SpinTerm`, replace each same-spin `t2[aaaa]`
    factor `(A,B,I,J)` by the two-term abab sum `t2ab(A,B,I,J) − t2ab(B,A,I,J)`
    (virtual-slot swap, S2.0 in `[v,v,o,o]` layout), splitting the host term in
    two; multiple such factors take the Cartesian product, signs folded into the
    coefficient. `t1[aa]`/`f[aa]` are already the single spatial block (S2.2a
    dropped their `bb` partner) so they pass through; the INTEGRAL `v[aaaa]` is
    left for S2.2c. This is the first step where coefficients change (one term →
    two). *Gate* (`S22bAmplitudeCollapseTests`, ccd+ccsd): no `t2[aaaa]` survives
    (every t2 is `abab`), the term count grew (split fired), tag/spin consistency
    is preserved, and the `abab` residual value is unchanged (~1e-13) — proving
    the symbolic split equals the numeric S2.0. Sign-mutation verified to fail.

  - **S2.2c — integral collapse `v[aaaa] → v[abab] − P(v[abab])` (~M).** The
    symbolic rewrite mirrors S2.2b exactly: the same-spin `v[aaaa](p,q,r,s)`
    splits into `v[abab](p,q,r,s) − v[abab](p,q,s,r)` (**ket**-slot swap, vs the
    virtual swap for `t2`), leaving `v[abab]` as the single spatial antisymmetric
    physicist integral `⟨pq||rs⟩`. Coefficients change (one term → two), same
    shape as S2.2b. The physicist→**chemist** `2J − K` re-expression is NOT here
    — it is an emit concern (S3) / arrives with the `rccsd` compare (S2.2d); S2.2c
    lands the residual in a single spatial `⟨pq||rs⟩` block.

    **Prerequisite — fix the fixture's antisymmetry (this was the S2.2c blocker,
    now diagnosed).** The rewrite is trivial; the gate needs a valid tensor pair.
    Two empirical findings from scoping:
    - ccgen's `v` is a generic **antisym-in-pairs** tensor (bra = slots 0,1; ket
      = slots 2,3; `v == v.T(1,0,3,2)`) — NOT a full `⟨pq||rs⟩` with particle
      symmetry. The evaluator (`residual_einsum`) only requires bra/ket
      antisymmetry.
    - The current `_closed_shell_tensors` **`t2` is not antisymmetric**
      (`t2 ≠ −t2.T(1,0,2,3)`), and its `v` is only spin-conserving. S2.1/S2.2a/b
      are pure block-slicing identities that pass on any `v` and any `t2`, so the
      non-antisym fixture slipped through — but it is NOT a valid CC tensor.
      Pairing it with a real antisym `v` diverges the evaluator (21/… terms).

    **S2.2c-1 — the rewrite. LANDED (structural gate).** `collapse_integrals`
    + `_split_vaaaa` in `ccgen/spin.py`: split each same-spin `v[aaaa](p,q,r,s)`
    into `v[abab](p,q,r,s) − v[abab](p,q,s,r)` (the **ket**-slot swap, vs the
    virtual/bra swap for `t2`), exactly parallel to `collapse_amplitudes`. After
    the full S2.2a→b→c pipeline every doubles factor is a single spatial block
    (`t2`/`v` abab, `t1`/`f` aa). *Gate* (`S22cIntegralCollapseStructureTests`,
    ccd+ccsd): no `v[aaaa]` survives, the split fired, tag/spin consistency;
    mutation-verified (skipping the split fails the gate).

    **S2.2c-0 — the numeric no-op gate is IMPOSSIBLE here; this is a real
    finding, not a TODO.** Unlike the `t2` collapse (S2.2b, numeric no-op ~1e-13
    on the synthetic fixture), the `v` collapse cannot be numerically gated on
    ccgen's own `v`, for a structural reason:

    - **ccgen's `v` is spin-conserving-PER-LINE, not a full antisymmetric
      integral.** Each line (slot k / k+2) conserves spin; the exchange
      contribution lives in *separate ccgen terms*, not folded into `v`. So the
      closed-shell relation `v[aaaa] = v[abab] − v[abab](ket swap)` fails on
      ccgen's `v`: the ket-swapped `abab` entry is spin-forbidden → zero, so the
      relation degenerates to `v[aaaa] = v[abab]`. The split is then NOT a no-op
      and diverges (~68 on the fixture).
    - A **real antisymmetric** integral (which folds in exchange) *does* satisfy
      the relation, but is NOT evaluator-consistent: it fills the `abba`
      exchange block (e.g. `v(iα,cβ,kβ,aα)` ≈ 0.15) that ccgen's block model
      requires zero, so `residual_einsum` diverges on the ring `t2·v` terms.
      Confirmed the minimal reproducer: the ring term integrates to
      `−t2[baba]·v[aaaa]` with `v(i,c,k,a)`, whose `abba` slice is nonzero for
      the real integral but must be zero in ccgen's convention.
    - **GHF ruled out** (closed-shell GHF is spin-mixed, no clean α/β partition).
    - The full PySCF-`uccsd` route was built (real `t2` from `cc.t2` blocks:
      antisym ~1.9e-8, block reln ~4e-9; real `v` from `ao2mo` `(pr|qs)`:
      antisym ~1e-15, block reln 0) — and it confirmed exactly this: the real
      `v` breaks the evaluator on the same ring terms. It is the same wall from
      the integral side.

    **Conclusion:** the `v[aaaa]→v[abab]−P` collapse is the physicist→chemist
    `2J − K` step, and its correctness can only be validated once the residual is
    expressed in the **chemist single-integral** form on **real integrals** —
    i.e. at S2.2d, end-to-end against `rccsd.update_amps`, not as a per-step
    no-op. The rewrite is structurally landed; its numeric proof is deferred to
    S2.2d by necessity, not by choice.

  - **S2.2d — merge + numeric proof of the whole collapse vs PySCF (~M; carries
    the deferred S2.2c validation).** Input: the collapsed CCD/CCSD doubles
    `SpinTerm`s (all single-block: `t2`/`v` abab, `t1`/`f` aa). Measured on CCD
    doubles: manifold 16 → amp-collapse 23 → integral-collapse 33 terms, 25
    distinct raw signatures / 6 obvious dup groups; the `2J−K` pair is already
    visible as e.g. `−t2(b,c,j,k)v(i,c,k,a)` + `+t2(b,c,j,k)v(i,c,a,k)`.

    **The key de-risking decision (from studying the proven oracle):** do NOT
    gate by reproducing `rccsd.update_amps` element-wise — that needs a matched
    physicist↔chemist `v` slicing, exactly the wall S2.2c hit. Instead reuse the
    **"evaluate at PySCF's converged amplitudes"** trick that
    `ccgen_energy_at_pyscf_amps` (`test_reference_vs_pyscf.py`) already uses for
    the GCC path: at the CC solution the amplitude residual is ~0 and the energy
    is exactly `E_corr`. Plugging PySCF's converged RCCSD `(t1,t2)` into the
    collapsed RCC residual must give ~0 (and the collapsed RCC energy = PySCF
    `E_corr`). The energy is a fully-contracted scalar → **convention-robust**,
    so it sidesteps the `v`-slice confound that blocked a per-term S2.2c gate.

    Sub-steps:
    - **S2.2d-0 — spatial RCC residual evaluator. LANDED.**
      `_rcc_doubles_residual` in `test_spin.py`: sums the single-block
      `SpinTerm`s as plain spatial contractions (each collapsed factor is one
      fixed block, so `_eval_spinterm`'s spin+space slice IS the spatial slice —
      no new evaluator needed, just the sum wrapper). *Gate*
      (`S22d0SpatialResidualTests`, ccd+ccsd): on the synthetic fixture the
      **amplitude-collapsed** (post-S2.2b) residual reproduces the sliced `abab`
      GCC residual to ~1e-13, PySCF-free — reconnecting the collapsed form to the
      S1/S2.1 identity. **Refined finding:** the gate MUST run on the amp
      manifold, not the full (post-S2.2c) one — the integral collapse is NOT
      value-preserving on the synthetic spin-conserving `v` (`v[aaaa]=v[abab]−P`
      needs exchange `v` lacks; measured ~74 off). The test asserts BOTH the amp
      baseline holds AND the full collapse does NOT match on synthetic `v`, so
      the deferral of the `v`-split's numeric proof to S2.2d-2 (real integrals)
      is pinned as an explicit invariant, not a silent gap. Mutation-verified.
    - **S2.2d-1 — merge like terms. LANDED.** `merge_terms` + `_merge_signature`
      in `ccgen/spin.py`: group collapsed `SpinTerm`s by a factor-order- and
      summed-index-relabel-invariant signature (external names verbatim; summed
      names → positional placeholders, minimized over summed permutations ×
      factor orderings), sum coefficients, drop zero groups. *Gate*
      (`S22d1MergeTests`, ccd+ccsd): residual value unchanged (merge is pure
      algebra, so value-preserving on the synthetic fixture ~1e-14 regardless of
      `v`), term count actually dropped (33→25 on ccd), idempotent, and — the
      payoff — the **RCC `2J − K` coefficients appear** (merged `|coeff|` now
      includes `2` and `4`, absent from the ≤1 un-merged coeffs). That is the
      genuine derivation content of the whole S2 collapse surfacing. Mutation-
      verified (dropping the coefficient sum fails the value gate).
    - **S2.2d-2 — numeric proof vs PySCF RCCSD. BLOCKED; ROOT CAUSE ISOLATED to
      the S1.2 block filter (fully diagnosed, verified both ways).** The plan
      (evaluate merged residual at PySCF `(t1,t2)`, expect ~0) gave ~0.06.
      Tracing it on water/STO-3G RCCSD (nocc 5, nvir 2, `E_corr −0.0498`)
      produced a clean, verified chain:

      - **GCC is correct on real antisymmetric tensors.** Built spin-orbital
        `v = ⟨pq||rs⟩` (antisym, from `ao2mo` `(pr|qs)` minus exchange), `t2`
        from `cc.t2` closed-shell fill (verified antisym + block relation), `f`
        diagonal. GCC doubles residual at PySCF amps = **3.4e-7**, energy matches
        PySCF to **1.3e-8**. So GCC folds exchange INTO the antisymmetric `v`
        (NOT into separate terms — the earlier guess here was wrong).
      - **The raw S2.1 identity fails on those same real tensors** —
        `ucc_manifold["abab"]` == GCC abab slice is off **0.064** (passes ~1e-13
        on the synthetic fixture). Per-term breakdown: the mismatch is on every
        term whose factors have nonzero content in a spin-non-conserving-per-line
        block. Minimal reproducer: `f(a,c)·t2(b,c,i,j)` — UCC yields **zero
        survivors** (block filter drops it) but GCC gives 100 nonzero elements,
        because real `t2` (and `v`) have nonzero exchange (`abba`) blocks the
        filter assumes zero.
      - **Exact culprit + fix, both verified:** the S1.2 `block_exists` filter is
        the sole error. Re-running the UCC integration with the filter REMOVED
        (sum over ALL summed-spin cases, keep every case) reproduces the GCC abab
        slice on the real antisymmetric tensors to **2e-17**. So the machinery is
        right; only the filter — valid solely when forbidden blocks are zero
        (spin-conserving tensors) — is wrong for real integrals.
      - **Why the synthetic fixture hid it:** `_spin_structure_all` / the
        `_closed_shell_tensors` `v` are spin-conserving-per-line (forbidden
        blocks zeroed), so the filter drops exactly the zero terms and the
        identity holds. Real `⟨pq||rs⟩`/`t2` are antisymmetric and violate this.

      **Implication — this is the real content of spin adaptation, previously
      hidden.** The block filter is not a harmless optimization: dropping the
      "forbidden" cases silently discards the exchange contributions a real
      antisymmetric integral carries. A correct UCC/RCC reduction must either
      (a) keep those cases and re-express the antisymmetric `v` as chemist
      Coulomb + explicit exchange terms (the genuine `2J − K` derivation — where
      the exchange becomes the separate swapped-`v` term the S2.2c split already
      produces), or (b) prove that on a real antisymmetric `v` the filtered sum
      plus the collapse coefficients equals the no-filter sum. The `2e-17`
      no-filter result is the anchor: it says the target is well-defined and the
      only question is bookkeeping the exchange back in.

      **RESOLUTION — LANDED (`ucc_integrate_term_antisym`).** The fix is exactly
      the "keep the cases and re-express" route. A forbidden block of an
      ANTISYMMETRIC rank-4 factor is not zero — it equals an allowed
      (spin-conserving-per-line) block reached by swapping the bra pair (slots
      0,1) and/or ket pair (slots 2,3), each swap carrying a sign `−1`. So
      `_antisym_to_allowed` maps every factor into its allowed block with that
      sign instead of dropping the case; a case is dropped only when a rank-2
      line (f/t1) has genuinely mismatched spins. The `−1` swap signs ARE the
      exchange (`−K`); with the S2 collapse + merge they become the RCC `2J − K`
      combinations. *Gate* (`S1AntisymIntegrationTests`, PySCF-guarded): summed
      over the whole CCD and CCSD manifolds, `ucc_integrate_term_antisym`
      reproduces the GCC abab (doubles) and aa (singles) residual on the REAL
      antisymmetric water/STO-3G integrals to **~1e-16**; plus a tensor-oracle
      check that ccgen's GCC CCSD residual vanishes and energy == `cc.e_corr` at
      converged amps. Exchange-sign mutation-verified. The plain
      `ucc_integrate_term` (filter) is retained — it is exact for the synthetic
      spin-conserving fixture the S1/S2 structure tests use, and
      `_antisym_to_allowed` degrades to it there.

      **END-TO-END LANDED (`S22dEndToEndTests` + `_rcc_doubles_pipeline`).** The
      full S2.2a→d collapse rebuilt on the antisym integration —
      `ucc_integrate_term_antisym` → `canonicalize_spin_blocks` →
      `collapse_amplitudes` → `collapse_integrals` → `merge_terms` — reproduces
      the GCC abab residual on the REAL antisymmetric water/STO-3G integrals to
      **~1e-16 at every stage**, and the merged spatial RCC doubles residual
      **vanishes at PySCF's converged RCCSD amps (3.4e-7)** — the collapsed
      equation IS the RCC residual. Merged CCSD coefficients carry the RCC
      `2J − K` combinations (`{−2, 2, 4, −3/2, …}`). Two gates: the identity
      (merged RCC == GCC abab for any amps, ~1e-10) and vanishing-at-solution;
      merge-mutation-verified. **This is the original S2.2d-2 goal, achieved.**
      The SINGLES spatial residual runs through the identical pipeline
      (`_rcc_pipeline(method, "singles")`, external `{a,i}` — the collapse steps
      act on `t2[aaaa]`/`v[aaaa]` factors so they are block-agnostic): merged RCC
      singles == GCC aa slice ~1e-16 and vanishes at converged amps
      (`test_ccsd_rcc_singles_matches_gcc_slice`). **S2.2 (symbolic collapse) is
      complete end-to-end on real integrals for CCD/CCSD, both singles and
      doubles.** Remaining in the layer: S3 emit wiring.

  Do the singles residual after doubles (same four sub-steps, smaller); CCSD
  needs both. CCD (doubles only) is the first target so S2.2a–d land against the
  simpler manifold before singles cross-terms enter.

- **S2.3 — RCC energy end-to-end (~S).** RCC energy from the collapsed equations
  reaches the PySCF RCCSD energy (iterate harness on the single-block RHF
  tensors). *Gate:* ~1e-9 on a small closed-shell system.

**Recommended first move:** S2.0 + S2.1 — numeric, gated purely against PySCF,
*prove the model before writing collapse code*. **DONE** (see LANDED above): the
"abab + substitution" model is validated for CCD/CCSD before any symbolic work.
S2.2 (symbolic collapse) is scoped into S2.2a–d below. **S2.2a, S2.2b, and the
S2.2c-1 rewrite are LANDED** (block canonicalization, amplitude collapse,
integral `v[aaaa]→v[abab]−P` split — the last gated STRUCTURALLY only). A firm
finding fell out of S2.2c: ccgen's `v` is spin-conserving-per-line, so the `v`
collapse **cannot** be numerically gated per-step (the relation needs exchange
`v` carries elsewhere); its numeric proof necessarily moves to **S2.2d**, now
scoped into S2.2d-0/1/2 below. **S2.2d-0 and S2.2d-1 are LANDED**
(spatial RCC residual evaluator; merge like terms — where the RCC `2J − K`
coefficients appear). **S2.2 (symbolic collapse) is COMPLETE end-to-end on
real integrals for CCD/CCSD doubles.** The S1.2-block-filter blocker was resolved
by `ucc_integrate_term_antisym` (re-express each forbidden factor into its
allowed block via bra/ket swaps with sign `−1` = the `−K` exchange, instead of
dropping it), and the full S2.2a→d pipeline rebuilt on it
(`_rcc_doubles_pipeline`) reproduces the GCC abab residual on real antisymmetric
water/STO-3G integrals to ~1e-16 and the merged RCC residual vanishes at PySCF's
converged RCCSD amps (`S22dEndToEndTests`). The RCC `2J − K` coefficients appear
in the merged form. Singles run through the identical pipeline (also gated on
real integrals). **S2.2 is complete for CCD/CCSD singles + doubles.** Next: S3
(emit wiring — route adapted terms into lowering + confirm the emitters produce
spin-blocked RCC kernels reaching the PySCF energy).

- **S3 — wire the adapted terms into lowering + emit (~M).** The S2 pipeline
  produces merged spatial `SpinTerm`s (RCC, single-block per factor, with the
  `2J − K` coefficients). The emit path (`emit/planck_tensor_cpp.py` via
  `lowering/restricted_closed_shell.py`) consumes `AlgebraTerm`s. S3 bridges the
  two. Existing state that shapes the work:
  - `restricted_closed_shell` lowering already exists but is **layout-only** — it
    "does not attempt a full symbolic spin summation" (its own docstring); it
    re-lays-out spin-orbital terms into occ/vir blocks. Our `SpinTerm`s are
    ALREADY spin-summed to spatial RCC, so lowering becomes a near-passthrough on
    them (block signature + spatial index order), not a spin step.
  - The end gate is the **"evaluate at PySCF converged amps"** trick that already
    works for GCC (`ccgen_energy_at_pyscf_amps` in `test_reference_vs_pyscf.py`,
    and the S2.2d `_rcc_pipeline` residual). There is NO `ccgen_iterate_amps`
    harness to build — reuse the energy-at-amps scalar gate (convention-robust).

  Sub-steps:
  - **S3.0 — `SpinTerm → AlgebraTerm` bridge. LANDED.**
    `spinterm_to_algebraterm` in `ccgen/spin.py`: each factor → `Tensor` over the
    spatial `SpinIndex.base` indices; free/summed split by external-name
    membership (de-duplicated, first-appearance); coefficient as `Fraction`;
    `connected=True`. *Gate* (`S30BridgeTests`, PySCF-free — the bridge is a pure
    structural transform, tensor-value-independent): every converted term
    preserves the coefficient, the factor names + per-factor spatial index names
    AND spaces, and the free/summed partition (`free = allnames ∩ externals`,
    `summed = allnames − externals`, disjoint+complete). CCD/CCSD, singles +
    doubles. Coefficient-mutation-verified. (Per-term evaluation equivalence to
    the SpinTerm was confirmed manually via matched slicing; the structural gate
    is the durable check.)
  - **S3.1 — lower the bridged terms. LANDED.** The converted RCC `AlgebraTerm`s
    lower cleanly through `lower_term_restricted_closed_shell` — coefficient
    carried, canonical free indices in the manifold occ/vir signature, every
    factor gets a valid block signature (o/v glyphs, length = rank), amplitude
    factors in their canonical block (`t1→ov`, `t2→oovv` — note a singles
    residual still contains `t2` factors), and `v` factors mapped to a canonical
    ERI block (`oovv/ovov/…`) with a ±1 phase. *Gate* (`S31LoweringTests`,
    PySCF-free): matches the STRUCTURAL house style of the existing
    `test_restricted_closed_shell_lowering_*` regressions — block/space/phase
    layout, not a numeric round-trip; the numeric proof is the S3.2 energy gate.
    Mutation-verified (corrupting the block-glyph map fails it). Since the terms
    are already spatial RCC, lowering acts as layout-only, exactly as intended.
  - **S3.2 — end-to-end energy + emit. NUMERIC HALF LANDED; C++ emission
    remains.**
    - *Numeric (LANDED, `S32EnergyTests`).* The spin-adapted RCC ENERGY
      expression (`E = f_ia t1 + ¼ t2 v + ½ t1 t1 v`) runs through the same
      pipeline with an EMPTY external block (fully contracted scalar) and reaches
      PySCF's RCCSD `E_corr` to **~1e-8** at converged amps on real water/STO-3G
      integrals. `_rcc_pipeline` now handles `block="energy"`. Together with the
      singles+doubles residuals vanishing (`S22dEndToEndTests`), this is the
      convention-robust "evaluate at PySCF amps" end-to-end proof of the WHOLE
      adapted RCC equation set (energy + both residuals) — the numeric gate the
      doc called for.
    - *C++ emission (remaining).* Route the adapted RCC `AlgebraTerm` dict
      (energy/singles/doubles from `spinterm_to_algebraterm` on the pipeline
      output) through `emit_planck_translation_unit`, and confirm the emitted C++
      compiles against the real CC headers (the `tau` A1
      `test_generated_source_compiles` harness is the template). This is
      mechanical codegen wiring, not new algebra — the correctness is already
      proven numerically above.

  A UCC lowering (distinct α/β blocks) is deferred — RCC is the closed-shell
  production target and the harder coefficient case; UCC reuses the same bridge
  with the block tags kept spin-resolved.

- **S4 — higher rank + engine-agnostic (~M).** Confirm adaptation is
  engine-agnostic (it operates on `AlgebraTerm`s) and extends to CCSDT (gated vs
  `rccsdt` / `uccsdt`).

## Honest boundaries

- **S2 (RCC coefficient collapse) is the hard part** — the α ≡ β reduction's
  coefficient algebra is where the real derivation lives. S0/S1 turned out to be
  bookkeeping, exactly as expected: the UCC raw GCC coefficients came out right
  unchanged (S1.3b). S2 is the one step where coefficients genuinely change.
- **RESOLVED — S2.2 complete end-to-end on real integrals (CCD/CCSD doubles).**
  The `block_exists` filter was not a harmless optimization — S1.2/S1.3b/S2.1
  were all gated on a spin-conserving-per-line synthetic `v` whose forbidden
  blocks are zero, so the filter only ever dropped zero terms. On a REAL
  antisymmetric `⟨pq||rs⟩`/`t2` those blocks are nonzero (they ARE the exchange)
  and the filter silently discarded them (raw S2.1 off ~0.06).
  `ucc_integrate_term_antisym` re-expresses each forbidden factor into its
  allowed block via bra/ket swaps carrying `−1` (the `−K`), and the full
  S2.2a→d pipeline rebuilt on it reproduces the GCC residual on real integrals
  to ~1e-16 and vanishes at PySCF's converged RCCSD amps. The `2J − K`
  coefficients appear in the merged form. Singles run through the identical
  pipeline (gated on real integrals). Complete for CCD/CCSD singles + doubles;
  remaining for the layer: S3 emit wiring.
- **On the production critical path (intent confirmed).** The ccgen-generated
  kernels are intended to replace the hand-written `src/post_hf/cc/` solvers in
  production, once D7 (dressing on diagrams) is done. Production RHF/UHF CC needs
  spatial RCC/UCC kernels, and the generated path is spin-orbital (GCC) — so this
  layer (S1 UCC, S2 RCC) is a **prerequisite for the production swap**, not
  optional research. The "not compiled into any binary" state elsewhere in these
  docs is the current state, not the end state.
- **Not a rewrite.** An insertable stage; generation, lowering, and emit are
  untouched. But S1+S2 are each large.
