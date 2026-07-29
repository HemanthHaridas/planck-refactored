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

- **S4 — higher rank (CCSDT/CCSDTQ) + engine-agnostic (~M-L). NUMERICALLY
  COMPLETE through rank-8.** The core rank generalization (`_antisym_to_allowed`)
  is landed and NUMERICALLY GATED at rank-6 (S4a.0c + S4a.1) AND rank-8 (S4d —
  the production antisym integration reproduces the GCC quadruples slice on an
  FCI-limit-converged closed-shell antisym `t4`). The rank-2n amplitude splitter
  (S4b) is landed; the integral splitter is a confirmed no-op (S4c — `v` is always
  rank-4). The rank-6 `t3` fixture (map.1→map.3) is complete. Only S4a.2 (the
  optional unified lift consolidation) remains, and S4d showed it is not needed
  for correctness. The GCC generation handles
  arbitrary rank (its `_line_pairs`/`block_exists` are general rank-2n), and the
  diagram engine emits CCSDT (triples, 414 terms) and CCSDTQ (quadruples, 2728
  terms) fine (`generate_cc_equations(m, engine="diagram")`). But the
  spin-adaptation layer has rank-4 hardcodings, so it does NOT extend as-is —
  and this bites even the CCSDTQ singles/doubles residuals, which contain `t3`
  (rank-6) and `t4` (rank-8) factors (measured: doubles 11/79 terms, triples
  140/428, quadruples 1612/2728 have a rank>4 factor; max factor rank 8).

  The three rank-4 hardcodings:
  1. **`_antisym_to_allowed` — GENERALIZED (LANDED).** The old version had only
     rank-2/rank-4 candidate swaps; on a rank-6/8 factor it fell through and
     returned `None` ("genuinely zero") for cases actually reachable by
     antisymmetry, silently DROPPING valid terms. Replaced with a general rank-2n
     version: a factor is antisym within the bra group (slots `0..n-1`) and ket
     group (`n..2n-1`); it maps to an allowed (spin-conserving-per-line) block iff
     `sorted(bra_spins) == sorted(ket_spins)` (multiset match), via a within-group
     permutation whose parity product (`_permutation_parity`) is the sign;
     `None` only on a genuine multiset mismatch. On rank-4 it picks a different
     canonical block than the old 4-candidate path for 2 patterns (`abab` vs
     `baba`) — but those are **provably equivalent** (they differ by a bra-swap +
     ket-swap, net sign +1, block flips; verified to evaluate identically to 0.0
     on a random antisym tensor). Confirmed to reproduce GCC at rank-4 both raw
     and through the full collapse+merge pipeline (~1e-17), so it is a safe
     **drop-in replacement** — all rank-4 numeric gates stay green. Gated at
     rank-6 STRUCTURALLY (`S4HigherRankTests`): a rank-6 `t3` factor maps to an
     allowed block with ±1 (not `None`) on multiset match and `None` only on
     mismatch, and the CCSDT triples manifold integrates to a nonzero survivor
     set (the old bug dropped them all). Mutation-verified (re-limiting to
     rank ≤ 4 fails the gate).
  2. **`_split_t2aaaa` / `_split_vaaaa`** are still hardcoded 4-index — the
     same-spin higher-rank amplitude blocks (`t3[aaaaaa]` etc.) need their own
     antisym split, or a general rank-2n splitter. **The rank-2n relation is now
     PINNED (S4b.0, LANDED):** the all-alpha same-spin block `t_n[a..a]`
     reconstructs from the mixed block whose bra spins are `(a,..,a,b)` (single
     beta bra-slot at position n-1) by **BRA-ONLY antisymmetrization** — the
     signed sum over placing that beta bra-slot in each of the n bra positions,
     with the transposition sign, **ket FIXED**. At n=2 this is exactly
     `_split_t2aaaa`'s `t2[aaaa] = t2[abab] − t2[abab](bra swap)`; at n=3 it is a
     3-term sum reproducing `t3[aaaaaa]` from the `aab` block. Verified on the
     real UCCSDT closed-shell antisym fixture to ~1e-17 at both ranks
     (`S4bZeroCollapseRelationTests`); a JOINT bra+ket swap does NOT reproduce it
     (~0.014), so bra-only is load-bearing.

     **S4b.1 + S4b.2 LANDED.** `_split_same_spin_amplitude` in `ccgen/spin.py`
     replaces `_split_t2aaaa` (kept as a back-compat alias): a same-spin all-alpha
     `t_n[a..a]` factor (any rank-2n) splits into the fixed mixed block (bra/ket
     spins `(a,..,a,b)`, block `"abab"` at n=2, `"aabaab"` at n=3) by permuting the
     VIRTUAL base indices so each virtual takes a turn in the beta slot (ket base
     fixed), sign = the base-permutation parity. `collapse_amplitudes` now
     dispatches via `_is_same_spin_amplitude` (name starts `t`, even block length
     >= 4, all-`a`) so t3/t4 collapse too, not just t2. *Gates*
     (`S4bSplitterTests`): the summed split reproduces the all-alpha block on the
     real fixture at rank-4 AND rank-6 (~1e-12); `collapse_amplitudes` leaves no
     all-alpha t3 factor (3 terms out); t1 is not split; tag/spin consistency
     holds. **Byte-identical to the old `_split_t2aaaa` at n=2**, so the S2.2b
     rank-4 regression (`S22bAmplitudeCollapseTests`) stays green. The integral
     splitter `_split_vaaaa` (S4c) is separate; scan first whether any rank>4 `v`
     even appears (likely a no-op — `v` stays rank-4 across CCSDT/CCSDTQ).
  3. The closed-shell block relations generalize (`t3aa… = t3 mixed − P`) but the
     coefficient collapse for rank ≥ 6 is unverified.

  **Remaining for S4:** (a) the rank-6/8 NUMERIC gate — the S1' identity
  (`ucc_integrate_term_antisym` == GCC slice) at rank-6/8. This needs a
  **closed-shell ANTISYMMETRIC `t3`/`t4` fixture** (both properties at once): the
  antisym re-expression is INVALID on spin-structured tensors whose forbidden
  blocks are artificially zeroed (a zero forbidden block must NOT be re-expressed
  to a nonzero allowed one — verified this breaks, 108/126 terms), and raw random
  antisym tensors aren't closed-shell (α≢β).

  ### S4a — the closed-shell spatial→spin-orbital-antisym amplitude lift (`t3`, then arbitrary order)

  This is the load-bearing piece of the rank-6/8 numeric gate: given a closed-
  shell RHF-based spatial amplitude, produce the genuinely-antisymmetric
  spin-orbital `t_n` that ccgen's GCC equations consume. It is `so_t2` (working,
  in `_real_antisym_tensors`) generalized to rank ≥ 3.

  *Oracle (confirmed working):* **`pyscf.cc.rccsdt.RCCSDT`** (PySCF 2.13.0)
  converges and gives `cc.t1/t2/t3` + `cc.e_corr`. `cc.t3` is triangular-packed;
  `tamps_tri2full_rhf(cc, cc.t3)` → full `[o,o,o,v,v,v]`. That `t3full` is the RCC
  **symmetric** representation: symmetric under simultaneous particle interchange
  `(i,a)↔(j,b)` (verified `0`), NOT antisymmetric in `(i,j)` alone (`~1e-3`).
  (`RCCSDTQ`/`cc_order=4` gives `t4` the same way for the rank-8 gate.)

  *What's been tried:* two antisymmetrization-based lifts (full-bra/ket ÷6; and
  same-spin-group). Both got the ENERGY right to 1e-9 on LiH but that was
  misleading — S4a.0a (below) showed on N2 that the same-spin-group lift produces
  an **all-zero `t3so`**, so any antisymmetrization-of-`t3full` approach is wrong.
  The energy survived only because LiH's `t3` is negligibly small.

  *Diagnostics gathered (these reshape S4a.0 — the earlier "missing scalar
  weight" hypothesis is DISPROVEN):*
  - Scaling the lifted `t3so` by any factor (0.25…6) leaves the triples residual
    **unchanged** at ~2.4e-3 — so the error is NOT a scalar weight.
  - With `t3so = 0` the triples residual is the **same** ~2.4e-3, and even a
    `t3` solved directly from the residual (`t3corr = R0/e_ijkabc`) only drops it
    to ~5e-4 — so on this system `t3` barely moves the residual.
  - Root reason: **LiH/STO-3G is too weakly correlated** — `|t3full| ≈ 0.0019`,
    the same order as the residual floor. A ~few-% lift error is invisible
    against a near-zero `t3`. Strong-correlation systems have `|t3| ≈ 0.03`
    (`N2/STO-3G` 0.0315; linear `H4` stretched 0.0416) — **~20× larger**, so a
    lift error there shows up ~20× more clearly. HF-stretched has `t3 ≈ 0` (bad
    fixture). So the gate needs a strong-correlation system, not LiH.
  - PySCF's `t3_spin_summation_inplace_` is C-backed pattern code
    (`P3_full`/`P3_201`/`P3_422`) used *inside* its iteration, NOT a clean
    "give me the antisym `t3`" bridge (applying `P3_full` `-1/6, beta=1` to
    `t3full` barely changes it), so reverse-engineering it is the wrong lever.

  **S4a.0a DONE — and it found the root cause: the lift approach is wrong, not
  just mis-weighted.** Built the N2/STO-3G RCCSDT fixture (`|t3| = 0.0315`,
  nocc 7 / nvir 3). Two decisive results:
  - The same-spin-group-antisymmetrizer lift produces a **literally all-zero
    `t3so`** on N2 (norm 0) — it contributes *nothing* to the triples residual
    (identical to `t3so = 0`). It "worked" on LiH only because LiH's `t3` is so
    tiny the zero was within the residual noise.
  - **Why it zeros: antisymmetrizing the RCC-symmetric `t3full` self-cancels.**
    The lift antisymmetrizes over line-paired swaps within same-spin groups —
    but a line-paired simultaneous swap `(i,a)↔(j,b)` IS exactly `t3full`'s
    built-in RCC symmetry (verified `t3full[i,j,k,a,b,c] − t3full[j,i,k,b,a,c] =
    0.0`). So the antisymmetrizer subtracts `t3full` from itself. (`t2` escaped
    this because its same-spin `aaaa` block IS `t2ab − P(t2ab)` with a genuinely
    non-symmetric mixed block; the rank-3 symmetric rep is different.)
  - Corollary: **the antisym spin-orbital `t3` is NOT a permutation of `t3full`**
    — antisymmetrization annihilates the symmetric rep's content. The lift must
    be the **spin-summation INVERSE** (the antisym `t3` that, spin-summed
    forward, gives `t3full`), a genuine rank-6 transform, not a reshuffle.
  - Also: the fock-diagonal `t3corr = R0/e` is NOT a clean oracle — the triples
    residual has many non-fock `t3` terms (`W·t3`, …), so `R0/e` only captures
    part (drove N2 residual to 2e-3, not ~0). A correct oracle is ccgen's own
    GCC-CCSDT converged `t3` (Jacobi-iterate the GCC residual to self-consistency
    on the real integrals) or the true spin-summation inverse of `t3full`.

  **S4a.0b IN PROGRESS — a MUCH better oracle found: PySCF `uccsdt`.** Instead of
  inverting RCCSDT's symmetric `t3full`, use **`pyscf.cc.uccsdt.UCCSDT`** (also
  in 2.13.0). On a closed-shell UHF≡RHF reference it converges to the same
  `e_corr` (N2: −0.2194) and stores `t3` in explicit SPIN BLOCKS
  (`aaa`, `aab`, `abb`, `bbb`); `tamps_tri2full_uhf(cc, cc.t3)` unpacks them to
  full tensors. Crucially the `aaa` block is **genuinely antisymmetric** (verified
  `aaa == −aaa.T` in both `(i,j)` and `(a,b)`, unlike RCCSDT's symmetric
  `t3full`). So the antisym spin-orbital `t3` assembles DIRECTLY from these
  blocks — no spin-summation inverse to derive.

  *Progress + the remaining precise task:*
  - **UHF symmetry-breaking gotcha (found + fixed).** A bare `scf.UHF(N2)` at
    1.3 Å converges to a symmetry-BROKEN solution (α/β MOs differ by 0.28 though
    `<S²>≈0`), so `aaa ≠ bbb` (off ~0.002). Fix: build the UHF reference from RHF
    orbitals — `mf = scf.addons.convert_to_uhf(scf.RHF(mol).run())` — then
    `aaa == bbb` to 3e-18 and the fixture is genuinely closed-shell. Any
    UCCSDT-fixture code MUST do this convert, not a fresh UHF.
  - **Block layouts pinned:** `aaa/bbb` are `[i,j,k,a,b,c]`; `aab` is
    `[i,j,a,b,k,c]` (verified antisym in axes (0,1) alpha-occ and (2,3)
    alpha-vir; beta `k,c` last); `abb` is `[k,l,c,d,i,a]` = beta-occ,beta-occ,
    beta-vir,beta-vir,alpha-occ,alpha-vir (antisym in (0,1)&(2,3)).
  **Mixed-block mapping — measured facts + the precise remaining derivation.**
  Same-spin-swap antisymmetry is EXACT (0.0); the error is the physical LINE-SWAP
  antisymmetry (swap two `(vir,occ)` line pairs together). Measured `aab`/`abb`
  symmetries (both blocks identical structure): **antisym under the occ-pair swap
  (axes 0,1) ALONE and the vir-pair swap (axes 2,3) ALONE, SYMMETRIC under the
  joint swap.** That joint-symmetry is the trap: a physical line-swap of two
  same-spin lines is exactly the joint (occ+vir) swap → the block returns the
  SAME value, but the spin-orbital `t3` needs a `−1`. Failing example pinned:
  `t3so[1,3,0, 7,13,8]` (spins ββα/ββα, abb-type) vs its line-swap partner —
  both read `−0.00629` instead of ±.

  So the value must be `sign(P) × blk[canonical-line-order]` where `P` is the
  permutation of the THREE LINES (not axes) from the entry to canonical, and
  `sign(P)` is its parity. The block's own single-axis antisymmetry must NOT be
  relied on for the within-group swap (it's symmetric under the joint one);
  instead the explicit line-permutation parity supplies every sign. Attempts so
  far conflate the two — patching one swap's sign breaks another (fixing the
  same-spin transposition left the cross-spin `(a,i)↔(c,k)` at 0.002 while
  `(a,i)↔(b,j)` stayed 0.015).

  **LITERATURE + the authoritative PySCF convention (found — this de-risks the
  whole remaining derivation).** The `pyscf.cc.uccsdt` module docstring states
  its equations are **"derived from the GCCSDT equations in Shavitt & Bartlett,
  *Many-Body Methods in Chemistry and Physics: MBPT and Coupled-Cluster Theory*,
  Cambridge University Press (2009), DOI 10.1017/CBO9780511596834."** That
  textbook (the standard CC reference) is the source of truth for the
  spin-orbital↔spatial `t3` convention and the antisymmetrizer signs. The
  spin-adapted-CCSDT primary literature is Scuseria & Schaefer, *Chem. Phys.
  Lett.* **152**, 382 (1988) (original full CCSDT) and the spin-restricted CCSDT
  working equations in *J. Chem. Phys.* **117**, 7872 (2002). For the closed-shell
  spatial reduction specifically, the UGA-based CCSD reformulation is Scuseria,
  Janssen & Schaefer, *J. Chem. Phys.* **89**, 7382 (1988) — the rank-2 analog of
  exactly this lift. **The PySCF docstring also gives the exact storage
  conventions**, which resolve most of what the trial-and-error was groping for:
  - blocks are `(t3aaa, t3aab, t3bba, t3bbb)` — **`bba` NOT `abb`** (block [2] is
    2-beta-1-alpha), stored with **`i<j` and `a<b`** (symmetry-unique triangle;
    `aaa/bbb` use `i<j<k, a<b<c`). `tamps_tri2full_uhf` unpacks to full antisym
    tensors.
  - `aab` full layout `[i,j,a,b,k,c]` = (nocca,nocca,nvira,nvira,noccb,nvirb);
    `bba` `[i,j,a,b,k,c]` = (noccb,noccb,nvirb,nvirb,nocca,nvira).
  - **`t2ab` is `[i,a,j,b]`** (nocca,nvira,noccb,nvirb) — differs from BOTH rccsd's
    `[i,j,a,b]` AND `pyscf.cc.uccsd`'s `[i,j,a,b]` (the docstring flags this).

  With these pinned, the remaining error is narrow: a direct block read still
  fails the PHYSICAL line-swap `(a,i)↔(b,j)` at ~0.012 (checked with the correct
  line-swap invariant, not the earlier bra-only test which wrongly compared
  spin-conserving vs spin-broken entries). So the residual bug is in the
  mixed-block **line-pairing / relative sign vs `aaa`**, to be resolved by
  matching the Shavitt-Bartlett antisymmetrizer, not by guessing.

  *Scoped sub-steps for the mapping:*
  - **map.1 — pin the canonical read. LANDED.** `_uccsdt_t3_blocks` (the
    RHF→UHF-converted N2 UCCSDT fixture) + `_t3so_canonical_read` in
    `test_spin.py`: the closed-form entry `t3so[a,b,c,i,j,k]` for each spin
    pattern reads the block value at the entry's spatial indices —
    `aaa/bbb[I,J,K,A,B,C]`, `aab/bba[I,J,A,B,K,C]` (the pyscf.cc.uccsdt
    docstring's block layouts; block[2] is `bba`, 2-beta-1-alpha, NOT `abb`).
    *Gate* (`S4aMap1CanonicalReadTests`, PySCF-guarded): on CANONICAL entries
    (each ccgen line `(a,i)/(b,j)/(c,k)` spin-conserving, in the block's stored
    slot order) `t3so == block` for all four spin patterns to 0.0, plus the
    closed-shell `aaa == bbb` fixture check. Inherits the single occ-pair/vir-pair
    antisym from the blocks; the physical line-swap antisymmetry (line reorder)
    is map.2, where `_t3so_canonical_read` returns `"MIXED-ORDER"`. No sign was
    needed for the canonical read — the block read is exact as-is; the
    Shavitt-Bartlett antisymmetrizer signs enter at map.2 for the reordered
    lines.
  - **map.2 — the general (any line order) read. LANDED — and it CORRECTED the
    gate's premise.** `_t3so_read` + `_read_ascending` + `_line_parity` in
    `test_spin.py`. The read sorts the bra (virtuals `a,b,c`) and the ket
    (occupieds `i,j,k`) by spin INDEPENDENTLY (sign = product of the two
    parities), landing on a spin-conserving ascending arrangement, then reads the
    block — the one non-face-value case is the `(0,1,1)` multiset, which PySCF
    stores as `bba` in majority-first `(1,1,0)` order (one extra line-perm with
    its parity).

    **The doc's original map.2 gate was a MISCONCEPTION.** It demanded the three
    PHYSICAL line-swaps `(a,i)↔(b,j)` etc. vanish (`transpose(1,0,2,4,3,5)` → 0).
    They do NOT and MUST not: a genuine GCC `t3` is antisymmetric INDEPENDENTLY
    within the bra and within the ket, so a JOINT line swap is `(−1)(−1) = +1` —
    it is SYMMETRIC under a joint line swap. PySCF's raw `aaa` block confirms this
    directly (`test_ground_truth_block_symmetry`: `aaa == aaa.transpose(joint)`
    to 1e-12, while lone occ-swap and lone vir-swap are each antisym). This is
    also exactly the convention production `spin.py::_antisym_to_allowed` consumes
    (sorts bra and ket spin-multisets independently, sign = product of parities).
    The prior ~0.012 "failure" was measuring the wrong invariant. *Gates*
    (`S4aMap2GeneralReadTests`, PySCF-guarded): (1) `_t3so_read` is antisym under
    every lone vir-swap and lone occ-swap to ~1e-11; (2) it is SYMMETRIC under the
    joint line swap (pinned so the finding cannot silently regress); (3) the
    ground-truth block symmetry above; (4) it reproduces map.1's canonical block
    reads exactly (aaa + aab slots). map.3 (`t2ab` layout) is unaffected.
  - **map.3 — fix the fixture `t2ab` layout. LANDED.** `_uccsdt_so_tensors` in
    `test_spin.py` builds the full spin-orbital `(t1,t2,t3,v,f)` fixture from a
    converged UCCSDT reference, indexing `t2ab` as **`[i,a,j,b]`**
    (nocca,nvira,noccb,nvirb) — vs rccsd's AND `pyscf.cc.uccsd`'s `[i,j,a,b]` —
    with `t2aa/t2bb` as `[i,j,a,b]` (confirmed antisym, `aa==bb` closed-shell).
    *Gates* (`S4aMap3T2LayoutTests`, PySCF-guarded): the assembled spin-orbital
    `t2` is **bit-identical** to the validated `so_t2` fill (rebuilt from the
    transposed `t2ab`) to ~1e-16, `t2` is antisym in both pairs, and the GCC
    ENERGY at these amps hits UCCSDT's `e_corr` to ~1e-15 (a fully-contracted
    scalar → convention-robust, breaks on any `t1/t2/v/f` bug).

    **Deliberately NOT gated here (this is a real finding):** the "doubles
    residual < 1e-7" gate the earlier scoping put on map.3. With EVERY base
    tensor independently proven — energy 1e-15, `t2 == so_t2` 1e-16, `t3`
    round-trips to its UCCSDT blocks EXACTLY (0.0) and passes all map.2
    antisymmetry gates — the doubles/triples residuals at UCCSDT amps still sit at
    ~1e-3, splitting into a CCSD-part remainder (~9e-3) partially canceled by the
    `t3` terms (~6e-3). So the residual is NOT a fixture-layout problem (map.3);
    it isolates a remaining `t3` **contraction-convention** question (the
    slot-to-line assignment ccgen's GCC-CCSDT triples equation expects vs the
    `[a,b,c,i,j,k]` read). That belongs to **S4a.0c** (the numeric residual gate)
    and **S4a.1** (the rank-6 S1' identity), which this note explicitly separates
    from the layout fix. The layout fix is complete; the residual gate moves to
    S4a.0c.
  *Then S4a.0c gate:* `t3so` correctly antisym (independent bra/ket, per map.2 —
  NOT line-antisym; the fixture side is DONE) AND GCC triples residual < 1e-7 at
  UCCSDT amps on N2. **map.1/map.2/map.3 are all LANDED**; what remains for
  S4a.0c is closing the ~1e-3 residual, which is a `t3` contraction-convention
  question (slot-to-line assignment vs ccgen's GCC-CCSDT triples equation), not a
  fixture-assembly one — every base tensor and the `t3` round-trip are proven.
  - **S4a.0c — numeric gate on the strong-correlation fixture. LANDED.**
    `S4a0cTriplesResidualTests`: on the N2/STO-3G UCCSDT fixture (map.3's
    `_uccsdt_so_tensors`, `|t3| ~ 0.03`) ccgen's GCC CCSDT residual VANISHES at
    the converged UCCSDT amps — singles/doubles/triples all < 1e-7 (measured
    ~5e-13 / ~1.5e-11 / ~5e-13) and energy == `e_corr` (~1.7e-15), for BOTH the
    wick and diagram engines. This is the oracle S4a.0a identified (residual
    vanishing at a self-consistent full `t1/t2/t3`), with NO RCCSDT `t3full`
    inversion. It pins the whole map.1→map.3 t3 assembly end-to-end. (An interim
    scratch run showed ~1e-3; that was a transcription artifact in the throwaway
    prototype — the productionized `_uccsdt_so_tensors` vanishes cleanly.)
  - **S4a.1 — the rank-6 S1' identity. LANDED.** `S4a1Rank6IdentityTests`:
    production `ucc_integrate_term_antisym` (driving the general rank-2n
    `_antisym_to_allowed`) reproduces the GCC TRIPLES residual on the real
    closed-shell antisym integrals, sliced to a canonical external block —
    `aaa`-external ~3e-18 and `aab`-external ~2.6e-17 (gate 1e-10). This is the
    rank-6 analog of the rank-4 `S1AntisymIntegrationTests` and the numeric proof
    the S4 STRUCTURAL gate defers: it exercises the PRODUCTION path at rank 6, as
    a per-term algebraic identity (any amps), not residual-vanishing.
  - **S4d — rank-8 (CCSDTQ / `t4`) numeric gate. LANDED (via the FCI-limit
    route, not a `t4` oracle).** The doc originally scoped this as "gate `t4`
    against RCCSDTQ the same way [as uccsdt]" — but PySCF 2.13.0 has **no
    `uccsdtq`**, and `rccsdtq` gives only a SYMMETRIC triangular `t4full` that
    self-cancels under antisymmetrization (the S4a.0a wall, one rank up). A random
    antisym `t4` also fails (the collapse identity needs closed-shell α≡β;
    measured off by 54). **Resolution:** iterate ccgen's GCC CCSDTQ residual to
    self-consistency on RHF even/odd spin-orbitals — for a 4-electron system
    (H4/STO-3G) CCSDTQ == FCI, so the converged amps ARE the exact closed-shell
    antisymmetric tensors, no lift and no oracle needed. GHF was ruled out
    (spin-mixed, no α/β partition); the RHF even/odd basis gives the clean
    partition the adaptation needs. *Gates* (`S4dRank8IdentityTests`,
    PySCF-guarded): the iterated CCSDTQ energy reaches FCI (~3e-8), `t4` is
    closed-shell antisym (~4e-17), and production `ucc_integrate_term_antisym`
    reproduces the GCC QUADRUPLES residual sliced to a mixed `aabb` external block
    (the all-alpha rank-8 block is structurally impossible at 4 electrons) to
    ~1e-16, on a t4 perturbed ×0.5 so the residual is genuinely nonzero (a real
    identity test). This is the rank-8 analog of S4a.1, exercising the general
    rank-2n `_antisym_to_allowed` at rank 8.
  - **S4a.2 — generalize the lift to arbitrary order (~M). Now OPTIONAL /
    lower-priority.** State the lift as one rank-2n rule and provide a single
    `closed_shell_antisym_lift(spatial, n)` replacing the hand-written
    `so_t2`/`_t3so_read`. This is a consolidation refactor, not a capability gate:
    S4d showed the rank-8 numeric proof does NOT need a spatial→antisym lift at
    all (the FCI-limit iterate produces the antisym `t4` directly), so the
    unified lift is only worth building if the hand-written per-rank fills become
    a maintenance burden.

  Then: **(b)** the rank-2n splitters. **The AMPLITUDE splitter is LANDED (S4b):**
  `_split_same_spin_amplitude` generalizes `_split_t2aaaa` to rank-2n (byte-
  identical at n=2), wired into `collapse_amplitudes`; pinned at rank-4/6. **The
  INTEGRAL splitter (`_split_vaaaa`) is a CONFIRMED NO-OP (S4c):** the two-electron
  integral is fundamentally rank-4, so every `v` factor is rank-4 across CCSDT and
  CCSDTQ — no rank-6/8 integral exists in CC theory, so `_split_vaaaa` needs no
  generalization (`S4cIntegralRankTests` pins it). **(c)** the higher-rank collapse
  coefficient check + the rank-8 (`t4`/CCSDTQ) end-to-end gate (S4d) remain.
  Engine-agnostic is already effectively there — the layer operates on
  `AlgebraTerm`s and the diagram-engine manifolds feed the same
  `ucc_integrate_term_antisym` unchanged (and S4a.1 now runs on the diagram engine,
  proven equivalent to wick by S4a.0c).

## Honest boundaries

- **Why this layer exists at all: COST, not correctness (confirmed).** The GCC
  equations, evaluated on spin-orbitals built from an **RHF** reference, already
  give the exact closed-shell CC energy — verified: ccgen's GCC energy on
  RHF-derived amps == PySCF RCCSD `e_corr` to 1e-8, and == PySCF GCCSD (GHF ref)
  to 7e-9. GHF-CC and RCC are energy-equivalent for a closed shell (GHF reduces
  to RHF at the closed-shell minimum). So RCC/UCC are **NOT needed for the right
  number** — they exist purely to exploit spin symmetry for efficiency: the
  spin-orbital representation is ~16× the `t2` storage and ~64× the doubles-
  contraction flops of the spatial one (water/STO-3G; the ratio is
  `2^(2·rank)` and grows with system size). This is exactly the production
  driver: per [[ccgen_generated_kernels_to_production]] the generated kernels
  replace the hand-written `src/post_hf/cc` solvers, and production CC on real
  molecules cannot afford the spin-orbital blowup. So the spin-adaptation layer
  is a **performance prerequisite** for the production swap (alongside D7
  scaling), not a correctness one — a useful thing to keep straight: if the goal
  were only "get the energy," GCC-on-RHF suffices and this layer is unnecessary.
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
