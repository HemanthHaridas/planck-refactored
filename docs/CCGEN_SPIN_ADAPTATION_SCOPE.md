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

- **S2.0 — closed-shell block relations, numeric (~S).** Pin the relations RCC
  rests on: `t1a = t1b`, and `t2aa = t2ab − t2ab.transpose(pair-swap)`
  (determine the exact swap + sign against PySCF). *Gate:* reproduce PySCF's
  `t2aa` from its `t2ab` numerically. Settles the single most error-prone spot —
  the `P(t2ab)` swap convention — before any equation work.

- **S2.1 — RCC residual as a numeric block identity (~M).** Show the RCC single
  residual equals the UCC `abab`-block residual (S1's `ucc_manifold`) evaluated
  with the S2.0 substitution, on RHF tensors, vs PySCF `rccsd.update_amps`.
  *Gate:* per-element ~1e-12. **Proves the "abab + substitution" model is right
  before the symbolic collapse** — if it fails, the model is wrong and no
  symbolic work saves it.

- **S2.2 — symbolic collapse (~L, the core).** Produce the RCC spatial equation:
  take the UCC `abab` `SpinTerm`s, apply `t2aa → t2ab − P(t2ab)` and
  `t1a,t1b → t1` as term rewrites, and merge (where the `2J − K` combos form).
  *Gate:* the symbolic RCC residual evaluated matches PySCF `rccsd.update_amps`
  (~1e-12) and reproduces the S2.1 identity — the symbolic form is the validated
  model, not a new guess.

- **S2.3 — RCC energy end-to-end (~S).** RCC energy from the collapsed equations
  reaches the PySCF RCCSD energy (iterate harness on the single-block RHF
  tensors). *Gate:* ~1e-9 on a small closed-shell system.

**Recommended first move:** S2.0 + S2.1 — numeric, gated purely against PySCF,
*prove the model before writing collapse code*. RCC's `rccsd.update_amps` is the
strongest per-residual oracle in the whole ccgen effort — use it at every step.

- **S3 — wire the stage + emitter compatibility (~M).** Route adapted terms into
  lowering (`restricted_closed_shell` becomes layout-only on already-spatial
  terms; add a UCC lowering) and confirm the emitters produce spin-blocked
  kernels. *Gate:* emitted RCC/UCC kernel reaches the PySCF energy end-to-end
  (reuse the `ccgen_iterate_amps` harness on spatial tensors).

- **S4 — higher rank + engine-agnostic (~M).** Confirm adaptation is
  engine-agnostic (it operates on `AlgebraTerm`s) and extends to CCSDT (gated vs
  `rccsdt` / `uccsdt`).

## Honest boundaries

- **S2 (RCC coefficient collapse) is the hard part** — the α ≡ β reduction's
  coefficient algebra is where the real derivation lives. S0/S1 turned out to be
  bookkeeping, exactly as expected: the UCC raw GCC coefficients came out right
  unchanged (S1.3b). S2 is the one step where coefficients genuinely change.
- **On the production critical path (intent confirmed).** The ccgen-generated
  kernels are intended to replace the hand-written `src/post_hf/cc/` solvers in
  production, once D7 (dressing on diagrams) is done. Production RHF/UHF CC needs
  spatial RCC/UCC kernels, and the generated path is spin-orbital (GCC) — so this
  layer (S1 UCC, S2 RCC) is a **prerequisite for the production swap**, not
  optional research. The "not compiled into any binary" state elsewhere in these
  docs is the current state, not the end state.
- **Not a rewrite.** An insertable stage; generation, lowering, and emit are
  untouched. But S1+S2 are each large.
