# ccgen Derivation-Route Wiring and ERI Symmetry Fix

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

*Successor to the W1-W5 scope of the same name, rewritten as an answer once W4.3 went green (the scope's exemption expires when the work lands). Opened by `CCGEN_TWO_DRESSING_ROUTES.md`, which established that the derivation route was value-gated, worth 2-7x, and had no production caller — and recommended wiring it.*

This file answers a narrower architecture question:

**How does a derived-operator CC kernel reach production, and what was wrong with it?**

## Short answer

ccgen has two routes that produce dressed CC operators. Recognition matches hand-seeded Stanton-Gauss fingerprints; it is retired, 52% short on Be. Derivation builds operators from each term's own contraction tree; it was value-gated, worth 2-7x, and for months had no production caller — deferred in its own commit and never revisited. It has one now. Wiring it exposed an invalid ERI symmetry table that had been silently corrupting 41 of 288 emitted operator builders with a bogus sign, invisible to every existing gate because none of them emit and evaluate the actual C++ text against a non-antisymmetric fixture.

## Where the logic lives

- `python/ccgen/tensors.py` — `SPATIAL_ERI_SYMMETRIES` / `ANTISYMMETRIZED_ERI_SYMMETRIES`, the shared home for the fixed table
- `python/ccgen/generate.py` (lines ~1052/1064/1152, ~1060) — `dress_operators` wiring points
- `python/ccgen/lowering/restricted_closed_shell.py` — `_map_eri_tensor`, where the invalid table used to live
- `python/ccgen/dressing.py` — the retired recognition route; also carries its own `_ERI_PERMUTATIONS` pair (sign-free, deliberately separate)
- `python/ccgen/tests/test_emitted_builder_matches_spec.py`
- `python/ccgen/tests/test_eri_symmetry_tables.py`
- `docs/CCGEN_TWO_DRESSING_ROUTES.md` — established the derivation route's value and recommended wiring it
- `docs/CCGEN_MERGE_TRANSPOSES.md` — the follow-on for threading `merge_transposes` into this path
- `docs/CCGEN_KERNEL_SCALING_SCOPE.md` — the scaling ladder this route has not yet been re-run against
- `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` — O6, the prerequisite for UCC support here

## What invariants matter

### 1. A spatial physicist ERI has exactly four index symmetries, not eight

A spatial physicist `<pq|rs>` over real orbitals has exactly four index symmetries, all `+1`. The four single-swap relations `<qp|rs> = -<pq|rs>` and `<pq|sr> = -<pq|rs>` hold only for the antisymmetrized `<pq||rs>`. `restricted_closed_shell.py` carried the full 8-fold group, and its phase reached the emitted C++ directly through `_map_eri_tensor` returning `LoweredTensorFactor.phase` without re-deriving it. This corrupted 41 of 288 emitted operator builders with a bogus sign and a wrong block read (e.g. `ovov` swapped for `ovvo`), producing a converged-but-wrong answer 161x the tolerance on two independent systems.

Design rule:

- Define the spatial and antisymmetrized ERI symmetry sets exactly once, in a leaf module both the emitter and lowering layers import (`ccgen/tensors.py`), never re-derive or re-declare either set locally.
- A comment documenting an invariant is not sufficient — two warning comments already existed at the sites that broke, and neither prevented a third module from carrying the bad set. Only a test (`test_eri_symmetry_tables.py`, matched on shape so renaming cannot evade it) enforces it.

### 2. A fixture with more symmetry than the real object cannot see a defect that abuses symmetry

The value-preservation gate's fixture (`random_tensors`) antisymmetrizes `v`, under which the invalid relation is actually true: measured, 0/288 builders disagree on an antisymmetrized fixture versus 41/288 on a spatial one. The gate also never emits C++ at all (it validates Python objects) and skips single-step terms, covering only 27/142 doubles terms.

Design rule:

- A gate protecting a spatial (non-antisymmetrized) code path must use a fixture that is genuinely spatial, not one that happens to share the antisymmetric case's numerical answer. Ship an explicit vacuity control asserting the fixture is spatial and not antisymmetric, as `test_emitted_builder_matches_spec.py` now does.
- Validate the actual emitted C++ text, not just the Python objects that produced it — the defect existed entirely in how `_map_eri_tensor` rendered the table into text.

### 3. A census correlation is not causal until directly tested

Two operator censuses looked decisive during debugging and were each refuted. The first: the defect appeared to track operators spanning more than one distinct amplitude kind (singles 0, doubles 15, triples 91) — filtering all 106 out changed rank 2 by nothing. The second: it appeared to track operators read through more than one index binding, a perfect correlation across three manifolds — direct evaluation of all 616 terms refuted this too.

Design rule:

- Treat a census correlation as a lead, not a conclusion. Confirm causation by direct numeric evaluation (as in the five-step elimination below) before trusting it, especially on a third such correlation.

### 4. The two dressing routes must run at different points in the pipeline, by design

`recognized` dresses before spin-adaptation, because its hand-seeded specs declare GCC layouts that `adapt_intermediate_spec` must then transform. `derived` factorizes after, because it derives operators from whatever manifold reaches it, so its specs are already in the adapted layout.

Design rule:

- Do not move `derived` earlier in the pipeline to "unify" the two routes — it would declare one layout and build another.

### 5. `choose_determinant_backstop` binds the hand-written tensor path only

The `nso > 16 || ndet > 10000` limit recorded across several ccgen scopes does not constrain generated-route test cases. `PLANCK_RCCSDT_BACKEND=optimized` routes through `rccgen.cpp` to the arbitrary-order harness, which never consults the backstop — so small systems like LiH/STO-3G (nso=12, ndet=495) can exercise the generated route in 5 s, unlike the hand-written path.

Design rule:

- When sizing a new ladder point or test case for the generated route specifically, the `nso`/`ndet` backstop limit does not apply; size for wall-clock budget instead.

## What was fixed

1. **The wiring itself.** One dressing axis with a value, not a second boolean: `--dressing {none,recognized,derived}` (default `none`), `--dress-operators` kept as a deprecated alias to `recognized`, `PLANCK_CC_DRESSING={recognized,derived}` on the CMake side. Rejected the alternative (`--derive-operators` beside `--dress-operators`) on evidence in the tree: `print_cpp_planck` already carries 16 branches and `dress_operators` interacts at three separate points, and `generate.py:1060` records that a second emit call site had already forced UCC to be wired twice.
2. **The emitter seam.** `emit_factorized_from_equations` called the emitter itself, so it could not feed `print_cpp_planck`'s single downstream emit. Split into `factorize_equations(eqs, ...) -> (rewritten_eqs, kept_specs)` — the same `(eqs, intermediates)` pair the recognition route already threads — making `derived` a branch rather than a fork. The old entry became a thin, byte-identical delegate.
3. **The invalid ERI symmetry table.** `SPATIAL_ERI_SYMMETRIES` and `ANTISYMMETRIZED_ERI_SYMMETRIES` now live once in `ccgen/tensors.py` (a leaf both consumers import). Both `restricted_closed_shell.py` and the retired `recognized` route (which shared the same bad table and had no builder-level gate) now bind to the correct spatial set.
4. **Deleted the redundant emitter.** `emit_factorized_translation_unit` removed (-45 lines): it had no production caller (25 references, all tests), so "two emitters" was already one emitter plus dead weight. The generate-then-emit convenience moved into `test_factorize.py`, its only consumer.
5. **`print_cpp_planck` gained exactly one parameter** (`dressing`), none of the factorizer's seven selection knobs — deliberately, per the condition set for doing this merge after W4/W5 rather than before.

## Energies before and after the fix

| system | undressed | derivation-dressed (broken) | Δ |
|---|---|---|---|
| CH4/STO-3G | −39.8058445098 | −39.8058606381 | **−1.61e-05** |
| LiH/STO-3G | −7.8823242576 | −7.8823350582 | **−1.08e-05** |

161x the tolerance, on two independent systems, both converging cleanly (`rms(res)` 8.7e-11) — a converged-but-wrong answer, the same signature as the `SPIN_ADAPT` defect and the earlier 52% recognition defect.

| system | after the fix | Δ vs undressed |
|---|---|---|
| CH4 | −39.8058445096 | **2e-10** |
| LiH | −7.8823242576 | **exact, ten digits** |

CH4 also converges in 15 steps against 26 — the wrong fixed point took longer to reach.

## How it was found: five eliminations

Each step removed a layer, cheapest-first. Two of them refuted census-based hypotheses that looked decisive (see invariant 3 above).

| step | result |
|---|---|
| **D1** algebra | rewritten manifold SUM vs unrewritten: spatial doubles exactly 0.0, triples 1.4e-15. Clean. |
| **D2** per-term emit | one term rebuilt from its emitted loop: 3.6e-16. Clean. The `canonical_fock` term-count gap (148 vs 142) is a red herring — `max\|f_ov\| = 7.8e-17`. |
| **D3** operator reuse | 616 rewritten terms through the shared-operator path: worst 2.5e-16. Clean. |
| **D4** emitted text | interpreting the emitted C++ in Python reproduced the disagreement (5.06e-05 vs C++ 5.99e-05). Defect is in the emitter's rendering. |
| **D5** the table | one patched constant: 41/288 -> 0/288. |

## What now guards it

| gate | pins |
|---|---|
| `test_emitted_builder_matches_spec.py` | every `build_W_*` computes its own spec, by evaluating the emitted C++ text; ships a vacuity control asserting the fixture is spatial and not antisymmetric |
| `test_eri_symmetry_tables.py` | the relations verified on a real tensor; the odd ones verified FALSE on a spatial integral; no module redefines a signed table (matched on shape, so renaming does not evade); both consumers bind the shared object; `dressing.py`'s unsigned sets agree |
| `ch4_rccsdt_generated_sto3g`, `lih_rccsdt_generated_sto3g` | the generated route end to end, both requiring `PLANCK_CC_SPIN_ADAPT` |

Writing the guard found a third copy of the table shape — `dressing.py`'s own `_ERI_PERMUTATIONS` pair. Those are sign-free permutation sets used for canonicalization with parity computed separately, so they are deliberately not merged (different shape; merging would be a false unification). What they must share — which permutations belong to which basis — is what the gate checks.

## What was measured: first wall-clock numbers for the derivation route

Everything previously claimed for this route (2.0x-7.1x) was a FLOP model. Measured, same input, same binary configuration apart from `--dressing`:

| system | no/nv | undressed | derivation-dressed | speedup |
|---|---|---|---|---|
| LiH/STO-3G | 4/8 | 5.12 s | **1.64 s** | **3.12x** |
| CH4/STO-3G | 5/4 | 104.56 s | **28.94 s** | **3.61x** |

Medians of 3 and 2 runs; spreads 0.03-0.10 s (LiH) and 0.3-0.4 s (CH4). Energies identical to all printed digits on both, and CH4 takes 15 steps either way — so this is per-iteration work, not fewer iterations.

Both land inside the modelled range, worth stating plainly because `CCGEN_KERNEL_SCALING_SCOPE.md` gave good reason to expect otherwise: it measured the generated-vs-hand-written gap as a scaling defect that no cost model predicts. The model survived contact here; two points is not enough to say it survives generally, and the ratio does grow between them (3.12 -> 3.61).

## Validation strategy that should remain in place

- `test_emitted_builder_matches_spec.py` and `test_eri_symmetry_tables.py`, both evaluating actual emitted C++ text and both carrying their vacuity/non-antisymmetric fixture controls
- `ch4_rccsdt_generated_sto3g` and `lih_rccsdt_generated_sto3g` end-to-end regression cases, requiring `PLANCK_CC_SPIN_ADAPT`
- Cross-checking that the generated undressed kernel reproduces the hand-written baseline (currently 3e-10 on CH4), which is what makes any dressing disagreement attributable

## Related but separate outcome: pre-existing unrelated gate failures

The six failing selection-model gates (`test_savings_concentration` and five others) are pre-existing and unrelated to this work — baselined on a clean worktree both before and after, so they should not be mistaken for regressions introduced here.

## Remaining architecture concern

- **`merge_transposes` is not threaded** on the production path, so `derived` emits the un-merged 59 builders on spatial `ccsd` rather than 31. Scoped in `docs/CCGEN_MERGE_TRANSPOSES.md`, which also corrects a reading this document invited: the 1.4x -> 2.1x -> 3.7x figures are an operator-count reduction, while the modelled FLOP saving is only 1.02x-1.20x. The likely win is compile time, not speed, and should be measured before wiring.
- **The scaling ladder has not been re-run under dressing.** `CCGEN_KERNEL_SCALING_SCOPE.md` attributes the generated-vs-hand gap to H3 (n-ary contraction order) and recommends consuming `_optimal_contraction_order`. Dressing addresses the same hypothesis by a different mechanism, so the two fixes may overlap; the six-point ladder should be re-run with `--dressing derived` before the emitter change is attempted.
- **UCC is out of scope.** The emitter rejects spin-blocked manifold names (`Unknown manifold 'singles_aa'`), and recognition finds zero operators there. Needs O6 in `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` first.
