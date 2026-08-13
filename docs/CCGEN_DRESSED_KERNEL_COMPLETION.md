# Dressed CC kernels — completion record (V1.0 → V1.3, D0–D2)

Canonical record for the **landed** dressed-operator work: what is done, what validates it, and
what remains. Supersedes the per-step scope documents as the source of truth for "what is the
state" — those remain for design history and for the reasoning behind specific choices.

Last updated: 2026-08-13.

**One-line status:** dressed CC kernels generate, compile, link, and run, reproducing the
undressed correlation energy *and* iteration count at rank 3. Two items remain open (V1.3.2,
V1.3.5), neither blocking the validated path.

---

## What works, end to end

```
ccgen residual (diagram engine, canonical Fock)
   → dressed-operator recognition        Wmnij / Wabef / Wmbej / tau / tau_c
   → spin adaptation (RCC spatial)       + adapted intermediate specs
   → emit                                build_<op> + residual kernel
   → CMake -DPLANCK_CC_DRESS_OPERATORS=ON
   → compiles, links, RUNS = undressed
```

Reachable from the build. Default **OFF**, so the default build is byte-identical.

### The validating measurement

`h2` / `lih` / `bh3` RCCSDT-STO-3G, dressed build vs undressed reference:

| case | E_corr (both) | iterations |
|---|---|---|
| `h2` | −0.0205682660 | 12 vs 12 |
| `lih` | −0.0203779358 | 16 vs 16 |
| `bh3` | −0.0533629199 | 26 vs 26 |

Iteration count is part of the gate, not decoration: equal energy shows a shared fixed point,
equal iteration count shows the same *trajectory* — which is what "dressing is a pure
refactorization of the residual" actually claims. Driver:
`tests/dressed_kernel_equivalence.py`.

---

## Landed work, by area

### V1.0–V1.1 — adapt the dressed specs (all of a–f)

| step | what landed |
|---|---|
| V1.1a–d | adapt `definition_terms`, re-derive layout, block-key spec identity, recount usage |
| V1.1e | the faithfulness gate — **passing**, ~1e-14 numeric |
| V1.1f | `validate_intermediate_specs` — sig/rank/slots/definition-order + list-level checks |

V1.1e took four sub-steps and is the most instructive part of the effort:

- **e.0** — a genuine lost coefficient in `assemble_dressed_equation`.
- **e.2** — a **latent, pre-existing** orientation sensitivity in `ucc_integrate_term_antisym`:
  two writings of one integral (`v(k,b,c,j)` vs `−v(j,c,k,b)`, equal under `_eri_canonical`)
  integrated to 2 and 0. Fixed by canonicalizing `v` to one 8-fold orbit member inside
  `_antisym_to_allowed`. Side effect: the spatial emit *shrank* 73260 → 65431 bytes, merging
  orientation-duplicate terms.
- **e.2.5** — the residual `doubles = 14` was **not a defect**. The real bug was in
  `residual_eval.random_tensors`, which violated `<pq||rs> = <rs||pq>` (residual 2.35 vs ~1e-16
  for real integrals). With a symmetry-correct fixture the manifolds agree to ~1e-14.
- **e.3** — per-operator localization, so a regression names one operator.

### V1.2 — wire the composition into `print_cpp_planck`

Early return removed; one emit call site. Six pinned flag-matrix baselines held byte-for-byte.

Two defects surfaced, both caught by gates rather than review:

- **V1.2.2** — under `spin_adapt` the dressed specs were emitted in **GCC** form beside a
  spin-adapted residual; three of five layouts disagreed (`tau` `vvoo`→`oovv`, `tau_c` likewise,
  `Wmbej` `ovvo`→`oovv`). A live miscompile. V1.1f's validator is now wired in as an assertion at
  that exact point.
- **V1.2.4** — removing the early return **activated** `factorize_tau` under dressing. The
  "already mutually exclusive" claim in the parent scope was *unreachability*, not a guard. Now an
  explicit `ValueError`; CSE is forced off, mirroring the `spin_adapt` precedent.

### V1.3 — link and run

- **V1.3.0** — the co-inclusion asymmetry pinned. Non-arbitrary TUs co-include cleanly (differing
  amplitude types make the builders **overloads**); under `force_arbitrary` all take
  `ArbitraryOrderRCCAmplitudes`, so they are **redefinitions** — 5 errors. Conditional on exactly
  the registry's mode, and invisible in the mode probed first.
- **V1.3.1** — `--dress-operators` on the build generator + `PLANCK_CC_DRESS_OPERATORS` in CMake.
  Dressing is suppressed on the arbitrary-order companions for the collision reason above.
- **V1.3-emit** — the dressed TU had **never** been valid C++: `build_Wmnij`/`build_Wabef`
  referenced `tau(...)` with nothing declaring it, because `sibling_names` only controlled
  *rendering*. Fixed by binding referenced siblings the way `_emit_kernel` already did.
- **V1.3.3/V1.3.4** — links (6 builder symbols in the binary vs 1) and runs, per the table above.
- **V1.4** — builder order is a valid topological sort **of the emitted TU**: `tau`, `tau_c`,
  `Wmnij`, `Wabef`, `Wmbej`, zero forward references, across all three dressed configurations.

### D0–D2 — dressing's super-linear cost

`hypothesis_is_consistent` rebuilt `raw_multiset(residual_terms)` on every call — 7,461 times on
`ccsdt` triples, over an input that never changes. `n_hypotheses × n_terms` was the whole
super-linear term.

| | before | after |
|---|---|---|
| triples (399 terms) | 94.7 s | **6.9 s** |
| `raw_multiset` calls | ~7461 | **19** |
| rank-3 end-to-end | 293.7 s | **9.1 s** |
| rank-4 end-to-end | >25 min, abandoned | **61.6 s** |

The flat call count is the fix; timing follows. **D3 (pruning the hypothesis search) is not
needed** — one factor was the whole story, so a correctness-affecting change was avoided.

---

## What remains open

| item | status | blocks? |
|---|---|---|
| **V1.3.5** — pin the dressed config in `regression_cases.json` | not started (~S) | Nothing. Without it the V1.3.4 result can rot. |

**V1.3.2 is decided and landed** (route b): `_builder_symbol` names every builder
`build_<name>_<method>`, so two dressed TUs co-include cleanly — measured `rc=0`, 0 redefinitions
on the configuration that previously produced 5. Chosen over restricting dressing to one rank,
because the collision is a property of the naming scheme rather than of how many ranks are
enabled; a restriction would have left the trap armed for the next person to enable one.

**Rank 4 is therefore unblocked on both counts** — cost (61.6 s post-D1) and naming. The anchor
stays rank 3 only because that is where the validated end-to-end run is, not because rank 4 is
prevented.

---

## Gates that guard this

| gate | what it pins |
|---|---|
| `test_residual_symmetry.py` | V1.1e numeric faithfulness + the fixture's own symmetries |
| `test_dress_per_operator.py` | per-operator localization; F operators inert under canonical Fock |
| `test_intermediate_validity.py` | V1.1f spec validity, 9 negative cases |
| `test_emit_flag_matrix.py` | 6 flag-matrix baselines (hash + length), exclusions, dressed compile, builder order |
| `test_dressed_tu_coinclusion.py` | the co-inclusion asymmetry, both halves |
| `test_dressing_scaling.py` | `raw_multiset` call count is bounded, not per-hypothesis |
| `tests/dressed_kernel_equivalence.py` | dressed == undressed energy **and** iteration count |

Full ccgen suite: **759 tests OK**, 4 pre-existing expected failures, ~1021 s (was ~1390 s
pre-D1).

**Run numeric gates through `tests/pyscf/.venv/bin/python`** (pyscf 2.13.0). In the default
interpreter every pyscf gate reports `skipped` — a green run there is not evidence.

---

## Lessons that generalize

Recorded because each cost real time and each recurred:

1. **Gate on numeric residual values, not symbolic term counts.** A term multiset cannot
   distinguish *different algebra* from *the same algebra written in a symmetry-equivalent form*.
   That distinction produced the phantom `doubles = 14` and consumed five sub-steps.
2. **A numeric gate is only as good as its fixture's symmetry.** `random_tensors` violated
   `<pq||rs> = <rs||pq>` for a long time because nothing compared two exchange-related writings
   until dressing did. Assert the fixture's invariants, not just the result.
3. **Cumulative profile time is not a fix target.** `_eri_canonical` showed the largest number in
   the profile (864 s cumulative, 3 M calls); memoizing it bought 6 %. The real cause was a
   redundant *outer loop* no inner-function work would have reached.
4. **Prefer a call-count gate to a wall-clock one.** Deterministic, and it names the defect when
   it regresses.
5. **Each new *kind* of gate found a defect the previous kinds could not see** — layout (V1.2.2),
   flag interaction (V1.2.4), "does it even compile" (V1.3-emit). The gate I listed and skipped
   is the one that found the oldest defect. Run the gate you wrote down.
6. **"Already mutually exclusive" deserves a check.** V1.2.4's exclusion was unreachability, not
   a guard, and removing an early return silently activated the excluded flag.
7. **Never write spaghetti: when a defect lives in one mechanism, fix that mechanism.** Not a
   per-caller patch, not a boundary pre-pass, not a scope restriction that dodges it. Decided
   twice here against my own cost-based recommendation, and right both times:
   **V1.1e.2** (fix `ucc_integrate_term_antisym` rather than normalize `v` at the dress/adapt
   boundary — the defect turned out to be latent and pre-existing, so any future caller would have
   hit it) and **V1.3.2** (suffix the builder names rather than restrict dressing to one rank — the
   collision is a property of the naming scheme, not of the rank count). The corollary that keeps
   paying: prefer **one mechanism with a parameter** over two parallel paths
   (`adapt_intermediate_spec(adapter=…)`, `external_blocks(fold_spin_flip=…)`,
   `_dress_operator_equations(operators=…)`, `_builder_symbol(method, name)`), and never ship two
   overlapping normalizations of the same thing.

---

## Superseded scope documents

Retained for design history; **not** the source of truth for status:

| document | covers |
|---|---|
| `CCGEN_DRESS_ADAPT_COMPOSITION_SCOPE.md` | V1.0–V1.4 originals (V1.3 corrected in place) |
| `CCGEN_V11_SPEC_ADAPTATION_SCOPE.md` | V1.1a–f, and V1.1e's four sub-steps |
| `CCGEN_V11E2_ORIENTATION_INVARIANCE_SCOPE.md` | e.2's route decision + the wrong turn on the name-independent key |
| `CCGEN_V11E25_RESIDUE_SCOPE.md` | why `doubles = 14` was an artifact, and the fixture bug |
| `CCGEN_V12_EMIT_WIRING_SCOPE.md` | V1.2.0–V1.2.5 |
| `CCGEN_V13_LINK_AND_RUN_SCOPE.md` | V1.3.0–V1.3.5 — **still live for V1.3.2/V1.3.5** |
| `CCGEN_DRESSING_SUPERLINEAR_SCOPE.md` | D0–D4, including the `_eri_canonical` wrong turn |

`CCGEN_V13_LINK_AND_RUN_SCOPE.md` is the one to read before resuming: its V1.3.2 section holds
the naming decision, and its V1.3.5 section the regression-pinning plan.
