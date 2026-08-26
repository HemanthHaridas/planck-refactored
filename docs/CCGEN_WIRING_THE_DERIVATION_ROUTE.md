# How does the derivation route reach production?

**Scope for in-flight work. W1-W2 landed; W4.2a and W4.5 landed 2026-08-26; W3, W4.2b-W4.4 and W5 open.**
**W4 is no longer blocked** — its blocker was `PLANCK_CC_SPIN_ADAPT=OFF`, not a kernel defect
(`docs/CCGEN_SPIN_ADAPT_DEFAULT.md`); the flag now defaults ON and the generated undressed kernel
matches the hand-written baseline to 3e-10. **What blocks the rest of W4 is W3**: `--dressing` is
not implemented, so there is no derivation-dressed TU to compare against. Opened by
`docs/CCGEN_TWO_DRESSING_ROUTES.md`, which established that ccgen's *derivation* route (operators
from each term's contraction tree) is value-preserving at ranks 2-4, worth 2.0x-7.1x, and **has no
production caller** — deferred in its own commit (`f68f7e2`, "CCSD dressing stays D7.3's job") and
never revisited.

It is not a research question: the algebra is gated, the merge is implemented, and the emit bridge
exists. What is missing is that **there are two emitters** and production uses the other one.

**Where it stands.** The technical blocker is gone — the pipeline now accepts adapted equations
(W1) and the spatial TU compiles merged and un-merged, 59 → 31 builders (W2).

W3 was a decision, and it has been taken under the constraint *"the code should not be
spaghetti"*: **one `--dressing {none,recognized,derived}` axis, not a fourth boolean**, with the
emitters merged afterwards as a deletion rather than beforehand as an accumulation. The earlier
"new flag now, merge later" recommendation is withdrawn — see W3 for what the code itself says
about that. W4 (regression energies) and W5 (wall-clock) still gate it.

Retire this file when W1-W5 land; the answer belongs in `CCGEN_TWO_DRESSING_ROUTES.md`.

---

## The actual gap

Two parallel emitters that share exactly one parameter (`method`):

| | production | factorizer |
|---|---|---|
| entry | `print_cpp_planck` (`generate.py`) | `emit_factorized_translation_unit` (`factorize.py`) |
| called by | `generate_planck_cc_kernels.py` | **nothing** (this is W3) |
| dressing | `dress_operators=True` → the **retired** recognition route | derivation, merged |
| spin adaptation | `spin_adapt=`, `ucc=` | **none** |
| intermediates | `include_intermediates=`, three memory-budget knobs | `top_k` / `savings_fraction` / `memory_budget_bytes` |

```
production-only: dress_operators, factorize_tau, force_arbitrary, include_intermediates,
                 intermediate_{memory,peak_memory}_budget_bytes, intermediate_threshold,
                 spin_adapt, ucc
factorizer-only: canonical_fock, engine, factor_builder_bodies, max_operator_bytes,
                 memory_budget_bytes, merge_transposes, n_occ, n_vir, savings_fraction,
                 spatial, top_k
```

### The structural blocker, and the good news

**Removed by W1.** `emit_factorized_translation_unit` used to call
`generate_cc_equations(method, ...)` internally, so it could not be handed an already-adapted
manifold and could never emit spatial or UCC kernels — which is most of what production emits.
It is now a thin wrapper over `emit_factorized_from_equations`, which takes the equations.

The machinery underneath never had that limit, which is why W1 was a signature change rather than
new algebra. Measured on adapted input:

| input | merged operators | terms rewritten |
|---|---|---|
| spatial `ccsd` doubles | 31 | 64 |
| UCC `ccsd` doubles_abab | 86 | 71 |

Spatial now emits and compiles end to end. **UCC still does not**: the emitter rejects the
spin-blocked manifold names (`ValueError: Unknown manifold 'singles_aa'`). That is emitter work
downstream of O6 and deliberately out of scope here — see *What NOT to do*.

## Steps

Each step keeps `test_factorize_value_preservation` at 0/0 on both bases and both
`canonical_fock` settings. That gate, not the operator count or the TU size, is the acceptance
criterion throughout — a step that improves the numbers and reddens it has failed.

### W1 — LANDED (2026-08-25): the pipeline takes equations; the blocker is gone

`emit_factorized_from_equations(method, eqs, **selection)` is the pipeline body;
`emit_factorized_translation_unit` is now a thin wrapper that generates and delegates.

**Byte-identical**, verified by sha256 over six configurations before and after the split
(`ccsd`, `ccsd top_k=8`, `ccsd merge_transposes`, `ccsdt`, `ccsdt top_k=5`,
`ccsdt memory_budget`). No behaviour change.

**The blocker is lifted.** Handed `spin_adapt_equations(...)` output the new entry emits a
spatial TU: 82,793 chars, **31 builders**, braces balanced, each builder defined exactly once.
Previously impossible — the wrapper called `generate_cc_equations` internally and could only ever
emit GCC.

**UCC still fails, and deliberately so:** `ValueError: Unknown manifold 'singles_aa'`. The
emitter's manifold vocabulary does not include the spin-blocked names. That is emitter work
downstream of O6 (recognition finds zero operators on spin-tagged factors, and the tag-blind fix
is unsound), so it is out of scope here rather than patched.

*Gates:* `test_emit_from_equations_is_byte_identical` (wrapper == direct, four configurations)
and `test_emit_from_equations_accepts_a_spin_adapted_manifold`.

*Falsifiability:* dropping a kwarg from the wrapper fails 1 test; making the pipeline ignore the
caller's `eqs` fails 5.

Full suite after: 101 tests, the same 6 selection-model failures. No new breakage.

### W2 — LANDED (2026-08-26): the spatial TU compiles, merged and un-merged

Both halves answered, and both **actually run** here rather than skipping — Eigen and a C++23
compiler are present in this tree.

**It compiles.** `-fsyntax-only` against the real CC headers, exit 0 both ways:

| | builders | compiles |
|---|---|---|
| spatial, un-merged | 59 | yes |
| spatial, **merged** | **31** | yes |

Merging nearly halves the generated builder count and the TU still type-checks — so the merged
call sites really do resolve against the representative's `build_W`, not just in the algebra.
This is the check neither the value gate (symbolic, never compiles) nor W1's emit check (counted
builders and braces) could make.

**The ERI parity contract holds.** `_ERI_SYMMETRY_PERMUTATIONS` is exactly the four parity-`+1`
relations, each carrying an explicit `+1` sign, and the set is identical to
`_ERI_PERMUTATIONS_SPATIAL`. Worth re-checking rather than assuming, because the merge now runs
inside the emit path: this is the contract whose violation let a 52 % energy defect pass every
symbolic check, and the same one that produced two false operator merges in O1.

*Gates:* `test_spatial_factorized_tu_compiles` (both variants, skips without a toolchain) and
`test_emitter_folds_only_sign_preserving_eri_symmetries`.

*Falsifiability:* adding a `-1` permutation to the emitter's set fails the parity gate.

Full suite: 126 tests, the same 6 selection-model failures. No new breakage.

### W3 — RESCOPED (2026-08-26) under "the code should not be spaghetti"

The earlier framing offered three options and recommended "new flag now, merge later". **That
recommendation is withdrawn.** The constraint rules it out, and the codebase already contains the
evidence for why.

#### What the code says about adding a fourth flag

`print_cpp_planck` carries **16 branches**, and `dress_operators` interacts at three separate
points:

| where | interaction |
|---|---|
| `generate.py:1052` | mutually exclusive with `factorize_tau` — raises |
| `generate.py:1064` | overrides caller `engine` / `canonical_fock` kwargs |
| `generate.py:1152` | silently forces `include_intermediates=False` |

and the CLI carries a further comment reconciling the same pair. A `--derive-operators` flag adds
a **fourth** dressing-ish axis to a function that already needs prose to explain the three it has.
Every pairwise combination becomes a question someone has to answer, and most of them are
meaningless.

The decisive line is a comment already in the tree at `generate.py:1060`:

> `spin_adapt / factorize_tau / force_arbitrary silently unreachable under dressing; composing
> them is the point of V1.2, and **a second emit call site would fork the composition so V5 (UCC)
> had to be wired twice**.`

That is this exact mistake, already made once and already paid for. A parallel
`--derive-operators` path is a second emit call site by construction.

#### The shape that is not spaghetti

**One dressing axis with a value, not two booleans.**

```
--dressing {none,recognized,derived}      default: none
```

- `none` — today's undressed emit, byte-identical.
- `recognized` — what `--dress-operators` means now. Retired, kept only so old invocations
  reproduce; can print a deprecation line.
- `derived` — the factorizer route.

One parameter, three values, mutually exclusive by construction. The interactions above stay
exactly as many as they are today because there is still one dressing axis — `derived` inherits
`recognized`'s answers to all three (diagram engine, canonical Fock, no CSE) since they are
properties of *dressing*, not of which route derived the operators.

Internally this is one call site, not two: `print_cpp_planck` chooses which operator set to hand
the emitter and keeps a single composition path. The factorizer already exposes the right seam —
`emit_factorized_from_equations` takes equations (W1), so it can be fed the same adapted manifold
`print_cpp_planck` already builds.

`--dress-operators` becomes an alias for `--dressing recognized`, so nothing that exists today
changes meaning. That is what the earlier "repoint the flag" option got wrong: silently changing
what an existing flag emits would make an old command line reproduce different kernels.

#### Why not "merge the emitters" as a first step

It is the right end state and the wrong first move. Merging means `print_cpp_planck` absorbs the
factorizer's selection knobs (`top_k`, `savings_fraction`, `memory_budget_bytes`,
`max_operator_bytes`, `merge_transposes`, `n_occ`, `n_vir`) — seven more parameters on a
16-branch function, landed **before** W4 has shown the route produces correct energies.

Do it after W4/W5, when the route has earned it, and do it as a deletion: `print_cpp_planck` calls
`emit_factorized_from_equations` and `emit_factorized_translation_unit` goes away. One emitter,
fewer parameters than the sum of today's two.

#### Steps

- **W3.1** — add the `--dressing` enum with `--dress-operators` as an alias. No new emit path yet;
  `derived` raises "not yet wired". *Verify:* every existing invocation byte-identical, including
  `--dress-operators`.
- **W3.2** — implement `derived` inside `print_cpp_planck`'s existing dressing branch, feeding the
  adapted manifold to `emit_factorized_from_equations`. *Verify:* value gate 0/0; spatial TU
  compiles (W2's gate, now through the production entry); the three interaction points behave as
  they do for `recognized`.
- **W3.3** — *after W4/W5 pass:* delete `emit_factorized_translation_unit` and the recognition
  emit path, leaving one emitter. *Verify:* net negative diff, and no parameter added to
  `print_cpp_planck` that a caller does not use.

**W3.3 is the step that keeps this honest.** W3.1 alone is "new flag with no follow-through",
which is exactly how the derivation route was orphaned the first time. If W3.3 is not going to
happen, do not start W3.1.

### W4 — do derivation-emitted kernels compute the right energies? (~M, five steps)

**This is the step that can still find a correctness defect.** Everything before it is symbolic
or syntactic: the value gate evaluates a rewrite and never compiles, W2 compiles but never runs.
Nothing so far has executed a generated kernel and compared an energy.

Both prior CC defects on this branch were invisible to exactly the gates that precede W4 — the
rank-3 kernel was correct while its *solver* was wrong, and the 52 % dressed defect passed five
structural gates. Treat a W4 disagreement as outranking every performance claim in this document.

#### The trap this step must not fall into

Of the ten CC regression cases, **nine never reach a generated kernel.**
`choose_determinant_backstop` (`tensor_backend.cpp:243`) routes `nso <= 16 && ndet <= 10000` to
the determinant-space teaching backstop, which calls no generated code at all. Computed for
STO-3G:

| case | nso | ndet | path |
|---|---|---|---|
| `h2_rccsdt` | 4 | 6 | determinant |
| `be_rccsdtq` | 10 | 210 | determinant |
| `lih_rccsdt` | 12 | 495 | determinant |
| `water_rccsdt` | 14 | 1001 | determinant — and it *asserts* the handoff |
| **`ch4_rccsdt`** | **18** | **43758** | **tensor (generated)** |

So a W4 that runs "the CC cases" and reports green has, for nine of ten, proven nothing about the
derivation route. `ch4_rccsdt_sto3g` is the only in-tree case that exercises it — the same fact
that left the hand-written tensor solver ungated for its entire life.

**Narrowed 2026-08-26 (W4.5):** this constraint binds the **hand-written tensor** path, not the
generated one. `choose_determinant_backstop` is consulted inside `run_tensor_rccsdt`; the
`optimized` backend routes through `rccgen.cpp` to the arbitrary-order harness, which never calls
it. So a small case CAN exercise the generated route — LiH/STO-3G (`nso=12`, `ndet=495`) does,
matching hand-written to ten digits. The table above remains correct for the default routing.

#### Steps

##### W4.1 — DONE (2026-08-26): baseline pinned, and W4 has a build blocker

*(The "blocker" below — needing `-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` — still holds. The
separate `PLANCK_CC_SPIN_ADAPT` blocker found later is resolved; see W4.2a.)*

Baseline for `ch4_rccsdt_sto3g`, the only case that reaches the tensor path:

```
RHF   Total Energy   -39.7267328271
CCSDT Energy         -39.8058445095      converged in 24 steps, 0.18 s
```

Run as `python tests/run_regressions.py --suite extended --case ch4_rccsdt_sto3g` → PASS in
1.00 s. The `--suite extended` is required; a bare `--case` prints only "no cases selected".

**The blocker: this case does not use generated kernels today, and cannot in the current build.**

Its own assertion says so — the case pins the string `kernels=hand-optimized` — and the run
confirms it: `Stage-1 RCCSD warm start dimensions: nocc=10 nvirt=8 (kernels=hand-optimized)`,
with every iteration logging `kernel=native`. It reaches the **tensor backend** (correctly, being
above the determinant backstop) but the *hand-written* one.

Forcing the generated path fails with an actionable error:

```
$ PLANCK_RCCSDT_BACKEND=optimized ./build/hartree-fock ...ch4_rccsdt_sto3g.hfinp
[ERR] run_tensor_optimized_rccsdt: the generated rank-3 CCSDT kernel runs only in the
      arbitrary-order harness, which needs -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON.
```

`build/CMakeCache.txt` confirms `PLANCK_CC_ARBITRARY_LOWER_RANKS:BOOL=OFF` (as are
`PLANCK_CC_UCC` and `PLANCK_CC_DRESS_OPERATORS`).

**Consequence for W4.** The comparison is not "current kernels vs derivation-emitted kernels" on
one binary. It needs a **reconfigured build** with `-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON`, and
the comparison is then three-way:

| path | reachable today |
|---|---|
| hand-written tensor (`kernels=hand-optimized`) | yes — this is the baseline above |
| generated, undressed (arbitrary-order harness) | needs the reconfigure |
| generated, **derivation-dressed** (W3.2) | needs the reconfigure **and** W3.2 |

So W4.2's rebuild is not optional plumbing — it is the step that makes W4 possible at all, and it
must set that option. Worth noting the baseline is the *hand-written* path: a W4 disagreement
could mean the derivation route is wrong, **or** that generated-undressed already differs from
hand-written. Establishing the middle row first separates those two, and is cheaper than
debugging them together.

##### W4.2a — DONE (2026-08-26): the generated undressed kernel MATCHES the baseline

**This section previously read "the generated path does not converge; W4 is BLOCKED". That is
retracted.** The non-convergence was not a kernel defect: the build carried
`PLANCK_CC_SPIN_ADAPT=OFF`, the historical spin-orbital emit that `CMakeLists.txt` itself
documented as making the generated correlation energy ~4x wrong. **That flag now defaults ON**
(2026-08-26). Full answer: `docs/CCGEN_SPIN_ADAPT_DEFAULT.md`.

Re-run with `-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` (`SPIN_ADAPT=ON` now comes from the default),
`PLANCK_RCCSDT_BACKEND=optimized`:

```
[INF] RCCSDT[OPT] : Routing the ccgen-generated rank-3 CCSDT kernels through the
                    arbitrary-order harness (the representation they are emitted for).
[INF] RCCSDT     : Generated RCCSDT iterations ran 35 steps, E_corr=-0.0791116827
       Total RCCSDT Energy   -39.8058445098
```

| path | CCSDT total | vs baseline |
|---|---|---|
| hand-written tensor (W4.1 baseline) | −39.8058445095 | — |
| **generated, undressed** | **−39.8058445098** | **3e-10** |

Well inside the case's 1e-07 tolerance, and the **middle row of W4.1's table is established**: the
generated undressed kernel reproduces the hand-written one, so a later W4.3 disagreement can be
attributed to dressing rather than to the generated kernel or the harness. That separation was the
whole point of splitting W4.2.

The generated path is **positively identified**, not inferred — the run logs the `Routing …`
line, and `kernels=hand-optimized` does not appear. Gated by `ch4_rccsdt_generated_sto3g`, which
now asserts the correct energy (it previously pinned the broken behaviour) and requires **both**
`PLANCK_CC_ARBITRARY_LOWER_RANKS` and `PLANCK_CC_SPIN_ADAPT`, so it can never again run under the
defective emit.

Two further systems confirm the generated rank-3 kernel independently, both matching hand-written
to all ten digits: Be −0.0517702884 and LiH −0.0204594700.

##### W4.2 — split by W4.1 into two rebuilds, because there are two questions

**W4.2a — reconfigure with `-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` and rerun, undressed.**

This establishes the *middle* row of W4.1's table: does the generated, undressed kernel reproduce
the hand-written one? It needs nothing from W3, so it can be done now, and it is the cheaper
half.

*Verify:* `ch4_rccsdt_sto3g` under `PLANCK_RCCSDT_BACKEND=optimized` matches the W4.1 baseline
(CCSDT `-39.8058445095`) to the case's 1e-07 tolerance — and the run logs the generated path
rather than `kernels=hand-optimized`, positively identified, not inferred.

A disagreement here is a finding about the **generated kernel or the arbitrary-order harness**,
not about dressing, and it stops the ladder — the derivation route cannot be judged against a
baseline that already disagrees.

**W4.2b — rebuild again with W3.2's `--dressing derived`.**

*Verify:* the build succeeds, and the emitted TU **actually changed** — diff it, do not assume the
flag took effect. A silently-ignored flag makes W4.3 pass vacuously, which is the failure mode
this ladder keeps hitting.

Kernels are generated at configure time (`CMakeLists.txt:519`), so each of these is a
reconfigure-and-rebuild, not an incremental compile. Budget accordingly.

##### W4.3 — rerun and compare (~S, the gate)

*Verify:* `ch4_rccsdt_sto3g` energy matches W4.1 to the tolerance the case already asserts. **Any
disagreement is a correctness finding and stops the ladder** — it is not a tolerance to widen.

##### W4.4 — prove the comparison was not vacuous (~S)

W4.3 passing is only meaningful if the run actually used derivation-emitted kernels. Confirm it
did: check the emitted TU contains the merged operator names (`W_..._<shape-tag>`, and builder
count 19 rather than 27 on `ccsd`), and that the case did not silently take the determinant
backstop.

*Verify:* a positive identification of the code path taken, not an inference from a green tick.

##### W4.5 — DONE (2026-08-26): the ladder IS widened, via `optimized` not `tensor`

One case is thin evidence for a production route. This step asked whether the suite can provide a
second, and **it can** — but not by the mechanism this section proposed.

**Measured: `PLANCK_RCCSDT_BACKEND=tensor` does NOT bypass the determinant backstop.**
`choose_determinant_backstop` is called *inside* `run_tensor_rccsdt`
(`tensor_backend.cpp:2996`) off the reference size alone; the env var selects among three
backends (`ccsdt.cpp:22`), not whether the backstop fires. So forcing `tensor` on a small case
still lands in the determinant-space teaching backstop, and still yields nothing for this route.

**But `optimized` does bypass it.** That backend routes to the arbitrary-order harness via
`rccgen.cpp`, which never consults `choose_determinant_backstop` at all. So the `nso > 16 ||
ndet > 10000` constraint — recorded across several ccgen scopes as a hard ladder-design
limit — **does not apply to the generated path**, only to the hand-written tensor one.

Demonstrated end to end on **LiH/STO-3G** (`nso=12`, `ndet=495` — far below the threshold):

| LiH/STO-3G CCSDT | E_corr |
|---|---|
| hand-written | −0.0204594700 |
| **generated, via `optimized`** | **−0.0204594700** — all ten digits |

Input at `tests/inputs/regression/post_hf/lih_rccsdt_generated_sto3g.hfinp`, and it runs at
0.04 s/iteration against CH4's ~7 minutes, so it is also the cheaper development fixture.

**Landed:** `lih_rccsdt_generated_sto3g` is now a regression case — same `env` and
`requires_build_option` pair as the CH4 one, asserting `rccsdt_total_energy == -7.8823242576`
(1e-07) and `rhf_total_energy == -7.8618647876` (1e-08), and positively identifying the generated
path by its routing line. **PASS in 5.3 s**, and verified falsifiable (perturbing the expected
energy fails it).

So the generated route no longer rests on a single case, and the second one runs in seconds rather
than minutes — the fragility this step exists to flag is closed rather than merely recorded.

### W5 — cost, measured rather than modelled (~M)

Everything claimed for this route so far is `operator_savings` / `build_cost`, a FLOP model.
`CCGEN_KERNEL_SCALING_SCOPE.md` measured the generated-vs-hand gap as a **scaling** defect
(21.8x → 50.1x, no plateau) that no current cost model predicts.

*Verify:* wall-clock for a derivation-emitted kernel against the current one on a case that
actually reaches the tensor path — note `choose_determinant_backstop`
(`tensor_backend.cpp:243`) routes `nso <= 16 && ndet <= 10000` to the determinant backstop, which
never calls the generated kernel, so most small cases yield **no timing at all**. `ch4_rccsdt_sto3g`
is the known-good rank-3 point.

Expect the modelled 2-7x not to survive intact. That is worth knowing either way; it is the
first wall-clock number this route will ever have had.

## What NOT to do

- **Do not un-retire the recognition route.** Nothing here touches it. It is 52 % short on Be
  with five failed fix attempts behind it, and its seven `expectedFailure` gates stay as the
  tripwire.
- **Do not trust the value gate as a correctness gate for kernels.** It evaluates the symbolic
  rewrite; it never compiles or runs anything. W2's compile check and W4's energies are separate
  instruments for a reason.
- **Do not wire UCC in this pass.** The machinery runs on UCC input (86 merged operators), but
  recognition finds **zero** operators there and the obvious tag-blind fix is measured and
  unsound — it collapses 12 spin-tagged contractions onto one `Wmbej`. That is O6 in
  `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` and needs its own numeric gate first.
- **Do not re-pin the six failing selection-model gates as part of this.** They need their claims
  restated, not their constants moved, and the merge changes the distribution again. Separate
  work, tracked in `CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` (O4.6).

## Key code locations

| what | where |
|---|---|
| factorizer pipeline (W1) — takes equations | `emit_factorized_from_equations`, `python/ccgen/optimization/factorize.py:993` |
| its generating wrapper, still no caller | `emit_factorized_translation_unit`, same file `:948` |
| ~~internal generate call — the blocker~~ | removed by W1; the wrapper now delegates |
| production emitter | `print_cpp_planck`, `python/ccgen/generate.py:1023` |
| the CLI that chooses | `python/generate_planck_cc_kernels.py` (`--dress-operators`, `:114-147`) |
| merged operators + call-site plan | `manifold_operators_with_plan`, `factorize.py` |
| **the value gate** | `python/ccgen/tests/test_factorize_value_preservation.py` |
| spatial ERI symmetry contract | `_ERI_SYMMETRY_PERMUTATIONS`, `python/ccgen/emit/planck_tensor_cpp.py` |
| why this route, and what it is worth | `docs/CCGEN_TWO_DRESSING_ROUTES.md` |
| operator granularity and the merge | `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md` |
| unmodelled cost | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
