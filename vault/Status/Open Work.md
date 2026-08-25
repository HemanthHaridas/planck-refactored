---
name: Open Work
description: Canonical summary of known gaps, risks, and follow-up work in planck-refactored
type: status
priority: high
include_in_claude: true
tags: [status, open-work, canonical, roadmap]
---

# Open Work

Last updated: 2026-06-04

This is the canonical open-work document for the repository.
Use it with `vault/Status/Completion.md`. Older status snapshots and handoff
notes may still exist for design history, but they are no longer the source of
truth for what remains.

## Highest-priority correctness and robustness work

- (none currently — the ROHF MO-energy bookkeeping inconsistency is resolved;
  see Completion)

## Verification and regression gaps

- Strengthen the end-to-end spherical full-symmetry direct-SCF regression ladder beyond the current focused infrastructure tests and committed NH3/CH4 ladder
- Add durable regression coverage for remaining full-symmetry edge cases called out in the design notes:
  D3h, Oh, linear-group interplay, and lone-atom behavior
- Revalidate the CASSCF/PySCF gate suite after future optimizer work; the current tree matches the documented state, but the 11/11 suite was not freshly rerun during the May 25 consolidation review
- Keep documentation comments aligned with the implemented spherical symmetry representation; stale comments have already drifted once

## docs/ hygiene — two ccgen scope docs still owe an architecture rewrite

A file in `docs/` answers one architecture question or is a teaching guide; scoping **in-flight**
work is the only exception, and it expires when that work lands.

**Three of the original five are done (2026-08-16).** The CCSDTQ trio collapsed into one answer, as
predicted — they were split by *effort* and merged once regrouped by *question*:

| retired | into |
|---|---|
| `CCGEN_R3_HIGHER_RANK_BRIDGE_SCOPE.md` (295) | `docs/CCGEN_CCSDTQ_MULTISECTOR.md` |
| `CCGEN_KERNEL_WIRING_MULTISECTOR_SCOPE.md` (225) | same |
| `CCGEN_CCSDTQ_FCI_VERIFICATION_SCOPE.md` (128) | same |
| `CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md` (181) | `docs/CCGEN_TENSOR_ACCESSOR.md` |

All three CCSDTQ docs carried **stale headers contradicting their own content** — the bridge doc
advertised a rank-8 `xfail` that no longer exists in the code, the verification doc kept a "Why it
is still RED" section under a GREEN status line, and the wiring doc claimed "two gaps, both open"
when both were closed. Verified before rewriting: 12 bridge tests pass, the Be CCSDTQ==FCI oracle
passes (12m01s), and `be_rccsdtq_sto3g` passes end-to-end. **Do not trust a status header without
running its gate** — four such headers were found false in one session.

Remaining, deliberately deferred until the UCC work (U1–U5) lands:

| doc | lines | landed work |
|---|---|---|
| `CCGEN_SPIN_ADAPTATION_SCOPE.md` | 892 | S0–S4 |
| `CCGEN_KERNEL_WIRING_AND_BENCHMARK_SCOPE.md` | 331 | kernel wiring + benchmarks |

**Four became due 2026-08-22.** The UCC numeric ladder completed (F1/F2/F3), and then U1 completed
on top of it. `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md`, `CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE.md` and
`CCGEN_U1_UCC_ADAPT_SCOPE.md` are all finished work, so the scoping exemption has expired for each.
(`CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` stays exempt — U2–U5 are genuinely in flight.
The rank-6 gap closed, so that handoff has become
`CCGEN_UCC_NUMERIC_VALIDATION.md` — the target the three older UCC scope docs should merge into.) They should merge into **one** answer — they are two halves of a single question, split by
effort exactly as the CCSDTQ trio was:

- **`CCGEN_UCC_NUMERIC_VALIDATION.md`** — how do you check that a spin-block CC residual is right?
  The U1 doc merges in here: its answer is the same question one layer up (how do you know the
  GCC→UCC *adaptation* is right), and its four PySCF-interface defects belong with the fixture
  lessons rather than in a step ladder.

Keep, because each cost real investigation: the per-target-pairing correction to the closed-shell
oracle; the closure relations and why F1's fixture cannot serve it; the `(occ…,vir…)` vs
`(vir…,occ…)` transpose; the `f_ov`-on-both-sides measurement table with the falsified first
hypothesis; and both vacuous-pass traps (converged amplitudes, OH/STO-3G). Drop the F-numbering, the
per-step *Verify:* lines, and the three-option F2.0 design table now that A is built and proven.
Deferring only until U1.2 has consumed the evaluator, in case that surfaces more.

`CCGEN_SPIN_ADAPTATION_SCOPE.md` is the reference U1 works against, so rewriting it now risks
discarding scope still load-bearing for unstarted work. Target questions:

- **`CCGEN_SPIN_ADAPTATION.md`** — how does a spin-orbital CC equation become a spatial one?
  (the rank-4 multi-sector half of this is already answered in `CCGEN_CCSDTQ_MULTISECTOR.md`)
- **`CCGEN_KERNEL_WIRING.md`** — how does a generated kernel reach a runnable binary, and what does
  it cost?

When doing it: read `CCGEN_SPIN_ADAPTATION_SCOPE.md` in full first, and move any still-live UCC
scope into `CCGEN_U1_UCC_ADAPT_SCOPE.md` rather than dropping it. Keep the measured numbers, the
ruled-out hypotheses and the wrong turns — they are part of each answer. Drop step numbering,
gates-to-write and sequencing diagrams.

Judged compliant in the same audit, for the record: `CCGEN_TEACHING_GUIDE`, `CCGEN_REPORT`,
`CCGEN_GENERATION_AND_VALIDATION` (teaching/report); `CCGEN_HIGHER_OPERATOR_REUSE`,
`CCGEN_DIAGRAM_REPRESENTATION_SCOPE`, `CCGEN_INTERMEDIATE_MEMORY_LOCALITY_SCOPE` (already
question-shaped, work unstarted); `CCGEN_ARBITRARY_ORDER_UCC_SCOPE`, `CCGEN_U1_UCC_ADAPT_SCOPE`
(genuine in-flight scope).

`CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE` was in that list and has been **deleted** (2026-08-16): it
scoped V2–V6 for the dressed route, which is **retired** (see Completion — dressing and spin
adaptation do not compose, 52 % short on Be). The doc never acknowledged the retirement, so it read
as live scope inviting work the project has decided against — the "resumes an abandoned route" harm
this rule exists to prevent, and worse than a stale header because a full ladder looks actionable.
Its two still-binding design constraints (U1 must accept an already-dressed manifold; block-keyed
intermediate naming) were moved into `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md`, where they apply; the
retirement answer `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` already records what was kept and what to
check first if dressing is ever revisited.

### Active ccgen scopes, audited 2026-08-16 (verified against code, not headers)

| scope | state |
|---|---|
| `CCGEN_ARBITRARY_ORDER_UCC_SCOPE` + `CCGEN_U1_UCC_ADAPT_SCOPE` | **U0 + U1 COMPLETE and numerically validated; U2 IN PROGRESS.** **UCC now reaches the FCI limit directly** — `U15UccReachesFciLimitTests` solves the generated manifold to self-consistency on LiH+/6-31g (3-electron doublet, so CCSDT is exact) and hits FCI to **3.7e-14**; the obvious system Li/STO-3G is a *vacuous* gate there (t3 worth 0, a broken T3 passes), LiH+ makes the triples worth 8.1e-8. Also verified at rank 4 vs PySCF UCCSD (~6e-16) and rank 6 vs GCC-sliced (**1.6e-17**). **The gap that actually mattered: the GCC→UCC adaptation had 22 call sites and NO numeric gate** (structural checks only); `U14c3UccIsGccSlicedAtRankSixTests` closes it. U1.3 is DEAD (U1.1 designed its hazard out). **U2 is STRUCTURALLY COMPLETE** — U2.1 landed `build_ucc_block_denominator`; **U2.2 landed** `build_ucc_denominator_cache` + `ArbitraryOrderDenominatorCache::{sectors,sector_tensor}`, removing the B4 assumption that a sector reuses its rank's reference denominator (true for RHF where eps is spin-free, false under UHF, where `abab` differs in *shape* too). One code path, not two: `sector_tensor` falls back to `tensor(rank)` when no per-block entry is stored, so RHF is bit-identical — verified by building with and against the change (`be_rccsdtq_sto3g`, the only landed method carrying a sector, `-14.4036550465` to every digit; extended suite 107/107). Gate verified falsifiable against three mutations. **The scope doc's remaining U2 item — make the state's reference a variant — should NOT be done**: measured, the generated kernels touch only `f_oo`/`f_ov`/`f_vv` and `orbital_partition`, never `RHFReference` as a type, so a variant changes every kernel signature and every generated TU for no gain. Those three Fock blocks need the same spin split the ERIs do — that is U3, one change, not two. **Next is U3 + its emitter half.** **The rank-6 PySCF gap is CLOSED** (triples 2.3e-15, was rel 1.9e-3): the defect was in the comparison harness — `update_amps_uccsdt_tri_` updates t1/t2 **in place** before building the T3 intermediates, so `R = (t_new − t)·D` recovered through it is the residual at *different amplitudes*. Fixed by calling `compute_r3_tri_uhf` directly. Nine convention hypotheses had been falsified against it; none could have been right. Full answer: **`docs/CCGEN_UCC_NUMERIC_VALIDATION.md`** |
| `CCGEN_ARBITRARY_ORDER_UCC_SCOPE` (U3/U4/U5) | **U3, U3b AND U4 LANDED; the emitted UCC TU now COMPILES. U5.3c / U5.4 / U5.5 remain.** U3.0–U3.4 spin-blocked ERIs/Fock + emitter routing + open-shell MP2 limit; U4.0–U4.3 an ALL-SECTORS runtime bundle + the `--ucc` switch. Emitted UCC TU has **zero** untagged reads; RCC emit byte-identical throughout (SHA-256 pinned). **Four scope corrections worth carrying.** (1) U3's per-tag canonical block set had to be **derived** — a mixed block's orbits reach only 11 of 16 patterns, so the first restricted emit raised `NotImplementedError` on `vovv`; 6 arrays same-spin, **10** mixed. (2) U3.4 needed **no solver** (first-order MP2 amplitudes are closed-form), so it did not depend on U5 as scoped. (3) **U4.1 was not work at all** — pack/unpack and the update loop already tolerated an empty `by_rank`; U4.0's own gate had *asserted* the update was unreachable, inferring it from `max_rank()==0`, and the inference was wrong. (4) U4 was not a guard question: `validate_kernel_bundle` *required* `residuals_by_rank.size() == max_excitation_rank` while UCC pushes 0, so a UCC bundle was rejected before it ran; promoting one block per rank into the reference slot cannot fix it, because `rank_dims` gives one shape per rank while `aaaa` and `abab` differ in shape. **U4.2 fixed a real out-of-bounds read** (`by_rank[rank-1]` on a state with no reference blocks) that U4.0 had made reachable — removing the guard segfaults, exit 139. **Gate lessons**: a gate that re-implements the routing inline measures a *simulation* of the old code and cannot observe the fix (stayed red at 37 after U3.2 landed); a spin-blind-permutation mutation SURVIVED an array-name gate because what moves is index *order*; and two guards rejecting the same fixture means "it was rejected" asserts nothing — name the guard. **U5 is rescoped ~S → ~M and scoped U5.0–U5.5.** `build_ucc_{spin_block_cache,fock_blocks,denominator_cache}` have **no production callers** (measured — tests only), and `prepare_generated_arbitrary_order_state` still builds the RHF reference and RHF cache unconditionally. **The constraint that shapes the step: the RCC and UCC TUs COLLIDE** — measured `compute_ccsdt_energy` + `make_generated_ccsdt_kernels` at rank 3 (rank 2 shows only one, because RCC emits no bundle below the arbitrary floor, so it *understates* it), and the generator writes **one filename per method**. A UCC build today would overwrite the RCC TU and then fail to link, so UCC needs distinct kernel names AND a distinct filename, not just a flag — that is U5.0, **LANDED** (rank 4 verified clean out-of-band too). **U5.1a is also LANDED**: `ucc_canonical_blocks()` — 24 arrays (7 aaaa / 10 abab / 7 bbbb), derived from each tag's own symmetry group and **not** passed across the codegen boundary. Gated on **both** sides (C++ executes the rule; Python parses the C++ tags + permutation table and compares), falsifiable in three directions including a *Python-side* drift — the one a C++-only gate would miss. **Build-diagnostics traps recorded**: `Terminated: 15` from cc1plus is SIGTERM (the compiler was killed, on the slow `-O1` registry TU), NOT a compile error; and a green `ctest` is not evidence `hartree-fock` builds, since the CC unit binaries do not link it — both appeared together in one session and the ctest result said nothing about the failure. Remaining is U5.1b prepare-path wiring — and **checking how RCC does it changed that step's design**: `build_tensor_cc_block_cache` passes **no block vocabulary at all** (the set IS the struct's seven named members, built unconditionally, and it over-builds — measured, ccsd and ccsdt both read 6 of 7, `ovvo` never touched). So U5.1 must NOT take a `blocks` parameter; the UCC set is the same property one level up, one array per orbit of the 16 o/v patterns under each tag's own symmetry group = 7+10+7 = **24 arrays**, fixed by reference type not method, and derived from U3.0's predicate so the C++ and emitter sets cannot disagree by construction. An earlier draft had it coming from the bundle's `sector_tags`, which is wrong twice: those are *amplitude* tags, and RCC communicates no vocabulary in the first place. If 24 blocks proves too heavy at rank 4, trim by reference symmetry, **never by method** — that reintroduces the coupling. **U5.1b LANDED** (`prepare_generated_ucc_state`; the state deliberately holds the ERI cache in CHEMISTS order because `rebind_physicist` builds a fresh cache from the seven named members and never copies `spin_blocks` — calling it would silently discard all 24 UCC blocks and return something structurally valid). **U5.2 LANDED** (`rebind_physicist_ucc`) — and it took **two opposite wrong answers** to get right. Measured, all 24 blocks: `chem(swap(space),swap(tag)) == stored(space,tag)` (the storage permutes the spin tag) and `swap_mid(stored[key]) == physicist[key]` (the rebind must NOT reapply it). So the rebind is a plain per-key `swap_mid_axes` that copies `spin_blocks` — the omission that makes the RCC version unusable. U5.2a's `ucc_rebind_source` (three mixed blocks 'needing' a bra↔ket hop) was built on the premise that a stored key names a *chemists* pattern; it does not, every block is self-sourced, and that function is **removed**. U5.2a had also retracted U3's cross-source claim; **that retraction is itself withdrawn** — U3 was right about the storage, wrong only in extending it to the rebind. **THE REUSABLE LESSON**: a same-spin block cannot discriminate spin-routing hypotheses (`aaaa` is invariant under the tag permutation and symmetric under the mid swap), so validating against it and reading agreement as confirmation produced BOTH wrong answers. This scope has hit that degenerate vacuity **four times**, once while writing the warning about it — assertions about spin routing must use a MIXED block with asymmetric extents, where a wrong hypothesis changes the SHAPE. **U5.2c LANDED**, so U5.0–U5.2 are complete: the UMP2 energy re-derived through the REBOUND cache (physicist `(i,j,a,b)`) matches the one U3.4 builds from the chemists cache (`(i,a,j,b)`) — first check spanning reference → spin-blocked transform → rebind. It is not redundant with the structural gate: a **no-op** rebind is structurally plausible at every level (right block count, keys, dims) and is caught by the energy assertions alone (same-spin -0.0130 vs -0.1001, mixed -0.1320 vs -0.1248). **U5.3a LANDED** — `make_generated_ucc_kernels` (a SIBLING of the RCC lookup, since the two bundles differ in *shape*: RCC has one reference residual per rank, UCC has none and drives `sector_residuals`) plus the `PLANCK_CC_UCC` option, default OFF. Without the UCC TUs the lookup **errors** rather than falling back to the RCC bundle — that fallback would run a restricted kernel against an unrestricted reference and return a plausible wrong number. **This is the first real test of U5.0's naming work**: a `-DPLANCK_CC_UCC=ON` tree links BOTH kernel sets into one binary (`compute_ccsdt_energy` and `compute_ucc_ccsdt_energy` both present, zero duplicate symbols); until now the collision fix was gated only on emitted *text*. That scratch tree has an unconfigured basis path, so it validates **linking only** — runtime stays with the main tree, where the default path is unchanged and no UCC TU is emitted. **CMake trap recorded**: a conditional `COMMAND` in `add_custom_target` cannot use a generator expression — an empty `$<...:COMMAND>` expands to a bare newline and the shell fails with "syntax error near unexpected token `newline'", which reads like a CMake bug rather than a misuse; use a `POST_BUILD` step. **U5.3b LANDED** — `ucc2`/`ucc3`/`ucc4` route via a new `PostHF::UCCGEN` to `run_uccgen`; in a build without UCC kernels it **errors naming `-DPLANCK_CC_UCC=ON`** rather than falling back to the RCC bundle (that fallback would run a restricted kernel against an unrestricted reference and return a plausible wrong number). Warm-start and `.ccamp` are deliberately NOT carried over (warm start recurses through the RCC registry and would seed an unrestricted solve from restricted bundles; the `.ccamp` meta carries one `(n_occ, n_virt)` pair and cannot describe spin-resolved amplitudes). **U5.3c NEXT, raised as an architecture divergence**: after U5.3b the *generated* path has two drivers doing identical steps. But `ccsdtq.cpp` is **already generic** — measured, nothing in it is rank-4-specific and `cc3`/`cc5`/`cc6` route through it today, so it is **misnamed, not miswritten**. Rename to `rccgen.{cpp,h}` / `run_rccgen` (no new abstraction; my first proposal was to extract a policy-parameterised driver, which unifies two things by adding a third). It also fixes a live bug: a `cc6` run logs `RCCSDTQ` **eight times**. Labels become rank-derived with rank 4 UNCHANGED (`cc4 → Total RCCSDTQ Energy`), because three consumers parse that string — `run_regressions.py:33`, `be_rccsdtq_sto3g`'s `contains`, and `ccsdtq_fci_acceptance.py` — and all three run rank 4 only. Keep `PostHF::RCCSDTQ` and the `be_rccsdtq_sto3g` ID. **U5.3c also fixes the KEYWORD SURFACE**, and a correction belongs with it: I had said UCC 'invented a parallel vocabulary because uccsd was taken' — wrong, RCC did the same (`ccsd`/`ccsdt` are the HAND-WRITTEN solvers, so generated RCC invented `cc3`…`cc6`). The convention is symmetric; the **coverage** is not, from three causes of which only two are defects: (1) UCC lacks method-named aliases at ranks 3–4 — add `uccsdt_gen`/`uccsdtq_gen`; (2) UCC's rank-4 ceiling is a hand-written `switch` in the registry, not a real limit (emitter is rank-generic, verified at ranks 5/6) — follow `PLANCK_CC_MAXORDER` and add `ucc5`/`ucc6`; (3) RCC having no rank-2 generated keyword while UCC has `ucc2` is **correct and should stay** — RCC's `generated_floor` is 4 because hand-written covers 2–3, while `ucc2` exists because U5.4 needs it as the comparison against hand-written UCCSD. **Polymorphic `ccsdtq` (pick RCC/UCC from the reference) is explicitly OUT of scope**: measured, keyword handlers run in input-file order and `correlation` before `scf_type` still works, so resolving there would make one keyword mean different things by line order; and it would apply equally to `ccsd`/`ccsdt`, i.e. to the hand-written solvers. **U3b WAS THE BLOCKER (now landed), found by running U5.4** — and it is a gap I left: when U3 was first scoped I raised 'loop bounds and result allocation must come from the block tag' verbally, it never reached the written scope, and U3 landed as complete having fixed only ERI/Fock **routing**. Everything downstream was gated structurally and never executed a kernel, so nothing caught it. `correlation ucc2` now runs the whole chain and fails at `sector residual shape mismatch at (rank 1, tag aa)`: every kernel opens `const int no = reference.orbital_partition.n_occ` and `build_ucc_fock_blocks` never sets that, so both counts are **0**. Measured in the CCSD UCC TU: **1182** `< no`, **1192** `< nv`, 12 partition reads, 5 result allocations — all spin-blind. This is also the half of U2's withdrawn reference-variant question I got wrong (right about the Fock blocks, wrong about `orbital_partition`). **Tractable because**: indices carry no spin (the U1 bridge drops it), but slot *k* of a factor `t2_abab`/`v_aaaa` carries `tag[k]` — verified **2346 assignments, zero conflicts** across the manifold. **U3b.0 + U3b.1 LANDED**, both inert until U3b.2 connects them. U3b.0 `ucc_term_index_spins`: slot *k* of `t2_abab`/`v_aaaa`/`f_bb` carries `tag[k]`, read **POSITIONALLY not by space** (`t2_abab`'s slots are `(vir,vir,occ,occ)`, so a space-grouped reading agrees on `aaaa`/`bbbb` and differs on `abab` — the fourth instance of the same-spin-can't-discriminate trap, and the first where the gate was written for it up front). It RAISES on a conflict rather than picking a spin, because a disagreement means the slot mapping is wrong (R3.1.2 failure mode). U3b.1 adds four counts + `occupied_count`/`virtual_count` to `CanonicalRHFCCReference`, **additive** — `orbital_partition` is untouched because RCC kernels read it (6 reads each in the rank-3 TU) and is deliberately left **DEFAULT** on a UCC reference: filling it makes `ucc2` *appear* to work (right for `aaaa`, silently wrong for `abab`/`bbbb`), and the gate asserts against that shortcut. **U3b.2 LANDED, so U3b is COMPLETE and the emitted UCC TU COMPILES** at ranks 2 and 3 — the first time any UCC gate fed emitted text to a compiler, and precisely why the spin-blind bounds survived U3/U4/U5.0–U5.3b: every earlier gate inspected *text*. Three scope corrections. (1) **U3b.2b and U3b.2c are ONE atomic change**, proven by the compiler: 2b removes the `const int no/nv` declarations, so 2b-without-2c emits a TU failing with 16 `use of undeclared identifier 'no'` errors at the result allocations. (2) **Only TWO emitter sites are reachable, not three** — `include_intermediates` is forced off under `ucc`, so the intermediate-builder preamble is dead code on this path. (3) The **chunked `_partN` preamble is a site CCSD never reaches** (rank 2 has no kernel over the 256-term threshold; all four rank-3 triples blocks are), so a CCSD-only gate passes with every `_part` preamble spin-blind. Counts are read off the four public members rather than `occupied_count('a')`, whose `std::expected` error path cannot fire when the emitter writes the spin literal itself. **THE REGRESSION IT SHIPPED, and the gate gap behind it:** keying the spin map on "the map came back empty" rather than on `ucc` ALSO fires on the SPIN-ADAPTED path, whose rank-4 sector amplitudes are named `t4_aaabaaab` — 661 such terms got spin-suffixed bounds over a `no`/`nv` preamble and that TU stopped compiling. **Neither RCC gate could see it**: the SHA-256 pin emits with `spin_adapt=False`, and `test_emit_flag_matrix.py` pins `spin_adapt` only at `METHOD = "ccsd"`, below the rank where those names exist — two comprehensive-looking gates blind to the same defect for two different reasons. Fixed, and the pin now covers `spin_adapt` at ranks 2/3/**4** and BOTH engines (`diagram` and the default differ by 2038 lines at rank 2 at *identical byte length*, so a length check alone is insufficient). **Also fixed a pre-existing stale assertion** in `test_spin.py`, red since U3.2 (`8e4bb0c`) which deliberately inverted the contract it asserted; its Fock half had gone stale the same way in U3.3 and was masked behind the first failure. Full ccgen suite **890/890, zero failures** (baseline had 3). **The vacuity trap hit twice more here — fifth and sixth instances — both caught by MUTATION TESTING, not review**: a spin-adapted compile gate written at rank 3 and the extended SHA-256 pin written at ranks 2–3 each passed under the exact mutation they existed to catch. Mutation-test every new gate in this scope before trusting a pass. **U5.4 DONE — `ucc2` reproduces hand-written UCCSD EXACTLY** (B/STO-3G -0.0402694793 both, 12 iterations; H2O+/STO-3G C1 to 1e-10). The stack ran end-to-end for the first time and returned a *wrong* number, which took **two** fixes. **(1) The convention** (`fe744e6`): `v_aaaa` means the ANTISYMMETRIZED `<ij||ab>` to ccgen and the plain `<ij|ab>` to the C++ cache — both sides correct in isolation, never checked against each other; unfixed it returned -0.0705299626 vs -0.0402694793. Fixed on the EMITTER side, chosen architecturally: antisymmetrizing the cache is **not uniformly definable** (a mixed block's exchange partner is a different SHAPE — `oovv_abab` is (noa,nob,nva,nvb), partner (noa,nob,nvb,nva) — so it is a conditional over 12 of 28 blocks leaving one accessor with two meanings), **contradicts a rule `ucc_blocks.cpp:32` already states**, and **silently redefines what three landed C++ gates assert on**. **(2) The routing** (`e33c09b`): the first implementation swapped the last two ARGUMENT POSITIONS rather than the two KET slots. Those coincide only for `oooo`/`oovv`/`vvvv`; on `ooov`/`ovov`/`ovvv` the swap crosses occ/vir and reads the wrong array (`<ic|ka>` is `ovov`, its ket-swapped partner `<ic|ak>` is `ovvo`) — **90 of 180 emitted exchange pairs affected**. The partner is now re-resolved through the SAME block search as the direct read. **Two things the compiler caught that reading would not have**: the new `ovvo` partner needed a bound view (`_eri_blocks_used` re-derived the search independently and went out of step — its own docstring warned against exactly that), and the partner's permutation carries its OWN sign. **THE GATE ENCODED THE BUG** — `test_ucc_eri_convention.py` asserted the partner swapped "the LAST TWO slots", a description of what the code did rather than the contract, so it passed with the bug and could never have failed with it; now rewritten to assert the routing (`ovov`→`ovvo`) and mutation-verified. **TWO REUSABLE LESSONS.** (a) **An exact rational ratio is evidence of a CONSTANT, and a constant is as likely to be a configuration default as a coefficient bug** — `cc_damping` defaults to 0.8, the Jacobi update is `delta = damping*R/D`, so iteration 1 sits at exactly 80% of MP2 on every channel and every system; that masqueraded as a structural defect and cost two investigation steps. Grep the knobs before theorising about the equations. (b) **The localization instruments that worked**: first order at `cc_damping 1.0` (iteration 1 must equal UMP2 exactly — it does, to ten digits, clearing the ERI blocks, denominators, rebind, write-back and energy coefficients in ONE measurement), then iterate-by-iterate against the hand-written solver with DIIS off (first divergence names the order — clean at 1, +1.08e-04 at 2, so the defect was linear in t2). **EIGHTH fixture-vacuity instance, and the first where the fixture PASSED its own non-vacuity check**: `b_ucc2_sto3g`'s `oovv_aaaa` satisfies `v(ijab)==v(ijba)` identically (high-symmetry atom, two degenerate 2p virtuals), so both same-spin channels are exactly zero at first order and any same-spin assertion on it passes vacuously — four different orbital counts and a non-trivial E_corr say nothing about degeneracy WITHIN a channel. C1 fixture `h2o_cation_ucc2_sto3g.hfinp` committed and is the one to reason on. **Full answer: `docs/CCGEN_UCC_ERI_ANTISYMMETRY.md`.** **U5.4 IS NOW FULLY DONE**: the runner gained `requires_build_option` (a case names a CMake BOOL, checked against the build tree's `CMakeCache.txt`), so `b_ucc2_sto3g` is registered and SKIPS in a default build rather than failing — reported as `[SKIP]`, never counted as a pass. Verified both directions: skips in `build/`, passes in `build-ucc/`. **Build note:** `-DPLANCK_CC_MAXORDER=2` does NOT build — `tensor_backend.cpp` hard-includes `ccsdt_planck_generated.cpp`, so rank 3 is the floor; and a `make` failure can still report exit code 0, so check for the binary, not the code. **U5.5 RESCOPED — "open-shell UCCSDTQ == FCI" is IMPOSSIBLE as written** (`docs/CCGEN_U55_UCC_FCI_SCOPE.md`). Three requirements fight: `== FCI` needs `n_elec <= 4` for CCSDTQ to be exact; a worthwhile T4 needs >= 2 electrons of EACH spin; open shell needs `n_alpha != n_beta` — and 4 electrons with 2 of each spin is a CLOSED shell. Open-shell at 4 e- is 3a/1b, and one beta electron can be excited at most once, so the `aabb` T4 sector is identically zero. **Measured on triplet Be/STO-3G** (`<S^2>` exactly 2.000000, no contamination): in-tree FCI `-14.2866221716` and hand-written UCCSD `-14.2866221716` are IDENTICAL to 1e-10 — T3 and T4 both contribute nothing, so a broken implementation of either passes. That is U1.5's Li/STO-3G vacuity one rank up, hit inside the only system the old scope's own constraints allowed. **Gate an INTERVAL on B/STO-3G instead** (5 e-, 3a/2b, so `aabb` T4 is live): FCI `-24.1892649766` vs UCCSD `-24.1892581442`, a 6.8e-6 gap T3+T4 must recover; `ucc4` must land strictly between and near FCI. Both bounds bite — no contribution fails the lower, over-correction fails the upper (below FCI was the B5 signature). Land the cheaper `ucc3` interval first. **Constraints measured**: in-tree FCI REJECTS a UHF reference (`fci.cpp:39`, RHF/ROHF only) so the reference must come from a separate ROHF run — sound because FCI is reference-independent, but worth stating; `ucc4` needs `PLANCK_CC_MAXORDER>=4` (build-ucc is at 3); rank-4 generation MEASURED at 579s, of which `ucc_adapt_equations` is **1.8s** — the cost is entirely the GCC `generate_cc_equations` step the RCC path already pays, so rank-4 UCC is NOT meaningfully more expensive than rank-4 RCC (18533 terms / 14 sectors; rank 3 is 2598 / 9). An earlier draft called this "had not completed after ~10 minutes" and named it the practical risk — that was a `tail`-buffered probe misread as a hang. The real rank-4 risk is COMPILE time against the `-O1` registry pin, not generation; and `ucc4` prints `Total Correlated Energy` (UCCGEN is absent from the `method_label` chain), so the closed-shell harness's `Total RCCSDTQ Energy` anchor cannot be reused — this intersects U5.3c. **U5.5a LANDED** — `b_ucc3_sto3g` registered with a `metric_close` pin plus BOTH interval bounds, and it behaves exactly as scoped: UCCSD `-24.1892581442` < ucc3 `-24.1892636163` < FCI `-24.1892649766`, with T3 recovering **80.1%** of the gap, so the gate is NON-VACUOUS (the thing "== FCI" could not achieve on any system). Both bounds mutation-verified by moving the BOUND, not the expectation — a first attempt moved `expected` and proved nothing, since the bounds test the MEASURED energy. **The runner plumbing landed with it** and unblocked U5.4's case too. Extended suite in the default build: 107 passed, 0 failed, 2 skipped. **U5.5b (rank 4) remains** — needs `PLANCK_CC_MAXORDER=4` on a UCC tree; generation is ~10 min (measured) and compile time is the open risk. **U5.3c LANDED — rank 4 verified byte-identical.** `ccsdtq.{cpp,h}` → `rccgen.{cpp,h}`, `run_rccsdtq` → `run_rccgen` (6 files); the scope's "misnamed, not miswritten" claim held on inspection (the only `4`s are the `generated_floor`), so the rename IS the fix with no new abstraction. **The live mislabel bug is gone**: a `cc3` run announced itself as `RCCSDTQ` twice and `cc6` eight times; now `RCCSDT` and `RCC6`. `rcc_method_label(rank)` is declared in `rccgen.h` so the solver tags and the driver's energy label share ONE rule, and `rcc_method_label(4) == "RCCSDTQ"` keeps the three string-parsing consumers (`run_regressions.py:33`, `be_rccsdtq_sto3g`'s `contains`, `ccsdtq_fci_acceptance.py`) untouched by construction. **Verified**: `be_rccsdtq_sto3g` `-14.4036550465` / `Total RCCSDTQ Energy` / 12 iterations identical before and after; the CCSDTQ==FCI gate passes with the same 6.160e-08 gap; extended 107/107, smoke 35/35. **Keyword surface**: added `uccsdt_gen`/`uccsdtq_gen` and `ucc5`/`ucc6`, and the registry's rank-4 ceiling now follows `PLANCK_CC_MAXORDER` (measured: `PLANCK_CC_METHODS` is derived from it and drives BOTH generator invocations, so UCC TUs exist at ranks 5-6 exactly when RCC's do). All four new keywords reach the registry and error on the missing build option, while `ucc9` still fails as "Invalid Correlation" — so the check discriminates. **Correction worth carrying**: `ucc_independent_blocks` takes the AMPLITUDE rank (twice the excitation rank); the sector counts cited in the scope (6 and 7 at excitation ranks 5 and 6) were right but the call was first written in the wrong units. Coverage note: every UCC gate so far ran through the **diagram** engine only; `wick` is residual-equal by documentation but unpinned for UCC |
| `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE` + `CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE` | **COMPLETE — F1, F2.0–F2.4 and F3 all landed.** The UCC residuals are validated against PySCF UCCSD (CH3/STO-3G, all five blocks) to **~6e-16** — machine precision, gated at 1e-13 rather than the scoped 1e-10. Until this, every landed UCC residual was gated structurally only. Three scope claims were corrected by building it: the closed-shell oracle is a **per-target pairing, not a block sum**; the PySCF amplitude mapping is a **transpose, not a pure rename** (PySCF is `(occ…,vir…)`, ccgen is `(vir…,occ…)`); and **`f_ov` must be zeroed on BOTH sides** — one-sided zeroing is *worse* than neither (8e-9 → 9e-9 → 6e-17), since Planck CC kernels are canonical-Fock by construction while PySCF's `f_ov` is convergence noise that `update_amps` uses. Both vacuous-pass traps avoided and asserted. **U1.2 is unblocked; U1.3–U5 are the remaining UCC work** |
| `CCGEN_ARBITRARY_HARNESS_COST_SCOPE` | **research, not started** — H0 profile is blocking |
| `CCGEN_DRESSING_VS_PRODUCTION_CODES_SCOPE` | **research, D0 answered** — opened by "CFOUR/MRCC ship dressing as their only route, why did ccgen's fail?". D0 found the *derivation* route (`factorize.py`) also fails value preservation, **on GCC**, where there is no spin adaptation to blame: 23/66 `ccsd` doubles terms do not reproduce their source (‖diff‖/‖R‖ = 3.73e-01). So the retirement's decision stands but its stated reason does not. **The factorizer has no numeric gate** — its 47 tests compare factor `Counter`s, which cannot see index order. D1–D3 open |
| `CCGEN_KERNEL_SCALING_SCOPE` | **research, partly open** — H1 (memory-bound) untestable on the current ladder (tops out at 0.49 MiB `t3`); overlaps the cost scope, which hands off to it |

Two docs carried self-contradicting status lines ("nothing here is landed" above a LANDED entry) and
were corrected in the same pass: `CCGEN_ARBITRARY_ORDER_UCC_SCOPE` (U0) and
`CCGEN_KERNEL_WIRING_AND_BENCHMARK_SCOPE` (W0).

## Spherical-basis work still intentionally guarded off

- Spherical analytic gradients (and therefore geomopt / freq) for the post-HF
  correlated paths (RMP2 / UMP2). RHF, UHF, and ROHF spherical gradients,
  geomopt, and frequencies are all landed (ROHF via the same build-W-in-the-
  spherical-basis-then-lift-once pattern the RHF/UHF paths use). MP2 gradients
  still need the response-machinery audit before the same lift adapter (Phase 1)
  can be wired through `compute_rmp2_gradient` / `compute_ump2_gradient`.
  Boundary markers: `water_rmp2_spherical_{gradient,geomopt}_rejected`.
- Spherical PCM
- Spherical DFT and TDDFT
- Any additional spherical workflows not already covered by the landed
  single-point, RHF/UHF-gradient, and RHF/UHF-geomopt-and-freq allow-list

## Symmetry follow-up

- Conventional-path symmetry-unique ERI storage remains out of scope; current full-group reduction is a direct-SCF feature
- ROHF is still outside the full-symmetry direct-SCF implementation scope
- The full-symmetry performance story still has room to improve even after the persisted-skeleton and monomial-operator wins; the remaining major option is a true memory-direct contraction that avoids materializing the dense `nb^4` buffer

## DFT and response-method gaps

- Double-hybrid functionals remain single-point only; analytic gradients,
  geometry optimization, frequencies, and TDDFT are still unimplemented there
- For range-separated functionals, `ImaginaryFollow` and `LinearResponse`
  (TDDFT) remain gated / unvalidated even though gradient-driven workflows are
  now landed
- Analytic Hessian remains unimplemented; frequencies are currently semi-numerical
- DFT imaginary-mode following is not implemented
### TICKET: MPI rank-split the DFT grid layer (Gap 2) — the measured DFT scaling cap

**Priority: highest DFT-HPC item.** This is the single change that gives DFT
HF-like MPI scaling.

**Measured problem** (`scale.json`, Notchpeak notch460, post-#151, 6-31g/os):
DFT strong-scaling walls at **3.5× on 16 ranks (22% efficiency) at nb=208 and
degrades to 20% at nb=312**, while HF on the same ladder holds **10× / 63% and
rises with size**. The DFT/HF per-iteration ratio grows 4.8×→10× from nb=104 to
416 — because #151 distributed the J/K but the grid is still replicated, so the
grid's share of DFT wall time rises with both system size and rank count.

**Root cause:** `grep -rn "Mpi::\|USE_MPI" src/dft/` is empty. Every rank
rebuilds the full grid and evaluates the full XC. The grid loops are
OMP-parallel as of #152 (`xc_grid.cpp:83`, `if (!omp_in_parallel())`) but have
**no MPI rank split** — OMP threads work within a rank; nothing distributes
across ranks. (This supersedes the older "grid loops are still serial" note —
they are threaded now; what's missing is the rank split.)

**The work:**
- Partition grid points (or whole Becke atomic batches) by rank in
  `evaluate_density_on_grid` / the `xc_grid.cpp` density+XC loops.
- Reduce the XC matrix (`nb²`) across ranks with `Mpi::allreduce_inplace`,
  alongside the existing Fock reduce — one more reduction, not a new pattern.
- **Determinism constraint (load-bearing):** the DFT XC reduction is the
  historical jitter site (see the DFT XC Reduction Determinism note). The
  cross-rank sum MUST be in fixed rank order, never completion order, never
  `omp critical`. This is the medium-risk part of an otherwise ~M change.

**Acceptance:**
- `energy(-n k) == energy(serial)` bitwise across k ∈ {1,2,4,8,16} on a DFT
  case at nb where a partition bug bites (16-water B3LYP, nb=208). This is also
  Gap 3's missing CI tripwire — land them together.
- DFT strong-scaling efficiency at 16 ranks rises materially off the measured
  22% baseline; HF-like (>60%) is the target, grid being the last serial piece.

**Not in scope:** grid layer already OMP-threaded (#152); this is MPI only.
The `277ba10` (#151-only, pre-#152) attribution split is write-up-only and does
not block this. Full measured rescope in `docs/HPC_REMAINING_SCOPE.md`.
- Coarse/low-quality DFT grids can still show noticeable orientation sensitivity
  under symmetry reorientation; the validated symmetry-on gradient regression is
  intentionally pinned to `grid ultrafine`

## SCF, post-HF, and workflow gaps

- ROHF post-HF: FCI, CASSCF, and RASSCF now accept ROHF references; RMP2/UMP2
  and the coupled-cluster paths remain RHF/UHF only for ROHF inputs
- ROHF CASSCF/RASSCF only support a closed, doubly-occupied inactive core; a
  spin-polarized open inactive core (distinct alpha/beta core orbitals, with the
  unrestricted core Fock, core energy, and response-block changes it implies)
  is out of scope and stays rejected by the parity guard
- ROHF stability analysis and PCM remain incomplete (ROHF analytic gradients,
  and the geomopt / frequency workflows built on them, are now landed
  Cartesian-side — see Completion)
- The ccgen `TensorOptimized` RCCSDT backend is still treated in-tree as an experimental / phase-4 path

## ccgen generated-kernel performance

The dominant cost — the out-of-line, allocating tensor accessors — is fixed (see Completion).
What remains is the **scaling defect** the six-point ladder exposed: the generated-vs-hand-written
ratio grows from 21.8× to 50.1× with no plateau, and the generated cost does not obey a single
`o^a v^b` power law (21.4% residual, concentrated at high `v`). Full measurement in
`docs/CCGEN_KERNEL_SCALING_SCOPE.md`.

- **Enumerate the terms whose contraction order is wrong.** The high-`v` residual structure points
  at multiple contraction regimes — different residual terms wanting different orders, with the
  emitter picking none. `docs/CCGEN_HIGHER_OPERATOR_REUSE.md` already records `t2·t3·v` as `o⁵v⁵`
  n-ary against `o³v⁴` factored, superlinear in both indices and consistent with the measured
  `o^0.93 v^0.34`. Do this term-level enumeration **before** any emitter change.
- **Then consume `_optimal_contraction_order` in the emitter.** `python/ccgen/tensor_ir.py`
  defines `BLASHint` (`:66`), `_detect_gemm` (`:198`), and `_optimal_contraction_order` (`:283`),
  and `grep BLASHint python/ccgen/emit/planck_tensor_cpp.py` returns nothing — the emitter computes
  and discards all of it. This is the asymptotic fix. It outranks loop fusion, which was measured
  at 0.62× (i.e. no gain) at small size.
- **Firm up the exponents.** `o` spans only 4→8 across six points and the fit still leans on its
  endpoints (leave-one-out moves `o` across +0.40..+1.18, though it keeps its sign in all six
  variants). Two or three points in `o=8..12` would settle it. Treat `o^0.9 v^0.3` as indicative,
  not settled, until then.
- **The memory-bound hypothesis is untested, not refuted.** The whole reachable ladder stays under
  0.85 MiB `t3`, inside L2, so a cache transition cannot fire on it. Testing needs cc-pVDZ-class
  systems (H2O/cc-pVDZ is 6.5 MiB `t3`); at ~50× generated-kernel slowdown that run should be
  time-boxed before committing to it. Not exclusive with the scaling defect — it could add a term
  on top once the working set spills.
- **Rank 4 has no point on the ladder.** Different tensor types, different code path, plus the
  `-O1` registry pin (`CMakeLists.txt:402`) that rank 3 does not carry. The fixed-rank-only
  accessor pass already demonstrated rank 3 is not a proxy for rank 4 — do not assume the rank-3
  exponents transfer. The standing follow-on behind that pin (chunk the giant residual kernels in
  the ccgen emit so any optimization level stays cheap) is now worth re-costing, since the accessor
  no longer dominates.
- **Ladder-design constraint, for whoever extends this.** `choose_determinant_backstop`
  (`src/post_hf/cc/tensor_backend.cpp:241`) routes any case with `nso ≤ 16` **and** `ndet ≤ 10000`
  to the determinant-space teaching backstop, which never calls the generated tensor kernel. Such a
  case produces **no timing at all**, silently, regardless of `PLANCK_RCCSDT_BACKEND`. Any new
  ladder point must satisfy `nso > 16 || ndet > 10000`.

## ccgen dressed intermediates

**LANDED. Only the UCC follow-on remains.** See `docs/CCGEN_DRESSED_KERNEL_PIPELINE.md` for the
full record.

The problem this section used to describe — generated kernels carrying only *syntactic* CSE, never
the Stanton-Gauss dressed operators — is solved. Dressed CC kernels now generate from the build
(`-DPLANCK_CC_DRESS_OPERATORS=ON`), compile, link, and run, reproducing the undressed correlation
energy **and** iteration count at rank 3, pinned by the
`dressed_kernel_equivalence_rccsdt` regression case.

Route note, because this section previously scoped the wrong one: the retired plan was Option A's
exact-cover term algebra (A1-A4). What actually shipped is diagrammatic recognition — dressed
operators are matched as a topological subgraph property, which made A3's subgraph-isomorphism
problem the *mechanism* rather than an obstacle. `dressing.py`/`dressed_equation.py` carry it;
the old `tau.py` exact-cover route is history.

What remains open on the dressed path: **nothing in V1**. The follow-on is UCC
(`docs/CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md`, `docs/CCGEN_U1_UCC_ADAPT_SCOPE.md`) — U0 landed, U1
scoped as U1.0-U1.5, U2-U5 (the C++ side) ahead.

### ccgen parallel generation is not equivalence-safe (separate defect)

`generate_cc_equations(method, parallel_workers=N>1)` produces a **different**
equation set than the serial (`workers=1`, default) path — not just reordered,
genuinely different coefficients/term counts (ccsd: singles 24 vs 27–29,
doubles 200 vs 154). Two independent order-dependent defects, each internally
deterministic:

1. **`_wickaccel` is not spawn-safe.** The C extension's `apply_deltas_layout`
   / `analyze_signature` return divergent index-layout results in a
   freshly-spawned worker vs the parent, corrupting relabeled terms (the energy
   manifold gets factors desynced from their summed-index lists, e.g.
   `f(i,a) t1(b,j)` with summed `(i,a)`). A `CCGEN_NO_ACCEL` env hook (added to
   `wick.py` / `canonicalize.py`) forces the pure-Python path and is inherited
   by spawned workers; it fixes the energy manifold but not defect 2.
2. **Pre-canonical exact merge is partition-local.**
   `merge_exact_term_into_buckets` dedups raw terms within a chunk before
   canonicalization; raw terms that combine when co-located in one chunk
   survive separately when split across chunks (singles: `-1/4` vs two `-1/8`).
   Making it global would defeat its streaming-memory purpose on large BCH
   expansions.

The default path is serial and *is* deterministic + correct; parallel is an
opt-in speed feature. The regression `test_parallel_generation_matches_serial`
is marked `@unittest.expectedFailure` with the root cause inline, and
`test_serial_generation_is_deterministic` pins the guarantee that holds. Real
fix = make the extension spawn-safe (rebuild `_wickaccel.cpp`) and lift the
raw-merge global; deferred as parallel generation is unused by the default
build. No bearing on the dressed-intermediate work above (that runs on the
serial path).

## BSSE follow-up

- DFT ghost / counterpoise support
- N-body counterpoise beyond two fragments
- Counterpoise-corrected gradients and geometry optimization
- Post-HF ghost-reference verification beyond the current SCF-level validated scope

## CASSCF

### Remaining work

#### P2: Optimizer simplification pass — mostly resolved; only cosmetic remainder

A suite-wide sweep of every CAS input recorded which candidate the merit
selector actually accepts. Result:

- **Per-root candidates** (`root*-coupled` / `root*-grad-fallback`): accepted
  **zero** times, yet cost a full per-root coupled solve every stagnant macro.
  **Removed** (see Completion). Dead weight, no behavior change.
- **`numeric-newton`**: the dominant accepted fallback (~125 accepted steps
  across the suite). **Load-bearing — must NOT be demoted.** The original P2
  deliverable to demote it behind `mcscf_debug_numeric_newton` was wrong.
- **Single-pair probes**: accepted exactly once, but that once is the
  load-bearing `probe-pair6-favored[uphill]` step on the SAD-uphill SA-2 canary.
  **Must NOT be removed.**

So the original P2 deliverables (demote numeric-newton, remove probes) are
disproven; only the per-root removal was correct, and it is done.

Cosmetic remainder (low value): make every transcript step label uniquely
identify the path taken. Not required for correctness or performance.

### Future hardening

- Plateau-escape convergence path (`casscf.cpp`, the `Treating the stationary
  orbital plateau as converged` branch) is **correct and load-bearing**, not a
  hack to retire. It is the only exit for a genuinely converged
  state-averaged solution: at an SA stationary point the gating quantity
  `sa_g = Σ_I w_I g_I` goes to ~1e-10 while the per-root screens
  (`root_screen_g` / `max_root_g`) plateau at an O(1e-2) nonzero value, because
  state-averaging makes only the *weighted* gradient stationary, not each
  individual root. With `mcscf_accept_uphill` the per-root convergence screen
  then never passes, so the plateau branch is the correct way to recognize "SA
  gradient converged, energy and step flat → done." This is exercised by
  `water_casscf_sa2_sto3g_sad_guess_uphill` (the only one of the four SA-2
  cases that uses it; the other three converge through the normal gate at
  `sa_g < 1e-5`).
- Keep the two water SA-2 SAD-start regressions, because they intentionally protect two distinct optimizer policies

## Performance and maintenance opportunities

- The DFT Coulomb/exchange (`build_coulomb_from_eri` / `build_exchange_from_eri`)
  contractions are now parallel and verified thread-count-invariant (see the
  Integral Engine note). The remaining DFT parallelization target is the grid
  layer, tracked above under the DFT gaps.
- Rework shell-pair construction to operate at shell granularity rather than per Cartesian AO component
- Eliminate remaining reversed-shell-pair reconstruction churn in gradient paths outside the already-fixed RHF path
- Deduplicate the full-group AO-transform machinery that still exists in both `group_operations.cpp` and `mo_symmetry.cpp`
- Extract a shared `SpatialQuartetLayout` (6-axis dims + strides +
  `spatial_index` + `resize_for_quartet`) and retrofit the OS, HGP, and Rys
  per-quartet scratch onto it. All three now carry near-duplicate per-quartet
  scratch structs — OS's `_eri_scratch`, HGP's `g_hgp_scratch`, and (as of
  PR #126) Rys's `RysScratch` — so the three concrete call sites exist to shape
  the shared interface. Only the spatial-layout core is common; the Boys `m`
  axis, HGP's `a0c0_accum`, OS's no-zero-init policy, and the differing
  accessors stay engine-specific. Bitwise-gate across all three engines
  (`planck-compute-2e`, `planck-hgp-engine-smoke`, plus the OS path via the
  existing ERI gates).
- Refactor `Calculator` only where it buys real safety or clarity: the leading candidates are grouping the loose MP2/UMP2 result cache and introducing a geometry-derived working-state object with a single invalidation point
