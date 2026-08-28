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
- **FU2 — the i-shell (L=6) spherical path has never been checked against another
  code.** L=6 is the only production angular momentum that bypasses
  `normalized_pseudoinverse` entirely and delegates to the recurrence oracle
  (`cart_to_sph_block_recurrence`), so it is a distinct code path from the f/g/h
  gates landed 2026-08-28. The input and its reference are committed but
  **deliberately unregistered** — Ne/cc-pV6Z is 140 spherical AOs and the
  conventional `nb⁴` ERI build makes it far too heavy for the suite (h, at 91 AOs,
  already takes ~37 s):
  `tests/inputs/regression/spherical/ne_rhf_spherical_ccpv6z_ishell.hfinp`,
  PySCF 2.13.0 RHF/cc-pV6Z spherical Ne = `-128.5470611007` Eh, expect ~1e-9.
  A disagreement implicates `spherical_recurrence.cpp`, not the pseudoinverse fix.
  See `docs/SPHERICAL_F_SHELL_ACCURACY_SCOPE.md` FU2.

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

~~Remaining, deliberately deferred until the UCC work (U1–U5) lands:~~
**Both retired 2026-08-26** — see the audit refresh below for what replaced them.

**The four that became due 2026-08-22 are DONE (2026-08-25).** All the UCC work landed, so the
exemption expired for every doc scoping it, and the merge planned here has been carried out:

| retired | into |
|---|---|
| `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` (1374 lines) | `CCGEN_UNRESTRICTED_CC.md` (204) |
| `CCGEN_U55_UCC_FCI_SCOPE.md` (214) | same — deleted, its exactness lesson and the triplet-Be vacuity measurement carried over |
| `CCGEN_U1_UCC_ADAPT_SCOPE.md` (446) | `CCGEN_GCC_TO_UCC_BRIDGE.md` (129) |
| `CCGEN_UCC_NUMERIC_FIXTURE_SCOPE.md` + `CCGEN_UCC_RESIDUAL_EVALUATOR_SCOPE.md` | `CCGEN_UCC_NUMERIC_VALIDATION.md` — deleted, the per-target-pairing correction and the fixture name carried over first |

Everything the plan said to keep was kept and re-verified against the tree rather than trusted:
the per-target-pairing correction, the `(occ…,vir…)` vs `(vir…,occ…)` transpose, the
`f_ov`-on-both-sides result, the β-majority folding table (checked empirically), and the
fixture-vacuity traps. Dropped: the F/U step numbering, per-step *Verify:* lines, the design tables
for options already built, and every hypothesis a later step falsified.

**Audit refresh 2026-08-26 — and both remaining rewrites are now DONE.**

| retired | into | lines |
|---|---|---|
| `CCGEN_SPIN_ADAPTATION_SCOPE.md` | `CCGEN_SPIN_ADAPTATION.md` | 892 -> 143 |
| `CCGEN_KERNEL_WIRING_AND_BENCHMARK_SCOPE.md` | `CCGEN_KERNEL_WIRING.md` | 331 -> 143 |

The stated deferral reason — "U1 works against it" — had expired, since U1-U5.5
all landed. Kept in both: every measured number, and the traps rather than a
summary of them. `CCGEN_SPIN_ADAPTATION` keeps the finding that most resists
intuition — **the layer exists for COST, not correctness**: GCC-on-RHF already
gives the exact closed-shell energy (1e-8 vs PySCF RCCSD), and RCC/UCC exist only
because the spin-orbital form is ~16x the `t2` storage and ~64x the doubles FLOPs
— plus four traps that each passed a gate first, including the synthetic `v`
whose forbidden blocks are zero so a filter "harmlessly" dropped the entire
exchange. `CCGEN_KERNEL_WIRING` keeps the flag table with the note that **two
flags have silently produced wrong answers**, both because a default preserved
historical rather than correct behaviour, and the correction that
`choose_determinant_backstop` binds the **hand-written path only** — several
ccgen docs still record its `nso > 16` requirement as universal.

Part C of the wiring scope (a `benchmark_generated_kernels.py` driver) is
retired unbuilt, deliberately: `run_regressions.py`'s `requires_build_option`,
`PLANCK_CC_T3_TIME` and `PLANCK_CC_FIXTURE_DIR` already cover what it proposed,
and the gap it existed to close — nothing proving the generated path ran — was
closed by gates asserting the routing line.

Two further findings from the same audit pass:

- **`CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` was not merely scope-shaped — its
  CONCLUSION was false as stated.** It answers "can dressing be combined with
  spin adaptation?" with "no, and no", reasoning throughout about the
  **recognition** route only. The **derivation** route does compose, is wired
  into production, and measures 3.12x/3.61x. A superseded-conclusion header was
  added rather than a rewrite, because its diagnosis of *why recognition* fails,
  its five falsified fix attempts and its 52 %-short measurement are all still
  accurate. **A reader landing there previously got a wrong answer with no
  forward pointer** — worse than a stale status line.
- **`CCGEN_KERNEL_PERFORMANCE_SCOPE.md` was an answer wearing a scope filename**,
  and its "Still open" P3 bullet had since been answered by
  `CCGEN_KERNEL_SCALING_SCOPE`. Renamed to `CCGEN_KERNEL_PERFORMANCE.md`, P3
  marked answered, the genuinely-open rank-4 `-O1` pin left named.

Judged compliant in the same audit, for the record: `CCGEN_TEACHING_GUIDE`, `CCGEN_REPORT`,
`CCGEN_GENERATION_AND_VALIDATION` (teaching/report); `CCGEN_HIGHER_OPERATOR_REUSE`,
`CCGEN_DIAGRAM_REPRESENTATION_SCOPE`, `CCGEN_INTERMEDIATE_MEMORY_LOCALITY_SCOPE` (already
question-shaped, work unstarted). `CCGEN_UNRESTRICTED_CC` and `CCGEN_GCC_TO_UCC_BRIDGE` were
in-flight scope at the time and have since been rewritten as answers (2026-08-25).

`CCGEN_DRESSED_KERNEL_VALIDATION_SCOPE` was in that list and has been **deleted** (2026-08-16): it
scoped V2–V6 for the dressed route, which is **retired** (see Completion — dressing and spin
adaptation do not compose, 52 % short on Be). The doc never acknowledged the retirement, so it read
as live scope inviting work the project has decided against — the "resumes an abandoned route" harm
this rule exists to prevent, and worse than a stale header because a full ladder looks actionable.
Its two still-binding design constraints (U1 must accept an already-dressed manifold; block-keyed
intermediate naming) were moved into `CCGEN_UNRESTRICTED_CC.md`, where they apply; the
retirement answer `CCGEN_DRESSING_AND_SPIN_ADAPTATION.md` already records what was kept and what to
check first if dressing is ever revisited.

### Active ccgen scopes, audited 2026-08-16 (verified against code, not headers)

| scope | state |
|---|---|
| `CCGEN_UNRESTRICTED_CC` + `CCGEN_GCC_TO_UCC_BRIDGE` | **COMPLETE — U0 through U5.5 all landed and numerically validated.** `ucc2` == hand-written UCCSD exactly, `ucc3` recovers 80.1% of the UCCSD→FCI gap, `ucc4` == FCI to all ten digits on an OPEN-SHELL system. Three regression cases (`b_ucc{2,3,4}_sto3g`) behind `-DPLANCK_CC_UCC=ON`, skipping cleanly in a default build via the runner's new `requires_build_option`. The full record — the four defects, the eight fixture-vacuity instances, the two generalizable lessons, and the measured costs — is in **`docs/CCGEN_UNRESTRICTED_CC.md`** and **`docs/CCGEN_UCC_ERI_ANTISYMMETRY.md`**; see `vault/Status/Completion.md` for the landed summary. **Remaining, neither blocking:** `wick`-engine coverage for UCC (every gate ran `diagram`; the two are documented residual-equal but unpinned), and a gate on the amplitude-antisymmetry convention (`ucc_amplitude_blocks` asserts it, nothing enforces it; measured satisfied to ~1e-16) |
| `CCGEN_UCC_NUMERIC_VALIDATION` | **COMPLETE.** The UCC residuals are validated against PySCF UCCSD (CH3/STO-3G, all five blocks) to **~6e-16** — machine precision. Until this, every landed UCC residual was gated structurally only. The three interface corrections that cost the most time (the closed-shell oracle is a per-target PAIRING not a block sum; the PySCF amplitude mapping is a TRANSPOSE not a rename; `f_ov` must be zeroed on BOTH sides, one-sided being worse than neither) are recorded in **`docs/CCGEN_UCC_NUMERIC_VALIDATION.md`**, which absorbed the two step ladders that scoped it |
| `CCGEN_ARBITRARY_HARNESS_COST_SCOPE` | **research, not started** — H0 profile is blocking |
| `CCGEN_TWO_DRESSING_ROUTES` | **ANSWERED (2026-08-25).** Opened by "CFOUR/MRCC ship dressing as their only route, why did ccgen's fail?" — the premise was wrong. ccgen has **two** dressing routes and production was wired to the weaker one: recognition (6 hand-seeded spin-orbital fingerprints, `dressing.py`, retired, 52 % short) and derivation (`factorize.py`, from each term's own contraction tree). The derivation route recognizes 5 of the 6 Stanton-Gauss operators **on spatial terms**, was built 8 days later, ships an emit bridge, and **has no production caller to this day** — deferred in its own commit with "CCSD dressing stays D7.3's job" and never revisited. It did fail value preservation (23/66 GCC terms) via two defects, both now fixed: `node_to_term` recorded only the top tree step's summed indices (20/52 malformed specs), and `_derived_name` discarded slot order so one name denoted several contractions. Now value-gated at ranks 2-4 (**0/2536 on quadruples**) and worth **2.0x-7.1x**, growing with rank — the retirement measured only `ccsd` and concluded the opposite. **Recommendation: wire the derivation route; leave recognition retired.** Full answer in `docs/CCGEN_TWO_DRESSING_ROUTES.md`; the operator-granularity half in `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`. **Open:** what CFOUR/MRCC actually do (literature, no longer blocking); UCC carry-over (recognition finds 0 operators there — the tag-blind "fix" is measured and unsound); 6 selection-model gates need re-deriving |
| `CCGEN_OPERATOR_IDENTITY_AND_REUSE` | **O1-O5 COMPLETE (2026-08-25); O6 open.** Answers "when are two derived contractions the SAME operator?" — the question D6's shape-tag fix created by over-splitting operators 12→27 (GCC) and 26→83 (rank 3). Transpose-equivalence is decided **symbolically** (`operator_identity.symbolic_transpose`), exact against a numeric oracle on both bases at two fixtures x three seeds. Merging is implemented end to end and reaches the emitted C++: **27→19 builders on `ccsd`, 254→69 at rank 4**, value-gated at **0/2536** on quadruples. The merge ratio **grows with rank** (1.4x → 2.1x → 3.7x) and roughly doubles the spatial dressing payoff. Two lessons worth carrying: only **sign-preserving** symmetries may be folded (using all 8 ERI permutations produced 2 false merges — the same blind spot as the 52 % defect), and the oracle's fixture must match the basis (`random_tensors` antisymmetrizes `t2`/`v`; ~30 of 48 apparent spatial misses were oracle false positives). **O6 open:** UCC carry-over — recognition finds 0 operators on spin-tagged factors, and the obvious tag-blind fix is measured and unsound |
| `CCGEN_WIRING_THE_DERIVATION_ROUTE` | **W1-W2, W3.1-W3.2, W4.2-W4.3 and W4.5 COMPLETE (2026-08-26). The derivation route has a production caller and computes the right energy.** Wired as ONE dressing axis with a value — `--dressing {none,recognized,derived}` plus `PLANCK_CC_DRESSING` — not a fourth boolean, on evidence in the tree (`print_cpp_planck` has 16 branches, `dress_operators` interacts at three points, and `generate.py:1060` records that a second emit call site already cost a double-wiring). **W4.3 went RED on the first end-to-end comparison** — CH4 off by 1.61e-05, LiH by 1.08e-05, both converging cleanly — and the cause was an **invalid ERI symmetry table**: `lowering/restricted_closed_shell.py` carried the full 8-fold group of the ANTISYMMETRIZED `<pq||rs>`, four members of which are false for spatial blocks, and its phase reaches the emitted C++ directly. **41 of 288** emitted operator builders read the wrong block with a bogus sign. Fixed by defining the spatial and antisymmetrized sets **once** in `ccgen/tensors.py`; CH4 now matches to **2e-10** and LiH **exactly**, and the retired `recognized` route is repaired too. **The full answer — how it was found in five eliminations, why every existing gate missed it (the value gate never emits C++, covers 27/142 doubles terms, and its fixture ANTISYMMETRIZES `v` so the bad relation is TRUE under it), the two operator censuses that looked decisive and were each refuted, and the gates that now gate it — is in `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`**, rewritten from scope into answer. **W3.3 and W5 COMPLETE (2026-08-26), so W1-W5 are all done.** W3.3: `emit_factorized_translation_unit` deleted (**-45 lines**) — it had **no production caller** (25 references, all tests), so "two emitters" was already one plus dead weight; the generate-then-emit convenience moved into `test_factorize.py`, its only consumer, because inlining `generate_cc_equations` at 25 call sites would have been a net POSITIVE diff to remove 13 lines. `print_cpp_planck` gained exactly one parameter (`dressing`) and none of the factorizer's seven selection knobs, the condition W3 set. **W5 — the route's first wall-clock numbers:** LiH 5.12s → **1.64s (3.12x)**, CH4 104.56s → **28.94s (3.61x)**, energies identical to all printed digits and CH4 taking 15 steps either way, so it is per-iteration work rather than fewer iterations. Both land inside the modelled 2.0x-7.1x — worth stating because `CCGEN_KERNEL_SCALING_SCOPE` gave good reason to expect the FLOP model NOT to survive contact; two points is not enough to generalise, and the ratio grows between them. **Two follow-ons scoped (2026-08-26):** (1) **`CCGEN_MERGE_TRANSPOSES_SCOPE`** — thread `merge_transposes` into the production dressing path (M1-M5, measure-before-wiring). It corrects a reading the parent doc invited: the 1.4x → 2.1x → 3.7x figures are an **operator-count** reduction while the modelled FLOP saving is only **1.02x-1.20x**, so the likely win is compile time (the registry TU is `-O1`-pinned and the dressed CCSDTQ TU is 13 MB), not speed — and the speed case, if any, is at rank 4. Prefer making it the default for `derived` over adding a fifth axis; extend the builder gate to call-site permutations, using rank 3+ because every `ccsd` merge permutation is a self-inverse swap and applying it backwards is undetectable there. (2) **`CCGEN_KERNEL_SCALING_SCOPE` revisited** — its H3 ("generated kernels evaluate each term n-arily; `t2·t3·v` is `o⁵v⁵` n-ary vs `o³v⁴` factored") is exactly what derivation dressing fixes, by a different mechanism than the `_optimal_contraction_order` consumption it recommends. W5's 3.12x/3.61x is **consistent with H3 but is NOT a measurement on that ladder** (end-to-end solve times on two systems, one of them off-ladder, versus isolated triples-residual timings — the two sets must not be combined). Two points cannot give exponents, so whether dressing reduces the SCALING or only the CONSTANT is unmeasured. **Re-run the six-point ladder with `--dressing derived` before consuming `_optimal_contraction_order`** — the two fixes may overlap rather than add. Also noted there: the backstop constraint binds the hand-written arm only, widening the usable ladder points on the generated side. UCC stays out of scope pending O6 |
| `CCGEN_SPIN_ADAPT_DEFAULT` | **RESOLVED (2026-08-26). Was never a kernel defect — the build flag `PLANCK_CC_SPIN_ADAPT` defaulted OFF, which `CMakeLists.txt` itself documented as the historical emit that makes the generated correlation energy ~4x wrong.** Opened as "the generated rank-3 CCSDT kernel converges to a wrong answer"; every measurement in that investigation was taken under the defective emit. **The flag now defaults ON.** With it ON the generated rank-3 kernel matches the hand-written path to all ten digits on three systems: Be −0.0517702884, LiH −0.0204594700, CH4 −0.0791116827 (hand-written −0.0791116825, 2e-10; total −39.8058445098 vs PySCF −39.8058445240, 1.4e-08). Rank 4 warm-starts from it in 6 steps against 12 cold. **What cracked it:** compare the CCSDTQ bundle's shared manifolds against the CCSDT bundle's — they are `cmp`-BYTE-IDENTICAL at ranks 1/2/3, excluding "the rank-3 kernel is wrong" in one step and moving the search to the build. The two trees differed in **three** flags, not the one that was assumed. **Coverage gap closed:** no case pinned the flag, so the suite was green while the binary under test was the defective emit. `requires_build_option` now accepts a LIST; `be_rccsdtq_sto3g` requires `PLANCK_CC_SPIN_ADAPT`, `ch4_rccsdt_generated_sto3g` requires it plus `PLANCK_CC_ARBITRARY_LOWER_RANKS` and is **inverted** to assert the correct energy. Gates added: `test_iterate_amps_fixed_point.py`, `test_spatial_residual_vs_pyscf.py`; env-gated `PLANCK_CC_FIXTURE_DIR` probe in `rccgen.cpp` (inert when unset). Findings that outlived it: there is **no rank-2 generated RCC kernel** (`generated_floor` is 3), so any "rank 2 works" datapoint is hand-written CCSD; **Be/STO-3G cannot validate a rank-3 kernel** (t1 and t3 both at machine zero — LiH/STO-3G at 1.6 Å is the fixture); the spatial representative block is `aabaab`, **not** all-alpha (`spin.py:577`); ccgen amplitudes are `(vir...,occ...)` vs C++ `(occ...,virt...)`; and MO phase freedom makes cross-implementation residuals incomparable elementwise (use the phase-invariant Frobenius norm). Full answer: `docs/CCGEN_SPIN_ADAPT_DEFAULT.md`. **W4 UNBLOCKED.** **Second flag found flipped (2026-08-26):** `ch4_rccsdt_sto3g` was FAILING on a clean tree — the **hand-written** tensor path diverging to `E_corr=nan` (`rms(R3)` growing 1.7e-03 → 6.1e-03 → 2.4e-02) — because commit `70a587d` (the W4.2a investigation) flipped `use_diis` to `.false.` in that input while testing whether DIIS mattered for the **generated** path, and left it flipped. The generated path is indifferent; **the hand-written restricted tensor CCSDT solver diverges without DIIS on this system**. Restored: 24 steps, `-0.0791116825`, exactly W4.1's baseline. Two lessons: the two ch4 cases **share one input file**, so probing one path silently broke the other (the input now carries a comment saying the setting is load-bearing); and "not DIIS (both settings reach the same value)" was true of the **generated path only** but was recorded as a general finding — when ruling a variable out, record which path it was ruled out for. Minor open: `be_rccsdtq_sto3g` asserts −14.4036551081 while both builds give −14.4036550465 (6.2e-08, passes at 1e-07, pre-existing and flag-independent); and the hand-written tensor solver's need for DIIS is a real fragility nobody has investigated |
| `CCGEN_KERNEL_SCALING_SCOPE` | **research, partly open** — H1 (memory-bound) untestable on the current ladder (tops out at 0.49 MiB `t3`); overlaps the cost scope, which hands off to it |

Two docs carried self-contradicting status lines ("nothing here is landed" above a LANDED entry) and
were corrected in the same pass: `CCGEN_UNRESTRICTED_CC` (U0) and
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
(`docs/CCGEN_UNRESTRICTED_CC.md`, `docs/CCGEN_GCC_TO_UCC_BRIDGE.md`) — U0 landed, U1
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
