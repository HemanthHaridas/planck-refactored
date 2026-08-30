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

## Build and packaging

- **The compiled-in default basis path has a spurious `install/` segment and
  never resolves.** `CMakeLists.txt` sets
  `BASIS_INSTALL_PATH = ${CMAKE_INSTALL_PREFIX}/install/share/basis-sets`, which
  `basis.h.in` bakes into the binary as the fallback when `BASIS_PATH` is unset —
  but `install(DIRECTORY basis-sets DESTINATION share)` puts them at
  `${prefix}/share/basis-sets`, with no `install/` component. So a build that has
  not had `BASIS_PATH` exported fails at basis loading:

  ```
  [ERR] Basis Parsing Failed : Cannot open basis file: /usr/local/install/share/basis-sets/sto-3g
  ```

  Reproduced 2026-08-30 on a clean default configure+build. It affects both the
  uninstalled build tree and an installed prefix, and `tests/run_regressions.py`
  does not export `BASIS_PATH` either, so **all 35 smoke cases fail without it**
  and pass with it. Everyone working in the repo has an export in their shell,
  which is why it has stayed invisible. Fix is one of: drop `/install` from
  `BASIS_INSTALL_PATH`, or change the `install()` destination to match — they
  must agree. Documented as a workaround in the README meanwhile.

## SCF convergence — the unclaimed 3x

- **Iteration count triples with system size, and nothing is attacking it.**
  Measured in the committed `scale.json` (HF/6-31g water chains, serial, same
  basis and guess throughout): **30 iterations at nb=104 rising to 91 at
  nb=416**. At a flat 30, nb=416 would take ~2065 s instead of 6263 s — a **3x
  multiplier on total cost that no parallel or kernel work touches**. DFT shows
  the same cliff earlier (13 iterations to nb=208, then 51 at nb=312) and
  **fails to converge entirely at nb=416**.

  This is the cheapest large win available, because the other two axes are known
  and expensive: the ERI Fock build is ~200x slower than libcint with four
  candidate optimizations each disproven by measurement
  (`docs/ERI_PERFORMANCE_SCOPE.md` — closing it needs a different engine), and
  MPI scaling is already 42-46 % efficient at 32 ranks. Iteration count
  multiplies with **every** method: HF, DFT, and every post-HF path sitting on a
  converged SCF.

  **Most of an SOSCF already exists**, which is what makes this tractable rather
  than a research project. `src/post_hf/casscf/aug-hessian.h` is a *generic*
  CIAH solver — the same algorithm PySCF's SOSCF uses — whose header states it is
  callback-driven specifically so it is **not** coupled to CASSCF data
  structures, and it is validated by the 11/11 CASSCF gate suite.
  `build_rhf_cphf_matrix` (`src/post_hf/rhf_response.h`) is the RHF orbital
  Hessian, with an RI form that avoids the `nao⁴` build; `uhf_response.h` is the
  unrestricted sibling. **The work is writing the callbacks and deciding when to
  switch, not deriving a Hessian or writing a trust-region eigensolver.**

  The one genuinely non-mechanical part is the DIIS→SOSCF switch criterion:
  second-order steps converge quadratically near a solution and can find a saddle
  far from one, so switching too early is worse than not switching. Scoped S1-S5
  in `docs/SOSCF_SCOPE.md`, starting from the `diis_error` already computed every
  iteration.

## FCI performance — two measured, independent items

Found while sizing an FCIQMC validation fixture; both stand on their own and are
worth doing whether or not FCIQMC happens.

- **The FCI sigma build is single-threaded.** `apply_ci_hamiltonian`
  (`src/post_hf/ci/ci.cpp:437-622`) — the iterative path taken by every space
  above `dense_threshold = 500` — has **zero** `#pragma omp`. The one pragma in
  the file is on the **dense** Hamiltonian build (`:221`), which runs only
  *below* 500 determinants, so it never fires for a case big enough to care.
  `rdm.cpp` is threaded (6 pragmas); `fci.cpp` and `strings.cpp` are not.
  **Measured on `build-full`** (genuinely OpenMP-enabled), N2/STO-3G
  (ndet = 14 400): **121.9 s at 1 thread, 123.6 s at 4 — flat**, and **100.0 %
  CPU with `OMP_NUM_THREADS=8`**, one core of eight. Same signature CC had before
  it was threaded. The outer determinant loop writes `sigma(i)` per determinant,
  so it is a scatter rather than the disjoint-slice shape the CC nests had —
  threading it needs either per-thread partial vectors summed in **fixed thread
  order** (the DFT J/K discipline) or a gather formulation. **Do not use
  `omp atomic` or completion-order accumulation**; that is the DFT-grid jitter
  defect.

- **~53 % of FCI runtime is `malloc`/`free`, not arithmetic.** Leaf-sample
  profile (21 048 samples, N2/STO-3G, 1 thread): the malloc family is ~53 %,
  `apply_ci_hamiltonian` itself 12.0 %, `get_excitation` 5.9 %. The cause is in
  the source: `get_excitation` (`ci.cpp:65`) returns
  `std::pair<std::vector<int>, std::vector<int>>` **by value**, and
  `slater_condon_element` calls it for both spin channels on **every matrix
  element** — up to four heap allocations per element, for vectors that hold **at
  most two entries each** (a Slater-Condon element vanishes beyond a double
  excitation). `std::array<int,2>` plus a count removes it. This looks like the
  cheapest large win in the CI engine, and it compounds with the threading item
  rather than competing with it.

  **Do this one before any FCIQMC work**, if that ever starts: the spawning step
  calls `slater_condon_element` in its innermost loop, far more often than FCI's
  sigma build, so building on the allocating version inherits the penalty. It
  also widens the ndet window where a deterministic FCI reference is affordable,
  which is exactly what the FCIQMC validation strategy is bounded by.

## Research: FCIQMC (scoped, deliberately not started)

- **Scoped as a research question, not a work item, because two prerequisites are
  unanswered and either one kills it.** `docs/FCIQMC_RESEARCH_SCOPE.md`.

  **Q1 — is there a target?** There IS a real window: `CIString` is a `uint64_t`
  giving `kMaxPackedSpatialOrbitals = 31`, and at `n_act` 20-31 the determinant
  count runs 3.4e10 to ~1e17 — addressable by the existing bitstring
  representation but far beyond a stored CI vector. **But nothing currently wants
  it**: CASSCF is validated at CAS(8,6) and smaller, and the regression suite tops
  out at 6 atoms. Name a molecule and active space someone cannot run today, or
  record that this is a capability looking for a use.

  **Q2 — can a stochastic method live in this validation culture?** This is the
  harder half and it is cultural, not technical. The suite carries **161
  `metric_close` assertions**, the tightest at 1e-9, and every recent perf change
  was gated on **bitwise identity** (the CC OpenMP work, the transpose merge, the
  DFT J/K builds). There is **no RNG anywhere in `src/`**. An FCIQMC energy is a
  mean with an error bar and cannot be gated that way; it needs a fixed-seed
  reproducibility gate plus a statistical gate with a blocking analysis, neither
  of which exists. **Answer Q2 by writing that gate against the existing FCI
  before implementing anything** — if it cannot be made to pass where FCI gives
  the exact answer, the method is not maintainable here.

  **The reusable half is genuinely large**: bitstring determinants, occupation and
  parity helpers, `slater_condon_element` (the spawn), `build_ci_diagonal` (the
  death step), and pre-transformed active-space integrals all exist in
  `src/post_hf/ci/`. Missing and real: an RNG policy, a dynamic sparse walker
  container (the existing `det_lookup` indexes a fixed enumerated space), an
  excitation generator with a consistent `p_gen`, and population control.

  **The validation fixture is identified and measured: `N2/STO-3G`** (10 orbitals,
  7α/7β, **ndet = 14 400**, FCI in **124 s** on this machine). It is the smallest
  system satisfying both constraints — FCI cheap enough that a gate can recompute
  the reference, and a determinant space large enough that a few-thousand-walker
  population is a genuine *sample* rather than covering the whole space. The
  existing FCI regression cases (H2/STO-3G at 4 determinants, water/STO-3G at 441)
  are **unsuitable**: the walker population would exceed the space and the gate
  would prove nothing about sampling. `Be/6-31g*` (ndet 8 281, FCI 46.5 s) is the
  useful second fixture because it has only 2α/2β against N2's 7α/7β — a
  two-electron system cannot exercise doubles between different occupied pairs,
  which is where `p_gen` bugs hide, so disagreement between the two isolates the
  excitation generator.

  **The reference cost grows fast, which bounds the whole validation strategy.**
  Two measured points give ~`ndet^1.78`, and C2/STO-3G (44 100 dets) already
  exceeded 10 minutes, so that is a lower bound; water/6-31G (1.66 M) is tens to
  hundreds of hours. So the deterministic reference exists only for ndet <~ 1e5 —
  precisely *not* the regime FCIQMC is for. That is the argument for making the
  **fixed-seed reproducibility gate** the primary one: it runs at any size, while
  the statistical gate can only ever run where FCI is affordable. Candidate inputs
  are committed under `tests/inputs/exploratory/fciqmc/`.

  **One structural tension worth deciding explicitly rather than discovering:**
  every parallel path in Planck is bitwise thread-count-invariant by design and by
  gate. FCIQMC's natural parallelization is not — the annihilation sum depends on
  arrival order. Either accept a fixed-order reduction, or document FCIQMC as the
  one path where that rule does not hold. **Do not make that exception silently.**

## Verification and regression gaps

- **The ccgen Python suite's NINE standing failures are FIXED (2026-08-29); the
  suite is clean.** All nine were red before the merge work and on a clean `HEAD`,
  and **none was a live product defect** — but two were misdiagnosed until someone
  looked closely, which is the general lesson: a standing red is a place nobody
  has looked recently, and the stated reasons drift as much as the code.
  **(C)** one gate asserted the antisymmetry defect W4.3 had fixed, demanding
  `-ovov(i,a,j,b)` where the emitter correctly emits `+ovvo(i,a,b,j)` (off by
  8.77e-01 vs exact on a spatial fixture); corrected, given the numeric
  justification it never had, plus a counter-assertion that the antisymmetric form
  is absent and a second test executing the claim on a deliberately
  non-antisymmetrized fixture; mutation-verified. **(B)** two tests meant to skip
  without pyscf but raised `NameError: gto` past their own `except ImportError`;
  they now carry `test_reference_vs_pyscf`'s `skipUnless(_HAVE_PYSCF)` guard,
  verified to SKIP without pyscf and to RUN when the flag is set. **(A)** six
  selection-model gates broke in `7bdfdaf1`, which correctly split operators
  26 -> 83 and redistributed the savings distribution they assert. Settled first
  by a value probe (**0 disagreements** across 4 seeds, GCC and spatial,
  non-vacuous and mutation-verified) that the factorizer reaches *different but
  equally valid* trees — so this was test debt, not a defect. Four gates restated
  against measured numbers (concentration as a **fraction** of the set, not a
  fixed top-5; key divergence by **median** rather than max; both CCSDTQ gates
  **searching** the budget range, since the divergence regime moved to a +5.77 %
  peak at 3200 GB that is a knife edge and would make a brittle constant), the
  invariance gate restated to assert the **reference count** (exactly 963 under
  every shuffle, while names and savings drift), and the sixth turned out not to
  be distributional at all — a plain test bug where the emitter used
  `engine="diagram"` and the comparison side took `generate_cc_equations`'
  `"wick"` default. Every restated gate is mutation-verified against the defect it
  claims to guard. **Still open, deliberately:** whether the factorizer should be
  made order-invariant by a canonical tie-break — a build-reproducibility
  argument, not a correctness one, recorded in the gate's own docstring. Full
  record in `docs/CCGEN_RED_TESTS.md`.
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
| `CCGEN_ARBITRARY_HARNESS_COST` | **ANSWERED (2026-08-29), and its own premise was wrong — the harness is ~1 %.** Effectively all cost is one rank-3 kernel call. Two fixes landed off the profile: chunk-duplicated operator builds (**1.76x**) and duplicate transpose-equivalent builders (**1.42x-1.52x**, see `CCGEN_MERGE_TRANSPOSES`). Three hypotheses died on the profile (every rank evaluated each iteration = 1.2 %; DIIS packing unmeasurable; intermediates collapse into the kernel). **Open: CC has no OpenMP at all**, rescoped in `CCGEN_CC_OPENMP` — post-merge the split is builders 13.8 % / residual 86.2 %, so the lever is the triples parts (modelled 2.74x at 4 threads), not the builders (1.12x) |
| `CCGEN_TWO_DRESSING_ROUTES` | **ANSWERED (2026-08-25).** Opened by "CFOUR/MRCC ship dressing as their only route, why did ccgen's fail?" — the premise was wrong. ccgen has **two** dressing routes and production was wired to the weaker one: recognition (6 hand-seeded spin-orbital fingerprints, `dressing.py`, retired, 52 % short) and derivation (`factorize.py`, from each term's own contraction tree). The derivation route recognizes 5 of the 6 Stanton-Gauss operators **on spatial terms**, was built 8 days later, ships an emit bridge, and **has no production caller to this day** — deferred in its own commit with "CCSD dressing stays D7.3's job" and never revisited. It did fail value preservation (23/66 GCC terms) via two defects, both now fixed: `node_to_term` recorded only the top tree step's summed indices (20/52 malformed specs), and `_derived_name` discarded slot order so one name denoted several contractions. Now value-gated at ranks 2-4 (**0/2536 on quadruples**) and worth **2.0x-7.1x**, growing with rank — the retirement measured only `ccsd` and concluded the opposite. **Recommendation: wire the derivation route; leave recognition retired.** Full answer in `docs/CCGEN_TWO_DRESSING_ROUTES.md`; the operator-granularity half in `docs/CCGEN_OPERATOR_IDENTITY_AND_REUSE.md`. **Open:** what CFOUR/MRCC actually do (literature, no longer blocking); UCC carry-over (recognition finds 0 operators there — the tag-blind "fix" is measured and unsound); 6 selection-model gates need re-deriving |
| `CCGEN_OPERATOR_IDENTITY_AND_REUSE` | **O1-O5 COMPLETE (2026-08-25); O6 open.** Answers "when are two derived contractions the SAME operator?" — the question D6's shape-tag fix created by over-splitting operators 12→27 (GCC) and 26→83 (rank 3). Transpose-equivalence is decided **symbolically** (`operator_identity.symbolic_transpose`), exact against a numeric oracle on both bases at two fixtures x three seeds. Merging is implemented end to end and reaches the emitted C++: **27→19 builders on `ccsd`, 254→69 at rank 4**, value-gated at **0/2536** on quadruples. The merge ratio **grows with rank** (1.4x → 2.1x → 3.7x) and roughly doubles the spatial dressing payoff. Two lessons worth carrying: only **sign-preserving** symmetries may be folded (using all 8 ERI permutations produced 2 false merges — the same blind spot as the 52 % defect), and the oracle's fixture must match the basis (`random_tensors` antisymmetrizes `t2`/`v`; ~30 of 48 apparent spatial misses were oracle false positives). **O6 open:** UCC carry-over — recognition finds 0 operators on spin-tagged factors, and the obvious tag-blind fix is measured and unsound |
| `CCGEN_WIRING_THE_DERIVATION_ROUTE` | **W1-W2, W3.1-W3.2, W4.2-W4.3 and W4.5 COMPLETE (2026-08-26). The derivation route has a production caller and computes the right energy.** Wired as ONE dressing axis with a value — `--dressing {none,recognized,derived}` plus `PLANCK_CC_DRESSING` — not a fourth boolean, on evidence in the tree (`print_cpp_planck` has 16 branches, `dress_operators` interacts at three points, and `generate.py:1060` records that a second emit call site already cost a double-wiring). **W4.3 went RED on the first end-to-end comparison** — CH4 off by 1.61e-05, LiH by 1.08e-05, both converging cleanly — and the cause was an **invalid ERI symmetry table**: `lowering/restricted_closed_shell.py` carried the full 8-fold group of the ANTISYMMETRIZED `<pq||rs>`, four members of which are false for spatial blocks, and its phase reaches the emitted C++ directly. **41 of 288** emitted operator builders read the wrong block with a bogus sign. Fixed by defining the spatial and antisymmetrized sets **once** in `ccgen/tensors.py`; CH4 now matches to **2e-10** and LiH **exactly**, and the retired `recognized` route is repaired too. **The full answer — how it was found in five eliminations, why every existing gate missed it (the value gate never emits C++, covers 27/142 doubles terms, and its fixture ANTISYMMETRIZES `v` so the bad relation is TRUE under it), the two operator censuses that looked decisive and were each refuted, and the gates that now gate it — is in `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`**, rewritten from scope into answer. **W3.3 and W5 COMPLETE (2026-08-26), so W1-W5 are all done.** W3.3: `emit_factorized_translation_unit` deleted (**-45 lines**) — it had **no production caller** (25 references, all tests), so "two emitters" was already one plus dead weight; the generate-then-emit convenience moved into `test_factorize.py`, its only consumer, because inlining `generate_cc_equations` at 25 call sites would have been a net POSITIVE diff to remove 13 lines. `print_cpp_planck` gained exactly one parameter (`dressing`) and none of the factorizer's seven selection knobs, the condition W3 set. **W5 — the route's first wall-clock numbers:** LiH 5.12s → **1.64s (3.12x)**, CH4 104.56s → **28.94s (3.61x)**, energies identical to all printed digits and CH4 taking 15 steps either way, so it is per-iteration work rather than fewer iterations. Both land inside the modelled 2.0x-7.1x — worth stating because `CCGEN_KERNEL_SCALING_SCOPE` gave good reason to expect the FLOP model NOT to survive contact; two points is not enough to generalise, and the ratio grows between them. **Two follow-ons scoped (2026-08-26):** (1) **`CCGEN_MERGE_TRANSPOSES` — DONE 2026-08-29 (M1-M5), and its prediction was wrong in the safe direction.** Threading `merge_transposes` into the production dressing path measures **1.42x (LiH) / 1.52x (CH4)**, energies bitwise identical and iteration counts unchanged — against a scoped expectation of "likely compile time, not speed" (1.02x-1.20x) and then a profile-weighted re-cost of 1.21x-1.36x. Both underestimates came from treating operators as equal-cost. It is now **unconditional for `derived`** (the preferred wiring; no flag kept — 10 lines, 8 of them comment) and the call-site gate landed as `test_merged_call_sites.py`, mutation-verified at ranks 3 AND 4. Compile time and TU size also improve (registry TU 1.5x smaller, ~1.1x faster). **Rank 4 needed no new mechanism** — same method list, same emit path, merging already unconditional — and the reduction roughly doubles with rank: **1615 -> 239 builders (6.8x)** vs rank 3's 3.2x, TU 11.0 MB -> 6.6 MB. Only a rank-4 end-to-end RUN remains untried (6.6 MB TU against the `-O1`-pinned registry). (2) **`CCGEN_KERNEL_SCALING_SCOPE` revisited** — its H3 ("generated kernels evaluate each term n-arily; `t2·t3·v` is `o⁵v⁵` n-ary vs `o³v⁴` factored") is exactly what derivation dressing fixes, by a different mechanism than the `_optimal_contraction_order` consumption it recommends. W5's 3.12x/3.61x is **consistent with H3 but is NOT a measurement on that ladder** (end-to-end solve times on two systems, one of them off-ladder, versus isolated triples-residual timings — the two sets must not be combined). Two points cannot give exponents, so whether dressing reduces the SCALING or only the CONSTANT is unmeasured. **Re-run the six-point ladder with `--dressing derived` before consuming `_optimal_contraction_order`** — the two fixes may overlap rather than add. Also noted there: the backstop constraint binds the hand-written arm only, widening the usable ladder points on the generated side. UCC stays out of scope pending O6 |
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
### HPC campaign — what is left (verified against the tree 2026-08-30)

Full measured rescope in `docs/HPC_REMAINING_SCOPE.md`. **Tiers 3 and 1 are done
and measured** for the default engine, HF and DFT: the `planck-mpi` front end, the
distributed direct-SCF Fock build (#146), the DFT J/K build (#151), and the **DFT
grid rank-split (#156 — Gap 2, CLOSED)**; `src/dft/driver.cpp` carries the
`Mpi::rank`/`size`/`allreduce_inplace` split. This entry previously described Gap 2
as the highest open DFT-HPC item and gave it a full ticket; it had already landed.

Measured on `scale.json` (notch386, 6-31g, os, 1 thread/rank): HF 42-46 %
efficiency at 32 ranks and near-linear to 16; DFT 46-65 % and **rising** with
system size. Both DFT walls — memory and scaling — are closed.

Remaining, in the doc's value order:

- **Gap 3 — no scale-proving fixture in CI (~S, the #1 item).** Every regression
  input is <= 6 atoms, so nothing fails if distribution silently regresses to
  replication. The cheap fix is one 16-water/6-31g (nb=208) case asserting
  `energy(-n 2) == energy(-n 4) == serial` **bitwise** — a correctness tripwire at
  a size where a partition bug can actually manifest, not a speed measurement.
  It is what stands between "DFT scales" (measured by a hand-run harness) and
  "DFT scaling cannot silently regress".
- **Gap 1 — two of four engines silently replicate (~S).** The one-shot
  `_compute_2e` tensor builds in HGP and Rys are not MPI-distributed; both carry
  an explicit in-tree admission (`hgp.cpp:1392`, `rys.cpp:1460`). `mpirun -n 8`
  with `engine hgp` in conventional mode is *correct but pointless* — every rank
  does 100 % of the work. OS already has the pattern (6 lines plus one
  `allreduce_inplace`). **Decide one of two — stripe them, or reject `-n > 1` with
  a clear message — but do not leave it silent.** Arguably YAGNI: HPC runs use
  direct SCF, which is already striped for every engine.
- **Gap 5a — post-HF OpenMP: PARTIALLY CLOSED (2026-08-30).** The gap was that
  every `src/post_hf/cc/*.cpp` backend and every generated kernel had **zero**
  `#pragma omp`. The **generated** side is now threaded: `planck_tensor_cpp.py`
  emits `collapse(3)` behind `CCGEN_OMP_COLLAPSE`, measured **3.22x at 4 threads**
  with energies bitwise identical across thread counts
  (`docs/CCGEN_CC_OPENMP.md`). **Still open: the hand-written path** —
  `tensor_backend.cpp` has 0 pragmas, and MP2 is unprofiled. The gap's own advice
  still applies: profile first, because which post-HF path is the real wall-time
  sink at a realistic size has never been measured. Its determinism constraint
  (scalar-accumulator terms are not thread-invariant under a naive `reduction`)
  was handled on the generated side by keeping the inner summed loop serial
  within a thread.
- **Gap 5b — dense post-HF MPI stripe (~M).** Untouched: `grep -rl "Mpi::"
  src/post_hf/` is empty. The reclassification is the useful part — RI is a
  *memory* strategy, not a distribution prerequisite, so the dense MP2/CC
  contractions can be rank-striped on the outer MO index today. RI-post-HF
  distribution stays the separate, memory-motivated tail (Tier 2).
- **DFT nb=416 fails to CONVERGE, not OOM.** 32-water RKS hits max_cycles at the
  serial baseline (4.8 GB, it ran and iterated). A convergence-robustness item
  (guess/DIIS/damping at large water chains), **wholly separate from the HPC
  track** — do not conflate it with a scaling defect.

**Closed since that doc was written:** Gap 4 (`scale_bench.py`'s Q2 verdict now
keys on the DFT RSS nb-exponent rather than a raw >50x ratio, `2b764b21`) and the
Q1 verdict (Karp-Flatt serial fraction rather than efficiency-at-max-ranks,
`ebf8ae5c`).
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

**CLOSED (2026-08-29): the dressed-vs-undressed ladder re-run is abandoned, and
the measurement route with it.** `CCGEN_KERNEL_SCALING_SCOPE` and
`CCGEN_KERNEL_PERFORMANCE` both recommended re-running the six-point ladder under
`--dressing derived` before consuming `_optimal_contraction_order`, to settle
whether the two fixes overlap. That cannot be done.

`PLANCK_CC_T3_TIME` cannot fire in any build — it sits in the
`use_generated_kernels` branch the rank-3 representation fix rerouted away from. A
replacement three-arm probe was built and established the blocking fact: **the
hand-written and generated arms have no residual-level agreement gate.** They are
distinct solvers with distinct amplitude representations, both individually
correct (each converges to `-0.0791116825` on CH4, PySCF to 1.4e-08), and no
shared intermediate state exists at which their residuals are elementwise
comparable. Four framings failed; `restore` belongs to a wedge-packed *amplitude*
and annihilates a raw residual by 2.0e+05, which
`CCGEN_RANK3_KERNEL_AND_SOLVER.md:21-24` had already established.

What remains measurable is whole-iteration timing validated by converged energy —
which describes *"solver iteration"*, not *"triples kernel"*, since each arm's
overhead is inside it, and so cannot adjudicate the exponent question it was meant
to answer.

**Consequence: the generated-vs-hand-written gap is actionable only through
code-level comparison and FLOP estimates.** The probe, the scope doc, and the
build trees were removed rather than left as an attractive dead end. The recorded
fits (`o^4.87 v^4.52` generated, `o^3.94 v^4.18` hand-written, ratio
`o^0.93 v^0.34`) and the term-level enumeration below remain the basis for the
emitter work; they do not need this re-run.


The dominant cost — the out-of-line, allocating tensor accessors — is fixed (see Completion).
What remains is the **scaling defect** the six-point ladder exposed: the generated-vs-hand-written
ratio grows from 21.8× to 50.1× with no plateau, and the generated cost does not obey a single
`o^a v^b` power law (21.4% residual, concentrated at high `v`). Full measurement in
`docs/CCGEN_KERNEL_SCALING_SCOPE.md`.

- **~~Enumerate the terms whose contraction order is wrong.~~ DONE (2026-08-29) —
  `docs/CCGEN_WHY_GENERATED_IS_SLOW.md`.** Two causes; **the larger one already
  ships.** *(1) Contraction order — FIXED.* The undressed emitter gives every term
  its own `o³v³` nest evaluated n-arily; **391 of 824 terms carry a four-index
  inner sum** (`o⁵v⁵` vs `o⁴v³` factored) = **83–90 % of generated FLOPs**.
  **`--dressing derived` eliminates all 391** — 824 nests → 414, zero four-deep —
  modelled **10x→18x growing with size**, moving the *exponents*
  (`o^4.92 v^4.94` → `o^4.42 v^4.40`), consistent with the measured 3.12x/3.61x.
  Census validated against the ladder: undressed model `o^4.92 v^4.94` vs measured
  `o^4.87 v^4.52`, **`o` agreeing to 0.05**. Caveat: the hand-written side is not
  FLOP-bound the same way, so a modelled *ratio* overpredicts by 2 orders of
  magnitude — trust the generated-side model only.
- **~~THE NEXT LEVER: fuse loop nests.~~ BUILT AND REFUTED (2026-08-29).** Fusion
  is implemented (`CCGEN_FUSE_LOOPS=N`, default 0, byte-identical when off) and
  reduces the dressed rank-3 triples kernel **806 -> 15 nests (54x)**, halving the
  TU and dropping 845 KB of binary. **It changes runtime by 0-3 %, inside noise, at
  three sizes spanning 7x in `t3`** (BH3 9.52->9.54 s, CH4 29.59->28.60 s,
  HF/6-31G 154.00->154.54 s). Energies bit-identical at every fusion level; both
  generated-route gates pass.

  **The traffic model is refuted on its own stated criterion** ("negligible at BH3,
  material at larger `t3`"): HF/6-31G is 3.4x BH3's working set and shows +0.35 %.
  It counted a full `o³v³` read+write per nest, but consecutive nests over the same
  `result` hit the same cache lines and `t3` never leaves L2 at reachable sizes —
  it priced traffic already served from cache. **Keep fusion as a compile-time and
  code-size lever** (the registry TU is `-O1`-pinned because these are pathological
  to compile); it is not a speed lever.

  **Re-tested after H5 and still ~0** — BH3 5.69→5.49 s (−3.5 %), CH4 16.95→17.01 s
  (+0.4 %), HF/6-31G 97.23→96.19 s (−1.1 %); non-monotonic, energies bit-identical.
  Worth re-testing rather than citing the first result, because H5 raised the
  residual's share of runtime **32.3 % → 54.9 %** (nearly doubling fusion's Amdahl
  leverage) *and* changed operand residency during the nests — operators are now
  built once and stay resident, instead of 270 being rebuilt immediately before each
  part's terms. **Amdahl amplifies a real effect, not a null one.** The refutation
  now holds under **two different memory regimes**, which is stronger than the
  original single measurement.

- **H5 LANDED (2026-08-29): dressed operator builds are hoisted out of the
  `_partN` chunks — 1.76x at rank 3, 20.8x fewer builder calls at rank 4.**
  `_emit_chunked_kernel` emitted every dressed operator inside *every* part, so the
  duplication factor equalled the part count. Rank-3 triples: 1080 calls for 270
  operators. **Rank-4 quadruples: 16 092 for 894** — 18 parts — so the defect scaled
  with kernel size, worst at the production target. Now built once into a generated
  `<kernel>_ops` struct and passed by `const&`.
  Measured: CH4 29.59 s → **16.81 s (1.76x)**, BH3 1.71x, builder time 2.64x,
  residual 1.04x (the check that the decomposition is sound). **`E_corr` bitwise
  identical** on both generated gates — hoisting reassociates nothing. Rank-4 TU
  12.8 → 10.5 MB with 48 170 fewer call sites, a compile-time win on an `-O1`-pinned
  TU. Undressed path byte-identical. Not done: a rank-4 dressed end-to-end run
  (10.5 MB TU + `-O1` registry); if one is ever built, check `be_rccsdtq_sto3g`
  against `-14.4036550465`.
  **What it leaves:** builders are still 45 % of rank-3 runtime after removing 75 %
  of the calls, so further gain is about what a single build costs — which is H6.

- **HOTSPOTS RANKED (2026-08-29, post-H5, HF/6-31G), and the top mergeable one is
  now FIXED.** The ranking that opened this entry:

  | hotspot | share | fixable |
  |---|---|---|
  | triples residual `part1` (`o²v²` terms) | **44.8 %** | hard — both models spent |
  | **`build_W_t2t2v_oooovv`, 38 builders** | **20.4 %** | **DONE — `merge_transposes`** |
  | triples residual `part0` | 13.8 % | hard |
  | `build_W_t1t3v_oooovv` | 8.9 % | done, same lever |
  | `build_W_t1t1t2v_oooovv` | 5.6 % | done, same lever |
  | everything, at once | 100 % | **OpenMP** — the largest remaining lever; rescoped, see `CCGEN_CC_OPENMP` |

  **M1-M5 landed 2026-08-29; `merge_transposes` is unconditional for
  `--dressing derived`.** See `vault/Status/Completion.md` and
  `docs/CCGEN_MERGE_TRANSPOSES.md`. Measured **1.42x (LiH) / 1.52x (CH4)**,
  energies bitwise identical, iteration counts unchanged — above both the
  operator-count model (1.02x-1.20x) and the profile-weighted re-cost
  (1.21x-1.36x). The remaining ranked hotspots are the two triples-residual parts,
  which no current lever addresses, so **H6 (OpenMP) is now the largest remaining
  item** — CC is still the only hot path in Planck with no threading.

  Incidental finding: **chunking splits by term COUNT, not cost.** All three heavy
  parts hold 256 nests but differ 18x in modelled cost (`part1` collects the `o²v²`
  terms). Harmless today; it would make any per-part parallelism badly unbalanced,
  and a cost-weighted split is the cheap fix if that is ever needed.

- **CC is the only hot path in Planck with NO OpenMP, and it is the largest
  remaining lever — RESCOPED 2026-08-29 with fresh measurements, which moved both
  the number and the plan.** The claim holds: **0 `pragma omp` in
  `src/post_hf/cc/*.{cpp,h}` and 0 in the emitted kernels**, against 8+ other files
  under `src/` that carry them. ERI, Fock, the 4-index transforms and the DFT J/K
  builds are all threaded; CC is not.

  **What changed.** The old estimate (3.86x at 4 threads, "start with the
  builders") rested on H5's 45.1 % / 53.7 % split. `merge_transposes` has since cut
  builder work hard: measured now, **builders 13.8 %, residual 86.2 %** (HF/6-31G,
  21 341 leaf samples; CH4 agrees at 17.2 / 82.8). So **threading only the builders
  now caps at 1.12x**, and the residual is the whole story — concentrated in
  `triples_residual_part1` at **63.6 %** and `part0` at 19.6 %. Threading the three
  triples parts models **2.74x at 4 threads**.

  **Two evidence corrections.** (1) The "98.8 % CPU on 8 cores" observation was
  taken on a tree where `OpenMP_CXX_FLAGS` is `NOTFOUND` and `-DUSE_OPENMP` never
  reaches the compile line — **every** Planck pragma was inert in that binary, so
  the number could not distinguish "CC has no pragmas" from "this build has no
  OpenMP". **Now measured on `build-full`**, which is genuinely threaded (`-fopenmp`,
  libgomp linked, g++-15, Release, MAXORDER=4): CC is flat **78.03 s -> 78.49 s**
  from 1 to 4 threads at 99.1 % CPU, while direct-SCF HF/cc-pVTZ in the **same
  binary** goes **2.70 s -> 0.81 s (3.3x)**. That positive control is what makes it
  a measurement rather than an inference, and `build-full` is the baseline O2 must
  beat. (Pick the control carefully: a small DFT case shows nothing, and a larger
  one is grid-bound because the DFT grid layer is itself unthreaded.) (2) "The emitter
  never emits a pragma" is false: `emit/cpp_loops.py:331` already emits
  `#pragma omp parallel for collapse(n) schedule(dynamic)`, and
  `print_cpp_optimized` defaults `use_openmp=True`. That is a *different* emit path
  from production's `print_cpp_planck`, so it is not a switch to flip — but it is a
  working in-tree precedent for the exact pragma shape, which reduces the work.

  **The binding constraint is granularity, not safety.** Each triples nest is 6-deep
  over `(i,j,k,a,b,c)` writing disjoint `result(...)` slices with no cross-thread
  reduction — the DFT J/K shape (bitwise thread-invariant), not the DFT-grid shape
  that caused the historical jitter. But the outer `i` trip count is `no` = **5** on
  every reachable test system, so a bare `parallel for` gives 1-2 iterations per
  thread. **`collapse(2)` is required** (25 trips), `collapse(3)` available (125);
  writes stay disjoint under both since the collapsed indices are all output
  indices. Verify bitwise across `OMP_NUM_THREADS`=1/2/4/8 rather than assuming it.
  Scoped O1-O4 in `docs/CCGEN_CC_OPENMP.md`.

  **O2 DONE (2026-08-30): 1.93x at 4 threads, bitwise deterministic.** One
  `#pragma omp parallel for collapse(3) schedule(static)` on each of `part1`'s 256
  nests, hand-edited into the generated file (the emitter is O4's job). HF/6-31G
  through the generated rank-3 route: **78.67 s -> 40.85 s at 4 threads**, 37.37 s
  at 8, against an Amdahl ceiling of **1.94x** for a 64.6 % part — so `part1` is
  now essentially fully parallel and nothing further is available inside it.
  `collapse(3)` beats `collapse(2)` consistently (40.85 vs 42.63 s) because 125
  chunks balance across 4 threads where 25 do not; both are bitwise identical.
  **Correctness verified the way the DFT J/K builds were:** every `E_corr`, `dE`,
  `rms(res)` and `rms(step)` matches across all 15 iterations at `OMP_NUM_THREADS`
  = 1/2/4/8 and against the unthreaded baseline. CPU utilization moved 99.1 % ->
  160.6 %, the cheap check that the pragma fires before any timing is read.
  **Two build-mechanics traps recorded in the scope:** `make hartree-fock`
  regenerates the file and silently wipes hand-edits (256 pragmas -> 0, no error),
  so compile the object and link directly; and a copied build tree rebuilds into
  the ORIGINAL, because `CMAKE_CACHEFILE_DIR` is absolute.

  **O3 DONE (2026-08-30): 3.11x at 4 threads**, and it found a dead-work defect
  along the way. Threading all four triples parts (806 nests) gives **2.81x**,
  landing almost exactly on the modelled 2.74x. Then, inspecting the triples entry
  point in order to thread its 88 operator builds, it turned out
  `compute_ccsdt_triples_residual` **builds every operator twice** — once as 88
  `const auto` locals, then again inside the `ops` aggregate — and **the locals are
  never referenced**. Deleting them is worth a further 4.9 s serial / 4.3 s at 4
  threads (~6 %), taking the total to **73.79 s -> 23.70 s (3.11x)**, or **3.41x**
  against the original unthreaded binary. CPU utilization went **99.1 % -> 359.9 %**
  on 4 performance cores. Bitwise identical at 1/2/4/8 threads and against the
  unthreaded baseline throughout — that removing dead builders changes no number is
  itself the proof they were unused. **The defect is in the emitter**, not this
  file: `planck_tensor_cpp.py:1173-1178` emits the intermediate builds
  unconditionally and then `_emit_chunked_kernel` re-emits them into the struct —
  H5 added the hoist without removing what it superseded, so every chunked kernel
  pays twice. Expected to matter more at rank 4 (894 operators vs 88). **O4 must
  fix both**: emit the pragma, and skip the duplicate emission on the chunked path.
  The triples builders never needed threading — they needed deleting.

  **O4 DONE (2026-08-30): both changes are in the emitter, and the default emit is
  unchanged apart from the dead builds.** `_emit_kernel` now skips the
  intermediate-build emission on the chunked path (the `ops` struct already
  carries them), and `emit_planck_term` takes an `omp_collapse` threaded from
  `CCGEN_OMP_COLLAPSE`, following `_fuse_loops_setting`'s env-var precedent rather
  than adding a `print_cpp_planck` parameter. Measured HF/6-31G: the **shipping
  default** gains **78.67 s -> 76.68 s (2.6 %)** from the dead-builder fix with no
  threading at all, and `CCGEN_OMP_COLLAPSE=3` gives **74.37 s -> 23.11 s
  (3.22x at 4 threads)**, slightly better than O3's hand-edit because the emitter
  also annotates the singles/doubles residuals. Energies bitwise identical at
  1/4/8 threads and against the unthreaded baseline; the pragma-disabled TU differs
  from the pre-O4 emit ONLY by the 88 removed dead builder lines. Regression cases
  `lih_rccsdt_generated_sto3g`, `ch4_rccsdt_generated_sto3g` and `be_rccsdtq_sto3g`
  pass; ccgen suite 876/0 unchanged. **New gate `test_chunked_kernel_builds_once.py`**,
  mutation-verified. The lesson worth keeping: the duplicate build was invisible to
  every existing gate because it is semantically a **no-op** — every value gate was
  correct while the work was done twice. Only wall-clock or reading the emitted text
  could see it, which is why the new gate reads the text.

- **The residual generated-vs-hand-written gap is mostly NOT a codegen defect.**
  **The two paths are different solvers and their wall-clock is not a like-for-like
  ratio** — wedge-packed vs dense amplitudes, cheap dressed intermediates vs a full
  generated kernel per rank, and **40 vs 16 iterations on CH4**. The same reasoning
  that forbids comparing their residuals elementwise
  (`CCGEN_RANK3_KERNEL_AND_SOLVER.md`, and T2.5 of the retired ladder work) applies
  to their timings. Quote it as "the generated production path costs Nx end to
  end", never as "the generated kernel is Nx slower".

  Both codegen hypotheses are spent: the path is **not FLOP-bound** (cause 1's
  modelled 11.2x realised as 3.62x) and **not traffic-bound** (cause 2 cut
  traversals 54x for ~0 %). What remained was answered by profiling rather than
  modelling (`docs/CCGEN_ARBITRARY_HARNESS_COST.md`): **it is not solver design
  either — the harness is ~1 %**, and the four hypotheses held out for it (every
  rank evaluated each iteration, no materialized intermediates, dense DIIS packing)
  are all dead. The cost was redundant operator construction inside the kernel, now
  fixed twice over (1.76x + 1.5x), leaving OpenMP as the open lever. The rule that
  made this work: profile generated-vs-generated across a configuration change,
  never against the hand-written path. Whether the emitter has anything left
  is unmeasured; two attempts to find it by modelling went 1-for-2.
- **~~Then consume `_optimal_contraction_order` in the emitter.~~ PROBABLY
  REDUNDANT.** It targets exactly the 391 terms `--dressing derived` already
  eliminates. `tensor_ir.py:283` still computes and discards it and
  `grep BLASHint python/ccgen/emit/planck_tensor_cpp.py` still returns nothing, so
  the lever is real — but the work is done by a route that is wired and
  value-gated. Re-check before building; fusion is the live lever.
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
