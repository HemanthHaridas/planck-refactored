# Which of the two surviving surfaces carries the rank-3 triples defect?

**Research scope, in-flight.** Continues `CCGEN_RANK3_TRIPLES_DEFECT.md` and
`CCGEN_RANK_PARITY_INVESTIGATION.md`. Rewrite as an architecture answer once the defect is found.

## What narrowed the surface

A term-level diff of the two emitted triples kernels, normalizing only the accessor syntax
(rank 3 emits `amplitudes.t2(i,j,a,b)`, rank 4 emits `t2({i,j,a,b})` — the same read):

| | terms |
|---|---|
| rank-3 `compute_ccsdt_triples_residual` | 811 unique |
| rank-4 `compute_ccsdtq_triples_residual` | 854 unique |
| **shared, byte-identical after syntax normalization** | **811** |
| **rank-3-only** | **0** |
| rank-4-only | 43 — every one references `t4` or `t4_aaabaaab` |

**The failing rank-3 kernel is a strict subset of a rank-4 kernel that matches PySCF `rccsdtq` to
1e-10.** The 43 extras are exactly the T4-coupling terms CCSDTQ should add. Reproduce with
`/tmp/claude-501/rank3_term_diff.py` (below).

A first pass at this diff reported "185 rank-3-only terms" and read them as a structural gap. That
was wrong: it collected `acc +=` lines from the whole rank-3 TU, sweeping in the singles, doubles
and energy kernels. Grouping accumulations by their enclosing `compute_*` function gives 0. The
error inverted the conclusion, so it is recorded rather than quietly corrected — and it is the
reason every step below groups by kernel.

### What that eliminates

| suspect | source | status |
|---|---|---|
| Spatial spin-adaptation lowering of an odd manifold | rank-3 doc suspect #1 (*leading*), parity P4 #3 | **Eliminated.** Those 811 terms are the adapted output and they are correct at rank 4. |
| Sz sector count / single-sector handling | parity P4 #1 | **Eliminated**, independently — parity P1 already killed it (rank 2 is correct with one sector). |
| The emitted term algebra generally | — | **Eliminated.** The emitted text is identical. |
| `restore_restricted_t3_structure` interaction | rank-3 doc suspect #2, parity P4 #2 | **Survives.** |
| The rank-3 C++ harness around it | not previously listed separately | **Survives.** |

The rank-3 doc's *leading* candidate is dead. Its second candidate is now one of only two, and the
other one was not on the list at all.

**The defect must live in something rank 4 does not execute.** Both survivors are in the C++
consumer, not the generator — which is a different layer from where the investigation has been
looking.

## The two surfaces

### S1 — `restore_restricted_t3_structure` and its convention

`tensor_backend.cpp:1977`, three sub-operations in sequence:

1. `apply_restricted_t3_permutation_symmetry` (`:1906`) — sums all 6 *simultaneous* occ/virt
   permutations. **Not idempotent** (a second application multiplies by 6).
2. `apply_restricted_t3_p3_full` (`:1938`) — subtracts the mean over the 6 virtual-only
   permutations: `x - mean(P3 x)`.
3. `purify_restricted_t3` (`:1965`) — zeroes elements where `i==j==k` or `a==b==c`.

Rank 4 has **no analogue**: every caller of `restore_restricted_t3_structure` is in
`tensor_backend.cpp`; `generated_arbitrary_runtime.cpp` never calls it.

**The sharpest single fact, and the reason S1 is not obviously guilty on its own:** the
hand-written branch calls the *same* `restore` on its own residual (`:2501`) and converges
correctly. So `restore` is not unconditionally wrong — it is wrong *for what the generated kernel
hands it*. That makes this a contract mismatch (kernel output convention vs restore's input
assumption), not a bug inside `restore`.

Both recorded symptoms are what such a mismatch looks like, and neither is what wrong equations
look like:

- **~45 % of elements unwritten, generated-live a strict subset of hand-live.** `purify` zeroes a
  diagonal set; step 1 spreads support across a permutation orbit. A kernel emitting a different
  canonical wedge than `restore` expects yields exactly a subset.
- **31 % sign flips on the shared elements** — the one feature the rank-3 doc records with no
  hypothesis. Step 2 is a signed operation (`x - mean(P3 x)`): an input already carrying some
  permutational structure can have its sign inverted where `mean(P3 x)` dominates. This is the
  first candidate explanation for the sign flips; it is a hypothesis to test, not a finding.

### S2 — the rank-3 harness (amplitude container, denominators, update path)

Rank 3 and rank 4 do not share a runtime. Rank 3: `RCCSDTAmplitudes` / `Tensor6D` through
`tensor_backend.cpp:2337`. Rank 4: `ArbitraryOrderRCCAmplitudes` / `TensorND` through
`generated_arbitrary_runtime.cpp` with `ArbitraryOrderDenominatorCache` and
`update_amplitudes_with_jacobi_diis`.

Two asymmetries make S2 independently suspicious rather than a fallback:

- **Input convention, not just output.** The `t3` handed to the generated kernel comes from a path
  that projects with `restore_restricted_t3_structure(triples.amplitudes.t3)` (`:2798`, `:2815`)
  before DIIS. So `restore` sits on **both** sides. If the generated kernel expects raw (unprojected)
  amplitudes while the hand-written one expects projected — or vice versa — the defect is on the
  input side and S1's output-side analysis would be chasing a symptom. **Test the input contract
  before the output one.**
- **`rebind_physicist` history.** Rank 3 was missing it entirely until `eb1c611`; the arbitrary path
  always had it (`generated_arbitrary_prepare.cpp:88`). One convention bridge was already found
  missing on exactly this surface — evidence the harness is where conventions get dropped, and a
  reason to check whether it is *fully* fixed rather than fixed for the blocks that were tested.

S1 and S2 are **not cleanly separable by inspection**: `restore` appears on both the input and
output side of the same loop. Step R1 exists to separate them by measurement.

## The investigation

Ordering rule: each step's verification is part of the step. The parity doc records five fixes that
passed a structural gate and made the physics worse — so every step below carries a numeric gate
from the start.

### R0 — re-establish the baseline (~S, do first)

The recorded probe numbers predate the accessor fix and an uncommitted working tree
(`tensor_backend.cpp` currently carries the P3 timing probe; `PLANCK_CC_INCLUDE_INTERMEDIATES`
is flipped OFF in the working tree vs ON in `build/CMakeCache.txt`).

Re-run the `PLANCK_CC_T3_DIFF=1` probe on `bh3_rccsdt_sto3g` and confirm the defect still reads
`max|gen-hand| ≈ 2.03e-02`, ~45 % missing, 31 % sign flips.

*Verify:* the three numbers reproduce, **and** the backend marker says `RCCSDT[OPT]` /
`kernels=ccgen-generated`. If they do not reproduce, stop — something else changed and the rest of
this scope is scoped against a stale measurement.

#### R0 finding: `build/` has no `CMAKE_BUILD_TYPE`, so the accessor fix is not in effect there

Found while starting the CH4 P2 run. `build/CMakeCache.txt` has `CMAKE_BUILD_TYPE` **empty**. Two
compounding consequences, both confirmed in a `sample(1)` stack profile of a running CH4 rank-4
solve:

1. **No `-DNDEBUG`, so the accessor bounds `assert` is live.** Every element access runs
   `detail::nd_index_valid` → `checked_product`, which walks a `std::vector<int>` per access. Those
   two symbols dominate the profile. This is precisely the per-access cost
   `CCGEN_TENSOR_ACCESSOR_FIX_SCOPE.md` removed — an empty build type silently reinstates it.
2. **The inlined accessors are not inlined.** `DYLD-STUB$$nd_flat_index` /
   `DYLD-STUB$$nd_index_valid` appear as real out-of-line PLT calls on the innermost loop.

The `-O1` pin on `generated_kernel_registry.cpp` (`CMakeLists.txt:402`) is deliberate and unrelated
— it stays; `-O3` on that TU costs 40+ min to compile.

**Consequence for this scope and for anything citing a CC timing number:** a run out of `build/` is
not the configuration any recorded CC measurement was taken in. Configure explicitly:

```bash
cmake -B <tree> -S . -DCMAKE_BUILD_TYPE=Release \
  -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON -DPLANCK_CC_INCLUDE_INTERMEDIATES=ON
```

Verify `-DNDEBUG` is present in `<tree>/CMakeFiles/hartree-fock.dir/flags.make` before trusting a
number. This is a specific instance of the standing rule in `CCGEN_KERNEL_SCALING_SCOPE.md` — never
compare against a stale or differently-configured `build/`.

A misread worth recording, since it cost a wrong inference: the CH4 process showed RSS ≈ 3.9 MB,
which was read as "it has not reached CC4 yet." It had — `compute_ccsdtq_quadruples_residual_part4`
was on the stack. CH4's T4 is 160,000 doubles ≈ 1.3 MB, so RSS carries no signal about CC progress
at this size. **Read the stack, not the footprint.**

### R1 — THE discriminating experiment: run rank 3 through the rank-4 harness (~M)

**This is the step that separates S1 from S2, and it should be run before any fix is attempted.**

`-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` emits `ccsdt_arbitrary_planck_generated.cpp` — the *same*
rank-3 equations against the *arbitrary-order* runtime (`ArbitraryOrderRCCAmplitudes`, no
`restore_restricted_t3_structure`, the path rank 4 uses and passes on). The registry already routes
rank 3 to `make_generated_ccsdt_kernels()` under that flag
(`generated_kernel_registry.cpp:~58`).

So the same equations can be run through both harnesses, isolating the harness as the only variable.

```bash
cmake -B /tmp/claude-501/r1arb -S . -DCMAKE_BUILD_TYPE=Release \
  -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON
make -C /tmp/claude-501/r1arb hartree-fock -j4
```

| outcome | reading |
|---|---|
| arbitrary rank-3 **correct**, `tensor_backend` rank-3 wrong | **S1/S2 confirmed, generator exonerated.** The defect is entirely in the rank-3 harness + `restore`. Proceed to R2. |
| **both wrong, identically** | The defect is upstream of both harnesses — but the term diff says the equations are shared with a passing rank 4, so this would point at something the *emit* does differently per rank (accessor form, kernel signature, block binding) rather than at the algebra. New surface; re-scope. |
| arbitrary rank-3 wrong in a *different* way | Two defects. Characterize separately; do not conflate. |

#### R1 — ANSWERED: row 1. The generated rank-3 kernel is CORRECT; the `tensor_backend` harness is wrong

CH4/STO-3G cart (`no=5 nv=4`, non-square, `nso=18 ndet=43758` so it clears the determinant
backstop). **The same `compute_ccsdt_triples_residual`, run through the two harnesses**, against
`pyscf.cc.rccsdt` as an external oracle:

| arm | harness | total / Eh | Δ vs PySCF |
|---|---|---|---|
| A | `tensor_backend` (**with** `restore_restricted_t3_structure`) | −39.8059200873 | **−7.56e-05** |
| B | arbitrary-order runtime (**no** `restore`) | −39.8058445091 | **+1.49e-08** |
| — | PySCF `rccsdt` | −39.8058445240 | — |

**Arm B agrees with PySCF to 1.5e-08 — a 5071× improvement over arm A, on identical generated
equations.** Arm B's residual 1.5e-08 is the same `|f_ov|`-scale Brillouin residue that parity P1
explained at rank 2 (PySCF carries the true-Fock `f_ov`; ccgen generates with
`canonical_fock=True`), i.e. arm B is correct to the precision the comparison can resolve.

Arm A is independently wrong on a *variational* check, which needs no Planck-side reference at all:
PySCF gives CCSDT − CCSDTQ = **+4.27e-05** (CCSDT recovers less correlation, as it must), yet arm A
lands **3.28e-05 below PySCF's CCSDTQ** — recovering more correlation than the higher method.
Planck's RHF sits ~7.5e-08 above PySCF's, three orders below the discrepancy, so the offset cannot
explain it; totals are compared throughout, per the parity doc's rule.

**Consequences:**

- **The generator, the equations, and the emit are exonerated at rank 3.** The 811 shared terms are
  correct — arm B proves it by executing exactly them and matching an independent code.
- **The defect is entirely in the rank-3 consumer**: `restore_restricted_t3_structure` and/or the
  `tensor_backend` harness around it. S1/S2 are the whole surface.
- `CCGEN_RANK3_TRIPLES_DEFECT.md`'s remaining-suspect list should be replaced: suspect #1 (spatial
  spin-adaptation lowering) is dead, and the defect title — "the generated rank-3 triples residual
  is wrong" — is itself wrong. The residual is right; what surrounds it is not.
- **The rank-parity hypothesis is dead**, and by a route the parity doc did not anticipate. Rank 3
  is not an odd-rank algebra failure: the same equations are correct under a different harness. The
  P2/P4 branches predicated on parity are moot. What remains true, and is now the sharper statement:
  *the arbitrary-order harness is correct at ranks 2, 3, and 4; only the rank-3 `tensor_backend`
  harness fails.*

Also settled, and it validated running R1 at all: the rank-3 doc's line *"Holds identically on both
emit paths (plain rank-3 and arbitrary-order)"* did **not** pre-empt this. `PLANCK_CC_T3_DIFF` exists
only in `tensor_backend.cpp`, so that claim compared the two emitted **TUs through one harness** —
never the arbitrary **runtime**. Both TUs were confirmed here to carry identical triples terms
(811/811, 0 differences either way), which is what makes the harness the only variable.

**A system-selection trap, paid for once here.** The first R1 attempt used LiH/STO-3G (small,
non-square, trusted gate value) and produced *no probe output at all* — `nso=12 ndet=495` routes to
`RCCSDT[DET-BACKSTOP]`, which never calls the generated kernel and converges correctly, looking like
a pass. Surveying in-tree STO-3G systems: LiH (`nso=12`) and H2O (`nso=14 ndet=1001`) are unusable;
BH3 clears the gate but is `no == nv == 4`, the square blind spot. **CH4/STO-3G is the only in-tree
system that is both non-square and reaches the generated rank-3 kernel.**

*Verify:* raw residual comparison at **fixed amplitudes**, not a converged energy — `restore` masks
raw error 11–29×. Compare against the hand-written residual exactly as `T3_DIFF` does.

**Caveat that must be controlled:** the rank-3 arbitrary companion is emitted *without* intermediate
builders (deliberately — the shape-named `build_W_*` symbols would collide with the ccsdtq TU; see
the comment in `generated_kernel_registry.cpp`). So it differs from the plain rank-3 TU in more than
its harness. Confirm the two TUs' residual **term sets** match before attributing any difference to
the harness — same normalized-term diff used above, which makes this cheap.

The rank-3 doc states the defect "holds identically on both emit paths (plain rank-3 and
arbitrary-order)". If that was measured **through `tensor_backend`** rather than through the
arbitrary-order runtime, it does not pre-empt R1. Establish which before spending the build — this
is a doc-reading step, not a measurement, and it may make R1 redundant or may confirm it is exactly
the missing experiment.

### R2 — pin the `restore` contract (~S) — **R1 implicated S1/S2; this is now the live step**

R1 narrowed the surface to the rank-3 consumer but did **not** separate S1 from S2 — `restore` sits
on both the input and output side of the `tensor_backend` loop, and arm B removed the whole harness,
not just `restore`. R2 and R3 split that.

Cheapest decisive variant, available because R1 established a correct oracle *inside Planck*: take
arm B's converged rank-3 amplitudes, evaluate the residual once in each harness, and diff. Arm A's
own `T3_DIFF` probe already does the generated-vs-hand comparison at fixed amplitudes; what is new
is that the correct answer is now known, so the probe's two sides can each be scored rather than
merely differenced.

#### R2 — ANSWERED: `restore` is implicated but is NOT the whole defect. Do not "fix" it by deletion

Arm C = arm A with the single `restore_restricted_t3_structure(triples_residual)` on the
**generated** branch skipped (`PLANCK_CC_T3_NO_RESTORE=1`, `tensor_backend.cpp:2491`; the
hand-written branch's own `restore` at `:2513` untouched, so that path stays byte-identical).

| arm | generated residual | converged? | Δ vs PySCF (total) |
|---|---|---|---|
| A | `restore` applied | yes | **−7.56e-05** |
| C | `restore` skipped | **NO** — stalls at `rms(R3)≈3.2e-04` | **+2.23e-05** |
| B | arbitrary harness (no `restore` anywhere) | yes | **+1.49e-08** |

**Removing `restore` cuts the error 3.4× and flips its sign, but destroys convergence.** That is not
a partial fix — it is evidence the double-symmetrization hypothesis is *incomplete*. Arm A overshoots
(more correlation than CCSDTQ, variationally impossible); arm C undershoots. The true answer sits
between them, so `restore` is applying roughly-but-not-exactly the wrong extra transform: a pure
double-application would leave a clean group-theoretic factor, and the residual +2.23e-05 says
something else is also mis-conventioned.

**Why arm C cannot converge — a coupling R1 could not see.** T3 does not only produce its own
residual; it feeds back into the singles/doubles residuals through
`add_dressed_triples_feedback_into_sd_residuals` (`tensor_backend.cpp:2291`), which is
**hand-written and consumes `amps.t3` in the compact one-representative convention**. Arm C leaves
the generated (fully-permuted) T3 unconverted, so that consumer is fed the wrong convention —
visible directly in the stall: `rms(R3)`, `rms(R1[T3])` and `rms(R2[T3])` all plateau together at
~1e-4 while `rms(SD)` stays small.

So the rank-3 `tensor_backend` harness has **two** convention boundaries, not one:

1. residual out of the generated kernel → `restore` (R2's line), and
2. T3 amplitudes → the hand-written SD feedback (`:2291`) and the amplitude projections at `:2803`,
   `:2810`, `:2826`.

Arm B is correct because the arbitrary harness crosses **neither** — it has no hand-written consumer
of T3 at all. That is why R1 showed a clean 5071× while R2 does not: R1 removed the whole boundary
set, R2 removed one of them.

**Consequence for the fix, and it changes its shape.** The right change is not deleting a line. It
is making the convention explicit at the boundary — one conversion between the generated
(fully-permuted) form and the compact form the hand-written consumers expect, applied at every
crossing — rather than a `restore` call that happens to suit one producer. Per the project's
standing rule: fix the mechanism, not the call site. A per-call-site patch would re-arm the trap at
the other crossings.

#### R2.1 + the missing control: the double-symmetrization story is WRONG. `restore` is innocent

Two runs killed it, and the second is the control that should have existed from the start.

| arm | residual producer | post-processing | converged? | Δ vs PySCF |
|---|---|---|---|---|
| A | generated | `restore` (default) | yes | −7.56e-05 |
| C | generated | none | **no** | +2.23e-05 |
| D | generated | compact = `p3_full`+`purify` (R2.1) | **no** | — |
| **E** | **hand-written** | **`restore` (default)** | **yes** | **+1.45e-08** |
| B | generated | arbitrary harness (none) | yes | +1.49e-08 |

**Arm E is the control: the hand-written tensor backend, same `restore`, same solver, same system —
and it lands at +1.45e-08, matching PySCF and agreeing with arm B to 4.0e-10.**

So `restore_restricted_t3_structure` composed with the `tensor_backend` rank-3 solver is **correct**.
It is not double-symmetrizing anything. R2's hypothesis — that the generated kernel's explicit
permutation expansion collides with `restore`'s permutation sum — is **refuted**: if it were true,
removing (arm C) or halving (arm D) the transform would have improved matters, and instead both
destroyed convergence while arm E sails through with the full transform intact.

The term-count asymmetry that motivated the hypothesis (824 generated accumulations vs 186
hand-written call sites, collapsing to 24 factor skeletons) is real but was **over-read**. A
permutation-expanded emission is not the same thing as a tensor that already carries the symmetry
`restore` builds; the modelling that "confirmed" it assumed `G == perm_sym(x)` exactly, which was
never measured on the actual residual.

**What the surface actually is, after arm E.** The defect is in the generated kernel's output *as
consumed by `tensor_backend`* — not in `restore`, and not in the equations (arm B proves those are
right). Since the same generated kernel is correct under the arbitrary harness, and `restore` is
correct under the hand-written producer, the mismatch is in what the two harnesses hand the kernel
or expect back: block conventions, amplitude layout, or the T3-feedback coupling at `:2291` — not
the symmetry transform.

#### The control was missing because the hand-written tensor backend has NO regression gate

Checked against `tests/regression_cases.json`: **every** rank-3 CC case asserts a determinant-space
marker in `contains`, and every CC test system sits under `choose_determinant_backstop`'s
`nso <= 16 && ndet <= 10000`:

| case | nso | ndet | path actually exercised |
|---|---|---|---|
| `h2_rccsdt_sto3g` | 4 | 6 | `Determinant-space prototype` |
| `lih_rccsdt_sto3g` | 12 | 495 | `Determinant-space prototype` |
| `water_rccsdt_sto3g` | 14 | 1001 | `Using the determinant-space CCSDT backstop` |
| `be_rccsdtq_sto3g` | 10 | 210 | backstop-eligible |

`water_rccsdt_sto3g` goes further and *asserts* the handoff string
`RCCSDT[TENSOR] : Using the determinant-space CCSDT backstop` — a gate pinning that the tensor path
**declines to run**, not that it is correct.

So the PySCF-validated CC suite validates the **determinant-space prototype**. The hand-written
tensor rank-3 solver, and `restore` acting on a residual, are unreachable from the entire suite. The
"hand-written is trusted" premise underpinning the original defect framing — and this investigation
until arm E — rested on a validation that does not cover the compared path.

Same shape as the defect that started all this: `compute_ccsdt_triples_residual` had no caller for
months. Here the tensor solver has a caller, but no test system large enough to reach it.
**Linkage is not execution — and a green suite is not coverage if every case takes the other branch.**

**Gate to add regardless of how the fix lands:** a rank-3 CC case with `nso > 16 || ndet > 10000` and
`no != nv`, pinned against PySCF `rccsdt`. `ch4_rccsdt_sto3g` (`nso=18 ndet=43758 no=5 nv=4`, PySCF
total **−39.8058445240**) is the smallest such system in-tree and is already written as
`tests/inputs/investigation/ch4_ccsdt.hfinp`. Without it, every result in this document is
unreachable from CI.

`restore_restricted_t3_structure` has an *implicit* input contract that is documented nowhere:
`apply_restricted_t3_permutation_symmetry` is non-idempotent (×6), and the rank-3 doc records this
as "compensated by the repeated-index pre-scaling". So the generated residual must arrive
**already pre-scaled** in a specific convention.

Determine what that convention is and whether the generated kernel satisfies it:

- Feed the *hand-written* residual through `restore` and confirm idempotency behavior matches the
  documented compensation.
- Feed the *generated* residual through the same and compare the discrepancy against the ×6 factor
  and the `mean(P3 x)` subtraction — the two signed operations that could produce the 31 % flips.

*Verify:* a named convention (stated as an assertion on the kernel's output), plus a numeric
before/after at fixed amplitudes showing the discrepancy closing. **A structural argument alone is
not a pass** — this is precisely where the five prior fixes failed.

### R3 — the input-side contract (~S, run alongside R2)

Independent of R2 and cheap: determine whether the generated kernel is being handed **projected**
or **raw** `t3`, and which it expects. `restore_restricted_t3_structure(triples.amplitudes.t3)` at
`:2798`/`:2815` projects amplitudes before DIIS, so both kernels see projected input in the solve
— but `T3_DIFF` evaluates at whatever amplitudes the probe injects.

*Verify:* evaluate both residuals at (a) raw and (b) projected `t3` and report all four numbers. If
the generated/hand agreement changes materially between (a) and (b), the defect is an input-contract
mismatch and R2's output-side analysis is chasing a symptom.

### R4 — verdict (~S)

A written statement naming the surface, the mechanism, and the numeric before/after. Then rewrite
`CCGEN_RANK3_TRIPLES_DEFECT.md` and this file into one architecture answer, per the docs/ rule.

## Gates to attach when the fix lands

- **Raw-residual gate at fixed amplitudes**, not converged energy.
- **Non-square system.** `bh3`/STO-3G is `no == nv == 4`; a wrongly-ordered read stays in bounds
  there. Use `nv != no` — `water_rccsdt_sto3g` is `no=5 nv=2`.
- **The rank-4 gate must stay green.** Both surfaces are rank-3-only by construction, so
  `be_rccsdtq_sto3g` should be untouched — if it moves, the fix reached further than intended.
- **A reachability assertion**, in the spirit of `GeneratedKernelsAreReachableTests`: the rank-3
  defect survived because `compute_ccsdt_triples_residual` had no caller. Any new contract needs a
  gate that fails when the contract is violated, verified by re-injecting the violation.

## Open input: the rank-4 anchor (now secondary)

**R1 demoted this.** The original narrowing leaned on rank 4 being correct, so the Be exactness
confound mattered. R1 replaced that inference with a direct measurement — the rank-3 equations are
correct because they reproduce PySCF through a harness that has no `restore`, which is an argument
that does not route through rank 4 at all. The elimination table no longer depends on the rank-4
anchor.

It is still worth closing for the parity doc's own sake (rank 4 off the exactness limit), and it
comes nearly free: R1's arm B continues into the rank-4 solve after seeding from its converged
rank-3, so that run **is** the P2 CH4 datapoint.

The CH4/STO-3G P2 run (`no=5 nv=4`, 5 distinct quadruples,
`tests/inputs/investigation/ch4_cc4_sto3g.hfinp`) is **in flight, on a Release tree** — a first
attempt out of `build/` was abandoned once the R0 finding above showed it was running with live
bounds asserts and un-inlined accessors. Target, from `CCGEN_RANK_PARITY_INVESTIGATION.md`:

```
RCCSDTQ (PySCF, cart) = -39.8058872460     <- compare Planck's TOTAL, not e_corr
```

**If CH4 deviates materially, this scope's premise weakens** — the 811 shared terms would no longer
be anchored to a verified-correct kernel, and the eliminated suspects come back. Do not treat the
elimination table as settled until CH4 lands.

## What NOT to do

- **Do not fix at one call site.** The conventions live in the emitter and the harness contract; the
  `rebind_physicist` fix was already this mistake once (fixed rank 3, left the contract implicit).
- **Do not gate on converged energy.** `restore` masks raw error 11–29×.
- **Do not conclude from a structural gate.** Five prior fixes passed one and degraded the energy.
- **Do not trust a probe number without checking the backend marker** — a build silently selects the
  hand-written backend and prints nothing.
- **Do not re-litigate the eliminated suspects** without new evidence; the term diff is
  reproducible and cheap to re-run.
- **Do not use a square test system for a new gate.**

## Reproducing the term diff

Group `acc +=` accumulations by enclosing `compute_*` function, normalize `amplitudes.` prefix and
`({...})` → `(...)`, then set-diff. Grouping by kernel is load-bearing — not doing it is what
produced the wrong "185 rank-3-only" reading.

```
rank-3 TU : build/generated/cc/ccsdt_planck_generated.cpp
rank-4 TU : build/generated/cc/ccsdtq_planck_generated.cpp
```

## Key code locations

| what | where |
|---|---|
| `restore_restricted_t3_structure` (S1) | `src/post_hf/cc/tensor_backend.cpp:1977` |
| its three sub-ops | `:1906` (perm sym, non-idempotent), `:1938` (P3 mean subtract), `:1965` (purify) |
| generated-vs-hand branch + `T3_DIFF` probe | `src/post_hf/cc/tensor_backend.cpp:2325` |
| hand-written branch, same `restore` (converges) | `:2501` |
| amplitude projection before DIIS (S2 input side) | `:2798`, `:2815` |
| rank-4 harness, never calls `restore` | `src/post_hf/cc/generated_arbitrary_runtime.cpp` |
| `rebind_physicist` | `src/post_hf/cc/generated_arbitrary_prepare.cpp:40` |
| rank-3 arbitrary companion routing (R1) | `src/post_hf/cc/generated_kernel_registry.cpp` |
| rank-4 gate | `be_rccsdtq_sto3g` in `tests/regression_cases.json` |
| CH4 P2 input | `tests/inputs/investigation/ch4_cc4_sto3g.hfinp` |
