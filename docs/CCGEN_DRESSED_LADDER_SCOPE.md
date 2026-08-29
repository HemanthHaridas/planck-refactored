# How do we measure whether dressing fixes the generated kernel's SCALING?

**Scope for in-flight work.** Rewrite this into an answer once the measurement
lands — that is this directory's rule, and the exemption expires when the work
does.

`CCGEN_KERNEL_SCALING_SCOPE.md` measured the generated rank-3 triples kernel as
**21.8x -> 50.1x slower than hand-written, growing with size** — a scaling defect,
not a constant tax — and named two candidate fixes for the same hypothesis (H3,
n-ary contraction order). One is consuming the emitter's discarded
`_optimal_contraction_order`. The other, **derivation dressing**, has since been
wired and measured at 3.12x/3.61x end-to-end.

Both that document and `CCGEN_KERNEL_PERFORMANCE.md` say to settle which before
building the emitter change, because **the two fixes may overlap rather than
add**. This scopes that measurement.

## The question, stated so it can be answered wrong

Not *"is dressing faster"* — that is already measured. The question is:

> **Does dressing reduce the generated kernel's SCALING EXPONENTS, or only its
> CONSTANT?**

Those imply opposite next steps:

| dressed fit lands near | reading | consequence |
|---|---|---|
| `o^3.9 v^4.2` (the hand-written fit) | dressing fixed the contraction order | `_optimal_contraction_order` is **largely redundant** — do not build it |
| `o^4.9 v^4.5`, whole curve shifted down | dressing is a constant-factor win | H3 still open; the emitter change is **still the asymptotic fix** |
| between the two | they partly overlap | measure the residual gap before committing |

Two points cannot give exponents. This needs the six-point ladder.

## Why the existing probe cannot answer it

`PLANCK_CC_T3_TIME` times two arms — generated and hand-written — inside
`run_tensor_rccsdt_impl`'s `use_generated_kernels` branch
(`tensor_backend.cpp:2350`). **That branch no longer executes.** The rank-3
representation fix (`CCGEN_RANK3_KERNEL_AND_SOLVER.md`) rerouted `optimized`
away from it, correctly — the old solver's symmetry-packed representation is
incompatible with the dense one the kernels are emitted for, and converged to
`-7.56e-05`.

```
PLANCK_CC_ARBITRARY_LOWER_RANKS=OFF
  -> run_tensor_optimized_rccsdt returns std::unexpected(...), hard error
PLANCK_CC_ARBITRARY_LOWER_RANKS=ON
  -> routes to run_rccgen (the arbitrary-order harness), which has no probe
     (verified: grep PLANCK_CC_T3_TIME over rccgen.cpp,
      generated_arbitrary_runtime.cpp, solver_arbitrary.cpp -> no match)
```

Both branches were run, not inferred. The probe was left behind on the abandoned
path.

### A retracted claim, kept because the mistake is instructive

> An earlier revision of this document claimed the arbitrary-order companion TUs
> "stay undressed", concluded the dressed rank-3 kernel had **no reachable
> caller**, and cast doubt on W5's 3.12x/3.61x. **All of that was wrong.**

It rested on a stale `CMakeLists.txt` comment (describing a V1.3.1 suppression
that **V1.3.2 removed** by method-suffixing every builder) plus `build_W:0` in
two trees configured with `ARBITRARY_LOWER_RANKS=OFF` — trees where no companion
TU is emitted at all, so `0` was the expected result under either hypothesis and
the check had no discriminating power.

Measured directly, one method, one flag varied:

| TU | undressed | `--dressing derived` |
|---|---|---|
| `ccsd_planck_generated.cpp` (plain) | 0 `build_W` | **119** |
| `ccsd_arbitrary_planck_generated.cpp` (companion) | 0 `build_W` | **119** |

**Dressing reaches the arbitrary-order TU — the one the registry executes.**
`generate_planck_cc_kernels.py:179` passes `dressing` unconditionally;
`force_arbitrary` gates only `include_intermediates`. W5's numbers stand as
measured. The CMake comment is corrected and now carries the measurement inline.

*A comment is not a gate* — third instance in this subsystem, and the first where
a stale one **invented** a defect rather than hiding one. Verifying it cost one
generator invocation.

## Why three arms, not two

The natural fix — retarget the probe at dressed-vs-undressed — **loses what the
ladder was for.** That ratio has two generated numerators and no standard, so it
measures dressing's speedup while saying nothing about whether the scaling is now
*correct*. The hand-written kernel is not merely a baseline; it is the known-good
asymptotic reference, fitting `o^3.94 v^4.18` at 4.5% residual (textbook `o³v³`
output x one contracted index). Without it in the comparison there is nothing to
be correct *relative to*.

The alternative — reuse the hand-written column already recorded in
`CCGEN_KERNEL_SCALING_SCOPE.md` — assumes timings from a different binary and a
different code path are comparable to new ones. Plausible (it is an isolated
residual evaluation either way) but unverified, and this session has already
shown what an unchecked assumption costs.

**So: three arms, one binary, one fixture, one run.** Generated-dressed,
generated-undressed, hand-written. No cross-build comparison, and the standard
travels with the measurement.

## The two structural obstacles, found before scoping

Both are real and shape the design; neither was assumed.

**1. The hand-written kernel is file-local.** `build_dressed_triples_residual`
is declared in no header — `grep` over `src/post_hf/cc/*.h` returns nothing. It
lives in `tensor_backend.cpp` along with its intermediates
(`build_dressed_sd_intermediates`, `build_dressed_triples_intermediates`). The
arbitrary harness cannot call it as things stand.

**2. The two arms use different amplitude types.** The hand-written kernel takes
`RCCSDTAmplitudes` (fixed-rank `Tensor6D` members); the arbitrary harness uses
`ArbitraryOrderRCCAmplitudes` (a rank-indexed `TensorND` pack). No conversion
exists — `project_rccsd_warm_start_to_restricted` is RCCSD->RCCSDT, not
fixed->arbitrary. **The arms must be seeded from one source and converted, or
they are not evaluating the same amplitudes**, which would void the comparison
entirely.

**Both were resolved by T1, in the opposite direction to the one expected** —
obstacle 1 by *not* exporting the kernel (the probe is hosted where it already
lives), and obstacle 2 by the pre-existing `to_tensor_nd(const Tensor6D&)`, which
is layout-preserving because the two types already share an occ-first axis order.
See T1. They are left stated here because the reasoning that made them look
expensive is what selected the design, and because obstacle 2 is the same
representation mismatch that caused the `-7.56e-05` defect — the cheap conversion
does not make the agreement gate optional.

## Steps

Ordered so the cheapest step can kill the expensive ones.

### T1 — decide the arm-hosting question — **DONE (2026-08-28)**

**Decision: host the probe in `tensor_backend.cpp`, not the arbitrary harness.
This reverses the recommendation this scope was written with**, on evidence
gathered while doing T1. Nothing is exported and no existing signature changes.

The scope assumed exporting `build_dressed_triples_residual` would cost "a header
+ conversion". Measured, it costs far more, and the arbitrary-harness direction
costs far less than assumed.

**Why not export the hand-written kernel.** Its signature names four types, and
**three of them are also file-local to `tensor_backend.cpp`**:

| type | declared in |
|---|---|
| `ProductionSpinOrbitalChemistsSystem` | `tensor_backend.cpp` |
| `DressedTriplesIntermediates` | `tensor_backend.cpp` |
| `DressedSinglesDoublesIntermediates` | `tensor_backend.cpp` |
| `TensorTriplesWorkspace` | `tensor_backend.h` (already public) |

Exporting the kernel means moving three internal types into a header — a wide
diff, on types with existing in-file callers, to serve a diagnostic probe. That
is precisely the "drags in more than its intermediates" signal T1 was told to
stop on.

**Why the arbitrary harness comes to us instead.** Its entire surface is
*already public* in `generated_arbitrary_runtime.h`:
`evaluate_generated_arbitrary_order_residuals` (`:172`),
`ArbitraryOrderTensorCCState`, `GeneratedArbitraryOrderKernels`,
`rebind_physicist`, and — decisively — `to_tensor_nd` overloads for
`Tensor2D/4D/6D` (`:94-97`).

**This dissolves obstacle 2.** `to_tensor_nd(const Tensor6D&)` is a pure
reinterpretation: same `data` buffer, dims copied in order, **no permutation**
(`generated_arbitrary_runtime.cpp:117`). It is exact because the two layouts
already agree — `RCCSDTAmplitudes::t3` is documented `(i,j,k,a,b,c)` and
allocated `(n_occ x3, n_virt x3)` (`amplitudes.h:43`, `amplitudes.cpp:478`),
which is exactly `rank_dims`' occ-first order (`amplitudes.cpp:54-62`). **The
`(vir...,occ...)` transpose recorded in `CCGEN_SPIN_ADAPT_DEFAULT.md` is a
ccgen-Python-vs-C++ concern and does not apply between these two C++ types.**

So the conversion T2 was scoped to build already exists, is one call, and is
layout-preserving.

**Downstream-caller impact: none.** Verified by grep over `src/` and `tests/`:

- `build_dressed_triples_residual`, `build_dressed_triples_intermediates`,
  `build_dressed_sd_intermediates` — **no callers outside `tensor_backend.cpp`**.
  Nothing moves, so nothing can break.
- `to_tensor_nd` — **no callers outside its own definition file**. Consuming it
  adds a first caller rather than changing behaviour for an existing one.
- `evaluate_generated_arbitrary_order_residuals` and `rebind_physicist` are
  already public and already called by `rccgen.cpp`; the probe becomes an
  additional caller of an unchanged signature.

The one direction that does **not** exist is `TensorND -> Tensor6D`. The probe
does not need it: it seeds *from* the hand-written amplitudes and converts
forward, so all three arms read one source.

*Verified:* the decision names its file (`tensor_backend.cpp`), exports nothing,
and every symbol it consumes is either already public or has no external callers.

**Consequence for T2**, which should be read with this: the agreement gate is
unchanged and still mandatory, but its risk profile drops sharply — a
layout-preserving conversion between two types that already share an axis order
is far less likely to be wrong than the transpose the scope anticipated. Keep the
gate; the reason it exists is that this codebase has twice timed two different
equations, and a cheap gate against a now-unlikely failure is still worth its
cost.

### T2 — one fixture, three arms, provably identical amplitudes — **BUILT; GATE RED**

The probe is landed as `PLANCK_CC_T3_LADDER` (N repeats, inert when unset), hosted
in `tensor_backend.cpp` per T1 and placed OUTSIDE `if (use_generated_kernels)` so
it runs on the path that executes. Pure insertion; the probe-unset energy is
bit-identical.

**Its gate is red, and that is the deliverable working.** Localizing the mismatch
is T2.1-T2.6 below.

Seed from **one** `ArbitraryOrderRCCAmplitudes` (the `PLANCK_CC_FIXTURE_DIR`
path already does this), convert to `RCCSDTAmplitudes` for the hand-written arm,
and time all three from that single source.

*Verify — and this gate comes BEFORE any timing:* evaluate all three residuals
once and assert the **generated-dressed and generated-undressed agree to ~1e-12**
elementwise, and that the hand-written agrees with them to the same tolerance
after the layout transpose. A timing comparison across arms that are not
evaluating the same equation is worthless, and this codebase has produced exactly
that failure twice. **If the conversion is wrong, this gate fires and the timings
are never taken.**

Note ccgen amplitudes are `(vir...,occ...)` while C++ `rank_dims` is
`(occ...,virt...)` — the transpose is real and recorded in
`CCGEN_SPIN_ADAPT_DEFAULT.md`.

### T2.1-T2.7 — localize the frame mismatch — **RESOLVED (2026-08-29): no residual-level gate is achievable; see T2.5**

T2's gate is **red, and correctly so**. Both arms are individually correct — each
converges to `E_corr = -0.0791116825` on CH4, identical to ten digits and 1.4e-08
from PySCF — so this is a representation mismatch, not a defect in either kernel.
**Do not close it by loosening the tolerance.**

What is already established, by measurement rather than inspection:

| framing | `rel` on CH4 |
|---|---|
| restore both arms | **3.6e-02** (closest) |
| restore hand only | 8.4e-01 |
| restore neither | 1.0e+00 |

and, by grep: the arbitrary-order harness calls `restore_restricted_t3_structure`
**zero times** (`solver_arbitrary.cpp`, `generated_arbitrary_runtime.cpp`), while
the hand-written solver calls it before consuming its residual
(`tensor_backend.cpp:~2714`). Two self-consistent conventions.

That restore-both is *close but not equal* is the key datum: the permutation-orbit
convention is **part** of the difference and not all of it. Something else remains.

**Method rule for this sub-investigation, earned twice already.** Three framings
were tried by reading code and reasoning; all three missed. The rank-3
investigation records five such hypotheses, all wrong, with every correct result
coming from direct comparison. **From here, measure first and hypothesize second.**

#### T2.1 + T2.2 — **DONE (2026-08-29). Neither branch; the framing was wrong.**

T2.1 added an opt-in dual dump (`PLANCK_CC_T3_LADDER_DUMP=<dir>`) writing
`r3_gen_raw`, `r3_hand_raw` and `r3_hand_restored` in the same format
`rccgen.cpp`'s R4.2c dump uses, so all three are directly comparable.

T2.2 then answered the branch question, and the answer is **neither**:

| tensor (CH4, n=8000) | max\|v\| |
|---|---|
| `r3_gen_raw` | 1.758e-03 |
| `r3_hand_raw` | 7.001e-03 |
| `r3_hand_restored` | **3.556e-08** |

**`restore` annihilates the hand-written residual — by a factor of 2.0e+05.**
So the earlier "restore both -> rel = 3.6e-02, the closest framing" was not close
at all: it was comparing the generated arm against a **near-zero tensor**, and the
small number was the generated arm's own magnitude, not agreement. That reading is
**refuted**, exactly as T2.2 was placed to do.

**Why, mechanically.** `restore_restricted_t3_structure` is three stages, and the
second is a *projector that removes what the first creates*:

| stage | what it does | effect here |
|---|---|---|
| `apply_restricted_t3_permutation_symmetry` | unnormalized sum over the 6 SIMULTANEOUS occ+virt permutations | **x6** — output is fully symmetric under that permutation |
| `apply_restricted_t3_p3_full` | `x - (mean over the 6 VIRT permutations)` | **annihilates** it: for a simultaneously-symmetric tensor that mean is itself |
| `purify_restricted_t3` | zeroes `i==j==k` / `a==b==c` | negligible |

Reproduced independently in Python from the dumps —
`7.00e-03 -> 4.20e-02 -> 3.56e-08`, matching the C++ to all printed digits, so
this is the mechanism and not a coincidence of one build.

**What this means — and this was ALREADY ESTABLISHED, not discovered here.**
`restore` is not a symmetrizer that maps a residual into a comparable frame; it is
only meaningful in its own solver, applied to a wedge-packed amplitude carrying
full permutational symmetry. `CCGEN_RANK3_KERNEL_AND_SOLVER.md` states it
outright — *"the wedge packing and `restore` are one coupled convention, not two
independent choices"* (:21-24), with the two representations tabulated at :14-15
and the coupling verified by measurement (deleting either half diverges).

**The measurement above re-derives a documented result.** Its only new content is
the *magnitude* (2.0e+05) and the *stage* (`p3_full` undoes
`permutation_symmetry`), which pin the mechanism numerically. The conclusion was
already in the tree, and consulting that document first would have skipped all
three failed framings — the retired dressed-operator investigation had paid for
this answer already.

That is the second time in this session that an existing record would have
prevented an investigation (the first being the stale CMake comment, which caused
one in the opposite direction). **Read the retired investigations before
re-measuring their subject.**

The live comparison is therefore `r3_gen_raw` vs `r3_hand_raw`, with **neither**
restored: elementwise max-diff 7.08e-03 against a hand magnitude of 7.00e-03, and
multiset max-diff 5.24e-03. Since the multisets do **not** match, T2.3/T2.4 (the
permutation branch) are **dead** — this is not a relabelling.

#### T2.3 / T2.4 — **DROPPED.** T2.2's multiset test refutes the permutation hypothesis, and the `restore` decomposition T2.4 would have performed is answered above.

#### T2.5 — **DONE (2026-08-29). The arms are not comparable per-rank at fixed amplitudes.**

Measured on CH4 by evaluating the generated arm's rank-1 and rank-2 residuals
alongside rank 3, against the hand-written `residuals.r1`/`.r2` taken **after**
`add_dressed_triples_feedback_into_sd_residuals` (probe at :2332, feedback at
:2298), so the T3->SD feedback is folded into both sides:

| iteration | rank 1 | rank 2 | rank 3 |
|---|---|---|---|
| 1 | rel 9.4e-06, but `max\|hand\|` = **7.8e-11** | rel 9.0e-08, `max\|hand\|` = 4.2e-09 | rel 8.4e-01 |
| 2 | **rel 7.0e-01** | **rel 6.5e-01** | rel 9.9e-01 |
| 3 | **rel 2.8e+00** | **rel 1.1e+00** | rel 1.0e+00 |

**Iteration 1's agreement is vacuous** — both arms are at ~1e-11, i.e. zero. That
is the degenerate-probe trap recorded in `CCGEN_UNRESTRICTED_CC.md` ("a probe
whose output is degenerate proves nothing"), and reading it as agreement would
have been a false green.

From iteration 2 the arms disagree **at every rank**, at the same order as rank 3.

**So the disagreement is not a rank-3 property, and not a T3->SD partition** —
which is consistent with `CCGEN_RANK3_KERNEL_AND_SOLVER.md` having already
refuted that hypothesis (:75-84, "overshoots 2.5x the other way"). The
hypothesis this step existed to test is dead by its own measurement as well as by
the prior record.

**The conclusion, and it is a finding rather than a defect.** The two arms are
distinct solvers with distinct amplitude representations. Feeding the generated
kernel the hand-written solver's mid-iteration amplitudes does not put the two in
a common frame — it evaluates one solver's kernel at another solver's iterate.
Both remain individually correct (each converges to `E_corr = -0.0791116825`,
matching PySCF to 1.4e-08); what does not exist is a shared intermediate state at
which their residuals are elementwise comparable.

**A residual-level agreement gate between these two arms is therefore not
achievable**, and T2's original design — three arms compared elementwise before
timing — cannot be completed as specified. That is worth stating plainly rather
than engineering around: it took four framings and two refuted hypotheses to
establish, and the next person will otherwise try the same four.

#### T2.6 — **SUPERSEDED.** The gate cannot be made green; see T2.5. What replaces it is below.

#### T2.7 — the fallback, which was planned for this outcome (~M) — **NEXT**

Time the **whole residual evaluation per arm**, not a per-rank slice, and validate
by **converged energy** rather than by elementwise residual agreement:

- **arm A** hand-written: the existing `tensor` backend, timed per iteration.
- **arm B/C** generated undressed / dressed: the arbitrary harness, timed per
  iteration, selected by `PLANCK_CC_DRESSING` at configure time.

*Verify:* each arm converges to `-0.0791116825` on CH4 (they already do -- this is
the check that the arms are the same calculation), and report seconds per
iteration. That is a legitimate comparison of two correct solvers doing the same
job, which is what the ladder question actually asks.

**What this costs.** The per-iteration time includes each solver's own overhead,
not the isolated triples kernel. The hand-written arm builds intermediates the
generated one does not; the generated arm evaluates all ranks where the
hand-written path evaluates r1/r2 from cheap dressed intermediates. So the
resulting ratio is **not** comparable to the original ladder's 21.8x-50.1x, which
timed isolated residual evaluations. Report it as its own quantity and do not
combine the two.

**What survives, stated more carefully than the first draft of this section.**

A B-vs-C fit answers the LITERAL question -- did dressing move the exponents or
only the prefactor -- because both arms are measured identically and differ only
in `--dressing`. That much needs no arm A.

**It does not tell you whether the dressed exponents are GOOD**, and that is the
decision this ladder exists to inform. If dressing gives `o^4.4 v^4.3`, is that
fixed? Only the hand-written `o^3.94 v^4.18` says. Without it, a partial
improvement and a complete one look identical -- and they imply opposite answers
on whether to build `_optimal_contraction_order`. So arm A is **not** optional for
the decision; an earlier revision of this section said it was, and that was wrong.

**A second caveat, which applies to B and C equally.** T2.7 times per-ITERATION
solver work, not an isolated residual. Each arm carries its own overhead -- the
hand-written path builds dressed intermediates, the generated one evaluates every
rank -- and that overhead has its own `o,v` scaling sitting inside the
measurement. So these exponents describe *"solver iteration"*, not *"triples
kernel"*, and they are **not** the same quantity the original six-point ladder
fitted. B-vs-C remains a fair comparison (identical overhead on both sides), but
the absolute exponents from T2.7 must not be quoted against the original ladder's
`o^4.87 v^4.52`.

**Arm A is still available, and should be measured.** T2.5 killed the elementwise
residual GATE between arms, not the timing of arm A: `PLANCK_RCCSDT_BACKEND=tensor`
runs the hand-written solver end to end and its iterations can be timed exactly as
B and C are. What changes is the validation -- from "residuals agree elementwise"
to "all three converge to `-0.0791116825`", which is the check that the arms are
the same calculation. Weaker, and sufficient.

### T3 — report shape (~S)

One line per invocation, extending the existing `RCCSDT[T3-TIME]` format so the
ladder harness needs no reparsing:

```
no= nv= o/v= t3=MiB reps= gen_dressed= gen_undressed= hand= ints=
```

`t3` MiB stays: it is what makes an H1 cache transition **visible rather than
inferred**, and the whole reachable ladder sits under 0.85 MiB (inside L2), so H1
remains untestable here by construction — record it so that stays obvious.

Keep the existing fairness convention and its reason: **intermediates are built
once, outside the timed loop**, because the generated kernel builds none and
charging them per-repeat would overstate the hand-written arm. Report them
separately as `ints=`.

*Verify:* the line parses with the existing harness; `reps >= 3` and the spread
is reported, not just the mean.

### T4 — run the six points, both dressing arms (~M)

`bh3_sto3g`, `ch4_sto3g`, `hf_631g`, `h2o_631g`, `bh3_631g`, `c2h4_sto3g` —
spanning `o/v` 0.36-1.33 and `o` 4-8. Inputs are already written (see below).

*Verify:* per point, energies identical across arms. Then fit `o^a v^b` for each
of the three arms and report **leave-one-out on every fit**, not the residual.
The residual is what made two earlier drafts of this ladder quote wrong exponents
— three of four points once shared `o=5`, so least squares had nothing to
separate and loaded all divergence onto `o`, and the 6.5% residual looked
reassuring *precisely because* it was overfitting a nearly-fixed variable.

### T5 — answer the question, and only that (~S)

Compare the dressed fit against the two references and write the outcome into
this document as an answer, per the table at the top.

*Verify:* the conclusion names which of the three rows it is, and the
leave-one-out spread on the dressed `o` exponent. If that spread does not hold
its sign, **the ladder has not answered the question** and the honest report is
"insufficient", not a number.

## What this must not do

- **Do not combine these numbers with W5's 3.12x/3.61x.** Those are end-to-end
  solve times on two systems, one off this ladder; these are isolated residual
  evaluations. Combining them into one ratio would be wrong.
- **Do not quote a ratio without saying which pair.** Three arms means three
  possible ratios and they answer different questions.
- **Do not extrapolate past the measured range.** An earlier draft quoted ~69x at
  `o=10 v=40` off a four-point fit with endpoint-sensitive exponents; that is not
  supportable and was withdrawn.
- **Do not skip T2's agreement gate to get to timings faster.** It is the only
  thing standing between this and timing two different equations.

## What is already set up

- Two configured trees differing in exactly one flag, cache-diffed:
  `build-ladder-{undressed,dressed}` (`MAXORDER=3`, `SPIN_ADAPT=ON`,
  `DRESSING=derived`), both building clean. **Both have
  `ARBITRARY_LOWER_RANKS=OFF`** and so emit no companion TU — add that flag
  before using them for the real measurement.
- Six ladder inputs, in the scratch `ladder/` directory. The `ch4` copy preserves
  its load-bearing `use_diis .true.` (`CCGEN_SPIN_ADAPT_DEFAULT.md` records that
  the hand-written tensor solver diverges without DIIS on that system).
- A harness that reports *why* a point yields no timing rather than dropping it —
  which is how the probe blocker surfaced instead of a five-point ladder with one
  unexplained gap.

## Constraints that still hold

- `PLANCK_RCCSDT_BACKEND=tensor` is the **hand-written** path; only `optimized`
  selects generated kernels. **Check the backend marker before believing any
  number** — that rule caught the first dead end here.
- The determinant backstop (`nso <= 16 && ndet <= 10000`) binds the hand-written
  arm only; all six ladder points clear it regardless.
- Build with `make -j4`; these TUs are large enough that a full-width build is
  disruptive, and the dressed CCSDTQ TU is 13 MB.
- Always set an explicit `CMAKE_BUILD_TYPE` — an empty one drops `-DNDEBUG`,
  re-enables the CC bounds asserts, and makes every timing meaningless.

## Key code locations

| what | where |
|---|---|
| the orphaned two-arm probe | `PLANCK_CC_T3_TIME`, `src/post_hf/cc/tensor_backend.cpp:2350` |
| the seam to build on (seed, evaluate once, stop) | `PLANCK_CC_FIXTURE_DIR`, `src/post_hf/cc/rccgen.cpp:88` |
| generated residual evaluation (the path that runs) | `evaluate_generated_arbitrary_order_residuals`, `generated_arbitrary_runtime.cpp:173` |
| hand-written kernel (file-local — obstacle 1) | `build_dressed_triples_residual`, `tensor_backend.cpp` |
| the two amplitude types (obstacle 2) | `RCCSDTAmplitudes` / `ArbitraryOrderRCCAmplitudes`, `amplitudes.h:39,83` |
| the reroute that orphaned the probe | `run_tensor_optimized_rccsdt`, `tensor_backend.cpp` |
| the ladder this extends | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
