# How do we measure whether dressing fixes the generated kernel's SCALING?

**Scope for in-flight work.** Rewrite into an answer once the measurement lands —
this directory's rule, and the exemption expires when the work does.

`CCGEN_KERNEL_SCALING_SCOPE.md` measured the generated rank-3 triples kernel at
**21.8x -> 50.1x slower than hand-written, growing with size** — a scaling defect,
not a constant tax — and named two candidate fixes for the same hypothesis (H3,
n-ary contraction order): consuming the emitter's discarded
`_optimal_contraction_order`, or **derivation dressing**, which has since been
wired and measured at 3.12x/3.61x end-to-end. Both that document and
`CCGEN_KERNEL_PERFORMANCE.md` say to settle which before building the emitter
change, because **the two may overlap rather than add**.

## The question

Not *"is dressing faster"* — that is measured. The question is:

> **Does dressing reduce the generated kernel's SCALING EXPONENTS, or only its
> CONSTANT?**

| dressed fit lands near | reading | consequence |
|---|---|---|
| `o^3.9 v^4.2` (the hand-written fit) | dressing fixed the contraction order | `_optimal_contraction_order` is **largely redundant** |
| `o^4.9 v^4.5`, curve shifted down | dressing is a constant-factor win | H3 open; the emitter change is **still the asymptotic fix** |
| between the two | partial overlap | measure the residual gap before committing |

Two points cannot give exponents; this needs the six-point ladder.

**Three arms, because the middle row is not self-interpreting.** A dressed
`o^4.4 v^4.3` is only meaningful against the hand-written `o^3.94 v^4.18` (4.5%
residual, textbook `o³v³` × one contracted index). Without that standard, a
partial fix and a complete one look identical while implying opposite answers.

```
arm A   hand-written        PLANCK_RCCSDT_BACKEND=tensor
arm B   generated undressed  PLANCK_RCCSDT_BACKEND=optimized, DRESS_OPERATORS=OFF
arm C   generated dressed    PLANCK_RCCSDT_BACKEND=optimized, DRESS_OPERATORS=ON
```

B-vs-C answers the literal question (identical measurement, one flag apart);
A calibrates whether the answer is good.

## What is settled

### The probe had to be rebuilt, not retargeted

`PLANCK_CC_T3_TIME` sits in `run_tensor_rccsdt_impl`'s `use_generated_kernels`
branch (`tensor_backend.cpp:2350`), which the rank-3 representation fix rerouted
away from. `ARBITRARY_LOWER_RANKS=OFF` makes `optimized` hard-error;
`ON` routes to `run_rccgen`, which has no probe. Both branches run, not inferred.

Replaced by `PLANCK_CC_T3_LADDER` (N repeats, inert when unset), hosted in
`tensor_backend.cpp` and placed **outside** that branch. **T1 decided the host
against the original recommendation:** exporting `build_dressed_triples_residual`
would drag three file-local types into a header
(`ProductionSpinOrbitalChemistsSystem`, `DressedTriplesIntermediates`,
`DressedSinglesDoublesIntermediates`), while the arbitrary harness is *already*
fully public — including `to_tensor_nd`, a layout-preserving `Tensor6D ->
TensorND` conversion (both types are occ-first, so no transpose; the
`(vir...,occ...)` transpose in `CCGEN_SPIN_ADAPT_DEFAULT.md` is a
ccgen-Python-vs-C++ concern and does not apply here). Nothing is exported and no
existing signature changes.

### There is no residual-level agreement gate between arms — and this is the load-bearing result

T2 was specified as: seed one amplitude set, evaluate all three arms, assert
elementwise agreement to ~1e-12, *then* time. **That cannot be done.** Four
framings were tried and all failed:

| framing | `rel` on CH4 |
|---|---|
| restore both arms | 3.6e-02 |
| restore hand only | 8.4e-01 |
| restore neither | 1.0e+00 |
| per-rank (r1/r2/r3 separately) | 0.65 – 2.76 at every rank |

**Why, established by dumping both arms elementwise** (`PLANCK_CC_T3_LADDER_DUMP=<dir>`,
matching `rccgen.cpp`'s R4.2c format):

- **`restore` annihilates the hand-written residual by 2.0e+05** (7.00e-03 ->
  3.56e-08). Its stage 1 sums the 6 simultaneous occ+virt permutations (×6, output
  fully symmetric); stage 2 `apply_restricted_t3_p3_full` subtracts the
  virt-permutation mean, which for such a tensor *is* that tensor. Reproduced
  independently in Python from the dumps, matching the C++ to all printed digits.
  So "restore both -> 3.6e-02" was comparing against a near-zero tensor; the small
  number was the generated arm's own magnitude. **Refuted, not close.**
- **This was already known.** `CCGEN_RANK3_KERNEL_AND_SOLVER.md:21-24` states the
  wedge packing and `restore` are *one coupled convention*; `restore` is meaningful
  only on a wedge-packed **amplitude** inside its own solver. Applying it to a raw
  **residual** is a category error, and all four framings did exactly that. The only
  new content here is the magnitude and the stage.
- **The multisets of |value| do not match** (5.24e-03), so it is not an index
  permutation either.
- **The disagreement is at every rank**, not confined to rank 3 (rel 0.65–2.76 from
  iteration 2 onward, with the hand-written r1/r2 taken *after*
  `add_dressed_triples_feedback_into_sd_residuals`, so T3->SD feedback is folded
  into both sides). Iteration 1 appears to agree at 1e-16 but **both arms are at
  ~1e-11 there — vacuous**, the degenerate-probe trap `CCGEN_UNRESTRICTED_CC.md`
  records.

**The conclusion is a finding, not a defect.** The arms are distinct solvers with
distinct amplitude representations. Feeding the generated kernel the hand-written
solver's mid-iteration amplitudes does not put them in a common frame — it
evaluates one solver's kernel at another solver's iterate. Both are individually
correct: each converges to `E_corr = -0.0791116825` on CH4, matching PySCF to
1.4e-08. **There is no shared intermediate state at which their residuals are
elementwise comparable.**

### Hypotheses already dead — do not re-enter

`CCGEN_RANK3_KERNEL_AND_SOLVER.md:75-84` tested and rejected these during the
retired dressed-operator work, *before* this scope existed:

| hypothesis | verdict |
|---|---|
| double symmetrization (`restore` applied twice) | No — removing/halving made it worse |
| pure double count of the T3->SD feedback | No — overshoots `+1.90e-04`, 2.5x worse |
| stride mismatch (spin-orbital vs spatial extents) | No |
| unique-triangle DIIS pack/unpack lossy | Partly — half the coupled convention, not the discriminator |
| block convention (`rebind_physicist`) | No — all seven ERI blocks bitwise identical |

Two of these were re-derived here at real cost. **Read the retired investigations
before re-measuring their subject.**

## What remains: T6 — time whole iterations, validate by energy

The elementwise gate is replaced, not repaired.

### T6.1 — time one iteration per arm (~M)

Time the **whole residual evaluation per iteration** in each of the three arms,
rather than an isolated triples slice. Arm A via `PLANCK_RCCSDT_BACKEND=tensor`,
arms B/C via `optimized`, B and C differing only in a configure-time flag.

*Verify:* each arm converges to `-0.0791116825` on CH4. That is the replacement
validation — it establishes the arms are the same calculation, which is what the
elementwise gate was for. Weaker, and sufficient.

### T6.2 — run the six points (~M)

`bh3_sto3g`, `ch4_sto3g`, `hf_631g`, `h2o_631g`, `bh3_631g`, `c2h4_sto3g` —
spanning `o/v` 0.36-1.33 and `o` 4-8. Inputs are written (see below).

*Verify:* per point, all three arms converge to the same energy. Then fit
`o^a v^b` per arm and report **leave-one-out on every fit**, never the residual —
the residual is what made two earlier drafts of this ladder quote wrong exponents
(three of four points once shared `o=5`, so least squares loaded all divergence
onto `o` and the 6.5% residual looked reassuring *because* it was overfitting).

### T6.3 — answer, and only that (~S)

Compare the arm-C fit against arm B (did dressing move the exponents?) and against
arm A (are the dressed exponents good?), and write the outcome in as an answer per
the table at the top.

*Verify:* the conclusion names which row it is, plus the leave-one-out spread on
the arm-C `o` exponent. **If that spread does not hold its sign, the honest report
is "insufficient", not a number.**

## What this must not do

- **Do not quote T6 exponents against the original ladder's `o^4.87 v^4.52`.** T6
  times per-*iteration* solver work, so each arm's own overhead is inside the
  measurement — the hand-written path builds intermediates the generated one does
  not; the generated path evaluates every rank where the hand-written builds r1/r2
  from cheap dressed intermediates. That overhead has its own `o,v` scaling. These
  exponents describe *"solver iteration"*, not *"triples kernel"*. B-vs-C stays
  fair (identical overhead both sides); the absolute numbers are a different
  quantity.
- **Do not combine with W5's 3.12x/3.61x.** Those are end-to-end solve times on two
  systems, one off this ladder.
- **Do not quote a ratio without naming which pair.** Three arms, three ratios,
  three different questions.
- **Do not extrapolate past the measured range.** An earlier draft quoted ~69x at
  `o=10 v=40` off a four-point fit with endpoint-sensitive exponents; withdrawn.
- **Do not restore either arm's residual.** See above — it is a category error, and
  it cost four framings to establish.

## What is set up

- `PLANCK_CC_T3_LADDER=N` — the three-arm probe, inert when unset, energies
  bit-identical with it off.
- `PLANCK_CC_T3_LADDER_DUMP=<dir>` — elementwise dumps (`r3_gen_raw`,
  `r3_hand_raw`, `r3_hand_restored`) in `rccgen.cpp`'s R4.2c format.
- `build-ladder-{undressed,dressed}` — cache-diffed to exactly one differing flag,
  `MAXORDER=3 SPIN_ADAPT=ON ARBITRARY_LOWER_RANKS=ON DRESSING=derived`.
- Six ladder inputs in the scratch `ladder/` directory. The `ch4` copy preserves
  its load-bearing `use_diis .true.` — the hand-written tensor solver diverges
  without DIIS on that system (`CCGEN_SPIN_ADAPT_DEFAULT.md`).

## Constraints

- `PLANCK_RCCSDT_BACKEND=tensor` is the **hand-written** path; only `optimized`
  selects generated kernels. **Check the backend marker before believing any
  number** — that rule caught the first dead end here.
- `make_generated_rcc_kernels` floors at rank 4 without
  `ARBITRARY_LOWER_RANKS=ON`; rank 3 needs it or the probe reports
  `below the generated floor 4`.
- The determinant backstop (`nso <= 16 && ndet <= 10000`) binds the hand-written
  arm only; all six ladder points clear it regardless.
- `bh3` is `no == nv == 4`, where a wrong axis order stays in bounds and can agree
  by accident. Never validate on `bh3` alone.
- `make -j4`; always an explicit `CMAKE_BUILD_TYPE` (an empty one drops `-DNDEBUG`,
  re-enables the CC bounds asserts, and makes every timing meaningless).

## A retracted claim, kept because the mistake is instructive

An earlier revision claimed the arbitrary-order companion TUs "stay undressed",
concluded the dressed rank-3 kernel had **no reachable caller**, and cast doubt on
W5's 3.12x/3.61x. **All of that was wrong.** It rested on a stale `CMakeLists.txt`
comment (describing a V1.3.1 suppression that V1.3.2 removed) plus `build_W:0` in
two trees configured with `ARBITRARY_LOWER_RANKS=OFF` — where no companion TU is
emitted at all, so `0` was the expected result under either hypothesis and the
check had no discriminating power.

Measured directly, one method, one flag varied: the companion goes **0 -> 119**
`build_W` call sites under `--dressing derived`, same as the plain TU.
`generate_planck_cc_kernels.py:179` passes `dressing` unconditionally.
**W5's numbers stand as measured.** The CMake comment is corrected and now carries
the measurement inline.

*A comment is not a gate* — and this is the case where a stale one **invented** a
defect rather than hiding one.

## Key code locations

| what | where |
|---|---|
| the three-arm probe | `PLANCK_CC_T3_LADDER`, `src/post_hf/cc/tensor_backend.cpp` |
| the orphaned two-arm probe it replaces | `PLANCK_CC_T3_TIME`, same file `:2350` |
| generated residual evaluation (the path that runs) | `evaluate_generated_arbitrary_order_residuals`, `generated_arbitrary_runtime.cpp:173` |
| hand-written kernel (file-local by design) | `build_dressed_triples_residual`, `tensor_backend.cpp` |
| the layout-preserving conversion | `to_tensor_nd`, `generated_arbitrary_runtime.h:94-97` |
| `restore`, and why it must not be applied here | `restore_restricted_t3_structure`, `tensor_backend.cpp:~1981` |
| the reroute that orphaned the old probe | `run_tensor_optimized_rccsdt`, same file |
| the ladder this extends | `docs/CCGEN_KERNEL_SCALING_SCOPE.md` |
| hypotheses already refuted | `docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md:75-84` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
