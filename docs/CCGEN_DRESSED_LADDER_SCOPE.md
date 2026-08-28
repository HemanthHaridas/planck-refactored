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

Obstacle 2 is the load-bearing one. It is also exactly the mismatch that caused
the `-7.56e-05` defect, so it must be handled explicitly rather than by
convenience.

## Steps

Ordered so the cheapest step can kill the expensive ones.

### T1 — decide the arm-hosting question (~S, no code)

Two options, and T1 is choosing between them, not building:

| | host in `tensor_backend.cpp` | host in the arbitrary harness |
|---|---|---|
| hand-written arm | already local | needs a header + conversion |
| generated arm | needs the dressed arbitrary kernel reachable | already local |
| dressed vs undressed | one binary per arm either way | same |

**Prefer hosting in the arbitrary harness** and exporting the hand-written
kernel: it is the path that actually runs, so timings there are the ones that
matter, and the alternative resurrects the abandoned branch this whole problem
came from.

*Verify:* a written decision naming which file the probe lives in, and what has
to be exported. If exporting `build_dressed_triples_residual` drags in more than
its intermediates, stop and reconsider — that is a signal the seam is wrong.

### T2 — one fixture, three arms, provably identical amplitudes (~M)

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
