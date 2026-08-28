# Why can't the scaling ladder be re-run under `--dressing derived`?

**Because `PLANCK_CC_T3_TIME` is on a code path that no longer executes.** That
is the whole blocker, and it is a probe-placement problem, not an architectural
one. The dressed kernel itself runs fine.

`CCGEN_KERNEL_SCALING_SCOPE.md` and `CCGEN_KERNEL_PERFORMANCE.md` both recommend
re-running the six-point rank-3 ladder under `--dressing derived` **before**
consuming `_optimal_contraction_order`, since the two fixes may overlap. That
recommendation stands and is still worth doing; this note records why the
obvious way to do it does not work, and what to fix first.

## The blocker

The probe lives inside `run_tensor_rccsdt_impl`'s `if (use_generated_kernels)`
branch (`tensor_backend.cpp:2350`). Reaching that branch needs
`PLANCK_RCCSDT_BACKEND=optimized`. But `optimized` no longer runs that code:

```
PLANCK_CC_ARBITRARY_LOWER_RANKS=OFF
  -> run_tensor_optimized_rccsdt returns std::unexpected(...)
     "the generated rank-3 CCSDT kernel runs only in the arbitrary-order harness"
  -> hard error, no timing

PLANCK_CC_ARBITRARY_LOWER_RANKS=ON
  -> run_tensor_optimized_rccsdt calls run_rccgen (the arbitrary-order harness)
  -> never reaches run_tensor_rccsdt_impl, so the probe never fires
     (verified: grep PLANCK_CC_T3_TIME over rccgen.cpp,
      generated_arbitrary_runtime.cpp, solver_arbitrary.cpp -> no match)
```

Both branches were run, not inferred. The first errors out; the second has no
probe in it.

`CCGEN_RANK3_KERNEL_AND_SOLVER.md` made that reroute deliberately and correctly
— the old solver's symmetry-packed representation is incompatible with the dense
one the kernels are emitted for, and converged to `-7.56e-05`. The probe was
simply left behind on the abandoned path.

## A wrong claim this note previously made, and the correction

> **Retracted.** An earlier revision of this document claimed the arbitrary-order
> companion TUs "stay undressed", concluded that the dressed rank-3 kernel had
> **no reachable caller**, and cast doubt on W5's 3.12x/3.61x on that basis.
> **All of that was wrong.** It rested on a stale comment in `CMakeLists.txt`
> plus the absence of arbitrary TUs in two build trees that had
> `ARBITRARY_LOWER_RANKS=OFF` — i.e. trees where no such TU is emitted at all, so
> `build_W:0` was the expected result either way and proved nothing.

Measured directly, one method, one flag varied:

```
python generate_planck_cc_kernels.py --methods ccsd --engine diagram \
       --arbitrary-lower-ranks --spin-adapt [--dressing derived]
```

| TU | undressed | `--dressing derived` |
|---|---|---|
| `ccsd_planck_generated.cpp` (plain) | 2156 lines, **0** `build_W` | 3531 lines, **119** `build_W` |
| `ccsd_arbitrary_planck_generated.cpp` (companion) | 2195 lines, **0** `build_W` | 3703 lines, **119** `build_W` |

**Dressing reaches the arbitrary-order TU — the one the registry actually
executes.** `generate_planck_cc_kernels.py:179` passes `dressing=dressing`
unconditionally; `force_arbitrary` gates only `include_intermediates`. The
source comment at `:157` says so outright: *"Dressing DOES apply to the
arbitrary-order TUs, and must: those are the ones the kernel registry actually
executes."*

The CMake comment described a **V1.3.1** suppression that **V1.3.2** removed by
method-suffixing every builder (`build_tau_ccsdtq`). It was stale by two
revisions and has been corrected.

**W5's 3.12x/3.61x therefore stand as measured**, and the mechanism is exactly
what it claimed: `optimized` + `ARBITRARY_LOWER_RANKS=ON` runs the arbitrary
harness, and that harness's TU *is* dressed.

The lesson is the one this subsystem keeps re-teaching: **a comment is not a
gate.** Two prior instances are on record — the `PLANCK_CC_SPIN_ADAPT` default
and the ERI symmetry table, where warning comments existed and did not prevent a
third module carrying the bad set. Here a stale comment produced a false
conclusion in the other direction, inventing a defect that did not exist.
Checking it cost one generator invocation.

## What to actually do

1. **Move the probe to where the generated residual is evaluated** — the
   arbitrary-order harness (`generated_arbitrary_runtime.cpp` /
   `solver_arbitrary.cpp`), not `tensor_backend.cpp`. It needs to time one
   residual evaluation from fixed amplitudes, the same contract
   `PLANCK_CC_T3_TIME` has now. `PLANCK_CC_FIXTURE_DIR` already injects
   amplitudes and evaluates residuals once in that path, so the hook exists.
2. **Then re-run the six points** in both arms. The setup below is ready.
3. **Only then decide on `_optimal_contraction_order`.** If the dressed
   exponents fall toward the hand-written `o^3.94 v^4.18`, the emitter change is
   largely redundant; if the ratio merely scales down, it is still the
   asymptotic fix.

Note the comparison is **generated-dressed vs generated-undressed**, both
through the arbitrary harness. Do not mix in the hand-written kernel's timings
from the original ladder — that ladder timed `tensor_backend`'s hand-written
triples against the plain TU's generated one, which is a different pairing.

## What is set up and reusable

- Two configured trees differing in exactly one flag, cache-diffed:
  `build-ladder-{undressed,dressed}` (`MAXORDER=3`, `SPIN_ADAPT=ON`,
  `DRESSING=derived`). Both build clean. **Note both have
  `ARBITRARY_LOWER_RANKS=OFF`**, so they emit no companion TU — add that flag
  before using them for the real measurement.
- Six ladder inputs matching the scope's table (`bh3_sto3g`, `ch4_sto3g`,
  `hf_631g`, `h2o_631g`, `bh3_631g`, `c2h4_sto3g`), spanning `o/v` 0.36-1.33.
  The `ch4` copy preserves its load-bearing `use_diis .true.`
- A harness that reports *why* a point yields no timing rather than dropping it
  silently — which is how the blocker surfaced instead of a five-point ladder
  with one unexplained gap.

## Constraints that still hold

- `PLANCK_RCCSDT_BACKEND=tensor` is the **hand-written** path; only `optimized`
  selects generated kernels. Check the backend marker before believing any
  number — that rule is what caught the first dead end here.
- Energies must be identical between arms, per point.
- The determinant backstop (`nso <= 16 && ndet <= 10000`) binds the hand-written
  arm only; all six ladder points clear it regardless.

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
