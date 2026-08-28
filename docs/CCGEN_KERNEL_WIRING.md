# How does a generated CC kernel reach a runnable binary, and what does it cost?

A ccgen-emitted kernel is a `.cpp` file. Getting it *executed* is a separate
problem from getting it *emitted*, and for a long time the two were confused —
kernels existed, compiled, and were never called.

This answers the route from emitted file to running code, the flags that gate
each stage, and what the path costs measured rather than modelled.

## The route

```
ccgen emit  ->  #include or registry  ->  backend selection  ->  solver harness
```

Three generated artifacts enter a binary three different ways:

| artifact | how it enters | reached by |
|---|---|---|
| `ccsd_spinorbital_warm_start.inc` | `#include`d unconditionally in `tensor_backend.cpp` | always — the RCCSD warm start |
| `<method>_planck_generated.cpp` | `#include`d in `tensor_backend.cpp` | only when the tensor-optimized RCCSDT backend is selected |
| `<method>_arbitrary_planck_generated.cpp` | registry, behind `PLANCK_CC_MAXORDER` / `PLANCK_CC_ARBITRARY_LOWER_RANKS` | `rccgen.cpp` -> the arbitrary-order harness |

**The third row is the one that runs in production.** The plain per-method TUs
are compiled but their residual kernels have no caller at ranks 2-3 — the
registry says so outright: "rank 2 and 3 use the hand-written backends".

## The flags, and what each actually gates

| flag | gates |
|---|---|
| `PLANCK_CC_MAXORDER` | which ranks are emitted at all (default 3) |
| `PLANCK_CC_ARBITRARY_LOWER_RANKS` | lowers `generated_floor` from 4 to 3, so rank 3 routes to the arbitrary harness |
| `PLANCK_CC_SPIN_ADAPT` | spatial vs the historical spin-orbital emit. **Defaults ON since 2026-08-26** |
| `PLANCK_CC_UCC` | a second emit pass for spin-resolved kernels |
| `PLANCK_CC_DRESS_OPERATORS` + `PLANCK_CC_DRESSING` | dressed operators, and which route derives them |
| `PLANCK_RCCSDT_BACKEND` (env) | `determinant` / `tensor` / `optimized` at run time |

**Two of these have silently produced wrong answers**, both because a default
preserved historical behaviour rather than correct behaviour:

- `PLANCK_CC_SPIN_ADAPT=OFF` emitted algebra that is ~4x wrong, and cost a full
  investigation before anyone diffed the build cache
  (`CCGEN_SPIN_ADAPT_DEFAULT.md`).
- `PLANCK_CC_DRESSING` did not exist, so CMake hard-coded `recognized` and the
  derivation route was unreachable from a build
  (`CCGEN_WIRING_THE_DERIVATION_ROUTE.md`).

**The lesson both times: check the build cache before the code.** `grep '^PLANCK_CC'
<build>/CMakeCache.txt` and diff it against a known-good tree. Two separate
investigations lost days to a flag nobody had verified.

## The trap that hid all of this

`choose_determinant_backstop` (`tensor_backend.cpp:243`) routes any case with
`nso <= 16 && ndet <= 10000` to the determinant-space teaching backstop, which
**calls no generated code at all**. Of the CC regression cases, most land there.

For a long time `ch4_rccsdt_sto3g` was the only case above the threshold — and
it *asserts* `kernels=hand-optimized`, so it was green for its entire life while
never executing the generated kernel it was added to protect.

**Corrected 2026-08-26:** the backstop binds the **hand-written** path only.
`PLANCK_RCCSDT_BACKEND=optimized` routes through `rccgen.cpp` to the
arbitrary-order harness, which never consults it. So a small case *can* exercise
the generated route — `lih_rccsdt_generated_sto3g` (nso=12, ndet=495) does, in
5 s against CH4's ~250 s. Several ccgen documents still record the
`nso > 16 || ndet > 10000` requirement as universal; it is not.

## What it costs

Measured, not modelled. Same input, same binary configuration apart from the
flag under test.

**Generated vs hand-written** (`CCGEN_KERNEL_SCALING_SCOPE.md`, six-point ladder,
isolated triples residual): the generated kernel is **21.8x to 50.1x slower**,
and the ratio *grows* with system size — a scaling defect, not a constant tax.
The dominant constant factor, an out-of-line allocating tensor accessor, is
fixed (`CCGEN_KERNEL_PERFORMANCE.md`, 206x on rank 3).

**Undressed vs derivation-dressed** (`CCGEN_WIRING_THE_DERIVATION_ROUTE.md`,
end-to-end solve):

| system | undressed | dressed | speedup |
|---|---|---|---|
| LiH/STO-3G | 5.12 s | 1.64 s | **3.12x** |
| CH4/STO-3G | 104.56 s | 28.94 s | **3.61x** |

Energies identical, iteration counts unchanged — per-iteration work.

**These two sets are not comparable.** The first is an isolated residual
evaluation on a designed ladder; the second is end-to-end solve time on two
systems, one of them off that ladder. Combining them into a single ratio would be
wrong, and the scaling ladder has not yet been re-run under dressing.

**Compile time is a real cost.** `generated_kernel_registry.cpp` is pinned to
`-O1` (`CMakeLists.txt:408-415`) because a ~230k-line TU is super-linear to
optimize; under `SPIN_ADAPT=ON` with dressing it takes minutes on its own, and
the dressed CCSDTQ TU is 13 MB. Budget for it, and build with `make -j4` — a
full-width build on these TUs is disruptive.

## Benchmarking

There is no `benchmark_generated_kernels.py`, and the scope that proposed one is
retired with this rewrite. What replaced it, and works:

- **`run_regressions.py` with `requires_build_option`** — cases declare the flags
  they need and SKIP rather than fail elsewhere, which is what stops a case from
  silently measuring the wrong emit. It accepts a list.
- **`PLANCK_CC_T3_TIME=N`** — an opt-in isolated triples-residual timer, inert
  when unset. This is what the scaling ladder used.
- **`PLANCK_CC_FIXTURE_DIR`** — inject amplitudes, evaluate residuals once, dump
  per-rank tensors, exit. No solver, no DIIS. Built for the D4 diagnosis and the
  right instrument for any "which manifold disagrees" question.

A driver script would have added a fourth harness over three that already work.
The gap it was meant to fill — *nothing proves the generated path ran* — was
closed by gates that assert the routing line instead.

## Related

| doc | question |
|---|---|
| `CCGEN_SPIN_ADAPT_DEFAULT.md` | why a build flag made the kernel look broken |
| `CCGEN_WIRING_THE_DERIVATION_ROUTE.md` | how dressing reaches production, and the defect that exposed |
| `CCGEN_KERNEL_SCALING_SCOPE.md` | why generated is slower than hand-written, and by how much |
| `CCGEN_KERNEL_PERFORMANCE.md` | the accessor fix, the dominant constant factor |
| `CCGEN_GCC_TO_UCC_BRIDGE.md` | how adapted terms become runtime tensors |

## Key code locations

| what | where |
|---|---|
| the registry | `src/post_hf/cc/generated_kernel_registry.cpp` |
| the generated-rank floor | `make_generated_rcc_kernels`, same file |
| the production entry | `src/post_hf/cc/rccgen.cpp` |
| backend selection | `src/post_hf/cc/ccsdt.cpp:22` |
| the backstop (hand-written path only) | `src/post_hf/cc/tensor_backend.cpp:243` |
| codegen invocation | `CMakeLists.txt` (`ccgen-planck-kernels`) |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
