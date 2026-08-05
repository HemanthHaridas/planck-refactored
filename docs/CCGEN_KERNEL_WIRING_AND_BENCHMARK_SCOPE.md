# Wiring the generated CC kernels into a runnable binary, and benchmarking them

This document answers one question: **what does it take to run the ccgen-generated
CC kernels in a compiled binary, and how do we benchmark the tests that need
one?** It scopes three connected pieces — (A) the inventory of tests that require
a compiled binary, (B) the route to register the generated kernels so they are
actually executed, and (C) a benchmark script that runs them. Part B is scoped
first because the runtime tests in A depend on it.

Everything below is grounded in the current tree; nothing here is landed yet.

---

## Background: how the generated path reaches a binary today

Three distinct generated artifacts, three distinct wiring states:

| generated artifact | how it enters a binary | state |
|---|---|---|
| `ccsd_spinorbital_warm_start.inc` | `#include`d unconditionally in `tensor_backend.cpp` | **wired & run** (RCCSD warm-start) |
| `ccsdt_planck_generated.cpp` | `#include`d unconditionally in `tensor_backend.cpp` | **wired**; run only if the tensor-optimized RCCSDT backend is selected |
| `ccsdtq_planck_generated.cpp` | `#include`d in `generated_kernel_registry.cpp` behind `#if PLANCK_CC_MAXORDER >= 4` | **guarded off** — default build (MAXORDER=3) returns a "not available" stub |

The CCSDTQ path is the one that is not runnable in the default build. The driver
routes `correlation ccsdtq` / `cc4` → `run_rccsdtq` (`src/post_hf/cc/ccsdtq.cpp`),
which calls `make_generated_rccsdtq_kernels()`
(`generated_kernel_registry.cpp`). At the default `PLANCK_CC_MAXORDER=3` that
factory returns:

> "Generated RCCSDTQ kernels are not available in this build. Reconfigure with
> -DPLANCK_CC_MAXORDER=4 (or higher) and rebuild."

So the mechanism to run the generated CCSDTQ kernel **already exists** — it is one
CMake cache variable. The gap is not "how do we register them" (that is done); it
is (1) nothing exercises the MAXORDER≥4 path in CI, and (2) there is no runtime
regression case that runs a generated CCSDTQ energy to compare against a
reference. The dressed/factorized/memory-aware translation units from the
factorization and memory investigations are a *separate*, fully-unwired path
(they are emitted by `emit_factorized_translation_unit` but `#include`d nowhere).

---

## Part A — tests that require a compiled binary

Two tiers exist today, plus the tier this work adds.

### A1 — C++ unit tests (link + run their own `main`)

Built under `if(BUILD_TESTING)` in `CMakeLists.txt`, registered with `add_test`,
run via `ctest`. ~40 executables (`tests/*.cpp`). The CC-relevant ones:

- `planck-cc-arbitrary-solver` (`tests/cc_arbitrary_solver.cpp`) — exercises the
  arbitrary-order solver machinery (`solver_arbitrary.cpp` +
  `generated_arbitrary_runtime.cpp`) but **does not link
  `generated_kernel_registry.cpp`**, so it tests the solver, not the generated
  CCSDTQ kernels.
- `planck-eri-derivative-kernels`, `planck-compute-2e`, `planck-hgp-engine-smoke`,
  etc. — integral-engine gates, unrelated to generated CC.

None of the existing C++ unit tests run a *generated CC kernel end to end*.

### A2 — regression tests (run `hartree-fock` / `planck-dft` on input files)

`tests/run_regressions.py` + `tests/regression_cases.json` invoke the compiled
`hartree-fock` / `planck-dft` binaries on `.hfinp` inputs and diff printed
outputs / metrics. CC coverage: `h2_rccsdt_sto3g`, `lih_rccsdt_sto3g`, … — all
**determinant-space or hand-written** backends. **There is no `ccsdtq` / `cc4`
regression case at all**, so the generated kernels are never run by the regression
suite.

### A3 — Python compile-only gates (compile, do NOT run)

Six tests currently shell out to `c++ -fsyntax-only` on emitted TUs
(`test_factorize.py::test_factorized_tu_compiles`,
`test_footprint_guarded_tu_compiles`, `test_emit_memory_budget_compiles`,
`test_factored_builder_tu_compiles`, `test_stride_ordered_builder_tu_compiles`;
`test_tau.py::test_generated_source_compiles`). These prove the emitted C++ is
*valid* but never link or execute it — they are the closest thing to a
generated-kernel binary test today, and they stop one step short.

### A4 — the missing tier (what this work enables)

A test that **runs a generated CC kernel in a compiled binary and checks its
energy** against a reference (FCI / hand-written solver / PySCF). None exists.
This is the tier Part B unblocks and Part C benchmarks.

---

## Part B — scope: register the generated kernels so they run

The goal: make a generated CCSDTQ energy runnable and checkable in CI, and make
the factorized/memory-aware TUs runnable so their *numeric* (not just
compile-only) gates can exist. Small verifiable steps.

- **W0 — CI builds the MAXORDER=4 path (~S).** The registration is already
  correct; nothing exercises it. Add a CMake/CI configuration that builds
  `hartree-fock` with `-DPLANCK_CC_MAXORDER=4` so
  `ccsdtq_planck_generated.cpp` is compiled and `make_generated_rccsdtq_kernels()`
  returns real kernels. *Gate:* the MAXORDER=4 build links and
  `hartree-fock` starts; a `correlation ccsdtq` input no longer prints the
  "not available" stub.

- **W1 — a generated-CCSDTQ regression case (~S given W0).** Add a small
  closed-shell `cc4` input (e.g. a 4-electron system where CCSDTQ ≡ FCI) to
  `regression_cases.json`, gated on the generated energy matching the FCI/hand
  reference to ~1e-8. This is the first runtime test of the generated kernels —
  the A4 tier. *Gate:* `run_regressions.py` runs the case and the energy matches;
  the case is skipped (not failed) when the binary was built at MAXORDER<4.

- **W2 — a self-contained generated-kernel unit harness (~M).** For the
  factorized / memory-aware TUs (which are `#include`d nowhere), the cheapest
  runnable route is a standalone unit executable, not driver integration: a
  `tests/generated_kernel_energy.cpp` that builds a tiny reference, calls the
  emitted `build_W` + residual kernels, and checks the CC energy. The emitted TU
  is compiled into this one executable (an `add_executable` mirroring
  `planck-cc-arbitrary-solver`, plus the emitted `.cpp`). This turns every A3
  compile-only gate into a numeric one without touching the driver or the default
  build. *Gate:* the harness links the emitted TU, runs, and matches the
  arbitrary-order solver / FCI energy; default build untouched.

- **W3 — factorized/memory-aware emit selectable at generation (~S given W2).**
  The build-time codegen (`generate_planck_cc_kernels.py`, invoked by the
  `ccgen-planck-kernels` CMake target) emits the *plain* TU today. Add
  pass-through flags so it can emit the factorized / budgeted / stride-shaped TU
  (the `emit_factorized_translation_unit` options), selectable by CMake cache
  vars mirroring `PLANCK_CC_ENGINE`. *Gate:* the generated `.cpp` reflects the
  requested optimization and W2's harness runs it to the same energy (exactness
  is already Python-gated; this confirms it survives compilation + execution).

**Sequencing / risk.** W0+W1 are the low-risk unblock — they use the registration
that already exists and add the first runtime CCSDTQ test. W2 is the ~M piece (a
new test harness that compiles an emitted TU), and it is what converts the
memory/factorization investigations' compile-only gates into numeric ones —
i.e. it is the route to the "E2 / numeric energy" boundary both prior
investigations flagged as out of scope. W3 is polish on top.

**What NOT to do.** Do not flip the default `PLANCK_CC_MAXORDER` to 4 — it changes
what every default build compiles and slows the build (the ccsdtq generation is
the slow codegen step). Keep the generated-kernel execution opt-in behind the
existing cache variable; CI builds one extra configuration.

---

## Part C — scope: the benchmark script

A script that builds the relevant configurations and runs the compiled-binary
tests, reporting pass/fail and timing. It is a *driver over the existing harnesses*,
not a new test framework.

- **C0 — enumerate & run the compiled-binary tests (~S).** A script
  (`tests/benchmark_generated_kernels.py`) that: (1) configures + builds
  `hartree-fock` at MAXORDER=3 and MAXORDER=4 (two build dirs), (2) runs the
  `ctest` C++ unit tests and the `run_regressions.py` CC cases against each, (3)
  runs the W2 generated-kernel harness. Report: per-test pass/fail + wall time,
  and the MAXORDER=3-vs-4 delta. *Gate:* the script runs end to end on a machine
  with a C++23 compiler; it skips (not fails) configurations whose binary is
  absent.

- **C1 — timing the generated vs hand-written CC path (~S given C0 + W1).** For
  the one method where both exist (CCSDT: hand-written determinant/tensor backend
  vs generated tensor-optimized), run the same input through
  `PLANCK_RCCSDT_BACKEND={determinant,tensor,optimized}` and report energies
  (must agree) + wall times. This is the actual performance benchmark the
  generated path was built for. *Gate:* energies agree to ~1e-8 across backends;
  timings reported.

- **C2 — the memory/factorization emit, timed (~S given W2/W3).** Run W2's
  harness on the plain vs factorized vs memory-budgeted emitted TU for the same
  method, reporting energy (agree) + build-flop proxy + wall time — the runtime
  confirmation of the symbolic FLOP/stride wins from the memory investigation.
  *Gate:* energies agree; the factored TU's measured time is ≤ the flat TU's (the
  first wall-clock evidence for the symbolic model, the noted E2 boundary).

**Honest ceiling.** The benchmark needs a real toolchain and is slow (two full
builds + CC solves); it is a developer/CI tool, not part of the fast Python
suite. C2 is the only step that produces *wall-clock* numbers for the
factorization/memory work — everything prior in those investigations is a
symbolic-model gate, and C2 is precisely where the model meets the hardware.

---

## What this reuses

- The existing registration: `generated_kernel_registry.cpp` +
  `#if PLANCK_CC_MAXORDER >= 4` (Part B needs no new registration mechanism).
- `run_regressions.py` / `regression_cases.json` (W1, C0).
- `emit_factorized_translation_unit` (W3) and the arbitrary-order solver / FCI
  references (W2 oracle).
- The `add_executable`-per-test pattern under `BUILD_TESTING` (W2 harness).

See `CCGEN_GENERATION_AND_VALIDATION.md` (why the generated path is not yet
production-load-bearing) and `CCGEN_REPORT.md` §9 (the compiled-binary boundary
this work crosses).
