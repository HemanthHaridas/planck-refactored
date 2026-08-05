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

### The registration is rank-hardcoded, but the runtime is not

The mechanism to run the generated CCSDTQ kernel exists, but it is pinned to
rank 4 in four C++ places even though the layers above and below it are already
arbitrary-order:

- **Generation is arbitrary-order.** `generate_cc_equations` and
  `emit_factorized_translation_unit` handle any rank; the CMake codegen already
  maps ranks 2–6 to method names (`_planck_cc_method_by_rank = ccsd ccsdt ccsdtq
  cc5 cc6`) and `PLANCK_CC_MAXORDER` accepts 2–6. **Nothing stops a user
  generating a cc6 kernel today.**
- **The runtime solver is arbitrary-order.** `prepare_generated_arbitrary_order_state`
  takes `max_excitation_rank` as a parameter (validates `≥ 1`), and
  `GeneratedArbitraryOrderKernels` is a rank-generic bundle. The engine that runs
  the kernels does not care about rank.
- **But the registration + options ceiling at rank 4:**
  1. `generated_kernel_registry.cpp` has a single `make_generated_rccsdtq_kernels()`
     behind `#if PLANCK_CC_MAXORDER >= 4`, `#include`ing only
     `ccsdtq_planck_generated.cpp`. No cc5/cc6 include, no per-rank factory.
  2. `enum class PostHF` (`src/base/types.h`) stops at `RCCSDTQ` — no `RCC5`/`RCC6`.
  3. The io.cpp option table stops at `{"cc4", RCCSDTQ}` — no `cc5`/`cc6` keyword.
  4. The driver hardcodes `run_rccsdtq` → `prepare_generated_arbitrary_order_state(
     …, 4, …)` and `make_generated_rccsdtq_kernels()`.

So a user can emit a cc6 kernel but has no `PostHF` option to select it and no
registration to run it. **Part B is therefore not "build the MAXORDER=4 path" —
it is "make registration and the `%posthf` options arbitrary-order," so any rank
the codegen emits is automatically runnable.** The dressed/factorized/memory-aware
TUs are a further, fully-unwired path (`#include`d nowhere).

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

## Part B — scope: arbitrary-order kernel registration

The goal is **registration that scales with the codegen, not a per-rank ceiling**:
whatever rank `PLANCK_CC_MAXORDER` emits (up to 6) should be automatically
registered and selectable in `%posthf`, with no new C++ per rank. The runtime
solver and the codegen are already arbitrary-order (see Background); only the
registration, the `PostHF` enum, and the io.cpp options are rank-hardcoded. Small
verifiable steps, registration first.

- **W0 — a rank-parameterized registry (~M, the core generalization). LANDED
  (registry) — but exposed a downstream emitter defect (W0.1).** Replaced the
  single `make_generated_rccsdtq_kernels()` / `#if >= 4` with
  `make_generated_rcc_kernels(int rank)`: a per-rank `switch`, each case behind its
  own `#if PLANCK_CC_MAXORDER >= N`, calling the TU's `make_generated_<method>_kernels()`
  (4=ccsdtq, 5=cc5, 6=cc6, matching `_planck_cc_method_by_rank`).
  `make_generated_rccsdtq_kernels()` kept as a rank-4 alias for `run_rccsdtq`.
  Rank<4 and unbuilt-rank return errors naming the reconfigure. **Compiles at the
  default MAXORDER=3** (all cases guarded off; unchanged behavior). The header is
  extended, the driver call site is untouched.

- **W0.1 — fix the rank-≥4 amplitude-accessor emit (~S, was the BLOCKER). LANDED.**
  Building the registry at MAXORDER=4 surfaced a pre-existing emitter defect the
  `#if >= 4` guard had always hidden: the generated ccsdtq kernels called
  `amplitudes.t1/.t2/.t3(...)`, but the arbitrary-order runtime type
  `ArbitraryOrderRCCAmplitudes` (which the emitted bundle's residual lambdas take)
  exposes only `.tensor(rank)` — and that returns `std::expected<view>`, so even
  `.tensor(rank)(...)` does not compile. The ccsdtq TU had **never** compiled
  against the runtime; the guard kept anyone from finding out. **Fix (two parts):**
  (1) `_map_factor` emits a local `t<rank>({...})` for the arbitrary target instead
  of the `t1/t2/t3` shortcut; (2) `_emit_kernel` / `_emit_intermediate_builder`
  emit a per-kernel prologue binding each used rank's view once —
  `const auto t<rank> = amplitudes.tensor(rank).value();`
  (`_amplitude_view_bindings`), unwrapping the `expected` outside the loops.
  Gated on `arbitrary = amplitude_type == "ArbitraryOrderRCCAmplitudes"`, so the
  rank ≤ 3 tensor_backend path (`RCCSDTAmplitudes`, direct `.t1`) is
  **byte-unchanged** (verified: ccsdt still emits `.t1`, zero view-bindings).
  *Gate:* `test_generated_ccsdtq_tu_compiles_against_runtime` — the generated
  CCSDTQ TU compiles against the real CC headers, uses no `amplitudes.t[123](`,
  and binds views instead. This is the true unblock: the MAXORDER=4 registry +
  ccsdtq now compile together (verified `exit 0`).

- **W1 — arbitrary-order `%posthf` options (~S given W0).** The `PostHF` enum
  stops at `RCCSDTQ`; extend the CC path so `correlation cc5` / `cc6` (and the
  `ccsdtqp`/… aliases) parse and dispatch. Two shapes are possible — add `RCC5`/
  `RCC6` enum members, or (cleaner) carry the rank as an integer alongside a
  single `RCCGeneratedArbitrary` PostHF value so no enum edit is needed per rank.
  The driver then calls `run_rcc_generated(rank)` (a rank-parameterized
  generalization of `run_rccsdtq`) → `prepare_generated_arbitrary_order_state(…,
  rank, …)` (already rank-generic) → `make_generated_rcc_kernels(rank)` (W0).
  *Gate:* `correlation cc5` at MAXORDER=5 runs the generated rank-5 kernel;
  `correlation cc5` at MAXORDER=3 fails with the "reconfigure with
  -DPLANCK_CC_MAXORDER=5" message (not a parse error, not a crash).

- **W2 — CI builds a high-MAXORDER configuration (~S given W0/W1).** Add a CMake/CI
  configuration that builds `hartree-fock` at `-DPLANCK_CC_MAXORDER=4` (and,
  behind a slower opt-in job, 5/6) so the generated ranks are actually compiled
  and the W3 tests can run. *Gate:* the high-MAXORDER build links and starts; the
  arbitrary-order options from W1 are live.

- **W3 — arbitrary-order generated-CC regression cases (~S given W2).** Add small
  closed-shell inputs where CCSDTQ (and cc5 where feasible) ≡ FCI, gated on the
  generated energy matching the FCI/hand reference to ~1e-8 — the first runtime
  test of the generated kernels (the A4 tier), parameterized over rank rather than
  pinned to 4. *Gate:* `run_regressions.py` runs each rank's case and the energy
  matches; a case is **skipped** (not failed) when the binary's MAXORDER is below
  its rank.

- **W4 — a self-contained generated-kernel unit harness (~M).** For the
  factorized / memory-aware TUs (which are `#include`d nowhere), the cheapest
  runnable route is a standalone unit executable, not driver integration: a
  `tests/generated_kernel_energy.cpp` that builds a tiny reference, calls the
  emitted `build_W` + residual kernels, and checks the CC energy. The emitted TU
  is compiled into this one executable (an `add_executable` mirroring
  `planck-cc-arbitrary-solver` plus the emitted `.cpp`). This turns every A3
  compile-only gate into a numeric one without touching the driver or the default
  build, and works at any rank. *Gate:* the harness links the emitted TU, runs,
  and matches the arbitrary-order solver / FCI energy; default build untouched.

- **W5 — factorized/memory-aware emit selectable at codegen (~S given W4).** The
  build-time codegen (`generate_planck_cc_kernels.py`, the `ccgen-planck-kernels`
  target) emits the *plain* TU today. Add pass-through flags so it can emit the
  factorized / budgeted / stride-shaped TU (the `emit_factorized_translation_unit`
  options) per rank, via CMake cache vars mirroring `PLANCK_CC_ENGINE`. *Gate:*
  the generated `.cpp` reflects the requested optimization and W4's harness runs
  it to the same energy.

**Sequencing / risk.** W0 is the ~M core — the per-rank registry; once it lands,
W1–W3 make any emitted rank selectable and testable. W4 is the other ~M piece (a
harness compiling an emitted TU), the route to the numeric-energy ("E2") boundary
the prior investigations flagged. W5 is polish.

**What NOT to do.** Do not flip the default `PLANCK_CC_MAXORDER` — it changes and
slows every default build (high-rank codegen is the slow step). Registration stays
compile-time gated per rank; CI opts into the higher configurations. Do not add a
bespoke factory or driver path per rank (`run_rccsdtq`, `run_rcc5`, …) — that
recreates the ceiling one rank higher; W0/W1 parameterize on rank so the ceiling
is `PLANCK_CC_MAXORDER` alone.

---

## Part C — scope: the benchmark script

A script that builds the relevant configurations and runs the compiled-binary
tests, reporting pass/fail and timing. It is a *driver over the existing harnesses*,
not a new test framework.

- **C0 — enumerate & run the compiled-binary tests (~S).** A script
  (`tests/benchmark_generated_kernels.py`) that: (1) configures + builds
  `hartree-fock` at a baseline MAXORDER=3 and at a high MAXORDER (4, optionally
  5/6) — one build dir each, (2) runs the `ctest` C++ unit tests and the
  `run_regressions.py` CC cases against each, (3) runs the W4 generated-kernel
  harness. Report: per-test pass/fail + wall time, and the per-rank deltas.
  *Gate:* the script runs end to end on a machine with a C++23 compiler; it skips
  (not fails) ranks whose binary is absent.

- **C1 — timing the generated vs hand-written CC path (~S given C0 + W3).** For
  the one method where both exist (CCSDT: hand-written determinant/tensor backend
  vs generated tensor-optimized), run the same input through
  `PLANCK_RCCSDT_BACKEND={determinant,tensor,optimized}` and report energies
  (must agree) + wall times. This is the actual performance benchmark the
  generated path was built for. *Gate:* energies agree to ~1e-8 across backends;
  timings reported.

- **C2 — the memory/factorization emit, timed (~S given W4/W5).** Run W4's
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
