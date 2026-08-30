# What does threading the generated CC path buy?

**3.22x at 4 threads, and a 2.6 % win that has nothing to do with threading.**
Landed 2026-08-30. CC was the only hot path in Planck with no OpenMP at all; it
now emits `#pragma omp parallel for collapse(3) schedule(static)` on every
residual nest, behind `CCGEN_OMP_COLLAPSE` (default off).

| emit | 1t | 4t | 8t | **4t speedup** |
|---|---|---|---|---|
| default (`CCGEN_OMP_COLLAPSE` unset) | **76.68 s** | 75.82 s | — | 1.01x |
| `CCGEN_OMP_COLLAPSE=3` | 74.37 s | **23.11 s** | 18.31 s | **3.22x** |

HF/6-31G through the generated rank-3 route, on `build-full` (g++-15, Release,
MAXORDER=4, dressed + merged). Energies are **bitwise identical** at 1/2/4/8
threads and against the unthreaded baseline. CPU utilization went **99.1 % ->
359.9 %** on a 4-performance-core machine.

**The default row is the surprise.** It carries no threading — it is the
by-product described below, worth 2.6 % to every build whether or not anyone
enables OpenMP.

## The measurement that had to come first

The claim "CC is unthreaded" had been asserted from `grep` and from a "98.8 % CPU
on 8 cores" observation. That observation could not support it: it was taken on a
tree where `OpenMP_CXX_FLAGS` was `NOTFOUND` and `-DUSE_OPENMP` never reached the
compile line, so **every** Planck pragma was inert in that binary, not just CC's
absent ones. It was equally consistent with "CC has no pragmas" and "this build
has no OpenMP".

Re-measured on a genuinely threaded binary, with a positive control in the *same*
binary:

| | 1 thread | 4 threads | |
|---|---|---|---|
| **CC** (HF/6-31G, generated rank-3) | 78.03 s | 78.49 s | **flat** |
| **direct-SCF HF/cc-pVTZ** (control) | 2.70 s | 0.81 s | **3.3x** |

Threading demonstrably worked in that binary and CC alone was flat. **Pick the
control carefully**: a small DFT case is 0.18 s -> 0.17 s (too small to thread),
and even water/cc-pVDZ ultrafine reaches only 1.09x because it is *grid*-bound and
the DFT grid layer is itself unthreaded. An earlier draft of this work reported
"5.6x" for that small case — the 0.96 s it divided by was a **cold-start first
run**; warm it is 0.18 s. Measure warm, and choose a case actually bound by the
threaded path.

## Why `collapse(3)`, not a bare `parallel for`

Each residual nest is six deep over `(i,j,k,a,b,c)`, writing one disjoint
`result(i,j,k,a,b,c)` slice per iteration and accumulating into a thread-private
`acc`. Safety is not the constraint — **granularity is**. The outer index alone
has `no` = 5 trips on every reachable test system:

| system | no | nv | outer `i` | `collapse(2)` | `collapse(3)` |
|---|---|---|---|---|---|
| CH4/STO-3G | 5 | 4 | 5 | 25 | 125 |
| HF/6-31G | 5 | 6 | 5 | 25 | 125 |

Five iterations over four threads is 1-2 each. Measured, `collapse(3)` beats
`collapse(2)` by a consistent ~4 % (40.85 s against 42.63 s at 4 threads, ±0.1 s
across repeats) and costs nothing, since the collapsed indices are all output
indices and writes stay disjoint either way.

## How it got there, and what each step was worth

| step | change | 1t | 4t | speedup |
|---|---|---|---|---|
| baseline | none | 78.67 s | 78.49 s | 1.00x |
| `part1` only | 256 nests | 78.67 s | 40.85 s | 1.93x |
| all four parts | 806 nests | 78.67 s | 27.98 s | 2.81x |
| + dead builds removed | — | 73.79 s | 23.70 s | 3.11x |
| from the emitter | + singles/doubles | 74.37 s | 23.11 s | **3.22x** |

`part1` alone saturated its own Amdahl ceiling — 1.93x against a 1.94x bound for a
64.6 % part — so the only way further was to widen what is threaded. Threading all
four parts landed almost exactly on the modelled 2.74x.

**Quote the same-binary serial reference.** 3.22x is against 74.37 s, the threaded
binary at `OMP_NUM_THREADS=1`. Against the 80.92 s pre-work binary it would read
3.5x and would be pricing a rebuild as a threading win.

## The by-product: every chunked kernel was building its operators twice

Inspecting the triples entry point in order to thread its 88 operator builds
revealed that `compute_ccsdt_triples_residual` emitted them **twice** — once as 88
`const auto` locals, then again inside the `ops` aggregate — with the locals
**never referenced**, because the `_partN` functions read `ops`.

It is an emitter defect, not an artifact of one file: `_emit_kernel` emitted the
intermediate builds unconditionally, and *then* delegated large kernels to
`_emit_chunked_kernel`, which builds the same operators into the struct. H5
introduced that hoist and did not remove the emission it superseded, so every
chunked kernel paid twice.

Worth **4.9 s serial / 4.3 s at 4 threads** on HF/6-31G at rank 3 (88 operators),
about 2.6 % of the solve, and it should scale with operator count — rank 4 has
**894**.

**Why it survived every gate is the transferable part: the duplicate is
semantically a no-op.** Energies, residuals and every value gate were correct
while the work was being done twice. Only wall-clock or a reading of the emitted
text could see it. `test_chunked_kernel_builds_once.py` therefore reads the text,
and checks two things rather than one — no operator built twice, *and* no
`const auto` operator build before the `ops` aggregate — because a future change
could emit one without a matching struct entry and slip past a duplicate-only
check. Mutation-verified: restoring the unconditional emission turns both red
while the vacuity check stays green.

**The triples builders never needed threading. They needed deleting.** The 12-13 %
of runtime they occupied was mostly work that should not have run.

## Determinism

The historical DFT jitter came from a **cross-thread reduction summed in
completion order** (`dft_xc_reduction_determinism`). Neither site here has one:
builders write their own freshly-allocated tensors, and residual nests accumulate
into a thread-private `acc` before writing one disjoint slice, with the inner
summed loop serial within a thread so its accumulation order is unchanged. That is
the same structural property that made the DFT J/K builds bitwise
thread-count-invariant.

**That is a reason to expect determinism, not evidence of it.** Verified the way
the J/K builds were: every `E_corr`, `dE`, `rms(res)` and `rms(step)` matches
across all 15 iterations at `OMP_NUM_THREADS` = 1/2/4/8 and against the unthreaded
baseline — for every variant measured, not just the final one. **"Matches to
tolerance" is not the claim.**

## Using it

```bash
CCGEN_OMP_COLLAPSE=3 cmake --build build    # threaded CC kernels
```

Default (unset or 0) emits no pragma and is byte-identical to the pre-existing
emit apart from the removed dead builds. An env var rather than a
`print_cpp_planck` parameter, following `_fuse_loops_setting`'s precedent — W3's
condition is that a knob earns a parameter once it is staying. If threading
becomes the default, that is the moment for a CMake option, not before.

A guard (`len(free) >= omp_collapse`) leaves too-shallow nests alone; verified
across all 948 emitted pragmas that none collapses more levels than it has.

## Traps

- **Do not use a CPU-utilization percentage alone as evidence that something is
  unthreaded.** It cannot distinguish "no pragmas here" from "no OpenMP in this
  binary". Pair it with a positive control on the same binary, and confirm
  `-fopenmp` reaches the compile line.
- **`make hartree-fock` silently reverts hand-edits to generated files.**
  `ccgen-planck-kernels` is an unconditional dependency of the target, so a normal
  build regenerates and wipes them — 806 pragmas back to 0, no error. Compile the
  object and link directly, and check both `grep -c "pragma omp"` on the source
  and `nm | grep GOMP_parallel` on the object before trusting a number.
- **Do not copy a build tree to experiment in.** `CMAKE_CACHEFILE_DIR` is
  absolute, so the copy rebuilds into the *original* — silently corrupting the
  baseline being measured against.
- **Do not introduce a cross-thread reduction.** If a future term shape needs one,
  it must sum in fixed thread-index order, never completion order.
- **Do not thread the builders.** They are 12-13 % and, in the chunked kernels,
  were mostly dead work. Deleting beat threading.

## What is untested

**Rank 4 threading.** The dead-builder fix is verified there (`be_rccsdtq_sto3g`
passes), but no threaded rank-4 TU has been built. With 894 operators against 88,
the dead-builder share should be larger than rank 3's 2.6 %.

**Beyond 4 threads on this machine.** 8 threads gives 3.22x -> 4.06x of the serial
time but the machine has only 4 performance cores, so the 8-thread numbers say
little about a real 8-core run.

## Key code locations

| what | where |
|---|---|
| the pragma | `emit_planck_term`'s `omp_collapse`, `python/ccgen/emit/planck_tensor_cpp.py` |
| the knob | `_omp_collapse_setting`, same file |
| the duplicate-build fix | `_emit_kernel`'s `chunked` guard, same file |
| the gate for it | `python/ccgen/tests/test_chunked_kernel_builds_once.py` |
| the hot nests | `compute_ccsdt_triples_residual_part{0,1}` in the generated TU |
| where the split came from | `docs/CCGEN_ARBITRARY_HARNESS_COST.md` |
| the determinism precedent | DFT J/K builds, `src/dft/driver.cpp` |
| the determinism defect to avoid | `dft_xc_reduction_determinism` note |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
