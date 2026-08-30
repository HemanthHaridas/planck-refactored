# Scope: threading the generated CC path

**All four steps done (O1-O4), 2026-08-30.** Re-measured 2026-08-29/30
against the post-merge tree, which **invalidated the estimate and the
recommendation** carried in `CCGEN_ARBITRARY_HARNESS_COST.md`. That document told
its reader to re-measure the split before relying on its figures; doing so changed
the answer.

**The baseline is `build-full`** — an existing, genuinely OpenMP-enabled tree
(g++-15, Release, MAXORDER=4, dressed + merged). On it, CC is measurably flat from
1 to 4 threads (78.03 s -> 78.49 s) while direct-SCF HF in the *same binary* goes
3.3x. That pairing is what makes "CC is unthreaded" a measurement rather than an
inference.

## What changed, and why the old plan is now wrong

The prior scope was written when builders were **45.1 %** of runtime and advised
*"start with the builders: better granularity, no write sharing"*. Two landed
fixes (the chunk hoist, then `merge_transposes`) removed most of that work.
Measured now, on the same binary configuration:

| | prior (post-chunk-hoist) | **now (post-merge)** |
|---|---|---|
| `build_W_*` builders | 45.1 % | **13.8 %** |
| residual nests | 53.7 % | **86.2 %** |

(HF/6-31G, 21 341 leaf samples, AppleClang rank-3 build. **Confirmed across a
different compiler and truncation** on `build-full` (g++-15, MAXORDER=4, 33 538
samples): **12.1 % / 87.9 %**, with `part1` at 64.6 % against 63.6 %. CH4/STO-3G
agrees too: 17.2 % / 82.8 %. The split is a property of the dressed+merged kernel,
not of one toolchain.)

**So threading only the builders now caps at 1.12x, not the 1.51x the old scope
implied.** Amdahl on the measured split, 4 physical cores on this machine:

| threads | both sites | residual only | builders only |
|---|---|---|---|
| 2 | 2.00x | 1.76x | 1.07x |
| **4** | **4.00x** | **2.83x** | **1.12x** |
| 8 | 8.00x | 4.07x | 1.14x |

**The residual is now the whole story**, and it is concentrated:

| hotspot | share |
|---|---|
| `compute_ccsdt_triples_residual_part1` | **63.6 %** (64.6 % on `build-full`) |
| `compute_ccsdt_triples_residual_part0` | 19.6 % (20.1 %) |
| `..._part2` | 1.5 % (1.7 %) |
| doubles + singles residual | 1.2 % (1.2 %) |
| all builders combined | 13.8 % (12.1 %) |

Threading the three triples parts (84.7 %) models **2.74x at 4 threads**; `part1`
alone models 1.91x.

## The claim, now measured on a genuinely OpenMP-enabled build

The prior scope's evidence was a "98.8 % CPU on 8 cores" run taken on a tree
where `OpenMP_CXX_FLAGS:STRING=NOTFOUND` and `-DUSE_OPENMP` never reached the
compile line — **every** Planck pragma was inert there, so the number was equally
consistent with "CC has no pragmas" and "this binary has no OpenMP at all". It
could not distinguish them.

**Re-measured on `build-full`**, which is genuinely threaded — `-fopenmp` and
`-DUSE_OPENMP` on the compile line, `libgomp` linked, g++-15, Release, MAXORDER=4,
dressed + merged:

| | `OMP_NUM_THREADS=1` | `=4` | |
|---|---|---|---|
| **CC** (HF/6-31G, generated rank-3 route) | 78.03 s | **78.49 s** | **no change** |
| **HF/cc-pVTZ direct SCF** (same binary, ERI Fock build) | 2.70 s | **0.81 s** | **3.3x** |

At `OMP_NUM_THREADS=8` the CC run sits at **99.1 % CPU** — one core of eight.

**The second row is the control that the earlier attempt lacked.** It rules out
the alternative explanation directly: threading demonstrably works in this binary,
on this machine, at this thread count — the ERI Fock build scales 3.3x. CC alone is
flat. That is now measured rather than asserted from `grep`.

**Choose the control carefully.** A small DFT case is the wrong one: water/STO-3G
B3LYP is 0.18 s -> 0.17 s (too small to thread), and even water/cc-pVDZ ultrafine
is only 1.09x because it is **grid**-dominated, and the DFT grid layer is itself
unthreaded (a separate open item). An early version of this section quoted "0.96 s
-> 0.17 s, 5.6x" from that small case; the 0.96 s was a **cold-start** first run,
and warm it is 0.18 s. A direct-SCF HF case in a real basis exercises the threaded
ERI path and is the honest control.

The static picture agrees and is still worth re-checking after any emitter change:
**0 `pragma omp` in `src/post_hf/cc/*.{cpp,h}` and 0 in the emitted kernels**,
against 8+ other files under `src/` that carry them.

**2. "The emitter never emits a pragma" is false, and this reduces the work.**
`python/ccgen/emit/cpp_loops.py:331` already emits
`#pragma omp parallel for collapse(n) schedule(dynamic)`, with `use_openmp` and
`omp_collapse` parameters plumbed through `emit_term_tiled`, and
`print_cpp_optimized` / `print_cpp_blas` in `generate.py` default it to `True`.

That is a **different emit path** from the one production uses
(`print_cpp_planck` → `planck_tensor_cpp.py`), so it is not a switch to flip. But
it is a working precedent in-tree for the exact pragma shape needed, including the
collapse handling — copy its form rather than designing one.

## Why the residual needs `collapse`, not a bare `parallel for`

Each nest in `part1` is six deep over `(i,j,k,a,b,c)` writing
`result(i,j,k,a,b,c)`, so slices are disjoint per `i` and **no reduction is
involved**. But the outer trip count is `no`, which is **5** on every reachable
test system:

| system | no | nv | outer `i` | `collapse(2)` | `collapse(3)` |
|---|---|---|---|---|---|
| CH4/STO-3G | 5 | 4 | 5 | 25 | 125 |
| HF/6-31G | 5 | 6 | 5 | 25 | 125 |

Five iterations across four threads is 1–2 each — badly unbalanced, and the
imbalance is the *first* thing to rule out if a measurement disappoints.
`collapse(2)` over `(i,j)` gives 25 and `collapse(3)` gives 125; writes stay
disjoint under both, since the collapsed indices are all output indices.

**This constraint is why "start with the builders" was attractive before and is
not now.** The builders are 91 independent calls — ideal granularity, but only
13.8 % of the time.

## The determinism question, stated precisely

The historical DFT jitter (`dft_xc_reduction_determinism`) came from a
**cross-thread reduction summed in completion order**. Neither site here has a
cross-thread reduction:

- **Builders** write their own freshly-allocated tensors — no sharing at all.
- **Residual nests** accumulate into a thread-private `acc` and write one disjoint
  `result(...)` slice. The inner summed loop stays serial within a thread, so its
  accumulation order is unchanged.

That is the same structural property that made the DFT J/K builds bitwise
thread-count-invariant. **It is a reason to expect determinism, not evidence of
it** — the DFT J/K case was verified, and this must be too.

## Steps

Ordered so the cheapest step can kill the expensive ones.

### O1 — **DONE.** The baseline exists: use `build-full`

`build-full` is already the tree this step called for — `-fopenmp` and
`-DUSE_OPENMP` on the compile line, `libgomp` linked, `-O3 -DNDEBUG` (with the
usual `-O1` pin on `generated_kernel_registry.cpp` only), g++-15, MAXORDER=4,
`PLANCK_CC_DRESS_OPERATORS=ON`, `PLANCK_CC_DRESSING=derived`,
`PLANCK_CC_ARBITRARY_LOWER_RANKS=ON`, `PLANCK_CC_SPIN_ADAPT=ON`.

Baseline recorded, HF/6-31G through the generated rank-3 route
(`PLANCK_RCCSDT_BACKEND=optimized`, `E_corr = -0.1319388410`, 15 iterations):

```
OMP_NUM_THREADS=1    78.03 s
OMP_NUM_THREADS=4    78.49 s     <- the number O2 must beat
```

Its check — *"if enabling OpenMP moves the time at all, stop and explain"* — is
satisfied: enabling it moves nothing, because CC has no pragmas to activate. The
same binary's direct-SCF HF path goes 2.70 s -> 0.81 s over the same thread range,
so the baseline is threaded everywhere except here.

**Do not re-derive this on a fresh tree.** Any O2 comparison must run against
`build-full`'s 78.03 s / 78.49 s, on the same binary configuration, or it prices
a compiler change as a threading win.

### O2 — **DONE (2026-08-30). 1.93x at 4 threads, bitwise deterministic, at the Amdahl ceiling.**

One `#pragma omp parallel for collapse(N) schedule(static)` on each of the 256
nests in `part1`, hand-edited into the generated file — the emitter was left
alone, per this step's instruction to get a number first.

Measured on `build-full`, HF/6-31G through the generated rank-3 route:

| variant | 1t | 2t | 4t | 8t | **4t speedup** |
|---|---|---|---|---|---|
| `collapse(2)` | 79.52 s | 54.76 s | 42.63 s | 38.42 s | **1.87x** |
| **`collapse(3)`** | 78.67 s | — | **40.85 s** | 37.37 s | **1.93x** |

**The Amdahl ceiling for a 64.6 % part at 4 threads is 1.94x.** `collapse(3)`
reaches 1.93x of it, so `part1` is now essentially fully parallel and there is
nothing further to win *inside it* — the remaining serial time is elsewhere, which
is O3's subject. 8 threads adds little (2.11x) because this machine has 4
performance cores.

**`collapse(3)` is the better choice and the reason is granularity, as predicted.**
`collapse(2)` gives 25 chunks over 4 threads; `collapse(3)` gives 125. The gap is
small but consistent across repeats (40.83-41.00 s vs 42.58-42.67 s, ±0.1 s), and
it costs nothing — the collapsed indices are all output indices, so writes stay
disjoint either way.

**Correctness: bitwise identical at every thread count**, for both variants and
against the pre-O2 unthreaded baseline. Every `E_corr`, `dE`, `rms(res)` and
`rms(step)` matches across all 15 iterations at `OMP_NUM_THREADS` = 1/2/4/8. The
reduction-free argument holds in practice, not just on paper.

**CPU utilization moved 99.1 % -> 160.6 %** at 4 threads, which is the cheap check
that the pragma is firing at all before any timing is interpreted.

#### Two build-mechanics traps worth carrying into O3/O4

**1. `make hartree-fock` silently reverts the edit.** `ccgen-planck-kernels` is an
unconditional dependency of the `hartree-fock` target, so any normal build
regenerates the file and wipes hand-edits — 256 pragmas back to 0, with no error.
Compile the object and link directly instead:

```bash
cd build-full
rm -f CMakeFiles/hartree-fock.dir/src/post_hf/cc/generated_kernel_registry.cpp.o
make -f CMakeFiles/hartree-fock.dir/build.make CMakeFiles/hartree-fock.dir/build
```

Verify the edit survived (`grep -c "pragma omp"`) **and** that it reached the
object (`nm ... | grep -c GOMP_parallel`) before trusting a timing.

**2. Do not copy the build tree.** `CMAKE_CACHEFILE_DIR` is absolute, so a copied
tree rebuilds into the *original* — silently corrupting the baseline you are
measuring against. Work in `build-full` with a backup of the generated file
(`cp` it aside, restore when done).

### O3 — **DONE (2026-08-30). 3.11x at 4 threads — and it found a dead-work defect worth more than the threading of the builders.**

Two changes, measured separately.

**O3a — thread all four triples parts.** Same `collapse(3) schedule(static)`, now
on all 806 nests (`part0` 256, `part1` 256, `part2` 256, `part3` 38). Checked
first, not assumed: every part opens with the same `i/j/k` header and every write
is `result(i, j, k, a, b, c)`, so the collapse is valid and the writes stay
disjoint everywhere.

**O3b — the builders turned out to be building everything twice.** Inspecting the
triples entry point to thread the 88 operator builds revealed that
`compute_ccsdt_triples_residual` emits them **twice**: once as 88 `const auto`
locals, then again inside the `ops` aggregate initializer. The two sets are
identical (verified by diffing the builder-symbol lists), and **the locals are
never referenced** — only `ops` is passed to the parts. They were pure dead work.

| step | 1t | 4t | 8t | **4t speedup** |
|---|---|---|---|---|
| O2 (`part1` only) | 78.67 s | 40.85 s | 37.37 s | 1.93x |
| **O3a** (all 4 parts) | 78.67 s | **27.98 s** | 23.13 s | **2.81x** |
| **O3b** (+ dead builders removed) | **73.79 s** | **23.70 s** | 18.80 s | **3.11x** |

Against the original unthreaded binary (80.92 s), the combined result is **3.41x
end to end**. CPU utilization went **99.1 % -> 359.9 %** at 4 threads — near-perfect
use of the machine's 4 performance cores.

**Correctness: bitwise identical throughout.** Every `E_corr`, `dE`, `rms(res)`
and `rms(step)` matches across all 15 iterations at `OMP_NUM_THREADS` = 1/2/4/8
and against the original unthreaded baseline, for both O3a and O3b. Removing the
dead builders changes no number, which is itself the proof they were unused.

O3a lands almost exactly on the modelled 2.74x for the three triples parts. The
extra came from O3b, which the model did not know about.

#### The dead-builder defect, for O4

It is an **emitter** bug, not an artifact of this file.
`planck_tensor_cpp.py:1173-1178` emits the `const auto` intermediate builds
unconditionally, and *then* — for any kernel above `_KERNEL_CHUNK_TERMS` —
delegates to `_emit_chunked_kernel`, which builds the same operators into the
`<kernel>_ops` struct. H5 added the struct hoist but did not remove the emission
it superseded, so **every chunked kernel builds its operators twice**.

Worth 4.9 s serial / 4.3 s at 4 threads on HF/6-31G here — about **6 %** — and it
should scale with operator count, so rank 4 (894 operators against 88) is likely
to gain considerably more. **Fix it in the emitter as part of O4**, not as a
separate hand-edit: the chunked path should skip the standalone emission entirely.

#### What is left after O3

At 359.9 % of 4 cores there is little parallel headroom left on this machine. The
remaining serial work is the doubles/singles residuals (~1.2 %) and their
standalone builders, which are genuinely used and genuinely small. **The builders
inside the triples path never needed threading at all** — they needed deleting.

### O4 — **DONE (2026-08-30). Both changes are in the emitter; default emit unchanged apart from the dead builds.**

**The duplicate-builder fix.** `_emit_kernel` now computes `chunked` before
emitting the intermediate builds and skips them on that path
(`planck_tensor_cpp.py`), because `_emit_chunked_kernel` builds the same
operators into the `<kernel>_ops` struct that the parts actually read. The stale
comment claiming "the intermediate builds and amplitude-view bindings are
re-emitted per part" is corrected — H5 stopped that being true for the builds.

**The pragma.** `emit_planck_term` takes `omp_collapse`, threaded through
`_emit_terms` (both the fused and unfused paths) from a `CCGEN_OMP_COLLAPSE` env
var, following `_fuse_loops_setting`'s established pattern — an env var rather
than a `print_cpp_planck` parameter, since W3's condition is that a knob earns a
parameter once it is staying. Default 0 emits no pragma. A guard
(`len(free) >= omp_collapse`) means a nest that is too shallow is simply left
alone; verified across all 948 emitted pragmas that none collapses more levels
than it has.

**Measured, HF/6-31G, generated rank-3 route:**

| emit | 1t | 4t | 8t | **4t speedup** |
|---|---|---|---|---|
| default (`CCGEN_OMP_COLLAPSE` unset) | **76.68 s** | 75.82 s | — | 1.01x |
| `CCGEN_OMP_COLLAPSE=3` | 74.37 s | **23.11 s** | 18.31 s | **3.22x** |

The default row is the shipping path and carries the dead-builder fix alone:
**78.67 s -> 76.68 s, 2.6 %, with no threading**. The threaded row reproduces O3's
hand-edit and is slightly better (23.11 s against 23.70 s) because the emitter also
annotates the singles/doubles residuals, which O3 did not hand-edit.

**Correctness.** Energies bitwise identical at 1/4/8 threads and against the
original unthreaded baseline, on both rows. The emitted TU with the pragma
disabled differs from the pre-O4 emit **only** by the 88 removed dead builder
lines (diff checked line by line: 91 removed lines, all `const auto W_... =
build_W_...`, the section comment, and a blank). Regression cases
`lih_rccsdt_generated_sto3g`, `ch4_rccsdt_generated_sto3g` and `be_rccsdtq_sto3g`
all pass; the ccgen Python suite is 876 passed / 0 failed, unchanged.

**New gate: `test_chunked_kernel_builds_once.py`.** The duplicate build was
invisible to every existing gate, and that is the point worth carrying — it is
semantically a **no-op**, so energies, residuals and every value gate were correct
while the work was being done twice. Only wall-clock or a reading of the emitted
text could see it. The gate reads the text: no operator built twice in a chunked
entry point, and no `const auto` operator build before the `ops` aggregate (stated
separately, because a future change could emit one *without* a matching struct
entry and slip past the duplicate check). Mutation-verified — restoring the
unconditional emission turns both red, while the vacuity check stays green.

## What this must not do

- **Do not quote the 3.86x from the prior scope.** It came from the 45.1 %/53.7 %
  split, which no longer holds. Threading everything models 4.00x at 4 threads;
  **measured so far is 1.93x from `part1` alone**, and the modelled combined figure
  for the three triples parts is 2.74x.
- **Quote the same-binary serial reference, not the pre-O2 baseline.** O2 is
  1.93x against 78.67 s (the threaded binary at `OMP_NUM_THREADS=1`), which is the
  honest comparison; against the 80.92 s pre-O2 binary it would read 1.98x and
  would be pricing a rebuild as a threading win.
- **Do not use a CPU-utilization percentage alone as evidence that CC is
  unthreaded.** It cannot distinguish "no pragmas here" from "no OpenMP in this
  binary" — the prior scope's central number had exactly that ambiguity. Pair it
  with a **positive control** on the same binary (direct-SCF HF/cc-pVTZ threads
  3.3x), and confirm `-fopenmp` reaches the compile line. Pick a control that is
  actually bound by a threaded path and measure it **warm** — a cold first run
  inflates the ratio, which is how an earlier draft of this doc reported 5.6x for
  what is really 1.06x.
- **Do not thread the builders first.** 13.8 %, caps at 1.12x.
- **Do not introduce a cross-thread reduction.** If a future term shape needs one,
  it must sum in fixed thread-index order, never completion order — that is the
  DFT jitter defect exactly.
- **Do not accept "energies match to tolerance."** The claim is bitwise identical
  across `OMP_NUM_THREADS`, as the DFT J/K builds were verified.

## Key code locations

| what | where |
|---|---|
| the hot nests (63.6 % + 19.6 %) | `compute_ccsdt_triples_residual_part{0,1}`, generated `ccsdt_arbitrary_planck_generated.cpp` |
| the emitter that writes them | `_emit_chunked_kernel`, `python/ccgen/emit/planck_tensor_cpp.py` |
| the pragma form to copy | `emit_term_tiled`, `python/ccgen/emit/cpp_loops.py:331` |
| the builders (13.8 %) | the `<kernel>_ops` struct, same emitter |
| the determinism precedent to imitate | DFT J/K builds, `src/dft/driver.cpp` |
| the determinism defect to avoid | `dft_xc_reduction_determinism` note |
| where the split came from | `docs/CCGEN_ARBITRARY_HARNESS_COST.md` |
| the OpenMP-enabled baseline tree | `build-full` (g++-15, Release, MAXORDER=4, dressed+merged) |
| the positive control | direct-SCF HF/cc-pVTZ via `build-full/hartree-fock` — 3.3x at 4 threads |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
