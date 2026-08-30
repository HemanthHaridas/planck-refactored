# Scope: threading the generated CC path

**Scope for in-flight work. Not started.** Re-measured 2026-08-29 against the
post-merge tree, which **invalidated the estimate and the recommendation** carried
in `CCGEN_ARBITRARY_HARNESS_COST.md`. That document told its reader to re-measure
the split before relying on its figures; doing so changed the answer.

## What changed, and why the old plan is now wrong

The prior scope was written when builders were **45.1 %** of runtime and advised
*"start with the builders: better granularity, no write sharing"*. Two landed
fixes (the chunk hoist, then `merge_transposes`) removed most of that work.
Measured now, on the same binary configuration:

| | prior (post-chunk-hoist) | **now (post-merge)** |
|---|---|---|
| `build_W_*` builders | 45.1 % | **13.8 %** |
| residual nests | 53.7 % | **86.2 %** |

(HF/6-31G, 21 341 leaf samples. CH4/STO-3G agrees: 17.2 % / 82.8 %.)

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
| `compute_ccsdt_triples_residual_part1` | **63.6 %** |
| `compute_ccsdt_triples_residual_part0` | 19.6 % |
| `..._part2` | 1.5 % |
| doubles + singles residual | 1.2 % |
| all 91 builders combined | 13.8 % |

Threading the three triples parts (84.7 %) models **2.74x at 4 threads**; `part1`
alone models 1.91x.

## Two corrections to the prior scope's evidence

**1. The "98.8 % CPU on 8 cores" observation does not prove what it was used for.**
It was taken on a tree where `OpenMP_CXX_FLAGS:STRING=NOTFOUND` and `-DUSE_OPENMP`
is absent from the compile line — **every OpenMP pragma in Planck is inert in that
binary**, not just CC's absent ones. The default `build/` tree does have
`-fopenmp`. The observation is therefore consistent with "CC has no pragmas" and
equally with "this binary has no OpenMP at all", and cannot distinguish them.

**The underlying claim is nevertheless true**, confirmed independently by
inspection rather than by that measurement: **0 `pragma omp` in
`src/post_hf/cc/*.{cpp,h}` and 0 in the emitted kernels**, against 8+ other files
under `src/` that carry them. Re-verify with `grep`, not with a CPU-utilization
number, and make sure any before/after timing uses a tree that actually found
OpenMP.

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

### O1 — establish a trustworthy baseline (~S)

Build the dressed generated tree **with OpenMP actually found** (confirm
`OpenMP_CXX_FLAGS` is not `NOTFOUND` and `-DUSE_OPENMP` reaches the compile line —
the prior scope's baseline had neither). Record CH4 and HF/6-31G wall-clock at
`OMP_NUM_THREADS=1`.

*Verify:* energies match the committed values (`ch4_rccsdt_generated_sto3g`
`-0.0791116825`, HF/6-31G `-0.1319388410`) and the timings match the current
single-threaded numbers, i.e. enabling OpenMP alone changes nothing.

**If enabling OpenMP moves the time at all, stop and explain that first** — it
would mean some other pragma in the binary was previously inert and is now live,
which changes what any later comparison measures.

### O2 — thread `part1` only, measured (~S)

One `#pragma omp parallel for collapse(2)` on the hottest part. Hand-edit the
generated file first; do not touch the emitter until a number exists.

*Verify:* wall-clock at 1/2/4/8 threads, and **energies bitwise identical at every
thread count**. Expect ≤1.91x (part1 is 63.6 %). Below ~1.3x at 4 threads, suspect
load imbalance or task granularity and try `collapse(3)` before concluding the
lever is absent.

### O3 — extend to part0/part2 and the builders, if O2 justifies it (~S)

Same pragma shape. Models 2.74x combined for the three triples parts; the builders
add at most 1.12x on their own and are the last thing to do, not the first.

### O4 — move it into the emitter (~M, only after O2/O3 measure well)

Teach `planck_tensor_cpp.py` to emit the pragma, borrowing the form already in
`cpp_loops.py:331`. **One knob, not seven** — the same condition W3 set for the
dressing axis. A build option defaulting OFF is acceptable while it is new; a
per-nest tuning surface is not.

*Verify:* the emitted TU is byte-identical to today's when the option is off, and
the O2/O3 numbers reproduce when it is on.

## What this must not do

- **Do not quote the 3.86x from the prior scope.** It came from the 45.1 %/53.7 %
  split, which no longer holds. The measured figure is **4.00x** for both sites at
  4 threads, but the achievable one is bounded by the residual's granularity.
- **Do not use a CPU-utilization percentage as evidence that CC is unthreaded.**
  Use `grep` for pragmas, and check the build actually found OpenMP.
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

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
