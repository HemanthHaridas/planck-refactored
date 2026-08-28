# Why can't the scaling ladder be re-run under `--dressing derived`?

**Because the dressed rank-3 kernel has no reachable caller, and the probe that
would time it is on a code path that no longer executes.** Not a measurement
result — a blocker found while setting the measurement up, recorded so the next
attempt does not re-derive it.

`CCGEN_KERNEL_SCALING_SCOPE.md` and `CCGEN_KERNEL_PERFORMANCE.md` both recommend
re-running the six-point rank-3 ladder with `--dressing derived` **before**
consuming `_optimal_contraction_order` in the emitter, on the grounds that the
two fixes may overlap. That recommendation stands. It cannot currently be
carried out.

## The two-sided trap

The `PLANCK_CC_T3_TIME` probe lives inside `run_tensor_rccsdt_impl`, in its
`if (use_generated_kernels)` branch (`tensor_backend.cpp:2350`). Reaching that
branch requires `PLANCK_RCCSDT_BACKEND=optimized`. But `optimized` no longer
runs that code:

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

## The deeper half: dressing and the harness do not meet

This is the part that matters beyond the probe, because it is about which code
*runs*, not which code is *timed*.

| | dressed by `PLANCK_CC_DRESS_OPERATORS`? | runs at rank 3? |
|---|---|---|
| `ccsdt_planck_generated.cpp` (plain per-method TU) | **yes** | **no** |
| `ccsdt_arbitrary_planck_generated.cpp` (arbitrary companion) | **no** — CMake:72-74 | yes |

Measured on two trees differing in exactly one flag (cache-diffed):

| arm | `ccsdt_planck_generated.cpp` | `build_W` call sites |
|---|---|---|
| undressed | 14497 lines | **0** |
| dressed | 21876 lines | **1706** |

So dressing demonstrably reaches the plain TU. But the only backend that ever
called that TU's residual kernel was `run_tensor_rccsdt_impl(..., use_generated
= true)`, and `CCGEN_RANK3_KERNEL_AND_SOLVER.md` deliberately rerouted
`optimized` away from it — correctly, because that solver's symmetry-packed
representation is incompatible with the dense one the kernels are emitted for
(it converged to `-7.56e-05`).

The reroute fixed correctness and, as a side effect, orphaned the dressed
rank-3 kernel. `grep compute_ccsdt_triples_residual src/` returns only sites
inside the unreachable branch.

**This is the third instance of the same defect class in this subsystem**: a
generated kernel that compiles, links, and is never executed. The first was
`compute_ccsdt_triples_residual` having no caller for months; the second was
`emit_factorized_translation_unit` having no production caller. Both are
recorded in `CCGEN_TWO_DRESSING_ROUTES.md`. The pattern is that **linkage is
not execution**, and nothing in the build fails when they diverge.

## What this means for W5's 3.12x / 3.61x

W5 measured `PLANCK_RCCSDT_BACKEND=optimized` with
`ARBITRARY_LOWER_RANKS=ON` — i.e. end-to-end solve time through the
**arbitrary-order harness**, the same configuration
`lih_rccsdt_generated_sto3g` and `ch4_rccsdt_generated_sto3g` pin.

By the table above, the arbitrary companion TU is **not** dressed. That leaves
two possibilities, and this document does **not** resolve which:

1. `--dressing derived` reaches the arbitrary TU by some path the CMake comment
   does not describe, in which case the comment is wrong; or
2. the 3.12x/3.61x came from something other than dressed rank-3 triples.

**Do not treat W5's numbers as retired on the strength of this note** — they were
measured, twice, with energies matching the undressed baseline, and the comment
at CMakeLists.txt:72-74 is prose, not a gate. But they should not be quoted as
"dressing speeds up the generated kernel" until it is established which TU the
dressed builders were actually executing from. That is one `nm`/breakpoint
check, and it is the first thing to do here.

## What to do, cheapest first

1. **Settle where W5's speedup came from.** Build W5's exact configuration
   (`optimized` + `ARBITRARY_LOWER_RANKS=ON` + dressing) and check whether any
   `build_W_*` symbol is reached at run time — `nm` the binary for the symbols,
   then confirm execution with a counter or a breakpoint. This decides whether
   the CMake comment or the measurement is wrong, and everything else depends on
   it.
2. **If the arbitrary TU is genuinely undressed**, then dressing has no effect
   on the production rank-3 path at all, and "wire the derivation route" is not
   finished. Extending `--dressing` to the arbitrary companions is the real
   task; CMake:73-74 names the obstacle (every method's dressed builders would
   share one signature and collide in the co-including registry), which is a
   naming problem, not a hard one.
3. **Only then re-run the ladder.** Move `PLANCK_CC_T3_TIME` to wherever the
   generated residual is actually evaluated — the arbitrary harness — or add an
   equivalent probe there. Timing a kernel that does not run is worse than not
   measuring.

## What was set up and is reusable

- Two configured trees differing in exactly one flag, cache-diffed:
  `build-ladder-{undressed,dressed}` (`MAXORDER=3`, `SPIN_ADAPT=ON`,
  `DRESSING=derived`). Both build clean.
- Six ladder inputs matching the scope's table (`bh3_sto3g`, `ch4_sto3g`,
  `hf_631g`, `h2o_631g`, `bh3_631g`, `c2h4_sto3g`), spanning `o/v` 0.36-1.33.
  The `ch4` copy preserves its load-bearing `use_diis .true.`
- A harness that reports *why* a point yields no timing rather than dropping it
  silently — which is how this blocker surfaced instead of producing a
  five-point ladder with one unexplained gap.

## Constraints that still hold

- `PLANCK_RCCSDT_BACKEND=tensor` is the **hand-written** path. Only `optimized`
  selects generated kernels. Checking the backend marker before believing any
  number is a standing rule here and it is what caught this.
- Energies must be identical between arms, per point. Verified for the
  hand-written path (BH3/STO-3G `-0.0533629199` in both arms, as expected since
  dressing does not touch it).
- The determinant backstop (`nso <= 16 && ndet <= 10000`) binds the hand-written
  arm only; all six ladder points clear it regardless.

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`,
which are canonical.
