# Why does the generated rank-3 CCSDT kernel not converge, when rank 2 and rank 4 do?

**Defect scope. Not started.** Opened by W4.2a
(`docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md`), which set out to compare
derivation-dressed kernels against a baseline and found the baseline path itself does not work.

**This blocks W4, and therefore blocks wiring the derivation route to production.** Dressing
cannot be evaluated on top of a route that does not converge.

---

## The observation

`ch4_rccsdt_sto3g`, the only in-tree case that clears `choose_determinant_backstop`, built with
`-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON` and run with `PLANCK_RCCSDT_BACKEND=optimized`:

| path | E_corr | outcome |
|---|---|---|
| hand-written tensor | **−0.0791116825** | converged, 24 iterations, 0.18 s |
| generated, arbitrary harness | **−0.0565650696** | **never converges**, 100-iteration cap |

Gap **2.26e-02** against a 1e-07 tolerance. Both DIIS settings reach the same value; with DIIS on
the iteration freezes exactly (`rms(step) = 0.000e+00`, residual bit-identical across iterations),
with it off the residual decays ~3.5 %/iteration while `E_corr` sits at −0.05656 from iteration 45
onward. It is converging to a fixed point that is not the CC solution.

## The defect is local to rank 3 — measured, not assumed

Same runtime (`generated_arbitrary_runtime.cpp`), same solver (`solver_arbitrary.cpp`), same DIIS,
same denominators. Only the kernel differs:

| rank | build | result |
|---|---|---|
| 2 (`ccsd` on CH4) | `ARBITRARY_LOWER_RANKS=ON` | **works** — `−39.8056549601` |
| 3 (`ccsdt` on CH4) | `ARBITRARY_LOWER_RANKS=ON` | **fails** — no convergence |
| **4** (`ccsdtq` on Be) | **default build** | **works** — `−14.4036550465`, matches its gate, 13 iterations |
| 4 (`ccsdtq` on Be) | `ARBITRARY_LOWER_RANKS=ON` | **fails** — and the log says why (below) |

Rank 4 working in the default build is the load-bearing measurement: it exercises the same
arbitrary-order harness end to end, so the harness, the Jacobi/DIIS update, the denominators and
the packing are all exonerated. **What is left is the rank-3 kernel itself.**

### Rank 4 fails only because it is seeded from rank 3

```
[INF] RCCSDTQ[TENSOR] : Warm-started rank 4 from converged rank 3 (seeded T1..T3).
[ERR] run_generated_arbitrary_order_iterations: ... did not converge within 100 iterations.
```

With `ARBITRARY_LOWER_RANKS=ON`, `generated_floor` drops to 3 (`rccgen.cpp:116`), so rank 4 warm-
starts from the broken rank-3 solve and inherits its bad amplitudes. In the default build rank 4
cold-starts and converges in 13 iterations.

**So the option intended to enable rank 3 currently breaks rank 4 as a side effect.** That is a
second, independent consequence worth fixing even if rank 3 is deferred.

## What was believed, and why it went unnoticed

`vault/Status/Completion.md:311` records:

> `optimized` now lands at +1.44e-08 (5247× error reduction) and agrees with the hand-written path
> to 1.0e-10

from commit **`1986d0c`** (2026-08-16), *"the generated rank-3 CCSDT kernel is correct; its solver
was not"*.

**That result does not reproduce at the commit that made it.** `1986d0c` was checked out, built
with the identical configuration, and run on the identical input: `E_corr = −0.0565650696`,
**bit-identical to HEAD**, same 100-iteration failure.

So this is **not a regression**. The six UCC commits that touched the shared runtime afterwards
(`fbf1be6`, `9a2e2f0`, `49cebdc`, `9d8c483`, `ca8bcb9`, `7ca1465`) are exonerated by that test —
`ca8bcb9`'s "RCC untouched" claim holds.

### The gate that could not see it

`1986d0c` added `ch4_rccsdt_sto3g` as *"the ONLY in-tree rank-3 case that clears
`choose_determinant_backstop`"*. Its assertions:

```json
"contains": [
  "RCCSDT[TENSOR] : Standalone restricted tensor RCCSDT converged in",
  "kernels=hand-optimized"
]
```

It pins **`kernels=hand-optimized`** — the hand-written path. The case has been green throughout
while never once executing the generated kernel it was added to protect.

This is the same defect shape that commit diagnosed, one level up. Its own message says *"the
hand-written side was treated as a validated reference and never was"* and *"that coverage hole is
why this took five falsified hypotheses"* — and the fix then shipped with a gate carrying exactly
that hole.

## What to establish

Each step keeps the default build byte-identical; rank 4 there is the reference that must not move.

### R1 — a gate that fails today (~S, BLOCKING)

Before any fix, add a case that runs the **generated** rank-3 path and asserts the energy. Without
it the next fix has the same blind spot as the last one.

*Verify:* the new case is RED on the current tree, and its failure names the energy rather than a
missing log string. Then correct `Completion.md:311`, which currently states a validated result
that does not reproduce.

### R2 — stop rank 3 from breaking rank 4 (~S)

Independent of the kernel defect: with `ARBITRARY_LOWER_RANKS=ON`, rank 4 seeds from a
non-converged rank-3 solve. A warm-start seed should not be taken from a solve that failed to
converge.

*Verify:* `be_rccsdtq_sto3g` passes in **both** builds. This is worth doing first — it is small,
it is separable, and it stops one broken path from spreading.

### R3 — is it the kernel or the seed? (~M)

Rank 3 cold-starts even with `warm_start=on`, because `rank-1 >= generated_floor` is `2 >= 3`
(`rccgen.cpp:116`, unchanged since `b35fae3` — this is not a regression). Iteration 1 begins at
`E_corr = −0.0107`; the hand-written path begins from a converged RCCSD at −0.0789.

Seed the rank-3 solve from the converged rank-2 amplitudes and re-run.

- Converges to −0.0791116825 → the kernel is **correct** and the defect is the cold start plus a
  basin the solver cannot escape. Fix is the seed.
- Converges to −0.05656 again → the kernel is **wrong**, and R4 follows.

*Verify:* a stated answer with the energy behind it. This is the cheapest discriminator and it
should come before any kernel-level investigation.

### R4 — only if R3 says the kernel is wrong (~L, research)

Compare the generated rank-3 residual against the hand-written one at **identical amplitudes**.
`1986d0c` claims they are "BITWISE IDENTICAL across both harnesses at identical inputs" — that
claim is now suspect too and must be re-measured, not inherited.

The instrument exists: `test_factorize_value_preservation` evaluates residuals symbolically, and
`residual_eval.residual_einsum` can evaluate a term set numerically. A term-by-term diff between
the generated rank-3 triples residual and the hand-written one localises the defect to specific
terms, the way the factorizer's own value gate localised its two defects.

*Verify:* the disagreeing terms named, or a statement that they agree — in which case the defect is
in the packing/denominators for rank 3 specifically, and R3's answer was incomplete.

## What NOT to do

- **Do not widen the tolerance or raise the iteration cap.** Measured: the residual decays while
  `E_corr` stays at −0.05656 from iteration 45. More iterations buy a tighter residual at the
  wrong fixed point. This was tested.
- **Do not blame DIIS.** With DIIS off the iteration still heads to −0.05656; DIIS only changes
  *how* it stalls there (exact freeze vs slow crawl).
- **Do not look for a regression in the UCC commits.** `1986d0c` reproduces the failure
  bit-identically; the runtime changes after it are exonerated.
- **Do not trust `kernels=hand-optimized` as evidence the generated path ran.** That string is what
  made this invisible for ten days. Any new gate must positively identify the generated path —
  the run logs `Routing the ccgen-generated rank-3 CCSDT kernels through the arbitrary-order
  harness` when it takes it.

## Key code locations

| what | where |
|---|---|
| the failing convergence test | `run_generated_arbitrary_order_iterations`, `src/post_hf/cc/generated_arbitrary_runtime.cpp:321` |
| the update whose step goes to zero | `solver_arbitrary.cpp:305-320` (DIIS extrapolate, then `update_delta`) |
| tolerances, inherited from SCF | `rccgen.cpp:136-137` (`_tol_energy` / `_tol_density`, both 1e-10) |
| iteration cap | `rccgen.cpp:134`, `max(get_max_cycles(nbasis), 100)` |
| warm-start floor that skips rank 3 | `rccgen.cpp:116`, `rank - 1 >= generated_floor` |
| the generated rank-3 TU | `build-arb/generated/cc/ccsdt_arbitrary_planck_generated.cpp` (7247 lines) |
| the gate that could not see this | `ch4_rccsdt_sto3g` in `tests/regression_cases.json` |
| the claim to correct | `vault/Status/Completion.md:311` |
| what this blocks | `docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md` (W4) |

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
