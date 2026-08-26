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

### R1 — LANDED (2026-08-26): the generated path is gated, pinned to its OBSERVED behaviour

`ch4_rccsdt_generated_sto3g` runs the same input as `ch4_rccsdt_sto3g` but forces the generated
path and asserts what it actually does.

It pins the **defect**, not the fix:

```
expected_exit_code: 1
contains:
  "Routing the ccgen-generated rank-3 CCSDT kernels through the arbitrary-order harness"
  "Running generated arbitrary-order RCC tensor kernels (rank=3"
  "did not converge within 100 iterations"
  "E_corr=-0.0565650696"
```

Pinning the observed value rather than the correct one is deliberate. A gate asserting
`-39.8058445240` would fail today for three different reasons at once — wrong exit code, missing
`CCSDT Energy` line, wrong number — and could not distinguish "converged to the wrong answer" from
"did not converge". This one says exactly which behaviour is present, so a change in *any* of
those is visible.

**When the defect is fixed this case must FAIL**, and the JSON says so in a `known_failure` field:
flip `expected_exit_code` to 0, drop the non-convergence strings, and assert
`rccsdt_total_energy == -39.8058445240` to 1e-07. That inversion is the signal R1 exists to
provide.

Two mechanics were needed:

- **Per-case `env`** (`run_regressions.py`). The generated path is reachable only via
  `PLANCK_RCCSDT_BACKEND=optimized`; there is no input keyword. `subprocess.run` now takes
  `env={**os.environ, **case["env"]}` when a case declares one, `None` otherwise — so every
  existing case is unaffected (verified: `water_rccsdt_sto3g` still passes).
- **`requires_build_option: PLANCK_CC_ARBITRARY_LOWER_RANKS`**, reusing the mechanism the UCC
  cases added. In a default build it reports
  `[SKIP] ch4_rccsdt_generated_sto3g (needs -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON)` rather than
  failing.

**Why the sibling case could not be extended instead:** `ch4_rccsdt_sto3g` asserts
`kernels=hand-optimized`, and that assertion is correct for the hand-written path it guards. The
two need different `contains` and different exit codes, so they are two cases over one input.

*Verify:* the energy in `contains` is the value measured in W4.2a, and the run must log the
routing line — a gate that passed without it would be pinning the hand-written path again, which
is the exact failure this case exists to prevent.

### R2 — stop rank 3 from breaking rank 4 (~S)

Independent of the kernel defect: with `ARBITRARY_LOWER_RANKS=ON`, rank 4 seeds from a
non-converged rank-3 solve. A warm-start seed should not be taken from a solve that failed to
converge.

*Verify:* `be_rccsdtq_sto3g` passes in **both** builds. This is worth doing first — it is small,
it is separable, and it stops one broken path from spreading.

### R3 — ANSWERED (2026-08-26): the kernel is WRONG, not cold-started into a bad basin

R3 was scoped as "seed rank 3 from converged rank 2 and see". That turned out to be
unimplementable from input files — **RCC has no rank-2 generated keyword**, deliberately
(`io.cpp:675`: "a generated rank-2 RCC path would have no consumer", since hand-written covers
ranks 2-3). Seeding from hand-written RCCSD would need an amplitude-type conversion, i.e. code.

A better discriminator was already in the tree. **The Be CCSDTQ run performs a rank-3 sub-solve
internally**, and on Be that sub-solve *converges*:

```
RCCSDTQ[TENSOR] :(rank 3) Iter : 12  E_corr=-0.0139349127  rms(res)=1.868e-12
RCCSDTQ[TENSOR] : Warm-started rank 4 from converged rank 3 (seeded T1..T3).
```

12 iterations, residual 1.9e-12. So the kernel is **not** universally non-convergent, and the
CH4 non-convergence is not the primary defect.

But against the hand-written path on the same system:

| system | hand-written | generated rank-3 | gap |
|---|---|---|---|
| **Be** | −0.0517702884 | **−0.0139349127** (converged, res 1.9e-12) | **3.78e-02** |
| CH4 | −0.0791116825 | −0.0565650696 (never converges) | 2.25e-02 |

**The kernel converges to a wrong answer.** A residual of 1.9e-12 alongside a 3.8e-02 energy
error means it is driving a *different equation* to self-consistency — the residual it zeroes is
not the CCSDT residual.

That answers R3 against the benign branch: **not a seed problem, not a basin problem, not the
iteration cap.** Non-convergence on CH4 is a second symptom of one defect, not the defect.

Two consequences:

- **R2 is more urgent than first scoped.** Rank 4 in the arbitrary build is seeded from amplitudes
  that are *converged and wrong*, which is worse than unconverged ones — nothing downstream can
  detect the difference.
- **The vault claim is doubly refuted.** `1986d0c` states the residuals are "BITWISE IDENTICAL
  across both harnesses at identical inputs". A converged-but-wrong solution on Be is direct
  evidence against that, independent of the CH4 reproduction.

### R4 — where does the wrongness enter? (~L, five steps)

R3 established the kernel converges to a **wrong self-consistent answer**: on Be, `rms(res) =
1.9e-12` at an energy 3.78e-02 from the hand-written result. The residual being driven to zero is
not the CCSDT residual.

There are four places that can happen, and they are separable. The ladder narrows in that order,
cheapest first.

```
  (a) equations  ->  (b) emission  ->  (c) block binding  ->  (d) solver/denominators
      ccgen           C++ codegen       mo_blocks feed        the harness
```

**(d) is already largely exonerated** — rank 2 and rank 4 converge correctly through the same
solver, DIIS and denominators (see the localisation table above). Do not start there.

#### Fixture: use Be, not CH4

Be converges in 12 iterations at ~0.01 s; CH4 takes 100 at 3.4 s and never converges. A stable
wrong value is a far better comparison target than a moving one. The Be rank-3 sub-solve is
reachable inside the CCSDTQ run in the `ARBITRARY_LOWER_RANKS=ON` build.

| | E_corr |
|---|---|
| Be hand-written CCSDT | **−0.0517702884** |
| Be generated rank-3 | **−0.0139349127** |

##### R4.1 — DONE (2026-08-26): the EQUATIONS are clean; the defect is below ccgen

PySCF lives in `tests/pyscf/.venv` (2.13.0), not in any conda env — that is why the gate skipped
earlier and why the skip was not evidence of anything.

```
$ tests/pyscf/.venv/bin/python -m unittest \
    ccgen.tests.test_reference_vs_pyscf.ReferenceVsPyscfTests.test_ccgen_ccsdt_reaches_fci_limit
ok    (10.6 s)
```

**It RAN and passed** — the generated CCSDT residual solved in Python reaches the FCI total to 8
places on H3/6-31g. All three CCSDT FCI gates ran (`ccgen`, `diagram_engine`,
`diagram_weighted`), and the whole reference suite went from 20 skipped to **31/32 passing**.

Layer **(a) equations is clean.** The defect is downstream of ccgen.

**And the suite hands over the decisive comparison for free.**
`test_ccsdt_spin_adapted_solves_between_ccsd_and_fci` solves the generated, spin-adapted CCSDT
equations on **Be/STO-3G — the exact system R3 used**:

| Be/STO-3G CCSDT `E_corr` | value | |
|---|---|---|
| generated equations, solved in **Python** | −0.0517702744 | |
| **hand-written C++** | −0.0517702884 | agree to **1.40e-08** |
| **generated kernel, C++** (arbitrary harness) | **−0.0139349127** | off by **3.78e-02** |

The same equations that produce the right answer in Python produce a wrong one through the C++
kernel. That narrows the defect to **(b) emission, (c) block binding, or (d) packing** — and
since ranks 2 and 4 converge correctly through the same solver, (d)'s solver half is out too.

This also settles `1986d0c`'s "BITWISE IDENTICAL across both harnesses at identical inputs" claim:
the two harnesses demonstrably do not agree on Be. **Do not carry that claim forward.**

*Environment note for whoever runs the rest of this ladder:* use
`tests/pyscf/.venv/bin/python`, not the default interpreter. A `pyscf not importable` skip looks
identical to a pass in the summary line.

##### R4.2 — compare the emitted kernel against the Python residual at identical amplitudes (~M)

R4.1 already gave the *energy*-level comparison (Python −0.0517702744 vs C++ −0.0139349127 on
Be). This step goes one level down, to the residual, because an energy difference does not say
which tensor element is wrong.

Take one fixed, non-trivial amplitude set; evaluate the triples residual two ways — through
`residual_eval.residual_einsum` on the generated Python terms, and through the emitted C++ kernel
— and diff element-wise. Use the **converged Python amplitudes** from R4.1's Be solve as the
input, so both sides are evaluated at a point where the correct residual is known to be ~0: any
non-zero element on the C++ side is then directly the defect, with no solver dynamics in the way.

`1986d0c` claims these are "BITWISE IDENTICAL across both harnesses at identical inputs". **That
claim is now suspect and must be re-measured, not inherited** — R3's converged-but-wrong result is
evidence against it.

Precedent for the C++ side: `PLANCK_CC_T3_TIME` (`tensor_backend.cpp:2350`) already reaches into
the residual evaluation behind an env var, and the B5 work used a "fixed-point probe" that
injected known-correct amplitudes into live C++ state and evaluated once. Reuse that shape.

*Verify:* either a maximum element-wise difference at machine precision (layers b+c clean, defect
is elsewhere), or the disagreeing elements localised by index block.

##### R4.3 — if they disagree, which BLOCK? (~M)

The residual is built from ERI blocks (`oooo`, `ooov`, `oovv`, `ovov`, `ovvo`, `ovvv`, `vvvv`).
Report the diff per block rather than as one norm.

Two known-shaped hazards to check first, both of which have bitten this codebase:

- **physicist vs chemists convention** — `rebind_physicist` exists precisely because generated
  kernels index `<pq|rs>` while `mo_blocks` is chemists. A single un-rebound block gives exactly
  this signature: self-consistent, converged, wrong.
- **`ovvo` vs `ovov`** — O6 already found the same-spin fold is not applied in the UCC emitter.
  Rank-3 RCC uses both blocks.

*Verify:* the defect named as a block plus a convention, or an explicit statement that all blocks
agree.

##### R4.4 — if the blocks agree, check the amplitude packing (~M)

If (b) and (c) are clean but the solve is still wrong, the remaining suspect is how amplitudes are
packed into and out of the arbitrary-order state at rank 3 specifically — `1986d0c` changed
exactly this (wedge-packed vs dense) and rank 3 is the only rank where both representations have
ever been in play.

*Verify:* round-trip a known amplitude set through `pack`/`unpack` at rank 3 and compare.

##### R4.5 — write the gate, then fix (~S)

Whatever R4.2-R4.4 names, add a unit-level gate at that layer before fixing it. R1 gates the
end-to-end symptom; it will not tell a future reader *which* layer regressed.

*Verify:* the new gate is red on the current tree, and `ch4_rccsdt_generated_sto3g` inverts as
its `known_failure` field describes once the fix lands.

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
