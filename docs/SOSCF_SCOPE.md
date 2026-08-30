# Scope: second-order SCF (SOSCF)

**Scope for in-flight work. Not started.** Opened 2026-08-30 from a measurement
in the committed cluster data, not from a wish for a feature.

## The problem, measured

`scale.json` (notch386, HF/6-31g, water chains, serial, `engine os`):

| nb | iterations | s/iter | total |
|---|---|---|---|
| 104 | 30 | 4.8 | 143 s |
| 156 | 39 | 10.1 | 395 s |
| 208 | 47 | 16.0 | 754 s |
| 312 | 70 | 38.0 | 2662 s |
| 416 | **91** | 68.8 | **6263 s** |

**Iteration count triples as the system grows** — 30 → 91 over a 4× basis range,
on the same basis, same guess, same convergence thresholds. That is a **3×
multiplier on total cost that no amount of parallel or kernel work touches**: at a
flat 30 iterations, nb=416 would be ~2065 s instead of 6263 s.

Two more data points on the same axis:

- **DFT at nb=416 does not converge at all** — `rc=1, "RKS did not converge"` at
  the serial baseline. It ran, iterated, and hit `max_cycles`. The HPC rescope
  correctly classifies this as a convergence-robustness item, not a scaling one.
- The DFT rows converge in 13 iterations up to nb=208 and then jump to **51** at
  nb=312, which is the same cliff arriving earlier.

**Why this is the cheapest large win available.** The other two axes are known and
expensive: the ERI Fock build is ~200× slower than libcint and four candidate
optimizations were each disproven by measurement (`docs/ERI_PERFORMANCE_SCOPE.md`)
— closing it needs a different engine. MPI scaling is already 42-46 % efficient at
32 ranks. Iteration count is the one large factor nobody has attacked, and it
multiplies with **every** method: HF, DFT, and every post-HF path that sits on a
converged SCF.

## What exists already — this is mostly assembly, not construction

The two hard pieces of an SOSCF are **already in the tree, validated, and
deliberately decoupled**.

**1. The generic augmented-Hessian solver.** `src/post_hf/casscf/aug-hessian.h`
is a Co-Iterative Augmented Hessian (CIAH) solver modeled on PySCF's
`soscf/ciah.py:davidson_cc` — the same algorithm PySCF's own SOSCF uses. Its
header states the decoupling explicitly:

> The callbacks give the caller full control over how the gradient and
> Hessian-vector product are computed **without coupling this solver to the CASSCF
> data structures.**

It takes three `std::function` callbacks (`AugHessianHopFn`, `AugHessianGradFn`,
`AugHessianPrecondFn`) and an options struct carrying level shift, mode-selection
guard (`v0_min`), linear-dependence threshold and micro-iteration limits. It is
exercised by the 11/11 CASSCF PySCF gate suite.

**2. The RHF orbital Hessian.** `build_rhf_cphf_matrix`
(`src/post_hf/rhf_response.h`) is exactly the operator SOSCF needs, and
`build_rhf_cphf_matrix_ri` is the RI-fitted form that avoids the `nao⁴` build.
`solve_rhf_cphf` already drives it. `uhf_response.h` is the unrestricted sibling.

**So the work is: write the callbacks, and decide when to switch.** Not: derive
an orbital Hessian, or write a trust-region eigensolver.

## The one design question that is not mechanical

**When to switch from DIIS to SOSCF, and how to fall back.**

SOSCF converges quadratically *near* a solution and can diverge or find a saddle
far from one. Every production code therefore runs first-order steps until some
switch criterion and only then goes second-order. PySCF's default is a gradient
threshold plus a minimum cycle count.

This is the only part with real judgement in it, and it is where a naive
implementation makes things *worse* — switching too early costs micro-iterations
and can walk into a saddle; too late wastes the DIIS iterations SOSCF was meant
to replace.

**Do not invent a criterion.** Start with the DIIS error norm already computed
every iteration (`metrics.diis_error`, `src/scf/scf.h:21`), which is
`‖FPS − SPF‖` — the same quantity `is_converged` gates on. Switch when it drops
below a threshold *and* a minimum iteration count has passed, matching the shape
PySCF uses, then tune against the measured cases.

## Steps

Ordered so the cheapest step can kill the expensive ones.

### S1 — establish the baseline this is judged against (~S)

Record iterations-to-convergence for the current DIIS path across the ladder:
water chains at nb ∈ {104, 156, 208, 312, 416} for HF, and the DFT rows including
the nb=416 case that does not converge.

*Verify:* the numbers reproduce the `scale.json` column above. If they do not, the
comparison later is meaningless — find out why before writing any SOSCF code.

**Also record where the time goes per iteration.** If the Fock build is 95 % of an
iteration, halving iterations halves wall time; if the diagonalization is
significant at nb=416, SOSCF's micro-iterations eat some of the win. This decides
how to *report* the result honestly.

### S2 — RHF only, behind a flag, no fallback logic (~M)

Wire `build_rhf_cphf_matrix`'s Hessian-vector product and the orbital gradient
into `solve_augmented_hessian` as an alternative to the DIIS extrapolate +
diagonalize block in `run_rhf` (`src/scf/scf.cpp:645-700`). Switch on a **fixed
iteration number** first (`scf_soscf_start 8`, say) — not a criterion. The goal of
S2 is to prove the step is *correct*, not that the switch is smart.

*Verify:* on a case that converges today, SOSCF from iteration N reaches the
**same energy to 1e-10** as the pure-DIIS run. Same minimum, not just a converged
one — the failure mode of a bad second-order step is converging confidently to a
different stationary point.

**If the energies differ, stop.** That is a wrong Hessian or a wrong step, and no
amount of switch tuning fixes it.

### S3 — the switch criterion, measured (~S)

Replace the fixed iteration with the DIIS-error criterion above. Sweep the
threshold on the S1 ladder and record iterations-to-convergence at each.

*Verify:* iteration count **falls at large nb and does not rise at small nb**. The
second half matters more than the first: a change that helps nb=416 and costs
nb=104 five iterations is a regression for every regression case in the suite.

### S4 — UHF/ROHF, and DFT (~M)

`uhf_response.h` gives the unrestricted Hessian. ROHF's effective Fock makes its
orbital Hessian a different object — **scope that separately**, and do not assume
the UHF form transfers (the same trap the ROHF *gradient* hit, where the UHF
energy-weighted density was wrong and `W = PaFaPa + PbFbPb` was needed).

DFT reuses the same machinery with the XC kernel contribution added to the
Hessian-vector product. **The nb=416 non-convergence is the target**, and it is
the strongest single test available: it fails today.

### S5 — make it the default, or do not (~S)

A flag nobody sets is not a win. Decide on S3's data whether SOSCF is on by
default with a fallback to pure DIIS, or opt-in. If default: the regression suite
must be green **and** iteration counts must not regress on the small cases.

## What this must not do

- **Do not write a second augmented-Hessian solver.** `aug-hessian.h` is generic
  by construction and validated by 11 gates. If it needs a change to serve SCF,
  change it there — a second implementation is the thing its callback design
  exists to prevent.
- **Do not touch the DIIS path's behaviour when SOSCF is off.** The default emit
  must stay bitwise identical; every existing regression depends on it.
- **Do not judge success on wall-clock alone.** The claim is *iterations*, and
  wall-clock mixes in the micro-iteration cost. Report both.
- **Do not tune the switch on one system.** The measured cliff is size-dependent
  (HF 30→91, DFT 13→51), so a threshold fitted at nb=104 says nothing about
  nb=416.
- **Beware the saddle.** A second-order method finds *stationary points*, not
  minima. Planck already has `src/scf/stability.{h,cpp}` for exactly this
  question — if SOSCF starts converging to different energies than DIIS, run the
  stability analysis before assuming the Hessian is wrong.

## Key code locations

| what | where |
|---|---|
| the generic CIAH solver (reuse, do not rewrite) | `solve_augmented_hessian`, `src/post_hf/casscf/aug-hessian.h` |
| how CASSCF drives it (the pattern to copy) | `src/post_hf/casscf/aug-hessian-orbital.cpp:264` |
| the RHF orbital Hessian | `build_rhf_cphf_matrix`, `src/post_hf/rhf_response.h` |
| its RI form (no `nao⁴`) | `build_rhf_cphf_matrix_ri`, same header |
| the UHF sibling | `src/post_hf/uhf_response.h` |
| the insertion point in the SCF loop | `run_rhf`, `src/scf/scf.cpp:645-700` (DIIS extrapolate → diagonalize) |
| the switch quantity, already computed | `IterationMetrics::diis_error`, `src/scf/scf.h:21` |
| saddle-vs-minimum diagnosis | `src/scf/stability.{h,cpp}` |
| the motivating measurement | `scale.json`, and `docs/HPC_REMAINING_SCOPE.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
