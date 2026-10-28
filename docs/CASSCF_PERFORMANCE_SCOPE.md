# Why is CASSCF so much slower than PySCF, and what makes it fast?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This is an in-flight scope; when the work lands,
rewrite it as the answer to the question in the title.

## Headline (2026-10-28)

**CASSCF is 4.4-5x faster than at the start, with identical energies, and
PySCF is still 10-23x faster.** Two defects are fixed: an O(nb^5) integral
transform, and a finite-difference orbital Hessian. What remains is the
optimizer itself. `numeric-newton`, meant as a stagnation fallback, runs every
macro and wins most of them, at 2·npairs full CI-relaxed evaluations per macro.
Making the primary coupled step win is the next lever and the larger one.

Water CAS(4,4), Cartesian, 4 threads, `build-full`. PySCF 2.13.0
(`tests/benchmarks/casscf/pyscf_casscf_timing.py`).

| case | AOs / pairs | start | now (P2) | speedup | PySCF | still slower |
|---|---|---|---|---|---|---|
| 6-31G, C2v | 13 / 22 | 1.3-1.9 s | 0.58 s | 2-3x | 0.34 s | 1.7x |
| cc-pVDZ, C2v | 25 / 50 | 13.4 s | 2.7 s | 5.0x | 0.27 s | **10x** |
| cc-pVDZ, C1 | 25 / 138 | 44.5 s | 10.0 s | 4.4x | 0.43 s | **23x** |
| cc-pVTZ, C2v | 65 / - | ~176 s per macro | not rerun | | 1.0 s total | |

Every energy is unchanged to the last printed digit (cc-pVDZ -76.0440109051,
PySCF -76.0440109018; C1 -76.0781256198 vs -76.0781256214), and so is every
macro's winning candidate. The gap still grows with basis size, so a scaling
problem remains, not a constant.

## Where the time went

Three multipliers stacked:

1. **An O(nb^5 n_act) transform (fixed, P1).** `transform_eri_active_cache`
   looped over the full-MO index outside the first contraction. It was 87 %
   of a cc-pVDZ run.
2. **A finite-difference orbital Hessian (fixed, P2).** Every Hessian-vector
   product was a central difference of two full gradient rebuilds: about one
   rebuild pair per rotation pair per macro.
3. **`numeric-newton` every macro (open, P3).** `build_numeric_newton_step`
   runs whenever `npairs <= 256` (`casscf.cpp`), not only under stagnation as
   its comments say. It costs 2·npairs `evaluate`s (CI solve + two Fock
   builds + transform each) and wins 11/13 macros at 6-31G, 6/16 at cc-pVDZ,
   25/27 in C1. The primary `sa-coupled` step mostly loses.

Planck also takes more macros than PySCF (16 vs 9 at cc-pVDZ, 27 vs 17 in C1),
and its predicted dE is often the wrong sign. That is the same optimizer
question as item 3.

## Gates (every step)

- All 15 CASSCF/RASSCF/FCI regression cases (`tests/run_regressions.py
  --suite all --case ...`). The basin canary
  `water_casscf_sa2_sto3g_sad_guess_uphill` matters most: `numeric-newton`'s
  `/2eps` once amplified a 1e-15 reorder into a basin flip
  (`CASSCF_FOCK_REBUILD_COST.md`).
- The 11 PySCF-table inputs in `tests/benchmarks/casscf/pyscf_reference/`,
  against their recorded Planck energies.
- `planck-casscf-internal`.
- Stop if any gated energy moves by more than 1e-8 Eh or the canary flips.

## Done

### P0: measurement (2026-10-28)

`PLANCK_CASSCF_CALL_PROBE=1` prints per-macro counts (`CallProbe` in
`orbital.h`; temporary, delete when this scope is answered). Its Fock count
undercounts after P2, since the analytic action calls `_compute_fock_rhf`
directly. Per macro at the start:

| case | `evaluate` | FD gradient rebuilds | `(pu\|vw)` transforms | Fock | `numeric-newton` wins |
|---|---|---|---|---|---|
| 6-31G | 73 | 74 | 147 | 293 | 11 / 13 |
| cc-pVDZ | 129 | 128 | 257 | 514 | 6 / 16 |
| cc-pVDZ C1 | 306 | 122 | 428 | 857 | 25 / 27 |

`evaluate` = 2·npairs + ~29, so `numeric-newton` runs every macro. Starting
profile (cc-pVDZ, 1 thread): `transform_eri_active_cache` 87.3 %,
`_compute_fock_rhf` 8.3 %, everything else (CI, RDMs, Q) < 1 %. At cc-pVTZ
the first two macros took ~176 s each, ~1600x PySCF per macro.

### P1: (pu|vw) transform in O(nb^4 n_act) (2026-10-28, `9e240155`)

`(pu|vw) = (vw|up)` under the AO tensor's 8-fold symmetry, so the routine is
now `transform_eri(eri, nb, C_act, C_act, C_act, C)` plus a permutation. The
hand-rolled routine and its scratch are deleted (net -116 lines).
cc-pVDZ 13.4 -> 3.5 s, C1 44.5 -> 11.1 s.

Gate: against `transform_eri(C, C_act, C_act, C_act)` on an 8-fold-symmetric,
non-square, dense fixture. The old fixture had no ERI symmetry at all, which
no real tensor lacks. A u<->w permutation mutant fails it.

### P2: analytic orbital Hessian action (2026-10-28, `a330be6f`)

With `C(eps) = C(I + eps R)` (the Cayley step to first order):

- `dF_I = R^T F_I + F_I R + C^T G[dD_c] C`, `dD_c = 2C(R P_c + P_c R^T)C^T`;
  `dF_A` likewise with `P_gamma`.
- `dQ(p,t) = (R^T Q)(p,t) + sum_{a,x} R(a, n_core+x) Y[p,a,t,x]`, with `Y`
  built once per macro from `(pa|vw)` and `(pu|aw)`.
- `compute_orbital_gradient` is linear in `(F_I, F_A, Q)` at fixed gamma, so
  the action is that function applied to the derivatives.

Two Fock contractions per product instead of four plus two transforms. The
FD path stays as the oracle (null `analytic`) and is still what RI uses.
cc-pVDZ 3.5 -> 2.7 s, C1 11.1 -> 10.0 s.

Gate: analytic vs Richardson-extrapolated central FD to 9e-11 relative. Plain
central FD at 1e-4 resolves only 1.3e-6, too coarse to tell truncation from a
bug. Zeroing the w-leg term fails it at 5.3.

**Finding:** exact curvature changed no candidate choice, so the FD Hessian
was not why `sa-coupled` loses.

## In progress: P3, why does the coupled step lose to `numeric-newton`?

`numeric-newton` is the FD of the fully CI-relaxed gradient, i.e. the exact
orbital Hessian *including* CI relaxation. `sa-coupled` solves the coupled
orbital-CI Newton equations in `response.cpp`, so both should target the same
step.

### Found so far (2026-10-28, uncommitted probes under `PLANCK_CASSCF_CALL_PROBE`)

**1. The coupled solve never converges.** `coupled_converged=0` on all 56
macros of 6-31G, cc-pVDZ and C1. The orbital residual stalls at 1e-3..1e-2,
about the size of the gradient. `numeric-newton` is offered only when the
coupled step is unreliable or after two stagnant macros (`casscf.cpp`), so it
is offered every macro because the primary step is never trusted.

**2. Its build no longer runs when the gate is shut** (behaviour-neutral:
energies and every winning candidate identical). It saves nothing on these
cases, since the gate is always open, but it stops 2·npairs evaluations
being discarded once the coupled step is fixed.

**3. The trust cap is not the cause.** With `mcscf_max_rot 10` the solve
still never converges. (Uncapped C1 takes 21 macros instead of 27, same
energy; `numeric-newton` still wins.)

**4. The coupled equations' CI-coupling block is wrong in shape.** Along
R = the current gradient direction, compare three actions:

- `rel`: central FD of the fully CI-re-solved gradient (two `evaluate`s),
  the CI-relaxed Hessian.
- `oo`: the analytic fixed-CI action, exact by P2's gate.
- `cpl = oo + sum_I w_I G_oc c1_I(R)`: the coupled system's effective
  Hessian, using `build_coupled_response_blocks`.

`dr = rel - oo` is then purely the true CI-relaxation term,
`-H_oc H_cc^-1 H_co R`, and `dc = cpl - oo` is what the code supplies for it.
`dr` is 1-33 % of `|rel|`, far above FD noise. But `cos(dc, dr)` is at most
0.75 and often negative, and the best-fit factor `alpha` in
`rel ~ oo + alpha·dc` wanders from -2.1 to 5.5. That is not a prefactor
error. The coupling term points the wrong way.

So the defect is in one of the three pieces of the CI relaxation:

- `build_ci_response_rhs`: the orbital-to-CI coupling H_co R.
- the CI response solve: H_cc, including the `1e-4` preconditioner floor,
  the projection against c0, and E0.
- `orbital_correction_from_ci_step`: the CI-to-orbital coupling H_oc c1.

### Next

Gate each piece separately against a finite difference, in the same style as
P2's gate:

- **H_co R:** FD of the CI gradient `(H - E)c0` along an orbital rotation,
  CI fixed.
- **H_oc c1:** FD of the orbital gradient along `c0 + eps·c1` (renormalized),
  orbitals fixed.
- **The solve:** check `(H - E0)c1 = rhs` directly.

Fix the piece that fails, then confirm `cos(dc, dr) -> 1` and that the coupled
solve converges. Only then rerun the accepted-candidate sweep and, if
`sa-coupled` wins, return `numeric-newton` to stagnation-only. It is
load-bearing today (the uphill canary's path runs through it), so that is a
measured decision, not a cleanup.

Not yet ruled out but now secondary: block-Jacobi stalling in the coupled
iteration (it stops at the first scale that fails to reduce the residual).
Fixing the coupling block may or may not fix convergence on its own.

## Follow-ups, not started

- **RI path.** Still FD-Hessian and still `transform_eri_ri(C, C_act, C_act,
  C_act)` (full leg first). `water_casscf_ri_ccpvdz` takes 14 s against 2.8 s
  for the dense cc-pVDZ case. Measure before assuming the same defect.
- **`transform_eri`'s first quarter** has `mu` innermost at stride `nb^3`.
  A GEMM shape would also speed MP2/CC (gate `planck-transform-eri`).
- **cc-pVTZ** has not been rerun since P1.

## Out of scope

- CI cost: irrelevant at CAS(4,4); the large-CAS sigma build is done
  (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`).
- Direct or cached Fock builds: disproven (`CASSCF_FOCK_REBUILD_COST.md`).
