# Why is CASSCF so much slower than PySCF, and what makes it fast?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This is an in-flight scope; when the work lands,
rewrite it as the answer to the question in the title.

## Headline (2026-10-30)

**CASSCF is 7-44x faster than when this started (6-31G 7-10x, cc-pVDZ 23x,
cc-pVDZ C1 44x; cc-pVTZ was ~176 s per macro and never run to convergence),
and beats PySCF at 6-31G.
It is 2x behind at cc-pVDZ, 2-4x behind without symmetry, and 17-25x behind at
cc-pVTZ.** It needs no more macroiterations than PySCF anywhere. What is left
is the cost of each macro at large bases, which is dominated by full passes
over the stored ERI tensor.

Water CAS(4,4), Cartesian, 4 threads, `build-full`; PySCF 2.13.0
(`tests/benchmarks/casscf/pyscf_casscf_timing.py`). Wall time, macros in
brackets.

| case | AOs / pairs | start | committed (P4, `1468a4f7`) | working tree (P5) | PySCF | best vs PySCF |
|---|---|---|---|---|---|---|
| 6-31G | 13 / 22 | 1.3-1.9 s | 0.18 s (8) | 0.19 s (8) | 0.34 s (10) | **0.5x** |
| cc-pVDZ | 25 / 50 | 13.4 s | 0.58 s (8) | 0.59 s (8) | 0.27 s (9) | 2.1x |
| cc-pVDZ, C1 | 25 / 138 | 44.5 s | 2.35 s (17) | **1.01 s (11)** | 0.43 s (17) | 2.3x |
| cc-pVTZ | 65 / 142 | ~176 s per macro | 43.5 s (9) | **28.7 s (9)** | 1.16 s (9) | 25x |
| cc-pVTZ, C1 | 65 / 418 | - | 51.8 s (17) | **40.2 s (13)** | 2.30 s (20) | 17x |

Energies agree with PySCF to <= 4.4e-9 Eh: cc-pVDZ -76.0440109039 vs
-76.0440109018, C1 -76.0781256191 vs -76.0781256214, cc-pVTZ -76.0754286515
vs -76.0754286541, cc-pVTZ C1 -76.1092535260 vs -76.1092535271. The
working-tree column has not been through the regression gates yet.

**The water SA-2 references were re-pinned** (2026-10-28). With the exact
Hessian, the hcore and SAD starts leave a measured saddle (-74.7751377977) for
the true minimum (-74.7877864783). The minimum is PySCF's own SAD-start
answer. At P4, all 15 CASSCF/RASSCF/FCI regression cases pass.

## What made it slow

| # | cause | fix | commit |
|---|---|---|---|
| 1 | `transform_eri_active_cache` was O(nb^5 n_act): 87 % of a cc-pVDZ run | active legs first, O(nb^4 n_act) | P1 `9e240155` |
| 2 | every orbital Hessian product was an FD of two gradient rebuilds | analytic action | P2 `a330be6f` |
| 3 | the orbital-CI coupling dropped two terms, so the coupled solve never converged and `numeric-newton` (2·npairs full evaluations) ran every macro | exact coupling, then a direct solve of the CI-relaxed Hessian | P3b `69728821`, P3c `7a5031a8` |
| 4 | above 256 pairs the solve fell back to a few orbital-only diagonal steps | CI-relaxed subspace solve | P4a `1468a4f7` |
| 5 | a 0.2 decaying level shift and a fixed 0.2 step cap, tuned for an approximate Hessian, halved the convergence rate | trust region driven by the exact model | P4b `1468a4f7` |
| 6 | the dense CI-relaxed operator costs ~570 nb^4 Fock passes per macro at cc-pVTZ | matrix-free preconditioned solve | P5, in progress |

## What is left

1. **Stop the P5 subspace solve early** (it still uses the full space or the
   64-vector cap). Then drop the dense build, remove the temporary switches,
   and run the gates.
2. **Fewer passes over the ERI tensor.** Each `_compute_fock_rhf` is a
   bandwidth-bound sweep over 143 MB at cc-pVTZ; batch several densities into
   one sweep.
3. **`transform_eri`'s first quarter** has `mu` innermost at stride `nb^3`.
   A GEMM shape would also speed MP2/CC (gate `planck-transform-eri`).
4. **RI path:** still an FD orbital Hessian, and a full-leg-first transform
   (`water_casscf_ri_ccpvdz` 4.1 s).
5. **Delete the temporary probes** (`CallProbe`, `nn_probe`, `schur_probe`,
   `alpha_probe`, `fix_probe`, `hess_probe`, `lanczos_probe`,
   `PLANCK_CASSCF_DENSE_PAIRS`) once this scope is answered.

## Gates (every step)

- All 15 CASSCF/RASSCF/FCI regression cases (`tests/run_regressions.py
  --suite all --case ...`). The uphill canary
  `water_casscf_sa2_sto3g_sad_guess_uphill` and the two hcore/SAD SA-2 cases
  pin which basin each start reaches.
- The 11 PySCF-table inputs in `tests/benchmarks/casscf/pyscf_reference/`,
  against their recorded Planck energies.
- `planck-casscf-internal`.
- Stop if any gated energy moves by more than 1e-8 Eh or a basin changes.

## Record

### P0: measurement (`87d5a10f`)

`PLANCK_CASSCF_CALL_PROBE=1` prints per-macro call counts and the P3-P5
diagnostics. All of them are temporary. Starting profile (cc-pVDZ, 1 thread):
`transform_eri_active_cache` 87.3 %, `_compute_fock_rhf` 8.3 %, CI/RDMs/Q
< 1 %. At cc-pVTZ the first two macros took ~176 s each, ~1600x PySCF per
macro.

### P1: (pu|vw) transform in O(nb^4 n_act) (`9e240155`)

`(pu|vw) = (vw|up)` under the AO tensor's 8-fold symmetry, so the routine is
`transform_eri(eri, nb, C_act, C_act, C_act, C)` plus a permutation (net -116
lines). cc-pVDZ 13.4 -> 3.5 s.

Gate: against `transform_eri(C, C_act, C_act, C_act)` on an 8-fold-symmetric,
non-square, dense fixture. The old fixture had no ERI symmetry at all. A
u<->w mutant fails it.

### P2: analytic orbital Hessian action (`a330be6f`)

With `C(eps) = C(I + eps R)`:

- `dF_I = R^T F_I + F_I R + C^T G[dD_c] C`; `dF_A` likewise.
- `dQ = R^T Q + sum R(a, n_core+x) Y[p,a,t,x]`, with `Y` built per macro from
  `(pa|vw)` and `(pu|aw)`.
- The action is `compute_orbital_gradient` applied to the derivatives.

FD stays as the oracle; RI still uses it. cc-pVDZ 3.5 -> 2.7 s. Gate: vs
Richardson FD to 9e-11 relative; zeroing the w-leg fails it at 5.3. Exact
curvature changed no candidate choice, so the FD Hessian was not why
`sa-coupled` lost.

### P3: why the coupled step lost (`9e4d0672`)

- **The coupled solve never converged:** 0 of 56 macros. `numeric-newton` is
  offered whenever the coupled step is unreliable, so it ran every macro.
  Raising `mcscf_max_rot` to 10 changed nothing.
- **The coupled equations were wrong in shape.** Along R = the gradient,
  compare the code's CI-relaxation term `dc` with the true one `dr` (FD of
  the fully CI-re-solved gradient minus the exact fixed-CI action):
  cos(dc, dr) <= 0.75, often negative, and the best-fit prefactor ranges
  from -2.1 to 5.5. No constant factor fixes it.

### P3b: exact orbital-CI coupling (`69728821`)

Two terms were missing:

- **A. CI RHS (H_co).** `h_eff = F_I(act)` was differentiated as a commutator
  only, dropping `(C^T G[dD_c] C)_act`. Core-virtual rotations had no CI
  coupling at all.
- **B. Back-coupling (H_oc).** Only the 2-RDM response (Q) was fed back, not
  the 1-RDM response through `F_A` and `gamma·F_I`.

| variant | cos(d, dr) | \|dr - d\|/\|rel\| |
|---|---|---|
| as was | -0.50 .. 0.75 | 1e-2 .. 3e-1 |
| A only | 0.07 .. 0.75 | 3e-2 .. 3e-1 |
| B only | -0.80 .. 0.99999 | 9e-3 .. 2e-1 |
| **A + B** | **1.000000** | **7e-10 .. 1e-8** |

Both apply when the Hessian context is passed; without it the old behaviour
is kept bitwise. `ci_orbital_coupling` is the one builder behind both
callers.

Gate (`planck-casscf-internal`):

| piece | check | error | mutant |
|---|---|---|---|
| A | vs Richardson FD of `H(C) c0` | 1.9e-10 | dropping the core term: 0.24 |
| B | vs FD along `c0 +- eps c1` | 1.4e-11 | dropping gamma1: 0.42 |

### P3c: direct solve of the CI-relaxed Hessian (`7a5031a8`)

- `build_orbital_linear_operator` takes an optional CI-relaxation action, so
  the operator is the CI-relaxed Hessian: the one `numeric-newton` estimated
  by FD. It is solved directly, and "converged" means the solve succeeded.
  The old residual measured the unshifted, uncapped equations, so it sat at
  about `level_shift·|kappa|` (~3e-4) and could never reach 1e-6.
- **Escape-return guard.** An uphill escape that returns to its starting
  energy is not retried from there. At a true SA minimum the per-root
  gradients are nonzero by construction, so the uphill pass could always trim
  them and the canary cycled up and back down.
- `numeric-newton` is no longer offered; `evaluate` calls per macro 6-31G
  73 -> 22, cc-pVDZ 129 -> 22, C1 306 -> 41.

### The SA-2 re-pin (`ac43c647`)

Measured with `hess_probe` (the relaxed Hessian's spectrum per macro):

- **-74.7751 is a saddle.** The Hessian there has one negative eigenvalue
  (-0.054 to -0.065), along a symmetry-breaking mode where |g·v0| ~ 1e-16.
- **The run leaves it by amplifying roundoff.** The 1e-4 eigenvalue floor
  multiplies that component ~1e4 per macro (1e-16 -> 3e-4 by macro 9).
- **-74.7877864783 is a minimum.** Its lowest eigenvalue is +2.5e-2, and it
  is PySCF's SAD-start answer.

**Suppressing the escape** (dropping clamped modes with |g·v| <= 1e-8|g|)
was tried and rejected. It kept the hcore case on the saddle but broke three
others:

| case | with suppression | expected |
|---|---|---|
| uphill canary | stuck on the saddle | -74.7878 |
| spherical 6-31G* | -76.0256 | -76.0610 |
| C1 cc-pVDZ | -76.0440 | -76.0781, where PySCF itself escapes |

So the references moved. Which basin a symmetric start reaches is still set
by roundoff along a zero-gradient mode, in both codes; running such inputs
with symmetry on would make it deterministic.

### P4a: CI-relaxed matrix-free solve above 256 pairs (`1468a4f7`)

`OrbitalLinearOperator` carries the CI-relaxation action, so matrix-free
products include it. Above the dense limit, the old fallback (six diagonal
steps on the orbital-only action) is replaced by a subspace solve with the
dense eigenvalue-floor recipe. Committed as Lanczos; replaced in P5.

| case | dense | Lanczos (forced) |
|---|---|---|
| cc-pVDZ C1 | -76.0781256195, 17 macros, 2.67 s | -76.0781256198, 18, 1.82 s |
| cc-pVDZ C2v | -76.0440109021, 17, 1.16 s | -76.0440109018, 17, 1.21 s |

cc-pVTZ C1 (418 pairs) reaches PySCF to 1.3e-9.

### P4b: trust region from the exact model (`1468a4f7`)

**Diagnosis.** After P3c the Hessian was exact and positive definite on
6-31G, yet the energy fell linearly for ten macros:

- **The level shift** started at 0.2 and decayed x0.7 per accepted macro,
  above the Hessian's smallest eigenvalue for most of the run.
- **The 0.2 per-element cap** rescaled the whole step. Sweeps showed a low
  shift alone helps C2v but hurts C1 (17 -> 20-22 macros); loosening the cap
  is what C1 needs (shift 0.001 with cap 0.5: 6-31G 8 macros, C1 14).

**Model.** `predicted_dE` used a diagonal Hessian and was often the wrong
sign (nothing read it). It is now `g·s + ½ s·H s` with the exact relaxed
action. rho = actual/predicted is 0.89-1.30 on every macro measured, and
-> 1.000 near convergence.

**Policy.** The level shift starts at its 1e-3 floor and rises only on
rejection (x2) or stagnation (x1.5). `trust_rot` replaces `max_rot` wherever
a macro step is formed or capped (probes keep `max_rot`), and moves with the
model:

- doubles (to 5 x `mcscf_max_rot`) when rho >= 0.75 and the step hit the
  radius;
- shrinks x4 when rho < 0.25 or the model predicts an increase;
- halves when no step is accepted.

**Results.** 15/15 regressions pass; table energies within 2e-9 Eh. Macros:

| input | before | after |
|---|---|---|
| water STO-3G | 11 | 4 |
| water 6-31G | 17 | 8 |
| water cc-pVDZ | 17 | 8 |
| water SA-2 | 16 | 10 |
| ethylene SA-2 | 17 | 9 |
| water cc-pVTZ | 17 | 9 (PySCF 9) |
| water cc-pVTZ C1 | 24 | 17 (PySCF 20) |

### P5: matrix-free per-macro cost (in progress, uncommitted)

**Profile** (cc-pVTZ C2v after P4, whole run, 4 threads):

| function | share |
|---|---|
| `_compute_fock_rhf` | **76.3 %** |
| `transform_eri` | 15.6 % |
| thread wait | 7.3 % |
| everything else | < 1 % |

The dense relaxed operator is built one column per pair (142 at cc-pVTZ C2v),
at ~4 nb^4 Fock contractions per column. That is ~570 full passes over a 143
MB tensor per macro, against PySCF's ~17 JK builds, because PySCF's
augmented-Hessian solve uses ~10 products instead of the whole matrix.

**Tried:**

1. *Plain Lanczos at every size.* Never stopped early: its 1e-8 |g| test
   needed the full space (22, 50 vectors) or hit the 64 cap. At cc-pVTZ the
   cap made steps inexact: 25 macros, 74 s. Not taken.
2. *Preconditioned subspace solve* (working tree). The subspace grows by
   `(H x + g)` divided by the diagonal orbital Hessian, with the dense
   floor recipe in the subspace, and stops at `||H x + g|| <= 1e-5 |g|`. It
   still uses the full space or the cap, but is already faster and keeps the
   energies: see the headline's working-tree column.

**Next:** an inexact-Newton forcing term (e.g. `||r|| <= min(0.1, sqrt|g|)·|g|`)
to stop at ~10 products. Check whether the floor or the CI-response
tolerance keeps the residual from falling. Then drop the dense build and gate.

## Out of scope

- CI cost: irrelevant at CAS(4,4); the large-CAS sigma build is done
  (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`).
- Direct or cached Fock builds: disproven (`CASSCF_FOCK_REBUILD_COST.md`).
