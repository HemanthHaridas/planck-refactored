# Why is CASSCF so much slower than PySCF, and what makes it fast?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This is an in-flight scope; when the work lands,
rewrite it as the answer to the question in the title.

## Headline (2026-10-28)

**CASSCF is now 4-16x faster than at the start, level with PySCF at 6-31G and
4.5-6x behind it at cc-pVDZ.** Three defects are fixed and a fourth is in the
working tree:

- an O(nb^5) integral transform (P1);
- a finite-difference orbital Hessian (P2);
- two missing terms in the orbital-CI coupling (P3b);
- an iterative coupled solve that could never converge, now a direct solve of
  the exact CI-relaxed Hessian (P3c, uncommitted).

With P3c, `numeric-newton`, built every macro at 2·npairs full evaluations,
is no longer used at all.

**P3c is blocked by a basin change.** The water SA-2 hcore-start cases now
converge to the deeper -74.7877864783 stationary point instead of PySCF's
-74.7751377977. Two regression cases fail. See P3c below.

Water CAS(4,4), Cartesian, 4 threads, `build-full`. PySCF 2.13.0
(`tests/benchmarks/casscf/pyscf_casscf_timing.py`).

| case | AOs / pairs | start | P2 | P3c (uncommitted) | PySCF | P3c vs PySCF |
|---|---|---|---|---|---|---|
| 6-31G, C2v | 13 / 22 | 1.3-1.9 s | 0.58 s | **0.33 s** | 0.34 s | 1.0x |
| cc-pVDZ, C2v | 25 / 50 | 13.4 s | 2.7 s | **1.21 s** | 0.27 s | 4.5x |
| cc-pVDZ, C1 | 25 / 138 | 44.5 s | 10.0 s | **2.70 s** | 0.43 s | 6.3x |
| cc-pVTZ, C2v | 65 / - | ~176 s per macro | not rerun | not rerun | 1.0 s total | |

Energies are within 3e-9 Eh of the start everywhere except the SA-2 basin
change (cc-pVDZ -76.0440109021, PySCF -76.0440109018; C1 -76.0781256197 vs
-76.0781256214). C1 now takes 17 macros, the same as PySCF.

## Where the time went

1. **An O(nb^5 n_act) transform (fixed, P1).** `transform_eri_active_cache`
   looped over the full-MO index outside the first contraction. It was 87 %
   of a cc-pVDZ run.
2. **A finite-difference orbital Hessian (fixed, P2).** Every Hessian-vector
   product was a central difference of two full gradient rebuilds.
3. **`numeric-newton` every macro (fixed by P3b + P3c).** It was offered
   whenever the coupled step was "unreliable", and the coupled step was
   unreliable on every macro. Its 2·npairs `evaluate`s per macro (CI solve,
   two Fock builds and a transform each) were most of what P1 and P2 left.

`evaluate` calls per macro: 6-31G 73 -> 22, cc-pVDZ 129 -> 22, C1 306 -> 41.

## Gates (every step)

- All 15 CASSCF/RASSCF/FCI regression cases (`tests/run_regressions.py
  --suite all --case ...`), with particular attention to the four SA-2 cases.
  The uphill canary `water_casscf_sa2_sto3g_sad_guess_uphill` and the two
  hcore/SAD SA-2 cases pin which basin each start reaches.
- The 11 PySCF-table inputs in `tests/benchmarks/casscf/pyscf_reference/`,
  against their recorded Planck energies.
- `planck-casscf-internal`.
- Stop if any gated energy moves by more than 1e-8 Eh or a basin changes.

## Done

### P0: measurement (`87d5a10f`)

`PLANCK_CASSCF_CALL_PROBE=1` prints per-macro call counts, plus the P3
diagnostics below (temporary, delete when this scope is answered). The Fock
count undercounts after P2, since the analytic action calls
`_compute_fock_rhf` directly. The starting profile (cc-pVDZ, 1 thread):

| function | share |
|---|---|
| `transform_eri_active_cache` | 87.3 % |
| `_compute_fock_rhf` | 8.3 % |
| CI, RDMs, Q | < 1 % |

At cc-pVTZ the first two macros took ~176 s each, ~1600x PySCF per macro.

### P1: (pu|vw) transform in O(nb^4 n_act) (`9e240155`)

`(pu|vw) = (vw|up)` under the AO tensor's 8-fold symmetry, so the routine is
now `transform_eri(eri, nb, C_act, C_act, C_act, C)` plus a permutation (net
-116 lines). cc-pVDZ 13.4 -> 3.5 s.

Gate: against `transform_eri(C, C_act, C_act, C_act)` on an 8-fold-symmetric,
non-square, dense fixture. The old fixture had no ERI symmetry at all. A
u<->w mutant fails it.

### P2: analytic orbital Hessian action (`a330be6f`)

With `C(eps) = C(I + eps R)`:

- `dF_I = R^T F_I + F_I R + C^T G[dD_c] C`; `dF_A` likewise.
- `dQ = R^T Q + sum R(a, n_core+x) Y[p,a,t,x]`, with `Y` built per macro from
  `(pa|vw)` and `(pu|aw)`.
- The action is `compute_orbital_gradient` applied to the derivatives.

FD stays as the oracle (null `analytic`) and RI still uses it. cc-pVDZ 3.5 ->
2.7 s. Gate: vs Richardson-extrapolated FD to 9e-11 relative. Zeroing the
w-leg fails it at 5.3.

Exact curvature changed no candidate choice, so the FD Hessian was not why
`sa-coupled` lost.

### P3: why the coupled step lost (`9e4d0672`)

- **The coupled solve never converged:** 0 of 56 macros across 6-31G, cc-pVDZ
  and C1. That is why `numeric-newton`, gated on an unreliable coupled step,
  was offered every macro. Raising `mcscf_max_rot` to 10 changed nothing.
- **The coupled equations were wrong in shape.** Along R = the gradient
  direction, the code's CI-relaxation term `dc` and the true one `dr`
  disagree (`dr` = FD of the fully CI-re-solved gradient minus the exact
  fixed-CI action):
  - cos(dc, dr) <= 0.75, often negative;
  - the best-fit prefactor wanders from -2.1 to 5.5.

  No constant factor fixes it.

### P3b: exact orbital-CI coupling (`69728821`)

Two terms were missing, one on each side:

- **A. CI RHS (H_co).** `exact_active_one_body_derivative` differentiated
  `h_eff = F_I(act)` as a commutator only. It dropped
  `(C^T G[dD_c] C)_act`, so core-virtual rotations had no CI coupling at all.
- **B. Back-coupling (H_oc).** Only the 2-RDM response (Q) was fed back, not
  the 1-RDM response through `F_A` and `gamma·F_I`.

Attribution, rebuilding the coupling four ways against `dr` over 16 macros:

| variant | cos(d, dr) | \|dr - d\|/\|rel\| |
|---|---|---|
| as was | -0.50 .. 0.75 | 1e-2 .. 3e-1 |
| A only | 0.07 .. 0.75 | 3e-2 .. 3e-1 |
| B only | -0.80 .. 0.99999 | 9e-3 .. 2e-1 |
| **A + B** | **1.000000** | **7e-10 .. 1e-8** |

Both terms now apply when the Hessian context (C, ERIs or RI) is passed;
without it the old behaviour is kept bitwise. `ci_orbital_coupling` is the
one builder behind `orbital_correction_from_ci_step` and
`build_coupled_response_blocks`. `fock_response_mo` and
`core_density_response` are shared with P2.

Gate (`planck-casscf-internal`):

| piece | check | error | mutant |
|---|---|---|---|
| A | vs Richardson FD of `H(C) c0` | 1.9e-10 | dropping the core term: 0.24 |
| B | vs FD of the gradient along `c0 +- eps c1` | 1.4e-11 | dropping gamma1: 0.42 |

B is exact because the RDMs are quadratic in c.

At this commit the uphill canary limit-cycles: the still-iterative coupled
solve stopped at sa_g 7.9e-7, where the uphill pass fires. P3c addresses it.

## In progress: P3c, direct solve of the CI-relaxed Hessian (uncommitted)

**What changed (`response.cpp`, `casscf.cpp`):**

1. `build_orbital_linear_operator` takes an optional CI-relaxation action.
   The SA coupled solve passes `sum_I w_I H_oc c1_I(e_k)` (via
   `build_coupled_response_blocks`), so the dense operator becomes the
   CI-relaxed Hessian: the one `numeric-newton` estimated by FD. Each column
   costs ~4 Fock builds plus a small CI solve, and no transforms. The dense
   limit is 256 pairs, `numeric-newton`'s own limit; 128 stays for the
   orbital-only operator.
2. With that operator the coupled step is solved **directly**, and the
   block-Jacobi refinement is skipped. "Converged" means the direct solve
   succeeded. The old residual measured the unshifted, uncapped equations,
   so it sat at about `level_shift·|kappa|` (~3e-4) by construction and
   could never reach 1e-6: a second reason the solve never "converged".
3. **Escape-return guard** (`casscf.cpp`). An accepted uphill escape records
   its starting energy. If the run comes back to it within
   `tol_mcscf_energy`, the uphill pass is not retried from there.
   - Why it is needed: at a true SA minimum the per-root gradients are
     nonzero by construction, so the uphill pass can always trim them.
   - What happened without it: the canary cycled up and straight back down.
     Pre-P3b it converged only because `numeric-newton` happened to land
     where no uphill step cleared a 0.05 % threshold (sa_g 3.3e-9,
     max_root_g 7.686e-2). At P3c's landing point (sa_g 5.9e-8,
     max_root_g 7.662e-2) one does.
   - What is kept: the first escape from any point, so a genuine escape to a
     deeper basin still happens.

**Results.**

- **Speed:** headline table. `sa-coupled` wins nearly every macro;
  `numeric-newton` is never offered.
- **Uphill canary:** passes, -74.7877864784 via the plateau exit, in 22 macros
  (was 46), with one uphill escape that came back and was not repeated.
- **Gates:** 13 of 15 regression cases pass. The 11 table inputs are within
  3e-9 Eh except one (below).
- **Macro counts went up on some symmetric cases:** 6-31G 13 -> 17, water
  STO-3G 7 -> 11, ethylene 3-21G 3 -> 5. Each macro is far cheaper, so wall
  time still drops, but the step policy is leaving iterations on the table.

### Blocker: the SA-2 hcore/SAD cases change basin

`water_casscf_sa2_sto3g` and `water_casscf_sa2_sto3g_sad_guess` (and the
table input `water_cas44_sto3g_sa2`) now converge to -74.7877864783 instead of
-74.7751377977, PySCF's answer from the same start.

- Both are valid SA-2 stationary points; the new one is deeper.
- These cases do not enable uphill steps, so the escape guard is not
  involved. They passed at P3b, so the direct relaxed step itself takes a
  different path.
- The ethylene SA-2 case is unaffected (7e-10).

Not yet diagnosed. Candidates, cheapest first:

1. **Step size.** The direct step is the full CI-relaxed Newton step, capped
   only by `max_rot`. Early in the run, far from either minimum, it may jump
   further than the old block-Jacobi step and cross into the deeper basin.
   Compare the first few macros' steps and energies against P3b.
2. **Indefinite Hessian.** Away from a minimum the relaxed Hessian can have
   negative eigenvalues. The eigenvalue floor (1e-4) turns those directions
   into large steps along negative curvature, which is exactly what crosses a
   ridge. PySCF's `mc1step` uses an augmented-Hessian (trust-region) step
   there instead.
3. **Root ordering.** With two roots, an exact step may swap the tracked
   roots' character.

Decide by matching PySCF macro by macro on `water_casscf_sa2_sto3g`. The
regression tests pin PySCF's basin from these starts, so landing in a
different (even if deeper) one is a regression until shown otherwise.

## Follow-ups, not started

- **RI path.** Still FD-Hessian for the orbital-orbital block, and still
  `transform_eri_ri(C, C_act, C_act, C_act)` (full leg first).
  `water_casscf_ri_ccpvdz` takes 8.4 s.
- **Above 256 pairs**, the relaxed operator falls back to the old weak
  matrix-free path. It needs a Krylov solve on the same action.
- **`transform_eri`'s first quarter** has `mu` innermost at stride `nb^3`.
  A GEMM shape would also speed MP2/CC (gate `planck-transform-eri`).
- **cc-pVTZ** has not been rerun since P1.
- **Delete the P0/P3 probes** (`CallProbe`, `nn_probe`, `schur_probe`,
  `alpha_probe`, `fix_probe`) once this scope is answered.

## Out of scope

- CI cost: irrelevant at CAS(4,4); the large-CAS sigma build is done
  (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`).
- Direct or cached Fock builds: disproven (`CASSCF_FOCK_REBUILD_COST.md`).
