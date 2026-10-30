# Why is CASSCF so much slower than PySCF, and what makes it fast?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This is an in-flight scope; when the work lands,
rewrite it as the answer to the question in the title.

## Headline (2026-10-30)

**CASSCF is now 12-75x faster than at the start: ahead of PySCF at 6-31G,
2-5x behind at cc-pVDZ, and 20-40x behind at cc-pVTZ.** It takes as many or
fewer macroiterations than PySCF everywhere. What is left is per-macro cost
at large bases, not the optimizer.

Four defects are fixed and the step policy rebuilt:

- an O(nb^5) integral transform (P1);
- a finite-difference orbital Hessian (P2);
- two missing terms in the orbital-CI coupling (P3b);
- an iterative coupled solve that could never converge, now a direct solve of
  the exact CI-relaxed Hessian, with a Lanczos solve above 256 pairs (P3c,
  P4a);
- a level shift and step cap tuned for an approximate Hessian, now a trust
  region driven by the exact model (P4b).

`numeric-newton`, built every macro at 2·npairs full evaluations, is no longer
used.

**The water SA-2 references were re-pinned.** With the exact Hessian, the
hcore and SAD starts leave a measured saddle (-74.7751377977) for the true
minimum (-74.7877864783, PySCF's own SAD-start answer). All 15 CASSCF
regression cases pass.

Water CAS(4,4), Cartesian, 4 threads, `build-full`. PySCF 2.13.0
(`tests/benchmarks/casscf/pyscf_casscf_timing.py`). Times are wall time;
macros in brackets.

| case | AOs / pairs | start | P2 | P3c | **now (P4)** | PySCF | now vs PySCF |
|---|---|---|---|---|---|---|---|
| 6-31G, C2v | 13 / 22 | 1.3-1.9 s | 0.58 s | 0.33 s (17) | **0.18 s (8)** | 0.34 s (10) | **0.5x** |
| cc-pVDZ, C2v | 25 / 50 | 13.4 s | 2.7 s | 1.21 s (17) | **0.58 s (8)** | 0.27 s (9) | 2.1x |
| cc-pVDZ, C1 | 25 / 138 | 44.5 s | 10.0 s | 2.70 s (17) | **2.35 s (17)** | 0.43 s (17) | 5.5x |
| cc-pVTZ, C2v | 65 / 142 | ~176 s per macro | - | 85.6 s (17) | **43.5 s (9)** | 1.16 s (9) | 38x |
| cc-pVTZ, C1 | 65 / 418 | - | - | 72.0 s (24)* | **51.8 s (17)** | 2.30 s (20) | 23x |

\* with the P4a Lanczos solve; before it, 418 pairs fell to the weak
diagonal fallback.

Energies agree with PySCF to <= 3.4e-8 Eh (cc-pVDZ -76.0440109039 vs
-76.0440109018; C1 -76.0781256170 vs -76.0781256214; cc-pVTZ -76.0754286537
vs -76.0754286541; cc-pVTZ C1 -76.1092535258 vs -76.1092535271).

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

### P3c: direct solve of the CI-relaxed Hessian (`7a5031a8`, references re-pinned after)

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
- **Gates:** 15 of 15 regression cases pass after the SA-2 re-pin (below).
  The table inputs are within 3e-9 Eh except the re-pinned water SA-2 row.
- **Macro counts went up on some symmetric cases:** 6-31G 13 -> 17, water
  STO-3G 7 -> 11, ethylene 3-21G 3 -> 5. Each macro is far cheaper, so wall
  time still drops, but the step policy is leaving iterations on the table.

### Resolved: the SA-2 hcore/SAD basin change (re-pinned)

`water_casscf_sa2_sto3g` and `water_casscf_sa2_sto3g_sad_guess` now converge
to -74.7877864783 instead of the pinned -74.7751377977. Measured with a
temporary `hess_probe` (the CI-relaxed Hessian's spectrum per macro):

- **The old point is a saddle.** Near -74.7751 the Hessian has exactly one
  negative eigenvalue (-0.054 to -0.065).
- **The gradient has no component along that mode:** |g·v0| ~ 1e-16. It is
  symmetry-breaking, and these inputs run with symmetry off.
- **Planck still leaves it, by amplifying roundoff.** The eigenvalue floor
  (1e-4) multiplies the 1e-16 component by ~1e4 per macro (1e-16 -> 1e-12 ->
  1e-9 -> 8e-7 -> 3e-4), and the run falls off at macro 9.
- **The new point is a true minimum.** At -74.7877864783 the Hessian is
  positive definite (lowest eigenvalue +2.5e-2). It is also PySCF's own
  SAD-start answer.

Before P3c, the coupling Hessian was wrong and `numeric-newton` happened not
to amplify the component, so Planck stayed on the saddle.

**Suppressing the escape was tried and rejected.** The variant dropped clamped
modes whose gradient component is at noise level (<= 1e-8·|g|). It kept the
SA-2 hcore case on the saddle (-74.7751377973) but broke three others:

| case | result with suppression | expected |
|---|---|---|
| uphill canary | stuck on the saddle | -74.7878 |
| `water_casscf44_spherical_631gd` | -76.0256 | -76.0610 |
| C1 water/cc-pVDZ | -76.0440 | -76.0781, where PySCF itself escapes |

Following negative curvature is correct optimizer behaviour. So the two SA-2
references were re-pinned to the minimum (decided 2026-10-28), with the
evidence in `pyscf_reference/pyscf_reference_energies.md` and the
Completion gate table.

After the re-pin, all 15 CASSCF/RASSCF/FCI regression cases pass.

**Caveat worth keeping.** Which basin a symmetric start reaches is still set
by roundoff growing along a zero-gradient mode, in both codes. Running these
inputs with symmetry on would make the constrained answer deterministic; that
is a separate decision.

### P4a: CI-relaxed Lanczos solve above 256 pairs (2026-10-30)

Above the dense limit the relaxed operator used to fall back to six
diagonal-preconditioned line-search steps on the **orbital-only** action, with
no CI relaxation. `OrbitalLinearOperator` now carries the CI-relaxation action
(so matrix-free products include it too). `solve_orbital_action_system` runs
Lanczos on it from g, with full reorthogonalization, up to 64 vectors. It
applies the dense recipe (eigenvalue floor 1e-4) to the tridiagonal
projection, and stops when `beta_k |y_k| <= 1e-8 |g|`. It tends to the dense
step as the subspace grows; one action per vector.

Validated by forcing it on (dense limit 0, temporary override since removed):

| case | dense | Lanczos |
|---|---|---|
| cc-pVDZ C1 | -76.0781256195, 17 macros, 2.67 s | -76.0781256198, 18, 1.82 s |
| cc-pVDZ C2v | -76.0440109021, 17, 1.16 s | -76.0440109018, 17, 1.21 s |

At cc-pVTZ C1 (418 pairs, the natural path) it reaches PySCF's energy to
1.3e-9.

### P4b: trust region from the exact model (2026-10-30)

**The diagnosis.** After P3c the Hessian was exact and positive definite on
6-31G, yet the energy fell linearly for ten macros. Two policy knobs tuned for
the old approximate Hessian were the brake:

- **The level shift** started at 0.2 and decayed only x0.7 per accepted
  macro, reaching its 1e-3 floor at macro 15. Early on it exceeded the
  Hessian's smallest eigenvalue, so "Newton" steps were damped gradient steps.
- **The step cap** (0.2 per element) rescaled the whole step uniformly.
  Sweeps (initial shift 0.2 / 0.05 / 0.01 / 0.001; cap 0.2 / 0.5 / 10) showed
  the low shift alone helps C2v but hurts C1 (17 -> 20-22 macros). Loosening
  the cap is what C1 needs (shift 0.001, cap 0.5: 6-31G 8 macros, C1 14).

**The model.** The printed `predicted_dE` came from a diagonal
orbital-energy Hessian (`hess_diag`) and was often the wrong sign; nothing
read it. It is now `g.s + ½ s.H s` with the exact CI-relaxed action at the
old point (one extra action per macro). Measured: rho = actual/predicted is
0.89-1.30 on every macro of 6-31G and C1 cc-pVDZ, and -> 1.000 near
convergence. The units are right and the model is trustworthy far from the
solution.

**The policy** (`casscf.cpp`):

- The level shift starts at its floor (1e-3). The existing rules still raise
  it on rejection (x2) and stagnation (x1.5).
- `trust_rot` replaces the fixed `max_rot` everywhere a macro step is formed
  or capped (coupled solve, diagonal and fallback steps, aug-Hessian,
  `numeric-newton`, the cap helper; Frobenius cap 4 x `trust_rot`). The
  single-pair probes keep the fixed `max_rot`. It starts at `mcscf_max_rot`
  and moves with the exact model:
  - doubles (up to 5 x `mcscf_max_rot`) when rho >= 0.75 and the step hit the
    radius;
  - shrinks x4 (floor `mcscf_max_rot`/16) when rho < 0.25 or the model
    predicts an increase;
  - halves when no step is accepted.

**Results.** All 15 CASSCF/RASSCF/FCI regression cases pass, including the
uphill canary. The 11 table inputs are within 2e-9 Eh of their recorded
values. Macro counts fell everywhere:

| input | macros before -> after |
|---|---|
| water STO-3G | 11 -> 4 |
| water 6-31G | 17 -> 8 |
| water cc-pVDZ | 17 -> 8 |
| water SA-2 | 16 -> 10 |
| ethylene SA-2 | 17 -> 9 |
| water cc-pVTZ | 17 -> 9 (PySCF 9) |
| water cc-pVTZ C1 | 24 -> 17 (PySCF 20) |

## Follow-ups, not started

- **Per-macro cost at large bases is now the gap** (cc-pVTZ ~4.8 s per macro
  against PySCF's ~0.13 s). Profile cc-pVTZ before choosing. Likely
  candidates:
  - the dense relaxed-operator build (142 columns of ~4 nb^4 Fock
    contractions each);
  - the nb^4 Fock contractions in general (PySCF builds JK directly from
    shell quartets, not from a stored tensor);
  - `transform_eri`'s first quarter, which has `mu` innermost at stride
    `nb^3` (a GEMM shape would also speed MP2/CC; gate
    `planck-transform-eri`).
- **RI path.** The orbital-orbital block is still FD-Hessian, and the
  transform is still `transform_eri_ri(C, C_act, C_act, C_act)` (full leg
  first). `water_casscf_ri_ccpvdz` takes 4.1 s.
- **Delete the temporary probes** (`CallProbe`, `nn_probe`, `schur_probe`,
  `alpha_probe`, `fix_probe`, `hess_probe`) once this scope is answered.
- **Symmetric starts and saddles.** Which basin a symmetric start reaches is
  set by roundoff growing along a zero-gradient mode, in both codes. Running
  such inputs with symmetry on would make the constrained answer
  deterministic.

## Out of scope

- CI cost: irrelevant at CAS(4,4); the large-CAS sigma build is done
  (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`).
- Direct or cached Fock builds: disproven (`CASSCF_FOCK_REBUILD_COST.md`).
