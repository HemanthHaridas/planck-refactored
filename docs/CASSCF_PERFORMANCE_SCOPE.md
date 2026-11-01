# Why is CASSCF so much slower than PySCF, and what makes it fast?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This is an in-flight scope; when the work lands,
rewrite it as the answer to the question in the title.

## Latest: N5 (2026-11-01, uncommitted)

**cc-pVTZ 19.6 -> 13.3 s and cc-pVTZ C1 23.6 -> 15.0 s, from a loop
interchange in `transform_eri`; and C1 CASSCF is now bitwise reproducible.**

- **`transform_eri` (`src/post_hf/integrals.cpp`).** Quarters 1-3 had the
  contracted index innermost, so the first quarter read the AO tensor at
  stride nb^3; it was 37 % of a cc-pVTZ run. The contracted index now sits
  outside a unit-stride sigma loop. Each element still sums over the
  contracted index in ascending order, so the output is bit-identical: every
  table energy matches N4 to all printed digits, `planck-transform-eri`
  passes at 1 and 4 threads, and the core suite passes 85/85. MP2, CC,
  stability and hybrid-DFT exchange use the same routine.
- **RDM reductions (`src/post_hf/ci/rdm.cpp`).** The 1- and 2-RDM builds
  scheduled tasks dynamically and merged per-thread partials under
  `omp critical`, so they summed in completion order. Water/cc-pVDZ C1 at 4
  threads gave -76.0781256180 .. -76.0781256195 across identical runs
  (P5 and N4 too). They now use the FCI sigma build's discipline: 64 bins
  fixed by task index, `schedule(static, bin_size)`, bins summed in fixed
  order. The energy is -76.0781256190 in five 4-thread runs and at 1/2/8
  threads, and the whole trace is identical at 1/4/8 threads. New gate:
  `water_casscf44_c1_ccpvdz_threads1/4` (`atol 0`, threads1 pinned to PySCF
  at 1e-8); the pair fails on the old `rdm.cpp`.

## Where it stands

**CASSCF is 8-76x faster than when this started and beats PySCF at 6-31G.
It is 1.4-1.6x behind at cc-pVDZ and 6.5-11x behind at cc-pVTZ.** It needs no
more macroiterations than PySCF anywhere.

Water CAS(4,4), Cartesian, 4 threads, `build-full`; PySCF 2.13.0
(`tests/benchmarks/casscf/pyscf_casscf_timing.py`). Wall time.

| case | AOs / pairs | start | P4 `1468a4f7` | P5 `305de17a` | N4 `c3095a53` | **N5** | PySCF | N5 vs PySCF |
|---|---|---|---|---|---|---|---|---|
| 6-31G | 13 / 22 | 1.3-1.9 s | 0.18 s | 0.16 s | 0.15 s | **0.16 s** | 0.34 s | **0.46x** |
| cc-pVDZ | 25 / 50 | 13.4 s | 0.58 s | 0.59 s | 0.45 s | **0.43 s** | 0.27 s | 1.6x |
| cc-pVDZ, C1 | 25 / 138 | 44.5 s | 2.35 s | 0.72 s | 0.58 s | **0.59 s** | 0.43 s | 1.4x |
| cc-pVTZ | 65 / 142 | ~176 s per macro | 43.5 s | 24.3 s | 19.6 s | **13.3 s** | 1.16 s | 11x |
| cc-pVTZ, C1 | 65 / 418 | - | 51.8 s | 27.4 s | 23.6 s | **15.0 s** | 2.30 s | 6.5x |

N5 energies: cc-pVDZ -76.0440109040 (PySCF -76.0440109018), C1
-76.0781256190 (-76.0781256214), cc-pVTZ -76.0754286346 (-76.0754286541; 2e-8,
set by `tol_mcscf_grad 1e-5`), cc-pVTZ C1 -76.1092535242 (-76.1092535271).

**Two references were re-pinned from saddles to minima.** Water SA-2
(2026-10-28, -74.7751377977 -> -74.7877864783) and, in P5,
`oh_casscf_rohf_sto3g` (-74.3854276959 -> -74.3867176233). Both old values
have one negative relaxed-Hessian eigenvalue along a mode the gradient has no
component on; both new values are PySCF's own tight answers.

## Handoff

**Committed through `c3095a53` (N4).** **Uncommitted, gated, ready to commit
as N5:** `src/post_hf/integrals.cpp`, `src/post_hf/ci/rdm.cpp`, the new input
`tests/inputs/regression/post_hf/water_casscf44_c1_ccpvdz.hfinp` and its two
cases in `tests/regression_cases.json`. On that tree: 19/19
CASSCF/RASSCF/FCI cases, core suite 85/85, `planck-transform-eri` and
`planck-casscf-internal` pass, table energies bit-identical to N4.

**How to reproduce:**

- *Inputs.* Copy `tests/benchmarks/casscf/pyscf_reference/water_cas44_{631g,
  ccpvdz}.hfinp`. C1 variants: `sed 's/use_symm    .true./use_symm
  .false./'`. cc-pVTZ: `sed 's/cc-pvdz/cc-pvtz/'`. The uphill canary is
  `water_cas44_sto3g_sa2_sad_uphill.hfinp`.
- *Planck.* `export BASIS_PATH=$PWD/basis-sets`, then
  `OMP_NUM_THREADS=4 build-full/hartree-fock <input>`.
  `PLANCK_CASSCF_CALL_PROBE=1` prints the per-macro call counters.
- *PySCF.* `OMP_NUM_THREADS=4 tests/pyscf/.venv/bin/python
  tests/benchmarks/casscf/pyscf_casscf_timing.py [--c1] [basis ...]`.
- *Gates.*
  - `python3 tests/run_regressions.py --build-dir build-full --suite all
    --case <id>` for the 19 CASSCF/RASSCF/FCI cases (ids: grep
    `casscf|rasscf|_fci_` in `regression_cases.json`, minus fciqmc). `--case` without
    `--suite all` silently selects nothing.
  - The 14 inputs in `pyscf_reference/`, run directly, compared with
    `pyscf_reference_energies.md`.
  - `build-full/planck-casscf-internal`. Read its exit code directly, not
    through a pipe.

**Traps hit this session:**

- **The scratchpad was wiped** when the date rolled over. Keep nothing there
  that cannot be regenerated.
- **Build with `-j4`.**
- **Check the binary is newer than every edited source before trusting a
  run.** A `cp`-restored file keeps its old mtime, and a restore plus
  `touch` within the same second as the last link also skips the rebuild;
  `sleep 1` first.
- **`PLANCK_X=""` is set, not unset**: `getenv` returns non-null and
  `atoi("")` is 0. Use `env -u`.
- **A Python edit that asserts uniqueness writes nothing when it fails.**
  The `response.cpp` loop text appears twice (single-root and SA solves).
- **Rewriting `regression_cases.json` from Python** must use
  `json.dumps(d, indent=1)` (the default `ensure_ascii`); the file stores
  non-ASCII as `\u` escapes, and `ensure_ascii=False` rewrites unrelated
  cases.

## What made it slow

| # | cause | fix | commit |
|---|---|---|---|
| 1 | `transform_eri_active_cache` was O(nb^5 n_act): 87 % of a cc-pVDZ run | active legs first, O(nb^4 n_act) | P1 `9e240155` |
| 2 | every orbital Hessian product was an FD of two gradient rebuilds | analytic action | P2 `a330be6f` |
| 3 | the orbital-CI coupling dropped two terms, so the coupled solve never converged and `numeric-newton` (2·npairs full evaluations) ran every macro | exact coupling, then a direct solve of the CI-relaxed Hessian | P3b `69728821`, P3c `7a5031a8` |
| 4 | above 256 pairs the solve fell back to a few orbital-only diagonal steps | CI-relaxed subspace solve | P4a `1468a4f7` |
| 5 | a 0.2 decaying level shift and a fixed 0.2 step cap, tuned for an approximate Hessian, halved the convergence rate | trust region driven by the exact model | P4b `1468a4f7` |
| 6 | the dense CI-relaxed operator costs ~570 nb^4 Fock passes per macro at cc-pVTZ | matrix-free preconditioned solve, with a seed vector so it still sees zero-gradient negative modes | P5 `305de17a` |
| 7 | each product swept the ERI tensor 4 times, once for a density it had already built | two densities per sweep, memoized G[dD_c] | N4 `c3095a53` |
| 8 | `transform_eri` read the AO tensor at stride nb^3 in its first quarter | contracted index outside a unit-stride loop | N5 |

## What is left

1. **Re-profile cc-pVTZ** after N5. Before it: `transform_eri` first quarter
   37 %, batched `_compute_fock_rhf` 35 %, single-density
   `_compute_fock_rhf` 14 %, thread wait 10 %.
2. **The single-density `_compute_fock_rhf`** (inactive/active Fock and the
   post-response G[C_a gamma1 C_a^T]; also the conventional SCF's hot path)
   could use N4's register-accumulator form. Measure first; it needs the SCF
   gates, not just CASSCF's.
3. **RI path:** still an FD orbital Hessian, and a full-leg-first transform
   (`water_casscf_ri_ccpvdz` 4.1 s).
4. **Delete `CallProbe`** (`PLANCK_CASSCF_CALL_PROBE`) once this scope is
   answered.

Not needed: a new Davidson implementation, the AH solver for the relaxed
step (N2: seeded from g, it has the subspace solve's blind spot, and the
seeded solve already beats dense), a dense/matrix-free switch kept "just in
case", direct or cached Fock builds (disproven, `CASSCF_FOCK_REBUILD_COST.md`).

## Gates (every step)

- All 19 CASSCF/RASSCF/FCI regression cases (17, plus the
  `water_casscf44_c1_ccpvdz_threads1/4` pair) (`tests/run_regressions.py
  --suite all --case ...`). The uphill canary
  `water_casscf_sa2_sto3g_sad_guess_uphill` and the two hcore/SAD SA-2 cases
  pin which basin each start reaches, as does `oh_casscf_rohf_sto3g`.
- The 14 inputs in `tests/benchmarks/casscf/pyscf_reference/`,
  against their recorded Planck energies.
- `planck-casscf-internal`.
- `build-full/planck-transform-eri` and the core suite when touching a
  shared kernel (`transform_eri`, `_compute_fock_rhf`).
- Stop if any gated energy moves by more than 1e-8 Eh or a basin changes.

## Record (newest first)

### N4: two ERI sweeps per Hessian product (`c3095a53`)

A product made four `_compute_fock_rhf` sweeps for one root: G[dD_c] and
G[dD_a] in `analytic_g_action`, G[dD_c] again in each root's CI right-hand
side, and G[C_a gamma1 C_a^T] after each root's CI response. Now dD_c and
dD_a share one sweep (`fock_responses_mo`, a multi-density
`_compute_fock_rhf` overload, up to four densities per sweep), and the CI
right-hand side reuses G[dD_c] through a one-entry memo on
`OrbitalHessianContext` keyed on the AO density (a stale C cannot give a
wrong hit; the MO transform uses the current C). The post-response sweep
depends on the CI solve and stays separate: 4 -> 2 sweeps per product,
2 + nroots -> 1 + nroots for SA.

**The first version was 12 % slower at cc-pVTZ** (24.3 -> 27.2 s) while
faster at cc-pVDZ: accumulating into `G[d](mu, nu)` through a vector of
matrices kept the sums out of registers. Local accumulators over raw pointers
gave cc-pVTZ 24.3 -> 19.6 s, cc-pVTZ C1 27.4 -> 23.6 s. The single-density
build is untouched.

### N1-N3: making the matrix-free solve the default (`305de17a`)

- **N1, the 1-2e-8 Eh drift, was not the solve.** The matrix-free run crossed
  `tol_mcscf_grad 1e-5` one macro earlier (6-31G: 7.8e-6 at macro 7; dense
  was 1.01e-5 there and 2.1e-6 at macro 8), and dense at macro 7 is just as
  far from the converged energy. A tighter forcing cap changed nothing.
- **N2 (AH solver) dropped**, see "What is left".
- **N3.** `max_dense_pairs = 0` for the relaxed solve; deleted
  `PLANCK_CASSCF_DENSE_PAIRS` and the `nn`/`schur`/`alpha`/`fix`/`hess`/
  `subspace`/`lanczos` probes. The non-relaxed coupled solve keeps its dense
  limit (it has no matrix-free branch). The seed vector and the OH re-pin
  this needed are in P5.

### P5: matrix-free relaxed solve everywhere (`305de17a`)

**Profile** (cc-pVTZ C2v after P4, whole run, 4 threads):
`_compute_fock_rhf` 76.3 %, `transform_eri` 15.6 %, thread wait 7.3 %. The
dense relaxed operator is built one column per pair (142 at cc-pVTZ C2v), ~4
nb^4 Fock contractions per column: ~570 passes over a 143 MB tensor per
macro, against PySCF's ~17 JK builds.

**Tried and dropped:** plain Lanczos at every size. Its 1e-8 |g| test never
stopped early; at cc-pVTZ the 64 cap made steps inexact (25 macros, 74 s).

**What landed:** a preconditioned subspace (Davidson-style) Newton solve.

- Expand with `r_perp / |diag H|`, where `r_perp = r - V V^T r` is the
  residual outside the subspace. Inside it the residual is the deliberate
  floor modification `(T - T_floor) y`, which never vanishes once a mode is
  clamped; testing the full residual made every solve run to n or the cap.
- Stop at the inexact-Newton forcing term
  `||r_perp|| <= min(0.1, sqrt|g|)·|g|`.
- **The second vector is a fixed generic seed** (`1 + 0.5 sin(1+i)`, over
  the symmetry-allowed pairs), not a residual. Without it, forced on
  everywhere, the SA-2 hcore and SAD starts returned to the -74.7751 saddle:
  a Krylov space grown from g never contains a negative mode g is orthogonal
  to, so the solve cannot amplify it the way the dense floor recipe does
  (1e-16 -> 3e-4 over ~9 macros). With the seed it can, and every table
  input matches dense to <= 1e-10 Eh.

**The seed exposed a second saddle pin.** `oh_casscf_rohf_sto3g` (CAS(5,4),
`use_symm .false.`) went to -74.3867176162. Dense ends at -74.3854277116 with
one negative eigenvalue (-3.8e-3, |g.v0| ~ 2e-16) and converges before the
roundoff grows. PySCF (cart, `conv_tol 1e-10`, `conv_tol_grad 1e-6`) reaches
-74.3867176233 from its own unperturbed ROHF start and from three perturbed
ones, so the old pin was a saddle. Re-pinned to -74.3867176233 (Planck 7e-9
away). `use_symm .true.` does not change it: matrix-free still reaches
-74.3867, so the descent mode is symmetry-allowed in the group used.

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

### P3: why the coupled step lost (`9e4d0672`)

- **The coupled solve never converged:** 0 of 56 macros. `numeric-newton` is
  offered whenever the coupled step is unreliable, so it ran every macro.
  Raising `mcscf_max_rot` to 10 changed nothing.
- **The coupled equations were wrong in shape.** Along R = the gradient,
  compare the code's CI-relaxation term `dc` with the true one `dr` (FD of
  the fully CI-re-solved gradient minus the exact fixed-CI action):
  cos(dc, dr) <= 0.75, often negative, and the best-fit prefactor ranges
  from -2.1 to 5.5. No constant factor fixes it.

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

### P1: (pu|vw) transform in O(nb^4 n_act) (`9e240155`)

`(pu|vw) = (vw|up)` under the AO tensor's 8-fold symmetry, so the routine is
`transform_eri(eri, nb, C_act, C_act, C_act, C)` plus a permutation (net -116
lines). cc-pVDZ 13.4 -> 3.5 s.

Gate: against `transform_eri(C, C_act, C_act, C_act)` on an 8-fold-symmetric,
non-square, dense fixture. The old fixture had no ERI symmetry at all. A
u<->w mutant fails it.

### P0: measurement (`87d5a10f`)

`PLANCK_CASSCF_CALL_PROBE=1` prints per-macro call counts and the P3-P5
diagnostics. All of them are temporary. Starting profile (cc-pVDZ, 1 thread):
`transform_eri_active_cache` 87.3 %, `_compute_fock_rhf` 8.3 %, CI/RDMs/Q
< 1 %. At cc-pVTZ the first two macros took ~176 s each, ~1600x PySCF per
macro.

## Out of scope

- CI cost: irrelevant at CAS(4,4); the large-CAS sigma build is done
  (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`).
- Direct or cached Fock builds: disproven (`CASSCF_FOCK_REBUILD_COST.md`).
