# Why was CASSCF so much slower than PySCF, and what makes it fast?

Answered 2026-11-01 (opened as a scope 2026-10-28). Status lives in
`vault/Status/`.

## Answer

**It was slow for two reasons that compounded: the optimizer did far more
work per macroiteration than it needed, and every piece of that work read the
full nb^4 ERI tensor.** The optimizer's orbital-CI coupling was missing two
terms, so its Newton step never converged and a finite-difference Newton
fallback (2·npairs full energy evaluations) ran every macro; its step policy
was tuned for an approximate Hessian; and its selector re-evaluated the energy
22 times per macro. Underneath, the active-space transform was O(nb^5 n_act),
each Hessian-vector product made four passes over the tensor, and each energy
evaluation made four more.

**What makes it fast:** an exact CI-relaxed orbital Hessian, solved
matrix-free by a preconditioned subspace Newton solve; a trust region driven
by that exact model; taking the Newton step as soon as it improves; and one
object, the half-transformed integrals `H[mu,nu,lam,i] = (mu nu|lam i)` with i
over the core and active orbitals, built once per energy evaluation. Every
density the code contracts has the form `X Y^T + Y X^T` with X those occupied
columns, so H gives F_I, F_A, (tu|vw), (pu|vw), the Hessian's mixed blocks and
every Hessian-product Fock build as O(nb^3) contractions. The tensor is read
about twice per macroiteration.

**Result** (water CAS(4,4), Cartesian, 4 threads, `build-full`, PySCF 2.13.0
via `tests/benchmarks/casscf/pyscf_casscf_timing.py`; wall time):

| case | AOs / pairs | before | now | PySCF | now vs PySCF |
|---|---|---|---|---|---|
| 6-31G | 13 / 22 | 1.3-1.9 s | 0.069 s | 0.34 s | 0.20x |
| cc-pVDZ | 25 / 50 | 13.4 s | 0.177 s | 0.27 s | 0.66x |
| cc-pVDZ, C1 | 25 / 138 | 44.5 s | 0.266 s | 0.43 s | 0.62x |
| cc-pVTZ | 65 / 142 | ~176 s per macro | 1.49 s | 1.16 s | 1.3x |
| cc-pVTZ, C1 | 65 / 418 | not run | 1.86 s | 2.30 s | 0.81x |

Energies match PySCF to the convergence tolerance: cc-pVDZ -76.0440109040
(PySCF -76.0440109018), C1 -76.0781256196 (-76.0781256214), cc-pVTZ
-76.0754286346 (-76.0754286541; 2e-8, set by `tol_mcscf_grad 1e-5`), cc-pVTZ
C1 -76.1092535257 (-76.1092535271). Planck needs no more macroiterations than
PySCF anywhere (cc-pVTZ 9 vs 9, C1 10-17 vs 20). The remaining 1.3x at
cc-pVTZ C2v was accepted (2026-11-01); see "Not done".

## How the time went, step by step

| step | commit | cc-pVDZ | cc-pVDZ C1 | cc-pVTZ | cc-pVTZ C1 |
|---|---|---|---|---|---|
| start | | 13.4 s | 44.5 s | ~176 s/macro | - |
| P1 transform O(nb^4 n_act) | `9e240155` | 3.5 s | 11.1 s | | |
| P2 analytic Hessian action | `a330be6f` | 2.7 s | 10.0 s | | |
| P3c direct relaxed solve | `7a5031a8` | 1.21 s | 2.70 s | | |
| P4 Lanczos + trust region | `1468a4f7` | 0.58 s | 2.35 s | 43.5 s | 51.8 s |
| P5 matrix-free + seed | `305de17a` | 0.59 s | 0.72 s | 24.3 s | 27.4 s |
| N4 two sweeps per product | `c3095a53` | 0.45 s | 0.58 s | 19.6 s | 23.6 s |
| N5 unit-stride `transform_eri` | `bc36af7b` | 0.43 s | 0.59 s | 13.3 s | 15.0 s |
| N6 take the Newton step | `80eb776d` | 0.245 s | 0.367 s | 7.3 s | 8.5 s |
| N7 Fock builds from H | `9e55b67a` | 0.248 s | 0.328 s | 3.0 s | 3.7 s |
| N8 one pass over H; X, Z from H | `aad9d399` | 0.180 s | 0.285 s | 1.85 s | 2.33 s |
| N9 H per evaluation | `a224881a` | 0.177 s | 0.266 s | 1.49 s | 1.86 s |

## What made it slow

| # | cause | fix | step |
|---|---|---|---|
| 1 | `transform_eri_active_cache` put the full-MO leg first: O(nb^5 n_act), 87 % of a cc-pVDZ run | `(pu|vw) = (vw|up)`, so contract the three active legs first | P1 |
| 2 | every orbital Hessian product was a central FD of two gradient rebuilds | analytic action | P2 |
| 3 | the orbital-CI coupling dropped two terms, so the coupled solve never converged (0 of 56 macros) and `numeric-newton` ran every macro | exact coupling, then solve the CI-relaxed Hessian directly | P3b, P3c |
| 4 | above 256 pairs the solve fell back to a few orbital-only diagonal steps | CI-relaxed subspace solve | P4a |
| 5 | a 0.2 decaying level shift and a fixed 0.2 step cap, tuned for an approximate Hessian, halved the convergence rate | trust region driven by the exact model | P4b |
| 6 | the dense relaxed operator cost ~570 nb^4 Fock passes per macro at cc-pVTZ (one column per pair) | matrix-free subspace solve, with a seed vector | P5 |
| 7 | `transform_eri` read the AO tensor at stride nb^3 in its first quarter (37 % at cc-pVTZ) | contracted index outside a unit-stride loop | N5 |
| 8 | the step selector scored 3 candidates x 7 scales, 22 energy evaluations, every macro | take the full Newton step when it improves | N6 |
| 9 | each Hessian product made full passes over the tensor for densities of rank <= 2 n_occ | Fock builds from H in O(nb^3 n_occ) | N4, N7, N8 |
| 10 | the analytic Hessian's (pa|vw) and (pu|aw) were two more tensor transforms per macro | three O(nb^3 n_act^2) contractions of H | N8 |
| 11 | each energy evaluation made ~4 tensor passes (F_I, F_A per root, (tu|vw), the cache) and the Hessian rebuilt H | build H once per evaluation, derive the rest, reuse it | N9 |

## How it works now

### The orbital step: exact CI-relaxed Newton (P2, P3b, P3c)

The Hessian is the CI-relaxed one, `H_oo + H_oc (H_cc)^-1 H_co`: the curvature
of the energy when the CI vector is re-solved for every trial rotation.

- **Fixed-CI part (P2).** With `C(eps) = C(I + eps R)`:
  `dF_I = R^T F_I + F_I R + C^T G[dD_c] C`, `dF_A` likewise, and
  `dQ = R^T Q + sum R(a, n_core+x) Y[p,a,t,x]`, with `Y` built per macro from
  (pa|vw) and (pu|aw). The action is `compute_orbital_gradient` applied to the
  derivatives (`analytic_g_action`, `orbital.cpp`). The central FD is kept
  only as the test oracle and for the RI path.
- **CI coupling (P3b).** Two terms were missing, and both are needed:

  | variant | cos(d, dr) | \|dr - d\|/\|rel\| |
  |---|---|---|
  | as was | -0.50 .. 0.75 | 1e-2 .. 3e-1 |
  | A only: core-density term in the CI RHS, `(C^T G[dD_c] C)_act` | 0.07 .. 0.75 | 3e-2 .. 3e-1 |
  | B only: 1-RDM response in the back-coupling, via `F_A` and `gamma·F_I` | -0.80 .. 0.99999 | 9e-3 .. 2e-1 |
  | **A + B** | **1.000000** | **7e-10 .. 1e-8** |

  (`d` is the code's CI-relaxation term along R = g, `dr` the FD of the fully
  re-solved gradient minus the exact fixed-CI action.) Before this, no
  constant prefactor fixed `d` (best fit -2.1 to 5.5), so the coupled step
  could never win and the FD Newton fallback ran every macro.

### Solving without building H (P4a, P5)

A dense build needs one product per pair (142 at water/cc-pVTZ, 418 without
symmetry), each with its own CI response. `solve_orbital_action_system`
(`response.cpp`) runs a preconditioned subspace (Davidson-style) Newton solve
instead:

- the subspace grows by `r_perp / |diag H|`, the diagonal being F_I + F_A
  differences;
- inside the subspace the step uses the dense recipe: eigenvalues below 1e-4
  are floored before inversion;
- it stops when the residual **outside** the subspace,
  `r_perp = r - V V^T r`, meets the inexact-Newton forcing term
  `min(0.1, sqrt|g|)·|g|`. The residual inside the subspace is the deliberate
  floor modification `(T - T_floor) y` and never vanishes once a mode is
  clamped; testing the full residual made every solve run to n or the 64 cap;
- **the second vector is a fixed generic seed** (`1 + 0.5 sin(1+i)` over the
  symmetry-allowed pairs), not a residual. See the next section.

### Saddles the gradient cannot see

Two pinned references were saddles, not minima:

| case | old pin | lowest relaxed eigenvalue there | minimum (PySCF's tight answer) |
|---|---|---|---|
| water SA-2 STO-3G (hcore and SAD starts) | -74.7751377977 | -0.054 to -0.065 | -74.7877864783 |
| `oh_casscf_rohf_sto3g` CAS(5,4) | -74.3854276959 | -3.8e-3 | -74.3867176233 |

At both, the gradient is orthogonal to the negative mode to roundoff
(|g·v0| ~ 1e-16): the inputs run with symmetry off, but symmetric starting
orbitals give a symmetric gradient with no component along a
symmetry-breaking direction. A Newton step leaves only by amplifying that
roundoff: the 1e-4 eigenvalue floor multiplies it ~1e4 per macro (1e-16 ->
3e-4 by macro 9). PySCF leaves such saddles the same way.

So **a Krylov solve started from g can never leave**: its space never
contains v0. Forced on everywhere without the seed, the subspace solve sent
the SA-2 starts back to -74.7751. With the seed it sees the same negative mode
a dense solve sees, every table input matches the dense solve to <= 1e-10 Eh,
and it also found the OH saddle, which the dense solve had stopped on because
it converged before the roundoff grew. `use_symm .true.` does not pin OH
either: the descent mode is symmetry-allowed in the group used.

### Step control (P4b, N6)

- **Trust region.** `predicted_dE = g·s + ½ s·H s` with the exact relaxed
  action; rho = actual/predicted is 0.89-1.30 on every macro measured. The
  radius `trust_rot` doubles (up to 5 x `mcscf_max_rot`) when rho >= 0.75 and
  the step hit it, shrinks x4 when rho < 0.25 or the model predicts an
  increase, and halves when no step is accepted. The level shift stays at its
  1e-3 floor unless steps are rejected. Macros: water 6-31G 17 -> 8, cc-pVDZ
  17 -> 8, SA-2 16 -> 10, cc-pVTZ 17 -> 9 (PySCF 9), cc-pVTZ C1 24 -> 17
  (PySCF 20).
- **Take the Newton step (N6).** `select_best_candidate_step` takes the first
  candidate, the coupled Newton step, without scoring the others when at full
  scale it lowers the energy and either does not hurt the gradient or lowers
  the merit. Otherwise the full search over candidates x 7 scales and the
  stagnation probes run as before, so the uphill path is unchanged. 22 -> 2
  energy evaluations per macro. Requiring "gradient not worse" alone was not
  enough: far from convergence the Newton step raises the gradient while
  lowering energy and merit, and macros 1-2 still scored all 21.

### One half transform per evaluation (N7, N8, N9)

`OccupiedHalfTransform` (`orbital.h`) holds
`H[mu,nu,lam,i] = sum_sig (mu nu|lam sig) C(sig,i)` for i over the first
n_core + n_act MOs: one pass over the tensor, nb^3·n_occ doubles (11 MB at
water cc-pVTZ). Every density the code contracts is `X Y^T + Y X^T` with
X = those columns of C:

| density | Y |
|---|---|
| core, 2 C_c C_c^T (F_I) | C_core on the core columns |
| active, C_a gamma C_a^T (F_A) | C_act gamma / 2 |
| dD_c (orbital action, CI right-hand side) | 2 C R[:, core] |
| dD_a (orbital action) | C R[:, act] gamma |
| C_a gamma1 C_a^T (after each root's CI response) | C_act gamma1 / 2 |

With the ERI permutation symmetry, `G = J - K/2` with
`J(mu,nu) = 2 sum_{lam,i} H[mu,nu,lam,i] Y(lam,i)` and `K = T + T^T`,
`T(mu,nu) = sum_{lam,i} Y(lam,i) H[mu,lam,nu,i]`
(`occupied_two_electron_ao`), O(nb^3 n_occ) per density. One pass over H
forms both: the contiguous block `H[mu,a,:,:]` gives J(mu,a) and adds to every
T(mu,nu). This is PySCF's in-core `ppaa`/`papa` idea.

The mixed blocks come from H too (`transform_occupied_half`): (tu|vw),
(pu|vw), (pa|vw) and (pu|aw) are three O(nb^3 n_act^2) contractions each,
provided the leg paired with H's active index is contracted first (the other
order is O(nb^4 n_act) again).

`evaluate` builds H at the trial C and stores it in `McscfState::half`; the
next macro's `build_analytic_orbital_hessian` reuses the accepted state's H.
The RI path keeps the dense-density builders and the FD Hessian.

### Determinism (N5)

The 1- and 2-RDM builds (`src/post_hf/ci/rdm.cpp`) scheduled tasks
dynamically and merged per-thread partials under `omp critical`, i.e. in
completion order. Water/cc-pVDZ C1 at 4 threads gave -76.0781256180 ..
-76.0781256195 across identical runs: the trace diverged at ~1e-10 in macro 2
and the indefinite early region amplified it. They now use the FCI sigma
build's discipline: 64 bins fixed by task index, `schedule(static, bin_size)`
so a chunk is one bin, bins summed in fixed order. The trace is identical at
1/2/4/8 threads.

## What was tried and did not work

- **Raising `mcscf_max_rot` to 10** before P3: changed nothing. The step size
  was never the problem; the coupled equations were wrong in shape.
- **The FD Hessian as the culprit (P2):** making the curvature exact changed
  no macro's winning candidate. The missing coupling terms were.
- **Plain Lanczos at every size (P5):** its 1e-8 |g| test never stopped
  early; at cc-pVTZ the 64-vector cap made steps inexact (25 macros, 74 s).
- **Suppressing the saddle escape** (dropping clamped modes with
  |g·v| <= 1e-8|g|): kept the SA-2 hcore case on its saddle but sent three
  others wrong: the uphill canary stayed on the saddle, spherical 6-31G* went
  to -76.0256 (expected -76.0610), C1 cc-pVDZ to -76.0440 (expected -76.0781,
  where PySCF itself escapes).
- **The augmented-Hessian solver for the relaxed step (N2):** dropped
  unbuilt. It is also seeded from g, so it has the subspace solve's blind
  spot, and the seeded subspace solve already beat the dense one everywhere.
- **A tighter forcing term for the 1-2e-8 Eh end-point drift (N1):** changed
  nothing. The drift was the run crossing `tol_mcscf_grad 1e-5` one macro
  earlier (6-31G: 7.8e-6 at macro 7; dense 1.01e-5 there), and dense at that
  macro is just as far from the converged energy.
- **Batching densities into one tensor sweep (N4):** worked (cc-pVTZ
  24.3 -> 19.6 s) but N7 made it pointless and it was deleted. Its first
  version was 12 % slower at cc-pVTZ: accumulating into `G[d](mu, nu)`
  through a vector of matrices kept the sums out of registers.
- **Direct or cached Fock builds:** disproven earlier
  (`CASSCF_FOCK_REBUILD_COST.md`).

## Gates

- The 19 CASSCF/RASSCF/FCI regression cases (`python3 tests/run_regressions.py
  --build-dir build-full --suite all --case <id>`; ids: grep
  `casscf|rasscf|_fci_` in `tests/regression_cases.json`, minus fciqmc).
  `--case` without `--suite all` silently selects nothing. The uphill canary
  `water_casscf_sa2_sto3g_sad_guess_uphill`, the two hcore/SAD SA-2 cases and
  `oh_casscf_rohf_sto3g` pin which basin each start reaches;
  `water_casscf44_c1_ccpvdz_threads1/4` (atol 0, threads1 pinned to PySCF)
  pins thread-count invariance and fails on the old `rdm.cpp`.
- The 14 inputs in `tests/benchmarks/casscf/pyscf_reference/`, run directly,
  against `pyscf_reference_energies.md`.
- `build-full/planck-casscf-internal`, read by exit code, not through a pipe.
  It checks the analytic Hessian against Richardson FD (9e-11 relative; zeroing
  the w-leg fails it at 5.3), each coupling term against FD (dropping either
  fails it), and the H-based G against the dense build (2.8e-15; dropping the
  factor 2 in J fails it).
- `planck-transform-eri` and the core suite when touching a shared kernel
  (`transform_eri`, `_compute_fock_rhf`).

**Reproducing the table.** Copy
`tests/benchmarks/casscf/pyscf_reference/water_cas44_{631g,ccpvdz}.hfinp`;
C1: `sed 's/use_symm    .true./use_symm    .false./'`; cc-pVTZ:
`sed 's/cc-pvdz/cc-pvtz/'`. Run `OMP_NUM_THREADS=4 build-full/hartree-fock`
with `BASIS_PATH=$PWD/basis-sets`, and the PySCF side with
`tests/pyscf/.venv/bin/python tests/benchmarks/casscf/pyscf_casscf_timing.py
[--c1] [basis ...]`.

**Traps.**

- A `cp`-restored source keeps its old mtime, and a restore within the same
  second as the last link also skips the rebuild: `sleep 1` and `touch` before
  rebuilding, and check the binary is newer than every edited source.
- `PLANCK_X=""` is set, not unset (`getenv` is non-null, `atoi("")` is 0);
  use `env -u`.
- Rewriting `regression_cases.json` from Python must use
  `json.dumps(d, indent=1)` with the default `ensure_ascii`; the file stores
  non-ASCII as `\u` escapes, and `ensure_ascii=False` rewrites unrelated
  cases.

## Not done

Not scheduled; take one up only with a new target.

1. **Re-profile cc-pVTZ C2v** (1.3x behind PySCF). Whether the remaining time
   is the H build, the Hessian products (O(nb^3) each, but 28-64 per macro),
   the SCF before CASSCF, or thread wait has not been measured.
2. **A better preconditioner for the relaxed solve.** 28-64 vectors per macro
   at cc-pVTZ; near convergence cc-pVDZ still needs 34-50 of 50 for a
   relative residual of ~4e-3. The diagonal is F_I + F_A differences only.
3. **The RI path** has none of this: FD orbital Hessian, full-leg-first
   transform, dense-density Fock builds (`water_casscf_ri_ccpvdz` 4.1 s). An
   RI twin of H (`B^Q_{mu i}`) would give it the same structure.
4. **`CallProbe`** (`PLANCK_CASSCF_CALL_PROBE`, `orbital.h`) is temporary and
   can be deleted.

## Not in this scope

- CI cost: irrelevant at CAS(4,4); the large-CAS sigma build is done
  (`docs/FCI_SIGMA_BUILD_PERFORMANCE.md`).
