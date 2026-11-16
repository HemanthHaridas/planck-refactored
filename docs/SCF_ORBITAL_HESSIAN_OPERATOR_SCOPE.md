# Can the RHF orbital Hessian be matrix-free, so RHF and UHF share one path?

Scope, opened 2026-11-16. It expires when the work lands; rewrite it then as
the answer. Companion to `docs/RI_MP2_GRADIENT_RESPONSE_SCOPE.md`: doing
this first makes that fix fall out of it (see "Order").

## Status (2026-11-16)

**Steps 1–5 of "Order" are done and gated; the work is uncommitted.** The
RHF and UHF orbital Hessians are one matrix-free operator
(`Correlation::OrbitalHessian`), every consumer applies it, the four explicit
builders are deleted, and the fitted/exact choice is made in one place
(`SCF::two_electron_fock{,_uhf}`, keyed on `scf_ri_jk`). That fixed the
RI-MP2 gradient, and spherical RI-MP2 gradients now run.

| Gate | Result |
|---|---|
| 1: operator vs dense A (`planck-ri-cphf-matrix`, post-deletion build) | exact 4.5e-16; fitted vs exact 7.0e-5 (fitting error, as required); UHF on [x, x] vs RHF 0. Before the deletions, fitted operator vs the old `build_rhf_cphf_matrix_ri` was 1.2e-15 |
| 2: every MP2 gradient/geomopt/freq, HF/KS SOSCF and stability, and RI case (100) | 99 pass; the one failure is `water_ri_rmp2_spherical_gradient_rejected`, whose input was renamed during the run (that case is retired, see below) |
| 3: RI-MP2 gradient vs its own energy FD at 3e-7 Ha/Bohr | water RI-RMP2 6-31G* 1.3e-7 (was 3.9e-6); water⁺ RI-UMP2 1.2e-7 (was 1.6e-6); the STO-3G FD gates, tightened from 3e-4 to 3e-7, measure 2.4e-7 / 1.6e-7 |
| Spherical RI-MP2 (refusal removed) | water RI-RMP2 1.1e-7 (was 3.5e-6); water⁺ RI-UMP2 1.5e-7 (was 1.6e-6) |

On the final build (refusal removed): `--suite core` 174/174, and the six
RI-MP2 FD gates plus the RI-MP2 geomopt and frequency pins 10/10, no
re-pins needed.

## What remains

1. **Cost.** Time RHF SOSCF and an RMP2 gradient before (`712a299c`) and
   after on water/6-31G* (conventional) and a water cluster/cc-pVDZ
   (direct), on an idle machine. If the small conventional case regresses
   noticeably, decide whether `two_electron_fock` should contract a cached
   `_eri`.
2. **Docs.**
   - Rewrite this file and `docs/RI_MP2_GRADIENT_RESPONSE_SCOPE.md` as
     answers.
   - Update `docs/RI_JK_ROUTINE_SCOPE.md` "Still open" (and line ~518, which
     names the retired `_rejected` case), the teaching-guide RI-MP2 row
     (regenerate `docs/index.html` with `python3 docs/build_teaching_site.py`),
     and `docs/SPHERICAL_MP2_GRADIENTS.md` (line ~153 lists the refusal as
     kept).
   - Vault: a `Completion.md` entry (and fix line ~609, which cites the
     retired case), and the RI item in `Open Work.md`.
3. **Commit**, staging only this work's files (below).

## Working tree

Everything is **uncommitted** on `research/wpbeh-smooth`; the last commit is
`712a299c`. The tree also holds the user's unrelated, uncommitted FCIQMC work
(`README.md`, `docs/FCIQMC_SCALING_INVESTIGATION.md`, `src/base/types.h`,
`src/io/io.cpp`, `src/post_hf/ci/fciqmc*`, `src/post_hf/fciqmc_driver.cpp`,
`tests/fciqmc_walkers.cpp`, `docs/RYDBERG_ORBITALS_SCOPE.md`) and a Rydberg
hunk in `vault/Status/Open Work.md` (~line 137). **Do not stage those.**
When staging shared files, stage hunks by header and stop on any mismatch;
a staging slip once committed the Rydberg hunk and had to be amended out.

### Changes in source

| Order step | Change | Files |
|---|---|---|
| 1 | `SCF::two_electron_fock{,_uhf}` return `build_ri_fock_rhf/uhf` when the SCF ran RI-JK (error if the RI cache is not primed); header comment says it is the one place the choice is made | `src/scf/scf.{h,cpp}` |
| 2 | `Correlation::OrbitalHessian`: `rhf()`/`uhf()` factories, `operator()` (one `two_electron_fock{,_uhf}` call; NaN on failure), `diagonal()`, `materialize()` | `src/post_hf/orbital_hessian.{h,cpp}` (new), added to `POSTHF_DFT_SRC` in `CMakeLists.txt` |
| 3 | `solve_rhf_cphf`: the operator plus GMRES (diagonal preconditioner, 1e-10, as `solve_uhf_cphf`), with the same RHF/closed-shell/shape checks. `PLANCK_DEBUG_RHF_RESPONSE=1` still dumps A (via `materialize()`), rhs and z. `solve_uhf_cphf` uses `OrbitalHessian::uhf`. RHF and UHF SOSCF use the operator as `h_op` (`scf.cpp`, run_rhf ~line 920, run_uhf ~1605) | `src/post_hf/{rhf,uhf}_response.{h,cpp}`, `src/scf/scf.cpp` |
| 4 | Deleted `build_rhf_cphf_matrix`, `build_rhf_cphf_matrix_ri`, `build_uhf_cphf_matrix`, the old `UHFOrbitalHessian` and `RI::fitted_response`. `build_veff_from_density{,_spin_densities}` call `two_electron_fock{,_uhf}` with no RI branch. Comments that named the deleted functions now name `OrbitalHessian` | `src/post_hf/mp2_gradient.cpp`, `src/post_hf/ri/ri_eri.h`, `src/dft/{analytic_hessian.h,response_packing.h,driver.cpp}`, three `tests/dft_*.cpp` comments |
| Gate 1 | `tests/ri_cphf_matrix.cpp` compares the operator with a dense A it builds from `transform_eri`. The target `planck-ri-cphf-matrix` now builds from the full-stack lists (as `planck-udh-validation`, plus `BSSE_SRC`, which the DFT driver's counterpoise call needs to link) | `tests/ri_cphf_matrix.cpp`, `CMakeLists.txt` |
| 5 | Removed the spherical RI-MP2 gradient refusal (and the now-unused `needs_gradient` block) | `src/hf_driver.cpp` |
| Gates 3, 5 | New inputs `post_hf/water_ri_rmp2_gradient_631gd`, `open_shell/water_cation_ri_ump2_gradient_631gd`, `spherical/water_cation_ri_ump2_spherical_gradient_631gd`; `spherical/water_ri_rmp2_spherical_gradient_rejected` renamed (`git mv`) to `..._gradient_631gd`. Cases `*_ri_*mp2_*gradient_631gd_fd` (four, extended, `--atol 3e-7`) replace `water_ri_rmp2_spherical_gradient_rejected`; the two STO-3G RI-MP2 FD cases get `--atol 3e-7` | `tests/inputs/regression/...`, `tests/regression_cases.json` |

### If something fails

- **A dense MP2 gradient moves by more than ~1e-9:** check the GMRES
  residual (`solve->true_residuals`). The old path solved A exactly by QR,
  and the operator equals that A to 4.5e-16 (gate 1), so a difference can
  only come from the solve.
- **An RI-JK SOSCF or stability case moves:** the fitted operator now comes
  from `two_electron_fock` under `scf_ri_jk`, which equals the old fitted A
  to 1.2e-15. Check that the RI cache is primed at that point (the new error
  message says so if it is not).
- **An RI-MP2 FD gate misses by ~4e-6 again:** a response consumer reaches a
  fitted operator. `git grep build_ri_fock` should show only `scf.cpp`
  (inside `two_electron_fock`) and the RI-JK SCF loop.

The sections below are the original scope: the answer, design and plan that
steps 1–5 implemented.

## Answer

**Yes, and nothing in production needs the explicit matrix.** Every
consumer of an HF orbital Hessian only applies it to vectors:

| Consumer | What it needs | RHF today | UHF today |
|---|---|---|---|
| MP2 gradient Z-vector (`solve_rhf_cphf`, `solve_uhf_cphf`) | solve A z = −b | explicit A, pivoted QR | operator + GMRES ✓ |
| HF SOSCF (`scf.cpp`, RHF ~905, UHF ~1592) | `h_op = Amat * x` inside the augmented-Hessian solver | explicit A | explicit A built **column by column from the operator**, one Fock build per column |

There are four implementations of one quantity today:
- **RHF exact:** `build_rhf_cphf_matrix` builds the full AO ERI tensor
  (`ensure_eri`), an nao⁴ MO transform, then an explicit A.
- **RHF fitted:** `build_rhf_cphf_matrix_ri` builds an explicit A from
  fitted MO blocks (nov²·naux).
- **UHF:** the `UHFOrbitalHessian` operator, matrix-free in the gradient and
  materialized for SOSCF.
- **The fitted/exact choice** is made in three places with two different
  rules: `RI::fitted_response` (CPHF), and `mp2_use_ri` (the MP2 gradient's
  `veff`).

**The RHF action is one Fock build**, the same as UHF. For a trial rotation
x (n_virt × n_occ), with D = C_v x C_oᵀ and D_s = D + Dᵀ:

  (A x)_ai = (ε_a − ε_i) x_ai + [C_vᵀ · 2 G(D_s) · C_o]_ai,  G(D) = J(D) − ½K(D)

This is A_{ai,bj} = (ε_a−ε_i)δ + 4(ai|jb) − (ab|ji) − (aj|bi) contracted with
x: J(D_s)_ai = 2Σ x_bj (ai|bj), and K(D_s)_ai = Σ x_bj [(ab|ij) + (aj|ib)].
G is exactly what `SCF::two_electron_fock` returns. (For a closed shell,
the UHF action on [x, x] gives the same thing per spin, at the price of two
K builds.)

## Design: one operator and one two-electron builder

1. **`SCF::two_electron_fock{,_uhf}` become "the SCF's two-electron
   operator"** and honor `scf_ri_jk`: `build_ri_fock_rhf/uhf` when set, the
   direct build otherwise. Their header comment already says they are for
   response and gradient code. This is the one place the fitted/exact choice
   is made, keyed on the operator the SCF actually used.
2. **One orbital-Hessian operator** in `post_hf/` for RHF and UHF, the
   existing `UHFOrbitalHessian` generalized: RHF packs one (a, i) block and
   calls `two_electron_fock` with 2·G; UHF keeps its two blocks and calls
   `two_electron_fock_uhf`. Both have the orbital-gap diagonal for GMRES
   preconditioning.
3. **Consumers take the operator:**
   - `solve_rhf_cphf` becomes GMRES with the diagonal preconditioner, at the
     same 1e-10 relative tolerance as `solve_uhf_cphf`;
   - RHF and UHF SOSCF pass the operator as `h_op`; no matrix is built.
4. **Deleted:** `build_rhf_cphf_matrix`, `build_rhf_cphf_matrix_ri`,
   `build_uhf_cphf_matrix`, `RI::fitted_response`, and the `mp2_use_ri`
   branches in `build_veff_from_density{,_spin_densities}` (they call
   `two_electron_fock{,_uhf}`).

**Kept, as views of the operator:** `PLANCK_DEBUG_RHF_RESPONSE` dumps A by
applying the operator to unit vectors, only when the variable is set. That
is the same materialization `build_uhf_cphf_matrix` does today, so the MP2
benchmark comparison script (`tests/benchmarks/mp2/pyscf_reference/rmp2_grad`)
keeps working.

## What changes in results

- **Exact RHF and UHF:** nothing beyond solver tolerance. The RHF Z-vector
  moves from QR to GMRES at a 1e-10 relative residual; the UHF switch made
  the same change and moved gradients ≤ 6.5e-13 Ha/Bohr. SOSCF takes the
  same Newton steps up to the augmented-Hessian solver's own tolerance.
- **RI-MP2 gradients (exact SCF):** the response becomes exact. That is the
  RI-MP2 fix: water 6-31G*/cc-pVDZ-RIFIT moves from 3.9e-6 / 1.6e-6 to the
  1.3e-7 / 1.2e-7 FD floor (measured with the equivalent experiment).
- **RI-JK SCF:** the response stays fitted, now through
  `two_electron_fock`, as step 4 of `docs/RI_JK_ROUTINE_SCOPE.md` intended.
  The `*_rijk_soscf_stability_631gd` cases pin it.
- **HF SOSCF under `mp2_use_ri` alone** stops using a fitted Hessian for an
  exact SCF, so it is consistent again.

## Cost

- **UHF SOSCF:** n Fock builds per Newton step (the column-by-column matrix)
  become k ≈ 5–10 (one per augmented-Hessian iteration). Large win.
- **RHF SOSCF and the RMP2 Z-vector:** one AO tensor build plus an nao⁴
  transform and a dense QR become k direct Fock builds. Under direct SCF
  that is a large win in memory and time: the old path built the full tensor
  inside a direct run. Under a small conventional SCF, the old path could
  reuse a cached tensor, so direct builds may be slower there. Measure both
  on water/6-31G* and a water cluster/cc-pVDZ before deciding whether
  `two_electron_fock` should contract a cached `_eri` when one exists.
- **Fitted (RI-JK):** k RI Fock builds instead of nov²·naux, a win.

## Gates

1. **Before deleting anything:** for exact RHF, the new operator
   materialized on unit vectors must equal `build_rhf_cphf_matrix` to 1e-12,
   on water 6-31G* Cartesian and spherical. For RI-JK, it must equal
   `build_rhf_cphf_matrix_ri`. This becomes the rewritten
   `planck-ri-cphf-matrix`, which today compares the fitted A against the
   exact one; it turns into an operator-against-operator check (fitted vs
   exact actions on random vectors, to fitting accuracy).
2. **Unchanged within tolerance:** every MP2 gradient, geomopt and frequency
   case; every HF SOSCF and stability case; and the RI-JK gradient, SOSCF and
   stability cases. That covers dense RMP2/UMP2 at 1e-8 against PySCF, so the
   new solver must hold that.
3. **The RI-MP2 gates** from `docs/RI_MP2_GRADIENT_RESPONSE_SCOPE.md` F2 at
   3e-7, which fail today.

## Order

1. Make `two_electron_fock{,_uhf}` honor `scf_ri_jk`.
2. Add the RHF operator; pass gate 1 against both explicit builders.
3. Switch `solve_rhf_cphf` and both SOSCF paths to the operator.
4. Delete the explicit builders, `fitted_response` and the `veff`
   branches; run gate 2.
5. RI-MP2 gates (that doc's F2), then spherical RI-MP2 (its F3). Its F1 is
   done by steps 1–4, and its F4 does not arise.

## Not in this scope

- **The KS orbital Hessians** (`src/dft/ks_orbital_hessian.cpp`) have the
  same shape with an XC term: RKS is diag + 2J + 2f_xc − cK, and HF is that
  with no XC and c = 1. They need J and K separately with a coefficient,
  which the fused G = J − ½K does not give, and they already take the fitted
  route under RI-JK (step 14). Sharing the J/K part would be a later step.
- **The SCF loop's own Fock build**, which carries incremental-Fock state.
