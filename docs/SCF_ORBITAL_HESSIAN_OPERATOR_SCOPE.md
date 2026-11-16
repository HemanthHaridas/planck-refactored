# Can the RHF orbital Hessian be matrix-free, so RHF and UHF share one path?

Scope, opened 2026-11-16. It expires when the work lands; rewrite it then as
the answer. Companion to `docs/RI_MP2_GRADIENT_RESPONSE_SCOPE.md`: doing
this first makes that fix fall out of it (see "Order").

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
