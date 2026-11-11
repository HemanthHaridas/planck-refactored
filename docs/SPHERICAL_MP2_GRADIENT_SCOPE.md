# What does the RMP2/UMP2 gradient need to run in a spherical basis?

Scope, opened 2026-11-10; audited 2026-11-11 (below). Nothing here is
built. It expires when the work lands; rewrite it then as the answer.
Status lives in `vault/Status/Open Work.md` (priority 4).

## Why it matters

Every standard correlated basis (cc-pVXZ, def2) is spherical. MP2 energies
run spherical; MP2 gradients, geometry optimization and the workflows on top
of them are refused (`src/hf_driver.cpp`, the `reject("MP2 analytic
gradient")` in the spherical gate; boundary cases
`water_rmp2_spherical_{gradient,geomopt}_rejected`).

## The pattern that already works

SCF gradients lift every AO matrix to Cartesian once at the skin,
`M_cart = Cᵀ M_sph C` (`BasisFunctions::to_cartesian`), and reuse the
Cartesian derivative engine unchanged (`spherical_gradient_lift_pattern`
memory; `src/gradient/gradient.cpp`). Rule learned on ROHF: lift a finished
AO quantity once; never build it from separately lifted factors, since
C·Cᵀ ≠ I.

## Audit (2026-11-11)

Read in full: `src/post_hf/mp2_gradient.cpp` (939 lines),
`rhf_response.cpp` and `uhf_response.cpp`, plus the kernels and entries they
call (`gradient.cpp` `compute_{r,u}mp2_gradient`, `one_electron_gradient`,
`two_electron_gradient.h`). Line numbers are for `mp2_gradient.cpp` at
`df86e098`. Throughout, T is the n_sph × n_cart transform held on the basis.

### Design: the Lagrangian in the working basis, lifted at the skin

The orbital half is built in the working basis, and it is consistent there
as it stands. Everything the derivative kernels contract is lifted to
Cartesian once.

**Already consistent in the working basis (no change needed):**
- **MO coefficients and densities.** `RMP2Result::mo_coeff` and the UMP2
  `mo_coeff_{alpha,beta}` have working-basis rows, and so do the HF densities
  (`_info._scf.{alpha,beta}.density`).
- **The AO ERI tensor.** `ensure_eri` returns the spherical tensor in
  spherical mode (S0.2), so the Lagrangian contraction `imat(q,v) = Σ (pq|rs)
  dm2[p,v,r,s]` (L308–316, L779–791) is consistent once `nao` is the working
  size.
- **The overlap.** `imat_mo = Cᵀ imat S C` (L321, L795–796) uses `_overlap`,
  which is the working-basis overlap.
- **The RHF response.** `solve_rhf_cphf` / `build_rhf_cphf_matrix` size with
  `working_nbasis()` and transform the working ERI tensor.
- **The UHF response.** `uhf_response.cpp` is already correct. Its
  `nbasis()` (L87) is the Cartesian size of the density it has just lifted
  with `to_cartesian` (L80–81), and it projects the Fock back with
  `to_working` (L91–92). The old note here that it "still passes `nbasis()`"
  was a misreading.

**Rejected alternative:** running everything in Cartesian with lifted
coefficients `C_cart = Tᵀ C`. That needs a Cartesian ERI tensor (which
`ensure_eri` does not give in spherical mode) and a Cartesian overlap, while
the CPHF solve stays in the working basis. One Lagrangian would then span two
bases.

### Defects, by class

**1. Basis-size sites: working objects sized with the Cartesian count.**
- **L248, L421 — the frozen-orbital check** (fixed 2026-11-11; see below).
  It compared
  `active_mo.size()` with `nbasis()`. `active_mo` always lists every active
  MO, `n_mo_full` (`mp2_internal.cpp:100–105`), which in spherical mode is
  the spherical count. So **every** spherical RMP2 gradient would be refused
  as "frozen-orbital gradients are not implemented" before anything else ran.
  Compare with `result.mo_coeff.cols()` instead.
- **L251, L427, L681 — `nao`.** It sizes `dm2buf_full`, `Da`/`Db`,
  `imat_ao`/`Imat_σ` and every loop over them. `pair_half_density` takes its
  size from the coefficient rows (the working count), while `add_symmetrized`
  and `idx_dm2` index with `nao` (the Cartesian count). That reads past the
  buffer. Use `working_nbasis()`.

**2. Fock builds of a trial density.**
- **L140–148 and L165–174.** `build_veff_from_density` and
  `build_veff_from_spin_densities` hand a working-basis density to
  `_compute_2e_fock{,_uhf}` with the Cartesian `nbasis()`. Fix: lift the
  density with `to_cartesian`, build, and bring the result back with
  `to_working`.
- **Make it one helper.** That lift-build-project sequence already exists in
  three copies: `scf.cpp` `direct_fock_rhf` / `direct_fock_uhf` (L71–118,
  with incremental-build state) and `uhf_response.cpp` L78–93. Add one public
  working-basis helper, e.g. `SCF::two_electron_fock{,_uhf}(calc, pairs, P)`.
  Call it from `uhf_response` and from both veff builders; `scf.cpp` can wrap
  it. Do not write a fourth copy.

**3. Cartesian consumers: lift once at the skin.**

The lifts are exact because T does not depend on geometry (S0.5): for any
working-basis M, tr(M ∂X_sph) = tr(Tᵀ M T ∂X_cart).
- **`one_electron_gradient`** (L560, L881). Lift `dm1_total_ao` / `dm1tot`.
- **Hand-rolled overlap-derivative loops** (L578–599 RMP2, L888–902 UMP2).
  They walk the Cartesian `_basis_functions` with `_compute_1e_deriv_A`, so
  lift each finished matrix once: `imat_ao`, `zeta_ao` and `vhf_s1occ`
  (RMP2); `im1_ao`, `zeta_total` and `vhf_s1occ` (UMP2). They are finished AO
  quantities, so the ROHF rule ("never build from separately lifted factors")
  is met.
  - Optional simplification: `one_electron_gradient(calc, pairs, D, W)`
    already computes −Σ ∂S·W. Folding these loops into its `W` argument would
    delete about 40 lines and N² per-AO-pair `ShellPair` constructions, but
    changes the summation order, so it is not bitwise.
- **The separable part of Γ** in `two_electron_gradient` (L611–615 RMP2,
  L910–915 UMP2). It is an outer product of two AO matrices: `hf_dm1`,
  `dm1p`, `hf_dm1a/b`, `dm1pa/b`. Lift each factor. That is exact here
  because each AO index belongs to one factor. The ROHF pitfall was a matrix
  product contracting a shared index (`Pa·Fa·Pa`), which does not occur.
- **The pair density in Γ** (L618–621 `dm2buf_full`, L916–921 `Da + Db`).
  Γ is indexed over the Cartesian basis (`two_electron_gradient` sizes with
  `basis.nbasis()`). Build a Cartesian copy by re-running
  `pair_half_density` / `back_transform_half_density` with lifted
  coefficients `C_cart = Tᵀ C`. Both are linear in each coefficient matrix,
  so this equals the 4-index lift exactly, at the cost of the working build.
  - Keep the working copy for the Lagrangian contraction with the spherical
    ERI.
  - In Cartesian mode reuse the one buffer, so Cartesian output stays
    bitwise identical.
  - Spherical mode holds one extra n_cart⁴ buffer at peak.

**4. RI branches** (L138–139, L162–164, L301–305, L484–502, L738–772). These
are out of this scope; they belong to the combined RI item in Open Work.
With `mp2_use_ri`, spherical MP2 gradients must stay refused by name.

**5. Callers outside the HF driver.** `build_rmp2_lagrangian` and
`build_rmp2_energy_weighted_density` are shared with the RKS double-hybrid
gradient through `ks_veff`. Spherical DH gradients are refused separately
(`water_rks_b2plyp_gradient_spherical_rejected`), and in Cartesian mode
every change above is the identity. `geomopt.cpp` L125/L139 call the same
`compute_{r,u}mp2_gradient`, and its inner loop already rebuilds spherical
state (`rebuild_basis_dependent_state`).

**6. Workflows.** Correlated frequencies and imaginary-mode following are
refused in **every** basis (`hf_driver.cpp`, "Correlated frequencies are not
implemented"), so lifting the spherical gate enables `gradient` and `geomopt`
only. Step 4 below is corrected accordingly.

**Why frequencies are refused when geomopt is not:** it is wiring, not
physics.
- The geomopt step (`geomopt.cpp` L115–142) runs SCF, then RMP2/UMP2, then
  `compute_{r,u}mp2_gradient`.
- The semi-numerical Hessian (`freq/hessian.cpp` L100–120) picks the
  gradient at each displaced geometry by SCF type alone (RHF/UHF/ROHF). It
  never looks at `_correlation`.
- So MP2 + `frequency` would print MP2 energies with HF frequencies. The
  driver refuses it for that reason, "until the MP2 gradient is wired into
  the Hessian dispatch".

Step 5 adds that wiring.

### Found on the way: the frozen-core UMP2 gradient is silently wrong

This is separate from spherical support. `build_rmp2_*` refuses frozen
orbitals; `build_ump2_gradient_intermediates` has no such check. With
`mp2_frozen 1` on `water_radical_cation_uhf_ump2_gradient_sto3g` it runs and
prints a gradient with no warning. Measured against a central FD of the
frozen-core UMP2 energy (O z, h = 0.001 Å):

| | analytic | FD |
|---|---|---|
| all-electron | −0.12006018 | −0.12006025 |
| `mp2_frozen 1` | −0.16473947 | −0.12010349 |

The frozen-core analytic gradient is off by 4.5e-2 Ha/Bohr.

**Fixed 2026-11-11.** One helper, `has_frozen_orbitals`, now compares the
result's active-orbital count with the SCF's MO count, and all three builders
use it. Comparing with the MO count is also right in a spherical basis. The
RMP2 check used to compare with `nbasis()` and would have refused every
spherical RMP2 gradient; that is class 1 above, now fixed. Gates:
- `water_rmp2_frozen_gradient_rejected`
- `water_radical_cation_ump2_frozen_gradient_rejected`

### Why frozen-core MP2 gradients are refused

The refusal is not a policy choice. The builders lack the terms a frozen-core
gradient needs.

**1. The frozen-core energy depends on how the orbitals are split.** With
`mp2_frozen`, the correlation energy is not invariant under rotations that
mix frozen and active occupied orbitals, or frozen and active virtuals. The
split is fixed by the canonical orbital energies. So the relaxed density
needs two extra blocks that the all-electron gradient never has (PySCF
`grad/mp2.py` L124–132):
- frozen–active occupied: D_FA = I_FA / (ε_F − ε_A);
- frozen–active virtual: D_VF,VA = I_VF,VA / (ε_VF − ε_VA).

Here I is the orbital Lagrangian (`imat_mo` in Planck). Nothing in
`mp2_gradient.cpp` builds these blocks.

**2. The builders only see the active orbitals.** Both `RMP2Result::mo_coeff`
and `UMP2Result::mo_coeff_{alpha,beta}` hold the active columns only
(`mp2_rmp2.cpp` L107, `mp2_ump2.cpp` L113, from `make_eris_*`). So every
quantity the gradient builds from them spans the active space alone:
- `imat_mo`;
- the Z-vector right-hand side and solve over occupied × virtual;
- `W_ref = Σ ε_i C_i C_iᵀ`;
- the ζ weights;
- the occupied projector in `vhf_s1occ`.

In PySCF these run over the full occupied and virtual space (`nocc`
recounted to include the frozen orbitals, L115–121). The HF density
(`hf_dm1a/b` in UMP2) does include the frozen orbitals, which is why the
UMP2 result was off rather than obviously broken. The 4.5e-2 Ha/Bohr error
is mostly the frozen core's missing share of `W_ref` and of the
energy-weighted overlap term.

**To implement it:**
- carry the full MO coefficients and energies, plus the active mask, into
  the builders;
- build `imat_mo` over all MOs;
- add the D_FA and D_VF,VA blocks before the Z-vector right-hand side;
- run the Z-vector, ζ and W over the full space.

That is about PySCF's 10 lines, applied to the RMP2 and UMP2 builders and to
the double-hybrid Lagrangian that shares them. Gate: frozen-core RMP2 and
UMP2 gradients against PySCF `mp2.Gradients` with `frozen=1`, plus an energy
FD. This is what "Not in this scope" below refers to.

## Steps

0. ~~Refuse frozen-core UMP2 gradients.~~ Done 2026-11-11 (above); the
   frozen check in class 1 was fixed with it.
1. **Refactor, Cartesian bitwise — done 2026-11-11.** What landed:
   - **`SCF::two_electron_fock{,_uhf}`** (`src/scf/scf.{h,cpp}`) is the one
     working-basis Fock helper: lift the density, build, project back.
     `uhf_response.cpp` and both MP2 veff builders call it; the veff
     builders now return `std::expected`.
   - **`BasisFunctions::mos_to_cartesian`** (`src/basis/spherical.{h,cpp}`)
     is the one coefficient lift, Tᵀ C, and an identity in Cartesian mode.
     It replaces the two earlier copies, in `ks_orbital_hessian.cpp` and the
     TDDFT setup in `src/dft/driver.cpp`.
   - **In `mp2_gradient.cpp`:**
     - `nao` is the working size and `nao_cart` the kernel size;
     - `lift_to_cartesian` lifts every 1e, overlap and separable-Γ input
       once;
     - the pair densities come from two builders, `rmp2_pair_density` and
       `ump2_pair_densities`. They run once in the working basis for the
       Lagrangian, and again with Tᵀ C for Γ in spherical mode only.
       Cartesian mode reuses the one buffer.
   - **Gate unchanged.** Spherical MP2 gradients stay refused until step 2.

   Verification: 68 regression cases pass, 0 fail. They cover every MP2 and
   double-hybrid case, and every TDDFT, stability and SOSCF case, Cartesian
   and spherical, since those cover the changed code. A partial bitwise
   output comparison against the `328ec560` binaries showed differences
   only in timing columns and timestamps among the lines it printed; it was
   not rerun with a corrected filter. Add the skin
   lifts and the Cartesian pair-density rebuild (class 3). Every lift is the
   identity in Cartesian mode, so all MP2/UMP2 gradient cases must be bitwise
   unchanged:
   - `water_rmp2_gradient_*`
   - `water_radical_cation_ump2_gradient_smoke`
   - `water_rmp2_gradient_engine_os_vs_hgp`
   - the RKS double-hybrid gradient cases, which share the Lagrangian.
2. **RMP2 spherical gate.** Lift the gate for RMP2 only, and keep it for
   `mp2_use_ri`. Gate against PySCF `cart=False` analytic `mp2.Gradients`
   on water/6-31G* and C1 H2O2/6-31G*, plus a fourth-order energy FD on one
   of them. Set PySCF's `conv_tol_grad` (memory
   `pyscf_conv_tol_grad`).
3. **UMP2 spherical gate.** Same for UMP2: water⁺ doublet and one C1
   radical.
4. **Convert the rejection cases.** Turn the two cases into positive cases:
   - `water_rmp2_spherical_gradient_rejected` becomes a gradient case;
   - `water_rmp2_spherical_geomopt_rejected` becomes a geomopt case (PySCF
     `geometric_solver` E_opt, `geomopt_grad_tol` as in the PCM gates).

   Frequency stays refused for MP2 until step 5.
5. **MP2 frequencies (Cartesian and spherical).** Two loops each compute
   "SCF and gradient at this geometry", and only the geomopt one handles
   MP2. The steps:
   - **One shared routine.** Factor the per-geometry SCF → MP2 → gradient
     step out of `geomopt.cpp` into one routine that both loops call, and use
     it in `hessian.cpp`. Do not add a second MP2 branch to the Hessian.
   - **Lift the refusal.** Remove the correlated-frequency refusal in
     `hf_driver.cpp`. That enables `frequency`, `geomoptfreq` and
     `imaginaryfollow`.
   - **Gate.** RMP2 and UMP2 frequencies against PySCF: a finite-difference
     Hessian of PySCF's analytic `mp2.Gradients`, as the ROHF frequency gate
     does. Use one Cartesian and one spherical case.
   - **Convert the rejection cases.** `water_ri_rmp2_freq_rejected` and
     `water_radical_cation_ump2_freq_rejected` become positive cases. Keep
     the RI one refused if RI-MP2 frequencies are not validated; that case
     pins the guard order.

   Cost is 6N MP2 gradients (each with its own SCF and Z-vector solve) for N
   atoms, so this is for small molecules. In spherical mode this needs steps
   1–3 first; the Hessian loop already rebuilds the spherical state at every
   displaced geometry (`rebuild_basis_dependent_state`).
RI-MP2 gradients in spherical are **not in this scope**. The RI 3-centre
derivative tensors are Cartesian-only (`compute_3c_eri_deriv` refuses a
spherical basis). The work moves to the combined RI item in Open Work
(priority 9). Until then, step 2 keeps `mp2_use_ri` + spherical + gradient
refused by name at the driver gate.

## Not in this scope

Frozen-core MP2 gradients. They are refused in every basis, for RMP2 and
UMP2. What they need is in "Why frozen-core MP2 gradients are refused" above.
It is the next most-asked MP2 feature and gets its own scope after this.
