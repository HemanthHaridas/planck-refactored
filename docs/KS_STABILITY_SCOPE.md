# Can `planck-dft` detect, and leave, an unstable Kohn-Sham solution?

Research scope, opened 2026-11-04. UKS internal stability and following are
implemented in the working tree (see Handoff); RKS stability is scoped below.
Status lives in `vault/Status/Open Work.md`.

## Why it matters

The default `hcore` guess can converge an open-shell SCF onto an excited,
symmetry-trapped saddle with nothing reported. Water⁺/cc-pVTZ: the core
Hamiltonian orbitals order 1a₁ 2a₁ 1b₂ 1b₁ 3a₁, so the guess already holds the
²A₁ beta hole, and UKS HSE06 converges to ²A₁ at −75.8470063, 0.066 Eh above
the ²B₁ ground state (−75.9126471 from `guess sad`; PySCF agrees with both to
the grid floor, 7e-6). ²A₁ is a saddle, but its descent direction mixes a₁
and b₁, which breaks C₂ᵥ, so a symmetric density never leaves it; PySCF's own
SCF started from the ²A₁ density stays there too. UHF does the same from
`hcore` and HF already escapes it (`stability_check` + `stability_follow`:
λ = −0.069, re-converges to the ²B₁ energy to every digit). Kohn-Sham had no
stability analysis, so a KS run could not notice.

## Design

The stability Hessian is the operator second-order SCF already applies, so no
new response code is needed:

- **UKS → UKS** (landed): the spin-conserving A + B is the UKS SOSCF operator,
  now `DFT::Driver::build_uks_orbital_hessian_op` (`src/dft/ks_orbital_hessian.cpp`):
  orbital-energy differences, J from the total trial density, the polarized
  analytic XC kernel, and −(c_fr K + c_sr K_SR) per spin, with
  dP_σ = C_v x_σ C_oᵀ + transpose.
- **RKS → RKS** (scoped): the singlet A^S + B^S is the RKS SOSCF operator
  `build_ks_orbital_hessian_op` with `kernel_scale = 2`. With dP = d₁ + d₁ᵀ,
  2·J[dP] gives 4(ai|bj) and 2·(−½K[dP]) gives −(ab|ij) − (aj|ib), exactly
  `analyze_rhf_stability`'s internal-real matrix plus the XC kernel.
- **RKS → UKS** (scoped): the triplet A^T + B^T is the UKS operator at the
  closed-shell reference (C_α = C_β = C, P_α = P_β = P/2) applied to the
  spin-antisymmetric rotation [κ, −κ]; the output is [y, −y] with
  y = (A^T + B^T)κ, and the triplet XC kernel (f_αα − f_αβ) comes from the
  polarized kernel unchanged. [κ, κ] gives the singlet, a cross-check of the
  RKS operator.

Eigenvalues come from the matrix-free `davidson_lowest_eigenpairs` with the
orbital-energy differences as preconditioner and 4 roots (a symmetry-breaking
mode is reached only from start vectors that touch it). The follow reuses
`SCF::rotate_along_internal_mode` (split out of the HF follow) and re-runs the
KS SCF.

**Eigenvalue convention.** Planck reports eigenvalues of A ± B, as the HF
analysis does, and they equal the eigenvalues of PySCF's orbital-Hessian
operator directly: UKS `gen_g_hop_uhf` gives −0.063809 for water⁺/cc-pVTZ ²A₁
(Planck −0.063853) and +0.069241 at 6-31G* (Planck +0.069275), differences at
the grid floor. What PySCF *prints* differs by a fixed factor per channel:
`uhf_internal` ×2 (−0.1276), `rhf_internal` ×4 (2.4717 vs 0.6179, H₂ 2.5 Å);
`rhf_external` ×1 (−0.5109 = A^T + B^T).

## Handoff

State on `feat/def2-ecp` as of 2026-11-05 (UKS check, follow and cost landed;
R0–R3 open):

- `src/dft/ks_orbital_hessian.{h,cpp}` — `UksOrbitalHessianInputs` and
  `build_uks_orbital_hessian_op`, lifted from the inline UKS SOSCF lambda. It
  prepares the polarized XC kernel once and returns `expected`.
- `src/dft/analytic_hessian.{h,cpp}` — `UKSXCKernel` (`prepare_uks_xc_kernel`
  / `apply`), the polarized twin of `RKSXCKernel`: ground densities and Libxc
  second derivatives once, GEMM contractions over 128-point blocks per apply.
  `compute_analytic_xc_hessian_vector_product_polarized` is prepare + apply,
  so the UKS double-hybrid response shares it.
- `src/dft/driver.cpp` — UKS SOSCF and `analyze_uks_stability` use the
  builder; `davidson_lowest_eigenpairs` caches each basis vector's product
  (it recomputed all of them every iteration); `run_initial_single_point`
  checks after a converged SCF and, with `stability_follow`, rotates once,
  re-converges and re-checks. RKS (`"UKS only"`) and spherical stability are
  refused.
- `src/scf/stability.{h,cpp}` — `rotate_along_internal_mode`, the rotation
  half of `follow_instability_and_rerun`, shared with the KS follow.

Verified:
- Water⁺/cc-pVTZ UKS HSE06 from `hcore`: λ_min −6.385299e-02 [UNSTABLE]; the
  follow re-converges to −75.9126471254 (the ²B₁ state Planck reaches from
  SAD) and re-checks stable (+6.650649e-02).
- Only that combination traps: water⁺ from `hcore` with PBE or B3LYP at
  6-31G*, 6-311G, cc-pVDZ and cc-pVTZ, and HSE06 at STO-3G and 6-31G*, all
  land on the ground state (λ_min ≈ +0.06–0.07).
- `planck-dft-analytic-hessian-polarized-production` passes on the prepared
  kernel and fails a mutated cross-spin slot (1e-8 tolerance).
- 16 SOSCF/UKS/double-hybrid regression cases, core 98/98, smoke 51/53 (the
  known `rccsdt` routing cases).
- Cases: `water_cation_uks_hse06_stability_ccpvtz` (extended, 51 s; unstable,
  follow, ground-state energy), `water_cation_uks_hse06_stability_631gd`
  (core; stable, λ pinned), `water_rks_stability_rejected` (smoke).

Pending: `docs/SOSCF_DFT.md` and the README keyword list should mention the
shared builder and `stability_check` in `planck-dft`.

## Next steps, in order

**C — cost of the UKS check — DONE 2026-11-05.** Measured first, on
water⁺/cc-pVTZ `grid fine` (SCF ~8 s):

| change | Davidson products | time |
|---|---|---|
| as landed | 167 | 172 s |
| cache each basis vector's product (it recomputed every one every iteration) | 30 | 31 s |
| residual tolerance 1e-6 → 1e-4 (λ unchanged to 7 digits; eigenvalue error ~ residual²) | 23 | 24 s |
| prepared polarized XC kernel (1.03 → 0.60 s per product) | 23 | 14 s |

Per product, before the kernel change: J 0.011 s, XC 0.53 s, short-range K
0.49 s. K is what remains; a trial density gets no help from the SCF's density
screening, so it is left. λ_min and the follow result are unchanged.

**R0 — fix the HF RHF → UHF channel first (found while scoping).**
`analyze_rhf_stability` builds its "external triplet" matrix as A^T − B^T
(`TripletAminusB`, `src/scf/stability.cpp`), which is algebraically identical
to the complex-orbital test it also reports. A real RHF → UHF rotation's
Hessian is A^T + B^T = (ε_a − ε_i)δ − (ab|ij) − (aj|ib). Measured on H₂/STO-3G
at 2.5 Å: Planck reports +0.0535 [stable]; A^T + B^T is −0.5109, and PySCF's
`rhf_external` agrees (−0.5109, "RHF/RKS → UHF/UKS instability"). At 1.6 Å:
Planck +0.2157 [stable], true −0.2561. So the textbook triplet instability of
stretched H₂ is reported stable. It is the same A − B / A + B mix-up the UHF
channel had (the comment above its matrix records that fix). Fix: the triplet
channel uses −(ab|ij) − (aj|ib); keep A^S − B^S as the complex test. Gates:
stretched H₂ unstable at both distances with PySCF's value, water/STO-3G
stable, and the RHF → UHF follow reaching PySCF's broken-symmetry UHF energy.
R2 below mirrors this channel, so it must be right first.

### RKS stability

**R1 — RKS → RKS (internal, real).** Davidson on `build_ks_orbital_hessian_op`
at the converged RKS orbitals (`kernel_scale = 2`). Follow with
`rotate_along_internal_mode`, re-run RKS. Gate: PySCF `rks.stability()`
internal (with its factor) on a stable case (water, B3LYP) and on one with a
real internal instability if one can be found at small size; else a gate that
λ_min matches PySCF on stable systems.

**R2 — RKS → UKS (external, triplet).** The UKS operator at the closed-shell
reference on [κ, −κ], Davidson in the κ space (dimension n_occ·n_virt). Follow
as HF does for RHF → UHF: switch the calculator to UKS (`DataSCF(true)`),
β HOMO–LUMO mix plus ±κ rotation, re-run the UKS SCF; `prepared` is reusable
since the geometry is unchanged. Gates: stretched H₂/STO-3G and N₂ at a
stretched bond (PBE and B3LYP), PySCF `rks.stability(external=True)`; the
follow lands on PySCF's UKS broken-symmetry energy; water stays stable.
Then lift the "UKS only" refusal.

**R3 — spherical.** The operators take working-basis C; the trial densities
go to the Cartesian J/K/XC builds through `to_cartesian` and the outputs back
through `to_working`, like the KS potential (`docs/SPHERICAL_DFT_SCOPE.md`).
Until then `validate_spherical_support` refuses stability analysis.

Not in scope: spin-flip UKS → GKS (the UHF analysis only has a diagonal
approximation, and Planck has no GHF/GKS to follow into); stability inside
geometry optimization steps.
