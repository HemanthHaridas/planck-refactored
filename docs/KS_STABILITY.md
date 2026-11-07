# Can `planck-dft` detect, and leave, an unstable Kohn-Sham solution?

**Yes, for RKS and UKS, in the Cartesian and spherical bases.** `stability_check` reports the
lowest orbital-Hessian eigenvalue of each channel:

- UKS: UKS → UKS.
- RKS: RKS → RKS (singlet) and RKS → UKS (triplet).

`stability_follow` rotates along the lowest unstable mode and re-converges.
A triplet instability promotes the run to UKS. Eigenvalues match PySCF to the
grid floor (≤ 2e-5) on every case below, and the follows land on PySCF's
lower state.

Also in the spherical basis (`docs/SPHERICAL_DFT.md`: λ ≤ 1.2e-7 from PySCF
`cart=False`, follow to every digit) and with ECPs (`docs/DEF2_ECP.md`
step 6: HI⁺ UKS PBE/def2-SVP λ_min +2.3e-9, the π-hole zero mode, PySCF
−1.4e-8). Nothing is open.
The HF-side fixes are gated. Status:
`vault/Status/Open Work.md`.

| case | channel | Planck | PySCF 2.13.0 |
|---|---|---|---|
| water⁺/cc-pVTZ UKS HSE06, `hcore` | UKS → UKS | −0.063853 [UNSTABLE] | −0.063809 |
| | after follow | −75.9126471254 (²B₁) | −75.9126406581 |
| water⁺/6-31G* UKS HSE06 | UKS → UKS | +0.069275 | +0.069241 |
| H₂/STO-3G 2.5 Å RKS PBE | singlet / triplet | +0.9301766 / −0.1448062 | +0.9301607 / −0.1448208 |
| | after follow (UKS) | −0.9302662713 | −0.9303000702 |
| H₂ 2.5 Å RKS B3LYP | singlet / triplet | +0.8647756 / −0.2230623 | +0.8647606 / −0.2230807 |
| | after follow (UKS) | −0.9377524499 | −0.9377675629 |
| water/6-31G* RKS B3LYP | singlet / triplet | +0.3080848 / +0.2522772 | +0.3080762 / +0.2522700 |
| water/6-31G* RKS HSE06 | singlet / triplet | +0.3367296 / +0.2817728 | +0.3367223 / +0.2817651 |

The ≤ 3.4e-5 energy gaps on stretched H₂ are the grid. The RKS energies
differ by 2.4e-5 before any follow; water differs by 8.5e-7.

## Why it matters

The default `hcore` guess can converge an open-shell SCF onto an excited,
symmetry-trapped saddle and report nothing. Example, water⁺/cc-pVTZ:

- The core-Hamiltonian orbitals order 1a₁ 2a₁ 1b₂ 1b₁ 3a₁, so the guess
  already holds the ²A₁ beta hole.
- UKS HSE06 converges to ²A₁ at −75.8470063, 0.066 Eh above the ²B₁ ground
  state.
- ²A₁ is a saddle, but its descent direction mixes a₁ and b₁, which breaks
  C₂ᵥ. A symmetric density therefore never leaves it, and PySCF's SCF started
  from that density stays there too.

UHF behaves the same and could already escape through its stability follow.
Kohn-Sham had no stability analysis.

## How it works

The stability Hessian is the operator second-order SCF already applies, so
there is no new response code.

- **UKS → UKS.** The spin-conserving A + B is the UKS SOSCF operator,
  `build_uks_orbital_hessian_op` (`src/dft/ks_orbital_hessian.cpp`). Its terms
  are:
  - orbital-energy differences;
  - J from the total trial density;
  - the prepared polarized XC kernel (`UKSXCKernel`);
  - −(c_fr K + c_sr K_SR) per spin, with dP_σ = C_v x_σ C_oᵀ + transpose.
- **Both RKS channels, one operator.** `analyze_rks_stability` builds that same
  UKS operator at the closed-shell reference (C_α = C_β = C,
  P_α = P_β = P/2). It needs polarized libxc handles, created from the
  resolved IDs as RKS TDDFT does.
  - The α block of its action on [κ, κ] is the singlet A^S + B^S:
    4(ai|bj) − (ab|ij) − (aj|ib) + XC, which is `analyze_rhf_stability`'s
    internal-real matrix.
  - On [κ, −κ] it is the triplet A^T + B^T, with the triplet kernel
    f_αα − f_αβ coming from the polarized kernel unchanged.
  - No RKS operator is involved, so the two channels cannot disagree about
    the kernel.
- **Eigenvalues.** `lowest_hessian_root` runs a matrix-free Davidson with
  4 roots, the orbital-energy differences as preconditioner, and a 1e-4
  residual. The eigenvalue error goes as residual², so λ is good to 7 digits.
  Four roots are needed because a symmetry-breaking mode is reached only from
  start vectors that touch it.
- **Follow** (`run_initial_single_point`). The most negative unstable channel
  is followed once, then the SCF re-converges and is re-checked.
  - Internal channels use `SCF::rotate_along_internal_mode`.
  - The RKS → UKS triplet uses `SCF::promote_rhf_to_uhf_along_triplet_mode`:
    β HOMO–LUMO mix plus ±κ, with the calculator switched to UHF.
  - Both functions are split out of the HF follow, so HF and KS share them.
  - The KS SCF reads the SCF type and seeded densities at call time, so the
    switch needs no plumbing.
  - After a promotion the run is UKS for every later step. `functionals` is
    therefore re-initialized polarized, and `validate_workflow_support` re-runs:
    a workflow UKS does not support (e.g. UKS double-hybrid geomopt) is refused
    rather than run unchecked.

### Eigenvalue convention

Planck reports eigenvalues of A + B, as the HF analysis does. Against PySCF:

| channel | PySCF operator | factor vs Planck | what PySCF prints |
|---|---|---|---|
| UKS → UKS | `newton_ah.gen_g_hop_uhf` | ×1 | `uhf_internal` ×2 |
| RKS/RHF → RKS/RHF | `newton_ah.gen_g_hop_rhf` | **×2** | `rhf_internal` ×4 |
| RKS/RHF → UKS/UHF | `stability._gen_hop_rhf_external` (2nd) | ×1 | `rhf_external` ×1 |

The ×2 for the singlet was pinned on HF: Planck water/STO-3G internal-real
gives 0.5607608, PySCF `gen_g_hop_rhf` gives 1.1215215. An earlier draft of
this doc assumed ×4, which is what `rhf_internal` *prints*.

## Cost

Measured on water⁺/cc-pVTZ UKS, `grid fine`, where the SCF itself takes
about 8 s:

| change | Davidson products | time |
|---|---|---|
| first version | 167 | 172 s |
| cache each basis vector's product (it recomputed every one each iteration) | 30 | 31 s |
| residual 1e-6 → 1e-4 (λ unchanged to 7 digits) | 23 | 24 s |
| prepared polarized XC kernel (1.03 → 0.60 s per product) | 23 | 14 s |

Before the kernel change, each product cost J 0.011 s, XC 0.53 s and
short-range K 0.49 s. K is what remains. A trial density gets no help from
the SCF's density screening, so it was left alone.

## Defects found on the way (all fixed)

**HF RHF → UHF channel computed the wrong matrix.** `analyze_rhf_stability`
built its triplet matrix as A^T − B^T. That is algebraically the
complex-orbital test it also reports, so the triplet row repeated the complex
row. It now uses A^T + B^T = (ε_a − ε_i)δ − (ab|ij) − (aj|ib)
(`TripletAplusB`).

| case | before | after | PySCF `rhf_external` |
|---|---|---|---|
| H₂/STO-3G 2.5 Å | +0.0535 [stable] | −0.5109055 [UNSTABLE] | −0.51090553 |
| H₂/STO-3G 1.6 Å | +0.2157 [stable] | −0.2560807 [UNSTABLE] | — |
| water/STO-3G | +0.4798247 (the complex value) | +0.3988887 | +0.39888868 |

The follow now reaches PySCF's broken-symmetry UHF to every digit:
−0.9338672048 at 2.5 Å and −0.9502279135 at 1.6 Å. The UHF channel had the
same A − B / A + B mix-up.

**A sweep of the suite.** All 38 open-shell (`scf_type uhf`) regression inputs
were re-run with `stability_check`. Results:

- **30 stable.** B/STO-3G UHF gives λ = −3.9e-17: the zero mode of the
  degenerate 2p, correctly reported stable.
- **One unintended saddle: `h2o2_cation_uhf_tight_tol_sto3g`.** λ = −0.07384714,
  the same as PySCF. Its pinned energy is PySCF's own hcore-guess SCF, and the
  follow lowers it 0.051 Eh. The case gates DIIS termination, not the ground
  state, so it was kept.
- **Spherical HF stability was broken.** The ERI transform was sized with the
  Cartesian `_shells.nbasis()`, while `ensure_eri` returns the spherical
  tensor:
  - Li/6-31G* failed to diagonalize;
  - water⁺/6-31G* reported λ = −15 [UNSTABLE];
  - HI⁺/def2-SVP segfaulted.

  Now `working_nbasis()`. Water⁺ spherical gives +0.08441901 against PySCF
  +0.08441900, and the conventional and direct runs are identical.
- **Eleven electron counts ignored ECP cores.** They summed `atomic_numbers`
  (Σ Z), which is also wrong with ghost atoms. Locations: stability,
  `rhf_response.cpp` (2), `mp2_internal.cpp` (2), CC `common.cpp`, FCI,
  CASSCF, the KS SCF scaffold, and RKS/UKS TDDFT.
  - On HI⁺ the occupied/virtual split was wrong: λ came out −0.1865 against
    PySCF +3.9e-10, and the follow *raised* the energy 0.14 Eh.
  - **HI/def2-SVP RMP2 returned E_corr −0.0059849856 against PySCF
    −0.1434184970, with no warning.**
  - All sites now use `total_nuclear_charge() − charge`. HI RMP2 gives
    −0.1434184969, HI⁺ λ −7.3e-12.
  - The ECP audit missed them because it searched for `nuclear_charge`,
    not for loops over `atomic_numbers`.
- **A spin with no occupied-virtual rotations** (H/STO-3G UKS) used to fail the
  run; it is now reported as an empty space.

## Gates

| case | suite | asserts |
|---|---|---|
| `water_cation_uks_hse06_stability_ccpvtz` | extended (51 s) | UKS unstable, follow to ²B₁ energy, re-check stable |
| `water_cation_uks_hse06_stability_631gd` | core | UKS stable, λ pinned |
| `water_rks_hse06_stability_631gd` | smoke, core | both RKS channels stable, λ pinned at 1e-6 |
| `h2_rks_{pbe,b3lyp}_triplet_follow_sto3g` | core (PBE also smoke) | triplet unstable, both λ, follow energy |
| `h2_rhf_triplet_{instability,follow}_sto3g`, `water_rhf_stability_stable` | smoke/core | the HF triplet fix; all fail on the old algebra |

- Runner metrics: `stability_rks_internal`, `stability_rks_triplet`.
- Building the RKS triplet on [κ, κ] fails all three RKS gates.
- `planck-dft-analytic-hessian-polarized-production` fails a mutated
  cross-spin kernel slot.

## What else landed

Spherical KS stability and SOSCF (`SPHERICAL_DFT.md`
step 4: the builders lift the MO coefficients, six gates), the spherical HF gates, the `rhf_response.cpp` site (it was
live through RHF SOSCF; `SPHERICAL_DFT.md` S0.6), and the ECP gates
`hi_cation_uhf_def2svp_ecp_stability` and `hi_rmp2_def2svp_ecp`. HI CCSD
(8.5e-8 off PySCF) stays unpinned in `DEF2_ECP.md`.

`SCF::analyze_uhf_stability` with an empty spin (checked 2026-11-06): an
empty β occupied space works (H/6-31G UHF reports both channels stable).
A spin with no virtuals logs `[WRN] Skipped` and the run continues. That
happens only when every orbital is occupied (H/STO-3G), where there is
nothing to rotate, so it is left as is.

Not in scope:
- spin-flip UKS → GKS (the UHF analysis has only a diagonal approximation,
  and Planck has no GHF/GKS to follow into);
- stability inside geometry-optimization steps.
