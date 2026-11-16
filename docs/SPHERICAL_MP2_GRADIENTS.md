# What does the RMP2/UMP2 gradient need to run in a spherical basis?

Answer (landed 2026-11-11 and 2026-11-12). The MP2 Lagrangian stays in the
working basis. Every quantity the Cartesian derivative kernels contract is
lifted to Cartesian once, at the skin. The one geometry routine that runs
SCF → MP2 → gradient serves geometry optimization and the Hessian. Spherical
and Cartesian RMP2/UMP2 now run `gradient`, `geomopt`, `frequency`,
`geomoptfreq` and `imaginaryfollow`, all gated against PySCF. Refused by
name: frozen-core MP2 gradients (every basis) and spherical RI-MP2
gradients. RI-MP2 frequencies were added on 2026-11-16
(`docs/RI_JK_ROUTINE.md` step 8). Status lives in `vault/Status/`.

## The design: Lagrangian in the working basis, lifted at the skin

This is the pattern the SCF gradients already used: `M_cart = Tᵀ M_sph T`
(`BasisFunctions::to_cartesian`), then the unchanged Cartesian derivative
engine. T is the n_sph × n_cart transform held on the basis. The lift is
exact because T does not depend on geometry: for any working-basis M,
tr(M ∂X_sph) = tr(Tᵀ M T ∂X_cart).

**Already consistent in the working basis (audited, unchanged):**
- the MO coefficients (`RMP2Result::mo_coeff`, UMP2 `mo_coeff_{alpha,beta}`)
  and HF densities;
- the AO ERI tensor (`ensure_eri` returns the spherical tensor in spherical
  mode), so the Lagrangian contraction `imat(q,v) = Σ (pq|rs) dm2[p,v,r,s]`
  is consistent once `nao` is the working size;
- the overlap in `imat_mo = Cᵀ imat S C`;
- the RHF and UHF responses (`rhf_response.cpp`, `uhf_response.cpp`).

**Lifted once (`mp2_gradient.cpp`):**
- `nao` is the working size and `nao_cart` the kernel size.
- `lift_to_cartesian` lifts the 1e density, the overlap-term matrices
  (`imat_ao`, `zeta_ao`, `vhf_s1occ`; UMP2 `im1_ao`, `zeta_total`) and the
  separable-Γ factors (`hf_dm1`, `dm1p`; UMP2 per spin). Each is a finished
  AO quantity, or an outer-product factor whose AO index belongs to one
  factor. The ROHF pitfall, a product contracting a shared index
  (`lift(Pa·Fa·Pa) ≠ lift(Pa)·lift(Fa)·lift(Pa)` since T·Tᵀ ≠ I), does not
  occur.
- The pair density in Γ is rebuilt with lifted coefficients Tᵀ C
  (`rmp2_pair_density`, `ump2_pair_densities`). Both builders are linear in
  each coefficient matrix, so this equals the 4-index lift exactly. The
  working copy stays for the Lagrangian; Cartesian mode reuses one buffer,
  and spherical mode holds one extra n_cart⁴ buffer at peak.

**Shared helpers it introduced:**
- `SCF::two_electron_fock{,_uhf}` (`src/scf/scf.{h,cpp}`): lift the density,
  build, project back. It replaced three copies of that sequence;
  `uhf_response.cpp` and both MP2 veff builders call it.
- `BasisFunctions::mos_to_cartesian` (`src/basis/spherical.{h,cpp}`): the one
  coefficient lift, an identity in Cartesian mode. It replaced copies in
  `ks_orbital_hessian.cpp` and the TDDFT setup.
- `Opt::run_scf_gradient(calc, warm_start)` (`src/opt/geomopt.cpp`): the one
  per-geometry SCF (+ MP2) + gradient step. Geometry optimization
  warm-starts it from the previous density. The Hessian
  (`src/freq/hessian.cpp`) runs it cold, so the +h and −h SCFs start from
  the same guess.

**Ruled out:** running everything in Cartesian with `C_cart = Tᵀ C`. That
needs a Cartesian ERI tensor (`ensure_eri` gives the spherical one) and a
Cartesian overlap, while the CPHF solve stays in the working basis, so one
Lagrangian would span two bases.

## The defects the audit found

1. **Working objects sized with the Cartesian count.** `nao` sized the pair
   density buffers and the Lagrangian; `pair_half_density` took the working
   count from the coefficient rows while `idx_dm2` indexed with the
   Cartesian one, reading past the buffer.
2. **Fock builds of a trial density** handed a working-basis density to the
   Cartesian `_compute_2e_fock{,_uhf}`. Fixed by the shared helper above.
3. **Cartesian consumers** (`one_electron_gradient`, the hand-rolled
   overlap-derivative loops, `two_electron_gradient`'s Γ) received working
   matrices. Fixed by the skin lifts.
4. **The frozen-orbital check** compared the active-MO count with `nbasis()`,
   so every spherical RMP2 gradient would have been refused before anything
   ran. Now `has_frozen_orbitals` compares with the SCF MO count.
5. **Frequencies were refused in every basis**, and that was wiring, not
   physics. The geomopt step ran SCF → MP2 → `compute_{r,u}mp2_gradient`,
   while the Hessian picked its gradient by SCF type alone and never read
   `_correlation`. MP2 + `frequency` would have printed MP2 energies with HF
   frequencies. Both loops now call `Opt::run_scf_gradient`.

## Found on the way: the frozen-core UMP2 gradient was silently wrong

`build_rmp2_*` refused frozen orbitals; `build_ump2_gradient_intermediates`
did not. With `mp2_frozen 1` on the water cation (STO-3G), against a central
FD of the frozen-core UMP2 energy (O z, h = 0.001 Å):

| | analytic | FD |
|---|---|---|
| all-electron | −0.12006018 | −0.12006025 |
| `mp2_frozen 1` | −0.16473947 | −0.12010349 |

An error of 4.5e-2 Ha/Bohr, printed with no warning. All three builders now
call `has_frozen_orbitals` and refuse. Gates:
`water_rmp2_frozen_gradient_rejected`,
`water_radical_cation_ump2_frozen_gradient_rejected`.

### What frozen-core MP2 gradients would need

The refusal is not a policy choice; the builders lack the terms.

1. **The frozen-core energy depends on how the orbitals are split.** It is
   not invariant under frozen–active rotations, so the relaxed density needs
   two blocks the all-electron gradient never has (PySCF `grad/mp2.py`
   L124–132): D_FA = I_FA / (ε_F − ε_A) for occupied, and
   D_VF,VA = I_VF,VA / (ε_VF − ε_VA) for virtual, with I the orbital
   Lagrangian (`imat_mo`).
2. **The builders see only the active orbitals.** `RMP2Result::mo_coeff` and
   `UMP2Result::mo_coeff_{alpha,beta}` hold the active columns, so
   `imat_mo`, the Z-vector, `W_ref`, the ζ weights and the `vhf_s1occ`
   projector all span the active space. PySCF runs them over the full space.
   The HF density does include the frozen orbitals, which is why the UMP2
   result was off rather than obviously broken; the error is mostly the
   frozen core's missing share of `W_ref` and of the overlap term.

To implement: carry the full MOs, energies and the active mask into the
builders; build `imat_mo` over all MOs; add D_FA and D_VF,VA before the
Z-vector right-hand side; run the Z-vector, ζ and W over the full space. That
applies to the RMP2 and UMP2 builders and the double-hybrid Lagrangian that
shares them. Gate: PySCF `mp2.Gradients` with `frozen=1`, plus an energy FD.

## Validation

**Gradients** (6-31G* spherical, every component, `atol` 3e-8, MP2 energy to
1e-9). The reference is a fourth-order FD (h = 0.005 Å) of PySCF's MP2
energy (`tests/pyscf/spherical_mp2_gradient_references.py`). Planck matches
every component to about 1e-8, the print precision.
- `water_rmp2_spherical_gradient_631gd`
- `h2o2_rmp2_spherical_gradient_631gd` (C1)
- `water_cation_ump2_spherical_gradient_631gd`
- `ch3_radical_ump2_spherical_gradient_631gd` (C1, all 12 components
  nonzero). It replaced the planned H2O2 cation, whose UHF does not converge
  from SAD in 6-31G*, even with SOSCF.

**Geometry optimization.** `water_rmp2_spherical_geomopt_631gd` reaches
E_opt −76.1955706453 Eh, PySCF's `geometric` GAU_TIGHT minimum to every
printed digit (`geomopt_grad_tol` 1.5e-5).

**Frequencies.** The reference is central differences of PySCF 2.15.0's
analytic MP2 gradient at Planck's h = 5e-3 Bohr, then
`thermo.harmonic_analysis` (`tests/pyscf/mp2_frequency_references.py`).
Every printed digit, frequencies and MP2 energy:

| Case | Frequencies (cm⁻¹) | E(MP2) |
|---|---|---|
| `water_radical_cation_ump2_freq_sto3g` (Cartesian UMP2) | 1347.71 / 4689.09 / 4868.83 | −74.6833354374 |
| `water_rmp2_spherical_freq_631gd` (spherical RMP2) | 1703.73 / 3919.24 / 4064.72 | −76.1953214676 |

The cost is 6N MP2 gradients for N atoms, each with its own SCF and Z-vector
solve, so MP2 frequencies are for small molecules.

**Refusals:** none left. Spherical RI-MP2 gradients were refused
(`water_ri_rmp2_spherical_gradient_rejected`) until the RI-MP2 response was
fixed (`docs/RI_MP2_GRADIENT_RESPONSE.md`); RI-MP2 frequencies until
`docs/RI_JK_ROUTINE.md` step 8.

### Why the gradient reference was PySCF's energy FD

PySCF 2.13's analytic `mp2.Gradients` misses the FD of its own MP2 energy:

| Case | Gap (Ha/Bohr) |
|---|---|
| CH3 radical UMP2 | 4.2e-7 |
| H2O2 RMP2 | 1.1e-7 |
| water cation UMP2 | 1.7e-7 |
| water RMP2 | 1.4e-8 |

The gap was the same in the Cartesian basis, and tightening its CPHF to
1e-13 / 300 iterations did not move it. Planck matched the FD to ~1e-8 in
both bases. It is fixed in PySCF 2.15.0 (CH3 C x: gap 6.5e-9; 2.14.0 still
4.1e-7) by the shared Krylov solver fix pyscf/pyscf#3334, where a fixed 1e-14
linear-dependence threshold stopped the solve early while it reported
convergence. On PySCF < 2.15, use an energy FD as the MP2 gradient
reference; from 2.15 the analytic gradient is usable, which is what the
frequency references rely on.

## Not in this scope

- **Frozen-core MP2 gradients**: what they need is above. The next
  most-asked MP2 feature; it gets its own scope.
- **Spherical RI-MP2 gradients**: done 2026-11-16. The RI derivative
  contraction lifts to Cartesian exactly (`docs/RI_JK_ROUTINE.md`
  step 16), and the RI-MP2 response defect is fixed
  (`docs/RI_MP2_GRADIENT_RESPONSE.md`).
- **RI-MP2 frequencies**: done 2026-11-16 (`docs/RI_JK_ROUTINE.md`
  step 8).
