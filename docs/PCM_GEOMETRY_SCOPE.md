# What does Planck need for PCM geometry optimizations and frequencies?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Result so far: the cavity is smooth and matches PySCF (step 1, 2026-11-07)

The energy is now a smooth function of the geometry. That was the
prerequisite for a gradient: the old cavity jumped. Planck's C-PCM uses the
same SWIG discretization as PySCF's default C-PCM and reproduces it to every
printed digit:

| Case (6-31G*, ε = 78.3553, 302 points) | PySCF 2.13.0 E / Eh | G_pcm / Eh | Tesserae |
|---|---|---|---|
| `water_rhf_pcm_swig_631gd` | −76.0172764854 | −0.0115156556 | 542 |
| `h2o2_rhf_pcm_swig_631gd` (C1) | −150.7676892561 | −0.0153189621 | 705 |
| `water_cation_uhf_pcm_swig_631gd` | −75.7506843658 | −0.1478733174 | 542 |
| `water_rks_pbe_pcm_swig_631gd` | −76.3221197790 | −0.0102913608 | 542 |

The references come from `tests/pyscf/pcm_swig_references.py` (`cart=True`,
Planck's GBS files, its Bohr constant and its unpruned 150 × 590 DFT grid).
Each case also matches PySCF's tessera count.

**Smoothness.** `water_rhf_pcm_smoothness_sto3g` scans an O–H stretch over
101 points, h = 0.002 Å. The fourth difference of the energy stays below
2.3e-9 Eh, against a 5e-9 bound. A binary built before the change fails the
same scan at 4.4e-5 Eh, because tesserae appeared and vanished at sphere
boundaries.

**Orientation.** The energy still depends slightly on orientation:
5.2e-6 Eh over five rigid rotations of C1 H2O2 at 302 points, the same as
PySCF. This is accepted; `docs/PCM_ROTATIONAL_INVARIANCE_SCOPE.md` gives
the reasons.

### What the cavity is

`build_pcm_state` (`src/solvation/pcm.cpp`) follows PySCF's
`solvent/pcm.py` (`gen_surface`, `get_D_S`) term for term:

- Atom J has a sphere of radius R_J = 1.2 × the modified Bondi radius, with
  H at 1.1 Å. Each sphere is sampled by an N-point Lebedev grid whose
  weights w_i sum to 4π.
- A point on atom A's sphere is faded out by
  F_i = ∏_{J≠A} h((r_iJ − R_in,J)/R_sw,J), with h(x) = x³(10 − 15x + 6x²).
  Points with w_i F_i ≤ 1e-16 are dropped.
- Each tessera carries a Gaussian charge of width ξ_i = ζ_N/(R√w_i), with
  ζ_N from the York-Karplus table. Then S_ij = erf(ξ_ij r)/r and
  S_ii = ξ_i√(2/π)/F_i. As a point fades, S_ii diverges, so its charge
  goes smoothly to zero.
- `_os_nuclear_primitive` takes a Gaussian-charge exponent
  (`ExternalCharge::exponent`, 0 = point charge). Point-charge output is
  identical, timings aside, on eight non-PCM inputs against a pre-change
  binary.

Three user-visible changes came with it, all to match PySCF's C-PCM:

- f(ε) is now (ε−1)/ε. The old (ε−1)/(ε+½) is COSMO's factor, used under
  the C-PCM name; for water it moves G_pcm by about 0.6 %.
- H uses the modified Bondi radius, 1.1 Å, in the cavity only.
- `surface_points_per_atom` must be a Lebedev size, default 302. It was a
  Fibonacci count, default 60. 74, 230 and 266 have no SWIG constant and are
  rejected.

The five existing PCM cases moved from 24 points to 302 and were re-pinned.
Water RHF/STO-3G now matches PySCF to every printed digit. The RKS PBE case
is 9.5e-7 from PySCF (it was 6.1e-4); what remains is the DFT grid
difference.

## What the gradient contains

dG_rxn/dR has three parts, matching PySCF's `solvent/grad/pcm.py`:

- **`grad_nuc`:** the nuclear potential at the tesserae,
  Z erf(ξr)/r, differentiated with respect to both the nucleus and the
  tessera. A tessera moves with its owning atom.
- **`grad_qv`:** the electronic potential, with the AO-pair integrals
  against Gaussian charges differentiated with respect to the basis centres
  (`int3c2e_ip1`) and to the charge centre (`int3c2e_ip2`).
- **`grad_solver`:** the derivative of S at fixed potential. With
  q = −f S⁻¹φ this is qᵀ(dS)q / (2f). S changes in two ways: off the
  diagonal because the tesserae move, and on the diagonal through F_i
  (S_ii = ξ_i√(2/π)/F_i, so through h′). ξ_i depends only on the weights
  and radii, not on the geometry. Under C-PCM the areas do not enter.

No CPHF is needed: the SCF with PCM is variational in the charges.

The OS engine already has the two integral derivatives for point charges:
`_os_nuclear_primitive_dC` (charge centre) and a raised/lowered angular
momentum form of `_os_nuclear_primitive` (basis centre). The basis-centre
form inherits the Gaussian exponent from the primitive unchanged.
`_os_nuclear_primitive_dC` has its own recurrence and needs the same
s = η/(ζ + η) scaling added.

## Steps

1. **Smooth cavity.** Done (above).

2. **Make the cavity's cost scale.** The 302-point default yields roughly
   180 tesserae per atom (542 for water), and two parts of the design now
   scale badly with it:
   - `PCMState` stores one n_basis² AO matrix per tessera. That is
     n_tess · n_basis² · 8 bytes; for example, about 2 GB at 3000 tesserae
     and n_basis = 300.
   - `evaluate_pcm_reaction_field` refactors the n_tess × n_tess matrix S
     every SCF iteration, which is O(n_tess³) per iteration.

   Fix: factor S once in `build_pcm_state`. Replace the per-tessera
   matrices with two fused sweeps: φ_el at every tessera, done like
   `_compute_electronic_potential` (which takes point charges only today),
   and V_rxn as one `_compute_external_charge_attraction` call over all the
   charges. This comes before the gradient, because `grad_qv` needs the same
   two contractions in derivative form.
   - Gate: the four SWIG cases and the five re-pinned ones are unchanged to
     their printed digits.
   - Gate: peak RSS and the setup/iteration timings on a medium molecule,
     measured before and after.

3. **Analytic gradient**, RHF/UHF and RKS/UKS, Cartesian, energy-only
   workflow `calculation gradient`. Lift the `calculation` gate in
   `src/io/io.cpp` for gradients only.
   - Gate: fourth-order finite difference of the energy to 1e-7 Ha/Bohr on
     C1 H2O2. FD is clean now that the energy is smooth.
   - Gate: PySCF's PCM gradient (`make_grad_object`) on the four SWIG cases.
   - Gate: the force components sum to zero. The atom-centred cavity
     translates with the molecule, so the sum must vanish to roundoff. Net
     torque does not vanish (see Orientation) and is not a gate.

4. **Geometry optimization and frequencies** through the gradient, and
   removal of the remaining `calculation` gate. The DFT double-hybrid
   gradients keep refusing solvent: they have no solvent response.
   - Gate: water in water optimized against PySCF's `geometric_solver`.
   - Gate: frequencies against a PySCF Hessian from finite differences of
     its analytic gradient.

5. **Spherical basis.** This was waiting on spherical DFT, which has landed,
   so it is now independent of steps 2–4. Lift the density to Cartesian
   before φ_el (as the ESP fix S0.1 did) and transform V_rxn to the working
   basis. Then remove the "PCM solvation" refusal in both binaries
   (`src/hf_driver.cpp` and the DFT `validate_spherical_support`).
   - Gate: PySCF `cart=False` on water/6-31G*, RHF and RKS. The gradient
     follows through the existing lift once step 3 lands.

## Not in this scope

IEF-PCM, SS(V)PE, COSMO's (ε−1)/(ε+½) as an option, and non-electrostatic
terms (cavitation, dispersion-repulsion, as in SMD). Also out: PCM for TDDFT
(linear-response or state-specific), for post-HF, and for ROHF. Restoring
SAO blocking under PCM is out too: the cavity is not symmetry-adapted
(`docs/PCM_ROTATIONAL_INVARIANCE_SCOPE.md`).
