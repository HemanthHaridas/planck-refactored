# What does Planck need for PCM geometry optimizations and frequencies?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Result so far: the cavity is smooth, matches PySCF, and scales (steps 1–2, 2026-11-07)

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

### What a PCM run costs (step 2)

The cavity stores no AO matrices. `build_pcm_state` builds S and keeps
only its LDLT factor. Each SCF iteration then does three things in
`evaluate_pcm_reaction_field`:

- φ_el at every tessera, in one sweep: `_compute_electronic_potential`
  with a Gaussian exponent per point, parallel over tesserae;
- one back-substitution with the stored factor;
- V_rxn as one `_compute_external_charge_attraction` call over all the
  tessera charges.

Memory is O(n_tess²) instead of O(n_tess · n_basis²). The cost is two
integral sweeps per iteration, as in PySCF.

Measured on naphthalene/6-31G* RHF (2138 tesserae, 166 basis functions,
8 threads), with the two binaries run back to back:

| | Stored matrices (before) | Fused sweeps (after) |
|---|---|---|
| Wall time | 157 s | 118 s |
| Time until the ERI tensor is ready (includes cavity setup) | 86 s | 48.5 s |
| Late SCF iteration | ~1.0–1.2 s | ~2.1–2.5 s |
| Peak RSS | 3.27 GB | 1.87 GB (the gas-phase run peaks at 2.10 GB) |

Both give the same energy, −383.3608259764 Eh, in 19 iterations. Setup
saves about 38 s, and each iteration costs about 1.2 s more. The old design
would only win again past roughly 30 SCF iterations, and it cannot avoid
its memory: the stored matrices alone were 471 MB here. Wall times vary by
about 10 % from run to run on this machine, so compare only runs made back
to back. Screening the two sweeps (skipping far shell pairs) is not done,
and would be the next lever if PCM iterations ever dominate.

All nine PCM regression cases and the smoothness gate pass unchanged.

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

2. **Make the cavity's cost scale.** Done (above). `grad_qv` will need
   the same two sweeps in derivative form.

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
