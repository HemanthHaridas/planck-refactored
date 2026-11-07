# What does Planck need for PCM geometry optimizations and frequencies?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Why it matters

Solution-phase structures and frequencies are routine. Planck's C-PCM
(`src/solvation/pcm.{h,cpp}`) is single-point only for HF and DFT, and is
not coupled to TDDFT or post-HF.

## Why gradients were not built: the cavity was not smooth

Until step 1 (below), `build_pcm_state` placed a fixed number of Fibonacci-sphere points on each
atom's scaled van der Waals sphere, gives every point the area
$4\pi R^2 / N$, and **discarded any point inside another atom's sphere**. The
code's own comment said this "can introduce small ripples in the energy as a
function of geometry … a reason analytic gradients are not implemented".

The effect is worse than ripples for optimization. As atoms move, points
cross sphere boundaries and appear or vanish, so the energy is piecewise
continuous with jumps, and its derivative does not exist at those geometries.
An analytic gradient of this surface would disagree with finite differences
and stall optimizers. A smooth cavity has to come first.

## Smooth discretizations

The standard fixes replace the hard in/out test with a switching function
that fades a tessera out as it is buried, and smear each point charge into a
Gaussian so that coincident charges do not blow up the self-interaction:

- **SWIG / ISWIG** (Lange and Herbert, 2010): Lebedev points, Gaussian
  charges, a smooth switching function. Energy and gradient continuous.
- **CSC** (Scalmani and Frisch, 2010): the Gaussian scheme in Gaussian.

PySCF's `solvent.pcm` implements C-PCM with Lebedev points, Gaussian charges
and a switching function, and has analytic gradients. **Adopting the same
scheme is what makes a tight PySCF gate possible**; with a different
discretization only loose agreement can be expected.

## What the gradient contains

With the smooth cavity, $dG_\text{rxn}/dR$ collects:
- the solute potential's derivative at fixed tesserae: nuclear part and
  electronic part (derivatives of point-charge AO integrals with respect to
  the basis centres; `os.cpp` already has nuclear-attraction derivatives with
  respect to both the AO centre and the charge centre);
- the tesserae moving with their owning atoms (derivative with respect to
  the charge centre);
- the derivative of the switching weights and the Gaussian-charge D matrix.

No CPHF is needed: the SCF with PCM is variational in the charges.

## Steps

1. **Smooth cavity** (Lebedev, Gaussian charges, switching function),
   energy only, replacing the Fibonacci hard-pruned surface behind the
   existing input keywords.
   - Gate: an energy scan along a bond stretch and a rotation, showing no
     jumps (finite-difference second derivative bounded).
   - Gate: PySCF C-PCM energies with the same Lebedev order, radii and
     scaling, HF and DFT, water and a C1 molecule, at ~1e-8 Eh.
   - The existing PCM regression references move; they are re-pinned to
     the new scheme, with the change recorded.
2. **Analytic gradient**, HF (RHF/UHF) and DFT (RKS/UKS), Cartesian.
   Gate: FD of the energy to 1e-7 Eh/Bohr, and PySCF PCM gradients.
3. **Geometry optimization and frequencies** through the gradient, and
   removal of the current gradient-workflow rejections for PCM. Gate: water
   in water optimized against PySCF; frequencies against a PySCF
   FD-of-analytic-gradient Hessian.
4. **Spherical basis**: follows from the spherical-DFT skin
   (`docs/SPHERICAL_DFT.md`) once that lands; V_rxn is built in
   Cartesian and projected.

## Step 1 status (2026-11-07): implemented, PySCF-matched, not yet registered

Done:

- `build_pcm_state` (`src/solvation/pcm.cpp`) is SWIG and follows PySCF
  `solvent/pcm.py` `gen_surface` / `get_D_S` term for term:
  - Lebedev grids from `DFT::MakeLebedevGrid` (weights sum to $4\pi$, which is
    PySCF's $4\pi w$);
  - the switching function `switch_h` with PySCF's $10^{-8}$ argument cutoff;
  - the $w F > 10^{-16}$ prune;
  - $\xi_i = \zeta_N / (R\sqrt{w_i})$ from the York-Karplus table;
  - $S_{ij} = \operatorname{erf}(\xi_{ij} r)/r$ and
    $S_{ii} = \xi_i\sqrt{2/\pi}/F_i$;
  - nuclear potential $Z\operatorname{erf}(\xi r)/r$.
- `_os_nuclear_primitive` (`src/integrals/os.cpp`) takes a Gaussian-charge
  exponent (`ExternalCharge::exponent`, 0 = point charge). The point-charge
  arithmetic is meant to be bitwise unchanged. The ESP reaction field
  (`esp.cpp`) uses $\operatorname{erf}(\xi r)/r$.

Decisions taken with it, all to match PySCF's C-PCM:

- **f(ε) is now (ε−1)/ε.** The old (ε−1)/(ε+½) is COSMO's factor, labelled
  C-PCM. For water it moves G_pcm by about 0.6 %.
- **Modified Bondi radii in the cavity:** H is 1.1 Å. The shared radius
  table, used by the ESP shells, keeps H at 1.2 Å.
- **`surface_points_per_atom` is a Lebedev size, default 302** (was a
  Fibonacci count, default 60). 74, 230 and 266 have no SWIG constant and
  are rejected. The `io.cpp` check (≥ 6) still runs first.

Gates (unregistered):

- PySCF 2.13.0 references from `tests/pyscf/pcm_swig_references.py`
  (`cart=True`, Planck's GBS and Bohr, 150 × 590 unpruned DFT grid):

  | Input | PySCF E / Eh | G_pcm / Eh | tesserae |
  |---|---|---|---|
  | `water_rhf_pcm_swig_631gd` | −76.0172764854 | −0.0115156556 | 542 |
  | `h2o2_rhf_pcm_swig_631gd` (C1) | −150.7676892561 | −0.0153189621 | 705 |
  | `water_cation_uhf_pcm_swig_631gd` | −75.7506843658 | −0.1478733174 | 542 |
  | `water_rks_pbe_pcm_swig_631gd` | −76.3221197790 | −0.0102913608 | 542 |

  Planck matches all four to every printed digit, with the same tessera
  counts.

- `tests/pcm_smoothness_scan.py`: an O–H stretch scan (water/STO-3G,
  110 points, h = 0.002 Å). It passes: the third difference is smooth
  (−1.7e-7 to −6e-8 Eh) and the fourth stays below 2.3e-9 Eh, against a
  5e-9 bound. It has not been run against the old cavity, so it is not
  mutation-checked.
- Rotation: 5.21e-6 Eh spread at 302 points on C1 H2O2, the same as PySCF
  (`docs/PCM_ROTATIONAL_INVARIANCE_SCOPE.md`).

Remaining for step 1:

1. Check that the point-charge integrals are bitwise unchanged (any
   non-PCM regression case).
2. Re-pin the five existing PCM cases. They use 24 points, which is not a
   Lebedev size, so they now fail at input. Their inputs change too.
3. Register the new cases and the smoothness scan.

## Not in this scope

IEF-PCM / SS(V)PE, non-electrostatic terms (cavitation, dispersion-
repulsion, as in SMD), PCM for TDDFT (linear-response or state-specific) and
for post-HF.

## Note

The known rotation dependence of the PCM energy (~5e-5 Eh, from the fixed
point orientation) shrinks with a denser Lebedev grid but does not vanish
with a smooth scheme either; PySCF has the same property.
