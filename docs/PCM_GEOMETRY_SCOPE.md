# What does Planck need for PCM geometry optimizations and frequencies?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Why it matters

Solution-phase structures and frequencies are routine. Planck's C-PCM
(`src/solvation/pcm.{h,cpp}`) is single-point only for HF and DFT, and is
not coupled to TDDFT or post-HF.

## Why gradients were not built: the cavity is not smooth

`build_pcm_state` places a fixed number of Fibonacci-sphere points on each
atom's scaled van der Waals sphere, gives every point the area
$4\pi R^2 / N$, and **discards any point inside another atom's sphere**. The
code's own comment says this "can introduce small ripples in the energy as a
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
   (`docs/SPHERICAL_DFT_SCOPE.md`) once that lands; V_rxn is built in
   Cartesian and projected.

## Not in this scope

IEF-PCM / SS(V)PE, non-electrostatic terms (cavitation, dispersion-
repulsion, as in SMD), PCM for TDDFT (linear-response or state-specific) and
for post-HF.

## Note

The known rotation dependence of the PCM energy (~5e-5 Eh, from the fixed
point orientation) shrinks with a denser Lebedev grid but does not vanish
with a smooth scheme either; PySCF has the same property.
