# Does the PCM energy depend on how the molecule is oriented?

Yes, slightly, and that is accepted. Since the SWIG cavity (2026-11-07,
`docs/PCM_GEOMETRY_SCOPE.md`) the dependence is smooth, small at the default
grid, shrinks as the grid gets denser, and is the same as PySCF's.
Planck does not build the cavity in a frame attached to the molecule, for
the reasons below.

## Why the energy is not exactly invariant

Each atomic sphere is sampled by a Lebedev grid whose axes are the lab axes.
Rotating the molecule rotates the atoms but not the grid, so the tesserae
move across each sphere and the discretized surface integral changes. The
exact surface integral is invariant; only its quadrature is not.

## Measured

C1 H2O2, RHF/STO-3G, water (ε = 78.3553), `use_symm .false.`. Energy spread
over five rigid rotations (identity, 0.37 rad about z, 0.91 about (1,1,0), 1.7
about (1,−2,3), 2.6 about (0.3,1,0.2)):

| Lebedev points per sphere | Spread / Eh | Tesserae (range over rotations) |
|---|---|---|
| 110 | 1.20e-5 | 281–289 |
| 302 (default) | 5.21e-6 | 693–705 |
| 590 | 1.10e-6 | 1290–1304 |
| 1202 | 8.70e-7 | 2505–2535 |

PySCF 2.13.0's default C-PCM on the same molecule and rotations gives
**5.21e-6 Eh** at 302 points, the same as Planck. Planck's cavity matches
PySCF's term for term: the four gates in `docs/PCM_GEOMETRY_SCOPE.md` agree
to every printed digit at 302 points.

The tessera count changes with orientation, but that is harmless now. A point
is dropped only once its switching weight is below 1e-16, so it carries no
charge when it leaves.

Before SWIG, the cavity was a Fibonacci spiral with a hard bury test and equal
areas. A C1 molecule spread **6.0e-5 Eh** at 24 points per atom, and 5.0e-6
at 400. The total cavity area jumped by up to 8 % between orientations,
because points appeared and vanished at sphere boundaries. That was a jump,
not just a quadrature error, and it is gone. Along an O–H stretch
(`tests/pcm_smoothness_scan.py`, water/STO-3G, 110 points, h = 0.002 Å) the
third difference of the energy falls smoothly from −1.7e-7 to −6e-8 Eh, and
the fourth difference stays below 2.3e-9 Eh. (The scan was not rerun against
the old cavity, so it is not yet mutation-checked.)

## Why not a molecule-attached frame

Rotating the grid with the molecule, by building it in the principal-axis
frame of the inertia tensor, makes the energy exactly invariant. With the old
Fibonacci cavity, a co-rotated spiral gave an identical tessera count and area
at every angle. It is not adopted, for three reasons:

- **It breaks smoothness, which the gradient needs.** The principal axes
  are not unique when moments of inertia are degenerate or nearly so (CH4,
  benzene, any symmetric top, linear molecules). There the frame, and with it
  the energy, jumps as the geometry passes through or near the degeneracy. A
  geometry optimization moves through such geometries, and a frequency
  calculation displaces from them.
- **It adds a gradient term.** A frame that depends on the geometry makes
  every tessera position depend on all the atoms, through the frame
  rotation, and not only on its own atom. The analytic gradient (scope step
  2) would need the derivative of the eigenvectors of the inertia tensor,
  which diverges at those degeneracies.
- **It would break the PySCF gate.** PySCF uses the lab-frame grid, so a
  rotated grid would only agree with it loosely.

At the default grid the dependence (5e-6 Eh) is below typical C-PCM model
error. Where it matters, a denser grid reduces it (590 points: 1.1e-6).

## What a user sees

- With `use_symm .true.`, libmsym reorients a symmetric molecule into a
  standard frame, so the energy does not depend on how its coordinates were
  typed. A C1 molecule has nothing to standardize against and keeps its input
  orientation.
- Regression cases therefore pin an orientation, not a frame-invariant
  number. The PCM gates use `use_symm .false.` so that both Planck and PySCF
  use the input frame.
- SAO blocking stays disabled under PCM (PR #141). A Lebedev grid has
  octahedral symmetry, but after a general rotation it is not adapted to the
  molecule's point group.

## Not answered here

Whether to raise the default grid density. Each step up costs more: the
influence-matrix factor is n_tess², and each SCF iteration does two integral
sweeps over the tesserae. 302 matches PySCF's default.
