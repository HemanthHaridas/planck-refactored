# How are ESP-derived atomic charges computed, and why is the grid the hard part?

Mulliken and Löwdin charges partition the density by which basis function owns
it. That is bookkeeping, not an observable: a Löwdin charge moves by tenths of
an electron between STO-3G and cc-pVTZ on the same geometry. CHELPG and RESP
instead fit charges to reproduce the molecule's own electrostatic potential
outside its van der Waals surface — the quantity a force field actually needs.

Three steps: generate sampling points, evaluate the potential there, solve a
constrained least-squares problem. **The third step is textbook, the second was
already in the tree, and essentially all the difficulty is in the first.**

---

## 1. The potential was already computable; the fusion is the only new physics

`_compute_external_charge_attraction` (`src/integrals/os.cpp`) builds the AO
potential matrix for charges at *arbitrary* points, and C-PCM has been
contracting exactly such matrices against the density for every cavity tessera
all along (`pcm.cpp:292`). So evaluating φ needed no new integral.

What it did need is a different **shape**. That routine sums its whole charge
list into one AO matrix — right for PCM, where each matrix is reused every SCF
iteration — but a fit needs φ at each point *separately*. Getting that from the
existing entry means one call per point, each allocating an `nbasis²` matrix
contracted once and discarded, or storing one matrix per point. A CHELPG grid is
~10 000 points; neither is affordable.

`ObaraSaika::_compute_electronic_potential` fuses the density contraction into
the shell-pair sweep: **O(npoints × npairs) time in O(npoints) memory**, with no
`nbasis²` temporary at all.

Two details that are easy to get wrong, both load-bearing:

- **`build_shellpairs` emits the upper triangle only**, so an off-diagonal pair
  stands for two elements of the Frobenius product and the diagonal for one.
  This weighting has no counterpart in the matrix builder, which makes it the
  one place the fusion can silently diverge. Forcing the weight to 1.0 moves the
  oracle disagreement from 2.2e-16 to 2.0e-01.
- **Parallelize over points, not pairs.** Over pairs would mean accumulating
  into a shared per-point vector — the cross-thread reduction that has caused
  determinism jitter in the DFT grid layer. Over points, each thread owns
  disjoint output and there is no reduction at all.

There is no `sym_ops` overload, deliberately: symmetry folding reconstructs
AO-pair orbits, which is a property of the *matrix*. A contraction already
reduced against a full unfolded density has nothing to fold.

### Validated cross-code, not just cross-engine

An oracle built from Planck's own matrix builder proves the fusion is
self-consistent, not that it is right — both share one kernel. That is the trap
the spherical transform hit, where every high-L gate was cross-*engine* and so
agreed while all were wrong.

PySCF's `mol.intor('int1e_grids')` is an independent implementation. Water/
STO-3G, converged RHF, five off-axis points: agreement to **4.6e-09 or better**.

The residual is not ESP error. Planck and PySCF round Angstrom→Bohr differently
(Planck keeps more digits), so at identical *input* coordinates the nuclei sit a
few parts in 1e8 apart — worth 6.9e-7 Eh in the nuclear repulsion. Feeding PySCF
Planck's own Bohr coordinates makes the repulsions agree to all ten printed
digits. **~1e-8 is therefore the floor for any Planck-vs-PySCF comparison on an
Angstrom input**, and tightening SCF convergence does nothing: `tol_energy`
and `tol_density` at 1e-13 reproduce the same values bitwise.

---

## 2. The solve: a Lagrange multiplier, not a penalty

Design matrix `A(k,a) = 1/|r_k − R_a|`, minimize `‖Aq − φ‖²` subject to
`Σq = q_total`:

```
[ 2AᵀA   1 ] [ q ]   [ 2Aᵀφ    ]
[ 1ᵀ     0 ] [ λ ] = [ q_total ]
```

The constraint is exact rather than approximate. The zero block guarantees a
negative eigenvalue, so the system is symmetric **indefinite** — `ldlt()` is
required, not merely preferred; `llt()` would fail.

**RESP** adds `a·Σ(√(q²+b²) − b)`, whose derivative contributes a *diagonal*
`a/√(q_k²+b²)`. Since that depends on `q`, the same system is re-solved until
the charges settle (~7 passes on water) — no new solver. Equivalence groups are
additional hard rows in the same Lagrange block, so grouped atoms agree to
2.2e-16 rather than approximately.

Hydrogens are exempt from the restraint by default: it exists to tame buried
heavy atoms the potential barely constrains, and hydrogens sit on the surface
where the data is good.

**Buried atoms fitting badly is not a defect.** An atom with little grid nearby
has almost no leverage on φ, so the unrestrained fit is free to give it a large
charge cancelled by its neighbours. Those charges reproduce the ESP and transfer
badly — which is the entire motivation for the restraint, not something to
chase.

---

## 3. The grid is where the difficulty actually lives

### CHELPG's rotational variance is real and does not converge away

A cubic lattice is not rotationally symmetric, so rotating the molecule changes
*which* points survive the exclusion test. On a field the atom-centred model
cannot represent exactly, the fitted charges move. Measured drift between two
orientations, against charges of order 0.9:

| spacing | drift | spacing | drift |
|---|---|---|---|
| 0.50 Å | 2.2e-01 | 0.20 Å | 6.1e-02 |
| 0.40 Å | 3.6e-01 | 0.15 Å | 7.8e-02 |
| 0.30 Å | 5.8e-02 | 0.10 Å | 2.8e-02 |

3–7 % throughout, **non-convergent in spacing** — a finer lattice is still a
lattice. This is CHELPG's known behaviour, not an implementation defect, and
`chelpg_grid` is kept because published CHELPG numbers need the cubic lattice,
variance included.

### `use_symm true` hides it for symmetric molecules, and is not the fix

`detectSymmetry` calls `msymAlignAxes` and writes a standardized frame into
`_standard` — the frame the grid is built in. On a rigidly rotated C2v water:

| | point group | grid pts | O charge |
|---|---|---|---|
| canonical, symm off | — | 8485 | −0.71334229 |
| rotated, **symm on** | C2v | **8485** | **−0.71334223** |
| rotated, symm off | — | 9232 | −0.71350913 |

Recovery to 6e-8 on a bit-identical grid. But it only helps molecules that *have*
symmetry — C1 falls through to `set_standard_from_bohr(bohr_coords)`, the
untouched input frame, and C1 is most of what anyone fits charges for. It also
makes charges depend on an SCF keyword, and libmsym reorders atoms so the
hydrogen charges swap position. **Do not present it as the remedy.**

### Connolly shells fix it — but only when sampled in a molecular frame

This is the finding most likely to be re-discovered the hard way. The obvious
reasoning is "spheres are rotationally symmetric, so sample spheres." That is
true of the *surface* and false of a **fixed discrete sampling** of it: placing
the same Fibonacci pattern around each atom leaves it pinned to the lab axes,
and rotating the molecule slides points across each surface exactly as the
lattice does.

Measured, that first implementation drifted **3.3e-02** — barely better than the
lattice it replaced — and against sampling density:

| pts/shell | 50 | 200 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|
| drift | 1.0e-1 | 3.3e-2 | 8.1e-3 | 8.7e-3 | 1.6e-2 |

Non-monotone, with a floor: not a resolution problem. Rotating the same
directions *with* the molecule gave **7.1e-15 at every density**, which
identified the cause rather than leaving it plausible.

`connolly_grid` therefore builds its direction frame from the molecule's own
geometry — Gram-Schmidt from the centroid, seeded by the farthest atom, ties
broken by index, all rotation-invariant choices — so the frame rotates with the
molecule by construction. Measured drift **6.9e-15**.

Two alternatives were considered and rejected. A principal-axis frame works for
C1 but makes axes swap discontinuously when moments of inertia are degenerate or
nearly so, trading a bounded variance for a discontinuity. Orientation averaging
is trivially correct but N× the cost and still only mitigates.

`fibonacci_sphere` lives in `src/base/sphere.h` because both the PCM cavity and
these shells need it; two copies would be free to drift apart.

---

## 4. The radii were already there, mislabelled

`ElementData::radius` was documented as `// Covalent radius (Angstrom)` and
holds **van der Waals** radii. H 1.20, C 1.70, N 1.55, O 1.52 match Bondi
exactly; the covalent values would be ~0.31 / 0.73 / 0.71 / 0.66. No new field
was needed, and the PCM cavity — which the comment had led to being accused of
scaling covalent radii — was correct all along.

One genuine data error: F was 1.350, matching no published set against Bondi
1.47, Mantina 1.47, Alvarez 1.46, Batsanov 1.45 and Hu 1.48. Corrected to 1.470.
Blast radius was nil: the radius has one consumer (`pcm.cpp:103`), the only
fluorine-bearing registered case is gas phase, and both PCM cases are water.

**Alvarez 2013 is deliberately not imported** despite being newest and broadest
(93 elements, the first real lanthanide/actinide values). It measures a
*different quantity*: his §5.4 states the radii mark the point of maximum slope
on the leading edge of the van der Waals peak, explicitly **not** a
closest-approach cutoff, naming Pd/Pt/Hg/U as elements where Bondi's values are
cutoff-like. Both consumers here want the cutoff sense. Importing it would move
75 of 93 entries by ≥0.10 Å, essentially always upward (mean −0.38 Å), inflating
every PCM cavity by ~0.4 Å.

PySCF's table is not a drop-in reference either: most of its apparent
disagreements are its own `unknown = 1.999999` filler at Sc–Co, Y–Rh, the
lanthanides and actinides — 49 elements where Bondi published nothing.

---

## 5. What the gates can and cannot establish

PySCF ships **no CHELPG or ESP-fitting module**, so unlike the potential itself
the fit has no cross-code oracle. The substitute is an *analytically known*
answer: fit a field generated by point charges placed on the nuclei, which the
model represents exactly, and require them back. Measured 1.1e-14.

**Three checks in this area were written, passed, and were wrong.** The pattern
is worth more than the individual bugs:

- Asserting the charge constraint on that same nuclear-centred field passed
  **with the Lagrange row deleted**, because the unconstrained solution already
  sums correctly there.
- Asserting rotational invariance on it passed against **two deliberately broken
  grids**, because an exact fit is insensitive to which points sample it.
- The RESP trade assertions were bare inequalities on a 0.5 % effect — they did
  catch an inert restraint, but sat one fixture change from passing on rounding.

**A property can only be gated on a fixture where it binds.** For a constraint
or a grid, that means a field the atom-centred model *cannot* represent. The
current gates therefore split: exactness on a representable field, constraint
and grid behaviour on an unrepresentable one, and minimum effect sizes plus an
iteration-count assertion for the restraint (an inert restraint leaves the normal
matrix unchanged between passes and exits at exactly 2).

`planck-esp-points` check 7 runs **both** grids over one fixture and requires the
CHELPG arm to keep failing — if it ever stops, the fixture has stopped measuring
grid orientation and the Connolly pass is vacuous.

### A mutation test does not prove a value is current

`water_rhf_connolly_sto3g` was committed asserting 1016 grid points and
O −0.709218; committed code produces 1012 and −0.708804. The values were
captured from a binary built before the molecular-frame fix and registered
without rebuilding — that fix changes which directions are generated, hence
which points survive burial.

Its mutation test passed throughout, because **a mutation test shows an
assertion is sensitive to change, not that its value is current**: perturbing a
stale expectation still turns the case red. Confirmed by rebuilding at the prior
commit with the later work stashed out entirely, which reproduces 1012. Capture
regression values from a binary you have just rebuilt, and verify the binary
contains the change rather than trusting a timestamp.

---

## 6. Scope

Landed for the HF path: potential at explicit points, both grids, the
constrained fit, the RESP restraint, and the `%begin_esp` wiring.

Not done, deliberately:

- **RESP stage-2 refitting** (methyl/methylene). It exists for AMBER
  compatibility specifically, and `equivalent` already covers the general case
  of forcing atoms to share a charge.
- **ESP in `planck-dft`.** It has no population analysis at all — one
  `populations/multipole.h` include and nothing else — so wiring ESP there means
  either duplicating the reporter or extracting a shared one.
