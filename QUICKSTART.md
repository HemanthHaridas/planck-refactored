# Quickstart

This guide gets you from zero to a converged Hartree-Fock calculation in five minutes.

---

## 1. Build

```bash
git clone https://github.com/HemanthHaridas/planck-refactored.git
cd planck-refactored
cmake -B build .
cmake --build build
cmake --install build
```

The first build fetches Eigen and libmsym automatically. The executable is `./install/bin/hartree-fock`.

---

## 2. First calculation

Save the following as `water.hfinp`:

```
%begin_control
    basis       sto-3g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    engine      os
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .true.
%end_geom

%begin_coords
3
0   1
O     0.000000    0.000000     0.117176
H     0.000000    0.756950    -0.468703
H     0.000000   -0.756950    -0.468703
%end_coords
```

Run it:

```bash
./build/hartree-fock water.hfinp
```

Expected output (abbreviated):

```
[INF]  SCF Mode :          Conventional
[INF]  2e Integrals :      Building ERI tensor (0.0 MB)
[INF]  2e Integrals :      ERI tensor ready

Iter  Energy              DeltaE         RMS(D)      ...
1     -73.24098642        73.24098642    ...
...
[INF]  SCF Converged after 18 iterations

    MO    Symmetry              Energy (Eh)
     1          A1                  -20.252
     2          A1                   -1.258
     3          B2                   -0.594
     4          A1                   -0.460
     5          B1                   -0.393  <-- HOMO
     6          A1                    0.582  <-- LUMO
     7          B2                    0.693

  Total Energy    -74.9659012173    -2039.190    -47012.6
```

The total energy is printed in Hartree, eV, and kcal/mol.

---

## 3. Open-shell calculation (UHF)

For open-shell systems set `scf_type uhf` and the correct multiplicity. Triplet water (M=3):

```
%begin_scf
    scf_type     uhf
    engine       os
    level_shift  0.3       # helps open-shell convergence
    diis_restart 2.0       # restart DIIS if error grows 2×
%end_scf

%begin_coords
3
0   3                      # charge=0, multiplicity=3
O  ...
```

UHF output includes spin contamination diagnostics:

```
<S^2> :   2.004321  (exact: 2.000000)
<S>   :   1.415390
```

---

## 4. Checkpoint and restart

### Same-basis restart

After a converged run, `water.hfchk` is written automatically. Add `guess read` to skip the ERI build and converge in 1–2 iterations:

```
%begin_scf
    scf_type    rhf
    engine      os
    guess       read
%end_scf
```

### Basis-set stepping (STO-3G → 6-31G)

1. Converge in the small basis — this saves `mol.hfchk`.
2. Change `basis` to the larger set in the **same input file**, keep `guess read`.
3. Planck detects the size mismatch, computes the cross-overlap, applies a Löwdin SVD projection, and uses the projected density as the starting point.

```
%begin_control
    basis    6-31g         # was sto-3g
    ...
%begin_scf
    guess    read          # projects sto-3g density into 6-31g
    ...
```

---

## 5. Analytic gradient and geometry optimization

### Gradient only

Set `calculation gradient` to compute the analytic nuclear gradient at the input geometry and stop. The gradient is printed in Ha/Bohr, one row per atom:

```
%begin_control
    basis       sto-3g
    calculation gradient
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    engine      os
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom

%begin_coords
3
0   1
O     0.000000     0.000000     0.100000
H     0.800000     0.000000    -0.500000
H    -0.800000     0.000000    -0.500000
%end_coords
```

Expected output (abbreviated):

```
[INF]  Nuclear Gradient (Ha/Bohr) :
[INF]    Atom   1:     0.00000000     0.00000000    -0.00819754
[INF]    Atom   2:     0.02280618     0.00000000     0.00409877
[INF]    Atom   3:    -0.02280618     0.00000000     0.00409877
[INF]  Gradient max|g| :   2.281e-02 Ha/Bohr
```

### Geometry optimization — Cartesian L-BFGS

Change `calculation gradient` to `calculation geomopt`. The optimizer runs L-BFGS with a strong-Wolfe line search and stops when the maximum Cartesian gradient component falls below `grad_tol` (default 3×10⁻⁴ Ha/Bohr):

```
%begin_control
    basis       sto-3g
    calculation geomopt
    ...
```

Per-step output:

```
[INF]  Opt Step 0 :  E = -74.9638050907 Eh   max|g| = 2.281e-02   rms|g| = 1.126e-02
[INF]  Opt Step 1 :  E = -74.9647769650 Eh   max|g| = 1.484e-02   rms|g| = 8.133e-03
...
[INF]  Geometry Optimization :  Converged in 4 steps
[INF]  Final Energy :           -74.9659011679 Eh
```

### Geometry optimization — internal coordinates (IC-BFGS)

Add `opt_coords internal` (or `ic`) in `%begin_geom` to optimize in redundant generalized internal coordinates. Bonds, bends, and torsions are auto-detected from the geometry using covalent radii. The Hessian is initialized diagonally and updated via BFGS. IC steps are back-transformed to Cartesian via Schlegel microiterations. This typically converges in fewer steps than the Cartesian optimizer:

```
%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
    opt_coords  internal
%end_geom
```

Water/STO-3G converges in 3 steps with IC-BFGS vs. 4 with Cartesian L-BFGS:

```
[INF]  IC System :   2 stretches, 1 bends, 0 torsions (3 total)
[INF]  Opt Step 0 :  E = -74.9638050907 Eh   max|g| = 2.281e-02   rms|g_ic| = 2.255e-02
[INF]  Opt Step 1 :  E = -74.9653705229 Eh   max|g| = 1.295e-02   rms|g_ic| = 1.226e-02
[INF]  Opt Step 2 :  E = -74.9658997244 Eh   max|g| = 7.026e-04   rms|g_ic| = 6.552e-04
[INF]  Opt Step 3 :  E = -74.9659012162 Eh   max|g| = 1.487e-05   rms|g_ic| = 1.532e-05
[INF]  Geometry Optimization :  Converged in 3 steps
[INF]  Final Energy :           -74.9659012162 Eh
```

### Constrained geometry optimization

Add a `%begin_constraints` block to freeze bonds, angles, dihedrals, or whole atoms during an IC-BFGS run. **Both** `opt_coords internal` and `coord_type zmatrix` are required.

Constraint line formats (atom indices are 1-based):

| Line | Meaning |
|---|---|
| `b  i  j` | Fix bond distance between atoms `i` and `j` |
| `a  i  j  k` | Fix angle `i`–`j`–`k` (`j` = vertex) |
| `d  i  j  k  l` | Fix dihedral `i`–`j`–`k`–`l` |
| `f  i` | Freeze atom `i` (all Cartesian DOFs) |

Example — optimize the H–O–H angle of water while holding both O–H bonds fixed:

```
%begin_geom
    coord_type  zmatrix
    coord_units angstrom
    use_symm    .false.
    opt_coords  internal
%end_geom

%begin_coords
3
0   1
O
H  1  0.9572
H  1  0.9572  2  104.52
%end_coords

%begin_constraints
    b  1  2   # fix O-H1 bond
    b  1  3   # fix O-H2 bond
%end_constraints
```

The log will confirm which coordinates are frozen before optimization starts:

```
[INF]  Constraints :   2 constraint(s) active
[INF]  Constraints :   2 IC(s) frozen, 0 atom(s) frozen
```

---

## 6. Vibrational frequency analysis

Set `calculation freq` to compute vibrational frequencies at the current geometry. The program uses a semi-numerical Hessian (central finite differences of analytic gradients): for each Cartesian DOF, two displaced SCF+gradient evaluations are performed, requiring 2×3N gradient calls in total (18 for water).

For physically meaningful results, run at a stationary point (optimized geometry) — imaginary frequencies indicate a saddle point.

```
%begin_control
    basis       sto-3g
    calculation freq
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    use_diis    .true.
    diis_dim    8
    engine      os
    guess       hcore
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom

%begin_coords
3
0   1
O     0.000000     0.000000     0.000000
H     0.000000     0.756951     0.585766
H     0.000000    -0.756951     0.585766
%end_coords
```

Output (abbreviated):

```
[INF]  Hessian :  Semi-numerical (central differences, h = 0.0050 Bohr, 18 evaluations)
[INF]  Hessian :    9/9 displacements done

[INF]  Vibrational Frequencies :
[INF]    Molecule: non-linear (6 T+R modes removed, 3 vibrational modes)
[INF]      Mode    Frequency (cm⁻¹)
[INF]         1         1749.xx
[INF]         2         3949.xx
[INF]         3         4131.xx
[INF]  Zero-point energy :  0.020xxx Eh  (12.xx kcal/mol)
```

### Combined geomopt + freq workflow

A common pattern: optimize to a minimum, then compute frequencies to confirm it is a true minimum (no imaginary frequencies):

1. Run `calculation geomopt` → saves `mol.hfchk` with `has_opt_coords = 1`
2. Run `calculation freq` with `guess full` → restores optimized geometry from checkpoint, runs SCF in one step, then computes Hessian

```
%begin_scf
    guess full   # restore geometry + density from checkpoint
%end_scf
```

---

## 7. Z-matrix input

Coordinates can be given in Z-matrix (internal coordinate) format. Set `coord_type zmatrix` in `%begin_geom`. Bond lengths are in the units from `coord_units`; angles and dihedrals are always in degrees.

```
%begin_geom
    coord_type  zmatrix
    coord_units angstrom
    use_symm    .false.
%end_geom

%begin_coords
3
0   1
O
H  1  0.9572
H  1  0.9572  2  104.52
%end_coords
```

Each row after the header gives the element symbol followed by reference atom index, bond length, and (for atom 3+) additional reference indices, angle, and dihedral. Atom indices are 1-based. The converted Cartesian coordinates are passed through the rest of the pipeline, so symmetry detection (`use_symm .true.`) works normally.

---

## 8. SCF mode

| Mode | Keyword | When to use |
|---|---|---|
| Conventional | `scf_mode conventional` | Default. Builds ERI tensor once; fast per-iteration. Use when memory allows. |
| Direct | `scf_mode direct` | Recomputes ERIs every iteration. Low memory, slower. |
| Auto | `scf_mode auto` | Selects conventional when `nbasis ≤ threshold` (default 100), otherwise direct. |

Memory for the ERI tensor scales as `nbasis⁴ × 8 bytes`:

| nbasis | Memory |
|---|---|
| 13 (water/STO-3G) | 0.2 MB |
| 50 | 50 MB |
| 100 | 800 MB |
| 150 | 4 GB |

For large systems set `scf_mode direct` or lower `threshold`:

```
%begin_scf
    scf_mode    direct
    ...
```

---

## 9. Basis sets

| Keyword | Description |
|---|---|
| `sto-3g` | Minimal; fast, qualitative |
| `3-21g` | Split-valence; improved geometry |
| `6-31g` | Standard split-valence |
| `6-31g*` | 6-31G + d polarization on heavy atoms |

---

## 10. Convergence tips

| Problem | Fix |
|---|---|
| UHF won't converge | Add `level_shift 0.2` – `0.5` |
| DIIS oscillates | Add `diis_restart 2.0` |
| Slow convergence | Increase `diis_dim` (default 8) |
| Wrong spin state | Check `multiplicity` in `%begin_coords` |
| Wrong energy on restart | Delete `.hfchk` and re-run from scratch |
| MO labels are Ag/Bg/Au/Bu instead of Eg/Eu | Expected — for non-Abelian groups (D3d, Oh, …) the program uses the largest Abelian subgroup (e.g. C2h for D3d). The active group is printed to the log. |

---

## 11. All input keywords at a glance

```
%begin_control
    basis        sto-3g | 3-21g | 6-31g | 6-31g*
    basis_type   cartesian | spherical
    calculation  energy | gradient | geomopt | freq   # freq = frequency analysis
    verbosity    silent | minimal | normal | verbose | debug
    basis_path   /path/to/basis-sets          # optional override
    grad_tol     3e-4                         # geomopt convergence threshold (Ha/Bohr)
    max_geomopt_iter  50                      # maximum geometry optimization steps
    hessian_step 5e-3                         # finite-difference step (Bohr) for freq
%end_control

%begin_scf
    scf_type       rhf | uhf
    scf_mode       conventional | direct | auto
    engine         os
    guess          hcore | density | full
    save_checkpoint .true. | .false.
    use_diis       .true. | .false.
    diis_dim       8
    diis_restart   2.0                        # 0 = off
    level_shift    0.0                        # Hartree; 0 = off
    max_cycles     150
    tol_energy     1e-10
    tol_density    1e-10
    tol_eri        1e-10
    threshold      100                        # auto-mode nbasis cutoff
    correlation    rmp2 | ump2
%end_scf

%begin_geom
    coord_type   cartesian | zmatrix
    coord_units  angstrom | bohr
    use_symm     .true. | .false.
    opt_coords   cartesian | internal           # geomopt coordinate system
%end_geom

%begin_coords
<natoms>
<charge>  <multiplicity>
# Cartesian (coord_type cartesian):
<symbol>  <x>  <y>  <z>
...
# Z-matrix (coord_type zmatrix):
<symbol>
<symbol>  <i1>  <r>
<symbol>  <i1>  <r>  <i2>  <angle>
<symbol>  <i1>  <r>  <i2>  <angle>  <i3>  <dihedral>
...
%end_coords

# Optional: constrained IC geomopt (requires opt_coords internal + coord_type zmatrix)
%begin_constraints
    b  i  j              # fix bond i-j
    a  i  j  k           # fix angle i-j-k
    d  i  j  k  l        # fix dihedral i-j-k-l
    f  i                 # freeze atom i (all Cartesian DOFs)
%end_constraints
```
