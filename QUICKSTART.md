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

## 5. Z-matrix input

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

## 6. SCF mode

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

## 7. Basis sets

| Keyword | Description |
|---|---|
| `sto-3g` | Minimal; fast, qualitative |
| `3-21g` | Split-valence; improved geometry |
| `6-31g` | Standard split-valence |
| `6-31g*` | 6-31G + d polarization on heavy atoms |

---

## 8. Convergence tips

| Problem | Fix |
|---|---|
| UHF won't converge | Add `level_shift 0.2` – `0.5` |
| DIIS oscillates | Add `diis_restart 2.0` |
| Slow convergence | Increase `diis_dim` (default 8) |
| Wrong spin state | Check `multiplicity` in `%begin_coords` |
| Wrong energy on restart | Delete `.hfchk` and re-run from scratch |
| MO labels are Ag/Bg/Au/Bu instead of Eg/Eu | Expected — for non-Abelian groups (D3d, Oh, …) the program uses the largest Abelian subgroup (e.g. C2h for D3d). The active group is printed to the log. |

---

## 9. All input keywords at a glance

```
%begin_control
    basis        sto-3g | 3-21g | 6-31g | 6-31g*
    basis_type   cartesian | spherical
    calculation  energy | geomopt | freq
    verbosity    silent | minimal | normal | verbose | debug
    basis_path   /path/to/basis-sets          # optional override
%end_control

%begin_scf
    scf_type       rhf | uhf
    scf_mode       conventional | direct | auto
    engine         os
    guess          hcore | read
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
```
