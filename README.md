### Planck

A Hartree-Fock quantum chemistry program implementing restricted and unrestricted SCF theory with an Obara-Saika integral engine, analytic nuclear gradients, geometry optimization, vibrational frequency analysis, DIIS convergence acceleration, symmetry detection, and binary checkpoint support.

### Features

- **RHF / UHF** — closed-shell and open-shell Hartree-Fock
- **Obara-Saika integral engine** — recursive 1e and 2e integrals (S, P, D, F, G, H shells)
- **Conventional and Direct SCF** — ERI tensor stored once (conventional) or recomputed per iteration (direct); auto-selection based on system size
- **DIIS** — Pulay extrapolation with optional automatic subspace restart
- **Level shifting** — virtual orbital energy raising for open-shell convergence
- **Symmetry detection** — point group via libmsym; standard-orientation coordinates
- **MO symmetry** — irreducible representation labels (A1, B2, Ag, Bu, …) assigned to each converged orbital; Cartesian AO coefficients are transformed to the real spherical harmonic basis and decomposed into symmetry species via libmsym's SALC machinery; the Cartesian→spherical block transform covers all shells supported by the integral engine (S through H, L=0–5); for non-Abelian groups (D3d, Td, Oh, …) the largest Abelian subgroup with all one-dimensional irreps is automatically selected (e.g. C2h for D3d) so every MO receives a unique, unambiguous label — the active group or subgroup is printed to the log; linear molecules (C∞v / D∞h) use a dedicated character-based handler
- **Post-HF** — RMP2 and UMP2 correlation energy corrections
- **Analytic nuclear gradients** — RHF and UHF nuclear gradients via the Obara-Saika AM-shift rule; all five terms (1e GTO-centre, nucleus-position V, ERI, Pulay/overlap, nuclear repulsion) assembled exactly
- **Geometry optimization** — L-BFGS optimizer in Cartesian coordinates or BFGS optimizer in redundant generalized internal coordinates (GIC); Wolfe/Armijo line search; convergence on max Cartesian gradient component
- **Vibrational frequency analysis** — semi-numerical Hessian from central finite differences of analytic gradients (2×3N evaluations); mass-weighted normal mode analysis with Eckart T+R projection; outputs vibrational frequencies in cm⁻¹ (imaginary encoded negative), zero-point energy in Ha and kcal/mol; linearity auto-detected
- **Checkpoint system** — binary `.hfchk` files (version 2); same-basis restart (`guess density`), full geometry+density restart (`guess full`), and cross-basis density projection (Löwdin SVD); checkpoint stores `has_opt_coords` flag set after a converged geometry optimization
- **Basis sets** — STO-3G, 3-21G, 6-31G, 6-31G\*

### Requirements

| Dependency | Version | Source |
|---|---|---|
| C++ compiler | C++23 | GCC ≥ 13 or Clang ≥ 17 |
| CMake | ≥ 3.15 | System package manager |
| Eigen | 3.4.0 | Fetched automatically |
| libmsym | latest | Fetched automatically |
| OpenMP | any | Optional; system package manager |

### Installation

### 1. Clone the repository

```bash
git clone https://github.com/HemanthHaridas/planck-refactored.git
cd planck-refactored
```

### 2. Configure

```bash
cmake -B build .
```

To disable OpenMP:

```bash
cmake -B build . -DUSE_OPENMP=OFF
```

To set a custom install prefix:

```bash
cmake -B build . -DCMAKE_INSTALL_PREFIX=/path/to/prefix
```

### 3. Build

```bash
cmake --build build
```

The first build fetches and compiles Eigen and libmsym automatically. Subsequent builds are incremental.

### 4. Install (optional)

```bash
cmake --install build
```

This installs the `hartree-fock` executable to `<prefix>/bin/` and the basis set files to `<prefix>/share/basis-sets/`.

### 5. Run

```bash
./build/hartree-fock molecule.hfinp
```

### Input File Format

Input files use an INI-style block format with the extension `.hfinp`. Each section is delimited by `%begin_<section>` and `%end_<section>` markers. Keywords are case-insensitive; boolean values accept `.true.` / `.false.`.

```
%begin_control
    ...
%end_control

%begin_scf
    ...
%end_scf

%begin_geom
    ...
%end_geom

%begin_coords
<natoms>
<charge>  <multiplicity>
<symbol>  <x>  <y>  <z>
...
%end_coords
```

### Section: `%begin_control`

General calculation settings.

| Keyword | Type | Values | Default | Description |
|---|---|---|---|---|
| `basis` | string | `sto-3g`, `3-21g`, `6-31g`, `6-31g*` | — | Basis set name |
| `basis_type` | enum | `cartesian`, `spherical` | `cartesian` | Angular function type. Only Cartesian is fully supported. |
| `calculation` | enum | `energy` / `sp`, `gradient` / `grad`, `geomopt` / `opt`, `freq` / `frequency` | — | Calculation type. `gradient` computes the analytic nuclear gradient at the input geometry and stops. `geomopt` optimizes the geometry. `freq` computes the semi-numerical Hessian and vibrational frequencies. |
| `verbosity` | enum | `silent`, `minimal`, `normal`, `verbose`, `debug` | `minimal` | Output level |
| `basis_path` | string | filesystem path | compiled-in | Override the basis set search directory |

### Section: `%begin_scf`

SCF procedure and convergence settings.

| Keyword | Type | Values | Default | Description |
|---|---|---|---|---|
| `scf_type` | enum | `rhf`, `uhf` | `rhf` | Wavefunction type |
| `engine` | enum | `os` / `obara-saika` | `os` | Integral engine (Obara-Saika recursive algorithm) |
| `correlation` | enum | `rmp2`, `ump2` | none | Post-HF correlation energy correction |
| `use_diis` | bool | `.true.`, `.false.` | `.true.` | Enable DIIS convergence acceleration |
| `diis_dim` | int | ≥ 2 | `8` | Maximum DIIS subspace size |
| `diis_restart` | float | ≥ 0 | `2.0` | Clear the DIIS subspace when the Pulay error grows by more than this factor relative to the previous iteration. Set to `0` to disable. |
| `level_shift` | float | ≥ 0.0 | `0.0` | Virtual orbital level shift in Hartree. Raises virtual MO energies to widen the HOMO–LUMO gap and suppress orbital swapping. Recommended `0.2`–`0.5` for open-shell systems. Set to `0` to disable. |
| `max_cycles` | int | ≥ 1 | auto | Maximum SCF iterations. Auto-scaling: 50 for small, up to 300 for large systems. |
| `tol_energy` | float | > 0 | `1e-10` | Energy convergence threshold in Hartree |
| `tol_density` | float | > 0 | `1e-10` | Density matrix convergence threshold (RMS and max element) |
| `scf_mode` | enum | `conventional`, `direct`, `auto` | `conventional` | ERI strategy. `conventional`: build the full ERI tensor once before the SCF loop and reuse it (fast per-iteration, higher memory). `direct`: recompute ERIs every iteration (low memory, slower). `auto`: selects `conventional` when `nbasis ≤ threshold`, otherwise `direct`. Only the Obara-Saika engine supports conventional mode; other engines always use direct. |
| `tol_eri` | float | > 0 | `1e-10` | ERI screening threshold (Schwarz) |
| `threshold` | int | ≥ 1 | `100` | Basis function count cutoff used by `scf_mode auto` to decide between conventional and direct. |
| `guess` | enum | `hcore`, `density`, `full` | `hcore` | Initial density guess. `density` loads only the density matrix from the checkpoint (geometry and 1e integrals are recomputed from the input file). `full` restores geometry, charge, multiplicity, and density from the checkpoint — useful to restart from an optimized geometry. Falls back to `hcore` if the checkpoint is missing or incompatible. Cross-basis projection is applied automatically when the checkpoint basis differs from the current basis. |
| `save_checkpoint` | bool | `.true.`, `.false.` | `.true.` | Write a `.hfchk` checkpoint file after successful convergence |

### Section: `%begin_geom`

Molecular geometry options.

| Keyword | Type | Values | Default | Description |
|---|---|---|---|---|
| `coord_type` | enum | `cartesian`, `zmatrix` / `internal` | `cartesian` | Coordinate specification type |
| `coord_units` | enum | `angstrom`, `bohr` | `angstrom` | Units for input coordinates |
| `use_symm` | bool | `.true.`, `.false.` | `.true.` | Detect molecular point group and reorient to standard frame using libmsym |
| `opt_coords` | enum | `cartesian`, `internal` / `ic` / `gic` | `cartesian` | Coordinate system for geometry optimization. `cartesian`: L-BFGS in 3N Cartesian coordinates. `internal`: BFGS in redundant generalized internal coordinates (bonds, bends, torsions); back-transforms to Cartesian via Schlegel microiterations. Only used when `calculation geomopt`. |

Geometry optimization convergence is controlled by two additional keywords accepted in `%begin_control`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `grad_tol` | float | `3e-4` | Convergence threshold: maximum Cartesian gradient component in Ha/Bohr |
| `max_geomopt_iter` | int | `50` | Maximum number of geometry optimization steps |

### Section: `%begin_coords`

Molecular geometry specification. The header lines are the same for both coordinate types:

```
<natoms>
<charge>  <multiplicity>
```

- **natoms** — number of atoms (integer)
- **charge** — total molecular charge (integer, can be negative)
- **multiplicity** — spin multiplicity M = 2S+1 (integer ≥ 1; 1 = singlet, 2 = doublet, 3 = triplet)

#### Cartesian format (`coord_type cartesian`)

```
<symbol>  <x>  <y>  <z>
...
```

Each subsequent line gives the element symbol and x, y, z coordinates in the units from `coord_units`.

#### Z-matrix format (`coord_type zmatrix`)

```
<symbol>
<symbol>  <i1>  <r>
<symbol>  <i1>  <r>   <i2>  <angle>
<symbol>  <i1>  <r>   <i2>  <angle>   <i3>  <dihedral>
...
```

- `i1`, `i2`, `i3` — 1-based indices of reference atoms
- `r` — bond length to atom `i1` (units from `coord_units`)
- `angle` — bond angle at `i1` relative to `i2` (degrees)
- `dihedral` — dihedral angle about the `i1`–`i2` bond relative to `i3` (degrees)

Example (water):

```
3
0   1
O
H  1  0.9572
H  1  0.9572  2  104.52
```

### Basis Sets

Basis set files are in Gaussian `.gbs` format and are installed to `<prefix>/share/basis-sets/`.

| Name | Keyword | Description |
|---|---|---|
| STO-3G | `sto-3g` | Minimal basis, 3 Gaussians per STO |
| 3-21G | `3-21g` | Split-valence |
| 6-31G | `6-31g` | Pople split-valence |
| 6-31G\* | `6-31g*` | 6-31G plus d polarization on heavy atoms |

### Checkpoint System

After a successful SCF, a binary checkpoint file `<input_stem>.hfchk` is written automatically (when `save_checkpoint .true.`).

### Same-basis restart

Set `guess read` in the next run with the same basis and geometry. The overlap and core Hamiltonian matrices are read directly from the checkpoint, skipping the 1e integral computation.

### Cross-basis projection (density projection)

To warm-start a larger basis from a converged smaller-basis checkpoint:

1. Converge in the small basis with `save_checkpoint .true.`
2. Change `basis` to the larger set and add `guess read`
3. Run — Planck detects the basis-size mismatch, computes the cross-overlap `S_cross = ⟨χ_μ^large | χ_ν^small⟩`, and applies a Löwdin SVD projection to transfer the occupied density into the new basis

The projected density is used as the SCF starting point, typically reducing the number of required iterations.

### Examples

### RHF single point — water, STO-3G

```
%begin_control
    basis       sto-3g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    use_diis    .true.
    diis_dim    8
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

### UHF triplet — open-shell water, 6-31G

```
%begin_control
    basis       6-31g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type     uhf
    use_diis     .true.
    diis_dim     8
    engine       os
    level_shift  0.3
    diis_restart 2.0
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .true.
%end_geom

%begin_coords
3
0   3
O     0.000000    0.000000     0.117176
H     0.000000    0.756950    -0.468703
H     0.000000   -0.756950    -0.468703
%end_coords
```

### Analytic gradient — water, STO-3G

```
%begin_control
    basis       sto-3g
    calculation gradient
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    use_diis    .true.
    diis_dim    8
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

### Geometry optimization — Cartesian L-BFGS

Set `calculation geomopt`. The optimizer uses L-BFGS with a strong-Wolfe line search and converges when the maximum Cartesian gradient component falls below `grad_tol` (default 3×10⁻⁴ Ha/Bohr):

```
%begin_control
    basis       sto-3g
    calculation geomopt
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
O     0.000000     0.000000     0.100000
H     0.800000     0.000000    -0.500000
H    -0.800000     0.000000    -0.500000
%end_coords
```

### Geometry optimization — internal coordinates (BFGS/GIC)

Add `opt_coords internal` in `%begin_geom`. Bonds, valence bends, and proper torsions are detected automatically using Alvarez (2008) covalent radii. The Hessian is initialized diagonally (0.5 Ha/Bohr² for stretches, 0.2 Ha/rad² for bends, 0.1 Ha/rad² for torsions) and updated via BFGS at each step. Steps exceeding 0.3 Bohr or 0.3 rad are scaled down before back-transforming:

```
%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
    opt_coords  internal
%end_geom
```

### Vibrational frequency analysis — water, STO-3G

Set `calculation freq` to compute a semi-numerical Hessian (central finite differences of analytic gradients) and perform mass-weighted normal mode analysis. For best results run at an optimized geometry (no imaginary frequencies):

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

Expected output at the HF/STO-3G optimized geometry:

```
[INF]  Vibrational Frequencies :
[INF]    Molecule: non-linear (6 T+R modes removed, 3 vibrational modes)
[INF]      Mode    Frequency (cm⁻¹)
[INF]         1         1749.xx
[INF]         2         3949.xx
[INF]         3         4131.xx
[INF]  Zero-point energy :  0.020xxx Eh  (12.xx kcal/mol)
```

The step size for finite differences can be changed with `hessian_step` in `%begin_control` (default 5×10⁻³ Bohr). For 2×3N gradient evaluations are required (18 for water). Imaginary frequencies are printed as negative values.

### Cross-basis restart — STO-3G → 6-31G

Run STO-3G first (saves `mol.hfchk`), then change the basis and add `guess read`:

```
%begin_control
    basis       6-31g       # changed from sto-3g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    use_diis    .true.
    engine      os
    guess       read        # reads mol.hfchk, projects density into 6-31G
%end_scf
...
```

### Output

The program prints a structured log to standard output. Key sections:

- **Input Parsing** — confirms input was read successfully
- **Symmetry Detection** — detected point group
- **Basis Construction** — number of shells and contracted functions
- **1e Integrals** — overlap, kinetic, nuclear attraction
- **2e Integrals** — ERI tensor size and build status (conventional mode only)
- **SCF Iterations** — energy, ΔE, RMS(ΔP), Max(ΔP), DIIS error, wall time per iteration
- **MO Symmetry** — when the detected point group is non-Abelian, a line `MO Symmetry: Using Abelian subgroup <X> of <G> for MO labels` is printed; otherwise `Using point group <G> for MO labels`
- **MO Energies** — orbital energies in Hartree with HOMO/LUMO labels and irrep labels when symmetry is enabled
- **⟨S²⟩ / ⟨S⟩** — spin contamination diagnostics (UHF only)
- **Converged Energy** — total energy in Hartree, eV, and kcal/mol; MP2 correlation and corrected total if post-HF enabled
- **Nuclear Gradient** — printed when `calculation gradient` or `calculation geomopt`; one row per atom showing ∂E/∂x, ∂E/∂y, ∂E/∂z in Ha/Bohr, followed by max and RMS norms
- **IC System** — when `opt_coords internal`, logs the count of stretches, bends, and torsions in the redundant GIC set
- **Opt Step N** — per-step log line: energy, max Cartesian gradient, and RMS IC gradient (IC mode) or RMS Cartesian gradient (Cartesian mode)
- **Optimized Geometry** — final Cartesian coordinates in Bohr after convergence
- **Vibrational Frequencies** — when `calculation freq`: molecule type (linear/non-linear), number of T+R modes removed, n_vib vibrational frequencies in cm⁻¹ (negative = imaginary), and zero-point energy in Eh and kcal/mol
- **Wall Time** — total elapsed time

### Section: `%begin_control` (additional keywords)

| Keyword | Type | Default | Description |
|---|---|---|---|
| `grad_tol` | float | `3e-4` | Geometry optimization convergence threshold: maximum Cartesian gradient component in Ha/Bohr |
| `max_geomopt_iter` | int | `50` | Maximum number of geometry optimization steps |
| `hessian_step` | float | `5e-3` | Finite-difference step size in Bohr for the semi-numerical Hessian |

## Build Options

| CMake variable | Default | Description |
|---|---|---|
| `USE_OPENMP` | `ON` | Enable OpenMP parallelism for integral loops |
| `BUILD_TOOLS` | `ON` | Build utility tools (`chkdump` checkpoint formatter) |
| `CMAKE_INSTALL_PREFIX` | `./install` | Installation prefix |

## Utility Tools

### `chkdump`

Converts a binary `.hfchk` checkpoint file to formatted human-readable text:

```bash
./build/chkdump mol.hfchk            # → stdout
./build/chkdump mol.hfchk mol.txt    # → file
```

Output includes: format version, SCF type, convergence status, total energy, nuclear repulsion, geometry (with element symbols), MO energies with HOMO/LUMO labels, overlap matrix, core Hamiltonian, density matrix, Fock matrix, and MO coefficients. Disable with `-DBUILD_TOOLS=OFF`.
