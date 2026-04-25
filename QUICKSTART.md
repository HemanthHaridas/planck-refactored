### Quickstart

This guide gets you from zero to a converged Hartree-Fock calculation in five minutes, then shows the shortest path into DFT and TDDFT.

### 1. Build

```bash
git clone https://github.com/HemanthHaridas/planck-refactored.git
cd planck-refactored
cmake -B build .
cmake --build build
cmake --install build
```

The first build fetches Eigen and libmsym automatically. The executable is `./install/bin/hartree-fock`.

### 2. First calculation

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
SCF Mode :                    Conventional
2e Integrals :                Building ERI tensor (0.0 MB)
2e Integrals :                ERI tensor ready

Iter  Energy              DeltaE         RMS(D)      ...
1     -73.24098642        73.24098642    ...
...
SCF Converged after 18 iterations

    MO    Symmetry              Energy (Eh)
     1          A1                  -20.252
     2          A1                   -1.258
     3          B2                   -0.594
     4          A1                   -0.460
     5          B1                   -0.393  <-- HOMO
     6          A1                    0.582  <-- LUMO
     7          B2                    0.693

  Total Energy                        -74.9659012173    -2039.190    -47012.6
Dipole Moment (origin at 0.000000, 0.000000, 0.000000 bohr)
Component        Electronic (au)      Nuclear (au)        Total (au)     Debye
X                       0.000000          0.000000          0.000000  0.000000
Y                       0.000000          0.000000          0.000000  0.000000
Z                      -1.23...           1.90...           0.67...   1.71...
Quadrupole Moment (traceless Cartesian tensor, au)
Component             Electronic           Nuclear             Total
XX                     ...
XY                     ...
XZ                     ...
YY                     ...
YZ                     ...
ZZ                     ...
```

The total energy is printed in Hartree, eV, and kcal/mol. After convergence, Planck also prints dipole and quadrupole components automatically from the final AO density matrix. Dipoles are reported in atomic units and Debye; quadrupoles are reported as a traceless Cartesian tensor in atomic units.

### 2b. First DFT calculation

Use `planck-dft` to run Kohn-Sham DFT with libxc functionals:

```bash
./build/planck-dft water.hfinp
```

Add a `%begin_dft` block to switch on DFT. For example, PBE/STO-3G:

```text
%begin_dft
    grid        coarse
    exchange    pbe
    correlation pbe
%end_dft
```

`scf_type rhf` gives an RKS reference and `scf_type uhf` gives a UKS reference.

### 3. Open-shell calculation (UHF)

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

The same `scf_type uhf` setting also selects UKS when you run `planck-dft`.

### 3b. TDDFT / linear response

Planck’s TDDFT driver lives in `planck-dft` and is requested with `calculation tddft` (aliases: `linearresponse`, `lr`, `td-dft`).

Minimal closed-shell singlet example:

```text
%begin_control
    basis       sto-3g
    calculation tddft
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    engine      os
%end_scf

%begin_dft
    grid        coarse
    exchange    pbe
    correlation pbe
    lr_nstates  3
%end_dft
```

Useful TDDFT controls:

```text
lr_nstates   5                # number of roots
lr_method    casida           # default; use tda for A-only response
lr_spin      singlet          # RKS: singlet or triplet
lr_spin      spin_conserving  # UKS/open-shell response
```

Current behavior:

- `lr_method casida` is the default and solves the full \(A/B\) problem.
- `lr_method tda` keeps the older Hermitian \(A\)-only approximation.
- RKS supports `singlet` and `triplet` response.
- UKS supports spin-conserving response blocks.
- Semilocal XC kernels are included for the supported LDA and GGA functionals.


### 4. Checkpoint and restart

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


### 5. Analytic gradient and geometry optimization

### Gradient only

Set `calculation gradient` to compute the analytic nuclear gradient at the input geometry and stop. The gradient is printed in Ha/Bohr, one row per atom. RHF, UHF, RMP2, and UMP2 gradients use this same entry point:

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


### 6. Vibrational frequency analysis

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


### 7. Z-matrix input

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


### 8. MP2 correlation energy and natural orbitals

Add `correlation rmp2` (closed-shell) or `correlation ump2` (open-shell) to the `%begin_scf` block to compute the second-order Møller-Plesset correlation energy after the SCF converges:

```
%begin_scf
    scf_type    rhf
    engine      os
    correlation rmp2
%end_scf
```

After a single-point RMP2 run, Planck automatically diagonalizes the unrelaxed MP2 one-particle density matrix and prints the natural orbital occupancies and their expansion in canonical MOs:

```
  MP2 Natural Orbital Occupancies :
    NO   1     1.999812
    NO   2     1.993021
    NO   3     1.985441
    NO   4     1.978033
    NO   5     1.964219
    NO   6     0.020113
    NO   7     0.011432
    ...

  MP2 Natural Orbitals (canonical MO expansion) :
    NO   1 =  +0.999902*MO1
    NO   2 =  +0.998731*MO2 +0.012401*MO6
    ...
```

Occupancies close to 2 indicate strongly occupied (nearly HF) natural orbitals; small values (0.01–0.1) indicate correlation-driven occupation of virtual space. This output is useful for selecting an appropriate active space before a CASSCF calculation.

To compute an analytic MP2 gradient, switch the control block to `calculation gradient` and keep the same correlation keyword. Closed-shell references use `correlation rmp2`; open-shell references use `scf_type uhf` with `correlation ump2`:

```
%begin_control
    basis       sto-3g
    calculation gradient
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type      uhf
    correlation   ump2
    engine        os
    level_shift   0.3
    diis_restart  2.0
%end_scf
```

The gradient section reports the correlated MP2 derivative and includes the usual maximum and RMS gradient norms.

### 9. Coupled cluster

#### RCCSD

`correlation ccsd` runs the conventional iterative RCCSD solver for canonical
closed-shell RHF references. It expands spatial orbitals into a spin-orbital
basis, forms the standard τ/τ̃ tensors and CCSD intermediates
(F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej), then iterates T1/T2 amplitudes
with DIIS. Scales O(N⁶); no system-size cap.

```
%begin_control
    basis       sto-3g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    correlation ccsd
    use_diis    .true.
    diis_dim    8
    engine      os
    guess       hcore
    save_checkpoint .false.
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom

%begin_coords
2
0   1
H     0.000000     0.000000    -0.370500
H     0.000000     0.000000     0.370500
%end_coords
```

#### UCCSD (determinant prototype)

`correlation uccsd` uses a determinant-space teaching prototype for open-shell
UHF references. Intended for small classroom-scale systems (≤ 12 spin orbitals
/ ≤ 1200 determinants). Boron atom is the canonical test case.

```
%begin_scf
    scf_type    uhf
    correlation uccsd
    use_diis    .true.
    diis_dim    8
    engine      os
%end_scf

%begin_coords
1
0   2
B     0.000000    0.000000    0.000000
%end_coords
```

#### RCCSDT — small system (determinant prototype)

`correlation ccsdt` on a small RHF system automatically uses the
determinant-space prototype: it builds the spin-orbital Hamiltonian, enumerates
all S/D/T excitations, evaluates `exp(−T) H exp(T) |Φ₀⟩` in the determinant
basis, and projects residuals onto S/D/T. LiH/STO-3G is the canonical test
(4 electrons, 6 spatial orbitals); triples contribution ≈ 10⁻⁵ Hartree.

```
%begin_scf
    scf_type    rhf
    correlation ccsdt
    use_diis    .true.
    diis_dim    8
    engine      os
    guess       hcore
    save_checkpoint .false.
%end_scf

%begin_coords
2
0   1
Li    0.000000     0.000000     0.000000
H     0.000000     0.000000     1.595000
%end_coords
```

#### RCCSDT — moderate system (tensor warm-start + determinant backstop)

The same `correlation ccsdt` keyword routes to the tensor production backend
automatically when the system exceeds the direct determinant-prototype limit
(nso > 12 or ndet > 1200). `choose_rccsdt_backend` makes this decision; no
keyword change is needed.

For moderate-size systems the tensor backend does not run alone — it is a
three-stage pipeline:

1. **Tensor RCCSD warm-start** — converges T1/T2 using the spin-orbital
   intermediate solver. Prints a per-block memory summary up front.
2. **Staged tensor T3 loop** — begins iterating the triples workspace for a
   fixed number of steps to build T3 amplitude quality.
3. **Determinant backstop** — if the system fits within nso ≤ 16 and the full
   Fock-space determinant count `C(nso, nelec)` ≤ 5000, the determinant solver
   takes over, warm-started from the T1/T2/T3 produced in stages 1–2, and runs
   to full convergence. Systems beyond nso=16 / ndet=5000 will error until the
   fully tensorized T3 residual engine is complete.

Water/STO-3G (nso=14, ndet=C(14,10)=1001) goes through all three stages —
the final convergence is through the determinant backstop with 1001 determinants.

```
%begin_scf
    scf_type    rhf
    correlation ccsdt
    use_diis    .true.
    diis_dim    8
    engine      os
    guess       hcore
    save_checkpoint .false.
%end_scf

%begin_coords
3
0   1
O     0.000000     0.000000     0.000000
H     0.000000    -0.757160     0.586260
H     0.000000     0.757160     0.586260
%end_coords
```

Log markers to watch for: `RCCSDT[TENSOR]` (tensor stages), `RCCSDT[DET-BACKSTOP]`
(determinant backstop iterations).

#### UCCSDT (determinant prototype)

`correlation uccsdt` extends the UCCSD determinant prototype to include triple
excitations (`max_rank = 3`). Same size limit as UCCSD; Boron/STO-3G is the
canonical test.

```
%begin_scf
    scf_type    uhf
    correlation uccsdt
    use_diis    .true.
    diis_dim    8
    engine      os
%end_scf

%begin_coords
1
0   2
B     0.000000    0.000000    0.000000
%end_coords
```

The `UCCSDT − UCCSD` triples correction for B/STO-3G is nonzero and
verifiable against `tests/pyscf/b_uccsdt_sto3g.py`.

---

### 10. CASSCF active-space calculation

CASSCF (Complete Active Space SCF) provides a multireference wavefunction by
performing a full CI expansion within a chosen active space of `nactorb`
orbitals containing `nactele` electrons. The RHF orbitals serve as the starting
reference; CASSCF then simultaneously optimizes the CI coefficients and orbital
rotations until convergence.

```
%begin_control
    basis       sto-3g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type           rhf
    correlation        casscf
    use_diis           .true.
    diis_dim           8
    engine             os
    guess              hcore
    nactele            4        # active electrons
    nactorb            4        # active orbitals
    nroots             1        # 1 = single-state; >1 = state-averaged
    mcscf_max_iter     200
    mcscf_micro_per_macro  4
    tol_mcscf_energy   1e-8
    tol_mcscf_grad     1e-4
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .true.
%end_geom

%begin_coords
3
0   1
O    0.000000    0.000000    0.117176
H    0.000000    0.757005   -0.468704
H    0.000000   -0.757005   -0.468704
%end_coords
```

Output includes the RHF reference energy, the CASSCF energy at each
macro-iteration, orbital gradient norm, and natural orbital occupation numbers:

```
[INF]  CASSCF  macro  0:  E = -75.9849xxxx Eh   |g| = x.xxxe-xx
...
[INF]  CASSCF Converged in N macro-iterations
[INF]  E(RHF)    = -74.9659xxxx Eh
[INF]  E(CASSCF) = -75.9851xxxx Eh
[INF]  Natural occupation numbers (active): 1.9xxx  1.9xxx  0.0xxx  0.0xxx
```

**Choosing the active space**: A good starting point is to include all strongly
correlated orbitals — bonding/antibonding pairs, lone pairs involved in bond
breaking, frontier orbitals. For H₂O with STO-3G, CAS(4,4) includes the four
valence-like orbitals. Start small and check the natural occupation numbers:
values near 0 or 2 indicate weakly correlated orbitals that may not need to be
active.

#### State-averaged CASSCF (SA-CASSCF)

Set `nroots N` to simultaneously optimize orbitals for N electronic states.
Provide `weights` (space-separated, automatically normalized) to use unequal
state averaging. Default is equal weights when `weights` is omitted.

```
%begin_scf
    scf_type           rhf
    correlation        casscf
    use_diis           .true.
    diis_dim           8
    engine             os
    guess              hcore
    nactele            2
    nactorb            2
    nroots             3
    weights            0.5 0.25 0.25   # S0 weighted 2×, S1 and S2 equal
    mcscf_max_iter     200
    tol_mcscf_energy   1e-8
    tol_mcscf_grad     1e-5
%end_scf
```

The iteration table prints `SA Grad` (convergence quantity,
`‖Σ_I w_I g_I‖∞`) and `MaxRootG` (diagnostic, max over roots).

To pin active orbitals by irrep when `use_symm .true.`:

```
    core_irrep_counts   A1=3 B1=1
    active_irrep_counts A1=1 B1=1
```

---

### 11. RASSCF active-space calculation

RASSCF extends CASSCF by partitioning the active space into three subspaces:
RAS1 (hole excitations limited by `max_holes`), RAS2 (full CI, like a small
CAS), and RAS3 (particle excitations limited by `max_elec`). Total active
orbitals must equal `nras1 + nras2 + nras3 = nactorb`.

```
%begin_control
    basis       sto-3g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control

%begin_scf
    scf_type    rhf
    correlation rasscf
    use_diis    .true.
    diis_dim    8
    engine      os
    nactele     4
    nactorb     4
    nroots      1
    nras1       1        # 1 orbital in RAS1; at most max_holes holes allowed
    nras2       2        # 2 orbitals in RAS2 (full CI)
    nras3       1        # 1 orbital in RAS3; at most max_elec electrons allowed
    max_holes   1
    max_elec    1
    mcscf_max_iter  50
    tol_mcscf_energy  1e-8
    tol_mcscf_grad    1e-5
%end_scf

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .true.
%end_geom

%begin_coords
3
0   1
O    0.000000    0.000000    0.117176
H    0.000000    0.757005   -0.468704
H    0.000000   -0.757005   -0.468704
%end_coords
```

### 12. SCF mode

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


### 13. Basis sets

| Keyword | Description |
|---|---|
| `sto-3g` | Minimal; fast, qualitative |
| `3-21g` | Split-valence; improved geometry |
| `6-31g` | Standard split-valence |
| `6-31g*` | 6-31G + d polarization on heavy atoms |


### 14. Convergence tips

| Problem | Fix |
|---|---|
| UHF won't converge | Add `level_shift 0.2` – `0.5` |
| DIIS oscillates | Add `diis_restart 2.0` |
| Slow convergence | Increase `diis_dim` (default 8) |
| Wrong spin state | Check `multiplicity` in `%begin_coords` |
| Wrong energy on restart | Delete `.hfchk` and re-run from scratch |
| MO labels are Ag/Bg/Au/Bu instead of Eg/Eu | Expected — for non-Abelian groups (D3d, Oh, …) the program uses the largest Abelian subgroup (e.g. C2h for D3d). The active group is printed to the log. |


### 15. Kohn-Sham DFT calculation

Kohn-Sham DFT uses a separate executable, `planck-dft`. It reads the same `.hfinp` format but requires a `%begin_dft` block that selects the exchange-correlation functional and integration grid. The `scf_type` keyword in `%begin_scf` controls the reference: `rhf` → RKS, `uhf` → UKS. Global hybrid functionals such as B3LYP and PBE0 include their own correlation through libxc, so the `correlation` keyword is ignored for those combined XC choices.

#### Minimal PBE/STO-3G single point — water (RKS)

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

%begin_dft
    exchange    pbe
    correlation pbe
    grid        normal
%end_dft

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

Run with:

```bash
./build/planck-dft water.hfinp
```

Expected output (abbreviated):

```
Theory :                      Kohn-Sham DFT
Reference :                   RKS
DFT Grid :                    Normal
Exchange :                    PBE
Correlation :                 PBE
...
RKS Converged :               E = -75.xxxxxxxxxx Eh after N iterations
Dipole Moment (origin at 0.000000, 0.000000, 0.000000 bohr)
Component        Electronic (au)      Nuclear (au)        Total (au)     Debye
X                       0.000000          0.000000          0.000000  0.000000
Y                       0.000000          0.000000          0.000000  0.000000
Z                      -1.24...           1.90...           0.66...   1.68...
Quadrupole Moment (traceless Cartesian tensor, au)
Component             Electronic           Nuclear             Total
XX                     ...
XY                     ...
XZ                     ...
YY                     ...
YZ                     ...
ZZ                     ...
DFT Energy :                  -75.xxxxxxxxxx Eh
Converged :                   true
```

The same post-SCF multipole analysis is available in `planck-dft`, because the dipole and quadrupole tensors are evaluated from the converged KS density just like in the HF executable.

#### Common functional combinations

| Input keywords | Functional name | Type |
|---|---|---|
| `exchange slater` + `correlation vwn5` | SVWN | LDA |
| `exchange b88` + `correlation lyp` | BLYP | GGA |
| `exchange b88` + `correlation p86` | BP86 | GGA |
| `exchange pbe` + `correlation pbe` | PBE (default) | GGA |
| `exchange pw91` + `correlation pw91` | PW91 | GGA |
| `exchange b3lyp` | B3LYP | global hybrid GGA |
| `exchange pbe0` | PBE0 | global hybrid GGA |

#### Grid quality

| Keyword | Angular pts (heavy atom) | Use when |
|---|---|---|
| `grid coarse` | ~110 | Quick tests |
| `grid normal` | ~194 | Default; most production runs |
| `grid fine` | ~302 | High-accuracy or difficult systems |
| `grid ultrafine` | ~590 | Benchmark-quality results |

#### Open-shell KS-DFT (UKS)

Set `scf_type uhf` for unrestricted Kohn-Sham. Multiplicity goes in `%begin_coords` as usual.

```
%begin_scf
    scf_type    uhf
    use_diis    .true.
    diis_dim    6
    engine      os
    level_shift 0.3
%end_scf

%begin_dft
    exchange    pbe
    correlation pbe
    grid        normal
%end_dft

%begin_coords
3
0   3
O     0.000000     0.000000     0.000000
H     0.758077     0.000000    -0.602602
H    -0.758077     0.000000    -0.602602
%end_coords
```

#### DFT geometry optimization

`calculation geomopt` works the same as for HF; the KS gradient drives each
step. Use `opt_coords internal` for IC-BFGS (recommended for non-linear molecules).

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
    diis_dim    6
    engine      os
    guess       hcore
    max_cycles  50
    tol_energy  1.0e-8
    tol_density 1.0e-8
%end_scf

%begin_dft
    exchange    pbe
    correlation pbe
    grid        coarse
%end_dft

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom

%begin_coords
2
0   1
H     0.000000     0.000000    -0.450000
H     0.000000     0.000000     0.450000
%end_coords
```

#### DFT frequency analysis

`calculation freq` (or `calculation optfreq`) computes the semi-numerical
Hessian from finite differences of KS gradients. Run at an optimized geometry
to get physically meaningful frequencies.

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
    diis_dim    6
    engine      os
    guess       hcore
%end_scf

%begin_dft
    exchange    pbe
    correlation pbe
    grid        normal
%end_dft

%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom

%begin_coords
2
0   1
H     0.000000     0.000000    -0.370000
H     0.000000     0.000000     0.370000
%end_coords
```

#### Custom libxc functional

To use any libxc functional by its integer ID, replace the enum keywords with numeric IDs:

```
%begin_dft
    exchange_id    1     # libxc LDA_X
    correlation_id 7     # libxc LDA_C_PZ
    grid           normal
%end_dft
```

### 16. All input keywords at a glance

```
%begin_control
    basis        sto-3g | 3-21g | 6-31g | 6-31g*
    basis_type   cartesian | spherical
    calculation  energy | gradient | geomopt | freq | optfreq | imagfollow
    verbosity    silent | minimal | normal | verbose | debug
    basis_path   /path/to/basis-sets          # optional override
    grad_tol     3e-4                         # geomopt convergence threshold (Ha/Bohr)
    max_geomopt_iter  50                      # maximum geometry optimization steps
    hessian_step 5e-3                         # finite-difference step (Bohr) for freq
    print_populations .true. | .false.        # Mulliken population analysis
%end_control

%begin_scf
    scf_type       rhf | rohf | uhf
    scf_mode       conventional | direct | auto
    engine         os | rys | auto
    guess          hcore | sad | density | full
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
    correlation    rmp2 | ump2               # MP2
                 | ccsd                      # RCCSD (iterative tensor, any size)
                 | uccsd                     # UCCSD (determinant prototype, small systems)
                 | ccsdt                     # RCCSDT (auto-selects determinant or tensor backend)
                 | uccsdt                    # UCCSDT (determinant prototype, small systems)
                 | casscf | rasscf           # multireference active-space SCF
    # CASSCF / RASSCF shared
    nactele        4                          # active electrons
    nactorb        4                          # active orbitals
    nroots         1                          # CI roots; >1 = state-averaged
    weights        0.5 0.25 0.25             # SA weights (one per root, auto-normalized)
    core_irrep_counts   A1=3 B1=1            # pin core orbitals by irrep
    active_irrep_counts A1=2 B1=2            # pin active orbitals by irrep
    mo_permutation      0 1 2 ...            # reorder MOs before MCSCF loop (0-based)
    mcscf_max_iter     200
    mcscf_micro_per_macro  4
    tol_mcscf_energy   1e-8
    tol_mcscf_grad     1e-4
    # RASSCF only
    nras1          1                          # orbitals in RAS1 (hole excitations)
    nras2          2                          # orbitals in RAS2 (full CI)
    nras3          1                          # orbitals in RAS3 (particle excitations)
    max_holes      1                          # max holes in RAS1
    max_elec       1                          # max electrons in RAS3
%end_scf

# planck-dft only: selects XC functional and numerical integration grid
%begin_dft
    exchange     slater | b88 | pw91 | pbe | b3lyp | pbe0  # exchange/XC functional (default: pbe)
    correlation  vwn5 | lyp | p86 | pw91 | pbe  # correlation functional (default: pbe)
    exchange_id  <int>                           # custom libxc exchange ID (overrides exchange)
    correlation_id <int>                         # custom libxc correlation ID (overrides correlation)
    grid         coarse | normal | fine | ultrafine   # integration grid quality (default: normal)
    use_sao_blocking    .true. | .false.         # default: .true.
    print_grid_summary  .true. | .false.         # default: .true.
    save_checkpoint     .true. | .false.         # default: .false.
%end_dft

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
