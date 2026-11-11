# What does `planck-dft` need for ghost atoms and counterpoise?

Answer, 2026-11-11 (opened as a scope on 2026-11-10). Status lives in
`vault/Status/Open Work.md` and `vault/Status/Completion.md`. The HF design
is in `docs/BSSE_PLAN.md`.

## Answer

`planck-dft` runs Boys–Bernardi counterpoise for RKS and UKS single points
through the **same** five-calculation driver as `hartree-fock`. Getting
there needed four things. Each is described below with its gate.

1. One counterpoise driver, with the method's single point passed in.
2. A ghost grid built with the element's Z.
3. Symmetry turned off when ghosts are present.
4. A KS convergence gate that sees the DIIS residual.

All gates are against PySCF 2.13.0, except where noted.

## 1. One driver, the sub-calculation passed in

`HartreeFock::BSSE::run_counterpoise(parent, run_sub)`
(`src/bsse/counterpoise.{h,cpp}`) builds the five sub-`Calculator`s and
runs each through `run_sub`. The orchestration (atom selection, ghosting,
fragment charges, the report) has nothing method-specific in it.

- **`hartree-fock`** uses the one-argument overload, which passes the HF SCF.
- **`planck-dft`** passes `DFT::Driver::run(sub)` from the CLI entry
  (`src/dft/driver.cpp`), and errors out if that run does not converge. Only
  single points are allowed; other calculation types are refused by name.
- **The sub-calculator copies `_dft`.** That carries the functional, the grid
  and the dispersion setting.
- **PCM is refused for both binaries.** `_solvation` is not copied, so PCM
  used to be dropped silently. A cavity around ghost atoms is also undefined.

Before this, `planck-dft` never read `_bsse`. A `%begin_bsse` block ran the
plain molecule and printed no counterpoise result. A short-lived refusal
(`df86e098`) closed that silent path until the driver landed.

Gates (B3LYP, cc-pVDZ spherical, 150 × 590 unpruned grid, `hcore` guess per
sub-calculation):

| Case | E(AB), BSSE and CP interaction vs PySCF |
|---|---|
| `he2_rks_b3lyp_cp_ccpvdz` | every printed digit |
| `water_dimer_rks_b3lyp_cp_ccpvdz` | every printed digit |
| `water_dimer_rks_b3lyp_d3bj_cp_ccpvdz` | BSSE equals the D3-free case to every digit: ghosts carry no D3 (`dispersion.cpp`) |

Reference script: `tests/pyscf/cp_dft_references.py`. The runner parses the
report into `cp_dimer_energy`, `cp_bsse` and `cp_interaction_energy`. The HF
He₂ counterpoise result did not change.

## 2. The ghost grid is built with the element's Z

A ghost keeps its real `atomic_numbers` entry, and the grid reads that. So
its radial count, Treutler ξ, pruning and Becke radius are those of the real
element. The index-0 entries of the radial tables are never reached for a
ghost. There are two reasons this is right:

- **Basis.** The ghost carries the element's tight basis functions, which a
  Z = 0 grid (H-like) integrates poorly.
- **Grid cancellation.** The dimer run and the ghosted-monomer run then use
  the same grid, so its error cancels in the counterpoise difference.

PySCF ≥ 2.14 does the same: pyscf/pyscf#3302 fixed PySCF's Z = 0 ghost grid
as a bug, checked against Q-Chem. Its gradient side is pyscf/pyscf#3597 and
#3598. On 2.13, `tests/pyscf/ghost_dft_references.py` reproduces the fixed
behaviour by building the grid on the same molecule with the ghosts made real.

Gates (water dimer, second monomer ghosted, 6-31G* Cartesian, 150 × 590):

- `water_dimer_ghost_rks_pbe_631gd`
- `water_dimer_ghost_rks_b3lyp_631gd`
- `water_dimer_ghost_cation_uks_pbe_631gd`

All three match PySCF to every printed digit. PySCF 2.13's own Z = 0 ghost
grid is 3.1e-8 Eh off on the PBE case, so the 1e-9 gate tells the two
treatments apart.

## 3. Symmetry is off when ghosts are present

Ghosts keep their real Z, so libmsym treated them as nuclei. He + Gh(He)
was detected as D∞h, and the RKS SCF failed to converge. `setup_symmetry`
(`src/dft/driver.cpp`) now turns symmetry off whenever ghosts are present,
with the same warning `hf_driver.cpp` gives.
`water_dimer_ghost_rks_pbe_631gd` runs with `use_symm .true.` and checks for
the warning.

## 4. The KS SCF gates convergence on the DIIS residual

Found through the He₂ BSSE, which first missed PySCF by 1.2e-6 Eh. The cause
was the isolated-He monomer: from `hcore` it stopped after 4 iterations, at
a stalled DIIS step where ΔP = 0 while FPS − SPF was still large. That left
it 6.2e-7 Eh above the minimum. This is the defect `SCF::is_converged`
already guards against in HF ("SAD isolated-atom bug"). The guard reads
`metrics.diis_error`:

- **The RKS loop** never set that field.
- **The UKS loop** set it only inside the level-shift / PT2-handoff branch.

Both loops now set it every iteration, as `run_rhf` / `run_uhf` do. A lone
He then converges in 11 iterations to −2.9070540938, PySCF's value.

**The KS loops need a higher stall floor than HF.** With HF's 1e-10, the
gate broke `water_rks_pbe_soscf_puredeadband_631g`:
- the pure-SOSCF deadband deliberately holds the orbitals at |g| ≈ 2e-9,
  where the KS DIIS residual reads 4.1e-10;
- with a 1e-10 floor, the gate never passed, and the run hit `max_cycles`.

`SCF::is_converged` therefore takes the floor as a parameter (default 1e-10,
unchanged for HF), and the KS loops pass `kKsDiisStallFloor = 1e-8`. That is
25× above the deadband residual and five orders of magnitude below the
lone-He stall (~1e-3). One mechanism, with a parameter.

## Not done

These remain nice-to-haves in Open Work:

- counterpoise-corrected gradients and geometry optimization;
- N-body counterpoise (more than two fragments);
- post-HF counterpoise;
- counterpoise under PCM, which is refused.
