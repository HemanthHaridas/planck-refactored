# What does Planck need for PCM geometry optimizations and frequencies?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Where it stands (2026-11-08)

**All three terms of the analytic PCM gradient exist, and each one matches
finite differences of the energy, but they are not wired into a gradient
yet.** A user still cannot run `calculation gradient` with PCM. Two steps
remain: 4d (wire the terms in and gate them against PySCF) and 5 (geometry
optimization and frequencies).

The groundwork is done. The C-PCM energy is a smooth function of the
geometry, matches PySCF's default C-PCM to every printed digit in both the
Cartesian and spherical bases, and costs the same wall time as the old
stored-matrix design with 5–7× less memory.

| Step | What | Status | Gate |
|---|---|---|---|
| 1 | Smooth SWIG cavity | Done 2026-11-07 | Four `*_pcm_swig_631gd` cases match PySCF; `water_rhf_pcm_smoothness_sto3g` |
| 2 | No stored AO matrices | Done 2026-11-07 | All PCM cases unchanged |
| 3 | Per-iteration cost back to parity | Done 2026-11-07 | Acene timing ladder |
| 6 | Spherical basis | Done 2026-11-08 | Three `*_pcm_swig_spherical_631gd` cases match PySCF `cart=False` |
| 4a | Cavity data for the gradient | Done 2026-11-08 | — |
| 4b | `grad_nuc`, `grad_solver` | Done 2026-11-08 | `planck-pcm-cavity-gradient-fd`: 2.8e-10 / 1.1e-9 Ha/Bohr |
| 4c | `grad_qv` | Done 2026-11-08 | `planck-pcm-qv-gradient-fd`: 1.6e-14 (6-31G*) / 7.7e-14 (cc-pVTZ) |
| 4d | Wire in, PySCF-gate | **Next** | — |
| 5 | Geomopt and frequencies | Open | — |

Each finished term has its own gate, and each gate fails under deliberate
breakage of that term (see the record below). If 4d's total disagrees with
PySCF, the per-term comparison in 4d can trace the gap to one term.

## What remains

### 4d. Wire the gradient in and gate it against PySCF

RHF/UHF and RKS/UKS, `calculation gradient` only, in both bases.

- **One function:** `Solvation::pcm_gradient(calc, shell_pairs, state,
  P_total, ao_atom)` returns natoms × 3. It gets q from one
  `evaluate_pcm_reaction_field` call at the final density, then sums
  `pcm_grad_nuc + pcm_grad_qv + pcm_grad_solver`. `pcm_grad_qv` already
  converts a spherical density to Cartesian.
- **Call it once at each gradient entry point.** `compute_{rhf,uhf}_gradient`
  take an optional `const PCMState *` and build `ao_atom` from the shell →
  atom map that `one_electron_gradient` already uses. The DFT
  `compute_analytic_ks_gradient` replaces its "omits PCM geometry
  response" warning (`src/dft/driver.cpp`) with the term. That warning is
  dead code today, because `io.cpp` rejects every PCM run that is not a
  single point.
- **Lift the gate in `io.cpp`** for `calculation gradient` only. ROHF and
  post-HF stay rejected. The DH gradients keep their own solvent refusals:
  they have no solvent response.
- **Gates:**
  - PySCF per term. `solvent/grad/pcm.py` exposes `grad_nuc`, `grad_qv`
    and `grad_solver` separately. Compare each on C1 H2O2 before comparing
    the total. When the total is the only check, partial cancellations hide
    which term is wrong (see the DH KS/PT2-halves lesson in Completion).
  - PySCF total (`make_grad_object`) on the four Cartesian SWIG cases and
    at least one spherical case, extending
    `tests/pyscf/pcm_swig_references.py`. Set `conv_tol_grad` (see the
    PySCF `conv_tol_grad` memory).
  - Fourth-order FD of the SCF energy to 1e-7 Ha/Bohr on C1 H2O2. This is
    the end-to-end check that the SCF's variational response needs no
    extra term.
  - The forces sum to zero to roundoff. Torque is not a gate (see
    Orientation below).
  - Mutation check: drop each of the three terms in turn and confirm that
    every gate fails.
- **Measure the cost** of `grad_qv` and `grad_solver` on the acene ladder.
  Neither has been timed. `grad_solver` is a serial O(n_tess²) loop marked
  `ponytail:` with its upgrade path.

### 5. Geometry optimization and frequencies

Lifting the gate is not enough here: the inner loops would lose PCM without
any error. `src/opt/geomopt.cpp` (`run_*` at ~95–99),
`src/freq/hessian.cpp` (~84–88) and the post-optimization symmetry SCF in
`src/hf_driver.cpp` (~1735) all call `run_*` without a PCM state.

- Each displaced geometry needs its own `build_pcm_state`, at 0.08–0.4 s
  per build on the acene ladder. Put the build in
  `rebuild_basis_dependent_state` or next to it, so all three loops share
  one path.
- Check that the DFT geomopt/freq callbacks rebuild through `prepare`
  (`prepared.pcm`) at every geometry.
- Gates:
  - water in water optimized against PySCF's `geometric_solver`;
  - frequencies against a PySCF Hessian from finite differences of its
    analytic gradient;
  - a geomopt case that asserts the final energy is the PCM energy. This
    catches the silent loss of PCM in the inner loop.

### Small items

- **Stale comment.** The SAO-suppression comment in `src/hf_driver.cpp`
  (~841) still describes the Fibonacci cavity. The reason it gives (the
  cavity is not symmetry-adapted) still holds for SWIG.
- **Deferred from step 3:** pair screening by a tolerance-linked prefactor
  bound, and a cheaper φ_el contraction. Take these up only if a larger
  system shows the sweeps dominating again.

## What the gradient contains

dG_rxn/dR has three parts, matching PySCF's `solvent/grad/pcm.py`:

- **`grad_nuc`:** the nuclear potential at the tesserae,
  Z erf(ξr)/r, differentiated with respect to both the nucleus and the
  tessera. A tessera moves with its owning atom.
- **`grad_qv`:** the electronic potential, with the AO-pair integrals
  against Gaussian charges differentiated with respect to the basis centres
  (PySCF `int3c2e_ip1`) and the charge centre (`int3c2e_ip2`).
- **`grad_solver`:** the derivative of S at fixed potential. With
  q = −f S⁻¹φ this is qᵀ(dS)q / (2f). S changes in two ways: off the
  diagonal because the tesserae move, and on the diagonal through F_i
  (S_ii = ξ_i√(2/π)/F_i, so through h′). ξ_i depends only on the weights
  and radii, not on the geometry. Under C-PCM the areas do not enter.

The gradient needs no CPHF, because the SCF with PCM is variational in the
charges.

## Record of the landed steps

### Step 1: the smooth cavity

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

The references come from `tests/pyscf/pcm_swig_references.py`, using
Planck's GBS files, its Bohr constant and its unpruned 150 × 590 DFT grid;
`cart` follows each input's `basis_type`. Each case also matches PySCF's
tessera count.

**Smoothness.** `water_rhf_pcm_smoothness_sto3g` scans an O–H stretch over
101 points, h = 0.002 Å. The fourth difference of the energy stays below
2.3e-9 Eh, against a 5e-9 bound. A binary built before the change fails the
same scan at 4.4e-5 Eh, because tesserae appeared and vanished at sphere
boundaries.

**Orientation.** The energy still depends slightly on orientation:
5.2e-6 Eh over five rigid rotations of C1 H2O2 at 302 points, the same as
PySCF. This is accepted; `docs/PCM_ROTATIONAL_INVARIANCE_SCOPE.md` gives
the reasons.

**What the cavity is.** `build_pcm_state` (`src/solvation/pcm.cpp`)
follows PySCF's `solvent/pcm.py` (`gen_surface`, `get_D_S`) term for term:

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

The ESP reaction field (`reaction_field_potential`) switched to
erf(ξr)/r in the same commit, but the `planck-esp-points` unit test kept
hand-built tesserae with ξ left at 0. Every potential then came out as 0,
and the test failed from that commit until 2026-11-08; no regression case
noticed, because real cavities always set ξ > 0. Its tesserae now carry
ξ = 0.3–1.1, small enough that erf is visibly below 1 at the near points,
and the reference uses erf. Putting plain 1/r back fails the test by
3.9e-4.

### Step 2: no stored AO matrices

The cavity stores no AO matrices. `build_pcm_state` builds S and keeps only
its LDLT factor. Each SCF iteration then does three things in
`evaluate_pcm_reaction_field`:

- φ_el at every tessera in one sweep, `_compute_electronic_potential`
  with a Gaussian exponent per point;
- one back-substitution with the stored factor;
- V_rxn as one `_compute_external_charge_attraction` call over all the
  tessera charges.

Memory is O(n_tess²) instead of O(n_tess · n_basis²). The cost is two
integral sweeps per iteration, as in PySCF.

On its own this was a memory win and a time loss. With direct SCF, so that
memory pressure plays no part, and the two binaries run back to back
(6-31G* RHF, 8 threads):

| Run | Stored matrices (before) | Fused sweeps (after step 2) |
|---|---|---|
| Naphthalene, 2160 tesserae, 166 basis functions | 14.3 s, 0.63 GB | 30.1 s, 0.13 GB |
| Anthracene, 2782 tesserae, 230 basis functions | 37.4 s, 1.40 GB | 76.6 s, 0.20 GB |

Both designs give the same energies and iteration counts. The old design
paid for its matrices once, at roughly the cost of one V_rxn sweep, and then
spent only a Frobenius product per tessera per iteration. The fused design
pays two full integral sweeps every iteration, about 1.05 s on naphthalene.
Step 3 won that time back.

**Correction.** This section first reported 157 → 118 s and 3.27 → 1.87 GB
for the default (conventional) SCF on naphthalene. Those numbers are
artifacts. That run builds a 6.07 GB ERI tensor on an 8 GiB machine. The
old design's extra 476 MB of stored matrices pushed it into compression and
paging, which slowed the "before" run. Peak RSS under that pressure (2.4 to
4.6 GB for a 6 GB tensor) does not measure memory use either. Do not time
conventional runs whose ERI tensor approaches RAM.

### Step 3: per-iteration cost back to parity

The target was the stored-matrix design's wall time (14.3 s and 37.4 s back
to back). Planck was already cheaper than PySCF per evaluation: PySCF's
`_get_vind` takes 3.1 s on naphthalene (1.5 s `_get_v` plus 1.5 s
`_get_vmat`; its build takes 2.0 s), against Planck's ~1.05 s.

**Profile.** Timers in `pcm.cpp` print at `verbosity verbose`
(`PCM Setup Timing`, `PCM Timing`). The ladder is the acenes, built the same
way (C–C 1.40 Å, C–H 1.08 Å), 6-31G* RHF, direct SCF, 8 threads
(4 performance + 4 efficiency cores), run back to back. Medians over SCF
iterations, before the fix:

| Molecule | AO pairs | Tesserae | φ_el / s | V_rxn / s | Solve / s | Late gas iteration / s | PCM overhead |
|---|---|---|---|---|---|---|---|
| Benzene | 5253 | 1538 | 0.160 | 0.140 | 0.001 | 0.098 | 309 % |
| Naphthalene | 13861 | 2160 | 0.566 | 0.486 | 0.001 | 0.473 | 219 % |
| Anthracene | 26565 | 2782 | 1.227 | 1.065 | 0.001 | 1.179 | 197 % |

- **Setup is negligible.** Building S takes 0.012–0.030 s and the LDLT
  0.065–0.375 s, once per run. The back-substitution takes ~1 ms per
  iteration.
- **The two sweeps are all of it, in equal parts.** Their cost tracks
  (AO pairs × tesserae) almost exactly: 37, 35 and 31 ns per pair-tessera
  for both sweeps together.
- **Idle time is end-of-region imbalance, not serial work.** In an earlier
  `sample` profile (conventional SCF), the main thread spent only ~2 % in
  serial code, the largest piece being the LDLT back-substitution at 1.2 %.
  The idle samples were barrier waits at the end of the Fock build and of
  the φ_el sweep: `schedule(static)` over tesserae gave each thread an equal
  share, and on mixed performance/efficiency cores those shares finished at
  different times.
- Inside the sweeps: `_os_nuclear_primitive` 10.3k samples, scalar Boys
  5.4k and `_nuclear_hrr` 3.6k.

Three structural causes followed from the code:

- **Per-component evaluation.** The sweeps walked `build_shellpairs`'s
  list, one entry per Cartesian AO pair (13861 for naphthalene). Each entry
  reran the full primitive VRR and HRR, so a (d|d) shell pair repeated the
  same VRR 36 times. Naphthalene has 2926 shell pairs.
- **Scalar Boys.** `_os_nuclear_primitive` called `boys(m, T)` once per
  order; the ERI kernels use the batched `boys_vec`.
- **Screening was a smaller lever than it first looked.** The `ShellPair`
  constructor already drops primitive pairs with |c·prefactor| < 1e-24
  (`kNegligiblePrimitivePair`), so distant pairs cost close to nothing.

**The fix: a shell-block kernel and a dynamic schedule.**
`src/integrals/os.cpp` builds point-charge integrals per shell pair:

- `nuclear_shell_blocks` groups the per-component pair list into shell
  pairs, finding each shell pair's first-component `ShellPair` by scanning
  the list rather than assuming its order.
- `nuclear_vrr_accumulate` runs one VRR per primitive pair and charge over
  the triangle ix+iy+iz ≤ LA+LB, with batched `boys_vec`, and contracts the
  m = 0 slice over primitives (and over charges, for a summed matrix).
- `nuclear_block_from_slice` runs the HRR once per component pair and
  applies the component norms.

Both sweeps use it: `compute_external_charge_attraction_impl` (V_rxn, and
V_ne for every calculation) and `_compute_electronic_potential` (φ_el, and
ESP). The φ_el sweep is `schedule(dynamic, 8)` over points, which stays
deterministic because each point is written by one thread.

| Molecule | φ_el / s (before → after) | V_rxn / s (before → after) |
|---|---|---|
| Benzene | 0.160 → 0.058 | 0.140 → 0.029 |
| Naphthalene | 0.566 → 0.202 | 0.486 → 0.092 |
| Anthracene | 1.227 → 0.466 | 1.065 → 0.200 |

V_rxn gains more (4.8–5.3×) than φ_el (2.6–2.8×): V_rxn does one HRR per
shell pair for all charges together, while φ_el still does one HRR and one
density contraction per (point, shell pair).

Whole PCM runs against the stored-matrix design, both binaries run back to
back twice:

| Run | Stored matrices | Shell-block fused |
|---|---|---|
| Naphthalene | 13.67 / 13.46 s, 0.63 GB | 13.65 / 13.93 s, 0.13 GB |
| Anthracene | 35.22 / 35.31 s, 1.40 GB | 35.80 / 35.93 s, 0.21 GB |

Time is back to parity, 0–3 % slower, with 5–7× less memory. Energies are
unchanged. (An earlier stop criterion, "PCM adds ≤ 25 % to a late
iteration", did not say which SCF mode. It reads 44 % under conventional SCF
and ~200 % under direct.)

**Effect on non-PCM output.** V_ne and ESP for every calculation now go
through this kernel. That changes the summation order, so results are not
bitwise identical. Against the pre-change binary, on HF and ECP gradients,
CHELPG, ESP points and spherical ESP:

- gradients and ESP/CHELPG charges are identical to every printed digit;
- otherwise only signed zeros (−0.000000 against 0.000000) and the tenth
  digit of the kJ/mol and kcal/mol columns differ;
- two cases moved by basin selection, not by error:
  - **HI⁺ UHF:** its degenerate π hole picks a different px/py mixture.
    The quadrupole's xy part rotates (XX 218.172892 → 218.122246,
    XY −0.754 → −0.404) with unchanged magnitude and trace, and the energy
    is unchanged.
  - **`ethylene_rhf_stability_unstable`:** the input was exact 90° twisted
    ethylene (D2d). The two carbon p orbitals are degenerate, so the
    HOMO/LUMO gap at iteration 2 is ~1e-15 and roundoff picks between a
    zwitterion and a covalent state. PySCF does the same: 1e-15 of noise on
    H_core flips its `1e` result in 9 of 10 seeds. The input now
    pyramidalizes the second CH₂ by 0.20 Å and uses `guess sad`. The gap
    opens to 8.4e-4, every PySCF guess reaches one zwitterion, and Planck
    matches it to every printed digit (−77.3844126695) with both kernels.
    Its real-internal, complex and triplet λ_min are −0.068, −0.073 and
    −0.078; it remains the suite's only gate on a negative RHF
    real-internal eigenvalue.

### Step 6: spherical basis

`evaluate_pcm_reaction_field` converts the working density to Cartesian with
`BasisFunctions::to_cartesian` before φ_el, and returns V_rxn through
`to_working`. The zero matrix returned when PCM is off is sized by
`working_nbasis()`. Every caller (the RHF/UHF SCF, the DFT SCF and the ESP
reaction-field report) already passed and expected working-basis matrices,
so no call site changed. Both refusals are removed. In Cartesian mode both
conversions are the identity, so the Cartesian output is unchanged.

| Case (6-31G*, `cart=False`) | PySCF 2.13.0 E / Eh | Tesserae |
|---|---|---|
| `water_rhf_pcm_swig_spherical_631gd` | −76.0158401249 | 542 |
| `water_cation_uhf_pcm_swig_spherical_631gd` | −75.7488908168 | 542 |
| `water_rks_pbe_pcm_swig_spherical_631gd` | −76.3201409489 | 542 |

Planck matches each case to every printed digit, which covers the RHF, UHF
and DFT call sites. The water cation uses `guess sad` (PySCF `minao`). From
`hcore`, both codes stop on the open-shell saddle at −75.6750550413,
0.076 Eh higher (UHF internal λ_min −0.0715; see vault Open Work). SAD
reaches the stable minimum, 1.8e-3 above the Cartesian case, close to
neutral water's 1.4e-3.

### Step 4a: cavity data for the gradient

`PCMState` keeps R_in,J and R_sw,J per atom (`switch_inner`,
`switch_width`). The S build is `pcm_influence_matrix(points)`, and
`build_pcm_state` factors its result. That was all the gradient needed. The
planned per-point w_i and per-atom R are not stored: ξ_i is already on each
point and does not depend on geometry.

### Step 4b: `grad_nuc` and `grad_solver`

`pcm_grad_nuc` and `pcm_grad_solver` (`src/solvation/pcm.cpp`) take the
charges q and return natoms × 3. They are separate functions so that 4d can
compare each one with PySCF.

- Tessera i moves rigidly with its owning atom A (ds_i/dR_A = I).
- Off the diagonal, S_ij = erf(ξ_ij r)/r with ξ_ij constant.
- On the diagonal, dS_ii = −S_ii · Σ_J (h′/h)(x_iJ) · dx_iJ, with
  h′(x) = 30x²(1 − x)². The division is safe: a tessera that was kept has
  h > 0 for every J.
- Terms where both points sit on the same sphere, and the owner's own
  nucleus in `grad_nuc`, cancel exactly and are skipped.

**Gate: `planck-pcm-cavity-gradient-fd`** (ctest). C1 H2O2, 302 points,
705 tesserae. Every displaced geometry rebuilds the cavity from scratch.
The FD is a five-point stencil at h = 1e-4 Bohr (`PCM_FD_H` overrides it).
Worst errors: `grad_nuc` 2.8e-10, `grad_solver` 1.1e-9 Ha/Bohr, against a
1e-8 tolerance. Both terms sum to zero over atoms to 1e-14. The test fails
if a displacement changes the tessera set; at h ≥ 2e-3 the count moves from
705 to 706.

- **The charges must be physical: q = −f S⁻¹ φ_nuc.** The first fixture
  used random |q| ~ 1e-2, and it made the FD meaningless. Nearly-faded
  tesserae have F_i down to 1e-16, so S_ii reaches about 1e16. A real solve
  gives q_i ~ F_i on those points, but a random charge makes q_i² S_ii
  dominate the energy (FD error 1.4e7).
- **The remaining misses are FD error, not a missing term.** The
  `grad_solver` error is 3.3e-4 at h = 1e-3, 9.9e-7 at 5e-4, 4.9e-8 at
  2.5e-4 and 1.1e-9 at 1e-4: it falls faster than h⁴ and converges on the
  analytic value. h is only C² at x = 0 and x = 1, so a stencil that
  straddles either end is poor.
- **Mutations.** Dropping the F_i term fails the gate at 0.49. Dropping the
  off-diagonal factor of 2 fails it at 1.08. Removing either skip still
  passes, which confirms they skip only terms that cancel exactly.

### Step 4c: `grad_qv`

The kernel is `ObaraSaika::_compute_electronic_potential_gradient`
(`os.cpp`), and `Solvation::pcm_grad_qv` wraps it. The wrapper converts the
density to Cartesian and negates the result, because φ_el = −P : V. The
caller passes the AO → atom map.

- **Design.** Per shell pair and owning atom, one VRR to L+1 runs per
  primitive pair and charge. It is `nuclear_vrr_accumulate` with two
  optional extra slices, weighted by 2α·w and 2β·w, rather than a second
  copy of the recurrence. dA and dB come from the up/down angular-momentum
  shift (2α·[a+1] − a·[a−1], and the same for b with β) through
  `_nuclear_hrr`. The integral depends only on relative positions, so the
  charge-centre term is −(dA + dB). Tesserae are grouped by owning atom,
  so the HRR runs once per (shell pair, atom).
- **Ruled out.** `_os_nuclear_primitive_dC` has its own recurrence without
  the Gaussian-charge scaling s = η/(ζ + η), and the per-component
  `_compute_nuclear_deriv_A_elem` is the slow pre-step-3 path, called six
  times per derivative. The translation identity made both unnecessary.
- **Determinism.** Shell pairs run in parallel. Results are stored per
  (shell pair, atom) and summed serially in shell-pair order, so the result
  does not depend on thread count.
- **Limit.** Shell pairs with L_A + L_B + 1 > 2·MAX_L (i-shell pairs) are
  refused, because they do not fit the slice.

**Gate: `planck-pcm-qv-gradient-fd`** (ctest). C1 H2O2, Cartesian 6-31G*
(34 AOs, d shells) and cc-pVTZ (100 AOs, f shells). It uses a fixed random
density and fixed charges on every 7th tessera of the real cavity
(101 points), with a five-point FD at h = 1e-3 Bohr. Every displaced
geometry re-reads the basis and moves each tessera with its atom. The FD
calls only the energy sweep, which shares the VRR but neither the shift nor
the translation term. Worst error: 1.6e-14 on 6-31G* and 7.7e-14 on
cc-pVTZ, against gradients of about 1e-2. The atom sum is 1e-17.

- **Mutations.** Each fails on both bases: dropping the a−1 term
  (1.7e-2 / 1.0e-1), weighting the B slice with α instead of β
  (5.9e-2 / 2.6e-2), dropping the charge-centre term (4.7e-3 / 1.1e-2,
  which also breaks translation), and an off-diagonal factor of 1 instead
  of 2 (8.8e-3 / 1.8e-2).
- The planned comparison against the per-component point-charge kernels
  was not built. The FD is independent of the kernel and covers Gaussian
  charges, which is the case PCM uses.

## Not in this scope

IEF-PCM, SS(V)PE, COSMO's (ε−1)/(ε+½) as an option, and non-electrostatic
terms (cavitation, dispersion-repulsion, as in SMD). Also out: PCM for TDDFT
(linear-response or state-specific), for post-HF, and for ROHF. Restoring
SAO blocking under PCM is out too: the cavity is not symmetry-adapted
(`docs/PCM_ROTATIONAL_INVARIANCE_SCOPE.md`).
