# What does Planck need for PCM geometry optimizations and frequencies?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Result so far: the cavity is smooth, matches PySCF, and scales (steps 1–3, 2026-11-07)

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

The references come from `tests/pyscf/pcm_swig_references.py` (`cart=True`,
Planck's GBS files, its Bohr constant and its unpruned 150 × 590 DFT grid).
Each case also matches PySCF's tessera count.

**Smoothness.** `water_rhf_pcm_smoothness_sto3g` scans an O–H stretch over
101 points, h = 0.002 Å. The fourth difference of the energy stays below
2.3e-9 Eh, against a 5e-9 bound. A binary built before the change fails the
same scan at 4.4e-5 Eh, because tesserae appeared and vanished at sphere
boundaries.

**Orientation.** The energy still depends slightly on orientation:
5.2e-6 Eh over five rigid rotations of C1 H2O2 at 302 points, the same as
PySCF. This is accepted; `docs/PCM_ROTATIONAL_INVARIANCE_SCOPE.md` gives
the reasons.

### What the cavity is

`build_pcm_state` (`src/solvation/pcm.cpp`) follows PySCF's
`solvent/pcm.py` (`gen_surface`, `get_D_S`) term for term:

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

### What a PCM run costs (step 2)

The cavity stores no AO matrices. `build_pcm_state` builds S and keeps
only its LDLT factor. Each SCF iteration then does three things in
`evaluate_pcm_reaction_field`:

- φ_el at every tessera, in one sweep: `_compute_electronic_potential`
  with a Gaussian exponent per point, parallel over tesserae;
- one back-substitution with the stored factor;
- V_rxn as one `_compute_external_charge_attraction` call over all the
  tessera charges.

Memory is O(n_tess²) instead of O(n_tess · n_basis²). The cost is two
integral sweeps per iteration, as in PySCF.

**It is a memory win and a time loss.** With direct SCF, so that memory
pressure plays no part, and the two binaries run back to back (6-31G* RHF,
8 threads):

| Run | Stored matrices (before) | Fused sweeps (after) |
|---|---|---|
| Naphthalene, 2160 tesserae, 166 basis functions | 14.3 s, 0.63 GB | 30.1 s, 0.13 GB |
| Anthracene, 2782 tesserae, 230 basis functions | 37.4 s, 1.40 GB | 76.6 s, 0.20 GB |

Both designs give the same energies and iteration counts. The old design
paid for its matrices once, at roughly the cost of one V_rxn sweep. After
that it spent only a Frobenius product per tessera per iteration. The fused
design pays two full integral sweeps every iteration, which is about 1.05 s
on naphthalene. Step 3 won that time back with a shell-block integral
kernel: the fused design now matches the stored-matrix design's wall time
to within 3 %, with 5–7× less memory (see step 3 below).

**Correction.** This section first reported 157 → 118 s and 3.27 → 1.87 GB
for the default (conventional) SCF on naphthalene. Those numbers are
artifacts. That run builds a 6.07 GB ERI tensor on an 8 GiB machine. The
old design's extra 476 MB of stored matrices pushed it into compression and
paging, which slowed the "before" run. Peak RSS under that pressure (2.4 to
4.6 GB for a 6 GB tensor) does not measure memory use either. Do not time
conventional runs whose ERI tensor approaches RAM.

All nine PCM regression cases and the smoothness gate pass unchanged.

## What the gradient contains

dG_rxn/dR has three parts, matching PySCF's `solvent/grad/pcm.py`:

- **`grad_nuc`:** the nuclear potential at the tesserae,
  Z erf(ξr)/r, differentiated with respect to both the nucleus and the
  tessera. A tessera moves with its owning atom.
- **`grad_qv`:** the electronic potential, with the AO-pair integrals
  against Gaussian charges differentiated with respect to the basis centres
  (`int3c2e_ip1`) and to the charge centre (`int3c2e_ip2`).
- **`grad_solver`:** the derivative of S at fixed potential. With
  q = −f S⁻¹φ this is qᵀ(dS)q / (2f). S changes in two ways: off the
  diagonal because the tesserae move, and on the diagonal through F_i
  (S_ii = ξ_i√(2/π)/F_i, so through h′). ξ_i depends only on the weights
  and radii, not on the geometry. Under C-PCM the areas do not enter.

No CPHF is needed: the SCF with PCM is variational in the charges.

The integral derivatives do not need `_os_nuclear_primitive_dC`, which has
its own recurrence without the s = η/(ζ + η) Gaussian scaling. The integral
depends only on relative positions, so the charge-centre derivative is
−(dA + dB). dA and dB come from the angular-momentum shift
(2α·[a+1] − a·[a−1], and the same for b with β) on one VRR taken to
L_A + L_B + 1. That VRR is `nuclear_vrr_accumulate` with L raised by one,
which already carries the Gaussian scaling. The per-component
`_compute_nuclear_deriv_A_elem` would also work, but it is the slow
pre-step-3 path, called six times per derivative.

## Steps

1. **Smooth cavity.** Done (above).

2. **Make the cavity's memory scale.** Done (above). `grad_qv` will need
   the same two sweeps in derivative form.

3. **Profile and fix the per-iteration PCM cost.** Step 2 made each
   iteration pay two full integral sweeps, which roughly doubled PCM run
   time with direct SCF (above). Planck is still cheaper than PySCF per
   evaluation: PySCF's `_get_vind` takes 3.1 s on naphthalene (1.5 s
   `_get_v` plus 1.5 s `_get_vmat`; its build takes 2.0 s), against
   Planck's ~1.05 s. The targets are Planck's own Fock build and the
   stored-matrix design it replaced.

   **Profile (item 1, done 2026-11-07).** Timers in `pcm.cpp` print at
   `verbosity verbose` (`PCM Setup Timing`, `PCM Timing`). The ladder is
   the acenes, built the same way (C–C 1.40 Å, C–H 1.08 Å), 6-31G* RHF,
   direct SCF, 8 threads (4 performance + 4 efficiency cores), run back to
   back. Medians over SCF iterations:

   | Molecule | AO pairs | Tesserae | φ_el / s | V_rxn / s | Solve / s | Late gas iteration / s | PCM overhead |
   |---|---|---|---|---|---|---|---|
   | Benzene | 5253 | 1538 | 0.160 | 0.140 | 0.001 | 0.098 | 309 % |
   | Naphthalene | 13861 | 2160 | 0.566 | 0.486 | 0.001 | 0.473 | 219 % |
   | Anthracene | 26565 | 2782 | 1.227 | 1.065 | 0.001 | 1.179 | 197 % |

   - **Setup is negligible.** Building S takes 0.012–0.030 s and the LDLT
     0.065–0.375 s, once per run. The back-substitution takes ~1 ms per
     iteration.
   - **The two sweeps are all of it, in equal parts.** Their cost tracks
     (AO pairs × tesserae) almost exactly: 37, 35 and 31 ns per
     pair-tessera for both sweeps together. That is the unscreened
     O(pairs × n_tess) growth.
   - **Idle time is end-of-region imbalance, not serial work.** In the
     earlier `sample` profile (conventional SCF), the main thread spent
     only ~2 % in serial code; the largest piece was the LDLT
     back-substitution at 1.2 %. Its barrier waits split 1147 samples at
     the end of the Fock build and 622 at the end of the φ_el sweep; the
     workers' 8.2k idle samples are the same barriers. φ_el uses
     `schedule(static)` over tesserae, so each thread gets an equal number
     of points, and on mixed performance/efficiency cores those shares
     finish at different times.
   - Inside the sweeps, the earlier profile has `_os_nuclear_primitive`
     10.3k samples, scalar Boys 5.4k and `_nuclear_hrr` 3.6k.

   Three structural costs follow from the code:
   - **Per-component evaluation.** The sweeps walk `build_shellpairs`'s
     list, which has one entry per Cartesian AO pair (13861 for
     naphthalene). Each entry reruns the full primitive VRR and HRR, so a
     (d|d) shell pair repeats the same VRR 36 times. Naphthalene has 2926
     shell pairs, a 4.7× smaller count; the gain in work depends on the
     primitive and angular-momentum mix and has to be measured.
   - **Scalar Boys.** `_os_nuclear_primitive` calls `boys(m, T)` once per
     order. The ERI kernels use the batched `boys_vec`.
   - **Screening, smaller than it first looked.** The `ShellPair`
     constructor already drops primitive pairs with
     |c·prefactor| < 1e-24 (`kNegligiblePrimitivePair`), so distant pairs
     carry short or empty primitive lists and cost close to nothing. A
     tighter, tolerance-linked screen would gain less than the N³ argument
     suggested.

   **Shell-block kernel and dynamic schedule (items 2 and 4, done
   2026-11-07).** `src/integrals/os.cpp` now builds point-charge integrals
   per shell pair:
   - `nuclear_shell_blocks` groups the per-component pair list into shell
     pairs, finding each shell pair's first-component `ShellPair` by
     scanning the list rather than assuming its order.
   - `nuclear_vrr_accumulate` runs one VRR per primitive pair and charge
     over the triangle ix+iy+iz ≤ LA+LB, with batched `boys_vec` and the
     same Gaussian-charge scaling as before. It contracts the m = 0 slice
     into one accumulator, over primitives (and over charges, for a summed
     matrix).
   - `nuclear_block_from_slice` runs the HRR once per component pair and
     applies the component norms.

   Both sweeps use it: `compute_external_charge_attraction_impl` (V_rxn,
   and V_ne for every calculation) and `_compute_electronic_potential`
   (φ_el, and ESP). The φ_el sweep is now `schedule(dynamic, 8)` over
   points. That stays deterministic, because each point is written by one
   thread.

   Per-iteration medians, direct SCF, same ladder:

   | Molecule | φ_el / s (before → after) | V_rxn / s (before → after) |
   |---|---|---|
   | Benzene | 0.160 → 0.058 | 0.140 → 0.029 |
   | Naphthalene | 0.566 → 0.202 | 0.486 → 0.092 |
   | Anthracene | 1.227 → 0.466 | 1.065 → 0.200 |

   V_rxn gains more (4.8–5.3×) than φ_el (2.6–2.8×). V_rxn does one HRR
   per shell pair for all charges together. φ_el still does one HRR and
   one density contraction per (point, shell pair).

   Whole PCM runs against the stored-matrix design, both binaries run back
   to back twice:

   | Run | Stored matrices | Shell-block fused |
   |---|---|---|
   | Naphthalene | 13.67 / 13.46 s, 0.63 GB | 13.65 / 13.93 s, 0.13 GB |
   | Anthracene | 35.22 / 35.31 s, 1.40 GB | 35.80 / 35.93 s, 0.21 GB |

   Time is back to parity, 0–3 % slower, with 5–7× less memory. Energies
   are unchanged.

   **Effect on non-PCM output.** V_ne and ESP for every calculation now go
   through this kernel. That changes the summation order, so results are
   not bitwise identical. Against the pre-change binary, on HF and ECP
   gradients, CHELPG, ESP points and spherical ESP:
   - gradients and ESP/CHELPG charges are identical to every printed digit;
   - otherwise only signed zeros (−0.000000 against 0.000000) and the tenth
     digit of the kJ/mol and kcal/mol columns differ;
   - two cases move by basin selection, not by error:
     - **HI⁺ UHF:** its degenerate π hole picks a different px/py mixture.
       The quadrupole's xy part rotates (XX 218.172892 → 218.122246,
       XY −0.754 → −0.404) with unchanged magnitude and trace, and the
       energy is unchanged.
     - **`ethylene_rhf_stability_unstable`:** from `guess hcore`, its
       trajectory matches to every printed digit through iteration 2 and
       then reaches a lower solution (−77.4195705637 instead of the pinned
       saddle −77.3837525369). That solution is still unstable in the
       triplet and complex channels, but no longer in the real-internal
       one. The input was exact 90° twisted ethylene (D2d): the two carbon
       p orbitals are degenerate, so the HOMO/LUMO gap at iteration 2 is
       ~1e-15 and roundoff picks between a zwitterion and a covalent state.
       PySCF does the same. Its `1e` guess gives the zwitterion and
       `minao`/`huckel` the covalent state, and 1e-15 of noise on H_core
       flips its `1e` result in 9 of 10 seeds. **Resolved:** the input now
       pyramidalizes the second CH₂ by 0.20 Å and uses `guess sad`. The gap
       opens to 8.4e-4, and every PySCF guess and noisy seed reaches one
       zwitterion. Planck SAD matches it to every printed digit
       (−77.3844126695) with both kernels, and the real-internal, complex
       and triplet λ_min are −0.068, −0.073 and −0.078. The case remains
       the suite's only gate on a negative RHF real-internal eigenvalue.

   **Remaining:**
   3. **Pair screening** by a tolerance-linked prefactor bound, only if a
      larger system shows the sweeps dominating again (see the correction
      above).
   - φ_el's per-(point, shell pair) HRR and density contraction are its
     remaining cost. Contracting the density into the HRR's output, or
     transposing the loop so each shell pair handles a batch of points,
     would be the next lever.

   **Gates:**
   - Every PCM regression case and the four SWIG gates are unchanged to
     their printed digits. This is not bitwise, because the summation order
     changes.
   - Non-PCM point-charge paths (V_ne, ESP, CHELPG) keep their printed
     output, or are rebased deliberately if they move to the new kernel.
   - The ladder above, with direct SCF, before and after, run back to back.
     Do not time conventional runs whose ERI tensor approaches RAM (see the
     correction under step 2).

   **Stop** when a direct-SCF PCM run on naphthalene and on anthracene is
   no slower than the stored-matrix design was (14.3 s and 37.4 s back to
   back). That needs about 3.5× on the PCM share of the run. Stop earlier
   if the remaining PCM time leaves these kernels. (An earlier stop
   criterion, "PCM adds ≤ 25 % to a late iteration", did not say which
   SCF mode. It reads 44 % under conventional SCF and ~200 % under direct.)
   **Status:** met to within 3 % (above). Items 2 and 4 are done; item 3
   is deferred.

4. **Analytic gradient**, RHF/UHF and RKS/UKS, Cartesian,
   `calculation gradient` only. One function,
   `Solvation::pcm_gradient(calc, shell_pairs, state, P_total)` returning
   natoms × 3, added once at each gradient skin:
   `compute_{rhf,uhf}_gradient` take an optional `const PCMState *`, and the
   DFT `compute_analytic_ks_gradient` replaces its "omits PCM geometry
   response" warning (`src/dft/driver.cpp`) with the term. That warning is
   dead code today, because `io.cpp` rejects every non-single-point PCM run.
   It recomputes q and φ from the final density with one
   `evaluate_pcm_reaction_field` call.

   - **4a. Cavity data.** `grad_solver` needs dF_i/dR, which
     `PCMSurfacePoint` cannot give. Store the Lebedev weight w_i (so ξ_i
     needs no recomputation) on each point, and R, R_in and R_sw per atom on
     `PCMState`. All three are already computed in `build_pcm_state`.
     Factor the S build out of `build_pcm_state` so a test can rebuild S at
     a displaced geometry.
   - **4b. `grad_nuc` and `grad_solver`.** These need geometry only, no
     integrals. Tessera i moves rigidly with its owning atom A
     (ds_i/dR_A = I). Off the diagonal, S_ij = erf(ξ_ij r)/r with ξ_ij
     constant. On the diagonal, dS_ii = −S_ii · dF_i/F_i. dF_i/dR comes
     from h′ for every J ≠ A and touches both R_A and R_J. `grad_nuc` is
     the derivative of Z_J erf(ξ_i r)/r with respect to the nucleus and to
     the tessera.
     - Unit gate: the analytic qᵀ(dS)q/(2f) against a central difference of
       qᵀSq at fixed q, with S rebuilt at ±h, on C1 H2O2. Do it before any
       integral work. A displacement can change the pruned tessera set
       (w F ≤ 1e-16); the test must keep the set fixed.
   - **4c. `grad_qv`.** A new shell-block derivative kernel, built as
     described under "What the gradient contains". Parallelize over shell
     pairs, using fixed-order per-thread partials of the natoms × 3 result.
     Never `omp atomic` (see the DFT XC reduction memory). It runs once per
     gradient, not once per iteration, so the φ_el-style cost is
     acceptable.
     - Unit gate: against the per-component `_compute_nuclear_deriv_A_elem`
       / `_C_xyz` with point charges (η = 0), which already exist and are
       already gated. Then against a finite difference of φ_el with a
       Gaussian charge.
   - **4d. Gates.**
     - PySCF per term. `solvent/grad/pcm.py` exposes `grad_nuc`,
       `grad_qv` and `grad_solver` separately. Compare each on C1 H2O2
       before comparing the total: when the total is the only check,
       partial cancellations hide which term is wrong (see the DH
       KS/PT2-halves lesson in Completion).
     - PySCF total (`make_grad_object`) on the four SWIG cases, extending
       `tests/pyscf/pcm_swig_references.py`. Set `conv_tol_grad` (see the
       PySCF `conv_tol_grad` memory).
     - Fourth-order FD of the energy to 1e-7 Ha/Bohr on C1 H2O2.
     - The forces sum to zero to roundoff. Torque is not a gate (see
       Orientation).
     - Mutation check: drop each of the three terms in turn and confirm
       that every gate fails.
   - **Lift in `io.cpp`:** `calculation gradient` only. ROHF and post-HF
     stay rejected. DH gradients keep their own solvent refusals.

5. **Geometry optimization and frequencies.** Lifting the gate is not enough
   here: the inner loops lose PCM without any error.
   `src/opt/geomopt.cpp` (`run_*` at ~95–99), `src/freq/hessian.cpp`
   (~84–88) and the post-optimization symmetry SCF in `src/hf_driver.cpp`
   (~1735) all call `run_*` without a PCM state. Each displaced geometry
   needs its own `build_pcm_state`, at 0.08–0.4 s per build on the acene
   ladder. Put the build in `rebuild_basis_dependent_state` or next to it,
   so all three loops share one path. Check that the DFT geomopt/freq
   callbacks rebuild through `prepare` (`prepared.pcm`) at every geometry.
   - Gate: water in water optimized against PySCF's `geometric_solver`.
   - Gate: frequencies against a PySCF Hessian from finite differences of
     its analytic gradient.
   - Gate: a geomopt case that asserts the final energy is the PCM energy.
     This catches the silent loss of PCM in the inner loop.

6. **Spherical basis. Done (2026-11-08).** `evaluate_pcm_reaction_field`
   lifts the working density with `BasisFunctions::to_cartesian` before
   φ_el, and returns V_rxn through `to_working`. The disabled-state zero
   matrix is now sized by `working_nbasis()`. Every caller (the RHF/UHF SCF,
   the DFT SCF and the ESP reaction-field report) already passes and expects
   working-basis matrices, so nothing changed at the call sites. Both
   refusals are removed. In Cartesian mode both lifts are the identity, so
   the Cartesian output is unchanged.

   | Case (6-31G*, `cart=False`) | PySCF 2.13.0 E / Eh | Tesserae |
   |---|---|---|
   | `water_rhf_pcm_swig_spherical_631gd` | −76.0158401249 | 542 |
   | `water_cation_uhf_pcm_swig_spherical_631gd` | −75.6750550413 | 542 |
   | `water_rks_pbe_pcm_swig_spherical_631gd` | −76.3201409489 | 542 |

   Planck matches each case to every printed digit, which covers the RHF,
   UHF and DFT call sites. The water cation lands 0.076 Eh above its
   Cartesian counterpart, in both codes, from the hcore guess. This is the
   open-shell hcore saddle (vault Open Work), not PCM. When step 4 lands,
   `pcm_gradient` must lift P_total once at entry, the same way.

   **Cleanup while here:** the SAO-suppression comment in
   `src/hf_driver.cpp` (~841) still describes the Fibonacci cavity. The
   reason (the cavity is not symmetry-adapted) still holds for SWIG.

**Recommended order:** 4a–4b, 4c, 4d, then 5 (6 is done). 4b and 4c are
independent of each other, and each has its own unit gate, so a failing
total in 4d can be traced to a term.

## Not in this scope

IEF-PCM, SS(V)PE, COSMO's (ε−1)/(ε+½) as an option, and non-electrostatic
terms (cavitation, dispersion-repulsion, as in SMD). Also out: PCM for TDDFT
(linear-response or state-specific), for post-HF, and for ROHF. Restoring
SAO blocking under PCM is out too: the cavity is not symmetry-adapted
(`docs/PCM_ROTATIONAL_INVARIANCE_SCOPE.md`).
