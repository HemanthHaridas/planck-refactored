# UKS double-hybrid gradient: one derivative architecture

Scope for ongoing work. When it lands, rewrite this file as the answer to its
question, or fold the answer into `docs/DH_IMPLEMENTATION.md` and delete it.

**Question.** The production UKS double-hybrid gradient failed. How does it reach
a working state with one derivative architecture shared with RKS, and no dense
path beside the streamed one?

## Current result (2026-11-10)

The UKS double-hybrid gradient works and passes finite differences. It shares
every engine with RKS: the AO derivative contraction, the AO pair density, the
Z-vector solver and the Eq. 40 MO-integral sectors. What still differs is the
amplitude storage (step 4), and RKS still carries dense reference adapters that
only tests use (step 1).

| Stage | RKS | UKS | Status |
|---|---|---|---|
| AO derivative contraction | streamed callbacks | same | shared (`16f4047b`) |
| AO pair density | bounded AO tiles | same | shared (`16f4047b`) |
| Z-vector | `solve_packed_zvector` (GMRES) | same | shared (`79fe31c7`) |
| MO integrals | five Eq. 40 sectors | five sectors per spin pair | shared (step 3) |
| Amplitudes | source-backed pair tiles | resident `t2_aa/ab/bb` | step 4 |
| Dense reference adapters | kept for test oracles | none | step 1 |

What the gradient is validated against:

| Gate | Result |
|---|---|
| UKS fourth-order FD, V4 grid | C1 H2O2⁺ 4.3e-9, HO2 1.8e-9, asymmetric water cation 9.1e-10 Ha/Bohr (limit 5e-8); translation ≤ 3.8e-13 |
| Closed-shell reduction | `scf_type uhf` on closed-shell C1 H2O2, V4 grid: within 2.4e-11 Ha/Bohr of RKS, same printed energy |
| RKS unchanged | Water and C1 H2O2 B2PLYP/STO-3G gradient and energy bitwise identical through every landed step |
| Mutations | `ab` weight `c`, exchange factor `-½a_x`, dropped `bb`, wrong spin frame on ν: each fails `planck-udh-two-particle-density` |
| Regression | `h2o2_cation_uks_b2plyp_v4_gradient_sto3g`, `extended`, about 45 s |

The FD numbers predate steps 2 and 3. Step 2 moved every UKS gradient by at
most 6.5e-13 Ha/Bohr, far inside the 5e-8 limit, and step 3 moved none of them
(bitwise), so they still hold; the full audit is rerun at final acceptance.

Shared so far:

- `for_each_dh_pair_ao_block` streams bounded AO blocks summed over a list of
  `DHPairChannel`s, so a UKS gradient is one derivative traversal, not three.
  `DHEq33TwoParticleSource` carries the separable terms and pair channels for
  either spin, and `build_dh_eq33_direct_non_xc_gradient` is the one assembly.
  UKS builds its source with `make_udh_two_particle_source`; its six `n⁴`
  tensors are deleted.
- `solve_packed_zvector` (`src/dft/dh_zvector.h`) is the one Z-vector solver:
  matrix-free GMRES with no dense fallback and a fresh final residual. RKS and
  UKS supply only their packing and action. On C1 H2O2⁺ this takes 46 response
  actions where the deleted dense UKS solve took 119 (details below).
- `build_dh_eq40_mo_eri_blocks_direct` builds the five Eq. 40 MO-integral
  sectors for both spins; UKS no longer forms three full `n⁴` MO tensors.
- Test oracles rebuild the dense objects from the production code:
  `tests/udh_dense_oracle.h` (two-particle tensors), `tests/dh_zvector_oracle.h`
  (Eq. 27 Hessians) and `tests/udh_mo_sector_oracle.h` (full MO tensors and
  their sector slices).

## Next steps

The steps are independent; each lands on its own and keeps the gradient
working. Step 1 is next: it is mostly a test refactor with no user-visible
gain, but it is lower risk than step 4, which goes last for the reasons given
there.

### Step 1. Remove the dense adapters (M, mostly test code)

Production deletions:

- The `hamiltonian_ao`, `overlap_ao` and `eri_ao` fields of
  `DHEq33NonXCDerivatives`, the dense branch of `build_dh_eq33_non_xc_gradient`,
  and `Eq33NonXCAccumulator`.
- `DHGammaStorage` and the six materialized fields of
  `DHEq46_47TwoParticleDensity`, with `eri_eightfold_symmetrize` and
  `materialized_eq47_bytes`.
- `build_dh_eq47_nonseparable_raw_ao`, `for_each_dh_eq47_nonseparable_raw_ao_tile`,
  `dh_eq47_penultimate`, and the unused `transform_tile`.
- The `dense_blocks` route through `build_dh_molecular_eri_blocks` in the
  reference overload of `build_dh_gradient_driver_contract`, which then has no
  production role. Its tests keep it as a test helper.

Test work is the bulk of the step. About 80 references to the materialized fields
sit in `dft_coulomb_response.cpp`, `dh_pt2_amplitude_density.cpp` and
`dh_o5_backtransform.cpp`. They move to a dense view rebuilt from the production
stream, the pattern `tests/udh_dense_oracle.h` already uses. Update
`test_dh_o2_workspace.py` and `test_dh_o5_backtransform.py`, and rewrite
`docs/DH_IMPLEMENTATION.md` §7 and §9 rule 7, which permit retained dense
adapters.

Gate: RKS gradient bitwise unchanged on water and C1 H2O2; all contract tests pass.

### Step 4. Amplitude storage (M–L)

`UMP2Result` has no tile source today. Its `t2_aa/ab/bb` are resident vectors, and
`RMP2T2TileSource` cannot be reused as it stands, because it packs on the
closed-shell pair symmetry `t_ij^ab = t_ji^ba`, which the `ab` block does not have.

- Add a spin-block tile source (ordered `(i,a)`/`(j,b)` tiles, no pair symmetry
  assumed), scratch-backed as RKS is. `mp2_ump2.cpp` writes it after
  `canonical_kernel`, so peak memory during the UMP2 kernel does not change.
- `UDHPT2Amplitudes` becomes a view over those tiles. Random-access consumers
  become tile loops: `build_udh_pt2_dprime`, `raw_channel`, `pair_channel`,
  `sector()` and the channel fill.
- The energy contract and `calculator._mp2_result` copies also read the resident
  vectors, and need the same treatment or a deliberate copy.

Do this last. It touches every UKS amplitude consumer and is the only step that
changes UMP2 itself, and its benefit only shows at sizes where the AO ERI tensor
is already the larger object.

Gate: tile reads equal the resident amplitudes exactly; UMP2 energies unchanged;
UKS gradient unchanged within 1e-9 Ha/Bohr; peak RSS measured on a fixture large
enough to show the saving.

### Final acceptance

After step 4, run the full `tests/udh_molecular_audit.py` (all `uks_u0` fixtures,
the larger-basis water cation, closed-shell reduction and the RKS regression),
then rewrite this file as an answer doc.

## Record of landed steps

### Step 3. MO integrals by sector (2026-11-10)

`sector()` in `udh_pt2_orbital.cpp` replaces one index of `(ia|jb)` at a time by
a general MO, so it reads exactly the five RKS sectors for each spin pair:
`(ov|ov)`, `(vv|ov)`, `(oo|ov)`, `(ov|vv)` and `(ov|oo)`, with electron 1 in the
first spin and electron 2 in the second. What changed:

- `build_dh_eq40_mo_eri_blocks_direct` takes separate occupied and virtual
  frames per electron and is declared in `driver.h`, so `udh_driver.cpp` calls
  it. RKS passes its one frame twice. An empty frame gives empty sectors.
- `DHMolecularERIBlocks` gains `n_occ_2`/`n_virt_2`; the typed RKS consumer
  checks them.
- `UDHFullMOIntegrals` (three full `n⁴` MO tensors) is replaced by
  `UDHMOIntegrals { DHMolecularERIBlocks aa,ab,bb; }`. `sector()` splits each
  general-MO loop into its occupied and virtual ranges, keeping the
  per-element accumulation order. Its input check is now sector sizes and
  finiteness; the old chemist-symmetry check on full tensors has nothing to
  act on.
- `UDHPT2DirectIntegrals` is still filled by a short reorder loop, not removed
  as first planned: it is laid out `[i][j][a][b]`, the `ovov` sector
  `[i][a][j][b]`.
- The four test files that build full MO tensors keep them as the independent
  oracle and slice sectors with `TestOracle::sectors`. The integral-mutation
  controls in `udh_pt2_orbital.cpp` now corrupt sectors (size, frame extent,
  Inf, NaN). `test_udh_pt2_orbital_contract.py` guards that no full UKS MO
  transform returns.

The conventional AO ERI tensor stays, for both spins. Removing it is the
separate RI-DH backend.

Results: gradient and energy are bitwise identical to step 2 on all six
`uks_u0` fixtures, the regression case and both RKS inputs (the per-element
accumulation order in `sector()` is unchanged, and the sector transforms give
the same values as the full ones). `planck-udh-pt2-orbital`,
`planck-udh-overlap-density`, `planck-udh-stationary-transfer`,
`planck-udh-zvector`, `planck-dh-pt2-amplitude-density` and
`planck-dft-coulomb-response` pass. Wall time is unchanged at STO-3G (C1 H2O2⁺
22.0 s to 20.5 s, within run-to-run noise).

The gain is memory. Computed from element counts, not measured:

| System (Cartesian AOs) | Three full MO tensors | Sectors | Ratio |
|---|---|---|---|
| C1 H2O2⁺ / STO-3G (12) | 0.47 MiB | 0.13 MiB | 3.5 |
| water cation / cc-pVTZ (65) | 409 MiB | 48 MiB | 8.6 |
| H2O2⁺ / cc-pVTZ (about 130) | 6.4 GiB | 0.71 GiB | 9.0 |

The AO tensor itself (one `n⁴`, 2.1 GiB for the last row) is now the largest
UKS object, and stays until RI-DH.


### Step 2. One Z-vector solver (2026-11-10, `79fe31c7`)

`solve_packed_zvector` takes a packed action, a right-hand side and an
orbital-gap diagonal. `solve_dh_zvector` (RKS, `a*no+i`) and `solve_udh_zvector`
(UKS, alpha then beta) supply only their packing and action. An empty pair
space returns an empty `z` before `gmres`, which rejects `n == 0`.

Deleted from production: the dense UKS Jacobian with its SVD, pivoted QR and
column-wise transpose residual; the `jacobian`, `adjoint_max_abs`,
`reciprocal_condition`, `rank` and `residual_ai` result fields;
`UDHZVectorOptions` (UKS now takes `DHZVectorOptions`); and
`DHZVectorBackend::DenseReference` with `dense_matrix_bytes`.

**Decision: the solver takes `A z = b`, not `A^T z = b`.** The old UKS solve
was `A^T z = -ℓ`; RKS was `A z = -L`. They agree because the physical joint KS
Hessian is self-adjoint. Production no longer checks that per run; the tests
do: `udh_zvector.cpp` (every audit fixture) and `udh_stationary_transfer.cpp`
check the oracle Jacobian is symmetric to 2e-12. A nonadjoint control pins the
convention: on such an operator the result satisfies `A z = -ℓ` and not
`A^T z = -ℓ`. What production gave up is the per-run adjointness and
condition-number check; a singular Hessian now surfaces as GMRES
non-convergence with no fallback, as it already did for RKS.

Results, `planck-dft` before and after on the same tree, V4 grid for UKS:

| Fixture | Pairs | GMRES iterations | Actions | Dense actions (1+2n) | max \|Δg\| (Ha/Bohr) |
|---|---|---|---|---|---|
| C1 H2O2⁺ (also the regression case) | 59 | 22 | 46 | 119 | 1.8e-13 |
| CN radical | 45 | 17 | 36 | 91 | 5.9e-14 |
| HO2 | 42 | 17 | 36 | 85 | 1.1e-13 |
| water cation | 22 | 11 | 24 | 45 | 1.1e-13 |
| water triplet | 18 | 12 | 26 | 37 | 6.5e-13 |
| H2⁺, zero beta | 1 | 0 | 2 | 3 | bitwise |
| RKS water, RKS C1 H2O2 | 10, 27 | 4, 12 | 10, 26 | n/a | bitwise |

Energies are unchanged in every case. The action count grows with the GMRES
iteration count rather than with the pair count, so the saving widens with
system size. One timing, C1 H2O2⁺: 40.5 s to 22.0 s total; the baseline shared
the CPU with a compile, so treat it as indicative only. RKS water GMRES `z`
matches the dense oracle to 2.8e-14.

### Shared streamed path (2026-11-09, `16f4047b`)

The defect it fixed: every UKS double-hybrid gradient stopped with
`UDH driver: invalid correction coordinate layout`. `d72742df` (2026-10-12) made
`build_dh_eq33_derivative_integrals` return only the two streamed callbacks, and
the dense UKS consumer then saw zero coordinates. The last recorded validation
before it was `039bd2df` (2026-10-09). The suite missed it because no
regression case ran this path, and the `planck-udh-molecular-runner` ctest entry
runs only the audit script's Python unit tests; the molecular audit itself is
run by hand.

Restoring the dense builder was ruled out: it would keep two derivative
architectures, which is what caused the break.

## Design

RKS and UKS differ only in where the density coefficients come from, so one
engine takes both.

| Term | RKS | UKS |
|---|---|---|
| Coulomb, Eq. 34 / 46 | `D(μν) P(κτ)` | `D_tot(μν) P_tot(κτ)` |
| Exchange | one term, factor `-½ a_x`, total densities | two terms, factor `-a_x`, spin densities |
| Pair, Eq. 35 / 47 | one channel, `(1+δ_ij) t̃`, one MO frame | three channels `aa`, `ab`, `bb`, weights `c`, `2c`, `c`, two frames |
| Eq. 40 MO sectors | one frame for both electrons | one block per spin pair, electron frames per spin |
| Z-vector | `a*no+i` packing | alpha then beta packing, same solver |

For a closed shell `D_σ = D/2` and `P_σ = P/2`, so the two UKS exchange terms sum
to `-½ a_x D P`, the RKS form. Both pair layouts are `(i,a,j,b)` →
`C_i C_a C_j C_b`.

## Out of scope

- Merging the `udh_*` and `dh_*` files into one spin-general pipeline. Decided:
  both spins share every engine (pair generator, derivative contraction, MO
  sector builder, Z-vector solver) and keep separate thin front ends for setup
  and spin-specific factors.
- UKS double-hybrid geometry optimization and frequencies. Open once this lands.
- The per-AO-element `_compute_eri_deriv_elem` traversal and its hardcoded
  Obara-Saika call, which both spins already share.
- Spherical, solvent, range-separated and RI double-hybrid gradients.
