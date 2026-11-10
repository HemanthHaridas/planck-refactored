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
amplitude storage, which is deferred to the RI-DH work (see "Deferred to
RI-DH" below). RKS no longer carries dense reference adapters
(step 1); the tests rebuild dense views in `tests/dh_dense_oracle.h`.

| Stage | RKS | UKS | Status |
|---|---|---|---|
| AO derivative contraction | streamed callbacks | same | shared (`16f4047b`) |
| AO pair density | bounded AO tiles | same | shared (`16f4047b`) |
| Z-vector | `solve_packed_zvector` (GMRES) | same | shared (`79fe31c7`) |
| MO integrals | five Eq. 40 sectors | five sectors per spin pair | shared (step 3) |
| Amplitudes | source-backed pair tiles | resident `t2_aa/ab/bb` | deferred to RI-DH |
| Dense reference adapters | none (test-side oracle) | none | step 1 |

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
  `tests/dh_dense_oracle.h` (dense derivatives and their callbacks, the RKS
  Eq. 46/47 tensors, the reference contract), `tests/udh_dense_oracle.h`
  (UKS two-particle tensors), `tests/dh_zvector_oracle.h`
  (Eq. 27 Hessians) and `tests/udh_mo_sector_oracle.h` (full MO tensors and
  their sector slices).

## Next steps

Steps 0 to 3 are done (see the record below), and step 4 is deferred to the
RI-DH work. Final acceptance (the full `tests/udh_molecular_audit.py`, V4 grid,
`--grid-policy g4-3-v4 --radial-points 135`) ran on 2026-11-10 and is blocked
by one fixture.

Passed: H2O2⁺, H2⁺ (zero beta), HO2, water cation and water triplet in
STO-3G, and the water cation in 6-31G (worst 1.1e-9 Ha/Bohr), all within
5e-8; closed-shell UKS = RKS to 8.5e-14 Eh and 1.1e-11 Ha/Bohr; RKS
full-coordinate FD on water (9.3e-9) and C1 H2O2 (1.2e-8). Three rejection
expectations in the audit were stale and are fixed: production now computes
UKS DH gradients (no longer a rejection case), the spherical refusal comes
from the spherical-DFT guard, and the solvent case lacked a dielectric, so
the input parser refused before the solvent guard was reached.

**Blocker: CN radical** (`cn_radical_asymmetric_b2plyp_sto3g`, ⟨S²⟩ 0.886).
Only the PT2 half along the C–N bond misses; the KS half is within 4e-9.

| FD step (Bohr) | 2e-4 | 1e-4 | 5e-5 | 2.5e-5 |
|---|---|---|---|---|
| bond-axis error, V4 grid | 2.2e-6 | 2.1e-7 | 8.7e-8 | 8.1e-8 |
| same, 590-point unpruned grid | | | −4.5e-7 | 2.2e-7 |

At large steps it is stencil truncation. Below that, on the qualified grid, a
step-independent ~8e-8 remains: systematic, equal and opposite on the two
atoms. The unpruned grid does not settle it: the error changes sign between
steps, so FD along the bond is not resolving a smooth energy there. Not
caused by this scope's work: step 1 touched no UKS code, step 3 left the CN
gradient bitwise and step 2 moved it by 5.9e-14, and no earlier CN FD pass is
recorded. Next: check the stability of the CN UHF state and repeat the FD
with tighter SCF convergence; decide whether CN belongs in the acceptance set
if it is a near-instability rather than a gradient defect. Then rewrite this
file as an answer doc.

## Record of landed steps

### Step 1. Dense adapters removed (2026-11-10)

Production never reached the dense path: `build_dh_eq33_derivative_integrals`
always set the direct callbacks, so the storage switch always chose
`ContractionOnly`. Removed from `src/dft/dh_pt2_gradient.{h,cpp}`:

- the dense `hamiltonian_ao`/`overlap_ao`/`eri_ao` fields of
  `DHEq33NonXCDerivatives`, the dense branch of `build_dh_eq33_non_xc_gradient`
  and `Eq33NonXCAccumulator`. The Eq. 47 tile context is now a required
  argument;
- `DHGammaStorage`, the six AO tensors of `DHEq46_47TwoParticleDensity`
  (it keeps `D`, `P`, `a_x`), `eri_eightfold_symmetrize` and
  `materialized_eq47_bytes`;
- `build_dh_eq47_nonseparable_raw_ao`, `for_each_dh_eq47_nonseparable_raw_ao_tile`,
  `dh_eq47_penultimate` and the `transform_tile` field of
  `DHEq47TiledExecution`;
- the one-argument `build_dh_gradient_driver_contract` and the `mo_eri`
  overload of `build_dh_stationary_products`. The remaining contract overload
  fails with a named error when the workspace lacks the direct callbacks,
  instead of falling back to a dense path;
- `build_dh_ks_fixed_density_fock_derivative` and its struct, which had no
  caller and read the dense derivative fields.

`build_dh_molecular_eri_blocks` and `for_each_dh_eq47_nonseparable_raw_ao_block`
stay: the first slices a dense MO tensor into sectors, the second is the
production pair stream with a one-channel wrapper.

Tests: `tests/dh_dense_oracle.h` holds the spin-agnostic dense views, rebuilt
from production code: `DenseDerivatives`, `callbacks_from_dense`, the
stream-to-dense `stream_channel`, `dense_gamma` (the six RKS tensors),
`dense_non_xc`, and `stationary_from_mo_eri` / `reference_driver_contract` in
place of the deleted overloads. `tests/udh_dense_oracle.h` builds on it.
`dft_coulomb_response.cpp`, `dh_pt2_amplitude_density.cpp`,
`dh_o5_backtransform.cpp`, `udh_two_particle_density.cpp` and
`udh_stationary_transfer.cpp` moved to it. `dh_o5_backtransform.cpp` now checks
the block stream against its eighth-order literal oracle instead of the
deleted direct backtransform. The Python contracts (`test_dh_o2_workspace.py`,
`test_dh_o4_contractions.py`, `test_dh_o5_backtransform.py`) now assert that no
dense storage mode, builder or symmetrizer remains.

Production shrinks by 450 lines (`dh_pt2_gradient.{h,cpp}`: 504 removed, 54
added). Results:

- RKS B2PLYP/STO-3G gradient and energy on water and C1 H2O2 identical to
  every printed digit before and after.
- All 29 `planck-dh-*`, `planck-udh-*` and `planck-dft-coulomb-response`
  ctest entries pass, as do the 99 DH and UDH Python contract tests, and the
  UKS regression case `h2o2_cation_uks_b2plyp_v4_gradient_sto3g`.
- Mutation: halving the Eq. 46 exchange factor in the production source fails
  both `planck-dft-coulomb-response` and `planck-dh-pt2-amplitude-density`,
  so the rewritten dense-oracle comparisons are live.

### Step 0. Unread UMP2 amplitude copy (2026-11-10, `bea7dd8b`)

`apply_ump2_result` (`src/post_hf/mp2.cpp`) copied `t2_aa/ab/bb` into
`calculator._mp2_result.ump2_t2_{aa,ab,bb}` on every UMP2 run, and nothing in
`src/` or `tests/` read them: one full extra amplitude set per run. The copies
and the three `types.h` fields are deleted. This is the "deliberate copy" item
the former step 4 listed, settled by deletion; it saves memory for every UMP2 caller, not
only double hybrids.

The 11 UMP2 and UKS double-hybrid regression cases pass.


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

## Deferred to RI-DH: UKS amplitude storage (former step 4)

Decided 2026-11-10: not done under this scope. Its benefit only shows at sizes
where the AO ERI tensor is already the larger object, and RI-DH is what removes
that tensor, so the two belong together. The design below is kept for that
work.

`UMP2Result` has no tile source today. Its `t2_aa/ab/bb` are resident vectors, and
`RMP2T2TileSource` cannot be reused as it stands, because it packs on the
closed-shell pair symmetry `t_ij^ab = t_ji^ba`, which the `ab` block does not have.

The HF UMP2 analytic gradient (`mp2_gradient.cpp`) also reads the resident
`t2_aa/ab/bb`, so the tile source is added beside them, not in their place.
This follows the RKS precedent, where `RMP2Result` carries `t2_source` beside
`t2`/`t2_pair_packed`: only the double-hybrid caller asks for tiles and leaves
the resident vectors empty. Moving the HF UMP2 gradient to tiles is out of
scope.

- Add a spin-block tile source (ordered `(i,a)`/`(j,b)` tiles, no pair symmetry
  assumed), scratch-backed as RKS is. `mp2_ump2.cpp` writes it after
  `canonical_kernel` when the caller requests it, so peak memory during the
  UMP2 kernel does not change.
- `UDHPT2Amplitudes` becomes a view over those tiles. Random-access consumers
  become tile loops: `build_udh_pt2_dprime`, `raw_channel`, `pair_channel`,
  `sector()` and the channel fill (`udh_driver.cpp`, `udh_pt2_gradient.cpp`).
- The energy contract also reads the resident vectors and needs the same
  treatment. (The `calculator._mp2_result` copies went in step 0.)

It touches every UKS amplitude consumer and is the only step that changes
UMP2 itself.

Gate: tile reads equal the resident amplitudes exactly; UMP2 energies unchanged;
UKS gradient unchanged within 1e-9 Ha/Bohr; peak RSS measured on a fixture large
enough to show the saving.

## Out of scope

- Merging the `udh_*` and `dh_*` files into one spin-general pipeline. Decided:
  both spins share every engine (pair generator, derivative contraction, MO
  sector builder, Z-vector solver) and keep separate thin front ends for setup
  and spin-specific factors.
- UKS double-hybrid geometry optimization and frequencies. Open once this lands.
- The per-AO-element `_compute_eri_deriv_elem` traversal and its hardcoded
  Obara-Saika call, which both spins already share.
- Spherical, solvent, range-separated and RI double-hybrid gradients.
