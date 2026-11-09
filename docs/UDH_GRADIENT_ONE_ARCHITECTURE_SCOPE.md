# UKS double-hybrid gradient: one derivative architecture

Scope for ongoing work. When it lands, rewrite this file as the answer to its
question, or fold the answer into `docs/DH_IMPLEMENTATION.md` and delete it.

**Question.** The production UKS double-hybrid gradient failed. How does it reach
a working state with one derivative architecture shared with RKS, and no dense
path beside the streamed one?

## Current status (2026-11-09)

The fix is accepted: the UKS gradient runs and passes finite differences on all
three fixtures. The shared streamed path is not yet committed. The steps below
remove the rest of the second architecture.

| Gate | Result |
|---|---|
| Production UKS gradient | Runs on C1 H2O2⁺ B2PLYP/STO-3G; forces sum to about 1e-12 Ha/Bohr |
| RKS unchanged | Water and C1 H2O2 B2PLYP/STO-3G gradient and energy bitwise identical to the previous build |
| Contract tests | `planck-dft-coulomb-response`, `planck-dh-pt2-amplitude-density`, `planck-udh-two-particle-density`, `planck-udh-stationary-transfer` pass |
| Closed-shell reduction | `scf_type uhf` on closed-shell C1 H2O2, V4 grid: within 2.4e-11 Ha/Bohr of RKS, same printed energy |
| Mutations | `ab` weight `c`, exchange factor `-½a_x`, dropped `bb`, wrong spin frame on ν: each fails `planck-udh-two-particle-density` |
| UKS fourth-order FD, V4 grid | C1 H2O2⁺ 4.3e-9, HO2 1.8e-9, asymmetric water cation 9.1e-10 Ha/Bohr (limit 5e-8); translation ≤ 3.8e-13 |
| Regression | `h2o2_cation_uks_b2plyp_v4_gradient_sto3g` registered, `extended`, about 45 s |

Built so far:

- `for_each_dh_pair_ao_block` streams bounded AO blocks summed over a list of
  `DHPairChannel`s, so a UKS gradient is one derivative traversal, not three.
- `DHEq33TwoParticleSource` carries the separable terms and pair channels for
  either spin. `contract_dh_eq33_two_electron_direct` takes only this.
- `build_dh_eq33_direct_non_xc_gradient` is the one production assembly, called
  by both spins.
- UKS builds its source with `make_udh_two_particle_source`. Its six `n⁴` tensors,
  `UDHERICoefficients` and `build_udh_non_xc_gradient` are deleted.
- `tests/udh_dense_oracle.h` rebuilds the dense tensors from the production source
  for the channel-level tests.

Where the two paths still differ:

| Stage | RKS | UKS | Step |
|---|---|---|---|
| AO derivative contraction | streamed callbacks | same | done |
| AO pair density | bounded AO tiles | same | done |
| Dense reference adapters | kept for test oracles | n/a | 1 |
| Z-vector | matrix-free GMRES | dense joint Jacobian, pivoted QR | 2 |
| MO integrals | five Eq. 40 sectors | three full `n⁴` transforms | 3 |
| Amplitudes | source-backed pair tiles | resident `t2_aa/ab/bb` | 4 |

## Remaining steps

Sizes are estimates. These steps are independent of each other, so the order below
is by payoff against risk. Each step lands on its own and keeps the gradient
working.

### 1. Remove the dense adapters (M, mostly test code)

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

### 2. One Z-vector solver (S–M)

Extract the packed core of `solve_dh_zvector` (GMRES, no fallback, fresh final
residual) so it takes an action, a right-hand side and a diagonal. RKS and UKS
each supply only their packing and action.

- UKS packs α then β, as `solve_udh_zvector` already does. The diagonal is the
  orbital-energy gap from `fock_mo`.
- The empty pair space (`n = 0`, the H2⁺ zero-beta fixture) returns before the
  solver, because `gmres` rejects `n == 0`.
- Delete the dense Jacobian, SVD and pivoted QR, and the `jacobian`,
  `adjoint_max_abs`, `reciprocal_condition` and `rank` result fields.
- Delete `DHZVectorBackend::DenseReference`. Its oracle moves to the tests, which
  build columns from the public `apply_dh_eq27_hessian` and
  `apply_udh_eq27_hessian`. `udh_stationary_transfer.cpp` and `udh_zvector.cpp`
  read `z.jacobian` directly and need this.

What production gives up: the per-run joint-adjointness and condition-number
checks. A near-singular Hessian then surfaces as GMRES non-convergence, with no
fallback, as it already does for RKS. The adjointness check stays in the tests.

Gate: `z` matches the test-side dense solve to 1e-10; UKS gradient unchanged within
1e-9 Ha/Bohr on all six `uks_u0` fixtures, whose iteration counts get recorded.

### 3. MO integrals by sector (M)

`sector()` in `udh_pt2_orbital.cpp` replaces one index of `(ia|jb)` at a time by a
general MO, so it reads exactly the five RKS sectors for each spin block:
`(ov|ov)`, `(vv|ov)`, `(oo|ov)`, `(ov|vv)` and `(ov|oo)`.

- Generalize `build_dh_eq40_mo_eri_blocks_direct` to separate occupied and virtual
  frames per electron, and move it out of `driver.cpp` so `udh_driver.cpp` can
  call it. `DHMolecularERIBlocks` gains the second electron's extents. RKS passes
  the same frame twice.
- Allow empty spin frames, which it currently rejects.
- `UDHFullMOIntegrals` becomes three sector blocks (`aa`, `ab`, `bb`). `sector()`
  splits each general-MO loop into its occupied and virtual ranges.
  `UDHPT2DirectIntegrals` is the `ovov` sector, so its extraction loop goes.
- Four test files build `UDHFullMOIntegrals` fixtures. They slice sectors from
  their full tensors in test code.

The conventional AO ERI tensor stays, for both spins. Removing it is the separate
RI-DH backend.

Gate: sector slices match the full transform to 1e-12; UKS gradient unchanged
within 1e-9 Ha/Bohr; RKS bitwise unchanged.

### 4. Amplitude storage (M–L)

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

## Design

RKS and UKS differ only in where the density coefficients come from, so one
engine takes both. Restoring the dense builder was ruled out: it would keep two
derivative architectures, which is what caused the break.

| Term | RKS | UKS |
|---|---|---|
| Coulomb, Eq. 34 / 46 | `D(μν) P(κτ)` | `D_tot(μν) P_tot(κτ)` |
| Exchange | one term, factor `-½ a_x`, total densities | two terms, factor `-a_x`, spin densities |
| Pair, Eq. 35 / 47 | one channel, `(1+δ_ij) t̃`, one MO frame | three channels `aa`, `ab`, `bb`, weights `c`, `2c`, `c`, two frames |

For a closed shell `D_σ = D/2` and `P_σ = P/2`, so the two UKS exchange terms sum
to `-½ a_x D P`, the RKS form. Both pair layouts are `(i,a,j,b)` →
`C_i C_a C_j C_b`.

## The defect

Every UKS double-hybrid gradient stopped with
`UDH driver: invalid correction coordinate layout`. `d72742df` (2026-10-12) made
`build_dh_eq33_derivative_integrals` return only the two streamed callbacks, and
the dense UKS consumer then saw zero coordinates. The last recorded validation is
`039bd2df` (2026-10-09).

The suite missed it because no regression case ran this path, and the
`planck-udh-molecular-runner` ctest entry runs only the audit script's Python
unit tests. The molecular audit itself is run by hand.

## Out of scope

- Merging the `udh_*` and `dh_*` files into one spin-general pipeline. Decided:
  both spins share every engine (pair generator, derivative contraction, MO
  sector builder, Z-vector solver) and keep separate thin front ends for setup
  and spin-specific factors.
- UKS double-hybrid geometry optimization and frequencies. Open once this lands.
- The per-AO-element `_compute_eri_deriv_elem` traversal and its hardcoded
  Obara-Saika call, which both spins already share.
- Spherical, solvent, range-separated and RI double-hybrid gradients.
