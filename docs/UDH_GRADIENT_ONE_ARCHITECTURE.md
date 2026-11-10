# UKS double-hybrid gradient: one derivative architecture

**Question.** The production UKS double-hybrid gradient failed. How does it
reach a working state with one derivative architecture shared with RKS, and
no dense path beside the streamed one?

**Answer (2026-11-10).** RKS and UKS run the same four engines: the streamed
AO derivative contraction, the bounded AO pair-density stream, one
matrix-free GMRES Z-vector solver, and one Eq. 40 MO-sector builder. Each spin
treatment keeps only a thin front end that supplies its density
coefficients, pair channels and packing. Production has no dense path at
all: no `n_ao⁴` two-particle tensor, no dense derivative arrays, no dense
Z-vector Hessian, no full MO-ERI tensor. The dense objects exist only in the
tests, rebuilt from the production producers. The full molecular audit
passes on every fixture (below).

What still differs between the spins is amplitude storage: UKS keeps its
`t2_aa/ab/bb` resident. That is deferred to the RI-DH work (last section).

## Architecture

| Stage | RKS | UKS | Shared through |
|---|---|---|---|
| AO derivative contraction | streamed callbacks | same | `build_dh_eq33_direct_non_xc_gradient` |
| AO pair density | bounded AO blocks | same | `for_each_dh_pair_ao_block` over `DHPairChannel`s |
| Z-vector | GMRES | same | `solve_packed_zvector` (`src/dft/dh_zvector.h`) |
| Eq. 40 MO integrals | five sectors | five sectors per spin pair | `build_dh_eq40_mo_eri_blocks_direct` |
| Amplitudes | source-backed pair tiles | resident `t2_aa/ab/bb` | deferred to RI-DH |
| Dense reference objects | none | none | test-side only |

RKS and UKS differ only in where the density coefficients come from:

| Term | RKS | UKS |
|---|---|---|
| Coulomb, Eq. 34 / 46 | `D(μν) P(κτ)` | `D_tot(μν) P_tot(κτ)` |
| Exchange | one term, factor `-½ a_x`, total densities | two terms, factor `-a_x`, spin densities |
| Pair, Eq. 35 / 47 | one channel, `(1+δ_ij) t̃`, one MO frame | channels `aa`, `ab`, `bb`, weights `c`, `2c`, `c`, two frames |
| Eq. 40 MO sectors | one frame for both electrons | one block per spin pair, electron frames per spin |
| Z-vector | `a*no+i` packing | alpha then beta packing |

For a closed shell `D_σ = D/2` and `P_σ = P/2`, so the two UKS exchange terms
sum to `-½ a_x D P`, the RKS form. Both pair layouts are `(i,a,j,b)` →
`C_i C_a C_j C_b`.

`DHEq46_47TwoParticleDensity` holds only `D`, `P` and `a_x`, and
`DHEq33NonXCDerivatives` holds only the two direct callbacks. A contract built
from a workspace without them fails with a named error rather than falling
back. Test oracles rebuild the dense objects from production code:
`tests/dh_dense_oracle.h` (dense derivatives and their callbacks, the RKS Eq.
46/47 tensors, the stationary products from a dense MO tensor, the reference
contract), `tests/udh_dense_oracle.h` (UKS two-particle tensors, on top of the
former), `tests/dh_zvector_oracle.h` (Eq. 27 Hessians) and
`tests/udh_mo_sector_oracle.h` (full MO tensors and their sector slices).

## Validation

Full `tests/udh_molecular_audit.py --grid-policy g4-3-v4 --radial-points 135`,
2026-11-10: **PASS, full suite.** Fourth-order centered energy FD at 1e-4 and
2e-4 Bohr in every Cartesian coordinate, limit 5e-8 Ha/Bohr.

| Fixture (B2PLYP) | Worst FD error | KS half | PT2 half | Rigid rotation | ⟨S²⟩ |
|---|---|---|---|---|---|
| CN radical | 7.8e-10 | 7.7e-10 | 4.7e-10 | 1.9e-8 | 1.314 |
| C1 H2O2⁺ | 4.3e-9 | 1.9e-9 | 4.1e-9 | 7.7e-8 | 0.754 |
| H2⁺, zero beta | 9.3e-12 | 9.3e-12 | 0 | 1.4e-10 | 0.750 |
| HO2 | 1.6e-9 | 1.4e-9 | 7.4e-10 | 2.8e-8 | 0.751 |
| water cation | 8.4e-10 | 8.5e-10 | 9.2e-11 | 1.1e-8 | 0.752 |
| water triplet | 7.2e-10 | 6.5e-10 | 8.5e-11 | 2.2e-8 | 2.002 |
| water cation, 6-31G | 1.1e-9 | 1.1e-9 | 7.8e-11 | 5.3e-7 | 0.751 |

STO-3G unless stated. Translated-input gradients agree to ≤ 3.7e-13 in every
case. Also in the audit:

- Closed-shell reduction: `scf_type uhf` on closed-shell water agrees with
  RKS to 8.5e-14 Eh and 1.1e-11 Ha/Bohr.
- RKS full-coordinate FD (`dh_cartesian_fd_audit.py`): water 3.0e-9 / 9.5e-9,
  C1 H2O2 4.2e-9 / 1.2e-8 at 1e-4 / 2e-4 Bohr.
- Refusals, each by the expected guard: spherical basis, ROKS, geometry
  optimization, frequencies, frozen core, RI, no stored T2, PT2 level shift
  and solvent.

Outside the audit: RKS B2PLYP gradients and energies on water and C1 H2O2
were identical to every printed digit through every step below; the
regression case `h2o2_cation_uks_b2plyp_v4_gradient_sto3g` (`extended`)
passes; four mutations (`ab` weight `c`, exchange factor `-½a_x`, dropped
`bb`, wrong spin frame on ν) each fail `planck-udh-two-particle-density`.

### The CN fixture sat on a saddle point

The first full audit failed on CN alone: a ~8e-8 Ha/Bohr PT2 residual along
the C–N bond that did not shrink with the FD step (2.2e-6, 2.1e-7, 8.7e-8,
8.1e-8 at 2e-4, 1e-4, 5e-5, 2.5e-5 Bohr), with the KS half clean. A denser
unpruned grid did not settle it; the error changed sign between steps.

The fixture started an open-shell UKS from `guess hcore`, which landed on a
saddle. `stability_check` on that state gives λ_min = −0.0173:

| CN start | KS energy (Eh) | λ_min | Notes |
|---|---|---|---|
| `hcore` (old fixture) | −91.2905395 | −0.0173 | saddle |
| `hcore` + `stability_follow` | −91.2953158 | +4.7e-10 | zero mode |
| `sad` (fixture now) | −91.3001560 | +0.0704 | ground state |

On the ground state CN passes at 7.8e-10. The other five fixtures reach the
same KS energy and a stable state (λ_min +0.015 to +0.54) from `hcore` and
`sad` alike. Lesson: an FD fixture's reference must be checked for stability
before a gradient miss is blamed on the gradient; this is the case the
open-shell `hcore` warning (`SCF::warn_open_shell_hcore_guess`) exists for.

## How it got here

All 2026-11-10 unless stated; each step left RKS bitwise unchanged.

**The defect (fixed by `16f4047b`, 2026-11-09).** Every UKS double-hybrid
gradient stopped with `UDH driver: invalid correction coordinate layout`.
`d72742df` (2026-10-12) made `build_dh_eq33_derivative_integrals` return only
the two streamed callbacks, and the dense UKS consumer then saw zero
coordinates. No regression case ran the path, and the
`planck-udh-molecular-runner` ctest entry runs only the audit script's unit
tests. Restoring the dense builder was ruled out: two derivative
architectures were the cause. Instead UKS moved onto the RKS stream
(`for_each_dh_pair_ao_block`, `DHEq33TwoParticleSource`,
`make_udh_two_particle_source`) and its six `n⁴` tensors were deleted.

**One Z-vector solver (`79fe31c7`).** `solve_packed_zvector` takes a packed
action, a right-hand side and an orbital-gap diagonal; RKS and UKS supply only
packing and action. Deleted: the dense UKS Jacobian with its SVD, pivoted QR
and transpose residual, and `DHZVectorBackend::DenseReference`. The solver
takes `A z = b`; the old UKS solve was `A^T z = -ℓ`. They agree because the
joint KS Hessian is self-adjoint, which the tests (not production) check to
2e-12, with a nonadjoint control pinning the convention.

| Fixture | Pairs | GMRES iterations | Actions | Old dense actions (1+2n) | max \|Δg\| |
|---|---|---|---|---|---|
| C1 H2O2⁺ | 59 | 22 | 46 | 119 | 1.8e-13 |
| CN radical | 45 | 17 | 36 | 91 | 5.9e-14 |
| HO2 | 42 | 17 | 36 | 85 | 1.1e-13 |
| water cation | 22 | 11 | 24 | 45 | 1.1e-13 |
| water triplet | 18 | 12 | 26 | 37 | 6.5e-13 |
| H2⁺, zero beta | 1 | 0 | 2 | 3 | bitwise |

Actions grow with GMRES iterations, not pair count, so the saving widens with
size. (The CN row is from the saddle-point reference.)

**MO integrals by sector.** `sector()` (`udh_pt2_orbital.cpp`) reads exactly
the five RKS sectors per spin pair; `build_dh_eq40_mo_eri_blocks_direct` takes
an occupied and virtual frame per electron, and `UDHMOIntegrals` holds three
`DHMolecularERIBlocks`. Bitwise on all fixtures. Memory, from element counts:

| System (Cartesian AOs) | Three full MO tensors | Sectors | Ratio |
|---|---|---|---|
| C1 H2O2⁺ / STO-3G (12) | 0.47 MiB | 0.13 MiB | 3.5 |
| water cation / cc-pVTZ (65) | 409 MiB | 48 MiB | 8.6 |
| H2O2⁺ / cc-pVTZ (about 130) | 6.4 GiB | 0.71 GiB | 9.0 |

The AO ERI tensor (2.1 GiB for the last row) is now the largest UKS object.

**Unread UMP2 amplitude copy (`bea7dd8b`).** `apply_ump2_result` copied
`t2_aa/ab/bb` into `calculator._mp2_result` on every UMP2 run and nothing read
them. Deleted, for every UMP2 caller.

**Dense RKS adapters removed (`21e2b864`).** Production never reached the
dense path (the driver always sets the callbacks). Removed: the dense
derivative fields and branch, `DHGammaStorage` and the six AO tensors,
`eri_eightfold_symmetrize`, the direct Eq. 47 backtransform and its mu-tile
stream, the one-argument reference contract and `mo_eri` stationary overload,
and the uncalled `build_dh_ks_fixed_density_fock_derivative`. Net −450 lines.
Halving the Eq. 46 exchange factor fails the rewritten oracle comparisons.

**Audit repaired.** Three refusal expectations predated production support
and aborted the full audit (production now computes the gradient; the
spherical refusal text changed; the solvent case lacked a dielectric, so the
parser refused before the solvent guard). The CN fixture moved to
`guess sad` (above).

## Deferred to RI-DH: UKS amplitude storage

Decided 2026-11-10: not done here. The saving only shows where the AO ERI
tensor is already the larger object, and RI-DH removes that tensor, so they
belong together. Design kept for that work:

- `RMP2T2TileSource` cannot be reused: it packs on the closed-shell pair
  symmetry `t_ij^ab = t_ji^ba`, which the `ab` block lacks. Add a spin-block
  tile source (ordered `(i,a)`/`(j,b)` tiles, no pair symmetry),
  scratch-backed, written by `mp2_ump2.cpp` after `canonical_kernel` only when
  the caller asks, so the UMP2 kernel's peak does not change.
- The HF UMP2 gradient (`mp2_gradient.cpp`) also reads the resident blocks,
  so the tile source goes beside them, as `RMP2Result::t2_source` sits beside
  `t2`; only the double-hybrid caller leaves them empty.
- `UDHPT2Amplitudes` becomes a view over the tiles; `build_udh_pt2_dprime`,
  `raw_channel`, `pair_channel`, `sector()`, the channel fill and the energy
  contract become tile loops.
- Gate: tile reads equal the resident amplitudes exactly; UMP2 energies
  unchanged; UKS gradient within 1e-9 Ha/Bohr; peak RSS measured on a fixture
  large enough to show it.

## Out of scope

- Merging the `udh_*` and `dh_*` files into one spin-general pipeline.
  Decided: the engines are shared; the thin per-spin front ends stay.
- UKS double-hybrid geometry optimization and frequencies (refused; the
  gradient they would call passes FD).
- The per-AO-element `_compute_eri_deriv_elem` traversal and its hardcoded
  Obara-Saika call, which both spins share.
- Spherical, solvent, range-separated and RI double-hybrid gradients.
