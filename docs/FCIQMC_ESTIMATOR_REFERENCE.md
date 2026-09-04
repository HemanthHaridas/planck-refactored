# FCIQMC Reference Determinant and Estimator Validation

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**What determinant should FCIQMC project on, and why were both estimators wrong on real molecules despite every existing energy gate being green?**

## Short answer

Both FCIQMC estimators were wrong on real molecules, in two independent ways, and every existing gate was green throughout because the suite compares energies, not wavefunctions. The projected energy `E = H_00 + (sum_j H_0j c_j) / c_0` is only an estimator of the ground-state energy when `|0>` carries a dominant share of the sampled wavefunction. Nothing in the code enforced that, and two separate mechanisms broke it: a reference determinant chosen by orbital index rather than energy, and a degenerate ground state with no restoring force inside its own manifold.

## Where the logic lives

- `src/post_hf/ci/fciqmc.{h,cpp}` — reference-determinant selection, the reweighting/re-anchoring fix
- `run_fci` / `run_fciqmc` wavefunction dumps (`C_I/C_0` and `<N_I>/<N_0>`), gated behind `verbosity verbose`
- `tests/fciqmc_validation.py` — the reproducing driver and its three sweeps
- `tests/inputs/exploratory/fciqmc/validation/` — deliberately unregistered fixtures
- `FCIQMC_DRIVER_AND_VALIDATION.md` — the sign-instability finding this doc's "neither estimator dominates" result mirrors

## What invariants matter

### 1. An energy gate cannot see a wavefunction-level defect

The regression suite compares energies, a single scalar contracted over the whole CI vector, so errors in the vector can cancel inside it. `n2_fciqmc_sto3g` passed at 0.32 sigma while the reference determinant was wrong by a factor of 14.2x in weight. The instrument that found both defects instead compares the wavefunction directly: `run_fci` prints `C_I/C_0` for its dominant determinants and `run_fciqmc` prints `<N_I>/<N_0>` over the accumulated *signed* per-determinant weight, in one format, on both paths, under `verbosity verbose`.

Design rule:

- When validating a stochastic or approximate method against an exact reference, compare the underlying object (the wavefunction / vector) directly in addition to the scalar it produces — a scalar can hide a large per-component error through cancellation.
- Accumulate and compare *signed* weights, summed over steps, not an instantaneous snapshot. An instantaneous population is shot noise (single-walker determinants flip sign step to step); a magnitude-only average would agree with a sampler that got every phase wrong.

### 2. The reference determinant must be chosen by energy, not by orbital index order

`reference_determinant` occupied the lowest-index orbitals, but on N2/STO-3G the Aufbau determinant is `0xbf` (orbitals [0,1,2,3,4,5,7]) because MO 6 lies above MO 7 in the converged SCF ordering — the code used `0x7f` instead, and the true reference carried 14.2x the weight of the wrong one everything was normalized against. The failure is silent because a weak anchor inflates the estimator's *variance* rather than producing an obviously wrong number.

Design rule:

- Select the reference determinant by minimizing the diagonal Hamiltonian element (`ops.diagonal`) over single occupied-to-virtual swaps from the Aufbau guess, not by orbital index order and not by reading SCF MO energies directly — `ops.diagonal` wraps the same `slater_condon_element` the propagator uses, so the reference cannot disagree with the Hamiltonian actually being sampled.
- A hill-climb search (not exhaustive, which is `C(n,k)^2` determinants) is sufficient — validate it against exhaustive search on a synthetic ordering before trusting it on production-sized systems.
- Remember the reference is also the starting population: a wrong reference corrupts equilibration dynamics too (the run spends time migrating away from a near-empty determinant), not just the projected-energy estimator. "The shift never touches the reference" is true of the estimator formula and false of the dynamics it depends on.

### 3. A degenerate ground state has no restoring force inside its own manifold, and longer sampling makes the drift worse

Any mixture of degenerate eigenstates is itself an eigenstate at the same energy, so the imaginary-time dynamics apply no restoring force within a degenerate manifold, and the population random-walks between partners. On C2/STO-3G (FCI roots 0 and 1 both `-74.6406501646`), increasing equilibration length from 20000 to 60000 steps moved the projected-energy deviation from +2.62 sigma to -5.57 sigma — below the variational minimum — with no other symptom: the sign is steady, the population is controlled, no warning fires under the old code.

Design rule:

- Do not assume more equilibration time is safe by default on a system that might have a degenerate or near-degenerate ground state — it can only make the drift worse, the opposite of the usual intuition about convergence.
- Warn when any determinant's weight exceeds a fixed multiple (2x) of the seeded reference's weight, and re-anchor the projection onto the largest-weight determinant exactly once, at the end of equilibration — never continuously, since a moving reference during sampling would change what the accumulated ratio-of-sums means partway through.
- Break ties in the re-anchoring scan deterministically (on the bitstring), because the population is stored in a hash map and an order-dependent tie-break would silently break fixed-seed reproducibility.

### 4. Neither FCIQMC estimator dominates — each is blind to a failure the other sees

The shift energy is immune to the degenerate-manifold drift in invariant 3 (-1.05/-0.14/-0.49 sigma across the same three C2 runs) because it responds to total population growth, indifferent to how weight is distributed inside a degenerate manifold. This is the mirror image of the sign-instability finding in `FCIQMC_DRIVER_AND_VALIDATION.md`, where the projected energy caught a broken run that the shift reported as perfectly converged.

Design rule:

- Always compute and cross-check both the shift and projected energy estimators. This is now demonstrated in both directions (each estimator catches a failure mode invisible to the other) rather than merely asserted as good practice.

## What was found and fixed

1. **Defect 1 — wrong reference determinant.** Fixed by selecting the reference via a hill-climb minimizing `ops.diagonal` from the Aufbau guess (see invariant 2). Measured on N2: shared determinants between the `run_fci`/`run_fciqmc` wavefunction dumps went 0 -> 16, projected error bar 3.27e-01 -> 1.34e-02 (24x tighter), shift deviation 0.92 -> 0.11 sigma.
2. **Defect 2 — degenerate-manifold drift.** Fixed by warning at 2x the seeded reference's weight and re-anchoring once at the end of equilibration (see invariant 3). Verified as a mutation test: the known-broken `eq60` case on C2 goes -5.57 to +2.27 sigma with the shift bit-identical (proving only the projection changed), and non-degenerate N2 is bitwise unchanged end to end.
3. **Fixture replacement.** C2/STO-3G cannot validate the projected energy because its Ms=0 ground state is a degenerate open-shell pair despite being a "closed-shell singlet" input — inferring non-degeneracy from that label is exactly the mistake this defect punished. Replaced by HF/6-31G (11 orbitals, 5a/5b, ndet = 213444, 4.8x C2), whose non-degeneracy was measured directly: roots 0 and 1 at `-100.1156979102` / `-99.7369526906`, a 10.31 eV gap. A population sweep on HF spans 0.047-0.469 walkers/determinant, entirely below saturation (C2 saturates at 44100 walkers).

## Validation strategy that should remain in place

- The wavefunction-comparison instrument (`C_I/C_0` vs `<N_I>/<N_0>`, `verbosity verbose`), since energy-only gates cannot see this class of defect
- `tests/fciqmc_validation.py --binary build-full/hartree-fock --exact <reference> --test all` — three sweeps (production length, walker population, coefficient ratios) against the HF/6-31G fixture
- `--exact` is required rather than defaulted, since a stale default would silently turn every sigma in the sweep tables into a comparison against the wrong molecule
- Fixtures in `tests/inputs/exploratory/fciqmc/validation/` stay deliberately unregistered (each sweep runs minutes to hours, outside any CI budget) but should be re-run manually whenever the estimator or reference-selection logic changes

## Remaining architecture concern

The fixtures that exercise this validation are explicitly kept out of CI due to their runtime, which means a future regression in either estimator or in reference-determinant selection would not be caught automatically — only by someone manually re-running `tests/fciqmc_validation.py` against the committed HF/6-31G exact reference.
