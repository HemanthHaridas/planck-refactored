# What determinant should FCIQMC project on?

Both FCIQMC estimators were wrong on real molecules, in two independent ways, and
**every existing gate was green throughout**. This answers why, and what the
reference determinant has to be.

The short version: the projected energy

```
E = H_00 + (sum_j H_0j c_j) / c_0
```

is only an estimator of the ground-state energy when `|0>` carries a dominant
share of the sampled wavefunction. Nothing in the code enforced that, and two
separate mechanisms broke it.

## Why energy gates could not see either defect

The suite compares **energies**. An energy is one scalar contracted over the whole
CI vector, so errors in the vector cancel inside it — which is exactly what
happened: `n2_fciqmc_sto3g` passed at 0.32 sigma while the reference determinant
was wrong.

The instrument that found both defects compares the **wavefunction**. `run_fci`
prints `C_I/C_0` for its dominant determinants; `run_fciqmc` prints `<N_I>/<N_0>`
over the accumulated **signed** per-determinant weight. One format, both paths,
`verbosity verbose`. On N2 the two dumps initially shared **zero** determinants.

Signed and summed over steps, not snapshotted: an instantaneous population is shot
noise (determinants holding one walker flip sign step to step), while `<N_I>`
converges to the wavefunction. A magnitude-only average would agree with a sampler
that got every phase wrong.

## Defect 1 — index order is not energy order

`reference_determinant` occupied the lowest-index orbitals. On N2/STO-3G the
Aufbau determinant is `0xbf` = orbitals [0,1,2,3,4,5,**7**], because MO 6 lies
above MO 7 in the converged SCF ordering. The code used `0x7f`.

The true reference carried **14.2x** the weight of the one everything was
normalised against.

**Why it survived:** the failure is silent. A weak anchor inflates the estimator's
VARIANCE rather than producing an obviously wrong number — N2's projected error
bar ran ~20x the shift's, which reads as ordinary noise.

**Fix:** minimise `ops.diagonal` over single occupied->virtual swaps from the
Aufbau guess. Deliberately *not* read from SCF MO energies: `ops.diagonal` wraps
the same `slater_condon_element` the propagator uses, so the reference cannot
disagree with the Hamiltonian being sampled. A hill-climb, not an exhaustive scan
(that is `C(n,k)^2` determinants) — validated against exhaustive search on a
synthetic ordering with 6/7 swapped.

Measured on N2: shared determinants **0 -> 16**, projected error bar
**3.27e-01 -> 1.34e-02 (24x)**, shift **0.92 -> 0.11 sigma**.

**It corrupted the shift too, indirectly.** The reference is also the starting
population, so the run spent equilibration migrating away from a near-empty
determinant. "The shift never touches the reference" is true of the estimator and
false of the dynamics.

## Defect 2 — degenerate ground states have no restoring force

Any mixture of degenerate eigenstates is itself an eigenstate at the same energy.
The imaginary-time dynamics therefore apply **no restoring force** within a
degenerate manifold, and the population random-walks between partners.

Measured on C2/STO-3G (FCI roots 0 and 1 both `-74.6406501646`; partners
`0x3f/0x6f` and `0x6f/0x3f` at +/-1.000000), varying **only** the equilibration
length:

| equil | partner/anchor | E_proj | sigma vs exact |
|---|---|---|---|
| 20000 | -0.861 | -74.6172886 | +2.62 |
| 40000 | -1.674 | -74.6413697 | -0.06 |
| 60000 | **-3.833** | **-74.7503958** | **-5.57** |

By 60000 the anchor holds a quarter of the partner's weight while the numerator
still samples the whole manifold. The ratio inflates negatively and reports an
energy **5.6 sigma below the variational minimum**.

**Nothing is unstable.** The sign is steady, the population is controlled, the
reference holds 743 walkers, no warning fires. The run is fine; the estimator is
measuring the wrong thing. **Longer equilibration makes it worse** — more time to
converge is also more time to drift.

**Fix:** warn when any determinant exceeds 2x the seeded reference's weight, and
re-anchor onto the largest-weight determinant **once**, at the end of
equilibration. Once, because a reference moving during sampling would change what
the accumulated ratio-of-sums means partway through. The scan breaks ties on the
bitstrings — the population is a hash map, and an order-dependent anchor would
break fixed-seed reproducibility.

Verified as a mutation test: `eq60` goes **-5.57 -> +2.27 sigma** with the shift
**bit-identical** (only the projection changed), and non-degenerate N2 is
**bitwise unchanged** end to end.

## Neither estimator dominates

The shift is immune to defect 2 (-1.05/-0.14/-0.49 sigma across those three runs)
because it responds to total population growth, which is indifferent to how weight
is distributed inside a degenerate manifold.

That is the **mirror image** of the sign-instability finding in
`FCIQMC_DRIVER_AND_VALIDATION.md`, where the projected energy caught a broken run
the shift reported as perfect. Each estimator is blind to a failure the other sees.
**That is the argument for computing both**, and it is now demonstrated in both
directions rather than asserted.

## Fixture consequence

C2/STO-3G cannot validate the projected energy. The replacement is **HF/6-31G**
(11 orbitals, 5a/5b, **ndet = 213444**, 4.8x C2), whose non-degeneracy was
**measured**: roots 0 and 1 at `-100.1156979102` / `-99.7369526906`, a **10.31 eV
gap**, against C2's `0.00e+00`.

Inferring non-degeneracy from the "closed-shell singlet" label is exactly the
mistake C2 punished — it *is* a closed-shell singlet input, and its Ms=0 ground
state is a degenerate open-shell pair.

A population sweep on HF spans 0.047-0.469 walkers/determinant, entirely below
saturation; C2 saturates at 44100 walkers.

## Reproducing

```
tests/fciqmc_validation.py --binary build-full/hartree-fock \
    --exact -100.1156979102 --test all
```

Three sweeps: production length, walker population, coefficient ratios. Fixtures
live in `tests/inputs/exploratory/fciqmc/validation/` and are **deliberately
unregistered** — each sweep is minutes-to-hours, which no CI budget absorbs.
`--exact` is required rather than defaulted: a stale default silently turns every
sigma in the tables into a comparison against the wrong molecule.

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
