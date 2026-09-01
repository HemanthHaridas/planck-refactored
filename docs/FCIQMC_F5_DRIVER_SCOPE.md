# Scope: F5 — driver wiring, the N2 regression gate, and the determinism decision

**Scope for in-flight work. Not started.** Step F5 of the ladder in
`FCIQMC_RESEARCH_SCOPE.md`. F1–F4 are landed and gated by
`planck-fciqmc-walkers`: the method runs, holds a population, and reports both
estimators with honest error bars.

**This step exists because everything validated so far runs on a synthetic
Hamiltonian.** `ToyHamiltonian` respects the sparsity of a real one and is checked
against exact diagonalization, but it is not a molecule: no SCF, no real
integrals, no basis set. Nothing yet demonstrates FCIQMC reproduces a *chemical*
answer.

## The deliverable that motivates the rest

**A regression case that reproduces N2/STO-3G deterministic FCI within its own
error bar.** N2/STO-3G is the research scope's primary validation fixture (10
orbitals, 7α/7β, ndet = 14 400, exact FCI cheap at ~8 s), chosen because it is the
smallest system where FCI is affordable *and* the determinant space is large
enough that a few-thousand-walker population is a genuine sample rather than
covering the space.

**Why this cannot be a unit test.** The gate would need a converged SCF for its
integrals — `h_eff = CᵀH_coreC` plus the transformed two-electron array — which
means linking the basis, integral and SCF machinery into a test that currently
links one file. And 14 400 determinants is 400x the current fixture, which already
uses the whole ~30 s budget. The honest home is a regression case driven by the
real binary.

## What must be built

The driver already has the shape to copy. `run_fci` (`src/post_hf/fci.h`) takes a
`Calculator` and shell pairs, reads options from `calc._active_space`, and writes
`_correlation_energy` / `_correlated_total_energy` back. `hf_driver.cpp:1401`
dispatches it off `PostHF::FCI`. FCIQMC needs the same three pieces plus its own
inputs.

## Steps

### F5.1 — the driver entry point — **DONE 2026-09-01. FCIQMC runs on real integrals.**

`run_fciqmc` (`src/post_hf/fciqmc_driver.{h,cpp}`), dispatched from
`hf_driver.cpp` on `correlation fciqmc`. Measured on H2/STO-3G against the exact
FCI `-1.1372744062`:

| estimator | value | blocked error | deviation |
|---|---|---|---|
| shift | −1.1375360199 | 2.76e-03 | **0.09σ** |
| projected | −1.1373278832 | 1.58e-04 | **0.34σ** |

**The integral transform is shared, not reimplemented.** `build_all_mo_ci_setup`
was *extracted* from `run_fci` — validation, electron counts, the packed-orbital
and `ci_max_dim` guards, `h_eff` and `ga` — and `run_fci` now calls it. Verified
behaviour-neutral: the FCI regression cases pass and N2 still gives
`-107.6529998854`, digit-identical. The Hamiltonian callbacks wrap
`slater_condon_element`, the same routine the deterministic CI uses, so the two
paths cannot disagree about the Hamiltonian — only about how they solve it.

**Gated** by `h2_fciqmc_sto3g` (extended suite), the first production consumer of
`metric_within_sigma`: both estimators asserted within 5 of their *own* blocked
error bars. Verified non-vacuous — against a deliberately wrong reference it fails
with the deviation reported in σ.

A collapsed or diverged population is a hard error naming the likely cause, never
a reported number.

**A build-hygiene trap worth recording.** A monitor watching only
`fciqmc_driver.cpp`'s timestamp fired on a build that predated the `io.cpp` edit by
one minute, so the first run failed with "Invalid Correlation : fciqmc" against
correct source. **Watching one file's timestamp does not prove the build included
every edit** — check all of them.

### F5.1 (original text) — the driver entry point

`run_fciqmc(Calculator&, const std::vector<ShellPair>&)`, mirroring `run_fci`:
build the determinant space's integrals the same way, run the propagator under
shift control, report the shift and projected energies with blocked error bars.

- **Verify:** on a system small enough to diagonalize (H2/STO-3G, 4 determinants)
  the reported energy matches `run_fci` on the same input within its error bar.
  This is the first time the two paths are compared on *real* integrals, so a
  disagreement here is an integral-plumbing defect, not a sampling one.
- **Do not reimplement the integral transform.** `run_fci` builds `h_eff` and `ga`
  from the converged reference; factor that out and share it, or the two paths
  will drift and every later comparison becomes ambiguous.

### F5.2 — input keywords — **DONE 2026-09-01**

Eleven keywords in `_scf_map`, each validated at parse time so a bad value fails
naming the keyword (`fciqmc_walkers must be positive`, `fciqmc_steps must be at
least 4`, …) rather than surfacing later as odd behaviour.

**Every parameter verified to change the run**, by varying each against a baseline
— nine moved the energy immediately. The tenth, `fciqmc_initiator`, appeared inert
at `n_add = 2.0`; investigation showed it is correctly plumbed and the **probe
value was below the walker scale**. With 5000 walkers on 4 determinants every
parent weight is ~1250, so a threshold of 2 never fires. At `n_add = 100` and
`1e9` it changes the answer as expected. Same fixture-saturation limit F4.5 hit.

**The reproducibility contract holds end-to-end through the real binary:** seed
4242 twice gives `-1.1382560651` identically, seed 9999 gives `-1.1373518204`.
That is F3.5's property verified at the driver level rather than in a unit test.

All parameters — seed included — are echoed to the output, so a result is
reproducible from its own log.

**A build-verification trap, and it defeated the fix for the previous one.**
F5.1's lesson was to check every edited file's timestamp against the binary. That
check *passed* while the binary still lacked the change, because a relink during
an in-flight build can produce a binary newer than its own inputs. A
`strings | grep -c` then returned 2 and looked like confirmation — but it was
matching the **error-message strings** (`"fciqmc_walkers must be positive"`), not
the map key. An exact match (`grep -qx`) showed the keyword genuinely absent.

> **A substring match on a binary is not evidence the symbol you care about is
> there, and a timestamp is not evidence a build finished.** Test the actual
> condition: build not running, *and* exact symbol present.

### F5.2 (original text) — input keywords

`correlation fciqmc` plus the parameters F4 established as load-bearing: walker
target, `zeta`, `xi`, timestep, spawn granularity, equilibration length, run
length, seed.

- **Verify:** every parameter is reachable from an input file and changing it
  changes the run. A keyword the parser accepts but ignores is worse than one it
  rejects.
- **The seed must be an input.** The reproducibility contract F3.5 established is
  worthless if the seed is not user-visible and recorded in the output.

### F5.3 — the N2/STO-3G regression gate

The deliverable. Compare FCIQMC's energy against the deterministic FCI reference
for the same input.

**The reference, measured** (`OMP_NUM_THREADS=4`, ~8 s):

```
N2/STO-3G, 10 orbitals, 7 alpha / 7 beta, CI dim = 14400
  Total FCI Energy      -107.6529998854
  Correlation Energy      -0.8864061248
```

- **Use `metric_within_sigma`** (G1), against the exact FCI value, with the
  blocked standard error as the uncertainty metric. This is what that assertion
  was built for and it has had no production consumer until now.
- **Assert the reported error bar is blocked, not naive.** A run that reports the
  naive error would pass a within-σ check while understating its uncertainty by
  ~5x (measured in F4.4). Gate the ratio, or report both.
- **Assert fixed-seed reproducibility** in the same case: two runs with the same
  seed give bitwise-identical output. That is the gate that survives at any system
  size, and it costs one extra invocation.
- **Budget it honestly.** If the run is too slow for the default suite, put it in
  `extended` — but measure first rather than assuming, and record the number.

### F5.4 — the determinism decision

**Do not start until F5.1–F5.3 are green**, and read §6 of the research scope
first. Every parallel path in Planck is bitwise thread-count-invariant, by design
and by gate. FCIQMC's natural parallelisation is not: the annihilation sum depends
on arrival order.

The FCI sigma build is the worked precedent — it threaded a scatter into a shared
vector and **kept** bitwise invariance for `kBins × dim × 8` bytes of
fixed-partition accumulators, at no measurable serial cost and 4.8 % idle. So the
burden is to show why FCIQMC cannot do what it did.

- **Decide explicitly, and write the decision down.** Either accept a fixed-order
  reduction, or document FCIQMC as the one path where bitwise thread-invariance
  does not hold. **Do not make the exception silently** — that is the failure this
  project has already paid for once.

## What this must not do

- **Do not gate the energy without an error bar.** A stochastic result quoted as a
  bare number invites a `metric_close` comparison, which is exactly the discipline
  mismatch Q2 asked about.
- **Do not tune the run parameters until the gate passes.** F4 established that
  `zeta` has a usable band and that the band is system-specific; a case that only
  passes at one hand-found setting is pinning an accident, not validating a method.
- **Do not report an energy from a run whose population collapsed or diverged.**
  F3 and F4 both established this; the driver must surface it as a failure rather
  than a number.
- **Do not let the N2 case be the only real-molecule gate.** Add at least one
  smaller one (H2 or LiH/STO-3G) that runs in the default suite, so a plumbing
  defect is caught in seconds rather than only by the slow case.

## Key code locations

| what | where |
|---|---|
| the method (F1–F4) | `src/post_hf/ci/fciqmc.{h,cpp}` |
| the unit gate | `tests/fciqmc_walkers.cpp` |
| the pattern to copy | `run_fci`, `src/post_hf/fci.{h,cpp}` |
| driver dispatch | `src/hf_driver.cpp:1401` |
| the `PostHF` enum | `src/base/types.h:73` |
| within-σ assertion | `metric_within_sigma`, `tests/run_regressions.py` |
| N2 input and its FCI reference | `tests/inputs/exploratory/fciqmc/n2_fci_sto3g.hfinp` |
| what F1–F4 established, and their traps | `docs/FCIQMC_SAMPLING_AND_DYNAMICS.md`, `docs/FCIQMC_POPULATION_CONTROL.md` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
