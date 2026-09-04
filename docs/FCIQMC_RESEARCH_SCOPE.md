# FCIQMC Research Case and Validation Strategy

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

**Status: COMPLETE. F1-F5 are landed and gated; FCIQMC runs from an input file and reproduces exact FCI on a real molecule.**

This file answers a narrower architecture question:

**Should FCIQMC be built in this codebase at all, and if so, how is a stochastic method validated in a codebase gated on exactness?**

## Short answer

Yes, on technical grounds, with one caveat that has not changed: nothing in this repository currently wants the window FCIQMC opens. `correlation fciqmc` on N2/STO-3G (14400 determinants, 0.69 walkers per determinant) agrees with exact FCI `-107.6529998854` at 0.32 sigma (shift) and 0.41 sigma (projected), gated by `n2_fciqmc_sto3g`. Both prerequisite questions below were answered before any FCIQMC code was written, and both answers held:

| | question | answer |
|---|---|---|
| **Q1** | Is there a calculation this codebase cannot do that FCIQMC could? | **Yes — Cr2 CAS(12,18).** Two atoms; 344.6M determinants; breaks both the time and memory walls |
| **Q2** | Can a stochastic method be validated in a codebase gated on exactness? | **Yes — G1-G4 built and passing** as CTest `planck-statistical-gate` |

What remains is a target, not a task: nobody has asked for the Cr2 binding curve, and while parallelism is scoped with its determinism policy already decided, it is worth building only the day a target exists. FCIQMC (Booth/Thom/Alavi, *JCP* **131**, 054106) samples the FCI wavefunction with a population of signed walkers evolving in imaginary time, instead of storing a CI vector, reaching determinant spaces deterministic FCI cannot at the cost of a stochastic error bar. The implementation details live in `FCIQMC_SAMPLING_AND_DYNAMICS.md`, `FCIQMC_POPULATION_CONTROL.md`, and `FCIQMC_DRIVER_AND_VALIDATION.md`; this document keeps the case for the work and the measurements that bound it.

## Where the logic lives

- `src/post_hf/ci/` — the determinant layer, shared by FCI and CASSCF; FCIQMC extends it rather than reimplementing it
- `slater_condon_element`, `ci.h:43` — the spawn's `H_ij`
- `build_ci_diagonal`, `ci.h:101` — the death step's `H_ii`
- `src/post_hf/fci.cpp` — the deterministic reference used for validation below ndet ~10^5
- `casscf_internal.h:27` — `kMaxPackedSpatialOrbitals` = 31, the existing bitstring ceiling
- `tests/{statistical_gate,blocking,reproducibility,mc_estimator}.py` — the statistical gate machinery (G1-G4)
- `tests/inputs/exploratory/fciqmc/cr2_casscf_target.hfinp` — the target case (not a regression case)
- `tests/inputs/exploratory/fciqmc/` — validation fixtures
- `docs/FCI_SIGMA_BUILD_PERFORMANCE.md` — the determinism precedent this work imitates
- `FCIQMC_SAMPLING_AND_DYNAMICS.md`, `FCIQMC_POPULATION_CONTROL.md`, `FCIQMC_DRIVER_AND_VALIDATION.md` — the full implementation ladder

## What invariants matter

### 1. A stochastic estimator needs its own class of gate, built and validated before the method exists

`metric_within_sigma` (G1), a Flyvbjerg-Petersen blocking analysis (G2), a fixed-seed bitwise reproducibility harness (G3), and a deliberately trivial end-to-end stochastic estimator (G4) were all built and gated against synthetic data with known answers, before any FCIQMC code was written. G1 asserts a 2-sigma deviation passes at 3-sigma and fails at 0.1-sigma, and that sigma = 0, negative sigma, NaN, and infinite sigma all fail rather than passing everything. G2 recovers the analytic correlation time of an AR(1) series across a 39x range, always biased slightly high (5-10%) by design, since overestimating sigma fails a gate loudly while underestimating passes one silently — the naive standard error understates sigma by up to 6.6x on correlated data. G3 digests raw IEEE bits, not text, since a print-precision comparison would hide the last-ulp reduction-order defects this codebase has been bitten by before. G4 asserts the mean within 3-sigma, that sigma scales as N^-0.478 against a theoretical N^-0.5, that the error bar is calibrated (`rms(deviation)/sigma ~= 1`), and — the anti-vacuity check — that the same trials fail against the naive sigma.

Design rule:

- Build and gate the statistical validation machinery against synthetic data with a known analytic answer first, entirely independent of the stochastic method it will eventually validate. If it cannot be made to pass there, the method under development is not maintainable in this codebase.
- Never gate a blocking/error-bar analysis on real stochastic output alone — only on synthetic series where the correct answer is known, so an under-reporting bug cannot silently pass its own check.

### 2. A fixture must share the structure whose violation you intend to detect

G4's first population was i.i.d. Gaussian, and a mutation restricting the sampler to half the space came back green — with i.i.d. values, every sub-range has the same mean, so genuine sampler bias moved the answer by only 0.58 sigma. A population that trends with index makes the same mutation 25.9 sigma out and catches a subtler 90%-coverage mutation too.

Design rule:

- When designing a fixture to catch a specific class of sampler bias, make sure the fixture has the structure that bias would actually violate (e.g., a trend across the sampled index), not just statistically plausible-looking data. This is the inverse trap of `CCGEN_MERGE_TRANSPOSES`, where a fixture was too general rather than too structureless — both directions defeat the gate's purpose.

### 3. Deterministic FCI cost does not follow one pooled scaling exponent

Deterministic FCI cost scales with electron count, not ndet alone — per-determinant cost varies 3.3x at comparable ndet, ordered by electron count. Fitting each regime separately gives ndet^1.56 for 6-7 electrons and ndet^1.14 for 2-3 electrons. A pooled three-point fit gave ndet^1.69, and a fourth point (BeH2/6-31G) run specifically to test that pooled fit came in at 36.7 s against a predicted 2.6 minutes — wrong by 4.3x, in the optimistic direction.

Design rule:

- Never use a pooled scaling exponent across systems with substantially different electron counts. Fit separate regimes, and validate any extrapolated fit with a held-out data point before trusting it for a cost estimate — the falsifying BeH2 input is committed for this reason.

### 4. A validation fixture must be large enough to make sampling genuine, not just large enough to be affordable

`h2_fci_sto3g` (4 determinants) and `water_fci_sto3g` (441 determinants) are unsuitable FCIQMC fixtures: the walker population would exceed the space, so the gate would prove nothing about sampling. The chosen primary fixture, N2/STO-3G (10 orbitals, 7 alpha/7 beta, ndet 14400), is the smallest system satisfying both constraints simultaneously — cheap enough for a gate to recompute the exact reference, and large enough that a few-thousand-walker population is a genuine sample rather than covering the whole space.

Design rule:

- Choose a stochastic-method validation fixture to be simultaneously affordable for an exact reference AND large enough that the walker population is a genuine sample of the space, not a fixture chosen for affordability alone.
- Use more than one fixture varying a different axis (Be/6-31g* at 2 alpha/2 beta isolates the excitation generator by having no doubles between different occupied pairs; C2/STO-3G at 6 alpha/6 beta triangulates electron count against a fixed orbital count) rather than a single confounded pair.

### 5. Bitwise thread-count invariance is achievable for FCIQMC by partitioning the work, not the output

Every parallel path in Planck is bitwise thread-count-invariant by design and by gate (the DFT J/K builds, the CC kernels, the ERI transforms, the FCI sigma build), and FCIQMC's natural parallelization (the annihilation step's floating-point sum) looked like a potential exception since it depends on arrival order. The FCI sigma-build work already threaded an analogous scatter into a shared vector and kept bitwise invariance at no measurable serial cost and 4.8% idle, which raised the burden: show why FCIQMC cannot do the same. It cannot be shown, and by a cleaner mechanism — partition the *parent* determinants by `hash(parent) % kBins`, accumulate spawns per bin privately, and merge bins in fixed bin order. Verified on a model of the spawn: identical result whether parents are visited in order, reversed, or shuffled.

Design rule:

- A fixed-order reduction is necessary but not sufficient for thread-count invariance — what must be deterministic is the *partition of work into accumulators*, not merely the order accumulators are summed afterward. `schedule(dynamic)` gives an accumulator a different subset of terms per run; keying accumulators by `omp_get_thread_num()` makes their contents depend on thread count. The working pattern is `partials[j / bin_size]` with a fixed bin count.
- Partition by the determinant that generates work (the parent), never by the determinant that receives it (the child) — binning by child fixes which accumulator receives a spawn but not the order arrivals reach it, so two threads spawning onto the same determinant would still race.
- Binning by determinant is invariant even to the bin count itself (each determinant maps to exactly one bin regardless of `kBins`), which is an easier property than the FCI sigma build had, where binning by index range meant a determinant could land in a different bin as `kBins` changed.
- Before restructuring any loop that skips negligible weight (a sparsity exploitation), measure what fraction of iterations the skip actually eliminates — a cheap-looking guard on an outer loop can carry asymptotic weight, as the FCI sigma build's refuted gather reformulation demonstrated (2.2-2.4x slower after moving a `|c| < 1e-15` test inward).
- Do not make a bitwise-invariance exception silently. Either accept a fixed-order reduction, or document the path as the one exception, as a decision — not a discovery made after the fact.

## What was found

1. **Q1 answered: Cr2 CAS(12,18) is a real, currently-blocked calculation this codebase already handles apart from determinant count.** Surveying standard multireference benchmarks against the measured FCI boundary, almost everything canonical is already reachable (N2/C2 full-valence dissociation, benzene/naphthalene pi systems, [Fe2S2], Cr2 at the textbook CAS(12,12)). Cr2 CAS(12,18) is the sharpest statement of the gap because it is the first case where both the time wall and the memory wall fail at once: 344 622 096 determinants, ~2 year FCI runtime, 2.76 GB per CI vector, ~22 GB for a Davidson subspace. Everywhere below it, time binds and memory does not (at `n_act` 14 a CI vector is only 0.09 GB against a 3-day solve). The 12 active electrons are the 3d^5 4s^1 valence on each Cr, and the sextuple Cr-Cr bond is the textbook case where single-reference methods get the binding curve qualitatively wrong, so the large active space is chemically motivated, not invented to be big. It needs no new determinant layer — `kMaxPackedSpatialOrbitals` is 31, so 18 active orbitals fit the existing bitstring representation; only the determinant count blocks. Verified runnable, not merely arithmetic: Cr is present in sto-3g/6-31g/cc-pVDZ/TZ/QZ, Cr2/STO-3G RHF converges in 44 iterations, and the CAS(12,12) rung starts and runs long, consistent with its estimate.

   | active space | ndet | FCI cost | one CI vector | status |
   |---|---|---|---|---|
   | CAS(12,12) | 853 776 | ~1.3 h | 7 MB | **reachable today** |
   | CAS(12,14) | 9 018 009 | ~2 d | 72 MB | painful |
   | CAS(12,16) | 64 128 064 | ~47 d | 513 MB | out of reach |
   | **CAS(12,18)** | **344 622 096** | **~2 yr** | **2.76 GB** | **the target** |

2. **The deterministic FCI boundary was remeasured** after the sigma build got ~17x faster (`FCI_SIGMA_BUILD_PERFORMANCE.md`), at 4 threads:

   | system | ndet | n-alpha | FCI wall | s per 10^3 det |
   |---|---|---|---|---|
   | Be/6-31g* | 8 281 | 2 | 2.72 s | 0.328 |
   | N2/STO-3G | 14 400 | 7 | 8.28 s | 0.575 |
   | C2/STO-3G | 44 100 | 6 | 47.7 s | 1.081 |
   | BeH2/6-31G | 81 796 | 3 | 36.7 s | 0.448 |

   Half-filled CAS at the 6-7 electron exponent projects: CAS(12,12) ~1.3 h, CAS(14,14) ~3 d, CAS(16,16) ~208 d, CAS(18,18) ~36 yr. The practical ceiling is `n_act` ~= 12, so the FCIQMC window opens around 13; the ~17x speedup bought about one `n_act` step, since a faster exponential is still exponential.

3. **Q2 answered: G1-G4 built and passing** as CTest `planck-statistical-gate` (2.0 s), all mutation-verified — see invariants 1 and 2 above.

4. **The reusable pieces of the existing CI layer were identified and confirmed sufficient.** Bitstring determinants (`CIString`), occupation/parity helpers, `slater_condon_element`, `build_ci_diagonal`, active-space integrals (`h_eff` + `ga`), and the deterministic reference all carried over as-is. `slater_condon_element` no longer heap-allocates per call (fixed alongside the sigma-build work), so the spawn step inherits that fix — the allocation prerequisite this scope originally named is satisfied. What had to be built new: an RNG policy (`RandomSource`, seeded and per-shard, with `derive(index)` independent of shard count), a sparse walker container (`WalkerPopulation`, a dynamic determinant-keyed map, since the existing `det_lookup` indexes only a fixed enumerated space), and an excitation generator with a consistent `p_gen`.

5. **The full implementation ladder (F1-F5) landed**, gated by `planck-fciqmc-walkers` (~34 s) plus the `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g`, and `h2_fciqmc_threads1/4` regression cases:

   | step | what | answer doc |
   |---|---|---|
   | **F1** | walker container, RNG policy | `FCIQMC_SAMPLING_AND_DYNAMICS.md` |
   | **F2** | excitation generator with `p_gen` | same |
   | **F3** | spawn / death / annihilation | same |
   | **F4** | shift control, estimators, initiator | `FCIQMC_POPULATION_CONTROL.md` |
   | **F5** | driver, keywords, N2 gate, determinism | `FCIQMC_DRIVER_AND_VALIDATION.md` |

6. **The determinism decision (2026-09-02): no exception.** FCIQMC keeps bitwise thread-count invariance — see invariant 5. Current state is entirely serial (zero `#pragma omp` in `fciqmc.cpp` or `fciqmc_driver.cpp`), verified bitwise-identical across `OMP_NUM_THREADS` = 1, 2, and 4 on N2/STO-3G today. This is a policy decision made for the threading that does not exist yet, before it is written rather than discovered afterward — so F5's eventual parallel step is a normal piece of work under the existing discipline, not an exception requiring separate sign-off.

## Validation strategy that should remain in place

- `planck-statistical-gate` (G1-G4), gated on synthetic data with known analytic answers, never on real FCIQMC output
- `planck-fciqmc-walkers`, `h2_fciqmc_sto3g`, `n2_fciqmc_sto3g`, `h2_fciqmc_threads1/4` — the full F1-F5 ladder
- The fixed-seed reproducibility gate as the *primary* validation instrument (it runs at any problem size), with the statistical/blocking gate restricted to small systems by construction, since a deterministic reference exists only below ndet ~10^5 — not the regime FCIQMC is actually for
- `p_gen` validation against brute-force enumeration on a tiny space before trusting any energy from a new or modified excitation generator — an excitation generator whose returned probability disagrees with its actual sampling distribution produces a plausible, converged, wrong energy, the same failure class as the spin-adapt default and the invalid ERI symmetry table found elsewhere in this codebase
- Clearing `__pycache__` between mutation-testing runs of the Python statistical gate machinery — a `cp` restore that preserves mtime within the cache's resolution can leave a stale `.pyc` running the mutated module while the file on disk is correct, misleading in either direction

## What this must not do

- Do not skip `p_gen` validation against brute-force enumeration.
- Do not gate on `h2_fci_sto3g` or `water_fci_sto3g` alone — both are small enough that the walker population covers the space, so neither can detect a sampling defect. They remain useful for correctness (F3), not for validating statistics.
- Do not reimplement the determinant layer — `src/post_hf/ci/` is shared by FCI and CASSCF and is the single representation; extend it there.
- Do not claim a speedup against FCI. They compute different things: FCI gives the exact energy, FCIQMC an estimate with an error bar. The valid comparison is reachable system size, not wall-clock.
- Do not read the ~17x FCI sigma-build speedup as weakening the case for FCIQMC — it moved the constant, not the ndet^1.56 scaling, and bought about one `n_act` step. What it genuinely changed is the validation budget.

## Related but separate outcome: prior art

- Booth, Thom, Alavi, *JCP* **131**, 054106 (2009) — the original method.
- Cleland, Booth, Alavi, *JCP* **132**, 041103 (2010) — the initiator approximation, which is what makes it practical.
- NECI (Alavi group) — reference implementation; most useful for what the excitation generators and population control actually look like.

## Remaining architecture concern

What remains is a target, not a task. Cr2 CAS(12,18) is a real blocked calculation on a molecule this code already handles, but nobody has asked for its binding curve. Parallelism is scoped and its determinism policy decided (invariant 5), and would be worth building the day a target exists — but building it before then would be speculative work against no current consumer. The method is implemented, validated, and unused.
