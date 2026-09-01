# Scope: F3 — spawn, death, annihilation

**Scope for in-flight work. Not started.** Step F3 of the ladder in
`FCIQMC_RESEARCH_SCOPE.md`. F1 (walker state, RNG) and F2 (excitation generator,
`p_gen`) are landed and gated.

This is the step where the pieces become a method. It is also the first step whose
output is a **physical number**, so it is the first that can be wrong in a way
that looks right.

## The algorithm, in the form F3 implements it

Each iteration applies `(1 - dt(H - S))` to the walker population:

```
for each occupied determinant i with weight c_i:
    SPAWN:  draw j from i;  child weight = -dt * H_ij * c_i / p_gen(i->j)
    DEATH:  c_i *= (1 - dt * (H_ii - E_ref - S))
ANNIHILATE: accumulate all spawned children into the population
             (opposite signs on the same determinant cancel)
```

F3 holds the shift `S` **fixed**. Population control is F4. With `S` fixed the
population grows or shrinks exponentially, which is correct and expected — the
energy estimator must still be right while it does.

**What F1 and F2 already supply:** `WalkerPopulation::add` *is* the annihilation
step; `draw_excitation` supplies `j` and `p_gen`; `slater_condon_element` supplies
`H_ij`; `build_ci_diagonal` supplies `H_ii`.

## The two things that can be silently wrong

**1. The `1/p_gen` division.** F2.5 established the lesson the hard way: when a
sampled quantity is used as a divisor, checking that the *estimator* is unbiased is
the wrong test — `E[1/X] ≠ 1/E[X]`, and the per-call acceptance estimate was
**1.72x** wrong in exactly the quantity the spawn uses while looking perfect in
`p_gen`. F3 divides by `p_gen` on every spawn, so this is live here, not a
historical note.

**2. The energy estimator.** Two estimators are standard and they fail differently:

- **Projected energy**: `E = Σ_j H_0j c_j / c_0`, a ratio of two stochastic
  quantities. **It is biased at finite population** — again `E[A/B] ≠ E[A]/E[B]` —
  and the bias shrinks as the population grows. This is a real, published property
  of the method, not a defect to fix, but it means a small-population test that
  agrees to 1e-6 is *suspicious*, not reassuring.
- **Shift energy**: the value of `S` that holds the population stationary. Requires
  population control, so it belongs to F4.

**F3 uses the projected energy and must state its population dependence**, rather
than quietly choosing a population where the bias is invisible.

## Steps

### F3.1 — one iteration, deterministically — **DONE 2026-08-31**

`propagate_deterministic` (`src/post_hf/ci/fciqmc.{h,cpp}`), gated by
`planck-fciqmc-walkers`. Matches a hand-computed matvec to **1e-12** on three
fixtures including an open-shell one. Also gated: the step touches only connected
determinants, propagating `+c` and `−c` gives exactly opposite populations while
their sum spawns nothing (annihilation survives the *propagation*, not just the
container), the propagator is linear, and the toy `H` is symmetric.
Mutation-verified against a spawn sign error, a dropped shift, and using the
child's diagonal in place of `H_ij`.

**Design note that made the check meaningful:** the Hamiltonian arrives as
callbacks (`HamiltonianOps`), not as `h_eff`/`ga`. That lets the gate drive the
dynamics with an *independently constructed* matrix — reusing
`build_ci_hamiltonian_dense` would verify the dynamics agree with the same
matrix-element code they call, which is consistency rather than correctness.

**The one failure was the TEST, and the diagnosis is worth keeping.** The first
run failed at 8.5e-2 on the closed-shell fixture while the 2-orbital case passed —
a pattern that points at same-spin doubles. It was not that, and not duplicate
connections either. The toy Hamiltonian filled **every** matrix entry, but a
physical `H` is exactly zero between determinants differing by more than a double
excitation, because it is a two-body operator. At `n_act=4, na=nb=2`, **9 of 35
pairs are unconnected**, so the reference matvec summed 9 contributions the
propagator correctly skipped. **A synthetic Hamiltonian must respect the sparsity
of a real one, or it is not a Hamiltonian and no propagator will reproduce it.**

### F3.1 (original text) — one iteration, deterministically

Implement `spawn`, `death` and the accumulate, but drive them with an
**enumerating** spawn (F2.2's `draw_uniform_excitation` over the full connection
set, or direct enumeration) rather than sampling. With every connection visited
exactly once and weighted by `H_ij`, one iteration is exactly `(1 - dt(H - S))c`
— a deterministic matrix-vector product.

- **Verify against the dense Hamiltonian.** Build `H` for H2/STO-3G (4
  determinants) with the existing `build_ci_hamiltonian_dense`, apply
  `(1 - dt(H - S))` by hand, and require the walker population to match **to
  machine precision**. No statistics involved.
- **Why this first:** it separates *the dynamics are wrong* from *the sampling is
  wrong*. Every later failure can then be attributed. If F3.1 does not match a
  matrix-vector product exactly, nothing after it is worth debugging.

### F3.2 — stochastic spawning, same fixed point — **DONE 2026-08-31**

`propagate_stochastic`. Draws connections instead of enumerating them and
reweights by `1/p_gen`; the death step stays deterministic (one diagonal element
per determinant, so sampling it adds variance for nothing). Unbiasedness was
verified numerically with a deliberately non-uniform `p_gen` before the code was
written.

Gated: the mean over 200k runs matches F3.1 within **5σ per component** on closed
and open shell, variance falls as `1/n_attempts`, raising `n_spawn_attempts` does
not change the expected step, death is exact, and a fixed seed reproduces bitwise.

**THE GATE WAS INITIALLY VACUOUS, and this is the finding worth carrying.** The
first version compared the mean against an absolute tolerance of `0.02`. Dropping
the `1/p_gen` reweighting entirely — the exact defect this step exists to catch —
**passed**, as did a 2x `p_gen` error. Spawn magnitudes here span **0.005 to 0.4**
across excitation classes, so `0.02` sat right at the size of the effect.

A fixed *relative* tolerance failed the other way: dominated by sampling noise on
the smallest components, it rejected correct code at 51 %.

The fix is to compare against each component's own **standard error**, accumulated
during the run. It is the only scale correct for every component at once and
requires no tolerance to be guessed. With it the two previously-passing mutations
are caught at **5553σ** and **226σ**.

**Generalizable: when the components of a checked quantity span orders of
magnitude, neither an absolute nor a relative tolerance is safe** — the former is
vacuous for the large components, the latter noise-dominated for the small ones.
Measure the standard error and compare in units of it. **This applies directly to
F3.3 and F3.4**, whose eigenvector components and energy contributions span a
similar range.

### F3.2 (original text) — stochastic spawning, same fixed point

Replace the enumerating spawn with `draw_excitation` and the `1/p_gen` weighting.
The population is now a random variable, but its **expectation** is unchanged.

- **Verify:** average the post-iteration population over many independent runs from
  the *same* start vector and require the mean to converge to F3.1's deterministic
  result, within 3σ using G2's blocking analysis. This is the assertion that
  catches a wrong `1/p_gen` — and it is a mean-of-a-linear-quantity, so unlike the
  energy it is *not* subject to a ratio bias.
- **Also verify:** the variance falls as 1/N_walkers. A spawn that is correct in
  the mean but has wrong variance scaling usually means `p_gen` is inconsistent
  with the sampling rather than wrong by a constant.

### F3.3 — imaginary-time propagation to the ground state

Iterate from a single walker on the reference determinant. With `S` fixed at the
reference energy the population should converge in *shape* to the ground-state
eigenvector.

- **Verify on H2/STO-3G (4 determinants, exact FCI `-1.1372744062`):** the
  normalized walker distribution converges to the exact ground-state eigenvector of
  the dense `H`, per component, within 3σ.
- **Verify the timestep bound.** `dt` must satisfy `dt < 2/(max|H_ii - S|)` or the
  propagation diverges. Assert that a deliberately too-large `dt` **does** diverge
  — an implementation that silently stays stable is not propagating what it claims.
  Confirmed on the same toy `H`: `dt = 0.9 × bound` is stable over 2000 iterations,
  `dt = 1.1 × bound` diverges.

### F3.4 — the projected energy, with its bias characterized

Report `E = Σ_j H_0j c_j / c_0`.

- **Verify:** on H2/STO-3G, `E` is within 3σ of `-1.1372744062` using G1's
  `metric_within_sigma` and G2's blocked error bar.
- **Verify the bias is a bias, not a bug:** measure `E` at several population
  targets and show the deviation from exact **shrinks with population**, roughly as
  1/N. A single population that happens to agree proves nothing; the *trend* is the
  evidence that the residual disagreement is the known finite-population bias.

  **Checked numerically before this scope was written** (4×4 toy `H`, Poisson
  sampling of the exact eigenvector, 4000 trials per point) so the step is not
  chasing a claim from memory:

  | N_walkers | mean E | bias | bias × N |
  |---|---|---|---|
  | 20 | −1.155547 | −1.82e-3 | −0.036 |
  | 80 | −1.154145 | −4.21e-4 | −0.034 |
  | 320 | −1.153896 | −1.72e-4 | −0.055 |
  | 1280 | −1.153785 | −6.1e-5 | −0.078 |
  | 5120 | −1.153747 | −2.2e-5 | −0.115 |

  `bias × N` stays within a factor of ~3 across a 256x population range, so ~1/N is
  the right expectation to gate against. Note the bias is **negative** here (the
  estimator sits below the exact value), which is the direction that would make a
  variational-looking result *more* convincing — another reason to gate the trend
  rather than a single number.
- **Open-shell fixture required.** F2 found that every closed-shell fixture was
  blind to an index bug that only appears when α and β counts differ. F3 must
  include one open-shell case for the same reason.

### F3.5 — determinism and the reproducibility gate

- **Verify:** a fixed seed reproduces the entire trajectory bitwise (G3's harness),
  across reruns. Serial only — threading is F5, and §6 of the research scope is the
  decision that gates it.
- **Verify:** the trajectory *changes* with a different seed. A run that ignores its
  seed would pass the first check trivially.

## What this must not do

- **Do not use `omp atomic` or a completion-order reduction** anywhere in the
  accumulate. F3 is serial; F5 decides the parallel policy, and the DFT-grid jitter
  defect is exactly this.
- **Do not tune `dt` until the tests pass.** `dt` has a derivable stability bound;
  if a test needs a smaller `dt` than the bound to pass, the bug is elsewhere.
- **Do not gate the energy on a single population size.** The projected energy is
  biased at finite population, so one agreeing number is not evidence. The trend is.
- **Do not gate only on H2/STO-3G.** 4 determinants cannot exercise doubles between
  different occupied pairs, and an open-shell case is required for the reason F2
  discovered.
- **Do not report an energy from a diverging or collapsing population without
  saying so.** With `S` fixed the population is not stationary by construction; an
  estimator quoted from a population that has collapsed to a handful of walkers is
  noise with a small error bar, which is the most misleading possible output.

## Key code locations

| what | where |
|---|---|
| walker state, RNG (F1) | `src/post_hf/ci/fciqmc.{h,cpp}` |
| excitation generator, `p_gen` (F2) | same |
| `H_ij` for the spawn | `slater_condon_element`, `ci.h:43` |
| `H_ii` for the death step | `build_ci_diagonal`, `ci.h:101` |
| dense `H`, for the F3.1 exact check | `build_ci_hamiltonian_dense`, `ci.cpp` |
| statistical gates | `tests/{blocking,reproducibility}.py`, `metric_within_sigma` |
| exact reference | `h2_fci_sto3g`, total FCI `-1.1372744062` |

---

Status lives in `vault/Status/Completion.md` and `vault/Status/Open Work.md`.
