# Why `numeric-newton` is the Preferred Orbital Step for Water CASSCF Runs

## Context

The CASSCF macro-optimizer evaluates multiple orbital step candidates at each
macroiteration and accepts whichever one best improves the merit function
`E_cas + 0.1·‖g_orb‖²`. For every water run tested (CAS(4,4)/STO-3G,
CAS(4,4)/6-31G, CAS(4,4)/cc-pVDZ at small active-space sizes), the
`numeric-newton` candidate wins every macroiteration, even when the full
coupled orbital/CI solver (`sa-coupled`) is available.

---

## 1. The Structural Cause: Tiny Non-Redundant Pair Space

For water CAS(4,4)/STO-3G the MO partitioning is:

| Block   | Count |
|---------|-------|
| Core    | 3     |
| Active  | 4     |
| Virtual | 0     |
| **Total MOs** | **7** |

The non-redundant orbital rotations are **core–active only**: 3 × 4 = **12 pairs**.
There are no core–virtual or active–virtual pairs because STO-3G water exhausts
all 7 basis functions with core + active orbitals.

The code gates `build_numeric_newton_candidate` on a hard limit
(`numeric_newton_pair_limit = 64`, [casscf.cpp:650](../src/post_hf/casscf/casscf.cpp#L650)):

```cpp
// casscf.cpp:969-970
const bool build_numeric_newton_candidate =
    use_numeric_newton_debug || static_cast<int>(opt_pairs.size()) <= numeric_newton_pair_limit;
```

With 12 pairs ≪ 64, the numeric-newton step is **always built** for water,
regardless of convergence state.

---

## 2. How `numeric-newton` Works

`build_numeric_newton_step` ([casscf.cpp:802–870](../src/post_hf/casscf/casscf.cpp#L802-870))
constructs the **exact fixed-CI orbital Hessian** column-by-column via
central finite differences:

1. For each rotation pair `k` in the non-redundant set:
   - Apply a ±5×10⁻⁴ orbital rotation along `e_k`
   - Re-run the full CASSCF evaluation (CI diagonalization + gradient) at each perturbed geometry
   - Take the central difference: `H[:,k] = (g(+ε) − g(−ε)) / 2ε`
2. Symmetrize `H ← ½(H + Hᵀ)`
3. Diagonalize, floor negative eigenvalues at `max(1e-4, level_shift)` (level-shifted Newton)
4. Compute the Newton step: `κ = −H⁻¹ g`
5. Cap the step at `max_rot = 0.20` radians

For water (12 pairs), this costs **24 extra full CASSCF evaluations** per
macroiteration — cheap for STO-3G (7 basis functions).

---

## 3. Why It Beats `sa-coupled`

### 3a. Off-diagonal Hessian coupling in the core–active block

All 12 non-redundant pairs for water CAS(4,4)/STO-3G are **core–active**.
This is the orbital block with the strongest off-diagonal Hessian coupling:
rotating a core orbital mixes it into the active space and vice versa,
inducing large cross terms between different rotation pairs.

The `sa-coupled` solver uses a **diagonal orbital Hessian preconditioner**
([orbital.h:130](../src/post_hf/casscf/orbital.h#L130), `hess_diag`) and
a matrix-free Hessian action ([response.cpp:719–725](../src/post_hf/casscf/response.cpp#L719-725))
applied through a block-Krylov iteration. The diagonal model systematically
underestimates the off-diagonal core–active coupling.

The numeric-newton step captures the **full 12×12 Hessian** exactly, including
all off-diagonal elements, yielding a better-conditioned Newton step.

### 3b. The `use_numeric_newton_fallback` trigger

```cpp
// casscf.cpp:1209-1212
const bool use_numeric_newton_fallback =
    use_numeric_newton_debug ||
    (static_cast<int>(opt_pairs.size()) <= numeric_newton_pair_limit &&
     (!coupled_step_reliable || stagnation_streak >= 2));
```

`coupled_step_reliable` requires `sa_coupled_result.converged == true`.
When the block-Krylov coupled solver does not converge all CI residuals below
tolerance — common in the first few macroiterations before the orbital space
is close to stationary — `coupled_step_reliable = false`, and the
numeric-newton candidate is included in the search **from macro 1 onward**.

### 3c. Candidate selection is a full competition

All candidates (in order: `sa-coupled`, `numeric-newton`, optionally
`sa-diag-fallback`, `sa-grad-fallback`) are evaluated at multiple step
scales `{1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625}`. The selection
loop ([casscf.cpp:1258–1312](../src/post_hf/casscf/casscf.cpp#L1258-1312))
keeps whichever (candidate, scale) pair achieves the lowest merit value. The
`numeric-newton` step consistently achieves a better descent than `sa-coupled`
because its Hessian is exact.

---

## 4. Evidence from the Log

Water CAS(4,4)/STO-3G (`water_cas44_sto3g.log`), all 7 macroiterations:

```
Macro   1  accepted=yes  candidate=numeric-newton  step_norm=7.18e-02  sa_g=1.12e-02
Macro   2  accepted=yes  candidate=numeric-newton  step_norm=1.13e-01  sa_g=1.24e-02
Macro   3  accepted=yes  candidate=numeric-newton  step_norm=1.79e-01  sa_g=1.22e-02
Macro   4  accepted=yes  candidate=numeric-newton  step_norm=2.52e-01  sa_g=7.55e-03
Macro   5  accepted=yes  candidate=numeric-newton  step_norm=2.22e-01  sa_g=1.61e-03
Macro   6  accepted=yes  candidate=numeric-newton  step_norm=6.18e-02  sa_g=5.13e-05
Macro   7  accepted=yes  candidate=numeric-newton  step_norm=1.65e-03  sa_g=4.59e-08
```

The gradient norm drops from ~10⁻² to 4.6×10⁻⁸ in 7 macroiterations, with
quadratic convergence near the end — exactly the signature of a true Newton method.

---

## 5. Why This Does Not Apply to Larger Systems

For systems with many virtual orbitals (e.g., ethylene CAS(4,4)/cc-pVDZ,
CAS(4,4)/6-31G with a larger basis), the number of non-redundant pairs exceeds
the 64-pair limit:

| System            | Basis   | Core | Active | Virtual | Pairs |
|-------------------|---------|------|--------|---------|-------|
| Water CAS(4,4)    | STO-3G  | 3    | 4      | 0       | 12    |
| Water CAS(4,4)    | 6-31G   | 3    | 4      | 6       | 32    |
| Water CAS(4,4)    | cc-pVDZ | 3    | 4      | 17      | 62    |
| Ethylene CAS(4,4) | STO-3G  | 5    | 4      | 3       | 44    |
| Ethylene CAS(4,4) | cc-pVDZ | 5    | 4      | 19      | 110   |

Once pairs > 64, `build_numeric_newton_candidate = false` and the
`sa-coupled` solver is the sole production path. The FD cost (2 × npairs
full evaluations) would become prohibitive for large bases.

---

## 6. Summary

| Factor | Effect |
|--------|--------|
| 12 non-redundant pairs (STO-3G, no virtual) | Always under the 64-pair gate; numeric-newton always built |
| All pairs are core–active | Strongest off-diagonal Hessian block; diagonal preconditioner weakest here |
| Exact 12×12 FD Hessian | Full off-diagonal information; true Newton step |
| 24 extra evaluations/macroiter | Negligible cost for 7-function STO-3G water |
| `coupled_step_reliable = false` early on | Numeric-newton included in candidate set from macro 1 |

The `numeric-newton` candidate is not a fallback for water — it is structurally
the best step available because the active space is small enough that the exact
Hessian is affordable and the approximate coupled solver cannot compete with it
in the core–active rotation block.
