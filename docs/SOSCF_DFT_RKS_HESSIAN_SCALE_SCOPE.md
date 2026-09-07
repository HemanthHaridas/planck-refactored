# RKS SOSCF Hessian scale defect — scope

In-flight scope. Fold the finding into `docs/SOSCF_DFT.md` invariant 2 and
delete this file when the fix lands (per `docs/docs_answer_one_question.md`).
Found while verifying `docs/SOSCF_DFT_HYBRID_SCOPE.md` step H1.

**Question this work answers:** the shipped RKS SOSCF `h_op` scales its
whole output by one constant (`H_true = 4·H_bare`, `docs/SOSCF_DFT.md`
invariant 2). That is correct for the orbital-energy-difference
(curvature) term but a factor of 2 too small for the kernel term
(`J_packed + xc_packed`). What does it take to fix the composition so the
Newton *direction* is right, not just its magnitude?

## Short answer

For a closed-shell RKS reference the exact second derivative of the total
energy along a single rotation `κ = t·(E_ai − E_ia)` splits as:

```
d²E/dt²  =  Tr(d²P/dt² · F₀)              (curvature)
         +  Tr(dP/dt · δV(dP/dt))         (kernel: J + V_xc + cₓ K)
```

with `P` the **occupancy-2** density, so `dP/dt = [R, P₀] = 2·dP` where
`dP = C_v·e·C_oᵀ + h.c.` is the convention `h_op` and
`compute_analytic_xc_hessian_vector_product` both build.

- **Curvature.** `Tr(d²P/dt²·F₀)` with `F₀` diagonal (`ε`) evaluates to
  `4·(ε_a − ε_i)` per unit direction. `h_op` supplies
  `diag_term ⊙ x = (ε_a − ε_i)·x`, so `4·diag_term` is exact. ✓
- **Kernel.** Bilinear in `dP/dt = 2·dP`, so
  `Tr(dP/dt·δV(dP/dt)) = 4·Tr(dP·δV(dP))`. And
  `pack_hessian_vector_product_cphf_order(δV(dP))_{ai} = ⟨a|δV(dP)|i⟩`
  while `Tr(dP·δV(dP)) = 2·⟨a|δV(dP)|i⟩` (δV symmetric). So the true
  kernel term is `8·[J_packed + xc_packed + K_packed]_{ai}`, **not 4×**.

The single `4×` in `H_true = 4·H_bare` therefore halves the kernel
contribution. Since the kernel is O(1%) of the diagonal for most
directions, `H_bare` is not a uniform multiple of `H_true`, so
`−H_bare⁻¹·g` is a slightly wrong *direction*, not merely a scaled step.
This is the likely cause of `docs/SOSCF_DFT.md` invariant 6's unexplained
"linear, not superlinear on water/6-31G/PBE" finding.

**UKS is already correct and must not be touched.** For the
occupancy-1 per-spin density, `dP^σ/dt = [R^σ, P₀^σ] = 1·dP^σ`, so both
the curvature term (`2·(ε_a−ε_i)`) and the kernel term
(`2·Tr(dP^σ·δV(dP^σ))`) scale by the **same** factor 2.
`H_true = 2·H_bare` is uniform and exact. The defect is RKS-only.

## Evidence (H1 probe, water/6-31G, `PLANCK_SOSCF_HYBRID_CHECK`)

Central-FD of the true total energy `E(κ)` vs the composed `h_op`, three
`(a,i)` directions, `h = {1e-2, 1e-3, 1e-4}`:

| functional | single `4×` ratio | `4×` diag + `8×` kernel ratio |
|---|---|---|
| Slater/VWN5 (LDA) | 0.982 – 1.004 | **1.000000 – 1.000001** |
| PBE (GGA) | 0.963 – 0.999 | ~0.993 – 1.002 (residual = GGA T3, separate) |
| B3LYP (GGA hybrid) | 1.001 – 1.008 | ~0.993 – 1.002 (same GGA T3 residual) |

The LDA row is the clean test: `δV_xc` is exact there (no T3
approximation), and the split convention lands on `1.000000` at all three
directions while the single `4×` misses by up to 1.8%. The residual on
the GGA rows after the split is a **different, pre-existing** limitation of
`compute_analytic_xc_hessian_vector_product`'s T3 term
(`docs/DFT_ANALYTIC_FXC_HESSIAN.md`) — direction-dependent sign, not a
scale — and is out of scope here.

## Where the code changes

`src/dft/driver.cpp`, RKS SOSCF branch in `run_ks_scf_scaffold`'s
`!unrestricted` loop:

- the `h_op` lambda (currently returns `diag ⊙ x + J_packed + xc_packed
  [+ K_packed]`)
- the `g` vector (`g(a·n_occ+i) = F_mo(n_occ+a, i)`) and the `x0 = -g`
  seed
- the trust-region cap `kSoscfMaxRot` interpretation (a rescaled `h_op`
  changes the natural step magnitude the cap sees)
- the comment block at `driver.cpp:2016-2020` stating
  `H_true = 4*H_bare` as "a MATCHING pair"

The UKS branch (`driver.cpp:2603+`) is **not** touched.

## The fix

Return the correctly-scaled Hessian-vector product from `h_op`:

```cpp
// RKS: curvature scales 4x, kernel (J + V_xc + K) scales 8x.
return 4.0 * diag_term.cwiseProduct(x)
     + 8.0 * (J_packed + xc_packed + K_packed);
```

and pair it with the true gradient `g_true = 2·g_bare`:

```cpp
g(a * n_occ_i + i) = 2.0 * F_mo(n_occ_i + a, i);
```

`solve_augmented_hessian` then solves `−H_true⁻¹ g_true`, the exact Newton
step. The AH solver only ever uses `h_op` and `g_op` together, so the pair
must move together — scaling one without the other changes the step.

Alternative considered and rejected: keep `h_op` returning `H_bare` and
apply the `4/8` split as a post-multiply outside the lambda. Rejected
because `solve_augmented_hessian` calls `h_op` internally in its Krylov
loop — the scaling has to be inside the callback, not wrapped around the
result.

## Steps

Each step is independently verifiable. Steps S1–S3 are the fix; S4–S5 are
regression coverage; S6 is doc cleanup.

### S1 — extend the H1 probe to assert the split, LDA only

The H1 probe already exists (`PLANCK_SOSCF_HYBRID_CHECK`,
`src/dft/driver.cpp`). Add a hard assertion path: when the functional is
LDA-like (`x_functional.is_lda_like()`), require every direction's
`(4·diag + 8·kernel)/FD` ratio within `1e-4` of 1.0 as `h → 1e-4`, and
`abort()` / log `[ERR]` if not. Leave GGA directions reported but not
asserted (the T3 residual is separate).

**Verify:** on Slater/VWN5 water/6-31G the probe passes silently; a
deliberate revert to a single `4×` inside the probe's `composed` makes it
fail naming the direction and ratio. This is the load-bearing correctness
gate for the convention — it must fail on the old scaling before the fix
is trusted.

### S2 — apply the scaling in the RKS `h_op` and `g`

Change the RKS `h_op` return to `4·diag ⊙ x + 8·(J + xc + K)` and
`g(a·n_occ+i) = 2·F_mo(n_occ+a, i)`. Nothing else.

**Verify:**
- `x0 = -g` still finite; `solve_augmented_hessian` still converges (`ah`
  result `converged=true`, `ah_iters` in single digits) on
  Slater/VWN5, PBE, and B3LYP water/6-31G.
- The S1 probe (now reading the production `h_op`) passes the LDA
  assertion.
- **The step is unchanged to within the old direction error**: log
  `‖κ_new − κ_old‖ / ‖κ_old‖` for the first SOSCF iteration. Expect
  O(1e-2) (the kernel/diag ratio) — a much larger change means the `g`
  scaling was missed.

### S3 — re-check the trust-region cap

`kSoscfMaxRot = 0.20` caps `step.cwiseAbs().maxCoeff()`. With `g` doubled
and `H` rescaled non-uniformly the raw Newton step magnitude shifts.
`docs/SOSCF_DFT.md` invariant 2 already records the cap is "rarely
binding" for RKS; confirm `cap_fired` stays `false` on the three test
systems from a converged-DIIS handoff (`scf_soscf_diis_tol 1e-3`).

**Verify:** `cap_fired=false` logged on all three; if it fires, the cap
constant needs a matching adjustment and that becomes part of this step.

### S4 — RKS SOSCF convergence-shape regression

The motivating symptom (`docs/SOSCF_DFT.md` invariant 6) is RKS
water/6-31G/PBE converging *linearly* under SOSCF. Add a regression case:
RKS PBE (or Slater/VWN5) water/6-31G, `scf_soscf_start` set, asserting
(a) SOSCF energy agrees with fully-converged plain DIIS to 10 digits
(unchanged requirement), and (b) the orbital-gradient ratio between
consecutive SOSCF iterations **decreases** (superlinear), not holding
near-constant.

**Verify:** the new case passes. If (b) still shows a constant ratio on
GGA, that isolates the remaining shape defect to the GGA T3 residual
(S-scope for `DFT_ANALYTIC_FXC_HESSIAN.md`), not the scaling — record
which. LDA should be unambiguously superlinear.

### S5 — full DFT regression + smoke, SOSCF off byte-identical

**Verify:** `ctest -R dft` (12/12) and the smoke suite unchanged. Every
SOSCF-off DFT run byte-identical to pre-fix (the `h_op` change is inside
`if (soscf_active)`, dead when SOSCF is not requested).

### S6 — update `docs/SOSCF_DFT.md`

- Invariant 2 (RKS): replace "`H_true = 4·H_bare`, a matching pair like
  RHF's 4-and-4" with the split: curvature `4×`, kernel `8×`, and the
  reason (`dP/dt = 2·dP` for occupancy-2, kernel bilinear). Keep the UKS
  paragraph unchanged — `2×` uniform is correct there and this scope
  confirmed it.
- Invariant 6: note the RKS "linear on water" finding is (partly or
  fully — per S4's result) explained by this scaling defect; update or
  remove the "open, recorded finding" language accordingly.
- The `driver.cpp:2016` comment: fix in the same commit.

**Verify:** doc no longer claims a single `4×` for RKS; `CLAUDE.md`
regenerated by the hook; `vault/Status/Completion.md` and
`vault/Status/Open Work.md` note the fix.

## Not in scope

- **UKS.** Analysed and confirmed correct (`2×` uniform). No change.
- **The GGA T3 residual** in
  `compute_analytic_xc_hessian_vector_product` — direction-dependent
  ~0.7% error on PBE/B3LYP after the scaling fix. Pre-existing,
  `docs/DFT_ANALYTIC_FXC_HESSIAN.md`'s subject, needs its own scope.
- **RHF/UHF SOSCF** (`docs/SOSCF.md`, `docs/SOSCF_UHF.md`) — those pair
  their own `build_{rhf,uhf}_cphf_matrix` with a measured constant and
  are not composed the DFT way; not affected by this finding. (RHF's own
  "4-and-4" is a genuine matching pair because its CPHF matrix already
  carries the kernel at full weight.)
- **PCM / SAO** — still rejected for DFT SOSCF regardless.

## Risks

- **Missing the `g` scaling.** If `h_op` is rescaled but `g` is left at
  `F_mo` (not `2·F_mo`), the Newton step is off by 2× in magnitude —
  the trust-region cap then fires constantly (S3 catches this).
- **The `kSoscfMaxRot` interaction.** A non-uniform `H` rescale changes
  what "a 0.20 max rotation" means relative to the true step; S3 exists
  to catch a cap that starts binding.
- **GGA masking.** The GGA T3 residual (~0.7%) is larger than the LDA
  verification tolerance, so S1 must assert on LDA only. Asserting on GGA
  would either fail spuriously or force a loose tolerance that can't see a
  future scaling regression.
