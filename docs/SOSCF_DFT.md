# DFT SOSCF (RKS and UKS)

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does second-order SCF work for Kohn-Sham DFT — how is the orbital
Hessian composed (analytic `fxc` contraction, not the finite-difference XC
kernel), how is each term scaled, and how do hybrids fit in?**

## Short answer

DFT SOSCF is the RHF/UHF pattern (`docs/SOSCF.md`, `docs/SOSCF_UHF.md`) with
one substitution: the KS orbital Hessian has no single dense matrix like
`build_rhf_cphf_matrix`. Its `h_op` is composed inline from four
linear-in-density pieces:

```
h_op(x) = diag_term ⊙ x  +  s · (J_packed + xc_packed + K_packed)
```

- `diag_term` — the orbital-energy-difference diagonal `ε(n_occ+a) - ε(i)`,
  virtual-major (`a·n_occ + i`). `DFT::Driver::orbital_energy_difference_diagonal`.
- `J_packed` — the Coulomb response. `J` is linear in the density, so
  `δJ(δP) = _compute_2e_j_direct(shell_pairs, δP, …)` — the same
  memory-direct builder the KS loop uses every iteration, on the trial
  `δP`. No new function.
- `xc_packed` — the analytic XC second-derivative contribution
  (`docs/DFT_ANALYTIC_FXC_HESSIAN.md`), then
  `pack_hessian_vector_product_cphf_order`.
- `K_packed` — **hybrids only**, the exact-exchange response. `K` is linear
  in the density exactly as `J` is, so this is just one more term of the
  same shape (see invariant 3).
- `s` — the kernel-vs-diagonal scale. **`s = 2` for RKS, `s = 1` for UKS**
  (invariant 2). The old code used `s = 1` for RKS too, which
  half-weighted the kernel and made the Newton *direction* slightly wrong;
  fixing it is what turned RKS PBE/LDA from linear to superlinear.

Everything else is reused: `solve_augmented_hessian`,
`apply_orbital_rotation`, per-spin semicanonicalization, the `scf_soscf_*`
window logic. Off by default.

**All hybrids are supported — RKS and UKS, global (B3LYP, PBE0) and
range-separated (HSE06).** Only **PCM** and **SAO/symmetry** emit a
one-time warning and fall back to plain DIIS.

## Where the logic lives

- `src/dft/driver.cpp` — the RKS SOSCF branch in `run_ks_scf_scaffold`'s
  `!unrestricted` loop, and the UKS SOSCF branch in the unrestricted loop;
  `C_soscf_prev`/`eps_soscf_prev` (RKS) and `Ca_soscf_prev`/`Cb_soscf_prev`/
  `epsa_soscf_prev`/`epsb_soscf_prev` (UKS) persisted every iteration;
  `soscf_enabled` gate excluding only `pcm` / SAO; the one-time
  `[WRN] DFT SOSCF :` warning block; the step deadband (invariant 7)
- `src/dft/analytic_hessian.{h,cpp}` — `orbital_energy_difference_diagonal`,
  and the `compute_analytic_xc_hessian_vector_product{,_polarized}`
  functions (`docs/DFT_ANALYTIC_FXC_HESSIAN.md`, including invariant 3a
  there — the combined-XC `fxc` guard)
- `src/dft/response_packing.{h,cpp}` — `pack_hessian_vector_product_cphf_order`
- `src/integrals/base.h` — `_compute_2e_j_direct`, `_compute_2e_k_direct`,
  `_compute_2e_k_uhf_direct`, reused for `δJ` / `δK`
- `src/post_hf/casscf/aug-hessian.h` — `solve_augmented_hessian`, reused
- `src/dft/driver.cpp` — `build_{closed_shell,unrestricted}_xc_kernel_blocks`,
  the FD-kernel oracle: verification only, not the production Hessian

## What invariants matter

### 1. The analytic path is wired directly; the FD-kernel oracle is not the production Hessian

Decision D1 originally chose the FD-kernel oracle first — cheap to reach,
scoped explicitly as a small-system correctness reference, because as a
SOSCF Hessian it is `O(n_occ·n_virt)` full-grid XC evaluations per Newton
step. That went stale: the analytic `fxc` path
(`docs/DFT_ANALYTIC_FXC_HESSIAN.md`) was built and verified first, so the
reference it was meant to serve as had already been consumed. Wiring a
second dense `O(n_occ·n_virt)`-grid-pass Hessian source when a verified
`O(1)` one exists would leave the production speedup undelivered.

Design rule:

- DFT SOSCF wires `compute_analytic_xc_hessian_vector_product`. The FD
  oracle stays in-tree as an independent correctness check for the wiring
  itself, never as the runtime Hessian.

### 2. The kernel scales `s = 2` for RKS and `s = 1` for UKS — a curvature/kernel split, not a single constant

The exact Newton step is `−H_true⁻¹ g_true`. In the κ-parametrization
`E(κ)` uses (rotation `κ_{ai}` in the α/β blocks), for the **closed-shell
RKS** density (occupancy 2, so `dP/dκ = [R, P₀] = 2·dP` where `dP` is the
`C_v·e·C_oᵀ + h.c.` convention `h_op` and
`compute_analytic_xc_hessian_vector_product` build):

- **curvature term** `Tr(d²P/dκ²·F₀)` with `F₀` diagonal → `4·(ε_a − ε_i)`
  per unit direction. `h_op` supplies `diag_term ⊙ x`, so this wants `4×`.
- **kernel term** `Tr(dP/dκ · δV(dP/dκ))` — bilinear in `dP/dκ = 2·dP`, so
  `= 4·Tr(dP·δV(dP))`, and `pack(δV(dP))_{ai} = ⟨a|δV(dP)|i⟩` while
  `Tr(dP·δV(dP)) = 2·⟨a|δV(dP)|i⟩`. So the true kernel is
  `8·[J_packed + xc_packed + K_packed]_{ai}` — **`8×`, not `4×`**.
- `g_true = ∂E/∂κ_{ai} = Tr(dP/dκ·F₀) = 4·F_mo(a,i)`.

Divide `g_true` and `H_true` by 4: pair `g = F_mo` (unscaled) with
`h_op(x) = diag ⊙ x + 2·(J + V_xc + K)`. That is `s = 2`.

**The old code used a bare `(J + V_xc)` — `s = 1`.** Magnitude was fine
(the diagonal dominates), but the kernel correction to the Newton
*direction* was half-weighted, which is why RKS water/6-31G/PBE converged
*linearly* under SOSCF where it should be superlinear (invariant 6 used to
record this as an unexplained finding; it is the scale bug). Verified: on
Slater/VWN5 (LDA — exact `δV_xc`) the composed `4·diag + 8·kernel` lands
on a central FD of the true total energy to ratio `1.000000` at three
directions as `h → 1e-4`, where the single `4×` misses by up to 1.8%.

**UKS is uniform `s = 1`.** The occupancy-1 per-spin density gives
`dP^σ/dκ = 1·dP^σ`, so both the curvature term (`2·(ε_a − ε_i)`) and the
kernel term (`2·Tr(dP^σ·δV(dP^σ))`) scale by the **same** factor 2 —
`H_true = 2·H_bare` with `H_bare = diag + kernel` uniform, `g_true = 2·g_bare`.
No RKS-style split.

Design rule:

- Verify the composed `h_op` against a finite difference of the **total**
  energy `E(κ)`, never `compute_analytic_xc_hessian_vector_product` alone
  against `E_xc(κ)` — the latter is missing the density-curvature term
  (`docs/DFT_ANALYTIC_FXC_HESSIAN.md`), which was D2.1's original mistake.
- RKS and UKS have different scale conventions — do NOT copy the RKS `2×`
  kernel factor into the UKS branch, and do not copy the UKS uniform
  scaling into RKS.
- LDA is the load-bearing check: `δV_xc` is exact there, so a ratio ≠ 1
  isolates a scale error. GGA has a small FD-cancellation floor on
  small-magnitude directions (invariant 8).

### 3. A hybrid's K response is one more linear-in-density term, matching the KS build's decomposition

The KS potential build (`assemble_current_ks_potential`) adds
`-0.5 · (c_fr·K_Coulomb[P] + c_sr·K_ShortRange[P, ω])` to the RKS Fock and
`-1.0 · (c_fr·K_C[Pσ] + c_sr·K_SR[Pσ, ω])` per spin to the UKS Fock. `K` is
linear in the density, so `h_op` gains the same term on the trial `δP`,
same coefficients, same sign, same prefactor:

- **RKS:** accumulate `c_fr·K_C[δP] + c_sr·K_SR[δP]` (both via
  `_compute_2e_k_direct`, `Coulomb` and `ShortRange` kernels) into one
  `dK`, pack `-0.5·dK`.
- **UKS:** `_compute_2e_k_uhf_direct(shell_pairs, δPa, δPb, …)` once per
  kernel gives `(δKa, δKb)`; accumulate `c_fr·δKa + c_sr·δKa` into `dKa`
  (and `dKb`), pack `-1.0·dKa` into the α block, `-1.0·dKb` into the β
  block.

Global hybrids (B3LYP, PBE0): `c_sr == 0`. Pure short-range screened
hybrids (HSE06): `c_fr == 0`. CAM-B3LYP: both nonzero — covered by the
accumulation by construction.

**No new integral code, no new CPHF matrix, no new scale convention** —
the direct K builders are already in the tree and already
screened-kernel-aware.

Two prerequisites had to land first, each recorded elsewhere:

- **The RKS kernel scale** (invariant 2 above). A hybrid's K term rides
  the same `s = 2` as J and V_xc; before the scale fix it was
  half-weighted along with them.
- **The combined-XC `fxc` double-count**
  (`docs/DFT_ANALYTIC_FXC_HESSIAN.md` invariant 3a). Every named hybrid
  (B3LYP, PBE0, HSE06) is a *combined* exchange-correlation libxc entry,
  so `compute_analytic_xc_hessian_vector_product{,_polarized}` was
  double-counting a correlation `fxc` for them. Guarded now.

Together those two are the reason the B3LYP/PBE0 scale probe went from a
0.7–5.6% direction-dependent miss to ratio `1.000000`.

### 4. UKS packing matches build_uhf_cphf_matrix — not a third convention

Two blocks in one vector: `[0, nova)` alpha, `[nova, nova+novb)` beta,
virtual-major within each. `δJ` is one call on the **total** trial density
`δP^α + δP^β` (matching the KS loop's own Coulomb build), then packed
separately into the α and β `(a,i)` blocks; `δK` is spin-resolved (α from
`δPa`, β from `δPb`) via `_compute_2e_k_uhf_direct`. Cayley rotation and
semicanonicalization run per spin channel — one shared step vector, two
`κ` matrices.

Design rule:

- Reuse the existing virtual-major CPHF convention
  (`pack_hessian_vector_product_cphf_order`, `build_uhf_cphf_matrix`) — the
  codebase already has two packings (that one and
  `ResponseExcitationSpace::flat_index`'s occupied-major); do not add a
  third.

### 5. Level shift is a DFT-wide no-op; semicanonicalization is unnecessary but kept

`grep -c level_shift src/dft/driver.cpp` returns 0 —
`calculator._scf._level_shift` is never read by the KS loop, RKS or UKS.
Verified: `level_shift 0.5` set vs absent gives byte-identical energy and
iteration count under plain DIIS. No guard needed, unlike UHF where the
field is live. Semicanonicalization disabled over a 200-cycle pure-SOSCF
window: RKS water/6-31G/PBE 126 vs 121 iterations, UKS triplet
water/6-31G 27 vs 30 — identical energy either way.

Design rule:

- Do not port RHF/UHF's level-shift guard to DFT — check the actual KS
  loop for the knob first. Keep semicanonicalization: pure gauge freedom,
  cheap.

### 6. A scope-cut gate needs a message, or the user's request is silently dropped

`soscf_enabled` excludes only PCM and SAO now — all hybrids (RKS and UKS,
global and range-separated) are in. A user setting `scf_soscf_start` by
habit on a PCM or SAO run got no warning and silently ran plain DIIS. A
one-time `[WRN] DFT SOSCF :` line is emitted before the loop when SOSCF is
requested but disabled, naming the reason and confirming the calculation
is still valid, just unaccelerated. Not a hard error — that would newly
fail every ordinary such DFT run with `scf_soscf_start` set.

Design rule:

- A gate that disables a requested feature emits a warning naming the
  specific reason. (RHF/UHF's own analogous guards do not yet — a small
  follow-on across all three SOSCF paths.)

### 7. Pure-SOSCF has a limit cycle at the density noise floor; a step deadband gives it a termination

With no DIIS handoff (`scf_soscf_cycles` large), the augmented-Hessian
solve returns a nonzero rotation for any `g ≠ 0`, so the Cayley transform
+ Löwdin cleanup keeps perturbing the density by `~1e-10` forever. At
`tol_density ≲ 1e-11` the run then hits `max_cycles` without converging —
this is **pre-existing** (the RKS scale fix improved the limit-cycle floor
~15×, it did not create it) and **grid-independent** (identical at
normal/fine/ultrafine).

The deadband: when `max|step| < tol_density`, hold the reference orbitals
(`C_new = C_soscf_prev`, `eps_new = eps_soscf_prev`), so
`next_density == density`, `RMS(D) = Max(D) = 0`, and `is_converged` can
fire. The DIIS-handoff mode (default `scf_soscf_cycles = 3`) never reaches
this — it hands back long before the step gets that small. Gated by
`water_rks_pbe_soscf_puredeadband_631g`.

### 8. The gradient-shrinkage shape, and where it is genuinely limited

Post the RKS scale fix, RKS PBE and LDA converge **superlinearly** (each
SOSCF step drops `|g|` ~50×), which the old `s = 1` code did not. The
recorded "linear on water/6-31G/PBE" finding was the scale bug.

Two residual limits, both **pre-existing and unrelated to the K term**:

- **The polarized GGA `fxc` for B88/LYP-based functionals** has a
  ~1e-4..1e-3 relative residual on small-magnitude directions in a
  whole-molecule FD probe. Verified this is **FD cancellation in the
  probe**, not an algebra error: the point-level FD of `δV_xc` for
  `gga_x_b88` / `gga_c_lyp` matches the analytic T1..T5 to ~1e-10, the
  same as PBE (`dft_gga_hessian_selfcheck` and its polarized twin now
  cover B88/LYP, closing the coverage gap that made this ambiguous). A
  second difference of `E ~O(75)` resolving `composed ~O(1)` at `h = 1e-4`
  carries ~1e-3 relative noise; PBE0 (analytic PBE grid) lands on
  `1.000000`, B88/LYP grid sums scatter more.
- **Triplet-radical UHF landscapes are shallow.** SOSCF's Newton steps
  escape spurious stationary points DIIS gets stuck near and can land in a
  UHF basin ~1e-4 Eh **lower** than DIIS's. That is SOSCF working, but it
  means a UKS hybrid gate cannot assert "10-digit vs DIIS" on triplet
  water — it needs a clean single-minimum open-shell system (the water
  cation, `h2o_cation_uks_pbe0_soscf_631g`).

Design rule:

- DFT SOSCF's pass criterion is energy agreement with fully-converged
  DIIS first (on a system where DIIS *has* a unique minimum), gradient
  shape second. LDA is the exact reference; GGA carries a small FD floor
  on small directions.

## What was fixed / built

1. **`orbital_energy_difference_diagonal`** — the RKS/UKS analogue of the
   one line `A(ai,ai) += eps(a) - eps(i)` inside `build_rhf_cphf_matrix`,
   standalone because DFT has no such matrix to reuse.

2. **The RKS `h_op` composition and SOSCF branch** — `diag_term`,
   `_compute_2e_j_direct` on `δP` for `J`, the analytic XC piece, the
   hybrid K term, kernel-scaled `s = 2`, packed virtual-major;
   `solve_augmented_hessian`; `kSoscfMaxRot = 0.20`;
   `apply_orbital_rotation`; occ-occ/virt-virt semicanonicalization; the
   step deadband.

3. **The UKS SOSCF branch** — the polarized analogue, uniform `s = 1`.
   Per-spin persisted state, `[0,nova)+[nova,novb)` packing, `δJ` from the
   total trial density, spin-resolved `δK`.

4. **The RKS kernel-scale fix** — `s = 1 → 2`, split as `4·diag +
   8·kernel` in the raw κ-parametrization. Superseded the old
   `H_true = 4·H_bare` single constant.

5. **The combined-XC `fxc` guard** — `drop_correlation_if_combined` in
   `analytic_hessian.cpp` (`DFT_ANALYTIC_FXC_HESSIAN.md` invariant 3a).

6. **The hybrid K term** — RKS `-0.5·(c_fr·K_C + c_sr·K_SR)`, UKS
   `-1.0·(...)` per spin, all built from the direct K builders.

7. **The one-time PCM/SAO warning** and the **switch trigger** (zero new
   code — structural copy of RHF's).

## Regression gates

| gate | what |
|---|---|
| `water_rks_lda_soscf_631g` | RKS LDA: SOSCF == DIIS to 1e-9; `dft_soscf_last_gradient ≤ 5e-4` (superlinear post-scale-fix, linear pre) |
| `water_rks_pbe_soscf_puredeadband_631g` | pure-SOSCF PBE at `tol 1e-11` converges via the deadband, not `max_cycles` |
| `water_rks_b3lyp_soscf_631g` | RKS global hybrid: energy == DIIS to 1e-9, no hybrid-blocked warning, superlinear |
| `water_rks_hse06_soscf_631g` | RKS range-separated hybrid: same, `≤ 5e-5` (the c_sr K branch tightens it ~9×) |
| `h2o_cation_uks_pbe0_soscf_631g` | UKS global hybrid on a clean single-minimum doublet: energy == DIIS to 1e-9, superlinear |
| `dft_analytic_hessian_polarized_production` :: `check_combined_no_double_count` | the `correlation_functional` arg is inert for a combined XC functional (B3LYP, PBE0), RKS + polarized |
| `dft_gga_hessian_selfcheck` + polarized twin | T1..T5 `δV_xc` vs grid-level FD for PBE, **B88, and LYP** |

Every `dft_soscf_last_gradient` gate is non-vacuity-verified against the
relevant mutation (kernel weight `2 → 1`, or dropping the c_sr K branch).

## Remaining architecture concern

### Wall-clock: RKS ≈ break-even, UKS a clear win

Measured one call to the analytic `h_op` (one `(a,i)` unit vector — the
Krylov-loop shape) against one call to the FD-kernel oracle building the
full dense Hessian, alongside real `ah_iters`. The FD oracle's
per-Newton-step cost is `≈ fd_kernel_full` regardless of `ah_iters`; the
analytic path's is `ah_iters × analytic_call`.

**RKS — not reliably faster at the two sizes measured:**

| system | `nov` | analytic (1 call) | FD-kernel (full) | crossover `ah_iters` | real `ah_iters` |
|---|---|---|---|---|---|
| water/6-31G/PBE | 40 | 0.131 s | 0.513 s | ≈3.9 | 6, 7, 4 |
| water/cc-pVDZ/PBE | 100 | 0.141 s | 2.131 s | ≈15.1 | 4, 23, 10 |

**UKS — reliably 3–5× faster at every size, the opposite of RKS:**

| system | `nova`/`novb` | analytic (1 call) | FD-kernel (full) | crossover `ah_iters` | real `ah_iters` |
|---|---|---|---|---|---|
| water/STO-3G/PBE triplet | 6 / 12 | 0.017 s | 0.155 s | ≈9.2 | 3, 3, 3 |
| water/6-31G/PBE triplet | 42 / 36 | 0.018 s | 0.50 s | ≈28 | 5, 6, 7 |
| water/cc-pVDZ/PBE triplet | 114 / 84 | 0.064 s | 2.06 s | ≈32 | 8, 10, 10 |

The `O(1)` vs `O(n_occ·n_virt)` asymptotic argument is real; whether it is
a wall-clock win depends on `ah_iters` vs the crossover at the actual
system size — comfortably favorable for UKS at modest sizes, borderline
for RKS there. The analytic path is correct and correctly-scaling in
both, and is the only path that scales to a large active space where the
FD oracle's one-time build itself becomes prohibitive. The hybrid K term
adds one direct sweep per Krylov iteration — same cost class as the `δJ`
sweep already there; the table's numbers are not re-measured for hybrids.

### Not done

- **PCM, SAO/symmetry** — still rejected with a warning. PCM needs the
  reaction-field response; SAO needs the block-diagonal response.
- **`scf_soscf_diis_tol` DFT-specific default** — not required
  (`_scf_soscf_diis_tol = 0.0`, SOSCF is opt-in; the criterion mechanism
  is verified working). A DFT-tuned default is a sweep for when the
  large-`nb` cluster ladder is available.
- **The large-`nb` motivation** is unreproduced for RKS (cluster access);
  measured and positive for UKS at modest sizes.
