# DFT SOSCF (RKS and UKS)

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does second-order SCF work for Kohn-Sham DFT, and why is its orbital
Hessian composed from an analytic `fxc` contraction rather than the
finite-difference XC kernel?**

## Short answer

DFT SOSCF is the RHF/UHF pattern (`docs/SOSCF.md`, `docs/SOSCF_UHF.md`) with
one substitution: the KS orbital Hessian has no single dense matrix like
`build_rhf_cphf_matrix`. Its `h_op` is composed inline:

```
h_op(x) = diag_term ⊙ x  +  J_packed  +  xc_packed
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

Everything else is reused: `solve_augmented_hessian`,
`apply_orbital_rotation`, per-spin semicanonicalization, the `scf_soscf_*`
window logic. Off by default. **Pure (non-hybrid) functionals only** —
hybrid / PCM / SAO emit a one-time warning and fall back to plain DIIS.

## Where the logic lives

- `src/dft/driver.cpp` — the RKS SOSCF branch in `run_ks_scf_scaffold`'s
  `!unrestricted` loop, and the UKS SOSCF branch in the unrestricted loop;
  `C_soscf_prev`/`eps_soscf_prev` (RKS) and `Ca_soscf_prev`/`Cb_soscf_prev`/
  `epsa_soscf_prev`/`epsb_soscf_prev` (UKS) persisted every iteration;
  `soscf_enabled` gate excluding `is_hybrid()` / `pcm` / SAO; the one-time
  `[WRN] DFT SOSCF :` warning block
- `src/dft/analytic_hessian.{h,cpp}` — `orbital_energy_difference_diagonal`,
  and the `compute_analytic_xc_hessian_vector_product{,_polarized}`
  functions (`docs/DFT_ANALYTIC_FXC_HESSIAN.md`)
- `src/dft/response_packing.{h,cpp}` — `pack_hessian_vector_product_cphf_order`
- `src/integrals/base.h` — `_compute_2e_j_direct`, reused for `δJ`
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
  itself (the role a `PLANCK_SOSCF_FD_CHECK`-style probe played for RHF/UHF),
  never as the runtime Hessian.

### 2. The scale convention is a matching pair, measured against PySCF — a different constant for RKS and UKS

Cross-checked against PySCF's `newton_ah.gen_g_hop_rhf` at multiple `(a,i)`
directions on identical water/STO-3G/PBE input, before writing Planck code.

**RKS:** `g_true = 2·g_bare` (`g_bare = F_mo(a,i)`),
`H_true = 4·H_bare` (`H_bare = diag_term + J_packed + xc_packed`). A
matching pair like RHF's 4-and-4, so the unscaled ratio `g_bare/H_bare`
already equals `g_true/H_true`. Verified in Planck's own code: `4·H_bare`
matches a central FD of the true total energy to ratio `1.000000` at three
directions, `h = {1e-2, 1e-3, 1e-4}`.

**UKS:** `d²E_total/dκ² = 2·H_bare_polarized` (unscaled per-spin
`dP = C_a·C_iᵀ + C_i·C_aᵀ`, no closed-shell 2× factor), `g_true = 2·g_bare`.
A **different constant from RKS's 4×**, measured not assumed to carry over.
Same conclusion: unscaled `g = F_mo` against unscaled `h_op`.

Design rule:

- Verify the composed `h_op` against a finite difference of the **total**
  energy `E(κ)`, never `compute_analytic_xc_hessian_vector_product` alone
  against `E_xc(κ)` — the latter is missing the density-curvature term
  (`docs/DFT_ANALYTIC_FXC_HESSIAN.md`), which was D2.1's original mistake
  (the one-term identity fails on water/GGA with the wrong sign).

### 3. UKS packing matches build_uhf_cphf_matrix — not a third convention

Two blocks in one vector: `[0, nova)` alpha, `[nova, nova+novb)` beta,
virtual-major within each. `δJ` is one call on the **total** trial density
`δP^α + δP^β` (matching the KS loop's own Coulomb build), then packed
separately into the α and β `(a,i)` blocks. Cayley rotation and
semicanonicalization run per spin channel — one shared step vector, two
`κ` matrices.

Design rule:

- Reuse the existing virtual-major CPHF convention
  (`pack_hessian_vector_product_cphf_order`, `build_uhf_cphf_matrix`) — the
  codebase already has two packings (that one and
  `ResponseExcitationSpace::flat_index`'s occupied-major); do not add a
  third.

### 4. Level shift is a DFT-wide no-op; semicanonicalization is unnecessary but kept

`grep -c level_shift src/dft/driver.cpp` returns 0 —
`calculator._scf._level_shift` is never read by the KS loop, RKS or UKS.
Verified: `level_shift 0.5` set vs absent gives byte-identical energy and
iteration count under plain DIIS. No guard needed (a `level_shift <= 0`
check would be dead code), unlike UHF where the field is live.
Semicanonicalization disabled over a 200-cycle pure-SOSCF window: RKS
water/6-31G/PBE 126 vs 121 iterations, UKS triplet water/6-31G 27 vs 30 —
identical energy either way.

Design rule:

- Do not port RHF/UHF's level-shift guard to DFT — check the actual KS loop
  for the knob first. Keep semicanonicalization: pure gauge freedom, cheap,
  no reason to drop it even where it measures as unnecessary.

### 5. A scope-cut gate needs a message, or the user's request is silently dropped

`soscf_enabled` excludes hybrids (need the unbuilt `K`-response), PCM, and
SAO. A user setting `scf_soscf_start` by habit on a B3LYP run got no
warning and silently ran plain DIIS. A one-time `[WRN] DFT SOSCF :` line is
emitted before the loop when SOSCF is requested but disabled, naming the
reason and confirming the calculation is still valid, just unaccelerated.
Not a hard error — that would newly fail every ordinary hybrid DFT run with
`scf_soscf_start` set.

Design rule:

- A gate that disables a requested feature emits a warning naming the
  specific reason. (RHF/UHF's own analogous guards do not yet — a small
  follow-on across all three SOSCF paths.)

### 6. The gradient-shrinkage shape is reported honestly, not assumed superlinear

On RKS water/6-31G/PBE the orbital gradient shrinks at a roughly constant
ratio (`1.58e-1 → 4.48e-2 → 1.29e-2`, ≈0.28) — linear, not the accelerating
ratio RHF/H2 showed. On H2/6-31G/PBE the same code is genuinely superlinear
(`4.60e-4 → 5.10e-5 → 5.64e-6`, ≈0.11). Investigated: the composed Hessian's
off-diagonal/diagonal coupling and condition number are comparable to RHF's
own (`‖H_offdiag‖/‖H_diag‖` 0.031 DFT vs 0.027 RHF; `cond(H)` 67 vs 57),
ruling out "the DFT Hessian is more diagonal-dominant". The water case's
linear rate is an open, recorded finding — not an algebra defect (the
callback was verified to ratio 1.000000 against `E(κ)`'s true second
derivative on this same system).

Design rule:

- DFT SOSCF's pass criterion is energy agreement with fully-converged DIIS
  first, gradient shape second — report the measured shape, do not assume
  superlinear because one system showed it.

## What was fixed / built

1. **`orbital_energy_difference_diagonal`** — the RKS/UKS analogue of the
   one line `A(ai,ai) += eps(a) - eps(i)` inside `build_rhf_cphf_matrix`,
   standalone because DFT has no such matrix to reuse. Verified against an
   independent hand-written loop on non-square fixtures.

2. **The RKS `h_op` composition and SOSCF branch** — `diag_term`,
   `_compute_2e_j_direct` on `δP` for `J`, the analytic XC piece, packed
   virtual-major; `solve_augmented_hessian`; `kSoscfMaxRot = 0.20`;
   `apply_orbital_rotation`; occ-occ/virt-virt semicanonicalization. The
   composed callback was verified against FD of the total energy at three
   directions before the branch was wired.

3. **The UKS SOSCF branch** — the polarized analogue, mirroring the RKS
   branch the way `docs/SOSCF_UHF.md` mirrors `docs/SOSCF.md`. Per-spin
   persisted state, `[0,nova)+[nova,novb)` packing, `δJ` from the total
   trial density.

4. **The one-time hybrid/PCM/SAO warning** — emitted before the loop in
   both RKS and UKS branches when a trigger keyword is set but SOSCF is
   disabled.

5. **Switch trigger — zero new code**, in both RKS and UKS. The gate was a
   structural copy of RHF's, which already carried the DIIS-error-criterion
   branch (third time this held: U4, D2.4, D3.4).

## Validation strategy that should remain in place

- **RKS:** same-energy to all 10 digits vs pure DIIS on water/6-31G/PBE
  (`-76.2895527467`); SOSCF-off default byte-identical to the pre-SOSCF
  tree. H2/6-31G shows SOSCF reaching `-1.1619037723` where plain DIIS
  stalls at `-1.1619034100` (cross-checked against PySCF's own DIIS run) —
  SOSCF routes around a real pre-existing plain-DIIS weakness, which is why
  "not H2 alone" is asked for.
- **UKS:** triplet water/STO-3G/PBE — SOSCF reaches `-74.8423080131` in 10
  iterations; plain DIIS at `tol 1e-9` stops early at `-74.8423073568`, and
  at `tol 1e-11` reaches exactly `-74.8423080131` in 102 iterations
  (bit-identical). Superlinear gradient shrinkage
  (`7.02e-3 → 2.96e-4 → 2.07e-5`).
- Full smoke (35/35) and DFT ctest (12/12) suites, SOSCF off by default.

## What was measured but is not kept

Whole-molecule probes (`PLANCK_D2_0_CHECK`, `PLANCK_D2_1_CHECK`,
`PLANCK_D2_2_2_CHECK`, `PLANCK_D2_3_NO_SEMICANON`, `PLANCK_D2_5_CHECK`,
`PLANCK_D3_0_CHECK`, `PLANCK_D3_1_CHECK`, `PLANCK_D3_3_NO_SEMICANON`,
`PLANCK_D3_5_CHECK`) verified each composition step against the true energy
or the FD oracle, once, then were reverted — same discipline as
`docs/DFT_ANALYTIC_FXC_HESSIAN.md`'s own probes. The permanent gates are the
point-level ctests for the `fxc` piece and the SOSCF-vs-DIIS
energy-agreement runs for the assembled `h_op`.

## Remaining architecture concern

### Wall-clock: RKS ≈ break-even, UKS a clear win

Measured one call to the analytic `h_op` (one `(a,i)` unit vector — the
Krylov-loop shape) against one call to the FD-kernel oracle building the
full dense Hessian, alongside real `ah_iters`. The FD oracle's
per-Newton-step cost is `≈ fd_kernel_full` regardless of `ah_iters` (each
subsequent Krylov iteration is a microsecond-scale dense matvec); the
analytic path's is `ah_iters × analytic_call`.

**RKS — not reliably faster at the two sizes measured:**

| system | `nov` | analytic (1 call) | FD-kernel (full) | crossover `ah_iters` | real `ah_iters` |
|---|---|---|---|---|---|
| water/6-31G/PBE | 40 | 0.131 s | 0.513 s | ≈3.9 | 6, 7, 4 |
| water/cc-pVDZ/PBE | 100 | 0.141 s | 2.131 s | ≈15.1 | 4, 23, 10 |

Real `ah_iters` straddle or exceed the crossover.

**UKS — reliably 3–5× faster at every size, the opposite of RKS:**

| system | `nova`/`novb` | analytic (1 call) | FD-kernel (full) | crossover `ah_iters` | real `ah_iters` |
|---|---|---|---|---|---|
| water/STO-3G/PBE triplet | 6 / 12 | 0.017 s | 0.155 s | ≈9.2 | 3, 3, 3 |
| water/6-31G/PBE triplet | 42 / 36 | 0.018 s | 0.50 s | ≈28 | 5, 6, 7 |
| water/cc-pVDZ/PBE triplet | 114 / 84 | 0.064 s | 2.06 s | ≈32 | 8, 10, 10 |

Real `ah_iters` (3–10) stayed in single digits across a >10× `nov` range
(conditioning, not dimension), and the FD-kernel oracle must
finite-difference both spin channels' directions, so its cost roughly
doubles at comparable `nov` — pushing the crossover **higher**, the reverse
of the worry that the doubled per-call cost would make UKS worse.

Neither claim is contradicted. The `O(1)` vs `O(n_occ·n_virt)` asymptotic
argument is real; whether it is a wall-clock win depends on `ah_iters` vs
the crossover at the actual system size — comfortably favorable for UKS at
modest sizes, borderline for RKS there. The analytic path is correct and
correctly-scaling in both, and is the only path that scales to a large
active space where the FD oracle's one-time build itself becomes
prohibitive.

### Not done

- **Hybrid / range-separated functionals** — need the `K` response on top
  of `fxc`. Rejected with a warning.
- **PCM, SAO/symmetry** — same.
- **`scf_soscf_diis_tol` DFT-specific default** — not required. There is no
  hardcoded default to re-tune (`_scf_soscf_diis_tol = 0.0`, SOSCF is
  opt-in), and the criterion mechanism is verified working (D2.4/D3.4). A
  DFT-tuned default is a sweep for when the large-`nb` cluster ladder is
  available.
- **The large-`nb` motivation** is unreproduced for RKS (cluster access);
  measured and positive for UKS at modest sizes.
