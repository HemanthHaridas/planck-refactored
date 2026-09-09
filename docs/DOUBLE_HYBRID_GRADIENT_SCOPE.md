# Double-Hybrid Analytic Gradient — Scope

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This is in-flight scope; fold it into an answer
doc when the work lands.

**Question this scopes:** what does an analytic gradient for a double-hybrid
KS functional (B2PLYP and the libxc `XC_HYB_PT2` family) actually require in
this codebase, given that the SCF-level double-hybrid energy already works?
Gradients unlock geomopt and frequencies for free (both drive
`compute_analytic_ks_gradient` in a loop / by finite difference of the
analytic gradient).

## Short answer

A double-hybrid gradient is **not** "the KS hybrid gradient plus the RMP2
gradient". The PT2 correction is evaluated on the **KS** orbitals, so its
orbital-relaxation (Z-vector) term must be solved against the **KS orbital
Hessian** — `(ε_a − ε_i)` diagonal + Coulomb response + scaled exact-exchange
response + analytic `fxc` XC-kernel response — not the HF CPHF matrix that
`compute_rmp2_gradient` uses today. That KS orbital Hessian already exists in
the tree as `compute_analytic_xc_hessian_vector_product{,_polarized}` +
`h_op`, built for RKS/UKS SOSCF (`docs/SOSCF_DFT.md`,
`docs/DFT_ANALYTIC_FXC_HESSIAN.md`). So the new work is:

1. an MP2-like Lagrangian (`P^(2)`, `W^(2)`, `L_ai`) built from **KS**
   orbitals and the **scaled** PT2 coefficient — structurally the existing
   `build_rmp2_gradient_intermediates` MO intermediates, minus the HF
   Z-vector solve;
2. a Z-vector solve using the **existing KS `h_op`** as the linear operator
   (CG, matrix-free — the same solve SOSCF already runs);
3. adding the relaxed-density contributions into the HF-like +
   XC-grid-derivative gradient that `compute_analytic_ks_gradient` already
   assembles;
4. lifting the `validate_workflow_support` gate for double hybrids on
   `Gradient` / `GeomOpt` / `Frequency` / `GeomOptFrequency` (RKS + UKS).

RKS first (B2PLYP is the validation target). UKS is the same three pieces
with the polarized `fxc` product and per-spin intermediates — do it in the
same PR only if the RKS Lagrangian factors cleanly per spin; otherwise a
fast follow.

## What already exists (do not rebuild)

| Piece | Where | Reused how |
|---|---|---|
| Double-hybrid SCF energy + scaled PT2 | `apply_post_ks_double_hybrid_correction`, `src/dft/driver.cpp:2892` | The converged KS state + `perturbative_correlation_coefficient` are the gradient's input |
| KS orbital Hessian `h_op` (diag + J + K + analytic `fxc`) | `src/dft/analytic_hessian.cpp`, `compute_analytic_xc_hessian_vector_product{,_polarized}`; composed in the SOSCF `h_op` in `src/dft/driver.cpp` | **This is the Z-vector operator.** Factor `h_op` out of the SOSCF path so the gradient can call it |
| HF-like KS gradient (core+Pulay, J, K full/LR, Vnn) | `compute_rks_gradient` / `compute_uks_gradient`, `src/gradient/gradient.cpp` | Unchanged; relaxed density adds into the same contraction |
| XC nuclear (grid-derivative) gradient | `DFT::Gradient::compute_xc_nuclear_gradient_{rks,uks}` | Unchanged for the SCF density; **new**: same routine re-run on the relaxed density difference for the `fxc·P^(2)` grid term (see Risk 2) |
| MP2 MO intermediates (`T2`, `P^(2)_ij`, `P^(2)_ab`, `W`, `L_ai`) | `build_rmp2_gradient_intermediates`, `src/post_hf/mp2_gradient.cpp` | Structure reused; **inputs change** from HF `C`, `ε` to KS `C`, `ε`, and the whole thing is scaled by `c_PT2` |
| Matrix-free CG Z-vector solve | the SOSCF augmented-Hessian / CIAH solve loop | Same solver, RHS = `−L_ai`, operator = KS `h_op` |
| Canonical-Fock guarantee (`f_ov = 0`, `f_oo`/`f_vv` diagonal) | KS orbitals diagonalize the converged KS matrix | The PT2 numerator uses `(ε_a+ε_b−ε_i−ε_j)` denominators exactly as MP2 does; no semicanonicalization needed. See `memory/cc_canonical_fock_only.md` for the analogous CC invariant |

## What is new

### N1 — PT2 Lagrangian on KS orbitals (`src/dft/dh_gradient.cpp`, new)

`L_ai = 2 Σ_bj (ai|bj)_KS T2_ibaj_stuff + Σ (P^(2) contracted with (ai|jk), (ai|bc))` —
i.e. the standard MP2 orbital-gradient RHS, but:
- integrals transformed with **KS** `C`;
- denominators from **KS** `ε`;
- **every term scaled by `c_PT2`** (`functionals.perturbative_correlation_coefficient`).

Cleanest path: parameterize `build_rmp2_gradient_intermediates` to take
`(C, eps)` explicitly and a scale factor, then call it with the KS reference
and **skip its internal `solve_rhf_cphf`** — the gradient driver owns the
Z-vector solve because the operator is different. Check
`src/post_hf/mp2_gradient.cpp:493` (`solve_rhf_cphf` call site) — that call is
what gets hoisted out and replaced.

### N2 — Z-vector against `h_op`

```
h_op(z) = −L_ai        # RHS is the scaled PT2 Lagrangian
```
`h_op` = the SOSCF KS orbital Hessian, matrix-free. Solve with CG. The
relaxed one-particle density is `P_relaxed = P^(2) + z` (occ-virt block from
`z`, occ-occ / virt-virt from `P^(2)`).

Prereq refactor: `h_op` is currently a lambda local to the SOSCF branch of
`run_ks_scf_scaffold`. Extract it to a free function
`build_ks_orbital_hessian_op(calculator, prepared, functionals, C, eps)`
returning `std::function<VectorXd(const VectorXd&)>` (or a small struct).
SOSCF and the gradient then share one definition — do **not** copy it
(`memory/no_spaghetti_fix_the_mechanism.md`).

### N3 — relaxed-density gradient contributions

Given `P_relaxed` and the energy-weighted `W^(2)`:
- `P_relaxed` contracts into the **same** HF-like derivative-integral engine
  the SCF density uses (`compute_rks_gradient` already contracts an AO
  density against `∂h/∂R`, `∂(ij|kl)/∂R`, scaled K) — add `P_relaxed` as a
  second density passed through the same path, or contract separately and
  sum;
- `W^(2)` contracts against `∂S/∂R` (the Pulay term), same as the SCF
  energy-weighted density;
- the XC piece: `P_relaxed` also couples through `fxc` to a grid term
  `∫ fxc(r) ρ_relaxed(r) ∇(AO·AO)` — this is the part with no existing
  caller. See Risk 2.

## Risks / unknowns

1. **The XC term in `E_PT2^x` -- RESOLVED by the paper (Eq. 33), it is
   `fxc` (v2), not `kxc` (v3).** The early guess (Risk 1, original) was
   that the relaxed-density XC gradient needs the *third* functional
   derivative. The paper's Eq. 33 shows it is the geometry (basis-function)
   derivative of the `fxc`-kernel matrix elements
   `<mu| V_xc^[2][rho_P] |nu>` -- `v2rho2` / `v2rhosigma` / `v2sigma2`
   (already in the wrapper, used by `dft_gga_hessian_selfcheck`) times
   AO grad/Hessian factors. NOT `Tr[D . V_xc^x]` (Risk 2's guess -- the
   paper explicitly says that naive form is wrong). Implemented as
   `compute_xc_kernel_nuclear_gradient` in N3.5.7; see
   `docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md`. Measured residual at
   1.87e-4 after N3.5.4, so this term is ~2e-4 -- not below a 1e-6 gate,
   it has to be implemented.

2. **`compute_xc_nuclear_gradient_rks` on the relaxed density is the WRONG
   operator.** Tested (N3.5.2): `g(dm1_corr_relaxed_ao) - g(0)` at every
   sign/scale made FD 8-22x worse. That routine is `dE_xc/dR` (contracts
   `vrho`/`vsigma`), missing the `fxc` response structure Eq. 33 requires.

3. **UKS Lagrangian factorization.** Same-spin / opposite-spin PT2 blocks
   give `L_ai^α`, `L_ai^β` coupled through the polarized `fxc`. The
   `_polarized` Hessian product exists; the open question is whether
   `build_ump2_gradient_intermediates` (`src/post_hf/mp2_gradient.cpp:944`,
   which calls `solve_uhf_cphf`) factors so its MO-intermediate half is
   reusable the way the RMP2 one is. If yes, UKS is the same PR. If the
   spin coupling is tangled into its Z-vector solve, split it out first.

4. **Frozen core.** If double-hybrid runs ever use a frozen core, the PT2
   Lagrangian and Z-vector must respect it. Check `calc._mp2` for a
   frozen-core flag and whether the SCF-level double-hybrid path already
   honors it — if the energy path is all-electron only, keep the gradient
   all-electron only and note it.

## Validation

- **FD self-consistency**: analytic gradient vs finite difference of the
  double-hybrid *energy*, water/STO-3G B2PLYP, target <3e-7 Ha/Bohr (same
  bar the HSE06 gradient met, `src/dft/driver.cpp:2850`).
- **PySCF cross-check**: PySCF has analytic B2PLYP gradients
  (`dft.RKS` + `mp2` gradient glue via `nuc_grad_method`). Water/STO-3G and
  water/6-31G*, target <1e-6 Ha/Bohr per component.
- **GeomOpt**: B2PLYP/6-31G* water, final geometry within ~7e-4 Å, energy
  within ~2e-5 Eh of PySCF `geometric` (matches the HSE06 geomopt bar).
- **Freq**: B2PLYP/STO-3G water, max |Δ| vs PySCF analytic-Hessian-FD within
  ~5 cm⁻¹ (HSE06 freq landed at 2.2 / 6.8 cm⁻¹).
- **UKS** (if in scope): H2O⁺ doublet, same three checks.

Regression cases to add to `tests/regression_cases.json`:
`water_b2plyp_gradient_sto3g`, `water_b2plyp_geomopt_631gd`,
`water_b2plyp_freq_sto3g`, and the FD self-consistency case
`water_b2plyp_gradient_fd`.

## Explicitly out of scope

- Double-hybrid **TDDFT** / `LinearResponse` — separate, harder (PT2
  response of excitation energies).
- Double-hybrid **ImaginaryFollow** — trivial once `Frequency` works (it
  reuses the semi-numerical Hessian eigenvector); fold in with a one-line
  gate change and a smoke test, don't scope separately.
- Range-separated **double** hybrids (ωB2PLYP): the range-separation LR-K
  response in `h_op` plus the PT2 — additive with this work but needs its
  own validation pass. Land plain B2PLYP first.
- Spherical-basis double-hybrid gradients — blocked on the same
  MP2-response spherical-lift audit that blocks plain RMP2/UMP2 spherical
  gradients (`vault/Status/Open Work.md`). Not this work.

## N1.3 progress notes

- **N1.3.1** (landed): `build_rmp2_lagrangian` added to `mp2_gradient.{h,cpp}`,
  no caller. The `imat_ao` contraction, fused into the derivative loop in
  `build_rmp2_gradient_intermediates`, is unfused into a standalone loop in
  the helper (RI + conventional branches).
- **N1.3.2** (landed): `build_rmp2_gradient_intermediates` routed through the
  helper. Conventional + RI RMP2 gradient on water/STO-3G byte-identical to
  the pre-refactor baseline; full extended suite 121/121.
- **N1.3.3** (skipped): a standalone check of `build_rmp2_lagrangian` on real
  orbitals adds nothing over N1.3.2 — it is the same function on the same
  `RMP2Result` with no hidden state (`ensure_eri` only touches an idempotent
  cache), and the geomopt cases already call it repeatedly. Its intent — "the
  helper's `Xvo` is correct when called standalone" — is subsumed by N1.4,
  which compares the Z-vector-relaxed `P_ao` end to end.
- **N1.3.4** (landed): `dh_gradient.cpp`'s `build_pt2_mo_intermediates` is a
  3-line delegation to `build_rmp2_lagrangian`; returns `RMP2Lagrangian`
  directly (`PT2MOIntermediates` deleted as a duplicate). The N1.0-N1.2
  inline prototype is gone.
- **N1.4** (skipped): the planned end-to-end "`P_ao` vs
  `build_rmp2_gradient_intermediates`" check has nothing left to verify for
  the RHF path -- `build_pt2_mo_intermediates` *is* `build_rmp2_lagrangian`
  (a pure function), and its RHF correctness is nailed by N1.3.2's
  byte-identical baseline plus extended suite 121/121. The one untested
  thing -- `build_rmp2_lagrangian` on KS (not HF) orbitals -- has no
  consumer until N2, so N1.4's intent folds into N2's FD-check of the full
  double-hybrid gradient.
- **N1.5** (landed): `build_pt2_mo_intermediates` takes `double pt2_scale`
  and multiplies every gradient-linear quantity (`doo`, `dvv`,
  `dm1_corr_mo/ao`, `veff_corr_ao`, `imat_ao/mo`, `Xvo`) by it. Scaling at
  the source -- the double-hybrid entry point -- so the returned struct is
  already correct for the `E_KS + c_PT2 E_PT2` functional and the N2 caller
  cannot forget a `1/c_PT2` factor. `build_rmp2_lagrangian` stays unscaled
  (RMP2 = full PT2). **`dm2buf_full` is deliberately NOT scaled**: it is the
  raw T2->AO buffer the RI 2e-gradient term consumes separately, and that
  term is scaled where applied -- scaling it here would double-count. The
  `pt2_scale = 0` -> gradient-reduces-to-KS-hybrid anchor needs a real
  system, so it is verified in N2's FD-check, not the synthetic unit test
  (which only confirms the scale arg is threaded and leaves the input guards
  intact).

## N2 progress notes

- **N2.1** (landed): the RKS SOSCF `h_op` lambda lifted into
  `src/dft/ks_orbital_hessian.{h,cpp}` as
  `build_ks_orbital_hessian_op(KsOrbitalHessianInputs)` -> a matrix-free
  `std::function`. Takes plain matrices/vectors + `shell_pairs` /
  `molecular_grid` / `ao_grid` pointers (not `PreparedSystem`), so it stays
  testable without linking SCF -- same reasoning `analytic_hessian.h` and
  `response_packing.h` use. `h_op(x) = diag_term(x) + s*(J + xc + K)`,
  `s = 2` for RKS (`kernel_scale`), CPHF (a,i) ordering. RKS-only for now.
  The RKS SOSCF branch in `run_ks_scf_scaffold` is now a pure caller.
  Gate: all 5 SOSCF regression cases pass with `dft_total_energy` matching
  to `atol 1e-9` and the superlinear per-iter `|g|` sequence intact (a
  scale error would show as linear convergence -- the case `_note`s gate
  exactly that).

- **N2.2** (skipped): a standalone FD-vs-analytic unit check of the composed
  operator adds little -- N2.1b already proves `build_ks_orbital_hessian_op`
  is byte-identical to the SOSCF operator (a *correct* KS Hessian: SOSCF
  converges superlinearly to the DIIS energy, and the SOSCF case `_note`s
  gate exactly the scale error that would degrade that to linear), and the
  XC term is independently verified by `planck-dft-analytic-hessian-production`.
  A real-basis binary that also links `build_rhf_cphf_matrix` for a
  scaling cross-check is heavy, and the one thing it would catch -- `Xvo`'s
  `build_rhf_cphf_matrix` convention vs `build_ks_orbital_hessian_op`'s
  convention when solving `h_op(z) = -Xvo` -- shows up unambiguously in
  N2.4's B2PLYP gradient FD-check (a 2x scale error there misses FD by
  ~1e-3..1e-2). So the scaling verification folds into N2.4.

- **N2.3** (landed): `solve_pt2_relaxed_density` in
  `src/dft/dh_relaxed_density.{h,cpp}` (split from `dh_gradient.{h,cpp}` so
  the MP2-side `build_pt2_mo_intermediates` and its unit test stay free of
  the libxc dependency). Builds `h_op` via `build_ks_orbital_hessian_op`,
  packs `-lag.Xvo` in CPHF order, solves `h_op(z) = -Xvo` (dense assembly
  by applying `h_op` to unit vectors, then `colPivHouseholderQr` -- a
  `ponytail:` note flags the O(dim) XC-HVP cost and the CG upgrade path),
  and assembles `P_ao = C(2I + gamma^1_diag + z_ov)C^T` exactly as
  `mp2_gradient.cpp:558-565`. RKS only. No caller yet -- N3 wires it into
  `compute_analytic_ks_gradient`. The real-basis checks (solve converges,
  `P_ao` symmetric with `tr(P_ao S) = n_elec`, `pt2_scale = 0` -> bare KS
  density, HF-limit cross-check vs `build_rhf_cphf_matrix`) all need a full
  SCF + grid, so they ride N3.4's end-to-end FD check.

## N3 -- assemble the double-hybrid gradient

N2 built the operator and the relaxed density; N3 wires them into
`compute_analytic_ks_gradient` and is where the first real numeric check
(FD of the B2PLYP gradient) becomes possible.

- **N3.1** (landed): `build_rmp2_energy_weighted_density` in
  `mp2_gradient.{h,cpp}` -- extracts `corr_relaxed_mo` / `P_mo` / `P_ao` /
  `zeta_ao` / symmetrized `imat_ao` / `dm1_corr_relaxed_ao` / `vhf_s1occ` /
  `W_ao` (was `build_rmp2_gradient_intermediates` lines 558-590 + 677) into
  a `RMP2RelaxedDensity` struct. Inputs: the shared `RMP2Lagrangian`, the
  solved Z-vector `z`, `mo_coeff`/`mo_energy`. `build_rmp2_gradient_intermediates`
  now routes through it (its `corr_relaxed_mo` debug print became `P_mo`; the
  final `W_ao` recompute became `rel.W_ao`). Gate: conventional + RI RMP2
  gradient byte-identical, 7 gradient/geomopt cases + core suite 71/71 pass.
  `solve_pt2_relaxed_density` now returns `PT2RelaxedDensity` (`P_ao`,
  `W_ao`, `dm1_corr_relaxed_ao`, `zeta_ao`, `imat_ao`) via the same shared
  helper -- only the Z-vector operator above it differs.
- **N3.2** (landed): in `compute_analytic_ks_gradient`, for a double-hybrid
  functional and RKS, gated behind `PLANCK_DFT_DH_GRADIENT`: run
  `rmp2_kernel` on the converged KS orbitals (they live in
  `calculator._info._scf` post-SCF), `build_pt2_mo_intermediates` with
  `c_PT2 = functionals.perturbative_correlation_coefficient`, then
  `solve_pt2_relaxed_density`. A `dft_allow_double_hybrid_gradient()` env
  hook opens the `validate_workflow_support` gate for the same four
  gradient-driven workflows (dev flag; step 6 lifts the gate and removes
  it). N3.2 logs diagnostics only (`c_PT2`, `tr(P_ao S)`, `max|P - P^T|`);
  N3.3 contracts. Verified on H2/STO-3G B2PLYP: chain runs end to end,
  `tr(P S) = 2.000000` (particle number), `max|P - P^T| = 0.000e+00`. Gate
  still rejects without the flag; DH energy / SOSCF / RMP2-grad / HSE06-grad
  regressions unaffected.
- **N3.3**: contract the PT2 correction into the gradient. The KS reference
  part `dE_KS/dR|_{P_SCF}` is already `compute_rks_gradient`; what N3.3 adds
  is `c_PT2 * dE_MP2_correction/dR`, and that is exactly what
  `build_rmp2_gradient_intermediates.electronic_gradient` computes MINUS the
  pure reference gradient it also folds in (it contracts `dm1_total_ao =
  hf_dm1 + dm1_corr_relaxed_ao` and cross-terms `hf_dm1(p,q)*dm1p(r,s)`).
  Rather than extract the ~160-line contraction block, use the identity it
  already satisfies. `build_rmp2_gradient_intermediates.electronic_gradient`,
  expanded in `dm1p = hf_dm1 + 2*dm1_corr_relaxed_ao` and `dm1_total_ao =
  hf_dm1 + dm1_corr_relaxed_ao`, is
  `[pure-reference 2J-K + core + Pulay + Vnn](hf_dm1)` +
  `c_PT2 * dE_MP2/dR`. The `vhf1_rs/rq` reference sub-terms carry the full
  `cx = 1` HF exchange (`+dI*hf_dm1(p,q)*hf_dm1(r,s)` Coulomb,
  `-0.5*dI*hf_dm1(p,s)*hf_dm1(r,q)` exchange), so the pure-reference part is
  exactly `compute_rhf_gradient` on those orbitals. Therefore:

      c_PT2 * dE_MP2/dR
        = build_rmp2_gradient_intermediates(KS, scaled_lag, KS_z).electronic_gradient
          - compute_rhf_gradient(KS orbitals).electronic
          - compute_nuclear_repulsion_gradient   (folded into electronic_gradient
                                                   by compute_rmp2_gradient, not here)

  and the KS reference `dE_KS/dR` is `compute_rks_gradient` (already in
  `calculator._gradient` from the main path). Sub-steps:
  - **N3.3a**: parameterize `build_rmp2_gradient_intermediates` to accept an
    optional pre-solved `(RMP2Lagrangian, z)` pair. When given (double-hybrid
    path) it skips its own `build_rmp2_lagrangian` + `solve_rhf_cphf` and
    uses them; when not (RMP2 path) unchanged. Gate: RMP2 gradient / geomopt
    byte-identical (same as N1.3.2 / N3.1).
  - **N3.3b/c** (landed together): `compute_analytic_ks_gradient` computes
    the reference-only electronic gradient via
    `build_rmp2_gradient_intermediates(KS, {zero_lag, zero_z})` (avoids the
    Vnn/static-helper problem and matches the exact grouping folded into
    `electronic_gradient`), then
    `corr = build_rmp2_gradient_intermediates(KS, {scaled_lag, KS_z}).electronic_gradient
    - ref_electronic`, and `calculator._gradient += corr`.
    **N3.3b c_PT2-linearity self-check** (`PLANCK_DFT_DH_GRADIENT_SELFCHECK`):
    re-run the chain at `2*c_PT2`, require `corr(2c) == 2*corr(c)`.
    **It failed first (rel 1.16)** -- the N1.5 decision to leave
    `dm2buf_full` unscaled was wrong: it feeds the conventional pair-density
    2e derivative term (`dm2v = 2*dm2buf_full`) AND the RI 2e-grad term
    directly, at full magnitude, so an unscaled buffer makes the whole
    correction non-linear in c_PT2. Fixed by scaling `dm2buf_full` in
    `build_pt2_mo_intermediates` too; linearity is now `rel 1.6e-13`. RMP2
    gradient stays byte-identical (that path never passes `pt2_scale != 1`).
    **No XC grid response term yet** -- N3.4's FD showed it is needed
    (~2e-4), and the paper (Eq. 33) identifies it as the `fxc` grid term,
    now N3.5.7. The HF-limit cross-check against `build_rhf_cphf_matrix`
    (old N2.2/N2.3c) rode N3.4's FD result.
- **N3.4** (landed): FD-checked water/STO-3G B2PLYP (`calculation gradient`,
  grid ultrafine, C1) vs central-difference of the B2PLYP energy, step
  1e-3 Bohr:

      max|analytic - FD| = 3.05e-4 Ha/Bohr   (step-independent 2e-3..2e-4)

  Not FD noise -- a real missing analytic term. Isolated by elimination:

  | path | FD error | verdict |
  |---|---|---|
  | B3LYP water gradient (pure global hybrid, no PT2) | 1.5e-7 | KS-hybrid gradient exact |
  | `water_rmp2_gradient_fd` (HF-like MP2 contraction, c_PT2=1) | ~1e-7 | MP2 contraction exact |
  | PT2 Z-vector via the KS orbital Hessian | 3.0e-4 | this is the residual |
  | PT2 Z-vector via HF CPHF (`PLANCK_DFT_DH_ZVECTOR_HFCPHF`) | 2.1e-3 | 7x worse -> KS Hessian is the RIGHT operator, scaling/packing OK |

  **The deferred N2.2/N2.3c operator-scaling check lands here, passing** --
  the KS orbital Hessian moves the gradient 7x closer to FD than HF CPHF,
  and a factor-of-2 scale error would not do that. The `pt2_scale = 0`
  anchor is covered by N3.3b's linearity check (rel 1.6e-13) plus the
  KS-only path being unaffected.

  The residual 3.0e-4 is exactly the scope's Risk 1: the PT2 relaxed
  density perturbs the XC potential, and the moving-grid XC gradient of
  that perturbation needs `kxc` (the third functional derivative,
  libxc `v3`), which the analytic `fxc` Hessian work stopped short of.

## N3.5 -- the tDH-gradient equations, and the one term still open

**The Neese/Schwabe/Grimme 2007 paper (JCP 126, 124115, DOI 10.1063/1.2712433)
is the reference.** Its closed-shell tDH gradient (Sec. III.A) is mapped
term-by-term to the code in
`docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md`, which also carries the
PySCF `grad/mp2.py` architecture map, the Planck<->PySCF sign-convention
analysis (the `dI` flip cancels -- do NOT transcribe PySCF's `de -=`
lines), and the full N3.5.4-N3.8 step list.

Findings:

- **N3.5.4 -- LANDED.** The Lagrangian RHS (Eq. 40, `L_ai = R(D')_ai + ...`)
  applies the response operator `R` -- which **includes the `f_xc` kernel
  `R^XC`** (Eq. 41) -- to the unrelaxed difference density. That is
  `veff_corr_ao` in `build_rmp2_lagrangian`, and it was hardwired to HF
  `J - 1/2 K`. A `KsVeffFn` callback (`J - 1/2 c_x K + f_xc-response`, the
  same pieces `build_ks_orbital_hessian_op` uses) now overrides it for the
  DH path; RMP2 stays byte-identical. **FD 3.05e-4 -> 1.87e-4.**

- **N3.5.5 -- reverted.** KS response for `vhf_s1occ` made FD worse
  (1.87e-4 -> 2.86e-4); small effect, unresolved, `vhf_s1occ` stays HF.

- **The `vhf1` HF `2J - K` derivative kernel is CORRECT.** Eq. 46's
  separable 2-particle density uses full HF exchange weight (`1/4`, `1/2`
  coefficients), not `a_x` -- the `a_x` scaling lives entirely in `E_SCF`
  and `R^XC`. So the earlier "K-weight fix" attempts
  (`PLANCK_DFT_DH_ZVECTOR_KERNEL_FIX`, `PLANCK_DFT_DH_XC_RESPONSE`) were
  wrong by the paper, not just mis-factored -- delete those blocks.

- **The remaining 1.87e-4 is the Eq. 33 `f_xc` grid term** (N3.5.7). The
  paper is explicit that the PT2 gradient's XC part is NOT `Tr[D . V_xc^x]`
  -- it requires the **second** functional derivative, because the SCF
  operator already carries `dV_xc/drho` and `E_PT2` is not stationary. It
  is the nuclear (basis-function) derivative of the `f_xc`-kernel matrix
  elements `<mu| V_xc^[2][rho_P] |nu>` contracted with the relaxed density
  `D` and the SCF density gradient -- one derivative order above
  `compute_xc_nuclear_gradient_rks` (which is `dE_xc/dR`, uses `vrho`, no
  `f_xc` -- that is why all three attempts with it overshot ~8-22x). Needs
  a new `compute_xc_kernel_nuclear_gradient` routine; cost "negligible" per
  the paper.

**Shipped state:** the RKS double-hybrid gradient runs end to end behind
`PLANCK_DFT_DH_GRADIENT`, **1.87e-4 Ha/Bohr off FD** on water/STO-3G
B2PLYP (down from 3.05e-4 via N3.5.4). The `validate_workflow_support` gate
still rejects DH gradient workflows without the flag. Solid: N1
(`build_rmp2_lagrangian`), N2 (`build_ks_orbital_hessian_op`,
`solve_pt2_relaxed_density`), N3.1-N3.3 (the chain + `c_PT2` scaling + the
HF-like contraction), N3.5.4 -- all gated by byte-identical RMP2 gradient
regressions + the SOSCF suite. N3.5.7 (the `f_xc` grid term) is the one
piece between here and lifting the gate (N3.6).

## Step order (as executed)

1-4 (N1, N2, N3.1-N3.3): landed -- shared `build_rmp2_lagrangian`,
   `build_ks_orbital_hessian_op` lifted from SOSCF, `solve_pt2_relaxed_density`,
   the chain + `c_PT2` scaling + HF-like contraction. FD 3.0e-4 off.
5. N3.5.4 (landed): KS response in the Lagrangian RHS via `KsVeffFn`. FD ->
   1.87e-4.
6. **N3.5.7 (open): the Eq. 33 `f_xc` grid term** -- new
   `compute_xc_kernel_nuclear_gradient` routine. This is the one piece left;
   see `docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md`.
7. N3.6: lift the `validate_workflow_support` gate, remove the dev flags,
   add regression cases.
8. N3.7 (UKS), N3.8 (ImaginaryFollow).
