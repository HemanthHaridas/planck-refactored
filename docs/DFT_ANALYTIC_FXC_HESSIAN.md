# DFT Analytic XC Hessian-Vector Product (`fxc`)

Canonical status now lives in:

- `vault/Status/Completion.md`
- `vault/Status/Open Work.md`

This file answers a narrower architecture question:

**How does Planck compute the analytic XC second-derivative contribution
`δV_xc` to the Kohn-Sham orbital Hessian — the `fxc`-kernel term a
CPHF/SOSCF `h_op` needs — without finite-differencing the grid, and how was
each term verified?**

## Short answer

`compute_analytic_xc_hessian_vector_product` (RKS) and
`compute_analytic_xc_hessian_vector_product_polarized` (UKS) in
`src/dft/analytic_hessian.{h,cpp}` contract libxc's second-derivative
kernel (`v2rho2`, `v2rhosigma`, `v2sigma2`) against a trial response
density `δρ`/`δ∇ρ` on the grid, once, and return `δV_xc` in the AO basis.
LDA and GGA, unpolarized and polarized.

The alternative — reusing the TDDFT finite-difference kernel builders
(`build_{closed_shell,unrestricted}_xc_kernel_blocks`) as a SOSCF Hessian —
is `O(n_occ·n_virt)` full-grid XC evaluations per Newton step. The analytic
path is `O(1)` grid passes regardless of how many occ-virt directions the
Krylov solver probes. It required wiring libxc's `xc_lda_fxc`/`xc_gga_fxc`
through the wrapper, which previously only ever called the first-derivative
`exc_vxc` family.

Every term was derived symbolically, verified point-by-point against the
FD-kernel oracle (which shares no formula with the analytic path), and once
end-to-end against PySCF's own `nr_rks_fxc`.

## Where the logic lives

- `src/dft/base/wrapper.h` — `evaluate_lda_fxc` / `evaluate_gga_fxc` on
  `DFT::XC::Functional`, plus `v2rho2_components()` /
  `v2rhosigma_components()` / `v2sigma2_components()` (the per-point
  component counts, which are NOT `spin_components()`/`sigma_components()`)
- `src/dft/analytic_hessian.cpp` — the T1..T5 contraction, LDA/GGA and
  unpolarized/polarized dispatch; the GGA branch is F3.3.3's `T1+T2+T3`
  decomposition ported from the now-deleted whole-molecule probe. Also
  `drop_correlation_if_combined` (invariant 3a) — zeroes the correlation
  `fxc` arrays under `is_combined_exchange_correlation()`
- `src/dft/response_packing.{h,cpp}` — `pack_hessian_vector_product_cphf_order`,
  the single translation from AO-basis `δV_xc` into the virtual-major
  `idx(a,i) = a·n_occ + i` layout `solve_augmented_hessian` consumes
- `src/dft/xc_grid.cpp` — `evaluate_density_on_grid`, reused unchanged
  because it is exactly linear in the density matrix (so `δρ = ρ[δP]`)
- `src/dft/ks_matrix.cpp` — `assemble_xc_matrix` / `accumulate_local_potential`,
  the first-derivative contraction that was symbolically differentiated to
  get the GGA T1/T2/T3 structure
- `src/dft/driver.cpp` — `build_{closed_shell,unrestricted}_xc_kernel_blocks`,
  the FD-kernel oracle (verification only, not the production Hessian)
- ctests: `dft_fxc_selfcheck`, `dft_gga_hessian_selfcheck`,
  `dft_gga_polarized_fxc_ordering`, `dft_gga_polarized_hessian_selfcheck`,
  `dft_analytic_hessian_production`, `dft_analytic_hessian_polarized_production`,
  `dft_hessian_vector_packing`, `dft_density_response_linearity`

## What invariants matter

### 1. libxc's per-point second-derivative component counts are not the first-derivative convention

Read from libxc's own `internal_counters_set_{lda,gga}`, not guessed:

| | unpolarized | polarized |
|---|---|---|
| `v2rho2` | 1 | 3 (independent `aa`/`ab`/`bb` pairs, not `nspin`) |
| `v2rhosigma` | 1 | 6 (2 rho-channels × 3 sigma-channels) |
| `v2sigma2` | 1 | 6 (independent sigma-sigma pairs) |

Getting a count wrong silently corrupts every downstream read. Mutation-
verified: an off-by-one polarized `v2rho2` count crashes on the size
assertion rather than passing quietly.

Design rule:

- Carry dedicated `*_components()` accessors for the second-derivative
  arrays, mirroring the existing `spin_components()`/`sigma_components()`;
  never reuse the first-derivative counts for `fxc` reads.

### 2. The FD-oracle comparison is structurally independent, not merely independently run

The oracle perturbs the density and re-evaluates the full XC potential from
scratch through the first-derivative `exc_vxc` path; the analytic path
contracts a second-derivative kernel through entirely different code
(`evaluate_{lda,gga}_fxc`). The two share no formula that could be wrong in
the same way on both sides — unlike an FD-of-itself self-consistency check,
which F2 showed is blind to a shared scale error.

Design rule:

- Verify against D1's FD-kernel oracle at every sub-step, not against a
  hand-derivation alone and not only once at the end. RHF SOSCF's own
  history is the precedent: a gradient/Hessian pairing individually
  plausible on each side was silently wrong by 2/4 until checked directly.

### 3. The GGA `δV_xc` has three structurally distinct terms and T3 is the one that gets dropped

Differentiating `V_xc = vrho·AA + 2·vsigma·(∇ρ·AG)` (with `σ = ∇ρ·∇ρ`, so
`δσ = 2∇ρ·δ∇ρ`) one more time gives:

```
δV_xc = [v2rho2·δρ + 2·v2rhosigma·(∇ρ·δ∇ρ)] · AA                 (T1)
      + 2·[v2rhosigma·δρ + 2·v2sigma2·(∇ρ·δ∇ρ)] · (∇ρ·AG)        (T2)
      + 2·vsigma · (δ∇ρ·AG)                                       (T3)
```

`AA = φ_μφ_ν`, `AG = φ_μ∇φ_ν + ∇φ_μφ_ν`. **T3 comes from `∇ρ` being an
argument of the existing `AG`-coupling factor**, not from differentiating
the `vrho`/`vsigma` coefficients — differentiating only the coefficients
silently truncates the Taylor expansion. The decomposition was checked two
ways before use: a direct multivariable chain rule, and a numeric check
against a Maxwell-consistent toy energy density `E(ρ,σ) = ρ³σ + ρ²σ² + ρσ`
(matched to `~7e-18`; a non-Maxwell-consistent toy `vrho`/`vsigma` gave a
real nonzero disagreement, which is the `v2rhosigma` equivalence the
contraction implicitly assumes).

Design rule:

- When differentiating an existing contraction, differentiate every factor
  that depends on the perturbed quantity, not just the named coefficients —
  and gate each term in isolation so a dropped term is a specific
  disagreement, not one combined mismatch.

### 3a. A combined exchange-correlation functional must not have `fxc[c_functional]` added on top

`compute_analytic_xc_hessian_vector_product` and `_polarized` sum
`fxc[exchange_functional] + fxc[correlation_functional]`. When the
exchange slot holds a *combined* exchange-correlation libxc entry (B3LYP,
PBE0, HSE06 — every named hybrid, and any input using one of those names),
that entry already carries the whole XC, so adding a second
`fxc[correlation_functional]` **double-counts a correlation `fxc`**. This
is the exact case the KS-matrix build guards on
(`src/dft/driver.cpp`, `if (exchange->is_combined_exchange_correlation())`
— "configured correlation is ignored"); the analytic Hessian must apply
the same guard.

`src/dft/analytic_hessian.cpp`'s `drop_correlation_if_combined` zeroes
`v2rho2_c`, and for GGA also `v2rhosigma_c` / `v2sigma2_c` / `vsigma_c`
(the last feeds the T3 `2·vsigma·δ∇ρ` term), under
`exchange_functional.is_combined_exchange_correlation()`, in all four
branches (RKS/UKS × LDA/GGA).

**Found latent** — DFT SOSCF rejects hybrids (`soscf_dft_hybrid_blocked`),
so no shipped SOSCF run reached the combined-XC analytic Hessian. It
surfaced via the RKS Hessian-scale probe
(`docs/SOSCF_DFT_RKS_HESSIAN_SCALE_SCOPE.md`): on B3LYP and PBE0 the
composed `h_op` missed a central-FD of the true total energy by
`~0.7% – 5.6%`, direction-dependent, while every separate-slot functional
(LDA, PBE, B88, LYP, PBE_X+LYP) was exact to `1.000000`. Zeroing the
spurious `fxc[c_functional]` lands every hybrid direction on `1.000000`.

Design rule:

- Any code that sums `fxc[x] + fxc[c]` must apply the
  `is_combined_exchange_correlation()` guard the KS-matrix build applies —
  a combined-XC functional carries its correlation in the exchange slot.

Gate: `tests/dft_analytic_hessian_polarized_production.cpp` ::
`check_combined_no_double_count` — asserts the `correlation_functional`
argument is inert (RKS and polarized entries give identical results
whether it is `gga_c_pbe` or the combined functional itself) for
`hyb_gga_xc_b3lyp` and `hyb_gga_xc_pbeh`. Mutation-verified: reverting
`drop_correlation_if_combined` makes all six assertions fail by
`0.012 – 0.018`.

### 4. The one-term trace identity is the wrong quantity to test the Hessian against

`Tr(δP·δV_xc(δP)) = ∂²E_xc/∂κ²` does **not** hold — it is missing a term.
The Cayley transform is nonlinear in `κ`, so
`P(κ) = P₀ + κ·δP + ½κ²·δ²P + …` with `δ²P ≠ 0`, and the full second
derivative is:

```
∂²E_xc/∂κ² = Tr(δP·δV_xc(δP))  +  Tr(d²P/dκ² · V_xc_ground)
```

Confirmed by porting PySCF's `nr_rks_fxc` and running both codes on
identical water/STO-3G/PBE input: PySCF's own analytic kernel reproduces
Planck's exact "disagreement" with the FD of `E_xc(κ)` alone (`-0.15740`
vs `9.68822`), while the two-term identity closes to 5+ sig figs
(`-0.15740 + 9.84562 = 9.68822`). The curvature term is the standard
orbital-energy-difference piece every Newton/CPHF formulation carries
separately (`fvv·x - x·foo` in PySCF's `gen_g_hop_rhf`, built before
`vind(dm1)`). `compute_analytic_xc_hessian_vector_product` supplies only
the `fxc`-kernel term, as it should, and does so exactly.

Design rule:

- Verify the fully-composed `h_op` against a finite difference of the
  **total** energy `E(κ)`, never `compute_analytic_xc_hessian_vector_product`
  alone against `E_xc(κ)`. Testing a decomposition that does not match how
  the total-energy FD check is assembled was D2.1's actual defect.

### 5. Symmetric verification directions produce spuriously passing or spuriously zero checks

Recurred three times in this work (D2.1's HOMO-LUMO even-in-κ finding,
D3.0's `(i=0,a=n_occ)` zero-cross-term, D3.1's same-spatial-index mixed
term giving bit-identical `E(±h,±h)`). A HOMO-LUMO direction is often
exactly even in `κ` — zero first-order coupling under the molecule's
point-group symmetry, not merely small. A same-spin-index choice for a
cross-spin term suppresses it to machine zero.

Design rule:

- Before trusting any single-direction FD check, confirm the checked
  quantity is nonzero on both sides — or deliberately choose indices that
  differ across whatever symmetry (spin, occ/virt numbering, point group)
  the system has.

## What was fixed / built

1. **`evaluate_lda_fxc` / `evaluate_gga_fxc`** added to
   `DFT::XC::Functional`, mirroring `evaluate_{lda,gga}_exc_vxc` (same
   chunked/threaded shape, same `is_{lda,gga}_like()` guards). Verified
   (`planck-dft-fxc-selfcheck`) that libxc's analytic second derivative
   reproduces a central FD of its own first derivative on `lda_x`,
   `lda_c_pw` (cross-spin term `aa=0.104, ab=-0.289, bb=0.327` at
   ρ_α=0.10, ρ_β=0.06 — `lda_x`'s is ~0 and would not exercise an `aa`↔`ab`
   swap), and PBE including the `v2rhosigma` mixed-partial equivalence.
   Mutation-verified.

2. **The point-level contraction ladder** (LDA unpolarized → LDA polarized
   → GGA T1/T2/T3 unpolarized → GGA polarized → `(a,i)` packing), each
   verified against the FD oracle to `~1e-8`–`~1e-11` on synthetic
   single-point / few-AO grids before the next. Every step mutation-
   verified. The polarized beta channel is written independently, not alpha
   with labels swapped.

3. **`pack_hessian_vector_product_cphf_order`** — a real production
   function (not a probe) in its own tiny translation unit, translating
   AO-basis `δV_xc` into the CPHF virtual-major `a·n_occ + i` convention.
   This codebase has two live conventions:
   `build_{rhf,uhf}_cphf_matrix` use virtual-major;
   `ResponseExcitationSpace::flat_index(i,a) = i·n_virt + a` (the FD
   oracle's, TDDFT's) is occupied-major. Verified on non-square fixtures
   (a transposition hides behind a square matrix's symmetry).

4. **`ResponseExcitationSpace` and the FD-kernel builders were moved out of
   `driver.cpp`'s anonymous namespace** (declared in `driver.h`) so they
   are callable from verification code. Pure move, behavior-neutral.

5. **`correlation vwn5` never resolved** — mapped to `"lda_c_vwn_5"`, which
   is not a libxc name (VWN5 is `lda_c_vwn`, no suffix). Zero regression
   coverage exercised it. One-line fix, found while building a test
   fixture.

## Validation strategy that should remain in place

- The seven point-level ctests above are the permanent gate — fast,
  always-run, zero production code path.
- The consuming SOSCF-vs-DIIS regression cases (`docs/SOSCF_DFT.md`)
  exercise the assembled `h_op` end to end.
- The FD-kernel oracle stays in-tree as an independent correctness check
  for future changes, even though it is not the production Hessian.

## What was measured but is not kept

Four whole-molecule probes (`PLANCK_FXC_F3_1_CHECK`, `_F3_2_CHECK`,
`_F3_3_4_CHECK`, `_F3_4_5_CHECK`) verified the algebra composes with the
real grid/AO-projection machinery, once, during derivation — then were
deleted outright per the project-wide decision that debug probes must
become standalone tests or be removed. Converting them was found
impractical: each needs a converged SCF's internal state `DFT::Driver`
does not expose.

One probe earned its keep before deletion: a fresh, independently-written
copy of F3.4.2/F3.4.3's algebra in the F3.4.5 probe was missing a
`σ_bb`-rooted term for the `beta-only`-driving-alpha-channel case — while
the point-level test file had it correctly. `"alpha-only x ⇒ δσ_bb = 0"`
is true; the converse (`"δρ_β nonzero ⇒ σ_bb-rooted terms matter"`) must be
re-derived at every site, not inferred from one working implementation.

## Remaining architecture concern

Hybrid and range-separated functionals need the exact-exchange (`K`)
response on top of `fxc` — the machinery `build_rhf_cphf_matrix` already
has for HF but which is unbuilt for the KS path. The analytic `fxc`
functions themselves are functional-agnostic; the gap is a `K`-response
contraction against an RKS/UKS `C`. `docs/SOSCF_DFT_HYBRID_SCOPE.md`
scopes closing it (the `K` response is one more linear-in-density term
built with the direct K builders already in the tree); until it lands,
DFT SOSCF rejects hybrids with a warning (`docs/SOSCF_DFT.md`).

When that lands, the combined-XC `fxc` double-count fixed in invariant 3a
goes from latent to live — `check_combined_no_double_count` is the unit
gate, and the hybrid SOSCF-vs-DIIS regression case added in
`SOSCF_DFT_HYBRID_SCOPE` H2 is the end-to-end one (a regression of the
guard shows there as a hybrid SOSCF convergence slowdown — linear instead
of superlinear — the same signature the RKS Hessian-scale fix had,
`docs/SOSCF_DFT_RKS_HESSIAN_SCALE_SCOPE.md`).
