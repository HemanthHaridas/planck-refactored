# Double-Hybrid Gradient: KS-veff Substitution (N3.5.4-N3.5.7)

In-flight scope. Fold into `docs/DOUBLE_HYBRID_GRADIENT_SCOPE.md` (or its
answer-doc successor) when this lands. Canonical status lives in
`vault/Status/`.

**Question this scopes:** the RKS double-hybrid analytic gradient
(`PLANCK_DFT_DH_GRADIENT`) is **3.0e-4 Ha/Bohr off finite difference** on
water/STO-3G B2PLYP. N3.4 isolated the residual to the PT2 correction's
Z-vector / mean-field terms; the literature + PySCF comparison
(`docs/DOUBLE_HYBRID_GRADIENT_SCOPE.md` N3.5) pinned it to Planck's RMP2
gradient hardwiring the **HF** mean field where a double hybrid needs the
**KS** one. This doc specifies the fix.

## Short answer

Three sites in the shared RMP2-gradient code call an HF Fock/veff builder.
For a double hybrid all three must use the KS mean-field response
(`J + V_xc + c_x K`) instead. PySCF gets this for free -- its `grad/mp2.py`
calls `mp._scf.get_veff` / `mp._scf.gen_response`, which dispatch to KS when
`_scf` is a KS object. Planck has no `_scf` object to dispatch on:
`build_veff_from_density` is a hardcoded HF Fock builder. So the fix is an
**optional KS-veff callback** injected into `build_rmp2_lagrangian` and
`build_rmp2_energy_weighted_density`, plus a gradient-side correction to the
`vhf1` derivative contraction. RMP2 (HF-reference) callers pass nothing and
stay byte-identical.

## The sign convention (read before touching any term)

Planck and PySCF disagree on the sign of the ERI derivative element:
`ObaraSaika::_compute_eri_deriv_elem` returns `dI` such that
`grad = + Sum 0.25 * gamma * dI` matches the analytic gradient
(`accumulate_eri_gradient_permutations`, `src/gradient/gradient.cpp`).
PySCF's `int2e_ip1` is `-d(pq|rs)/dR`, so PySCF writes `de -= ...`. **The two
flips cancel** -- both codes match analytic. Do **not** transcribe PySCF's
`de -=` lines. Every Planck helper this scope uses already carries Planck's
sign:

| helper | sign |
|---|---|
| `build_veff_from_density` | `+(J - 0.5 K)` -- same as PySCF `get_veff`'s `+(vj - 0.5 vk)` |
| `compute_two_density_exchange_kernel_gradient` (new, landed) | `grad += 0.25 * gamma * dI`, `gamma = -coeff * (P1 P2 + P2 P1)` -- same `accumulate_eri_gradient_permutations` path as the RHF 2e gradient |
| `compute_xc_nuclear_gradient_rks` | returns `+ dE_xc/dR`; the driver does `_gradient = wf_grad + xc_grad` |

The two earlier hand-written `dG` attempts failed on **factors** (`2*P^z`, a
4-term expansion), not sign. The forms below are factor-correct.

## Architecture map: PySCF `grad/mp2.py` vs Planck

RMP2 gradient FD-verified equal to PySCF to ~1e-7, so the structure is
sound. The pieces:

| quantity | PySCF | Planck | status |
|---|---|---|---|
| gamma^1 `doo/dvv` | `_gamma1_intermediates` -> `-dm1occ, dm1vir` | `rmp2_gamma1_intermediates` -> `-doo, dvv` (same `2 llT - llT`) | identical |
| `part_dm2` | `t2.transpose(0,2,3,1)*4 - t2.transpose(0,3,2,1)*2` | `C_v(p,a) C_v(q,b) (4 tab - 2 tba)` | identical |
| `dm2buf` | sym r<->s + halve diagonal (for `s2kl`-packed ERIs) | no r<->s sym (full unpacked ERIs) | different by design, gate-verified equivalent |
| `Imat` (generalized Fock) | `einsum('ipx,iqx->pq', eri0_s2kl, dm2buf)`, fused into the per-atom derivative loop | standalone loop `imat(q,v) += (pq\|rs) dm2buf[p,v,r,s]` (N1.3.1 unfused) | same value |
| `Imat -> MO` | `mo.T Imat S mo * (-1)` | `-mo.T imat_ao S mo` | identical |
| **`Xvo` RHS `vhf`** | `mp._scf.get_veff(mol, dm1_from_gamma1) * 2` -- **KS for KS `_scf`** | `veff_corr_ao = 2 * build_veff_from_density(dm1_corr_ao)` = **HF `2(J - 0.5K)`** | **N3.5.4** |
| `Xvo` assembly | `C_v.T vhf C_o + Imat[o,v].T - Imat[v,o]` | `C_v.T veff_corr C_o + imat_mo[o,v].T - imat_mo[v,o]` | identical structure |
| **Z-vector operator** | `cphf.solve(fvind)`, `fvind = C_v.T (mp._scf.get_veff(dm+dm.T)) C_o * 2` -- **KS** | `solve_rhf_cphf` (HF CPHF) for RMP2; **N2 already replaced it** with `build_ks_orbital_hessian_op` for the DH path (`solve_pt2_relaxed_density`) | done for DH |
| relaxed `dm1mo` | `[oo]=doo+dooT; [vv]=dvv+dvvT; += _response_dm1(Xvo)` (adds `[v,o]/[o,v]=z`) | `corr_relaxed_mo = dm1_corr_mo; [v,o]=z; [o,v]=zT` | identical |
| `zeta` / energy-weighted | `zeta_ij=0.5(e_i+e_j); zeta[vo]=e_o; zeta*dm1mo`; `+ make_rdm1e` (= `W_ref`) | identical `zeta_weights`; `W_ref + mo (zeta_w .* corr_relaxed) mo.T` | identical |
| **`vhf_s1occ`** | `p1 (mp._scf.get_veff(mol, dm1+dm1.T)) p1` -- **KS veff**; `dm1` = full relaxed correction (gamma^1 + z blocks, no `hf_dm1`) | `occ_proj (build_veff_from_density(dm1_corr_relaxed_ao + T)) occ_proj` -- **HF veff**; density arg matches | **N3.5.5** |
| `dm1p` | `hf_dm1 + dm1*2` (`dm1` = relaxed correction) | `hf_dm1 + 2 * dm1_corr_relaxed_ao` | identical |
| **`vhf1[k]` build** | per-atom loop: `einsum(eri1_ip1, hf_dm1)` Coulomb `- 0.5 einsum(...)` exchange (+ 2 more) = **HF `2J - K` derivative on `hf_dm1`** | fused loop: `vhf1[atom][c](r,s) += dI hf_dm1(p,q); (r,q) -= 0.5 dI hf_dm1(p,s);` (+ 2 more) = **same HF `2J - K`** | identical (HF -- **N3.5.6** for the relaxed-correction part) |
| final `de` term set | `+s1.im1` (both) `+h1.dm1` (full) `-s1.zeta` (both) `-2 s1.vhf_s1occ` `-vhf1.dm1p` | `+dST.imat_ao` (both) `+one_e(dm1_total)` `-dST.zeta` (both) `-2 dST.vhf_s1occ` `+vhf1.dm1p` | equivalent (sign-convention flip cancels; RMP2 FD confirms) |

**Three HF-vs-KS sites:** `Xvo` veff (N3.5.4), `vhf_s1occ` veff (N3.5.5),
`vhf1` derivative kernel for the correction part (N3.5.6). The Z-vector
operator (a fourth site) is already KS-correct for the DH path via N2.

## The KS mean-field response

For B2PLYP (a *truncated* DH -- orbitals stationary for `E_hyb`, PT2 a
non-SCF add-on; Hait & Head-Gordon arXiv:1803.10284, Neese/Schwabe/Grimme
JCP 126 124115), the KS response applied to a density `d` is

    R_KS[d] = J[d]  -  0.5 * c_x * K[d]  +  V_xc_response[d]

- `J[d]` -- `_compute_2e_j_direct(shell_pairs, d, ...)`, the memory-direct
  Coulomb builder (same one `build_ks_orbital_hessian_op` uses).
- `K[d]` -- `_compute_2e_k_direct(...)` for a global hybrid; for a
  range-separated DH (HSE-style, out of scope until N3.7) it splits into a
  full-range and a screened short-range piece with `c_fr` / `c_sr`.
- `V_xc_response[d]` -- the analytic `f_xc` Hessian-vector product,
  `DFT::Driver::compute_analytic_xc_hessian_vector_product(grid, ao_grid,
  ground_density, d, x_functional, c_functional)` -- **exactly the XC piece
  `build_ks_orbital_hessian_op` already composes**, in AO form (not packed).

`build_veff_from_density` returns `J[d] - 0.5 K[d]` (HF). The KS override
scales the `K` by `c_x` and adds `V_xc_response`. Same overall `+` sign.

The `Xvo` RHS multiplies this by 2 (`veff_corr_ao = 2 * ...`), matching
PySCF's `vhf = get_veff(...) * 2`.


## The equations (Neese/Schwabe/Grimme, JCP 126, 124115, 2007)

The paper's closed-shell formalism (Sec. III.A) is the reference. B2PLYP is
a *truncated* DH: KS orbitals stationary for `E_hyb` (combined XC + `a_x K`,
`a_x = 0.53`, no PT2), `E_PT2` a non-SCF add-on, `c_PT = 0.27`.

**Total gradient:** `E^x = E_SCF^x + c_PT E_PT2^x`.

- **`E_SCF^x` (Eq. 14)** = the KS gradient: `compute_rks_gradient` +
  `compute_xc_nuclear_gradient_rks`. Landed.

- **Z-vector (Eq. 27):** `(eps_a - eps_i) Z_ai + R(Z)_ai = -L_ai`, response
  operator (Eq. 41)
  `R(D')_munu = Sum D'_kt [4(munu|kt) - (muk|nut) - (mut|nuk)] + R^XC(D')_munu`
  where `R^XC` is the **f_xc second-derivative kernel** on `D'`. This is
  exactly `build_ks_orbital_hessian_op` (N2). The `a_x` exchange weight is
  in the `-(muk|nut) - (mut|nuk)` term... actually Eq. 41 as printed has
  coefficient 1 there, with `a_x` folded into `R^XC`'s HFX piece for the
  hybrid -- `build_ks_orbital_hessian_op` uses `-0.5 c_x dK` + the analytic
  fxc, which is the same operator.

- **Lagrangian (Eq. 40):**
  `L_ai = R(D')_ai + Sum_bc Sum_j t_cb^ij[(ac|jb) - (ab|jc)]`
  `      - Sum_kl Sum_b t_ab^kl[(ki|lb) - (kb|li)]`.
  The `R(D')_ai` term applies the response operator (INCLUDING `R^XC`) to
  the **unrelaxed** difference density `D'` (Eq. 37/38: the gamma^1
  blocks). -> this is `veff_corr_ao` in `build_rmp2_lagrangian`.

- **Effective 2-particle density (Eq. 46):**
  `Gamma^{SCF+PT2}_munukt = 1/2 P_munu P_kt - 1/4 P_muk P_nut`
  `                       + D_munu P_kt - 1/2 D_muk P_nut + Gamma^NS`.
  The `1/4`, `1/2` exchange coefficients are **full HF weight, NOT a_x**.
  The `a_x` scaling of exchange lives entirely in `E_SCF` and `R^XC`; the
  separable PT2 2e term is a pure MP2 object. -> **the `vhf1` HF `2J - K`
  derivative kernel in `build_rmp2_gradient_intermediates` is CORRECT for a
  double hybrid. No exchange-weight fix is needed.**

- **The XC term in `E_PT2^x` (Eq. 33).** The paper is explicit (p. 6): this
  is **NOT** `Tr[D . V_xc^(x)]` (the naive `rho_P^x -> D^x` analog). "The XC
  contributions to the PT2 gradient arises from the contraction of the
  relaxed PT2 difference density with the derivative of the SCF operator.
  Since the SCF operator already contains the first derivative of the XC
  potential and the PT2 energy is not stationary ..., a response-type term
  arises which requires the **second functional derivative of the XC
  functional**." Eq. 33's XC part:
  ```
  E_PT2^x|_XC = Sum_s INT Sum_z zeta(P)^(x) { f^{rho_s rho_s} rho_D^s
                  + 2 f^{gamma_ss z}(grad rho_P^s . grad rho_D^s)
                  + f^{gamma_ss' z}(grad rho_P^s' . grad rho_D) } dr
              + INT { 2 f^{gamma_ss}(grad rho_P^(x) . grad rho_D^s)
                  + f^{gamma_ss'}(grad rho_P^s'(x) . grad rho_D^s) } dr
  ```
  **CORRECTION (N3.5.7.5, after obtaining and reading the paper's actual
  text rather than relying on this transcription alone):** the line below is
  WRONG and caused N3.5.7.1-4 to build the wrong quantity. `zeta(P)^(x)` is
  **not** the geometry derivative of an AO-pair matrix element
  `<mu|V_xc^[2]|nu>` -- cross-referencing the paper's own definition of
  `zeta(D')` (used one page earlier in the AO-basis response operator)
  shows `zeta` ranges over `{rho_D', gamma_alphaalpha(D'), gamma_betabeta(D'),
  gamma_alphabeta(D')}`, i.e. `zeta(P)^(x)` is the geometry derivative of the
  SCF density's own scalar/vector quantities `rho_P` and `grad_rho_P`
  (`gamma(P) = grad_rho_P . grad_rho_P`) -- a **direct per-point scalar grid
  integral**, contracted with FIXED fxc-weighted coefficients built from
  `D`, no AO-pair scatter anywhere in this term. See N3.5.7.5 for the full
  finding, the correct building blocks already dead-code-present in the
  tree (`drho_channel`, `dg_axis_spin` in `dft_gradient.cpp`), and what
  remains open there.
  ~~i.e. the geometry (basis-function) derivative of the **f_xc-kernel matrix
  elements** `<mu| V_xc^[2][rho_P] |nu>` (v2rho2, v2rhosigma, v2sigma2 times
  basis grad/Hessian), contracted with the relaxed density `D` and the SCF
  density gradient. One derivative order above `compute_xc_nuclear_gradient_rks`.~~

## Steps

### N3.5.4 -- KS response in the Lagrangian RHS -- LANDED

`KsVeffFn = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>` added to
`mp2_gradient.h`. `build_rmp2_lagrangian` takes it (default `{}` = HF
`build_veff_from_density`); when set, `veff_corr_ao = 2 * ks_veff(dm1_corr_ao)`.
`build_pt2_mo_intermediates` forwards it; `RMP2PreSolved` carries it for the
`build_rmp2_gradient_intermediates` presolved path. The driver's DH block
builds the closure:
```cpp
ks_veff(d) = _compute_2e_j_direct(d)
           - 0.5 * c_fr * _compute_2e_k_direct(d, Coulomb)
           - 0.5 * c_sr * _compute_2e_k_direct(d, ShortRange, omega)
           + compute_analytic_xc_hessian_vector_product(grid, ao_grid,
               ground_density, d, x_functional, c_functional)
```
= exactly `R(D')` from Eq. 40/41. RI + DH gradient rejected up front.

**Result: water/STO-3G B2PLYP FD error 3.05e-4 -> 1.87e-4.** RMP2
conventional + RI gradient BYTE-IDENTICAL (callback null); RMP2 / geomopt /
B3LYP-SOSCF / HSE06-grad / B2PLYP-energy regressions all pass; c_PT2
linearity `rel 2.1e-13`.

**Gotcha:** the `PLANCK_DFT_DH_GRADIENT_SELFCHECK` c_PT2-linearity probe
re-runs `build_pt2_mo_intermediates` at `2*c_pt2` -- it must be passed the
SAME `ks_veff` (else it compares KS-veff vs HF-veff; briefly regressed to
rel 1.85e-2).

### N3.5.5 -- KS response for `vhf_s1occ` -- TRIED, REVERTED

Plumbing landed (`KsVeffFn` on `build_rmp2_energy_weighted_density` +
`RMP2PreSolved`, threaded from the driver), but using it made FD WORSE
(1.87e-4 -> 2.86e-4). The `vhf_s1occ` HF-vs-KS delta is real (`max|W_ks -
W_hf| = 1.05e-3`) but contributes `< 1e-4` to the gradient with the wrong
sign. `vhf_s1occ` in the paper is `W_ij^{PT2}` (Eq. 42) which carries
`-1/2 R(D)_ij` -- i.e. the response operator on the RELAXED `D`, not a
separate `get_veff`. The current HF `build_veff_from_density` there is a
PySCF-`grad/mp2.py` artifact (`p1 . get_veff(dm1+dm1.T) . p1`), and PySCF
has no validated DH gradient. Whether the paper's `-1/2 R(D)_ij` (with
`R^XC`) is the right replacement is unresolved; the effect is small and
revert is safe. **Shipped: `vhf_s1occ` stays HF.** The `KsVeffFn` params
are kept inert (default `{}`) for a later revisit.

### N3.5.6 -- delete the wrong `vhf1` K-weight block

Per Eq. 46, the separable PT2 2e term uses full HF exchange weight, so the
`vhf1` HF `2J - K` derivative kernel is correct as-is. The
`PLANCK_DFT_DH_ZVECTOR_KERNEL_FIX` block (K-weight correction via
`compute_two_density_exchange_kernel_gradient`) is **wrong by the paper**
and every empirical attempt overshot (~22x / ~8x). Delete it, and the
`PLANCK_DFT_DH_XC_RESPONSE` block, and the now-unused
`compute_two_density_exchange_kernel_gradient` helper (unless a UKS or
range-separated DH use appears -- check before removing the helper).

### N3.5.7 -- implement the Eq. 33 f_xc grid term

Split into small steps, each individually gated so a bad step is cheap to
isolate or revert.

#### N3.5.7.0 -- land N3.5.6 first -- LANDED

Both dead-end blocks deleted from `driver.cpp`, along with the now-orphaned
helper:

- the `PLANCK_DFT_DH_ZVECTOR_KERNEL_FIX` block (the K-weight correction via
  `compute_two_density_exchange_kernel_gradient`, wrong by Eq. 46 -- full HF
  exchange weight is correct there),
- the `PLANCK_DFT_DH_XC_RESPONSE` block (the failed first attempt at this
  term, `g(dm1_corr_relaxed_ao) - g(0)` via `compute_xc_nuclear_gradient_rks`,
  which made FD worse) -- replaced by a short comment pointing at N3.5.7 as
  the real term's landing spot,
- `HartreeFock::Gradient::compute_two_density_exchange_kernel_gradient`
  (`gradient.{h,cpp}`) -- had no other caller; the shared
  `compute_two_electron_kernel_gradient` helper it called stays (still used
  by `compute_closed_shell_exchange_kernel_gradient`).

Both binaries build clean. Verified inert: `water_rmp2_gradient_smoke`,
`water_rmp2_gradient_fd`, `water_dft_hse06_gradient_symm_ultrafine_fd`,
`water_rks_b3lyp_soscf_631g`, `h2_dft_b2plyp_sto3g` all pass; water/STO-3G
B2PLYP FD (`tests/dft_gradient_fd.py` with `PLANCK_DFT_DH_GRADIENT=1`)
reproduces **1.877e-4 Ha/Bohr**, matching the N3.5.4 baseline to the quoted
digits -- deleting the dead code changed nothing.

#### N3.5.7.1 -- point-level f_xc-kernel-matrix-element derivative, LDA only -- LANDED

New free function `DFT::Gradient::compute_xc_kernel_nuclear_gradient` in a
new `src/dft/dft_kernel_gradient.{h,cpp}` pair (mirrors `dft_gradient.{h,cpp}`'s
existing split). Signature shaped like `compute_xc_nuclear_gradient_rks` plus
the ground/relaxed density pair and both functionals:

```cpp
std::expected<Eigen::MatrixXd, std::string> compute_xc_kernel_nuclear_gradient(
    const HartreeFock::Molecule &mol, const HartreeFock::Basis &basis,
    const MolecularGrid &grid, const AOGridEvaluation &ao, const AOGridHessian &hess,
    const XCGridEvaluation &xc,
    const Eigen::Ref<const Eigen::MatrixXd> &ground_density_restricted,
    const Eigen::Ref<const Eigen::MatrixXd> &relaxed_density_restricted,
    const XC::Functional &exchange_functional, const XC::Functional &correlation_functional);
```

LDA term only (`f^{rho rho} rho_D`), by exact analogy with
`compute_xc_nuclear_gradient_rks`'s own LDA branch: same per-grid-point
`vtmp`/`vxc1` scatter, contracted against **P** (`ground_density_restricted`),
with `v2rho2 * rho_D(p)` playing `vrho`'s role as a frozen per-point scalar
coefficient. GGA functionals return an explicit error (not a silently
incomplete gradient) until N3.5.7.2. Same `drop_correlation_if_combined`
guard as `compute_analytic_xc_hessian_vector_product` (risk 2).

**Two findings, both in test construction, not the production routine --
recorded because each cost a debugging pass and the reasoning generalizes to
N3.5.7.2/.3:**

1. **`Tr[P . f(rho_P) . rho_D]`'s full nuclear derivative has THREE terms,
   not one** (confirmed by direct sympy differentiation of a two-atom toy
   model): `f'` moving with `rho_P`, the P-contracted AO-pair factor moving
   (what the routine computes), and `rho_D` moving. Eq. 33's Term 1 is
   *specifically* the middle piece -- a response-operator term that holds
   `f^{rho rho}` and `rho_D` frozen as per-point scalars, not the literal
   total derivative of that bilinear. A first FD test built the full
   three-term derivative and mismatched the analytic routine by a
   non-constant **1.4x-2.2x** across two random seeds -- the signature of
   comparing against the wrong reference quantity (the missing terms' size
   depends on the arbitrary P/D matrices), not a scale or sign bug.

2. **The `owner_atom` scatter inside the copied loop is a point-translation
   term, not part of N3.5.7.3.** For a real atom-centered quadrature the
   grid point translates rigidly with its owning atom, so (verified
   symbolically on a two-atom, two-basis-function model) a basis function
   centered on the point's own owner has **zero** net derivative from that
   point (function and point move together), while a basis function on a
   *different* atom sees the point move past it. `compute_xc_nuclear_gradient_rks`'s
   `owner_atom` direct-add is exactly this piece, and this routine's copy of
   that scatter carries it too -- it is **not** the same thing as the Becke
   partition *weight* derivative (N3.5.7.3's job, the separate
   `w_atomic*dpartition*exc_density`-shaped term). A second FD test version
   held the grid fixed (to exclude only the weight-derivative piece, as
   N3.5.7.1's original plan intended) and still mismatched, by a different
   non-constant ratio (0.7x-17x), because holding the grid fixed also drops
   the point-translation piece the analytic routine inherently includes.
   Fixed by rebuilding the grid (not just the basis) at each FD-displaced
   geometry while freezing the per-point `w*coeff` array by VALUE (grid
   points keep the same index-to-relative-offset meaning across geometries
   since `MakeMolecularGrid` is deterministic per atom).

With both fixes, analytic and FD agree to **ratio 1.000000** (6 significant
figures) across two seeds and two step sizes. Gated by
`tests/dft_kernel_gradient_fd.cpp` (CTest `planck-dft-kernel-gradient-fd`):
a real He2/STO-3G system (real basis/grid/AO-Hessian machinery is required
here, unlike the synthetic-single-point style of `dft_fxc_selfcheck.cpp` /
`dft_gga_hessian_selfcheck.cpp`, because this routine differentiates basis
functions with respect to nuclear position), arbitrary fixed symmetric P/D
matrices, central difference against `term1_at_current_geometry` (a second,
independent implementation of Term 1, not a reuse of the routine's
internals), plus a translational-invariance cross-check (`Sum_A grad_A = 0`).

#### N3.5.7.2 -- GGA piece -- LANDED

`compute_xc_kernel_nuclear_gradient` extended with the GGA branch: `rho_P`,
`rho_D` and their gradients projected onto the grid via direct AO
contraction (`2*gx.dot(P_sym*phi)` etc, matching `evaluate_density_on_grid`'s
own `2*grad_phi.P.phi` convention bit-for-bit); `v2rho2/v2rhosigma/v2sigma2`
and `vsigma` evaluated via `evaluate_gga_exc_vxc`/`evaluate_gga_fxc`;
`delta_vrho`/`delta_vsigma`/`delta_gradient_term` built exactly as
`compute_analytic_xc_hessian_vector_product`'s own GGA branch does (ground=P,
trial=D); the `wv0/wv1..3 -> aow/aow_x/y/z -> vtmp` construction copied from
`compute_xc_nuclear_gradient_rks`'s GGA branch with `vrho -> delta_vrho`,
`coeff -> delta_gradient_term`. Same `drop_correlation_if_combined` guard on
`v2rho2_c/v2rhosigma_c/v2sigma2_c/vsigma_c` (risk 2). The `owner_atom`
point-translation piece N3.5.7.1 found is inherent to the copied
`vtmp`/`vxc1` scatter structure, not LDA-specific, so the GGA branch inherits
it automatically -- no separate handling needed; only the Becke partition
*weight* derivative stays deferred to N3.5.7.3.

**One finding, in the test's independent reference formula, not the
production routine (same discipline as N3.5.7.1's two findings -- recorded
because it cost a debugging pass and the mechanism generalizes):** a first
FD test (`term1_gga_at_current_geometry`) built its frozen per-point
coefficients as `wv0 = w0*delta_vrho*0.5` -- copying the analytic routine's
own *internal* `wv0` from its two-term product-rule `aow`/`aow_x`
construction -- and mismatched the analytic routine by a **consistent
~1.8x** across both seeds and both step sizes. Consistency (as opposed to
N3.5.7.1's earlier *non-constant* 1.4x-2.2x and 0.7x-17x ratios) was itself
the signal: a constant ratio close to a small rational number is the
signature of a genuine missing/extra normalization factor, not a
structurally wrong reference quantity. The true physical matrix element the
routine differentiates is `V_munu(p) = w*[delta_vrho*phi_mu*phi_nu +
delta_gradient_term.(grad_phi_mu*phi_nu + phi_mu*grad_phi_nu)]` -- **no
extra 0.5** on the rho-rho term, matching `compute_analytic_xc_hessian_vector_product`'s
own `delta_v_xc_ao` convention exactly (verified both by re-deriving
`compute_analytic_xc_hessian_vector_product`'s own coefficient and
numerically, via FD on a toy two-Gaussian system: `Tr[P.V] =
w*(delta_vrho*rho_P + delta_gradient_term.grad_rho_P)`, no 0.5). The internal
`0.5` inside the analytic routine's own `wv0` exists **only** because that
routine's `aow`/`aow_x` construction explicitly differentiates both AO
factors of the product rule (unlike the LDA branch, which relies on an
implicit symmetric doubling through `P_sym` and needs no such factor); it is
an artifact of that particular derivative-construction mechanism, not part
of what `V` *is*. Fixed by using the unhalved `w0*delta_vrho`/`w0*delta_gradient_term`
in the test's independent reference.

With the fix, analytic and FD agree to **ratio 1.000000** (6 significant
figures) across two seeds and two step sizes, same as N3.5.7.1. Extended
`tests/dft_kernel_gradient_fd.cpp`'s He2/STO-3G FD gate with a
`check_gga_kernel_gradient` sibling using PBE (`gga_x_pbe`/`gga_c_pbe`, a
non-combined pair so `drop_correlation_if_combined` stays inert -- same
choice `dft_gga_hessian_selfcheck.cpp` makes), same moving-grid-with-frozen-
coefficient-values construction N3.5.7.1 established (rebuilding the grid,
not just the basis, at each displaced geometry -- N3.5.7.1's second defect
would have recurred identically here otherwise).

#### N3.5.7.3 -- Becke partition weight derivative -- LANDED

The one piece N3.5.7.1 confirmed was genuinely still missing (the
point-translation piece is already in, see N3.5.7.1's finding 2):
`compute_xc_nuclear_gradient_rks` also differentiates the quadrature
*weight* itself (`becke_partition_owner_derivatives`) and multiplies the
point-level energy-density-like quantity by that. Added the matching piece
inside the existing per-point loop (shares the per-point `dpartition` call,
not a separate pass): `w_atomic * dpartition(atom_A,q) * term1_scalar`,
where `term1_scalar` is the un-weighted per-point Term-1 quantity
(`delta_vrho*rho_P` for LDA; `delta_vrho*rho_P + delta_gradient_term.grad_rho_P`
for GGA -- built once per branch and reused by both the new weight-scatter
and the existing AO-derivative scatter), playing `exc_density`'s role
exactly.

**Mechanism reuse, not duplication:** `becke_partition_owner_derivatives`
(116 lines of Becke-weight-derivative algebra, previously file-local to
`dft_gradient.cpp`, used by both `compute_xc_nuclear_gradient_rks` and
`_uks`) was **exported** from `dft_gradient.h` rather than copied a third
time -- unlike the small per-TU shell/atom bookkeeping helpers
(`build_bf_shell_map` etc, ~15-30 lines, private wiring), this is real
shared algorithm, and this codebase's own convention (see the "no
spaghetti" project memory) is one mechanism with a parameter over parallel
copies once a helper crosses from trivial wiring into real logic. The move
is behavior-neutral: the function's body is untouched, only its linkage
changed (anonymous namespace -> `DFT::Gradient` public surface, declared in
the header). `dft_gradient.cpp`'s two existing call sites resolve it via
unqualified lookup in the same enclosing namespace unchanged.

**Verify -- and the doc's own original plan here was wrong.** This section
originally said the FD gate could switch to "re-evaluate `v2rho2` and
`rho_D` fresh at each displaced geometry" once this term landed, framing
that as "the true unrestricted moving-grid FD". That is NOT what Eq. 33's
Term 1 is, and building it would have reproduced N3.5.7.1's very first
mistake (the full three-term derivative of `Tr[P.f(rho_P).rho_D]`) in a new
disguise. What actually changed in the gate: `term1_at_current_geometry`
and `term1_gga_at_current_geometry` still freeze the fxc-kernel-derived
`coeff` (LDA) / `{c0,c1,c2,c3}` (GGA) arrays at their base-geometry VALUES,
exactly as N3.5.7.1/.2 established -- but the per-point WEIGHT is no longer
baked into those frozen arrays; it is now read fresh from the displaced
geometry's own grid (`grid->points(p,3)`) at every evaluation, since a
genuinely moving weight is precisely this term's FD signature. Verified:
analytic and FD agree to **ratio 1.000000** (6 significant figures) for all
8 cases (LDA x2 seeds, GGA x2 seeds, x2 step sizes) in
`tests/dft_kernel_gradient_fd.cpp`; all 13 `planck-dft-*` CTest targets and
the four gradient regression baselines
(`water_rmp2_gradient_{smoke,fd}`, `water_dft_hse06_gradient_symm_ultrafine_fd`,
`water_rks_b3lyp_soscf_631g`) still pass, confirming the
`becke_partition_owner_derivatives` export changed nothing about the
existing production gradient path.

`compute_xc_kernel_nuclear_gradient` now computes the FULL Term 1 of Eq. 33
(LDA + GGA + point-translation + Becke-weight-derivative). N3.5.7.4 is next.

#### N3.5.7.4 -- wire into the driver -- REVERTED. `compute_xc_kernel_nuclear_gradient`'s whole architecture is wrong; see N3.5.7.5

Wired as planned: `DFT::Gradient::compute_xc_kernel_nuclear_gradient(mol,
shells, grid, ao_grid, hess, *xc_grid, calculator._info._scf.alpha.density,
rd->dm1_corr_relaxed_ao, x_functional, c_functional)`, unconditional, result
added to `calculator._gradient` alongside `*corr` in `compute_analytic_ks_gradient`.

**Made the end-to-end FD residual WORSE, not better: 1.877e-4 -> 2.666e-4.**
Flipping the sign (`-= *xc_kernel_grad` instead of `+=`) made it worse again,
**3.214e-4** -- ruling out a simple sign-convention bug (the doc's own "sign
convention" section did not save this; the mismatch is structural, not a
missed minus sign). Reverted to inert (comment placeholder restored,
`dft_kernel_gradient.h` include removed from `driver.cpp`); production stays
at the 1.877e-4 baseline. `compute_xc_kernel_nuclear_gradient` and its own
FD gate (`tests/dft_kernel_gradient_fd.cpp`) both still stand and still
pass -- they correctly compute what they were built to compute; that
quantity is just not Eq. 33's Term 1, per N3.5.7.5's finding.

#### N3.5.7.5 -- root cause: `zeta(P)^(x)` is not an AO-pair derivative -- N3.5.7.1-3's whole architecture needs a rewrite

Re-derived directly from the primary source (Neese, Schwabe, Grimme, *J.
Chem. Phys.* **126**, 124115 (2007), full text obtained and read -- not the
secondary transcription this doc's own N3.5.7 preamble carried) rather than
continuing to guess-and-FD against the wrong quantity. The paper's Eq. (33)
(general spin-unresolved form, before the closed-shell collapse) is:

```
E_PT2^x = <Dh^x> + <W^PT2 S^(x)> + Sum_munukt Gamma_munukt^PT2 (munu|kt)^(x)
   + Sum_sigma [
       INT Sum_zeta zeta(P)^(x) { f^{rho_s rho_s zeta} rho_D^s
             + (2 f^{gamma_ss zeta} grad_rho_P^s + f^{gamma_ss' zeta} grad_rho_P^s') . grad_rho_D^s } dr
     + INT { 2 f^{gamma_ss} (grad_rho_P^s(x) . grad_rho_D^s)
             + f^{gamma_ss'} (grad_rho_P^s'(x) . grad_rho_D^s) } dr
   ]
```

**The critical misreading, found by cross-referencing the paper's own
definition of `zeta(D')`** (used a page earlier, in the AO-basis response
operator `R^alpha(D')_munu`, structurally the same kind of object): `zeta`
ranges over `{rho_D', gamma_alphaalpha(D'), gamma_betabeta(D'),
gamma_alphabeta(D')}` -- i.e. **`zeta(P)^(x)` is the geometry derivative of
the SCF density's own scalar/vector quantities `rho_P` and `grad_rho_P`
(via `gamma(P) = grad_rho_P . grad_rho_P`), not a derivative of an AO-pair
basis-function product `phi_mu*phi_nu`.** N3.5.7.1-3 built
`compute_xc_kernel_nuclear_gradient` around the latter (mirroring
`compute_xc_nuclear_gradient_rks`'s `vtmp`/`vxc1`/`owner_atom`
AO-pair-scatter machinery, contracted against `P`) -- structurally the wrong
operation, which is why the FD gates for N3.5.7.1-3 passed (they test the
routine against itself, correctly) while wiring the result into the real
gradient made the residual worse in both signs.

**The right building blocks already exist in the tree, unused.**
`dft_gradient.cpp`'s anonymous namespace defines `drho_channel` (= `d(rho_P(r))/dR_{atom,q}`)
and `dg_axis_spin` (= `d(grad_rho_P_axis(r))/dR_{atom,q}`), both taking
exactly `(P_sym, ao, hess, grid_point, atoms_bf)` and returning the scalar
per-point nuclear derivative of the density and its gradient respectively
-- **precisely what `zeta(P)^(x)` needs**. Neither has a caller anywhere in
`src/` (`grep -n "drho_channel\|dg_axis_spin" src/dft/*.cpp src/dft/*.h`
shows only their own definitions) -- they read as scaffolding built for
exactly this term and never wired up. `zeta(P)^(x)` reduces to `Sum_p w(p) *
d(rho_P)/dR * (fixed rho_D-and-fxc-weighted coefficient)` plus a second
piece weighted by `d(grad_rho_P)/dR` via `gamma(P)^(x) = 2 grad_rho_P .
d(grad_rho_P)/dR` -- a **direct scalar grid integral**, no AO-pair matrix
and no `vtmp`/`vxc1`/`atoms_bf` scatter at all. `becke_partition_owner_derivatives`
(exported in N3.5.7.3) is still needed for the point-translation and
weight-derivative pieces of `d(rho_P)/dR` itself, since `rho_P(r)` moves
both because the AO basis moves and because the grid point moves/reweights
-- same three-piece structure (basis-derivative, point-translation,
weight-derivative) as `compute_xc_nuclear_gradient_rks`'s own `d(rho)/dR`,
just consumed differently downstream.

#### N3.5.7.6 -- both open questions resolved against the paper PDF

The full text (JCP 126, 124115, obtained 2026) settles both N3.5.7.5 open
questions.

**1. The third functional derivative is REAL, not a parsing error.** The
paper's Eq. (23) and the text below it define `zeta(D')` to range over
`{rho_D'^alpha, rho_D'^beta, gamma_alphaalpha(D'), gamma_betabeta(D'),
gamma_alphabeta(D')}`, and Eq. (24) shows `gamma_alphaalpha(D') = 2
grad_rho_D'^alpha . grad_rho_P^alpha` -- a MIXED invariant, one factor from
`D'` and one from `P`. In Eq. (33)'s Term 1 the sum `Sum_zeta zeta(P)^(x)`
runs over the P-analog set `{rho_P^s, gamma_ss(P), gamma_ss'(P)}` (with
`gamma_ss(P)^(x) = 2 grad_rho_P^s . grad_rho_P^s(x)`), and `f^{... zeta}`
means ONE MORE functional derivative w.r.t. that invariant. So Term 1
genuinely needs `f^{rho rho rho}`, `f^{rho rho gamma}`, `f^{rho gamma
gamma}`, `f^{gamma gamma gamma}` -- libxc `v3rho3`, `v3rho2sigma`,
`v3rhosigma2`, `v3sigma3`. This is structurally forced: Term 1 is `d/dR` of
the response operator `R(D')` (Eq. 41), which already carries `f^{(2)}`;
differentiating `f^{(2)}` w.r.t. geometry pulls down `f^{(3)} x (drho_P/dR
or dgamma_P/dR)`. The paper builds on `f^{(2)}` "everywhere else" only
because everywhere else is not differentiating the kernel.

**2. The second `INT { ... }` line IS a separate additive integral, and it
is `f^{(2)}` only.** It reads `INT { 2 f^{gamma_ss}(grad_rho_P^s(x) .
grad_rho_D^s) + f^{gamma_ss'}(grad_rho_P^s'(x) . grad_rho_D^s) } dr` --
`v2sigma`-family (NOT third derivative), contracting `d(grad_rho_P)/dR`
against `grad_rho_D`. It is the piece where the `grad` in `R^XC`'s
`grad(phi_mu phi_nu)` gets its geometry derivative while `f^{(2)}` stays
put. GGA-only: vanishes identically for an LDA functional. It does NOT fold
into Term 1's `gamma(P)` branch (that one carries `f^{(3)}` and multiplies
`gamma_P^(x)`; this one carries `f^{(2)}` and multiplies `grad_rho_P^(x)`
directly).

**Closed-shell collapse.** For RKS `rho_P^alpha = rho_P^beta = rho_P/2`,
`grad_rho_D^alpha = grad_rho_D/2`, etc. Collapsing `Sum_sigma` and the
`sigma' != sigma` cross terms (the same way `R^XC(D') = R^alpha + R^beta`
collapses in Eq. 41) gives, per grid point, fixed scalar coefficients built
from libxc `v3*`/`v2sigma*` at the GROUND density times relaxed-density
(`D`) scalars/vectors, multiplying `drho_P/dR` (Term 1 LDA + part of GGA)
and `d(grad_rho_P)/dR` (Term 1 GGA `gamma` branch, and all of Term 2). The
exact closed-shell spin prefactors must be derived explicitly and
FD-verified -- N3.5.7.2's `~1.8x` lesson (a small-rational-number ratio =
a missing constant factor) applies directly.

Both are DIRECT per-point scalar grid integrals over `drho_channel` /
`dg_axis_spin` (`dft_gradient.cpp:139,173`, no callers, built for exactly
this). No AO-pair scatter. `becke_partition_owner_derivatives` (exported in
N3.5.7.3) supplies the point-translation and weight-derivative pieces of
`drho_P/dR` itself.

#### N3.5.7.6a -- CORRECTION: it is SECOND derivative, not third; and no moving-grid pieces

N3.5.7.6's "third functional derivative is REAL" reading is **wrong**, and
was caught by the S2/S3 FD gate (a genuine `d/dR` of the right functional,
vs the routine). Two corrections:

1. **The paper's own text (p.6) says SECOND:** "a response-type term arises
   which requires the evaluation of the **second functional derivative** of
   the XC functional." Eq. 33's XC term is `d/dR` of
   `Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>`, where `<mu|V_xc|nu>`
   (Eq. 10) carries the FIRST derivatives `df/drho`, `df/dgamma`.
   Differentiating those via the SCF density's non-stationary
   `rho_P^(x)` / `grad_rho_P^(x)` pulls down `f^{(2)}` (v2rho2,
   v2rhosigma, v2sigma2, vsigma) -- NOT `f^{(3)}`. The `f^{...zeta}`
   notation in Eq. 33 is `d^2 f / d rho_sigma d zeta`, structurally the
   same object Eq. 41's response operator uses. **S1's kxc wrapper is not
   needed for this term** (kept as a correct utility).

2. **Eq. 33 as WRITTEN has no moving-grid (Becke-weight /
   point-translation) correction.** `rho_P^(x)` (Eq. 15) is the
   basis-function derivative at a FIXED spatial point -- it is the
   integrand, not `d/dR` of a quadrature. So Eq. 33's Term1+Term2 is a
   plain `sum_p w_p * integrand_p` (= XC_II in the S5 breakdown), and it
   is NOT translationally invariant on its own; `sum_A grad_A = 0` holds
   only for the full `E_PT2^x`.
   **NB (S5):** the *scalar* `Phi_XC = sum D_munu <mu|V_xc[rho_P]|nu>` IS
   a grid integral, so its own `d/dR` (XC_I + XC_II + XC_III, S5-validated
   to rel 3e-9) does carry the moving-grid terms. Eq. 33's XC term is only
   the XC_II piece of that -- see S5 for why the other two do not belong.

Closed-shell, total-density convention (matching
`compute_analytic_xc_hessian_vector_product`'s GGA branch):

    LDA:  integrand(A,q) = rho_P^(x) * v2rho2 * rho_D
    GGA:  integrand(A,q) =
              [ v2rho2*rx + 2*v2rhosigma*(g.gx) ] * rho_D
            + 2*[ v2rhosigma*rx + 2*v2sigma2*(g.gx) ] * (g.grad_rho_D)
            + 2*vsigma * (gx . grad_rho_D)
    rx = rho_P^(x) (drho_channel), gx = grad_rho_P^(x) (dg_axis_spin),
    g = grad_rho_P (SCF gradient, frozen).

**`dg_axis_spin` had a real bug, found via a weighted-FD probe** (its
unweighted L1 norm matched FD to 2e-6, hiding a ~0.4-1.3% sign-correlated
error). The four product-rule terms of `d(grad_rho_P|_ag)/dR_{A,q}`
collapse (by P symmetry) to `2 * sum_{mu in A, all nu} P(mu,nu) *
[ -h_{ag,q}(mu) phi(nu) - g_q(mu) g_ag(nu) ]` -- the cross term is
`g_q(mu)*g_ag(nu)` (the atom-A function carries the q-derivative), and the
old code had `g_ag(mu)*g_q(nu)` plus a missing `sum_{mu,nu in A}` term.
Fixed and simplified to the single loop above; weighted-FD residual
`5e-8` (was ~1%). No prior callers, so behaviour-neutral for the tree.

### N3.5.7.7 -- the rewrite (S0-S5)

Supersedes N3.5.7.1-4. The old `compute_xc_kernel_nuclear_gradient` and
`tests/dft_kernel_gradient_fd.cpp` are the AO-pair-scatter architecture
N3.5.7.5/.6 disproved.

**S0 -- delete the dead architecture. LANDED.**
`src/dft/dft_kernel_gradient.{cpp,h}` (371 lines) and
`tests/dft_kernel_gradient_fd.cpp` (614 lines) removed; `CMakeLists.txt`
`planck-dft-kernel-gradient-fd` target and its four wiring blocks removed;
the driver placeholder comment rewritten to point at the S1-S4 plan. The
N3.5.7.3 `becke_partition_owner_derivatives` export in `dft_gradient.h`
stays -- the rewrite needs it.

**S1 -- libxc third-derivative wrapper. LANDED.**
`evaluate_lda_kxc` (`xc_lda_kxc` -> `v3rho3`) and `evaluate_gga_kxc`
(`xc_gga_kxc` -> `v3rho3`, `v3rho2sigma`, `v3rhosigma2`, `v3sigma3`) added
to `src/dft/base/wrapper.h`, mirroring `evaluate_lda_fxc` /
`evaluate_gga_fxc` one-for-one (new `v3*_components()` helpers,
npoints/size guards, chunked/threaded loop, pointwise-map). Gate:
`tests/dft_kxc_selfcheck.cpp` (`planck-dft-kxc-selfcheck`) -- `v3rho3`,
`v3rho2sigma`, `v3rhosigma2`, `v3sigma3` each vs a central-difference of
the corresponding libxc `fxc` block, three step sizes, `O(h^2)` tol, plus
mixed-partial cross-checks and family guards.

**Two libxc findings, both real:**
1. **`DISABLE_KXC` defaults ON in libxc.** The vendored ExternalProject
   built with NO third derivatives -- every functional has
   `XC_FLAGS_HAVE_KXC` unset, and `lda_x`/`pbe`/`b88`/`lyp`/`b3lyp` all
   report `kxc=0`. Fixed with `-DDISABLE_KXC:BOOL=OFF` in the libxc
   `CMAKE_ARGS` (`CMakeLists.txt`); additive, the vxc/fxc paths every
   existing DFT regression validated are byte-identical. Forces a one-time
   libxc rebuild (`rm -rf src/external/libxc/{src/libxc-build,install,src/libxc-stamp}`
   then reconfigure).
2. **Requesting v3 from a kxc-less functional is `exit(1)` inside libxc**,
   not an error return (`lda.c:41`, `gga.c:53`). So `evaluate_*_kxc` guard
   on a new `has_kxc()` predicate (`func_.info->flags & XC_FLAGS_HAVE_KXC`)
   and return `std::unexpected` -- the family check alone is not enough.

**S2 + S3 + S5 -- `compute_dh_xc_pt2_gradient`, the full geometry
derivative of `Phi_XC`. LANDED as a routine + FD gate; NOT wired in.**
`DFT::Gradient::compute_dh_xc_pt2_gradient` in a fresh
`dft_kernel_gradient.{cpp,h}` (filenames reused, contents new). Args: mol,
basis, grid, ao, ao Hessian, ground density `P`, relaxed density `D`, both
functionals.

**The quantity (Python/PySCF-derived and validated, S5).** The scalar
functional whose geometry derivative is Eq. 33's XC term is

    Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>
           = integral w * { vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D) } dr

(`<mu|V_xc|nu>` = the SCF operator's XC part, Eq. 10, FIRST functional
derivatives; `D` = relaxed PT2 difference density). Since `Phi_XC` is
itself a grid integral, `d/dR` is a TRUE geometry derivative with THREE
pieces -- verified in Python against a rebuild-the-molecule central
difference to **rel 3e-9** (He2/STO-3G PBE):

- **XC_I** -- basis-function derivative of `rho_D` (`drho_channel(D)`,
  `dg_axis_spin(D)`) against the FIRST XC derivatives:
  `sum_p w_p [ vrho*rho_D^(x) + 2*vsigma*(grad_rho_P . grad_rho_D^(x)) ]`.
  Dominant (~85% of `d/dR{Phi_XC}`).
- **XC_II** -- `rho_P` inside `V_xc[rho_P]` responds
  (`drho_channel(P)`, `dg_axis_spin(P)`) against the SECOND XC
  derivatives:
  `sum_p w_p [ (v2rho2*rx + 2*v2rhosigma*(g.gx))*rho_D
             + 2*(v2rhosigma*rx + 2*v2sigma2*(g.gx))*(g.grad_rho_D)
             + 2*vsigma*(gx.grad_rho_D) ]`
  (rx = drho_channel(P), gx_a = dg_axis_spin(P), g = grad_rho_P). ~15%.
- **XC_III** -- grid quadrature moving frame:
  `sum_p [dw_p/dR]*I_p` (Becke partition weight, via
  `becke_partition_owner_derivatives`) `+ sum_{owner(p)=A} w_p*[dI_p/dr_q]`
  (point translation, with `d(.)/dr_q = -sum_A(channel helper)`), where
  `I_p = vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D)`. ~0.1%, but
  load-bearing for exact translational invariance.

The routine computes all three, LDA (v2sigma-family = 0) and GGA, at the
GROUND density, `rho_P >= 1e-8` screened, combined-XC guard on the `_c`
arrays.

**FD gate:** `tests/dft_kernel_gradient_fd.cpp`
(`planck-dft-kernel-gradient-fd`) -- He2/STO-3G, Normal grid, strongly
diagonally-dominant random symmetric `P`/`D`. Reference: rebuild the
molecule (grid + basis) at each `+-h` displaced geometry and central-
difference `Phi_XC` -- a genuine geometry derivative (basis moves, grid
points move with owner, weights move). `sum_A grad_A = 0` is a valid
check now that XC_III is included. **LDA rel 3e-4 -> 7e-5, GGA rel 2e-5 ->
6e-6** (h = 1e-3 -> 5e-4, O(h^2)). 6 cases (LDA/GGA x 2 seeds x 2 steps).

**Exports from `dft_gradient.{h,cpp}` (was file-local anon namespace):**
`drho_channel`, `dg_axis_spin`, `atom_bf_lists` -- real shared algorithm.
`dg_axis_spin` was **fixed** in this arc (N3.5.7.6a -- cross-term index
swap; a weighted-FD probe caught it where an unweighted L1 norm matched
to 2e-6). Behavior-neutral for `compute_xc_nuclear_gradient_{rks,uks}`.

**S5 -- wire into the driver + measure. TRIED, REVERTED (3rd wall in
this arc, after N3.5.7.4 and S4).** The routine is FD-verified as
`d/dR{Phi_XC}` to rel 3e-9, but wiring it into
`compute_analytic_ks_gradient` STILL does not close the ~1.9e-4 Ha/Bohr
water/STO-3G B2PLYP FD residual. Component-resolved probes:

| config | water/STO-3G B2PLYP FD residual |
|---|---|
| baseline (nothing added) | `1.881e-4` (atom 2, y) |
| full `XC_I + XC_II + XC_III` | `3.517e-3` (atom 1, z) |
| `XC_I` only | `3.678e-3` |
| `XC_I + XC_II` (no XC_III) | `3.517e-3` |
| `XC_II` only | `2.650e-4` |
| `XC_II * 0.5` | `1.792e-4` |
| `XC_II * c_pt2` (0.27) | `~1.40e-4` (best; the `0.27` coincidence is suspicious) |

**Conclusions from the probes:**

1. **XC_I is NOT in the answer.** Adding it (~3.5e-3, ~85% of the routine)
   blows the gradient up. The paper's "not the naive `D^(x)`" (p.6) is
   *literal* -- Eq. 33 has no `rho_D^(x)` factor anywhere. XC_I is a real
   piece of `d/dR{Phi_XC}` but `Phi_XC` is the wrong scalar to fully
   differentiate.
2. **XC_III is negligible end-to-end** (Planck's grid, like the synthetic
   one, has moving-frame terms that cancel to ~1e-7).
3. **Eq. 33's Term1+Term2 IS XC_II** -- basis-only `rho_P^(x)` (Eq. 15),
   no moving grid, no `rho_D^(x)`. XC_II is FD-verified as
   `sum_munu D_munu R^XC[rho_P^(x)]_munu` with the SOSCF-validated
   total-density `v2rho2`, yet it **overshoots the baseline residual ~2x**
   (2.65e-4 vs 1.88e-4 needed), with a per-component sign structure -- the
   same shape N3.5.7.4 and S4 hit.

**Sec. II derivation done (2026) -- XC_II is EXACTLY Eq. 33 Term1+Term2.**
Worked the spin-resolved form and its closed-shell collapse, then
numerically validated against
`compute_analytic_xc_hessian_vector_product_polarized` (the SOSCF-validated
polarized `R^XC` operator):

> Eq.33_XC = Sum_sigma Sum_munu D^sigma_munu . R^XC_polarized[trial = rho_P^(x)]^sigma_munu

i.e. the polarized response operator applied to the SCF density's
basis-function derivative `rho^(x)` (per spin `rho_P^(x)/2`), contracted
with `D^sigma`. Closed-shell collapse (`rho_alpha = rho_beta = rho_P/2`,
`D^alpha = D^beta = D/2`, `Sum_sigma -> x2`):
`Sum_sigma D^sigma . R^XC_pol[rho_P^(x)/2]^sigma == D . R^XC_unpol[rho_P^(x)]`.
**Verified to machine precision (diff 1e-17) for both a linear (He2) and a
bent 3-atom (water-like) geometry with all components non-zero.** So:

- **There is NO closed-shell spin-factor bug.** The `v2rho2_ab/2`
  candidate above is DEAD -- the LDA collapse gives exactly
  `int v2rho2_unpol * rho_P^(x) * rho_D dr` (because
  `v2rho2_unpol = 1/2 (v2rho2_aa + v2rho2_ab)` and the spin sum + halved
  densities reconstruct it), and the GGA collapse matches the polarized
  form term-for-term.
- **The `ks_veff` for `vhf_s1occ` (N3.5.5) is inert now**, not harmful:
  re-tested with the CURRENT `ks_veff` closure (linear `f_xc` HVP, not
  N3.5.5's nonlinear `get_veff`), the `vhf_s1occ` KS-vs-HF swap changes
  the water/STO-3G B2PLYP FD residual by 0 (projected `R^XC(relaxed)` is
  ~0 there). N3.5.5's "made it worse" does not reproduce.

**So the ~2x end-to-end overshoot is NOT in the XC term** -- XC_II is
provably Eq. 33's XC contribution. It is in **how Planck assembles
`E_PT2^x` around it**: `build_rmp2_gradient_intermediates` is the *HF-MP2*
gradient path re-run on KS orbitals, and its `<D h^x>`, `<W^PT2 S^x>`,
`Sum Gamma (munu|kt)^x` terms are the HF-MP2 versions, not the paper's
DH-specific Eqs. 22-33. Candidates for the mismatch (all in the
already-committed DH gradient path, not this routine):
- **`W^PT2`'s `-1/2 R(D)_ij` piece (Eq. 42).** Planck's `vhf_s1occ` uses
  HF `J - 1/2 K`; the paper wants `R(D)` = the full response operator
  INCLUDING `R^XC(D)`. Inert for water (see above) but may bite on a
  system with real `rho_D` gradient structure.
- **`Gamma^PT2` (Eq. 46) vs Planck's `dm1p` / `dm2buf`.** The separable
  2e density the `vhf1` derivative kernel contracts is `hf_dm1 + 2*D`;
  Eq. 46's `Gamma^{SCF+PT2}` is `1/2 P P - 1/4 P P + D P - 1/2 D P +
  Gamma^NS`. Needs a term-by-term check that the DH path reproduces it
  exactly (it was validated for HF-MP2, where there is no XC operator).
- **`<D h^x>` -- `h` in the DH context.** The KS `h_core` is the same
  kinetic + nuclear as HF, so this is likely fine, but the `x` derivative
  of the KS one-electron operator has no XC part by construction, so any
  XC that "should" be in `<D h^x>` is not there.

**Term-by-term audit (started 2026).** Mapping `electronic` in
`build_rmp2_gradient_intermediates` to Eq. 33's closed-shell form:

| Planck accumulator | Eq. 33 term | density used |
|---|---|---|
| `two_e_terms` = `sum dI * 2*dm2buf_full` | nonseparable `Gamma^NS` (Eq. 35/47) | `t2 -> AO`, KS orbitals |
| `one_e_terms` on `dm1_total_ao` | `<D h^x>` | `hf_dm1 + 1*D'_relaxed` |
| `s_im1 + s_zeta + s_vhf` | `<W^PT2 S^x>` (Eqs. 42-45) | `imat`, `zeta`, `vhf_s1occ` |
| `vhf1_terms` = `sum vhf1[atom][q] .* dm1p` | separable `Gamma` (Eq. 46: `1/2 PP - 1/4 PP + DP - 1/2 DP`) | `dm1p = hf_dm1 + 2*D'_relaxed`, HF `2J-K` full-weight kernel |
| `vhf1_rs/rq/pq/ps` | same separable `Gamma`, other permutations | `hf_dm1 (x) dm1p` |

The full-HF-exchange-weight in `vhf1` (not `a_x`) is CORRECT per Eq. 46
(the paper is explicit: `a_x` lives in `E_SCF` and `R^XC`, not the
separable PT2 `Gamma`). So `Gamma^PT2` looks structurally right.

**Per-component probe (water/STO-3G B2PLYP, wiring XC_II only, unrelaxed
`D` = `lag->dm1_corr_ao`), analytic - FD:**

| component | baseline (no XC) | + XC_II | XC_II contributed | needed |
|---|---|---|---|---|
| atom1 (O) z | `-1.87e-4` | `-1.89e-4` | `~0` | `+1.87e-4` |
| atom2 (H) y | `+1.88e-4` | `+4.0e-5`  | `-1.48e-4` | `-1.88e-4` |
| atom2 (H) z | `+9.3e-5`  | `+2.02e-4` | `+1.09e-4` | `-9.3e-5` |

XC_II **fixes atom2-y** (the baseline's max residual) almost perfectly,
does **nothing** for atom1-z, and gets atom2-z the **wrong sign**. So the
missing term is XC_II (which XC_II supplies) **plus a z-directional term**
that pushes both z-forces negative -- one XC_II does not carry. (Relaxed
`D` gives a worse atom2-z; the `z_ov` block of the relaxed density is not
helping there.)

**The z-directional term is the open question.** Candidates, in order of
suspicion:
- **XC_III (moving grid), which is NOT negligible for a bent molecule.**
  The synthetic He2 test put XC_III at ~1e-7, but water's grid has real
  geometry dependence on the O-centred block -- a Becke-weight /
  point-translation term of O(1e-4) on the O z-force is plausible and
  matches the atom1-z signature (a term XC_II structurally cannot carry).
  The routine already computes XC_III; a clean wired probe (XC_II + XC_III,
  no XC_I) is the immediate next step -- the env-gated probes for this
  kept mis-firing.
- **`<W^PT2 S^x>`'s `-1/2 R^XC(D)` piece.** Inert for water at the SCF
  level (projected `R^XC(relaxed)` ~ 0), but its `S^x` contraction is a
  different object and was not separately checked.
- **A sign or factor in how `*corr` isolates the PT2 part** (the
  `full - ref_grad` subtraction with a zeroed Lagrangian) in the DH
  context -- the `ref_grad` was built for HF-MP2.

**The XC term (this routine, XC_II) is settled** -- Sec. II derivation +
polarized `R^XC` cross-check to machine precision. Everything unresolved
is in the surrounding DH-gradient assembly.

**The routine and its FD gate stay committed.**

#### N3.5.7.8 -- component probe wired; XC_III and the point-translation companion are both RULED OUT

The doc's own "immediate next step" (a clean wired probe of `XC_II +
XC_III`, no `XC_I`) is now built and run. The earlier env-gated probes
"kept mis-firing" because they were on/off flags that could not say WHICH
piece they added; this one takes an explicit **bitmask**:

- `compute_dh_xc_pt2_gradient` gained a trailing `unsigned parts = kXcAll`
  (`kXcI=1, kXcII=2, kXcIII=4, kXcIIt=8`). Default is the same full
  `XC_I+II+III` the FD gate verifies, so `planck-dft-kernel-gradient-fd`
  is unaffected and still passes.
- The driver's reverted comment block became a probe hook read from
  `PLANCK_DFT_DH_XC_PARTS` (bitmask) and `PLANCK_DFT_DH_XC_SCALE`
  (prefactor), logging what it added. **Unset = production unchanged**;
  `water_dft_hse06_gradient_symm_ultrafine_fd`, `water_rks_b3lyp_soscf_631g`,
  `h2_dft_b2plyp_sto3g`, `water_rmp2_gradient_{fd,smoke}` all pass.

**Fixture.** The doc's water/STO-3G B2PLYP case was ad hoc and is not in
the tree; rebuilt at `use_symm .false.`, `grid ultrafine`, `tol 1e-10`,
the standard `water_dft_hse06_gradient` geometry (O at origin, H's in the
xz plane). Baseline residual **2.424e-4 Ha/Bohr (atom 2, x)** -- the same
phenomenon as the doc's 1.881e-4 (atom 2, y), rotated: that run's water
was not in the xz plane, so its "y" is this run's "x". Per-component
signature is identical in shape. **Now committed** (it was ad hoc, and
re-deriving it cost a pass) at
`tests/inputs/exploratory/dh_gradient/water_b2plyp_gradient_fd.hfinp` --
exploratory, deliberately NOT a registered regression case (the FD driver
run is minutes). Reproduce with:

```
PLANCK_DFT_DH_GRADIENT=1 PLANCK_DFT_DH_XC_PARTS=<mask> \
  python3 tests/dft_gradient_fd.py \
    tests/inputs/exploratory/dh_gradient/water_b2plyp_gradient_fd.hfinp \
    --build-dir ./build --delta 1e-3 --atol 1e-2
```

| config | max residual | where |
|---|---|---|
| baseline | `2.424e-4` | a2-x |
| `XC_II` (parts=2) | `2.057e-4` | a2-z |
| `XC_III` (parts=4) | `2.424e-4` | a2-x |
| `XC_II+XC_III` (parts=6) | `2.057e-4` | a2-z |
| `XC_II+XC_IIt` (parts=10) | `2.164e-4` | a1-z |

**1. XC_III is NOT the z-directional term -- the doc's leading candidate
is dead.** The suspicion was that the synthetic He2 test understated
XC_III and that "water's grid has real geometry dependence on the
O-centred block". It does not: `parts=4` reproduces the baseline gradient
to **7 significant figures** on every component (a1-z `0.18961646` vs
`0.18961656`), and `parts=6` is `parts=2` to the same precision. XC_III
is ~1e-7 on real water exactly as on synthetic He2.

**2. The real defect XC_II has: it is not translationally invariant, and
that non-invariance is the same size as the residual it fails to close.**
`sum_A grad_A` is exactly `0` for the baseline analytic gradient and
`-1.0e-7` for the FD reference, but adding XC_II makes it **`-2.445e-4`
in z** -- i.e. XC_II injects a spurious net force of the same magnitude
as the whole problem. This was predicted structurally by N3.5.7.6a
("Eq. 33 as WRITTEN ... is NOT translationally invariant on its own")
but had never been measured end-to-end.

**3. Restoring invariance the obvious way does NOT fix it -- `kXcIIt`
tried and rejected.** Added XC_II's point-translation companion: the same
XC_II integrand scattered onto the owner atom with
`d/dr_q = -sum_A(channel)`, exactly as XC_III does for `I_p`. It works as
designed -- `sum_A grad_A` returns to `1.0e-8` -- but the residual gets
**worse** (2.057e-4 -> 2.164e-4), because the redistribution undoes the
one thing XC_II got right:

| component | baseline - fd | + XC_II | + XC_II + XC_IIt |
|---|---|---|---|
| a1-z | `1.404e-4` | `1.669e-4` | `2.164e-4` |
| a2-x | `2.424e-4` | **`2.44e-5`** | `1.717e-4` |
| a2-z | `-7.013e-5` | `-2.056e-4` | `-1.082e-4` |

So XC_II carries the **a2-x** physics almost exactly (contributes
`-2.181e-4` against a needed `-2.424e-4`, ratio 1.112) and the missing
piece is genuinely a *different* term that supplies the z structure --
not a redistribution of XC_II's own weight. `kXcIIt` is kept in the enum
(off by default, not in `kXcAll`) as the recorded negative result; delete
it if the eventual fix makes it meaningless.

**4. The `c_pt2 = 0.27` "best fit" is confirmed coincidence.** The needed
/ supplied ratios per component are `-5.29` (a1-z), `1.11` (a2-x),
`-0.52` (a2-z) -- no single scale factor exists, so the earlier `~1.4e-4`
at `XC_II * 0.27` was a max-norm artifact of two components crossing, not
a missing constant. The N3.5.7.2 "small rational number = missing factor"
heuristic does **not** apply here.

#### N3.5.7.9 -- the `full - ref_grad` isolation is EXONERATED, and the missing term is now known EXACTLY

The third and last candidate from the term-by-term audit -- "how `*corr`
isolates the PT2 part" -- is measured and clean. Two instruments:

**(a) Per-term dump.** `PLANCK_DEBUG_RMP2_TERMS=1` (already in
`mp2_gradient.cpp`, no new code) prints all 14 accumulators for BOTH the
zero-Lagrangian reference call and the full call, so the per-term PT2
contribution is a direct subtraction. Every term is translationally
invariant to **machine precision** (`sum_A grad_A` rel 1e-14 or better:
`two_e` 8.9e-15, `h1` 1.6e-14, `s_im1` 0.0, `s_zeta` 1.1e-14, `s_vhf`
5.4e-13, `vhf1` 8.8e-15, `electronic` 4.4e-14). The four `vhf1_{rs,rq,pq,ps}`
rows are NOT invariant individually but cancel exactly in pairs
(`rs+pq = 0`, `rq+ps = 0`) -- they are a decomposition of `vhf1`, not
separate contributions. Nothing in the assembly leaks a net force.

**(b) An exact FD reference for `*corr` itself.** Built in PySCF 2.13.0 by
finite-differencing `E(R) = E_KS_hyb(R) + 0.27 * E_MP2-on-KS(R)` --
the same total-energy function Planck's own FD driver differentiates, so
its `0.27 * dE_corr/dR` is precisely what `*corr` must equal. Cross-check
first: **PySCF `E_corr = -0.0384041028`, Planck `-0.0384040654`** (3.7e-8),
and the KS parts agree too, so the two codes share orbitals and
correlation energy before any gradient is compared.

| | a1-z | a2-x | a2-z | max err |
|---|---|---|---|---|
| FD truth (`0.27 dE_corr/dR`) | `9.8812e-3` | `-5.2017e-3` | `-4.9406e-3` | -- |
| Planck `*corr` | `1.0020e-2` | `-4.9600e-3` | `-5.0110e-3` | **`2.42e-4`** |
| PySCF `grad/mp2.py` on KS orbitals | `4.5858e-3` | `-3.7951e-3` | `-2.2929e-3` | `5.30e-3` |

**Planck's PT2 assembly is right to 2.4e-4; the PySCF harness is 22x
worse.** `max|Planck total - FD total| = 2.383e-4` equals
`max|Planck *corr - FD *corr| = 2.417e-4`, so the ENTIRE end-to-end
residual lives in `*corr` and is exactly this size -- nothing is hiding
in the KS part or in the subtraction.

**PySCF's `grad/mp2.py` is NOT a usable reference here, on two counts, and
this kills the N3.5.5 line of reasoning permanently.** Its `fvind`
(`grad/mp2.py:277`) calls `mp._scf.get_veff(mol, dm + dm.T)` -- the full
**nonlinear** KS `get_veff` on a small non-idempotent trial density, not
the linear `gen_response`. Built densely, that CPHF matrix has
`cond = 1.6e18` (numerically singular; the stock Krylov solver raises
`Krylov solver failed to converge`, which is how this surfaced), and the
gradient it produces is garbage (~3.5 Ha/Bohr). With the linear response
substituted, `cond = 37` and the eigenvalues are sane (min 0.53). All
three `get_veff` sites (lines 138 `Xvo`, 163 `vhf_s1occ`, 277 Z-vector --
the doc's N3.5.4 / N3.5.5 / N2 sites) were patched through every
combination; none reaches the FD truth (`planck` config 5.30e-3, `allhf`
7.20e-3, `allks` 5.23e-3). The driver comment at `driver.cpp:4065`
already suspected this; it is now measured.

**The missing term, exactly.** `FD - Planck`, in Ha/Bohr:

```
  Atom 1 (O):   0.00000e+00   0.00000e+00  -1.38775e-04
  Atom 2 (H):  -2.41742e-04   0.00000e+00   7.03883e-05
  Atom 3 (H):   2.41742e-04   0.00000e+00   7.03883e-05
```

**It is translationally invariant (`sum_A = 2.0e-6`) while XC_II is not
(`-2.445e-4`).** That is an independent confirmation of N3.5.7.8's
conclusion, reached from the opposite direction: **no multiple of XC_II
can be the missing term**, whatever the prefactor, because they differ in
a conserved quantity. The per-component ratios `missing/XC_II` are
`-5.23 / 1.11 / -0.52` -- the a2-x agreement at 1.11 that made XC_II look
close is coincidence, exactly as the earlier scale-factor analysis said.

**Where this leaves the arc.** All three candidates from the term-by-term
audit are now eliminated by measurement (XC_III ~1e-7; `vhf_s1occ`'s
`R^XC(D)` inert; the `full - ref_grad` isolation clean to 1e-14), and the
XC term itself is settled by derivation. The missing 2.4e-4 is a
translationally-invariant term that Eq. 33's XC contribution, as
implemented, does not supply. Two readings remain, and they are
distinguishable:

1. **The implemented XC_II is not the whole of Eq. 33's XC term** -- the
   Sec. II cross-check validated it against
   `compute_analytic_xc_hessian_vector_product_polarized`, i.e. against
   Planck's own `R^XC`, which is a consistency check, not an independent
   one. A direct FD of `sum_munu D_munu R^XC[rho_P^(x)]_munu` against a
   moving-geometry reference would separate "XC_II is correctly
   implemented" from "XC_II is the right formula".
2. **The relaxed density `D` fed to it is not the paper's `D`.** Eq. 33
   pairs the RELAXED PT2 difference density with `R^XC`; the probe used
   `rd->dm1_corr_relaxed_ao`, but the z-block convention there was never
   FD-verified independently of the rest of the gradient.

The exact target vector above is the instrument for both: any candidate
term can now be scored against it directly, per component, without
running the full FD driver.

**Reproduce:** `tests/inputs/exploratory/dh_gradient/` carries the Planck
input; the PySCF FD reference script is small enough to re-derive from
the recipe above (`E_KS_hyb + 0.27*E_MP2-on-KS`, `xc = "0.53*HF +
0.47*B88, 0.73*LYP"`, `cart=True`, `grids.level=6`, `h = 1e-3` Bohr) --
note PySCF has no `B2PLYP` alias in its vendored libxc, so the explicit
hybrid form is required.

### N3.6 -- lift the gate, add regressions

Once N3.5.7's FD passes: lift `validate_workflow_support` for double-hybrid
Gradient / GeomOpt / Frequency / GeomOptFrequency; remove
`PLANCK_DFT_DH_GRADIENT` and the dead-end flags; add PySCF-FD-anchored
regression cases (`water_b2plyp_gradient_sto3g`, `_geomopt_631gd`,
`_freq_sto3g`, plus the FD self-consistency case).

### N3.7 -- UKS

Polarized `f_xc` HVP (`compute_analytic_xc_hessian_vector_product_polarized`),
per-spin Lagrangian, `_compute_2e_k_uhf_direct`. Paper's Sec. II is the
unrestricted formalism throughout; Eqs. 22-33 are the spin-resolved
versions the closed-shell Sec. III.A reduces. Range-separated DH here too
(the `c_sr` short-range K piece in `ks_veff` and the kernel gradient).

### N3.8 -- ImaginaryFollow

Reuses the semi-numerical Hessian eigenvector; a one-line gate change + a
smoke test once Frequency works.

## Risks

1. **AO-form `f_xc`.** `compute_analytic_xc_hessian_vector_product` returns
   `delta_V_xc` in the AO basis already (not packed), so `ks_veff` uses it
   directly. Confirmed linear in the trial density, so any `d` is in
   domain.
2. **Combined-XC `f_xc` double-count.** `DFT_ANALYTIC_FXC_HESSIAN.md`
   invariant 3a: B2PLYP is a combined exchange-correlation libxc entry;
   the `f_xc` HVP must not add the correlation `f_xc` twice.
   `build_ks_orbital_hessian_op` already handles this (SOSCF B2PLYP-shaped
   cases pass); `ks_veff` calls the same function the same way. N3.5.7's
   new routine must use the same guarded `evaluate_gga_fxc` call.
3. **N3.5.7 density on the SCF-gradient side.** Eq. 33 pairs `grad rho_D`
   (relaxed) with `grad rho_P` (SCF). Get the pairing right -- one index
   of the f_xc kernel contracts `D`, the other contracts the SCF density's
   moving-grid derivative.
4. **RI.** `build_veff_from_density` under `use_ri` uses `build_ri_fock_rhf`;
   `ks_veff` has no RI analogue. DH + RI gradient is rejected up front (it
   is not RI by default). RI-DH is a separate follow-on.
