# Double-Hybrid Analytic Gradient: Handoff

**State:** in flight behind `PLANCK_DFT_DH_GRADIENT`. Production ships
double-hybrid **single points only**. The analytic gradient reproduces finite
difference to **3.887e-4 Ha/Bohr** (C1 H2O2) / **2.057e-4** (water).

**What this doc is for:** everything needed to pick the work up cold. Canonical
status lives in `vault/Status/`; the investigation record is
`docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md` and
`docs/DH_GRADIENT_XC_DERIVATION_SCOPE.md`.

---

## 1. The one-paragraph summary

The gradient is assembled from four terms (`<D h^x>`, `Gamma^PT2 (munu|kt)^x`,
the XC term, `<W^PT2 S^x>`). A first-principles derivation confirms **all four
are present and the XC term is correct**, so the residual is an **error inside an
existing term, not a missing one**. Of the four, three are verified against
independent FD-checked derivations; the **orbital Hessian feeding the Z-vector is
not**, and it carries **3.6x leverage** on the final gradient. That is the next
thing to do.

---

## 2. Where the residual is NOT (all measured, do not re-litigate)

| ruled out | evidence |
|---|---|
| the XC term | derived from scratch, all 4 channels matched (D6) |
| a missing XC term | `E_PT2` has no explicit grid dependence (D1) |
| `Gamma^PT2` exchange weight | `a_x` scaling is **16x/21x WORSE** |
| `W`'s diagonal blocks | derived independently, verified (D7) |
| the `ref_grad` subtraction | residual is **97% linear in `c_pt2`** (H2.1) |
| MP2 amplitudes | elementwise vs PySCF to **8e-9** (H2.2) |
| Z-vector *solve* | dense QR, resid **6.8e-18**, SPD (H2.3) |
| the KS-side machinery | KS-only gradient accurate to **3.66e-5**, 11x below |
| grid noise | ultrafine converged to **9.3e-5** |
| cross-code artifact | **Planck vs Planck's own FD = 3.56e-4** |
| the relaxed vs unrelaxed `D` | relaxed correct (cos 0.93 vs 0.75) |
| Eq. 42's `R^XC(D)` | 7% worse at full weight, 3% at the derived half weight |
| every `vhf_s1occ` variant | bounded: any multiple leaves >=72% of the target |

---

## 3. The single most valuable next step

**Put `build_ks_orbital_hessian_op` under an independent oracle.**

D8 verified the Z-vector's *channel decomposition* to 1.7e-11 but could not reach
the Hessian itself. Reproducing the MO channel as `L . dkappa/dR` needs `kappa`
extracted from `U = C0^T S(R0) C(R+h)`, and that is **contaminated**: `C(R+h)` is
orthonormal wrt `S(R+h)`, not `S(R0)`, so `U` mixes a genuine rotation with an
`O(dS)` metric mismatch. Measured `|U - I| = 3.7e-6` at `h = 1e-5` against
`O(h) = 1e-5` for a pure rotation. **Three extraction attempts each came up ~30x
short.**

The fix is the `U_ij = -1/2 S^(x)_ij` bookkeeping of the paper's Eqs. 19-21.

**Why this one:**
- the Z-vector moves the final gradient by **1.39e-3, 3.6x the residual**
  (measured by switching to HF-CPHF, which makes it worse)
- H2.3 established only that `A` is *well-formed* (symmetric, SPD, exactly
  solved) -- **not that it is right**; its coefficients have never been matched
  against Eq. 41 term by term
- `W`'s ov/vo blocks (D7's gap) are Z-vector-coupled, so **one model closes both**

---

## 4. Instruments (all committed, all reusable)

| instrument | what it does |
|---|---|
| `tests/pyscf/{water,h2o2,h2o2b}_b2plyp_dh_gradient_fd.py` | exact FD targets for `*corr` |
| `tests/inputs/exploratory/dh_gradient/*.hfinp` | two C1 fixtures + the water one |
| `tests/pyscf/dh_gradient_score.py` | scores a candidate; positive+negative controlled |
| `tests/pyscf/dh_xc_derivation_{model,check}.py` | D6 -- XC term, FD-exact to 2.2e-11 |
| `tests/pyscf/dh_w_derivation_{model,check}.py` | D7 -- `W`, FD-exact to 2.5e-12 |
| `tests/pyscf/dh_z_derivation_{model,check}.py` | D8 -- Z-vector, FD-exact to 1.7e-11 |
| `PLANCK_DFT_DH_XC_PARTS` / `_SCALE` | component probe on the XC routine |
| `PLANCK_DEBUG_RMP2_TERMS=1` | per-term gradient dump (14 accumulators) |
| `PLANCK_DFT_DH_ZVECTOR_HFCPHF` | swap the Z-vector operator to HF-CPHF |

**Fixture rule:** measure on a **C1** fixture. Water/C2v has only **3 independent
gradient components** -- fewer than the number of candidate terms -- so any three
candidates span the target exactly (an LSQ fit returns residual **1.8e-18** with
coefficients `-0.99 / 8.41 / 23011`). **Four attempts were reverted on
conclusions drawn there.** Score by cos in the translation-free subspace, and
require the coefficient to agree on **two** independent geometries.

---

## 5. Verified invariants the next model may rely on

**Planck's KS orbitals ARE canonical.** Checked directly rather than assumed --
`C^T F C` against the stored `eps`, on the real DH fixtures:

```
h2o2 C1  (use_symm .false.) : max|offdiag| = 3.55e-11  max|diag - eps| = 9.48e-11
water    (use_symm .false.) : max|offdiag| = 8.56e-14  max|diag - eps| = 2.81e-13
water    (use_symm .true. ) : max|offdiag| = 1.24e-13  max|diag - eps| = 4.26e-13
```

Both the plain and the **SAO-blocked** diagonalisation paths produce canonical
orbitals (`src/dft/driver.cpp:836` and `:862` -- the SAO path block-diagonalises
per irrep, and the off-diagonal blocks vanish by symmetry). This matters because
**every CC/MP2 kernel in the tree assumes it**, and D8's model is only well posed
under it.

**Corollary worth carrying:** MP2 is invariant **only under rotations within a
degenerate set**, not under arbitrary occ-occ / virt-virt rotations. "MP2 is
gauge-invariant" is false as usually stated, and a model that assumes it will be
ill-posed -- which is exactly how D8's first attempt failed.

---

## 6. Traps that cost real time

1. **A too-symmetric fixture.** See section 4. This one cost four reverted
   attempts.
2. **PySCF's `grad/mp2.py` is NOT a valid reference here.** Its `fvind` uses the
   *nonlinear* `mp._scf.get_veff` on a small non-idempotent trial density, giving
   a **singular** CPHF matrix (cond 1.6e18; the stock Krylov solver raises).
   Patched to the linear response it still lands **22x worse than Planck**.
3. **A model can be ill-posed and produce confident wrong "findings".** D8's
   first energy functional was not invariant under virt-virt rotations, so it was
   not a function of the SCF solution at all -- it produced three plausible
   failures (2.25e-1, 2.30e-1, 3.43e-2) that were pure gauge artifacts. **Check
   gauge invariance before trusting any orbital-response model.**
4. **A convention cannot be tested against a functional that never reads it.**
   D7's first run had Planck's `zeta` "failing" at 5.7e-2; the model's energy read
   only `diag(D)*eps`, so the off-diagonal weights were unconstrained. Zeroing
   them, all three conventions passed identically.
5. **Always include a scale control, not just on/off.** A `z = 0` probe reported
   "zeroing the Z-vector changes nothing" -- a spectacular-looking finding, and a
   pure artifact of an `if (false)` guard that missed the real assignment. The
   `z_mult = 10` control caught it.
6. **`XC_III` looks negligible and is not wrong.** It measures 4.6e-7 because its
   two sub-terms are individually large (`sum|point| = 1.06e-1`,
   `sum|weight| = 9.9e-3`) and **cancel 4e5-fold** -- a converged Becke grid is
   nearly translation-invariant as a quadrature. Do not "fix" it.

---

## 7. If the next step does not close it

The stop condition from `docs/DH_GRADIENT_RESIDUAL_SCOPE.md` still stands and
should be taken seriously: **the residual is 0.041 pm on a stiff X-H stretch**,
against the paper's own B2-PLYP accuracy claim of **0.3 pm MAD**. It is
0.686 pm on a torsion.

So if the orbital-Hessian audit comes back clean, run **H1.1** -- optimize the
fixtures with the DH gradient and compare against an FD-driven optimization. If
stiff coordinates agree to <0.05 pm, **ship it behind the flag with the residual
documented as a known bound** and stop hunting. A bounded, measured, non-blocking
limitation is a legitimate outcome.

The only decisive test of "the paper's equations are incomplete" is a cross-check
against ORCA (the paper's own implementation), which needs a licence. Without
that, H1 cannot be settled -- say so rather than substituting a weaker test.
