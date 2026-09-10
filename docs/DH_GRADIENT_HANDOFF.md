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
existing term, not a missing one**. **All four are now verified**, the orbital Hessian
included (2026-09-10, section 3) -- so the residual is an error in a term that
is individually correct as written, i.e. the equations Planck implements do not
match the ones the paper intends. **The next action is the stop condition in
section 7, not another term hunt.**

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
| **the orbital Hessian `h_op`** | diag/J/K exact to **3.7e-15** vs HF-CPHF (section 3) |
| **`h_op`'s XC channel** | its whole contribution is **0.36x/0.44x** the residual |

---

## 3. ANSWERED (2026-09-10): the orbital Hessian is clean

This section used to read "put `build_ks_orbital_hessian_op` under an
independent oracle" and was the named next step. **Done, and it comes back
clean -- the residual is not in the Z-vector operator.**

### The move that unblocked it: no finite difference at all

Three `kappa`-extraction attempts each came up ~30x short because
`U = C0^T S(R0) C(R+h)` mixes a genuine rotation with an `O(dS)` metric
mismatch (`|U - I| = 3.7e-6` at `h = 1e-5` against `O(h) = 1e-5`). The fix this
section prescribed was the `U_ij = -1/2 S^(x)_ij` bookkeeping of Eqs. 19-21.

**That is unnecessary. Do not build it.** The non-XC channels of the KS orbital
Hessian *are* the RHF CPHF matrix's couplings, so `build_rhf_cphf_matrix` --
a fully independent path (dense AO->MO ERI transform, textbook
`4(ai|jb) - (ab|ji) - (aj|bi)`) -- is an **exact** oracle for them. Comparing
operator-to-operator at a **fixed** geometry needs no displaced `C`, hence no
metric contamination and no gauge ambiguity.

**Generalizable:** when a quantity is a derivative, the reflex is to verify it
by finite difference. If it also has an algebraic identity to something already
implemented independently, that identity is the better oracle -- exact instead
of noise-floored, and it sidesteps whatever made the FD hard.

### What it measures

`PLANCK_DFT_DH_HESSIAN_AUDIT=1`, all three fixtures:

```
diag channel   : max|Hd - diag| = 0.000e+00   max|offdiag(Hd)| = 0.000e+00
J + K coupling : rel 3.6e-15 .. 3.8e-15  vs HF-CPHF
```

Gated **jointly** at machine precision: the `(a,i)` packing convention,
`kernel_scale = 2`, the `dP = C_v x C_o^T + h.c.` trial density, and the hybrid
`-0.5` K prefactor. Those are exactly the coefficients this section complained
had "never been matched against Eq. 41 term by term".

Non-vacuous -- three independent mutations, each caught **14 orders of
magnitude** above the clean value:

| mutation | rel |
|---|---|
| kernel scale `2 -> 1` | 4.7e-01 |
| K prefactor `-0.5 -> -1.0` | 1.1e+00 |
| drop the `h.c.` half of `dP` | 1.7e+01 |

### The XC channel is bounded out, not verified

`h_op`'s XC channel has no CPHF counterpart, so it gets a **scale control**
(`PLANCK_DFT_DH_HESSIAN_XC_SCALE`, per trap #5 -- never on/off). Bounding is
enough here:

| fixture | \|XC channel\| | residual | ratio |
|---|---|---|---|
| 1 (h2o2 C1) | 1.41e-4 | 3.89e-4 | **0.36** |
| 2 (h2o2b C1) | 1.18e-4 | 2.69e-4 | **0.44** |

Its **entire** contribution to the final gradient is under half the residual,
so no error in it -- not even zeroing it outright -- closes the gap. Scored as
a candidate: cos **-0.55 / -0.36**, scale spread **52.5%** -> the scorer's own
`INCONSISTENT -> fit artifact`. Linear in the scale (second difference
1.7e-05), so the bound extrapolates.

### What this leaves

H2.3 established that `A` is *well-formed* (symmetric, SPD, exactly solved);
this establishes that `A` is *right*. Every component of the Z-vector path is
now accounted for, so **the "error inside an existing term" framing has no
candidate term left inside the Z-vector**. The remaining suspect is the one
section 2's tail already names: the DH-specific `W^PT2` / `Gamma^PT2` of
Eqs. 42-46 versus the HF-MP2 forms Planck contracts -- a rewrite of the
surrounding assembly, not a one-term addition.

**Go to section 7.**

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
| `PLANCK_DFT_DH_HESSIAN_AUDIT` | channel-resolved oracle on `h_op` vs HF-CPHF (section 3); FD-free, mutation-verified |
| `PLANCK_DFT_DH_HESSIAN_XC_SCALE` | scale control on `h_op`'s XC channel, the one channel with no oracle |

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

**The orbital-Hessian audit has now come back clean (section 3), so this is
the live next action.** Run **H1.1** -- optimize the
fixtures with the DH gradient and compare against an FD-driven optimization. If
stiff coordinates agree to <0.05 pm, **ship it behind the flag with the residual
documented as a known bound** and stop hunting. A bounded, measured, non-blocking
limitation is a legitimate outcome.

The only decisive test of "the paper's equations are incomplete" is a cross-check
against ORCA (the paper's own implementation), which needs a licence. Without
that, H1 cannot be settled -- say so rather than substituting a weaker test.
