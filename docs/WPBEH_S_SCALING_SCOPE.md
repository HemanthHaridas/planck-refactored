# What should Planck do about libxc's discontinuous wPBEh exchange?

Research scope, opened 2026-11-06. Nothing has landed. Status lives in
`vault/Status/Open Work.md`. Rewrite this as the answer when it lands.

## The defect

libxc's `GGA_X_WPBEH` does not use the reduced gradient s directly. It first
rescales it with `s_scaling_2` (`maple/scaling.mpl`):

```
s_scaling_2(s) = s                               s < 1
               = s - ln(1 + exp(s - smax))       1 <= s <= 15   (smax = 8.572844)
```

At s = 1 the second branch gives 1 − 5.15e-4, so the scaled s jumps by
5.15e-4. The energy density jumps with it, by about 1e-4 relative
(+3.3e-6 at ρ = 0.1, +7.2e-5 at ρ = 1), and so does ∂e/∂σ.

This is deliberate upstream. libxc commit `173000399` (2020-01-27, M. Marques)
reads "Changed again s_scaling_2 to follow the old convention (even if that
leads to a discontinuous function)", and the source comment says the first
Henderson scaling is "apparently used by Gaussian". Planck's pinned libxc
(`4bd0e1e`) and PySCF 2.13.0's libxc 7.0.0 both carry it.

**Affected functionals.** Everything that builds on `XC_GGA_X_WPBEH`
(`src/hyb_gga_xc_hse.c`): the HSE family (HSE03, HSE06, HSE12, HSE12S, HSEsol)
and the wPBEh-based long-range-corrected hybrids (LRC-wPBEh and the WHS
variants). The HJS functionals use `gga_x_hjs` and are not affected. The exact
list is step 1.

## What it breaks, measured so far

- **Energy finite differences.** When a displacement moves a grid point's s
  across 1, the quadrature energy jumps by that point's weight times the jump.
  No analytic gradient sees this. On C1 H₂O₂ / 6-31G* HSE06 (δ = 5e-4 Bohr),
  7 of 354,000 points cross and carry 2.796e-8 Eh, which is the whole FD error
  (2.79e-8); with them removed the FD agrees with the analytic gradient to
  4.4e-8 (`tests/pyscf/hse06_s_scaling_jump.py`). The visible miss is 8.0e-6
  (Cartesian) to 1.5e-5 (spherical) Ha/Bohr, and it varies erratically with δ
  (1.3e-9 at δ = 1e-4).
- **Not broken:** Planck's HSE06 energy and analytic gradient. Both match
  PySCF to ≤ 1.2e-10, and PySCF reproduces the FD miss to 1e-12.
- **Passes by luck:** `water_dft_hse06_gradient_symm_ultrafine_fd`. None of its
  grid points cross s = 1 at the displacements it uses.

## What it might break, not measured

- **Semi-numerical frequencies.** They difference analytic gradients at
  ±0.005 Bohr. ∂e/∂σ also jumps at s = 1, so the gradient itself is
  discontinuous and a crossing point puts a step into a Hessian column.
  Size unknown.
- **Geometry optimization.** Energy and gradient steps of ~1e-8 can confuse a
  line search near convergence. Probably far below the 3e-4 force threshold,
  but not measured.
- **SCF and SOSCF.** V_xc is discontinuous at s = 1, and the analytic f_xc used
  by SOSCF is undefined at the jump. HSE06 SOSCF converges in every current
  gate, so this is a robustness question, not a known failure.

## Options

| option | what | cost | consequence |
|---|---|---|---|
| A. Accept and gate around it | Validate HSE-family gradients against PySCF, never against energy FD; make `tests/dft_gradient_fd.py` warn on wPBEh-based functionals | small | Planck stays identical to libxc, PySCF and Gaussian |
| B. Patch libxc at fetch | Make `s_scaling_2` continuous (e.g. `s_scaling_3`, or shift the s ≥ 1 branch) | small code, large blast radius | Every HSE-family energy moves away from PySCF and Gaussian; every PySCF-pinned HSE gate must be re-derived against a patched reference |
| C. Opt-in smooth variant | A keyword selecting a continuous scaling, defaulting to libxc's | medium (a patched libxc functional or a Planck-side wPBEh) | Two HSE06s in one code; only worth it if a workflow needs a smooth surface |
| D. Report upstream | Ask libxc for a continuous option | small | Upstream already chose the discontinuous form on purpose; low odds |

**Recommendation: A now, then measure whether B or C is ever needed.** A
matches what every other code computes. B buys smoothness by giving up the
cross-code agreement that validates the functional at all. Decide C only if
step 3 shows a real workflow cost.

## Steps

1. **List the affected functionals.** From `hyb_gga_xc_hse.c` and every
   `XC_GGA_X_WPBEH` user, map libxc IDs to the names Planck accepts. Add a
   one-line check to `DFT::XC` (or the wrapper) that reports whether a
   functional contains `GGA_X_WPBEH`, so the next two steps have one source of
   truth.
2. **Gate around it (option A).**
   - Replace `water_dft_hse06_gradient_symm_ultrafine_fd` with a PySCF
     comparison (grid response on, `conv_tol_grad` set). Keep the water
     symmetry-frame coverage it was added for (`vault` DFT Symmetry Frames
     gotcha), so compare the symmetry-on Planck gradient against PySCF.
   - Make `tests/dft_gradient_fd.py` print a warning naming the discontinuity
     when the functional contains `GGA_X_WPBEH`, so the next FD miss is not
     re-investigated.
3. **Measure the unmeasured.**
   - Frequencies: water and C1 H₂O₂ HSE06, Planck semi-numerical vs PySCF
     analytic Hessian, and Planck at two FD steps (0.005 and 0.002 Bohr). A
     step-dependent jump beyond ~0.1 cm⁻¹ is a real cost.
   - Geomopt: does an HSE06 optimization from a few starting geometries ever
     stall or oscillate near convergence?
   - SCF/SOSCF: count points with s within 1e-3 of 1 at convergence and check
     whether tight-tolerance HSE06 runs ever fail to converge.
4. **Decide B, C or nothing** from step 3, and record the answer here.

## Not in scope

Changing libxc's other functionals; meta-GGA scalings (`s_scaling_4` etc.);
a Planck-native wPBEh implementation unless step 4 chooses C.
