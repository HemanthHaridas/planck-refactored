# What does `planck-dft` need to support meta-GGA functionals?

Research scope, opened 2026-10-26 and merged with the former
`META_GGA_SUPPORT_SCOPE.md` on 2026-11-07. Nothing here is built. It expires
when the work lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md`.

## Why it matters

Several of the most used modern functionals are meta-GGAs: r2SCAN and its
hybrids, TPSS, M06-L, M06-2X, MN15, and the B97M and ωB97M families. libxc
evaluates them already. What is missing is Planck's grid layer.

## What already works

Input and functional construction need no change. The `Custom` path
(`resolve_libxc_name_or_id` in `src/io/io.cpp`, then
`DFT::XC::Functional::create`) resolves any libxc name or ID, so

```
exchange mgga_x_r2scan
correlation mgga_c_r2scan
```

already parses and builds an `XC::Functional`. `is_meta_gga_like()`
(`src/dft/base/wrapper.h`, `family() == XC_FAMILY_MGGA`) already exists. Two
checks in `validate_functional_compatibility` (`src/dft/xc_grid.cpp`) are what
stop a run:

- an explicit `is_meta_gga_like()` rejection ("tau/laplacian terms are not
  wired");
- `is_supported_semilocal()`, which admits only LDA and GGA.

`evaluate_ao_hessian_on_grid` (`src/dft/ao_grid.h`) already computes the
second derivatives of every AO at every grid point. The GGA nuclear gradient,
the SOSCF analytic fxc Hessian and the UKS double-hybrid driver use it. It is
the only extra AO machinery that meta-GGA gradients need.

## What a τ-dependent meta-GGA adds

The kinetic-energy density, per spin channel in libxc's convention, is

$$
\tau_\sigma(\mathbf r) = \tfrac12 \sum_{i\in\sigma} |\nabla\psi_i|^2
= \tfrac12 \sum_{\mu\nu} P^\sigma_{\mu\nu}\, \nabla\phi_\mu \cdot \nabla\phi_\nu .
$$

It is a contraction of the density matrix, the same shape as
`evaluate_density_channel`'s ρ and ∇ρ, with the AO gradient dotted into
itself instead of into φ. The KS matrix gains one term:

$$
V^{\tau}_{\mu\nu} = \sum_g w_g\, v_\tau(g)\, \tfrac12\, \nabla\phi_\mu(g) \cdot \nabla\phi_\nu(g) .
$$

Neither needs anything beyond the AO gradients the GGA path already evaluates
(`AOGridEvaluation::gradient`).

libxc's interface (`xc.h`) is

```cpp
void xc_mgga_exc_vxc(const xc_func_type *p, size_t np,
    const double *rho, const double *sigma, const double *lapl, const double *tau,
    double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau);
```

**Laplacian-dependent functionals** (`XC_FLAGS_NEEDS_LAPLACIAN`) are rare in
routine use. r2SCAN does not need the Laplacian: both `mgga_x_r2scan` and
`mgga_c_r2scan` set only `XC_FLAGS_NEEDS_TAU`. Reject Laplacian functionals by
name and pass a zero `lapl` buffer otherwise. If one is ever wanted, ∇²ρ is
the density contraction of the trace of `evaluate_ao_hessian_on_grid`, and its
Fock term is a separate contraction that does not touch the τ code.

## Steps

### 1. τ and V^τ in the SCF (RKS and UKS)

Every change adds to an existing per-point loop. No new files.

- **`ao_grid.h` / `xc_grid.h`.** Add a τ evaluator next to
  `evaluate_density_channel`. Add a `tau` vector to `DensityChannelOnGrid`.
- **`wrapper.h`.** Add `evaluate_mgga_exc_vxc`, a copy of the shape of
  `evaluate_gga_exc_vxc`: same chunked `omp parallel for schedule(static)`,
  same pointer arithmetic, same argument validation. Add
  `needs_laplacian()`, read from the libxc flags.
- **`xc_grid.cpp`.** Add `pack_tau` next to `pack_rho` and `pack_sigma`.
  Replace the meta-GGA rejection with a `NEEDS_LAPLACIAN` rejection. Extend
  `is_supported_semilocal()` to τ-only meta-GGAs. Add a third branch to the
  `is_lda_like()` / `is_gga_like()` dispatch in
  `evaluate_functional_on_grid`. Add `vtau` to `XCFunctionalGridResult`.
- **`ks_matrix.cpp`.** Inside the existing point loop in
  `accumulate_local_potential`, add V^τ as a rank update over
  `grad_{x,y,z}`, not over `values`. Gate it on `vtau.cols() > 0`. It uses the
  existing per-thread partials and fixed-order reduction, so thread-count
  invariance (DFT XC Reduction Determinism) needs nothing new.
- **Hybrids.** `is_hybrid()`, `exact_exchange_coefficient()` and the
  range-separation queries read libxc's mixing data, not `family()`.
  Meta-GGA hybrids (r2SCAN0, M06-2X, ωB97M without VV10) should therefore
  use the existing global and range-separated exact-exchange paths unchanged.
  The gate below checks this.
- **Every other workflow must refuse meta-GGA by name in this step.** The KS
  nuclear gradient chooses its terms by `xc_grid->vsigma.cols() > 0`
  (`src/dft/driver.cpp`). A meta-GGA admitted to the SCF would get a gradient
  that silently drops the τ term. The SOSCF and TDDFT kernels
  (`analytic_hessian.cpp`) and the double-hybrid code already refuse anything
  but LDA and GGA, but check that each one says "meta-GGA" in its message.
  Gradient, geomopt, frequency, geomopt+freq, SOSCF, stability, TDDFT and
  double hybrids each need a rejection case.

**Gate.** PySCF energies, with the same functional, for r2SCAN, TPSS, M06-L,
M06-2X, r2SCAN0 and an ωB97M member without VV10. Run them on water and on a
C1 molecule, RKS and one UKS radical (mirroring `h_dft_uks_b3lyp_sto3g`).

- Run a GGA control (PBE) through the same comparison first. It measures the
  grid floor between the two codes, and that floor sets the tolerance.
- Meta-GGAs are grid-sensitive. SCAN is the bad case, and r2SCAN was made to
  fix it, but it is still sharp near the τ → 0 and iso-orbital limits. Report
  the grid convergence of each gated functional, not a single grid. Use
  `grid ultrafine` or an explicit radial count for the committed cases, so a
  real defect is not confused with grid error
  (`docs/DFT_GRID_CONVERGENCE.md`).
- Check that 1 and 4 threads give bitwise-identical energies.

### 2. Analytic nuclear gradients, geomopt, frequencies

∂τ/∂R contracts AO second derivatives with the density, and
`evaluate_ao_hessian_on_grid` already provides them. This is one more term
class in the existing XC nuclear gradient, next to the `vsigma` term, plus
the matching grid-response term. It is not a rebuild of the gradient.

No CPHF or Z-vector solve is needed. τ is a functional of P, not of the
individual orbitals, and the KS energy is variational, the same reason the
LDA and GGA gradients need no CPHF. (The former support scope said the
opposite, claiming that "τ depends on occupied MO gradients, not just P". That
is wrong: its own formula writes τ as a contraction of P.)

**Gate.** Fourth-order energy FD on a C1 molecule. PySCF analytic gradients,
with `conv_tol_grad` set (see the PySCF conv_tol_grad memory). A water
geometry optimization with r2SCAN. Spherical-basis variants come for free
through the existing `to_cartesian` lift. Gate one of them anyway.

### 3. Response kernel for TDDFT, SOSCF and stability

These need the second derivatives with τ (`v2tau2`, `v2rhotau`, `v2sigmatau`)
in `analytic_hessian.cpp`, which handles only LDA and GGA today. The
alternative is to keep the step-1 rejections. Build it only if meta-GGA
excitations or SOSCF are wanted.

**Gate, if built.** PySCF TDA and Casida excitations. SOSCF reaching the DIIS
energy. Stability λ_min against PySCF.

### 4. VV10 nonlocal correlation

ωB97M-V, B97M-V and ωB97X-V need VV10. It is a separate two-grid integral with
its own potential and gradient, and it is not specific to meta-GGA. Scope it
only if those functionals are wanted. `docs/DISPERSION_D3.md` points
here for it.

## Out of scope

- Laplacian-dependent meta-GGAs, until one is wanted (see above).
- Double hybrids with meta-GGA parts. They are uncommon, and they would be a
  new XC-II and kernel contract in the double-hybrid gradient
  (`docs/DH_IMPLEMENTATION.md`).
- PCM with meta-GGA, beyond what step 1 inherits. The PCM reaction field does
  not depend on the functional, so single points should work. Gradients wait
  for `docs/PCM_GEOMETRY.md`.
- r2SCAN-3c-style composite methods. They also need D4 and a special basis.

## Depends on / enables

Independent of spherical DFT (landed) and of dispersion.
