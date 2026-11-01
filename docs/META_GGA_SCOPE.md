# What does `planck-dft` need to support meta-GGA functionals?

Research scope, opened 2026-10-26. It expires when the work lands; rewrite it
then as the answer. Status lives in `vault/Status/Open Work.md`.

## Why it matters

Several of the most used modern functionals are meta-GGAs: r2SCAN and its
hybrids, TPSS, M06-L, M06-2X, MN15, B97M and ωB97M families. `xc_grid.cpp`
rejects them by name in `validate_functional_compatibility` ("meta-GGA
functionals are not supported yet because tau/laplacian terms are not
wired"). libxc evaluates them already; the missing part is Planck's grid
layer.

## What a τ-dependent meta-GGA adds

The kinetic-energy density, in libxc's convention

$$
\tau(\mathbf r) = \tfrac12 \sum_i |\nabla\psi_i|^2
= \tfrac12 \sum_{\mu\nu} P_{\mu\nu} \nabla\phi_\mu \cdot \nabla\phi_\nu
$$

(per spin channel for UKS), and one more potential term in the KS matrix:

$$
V^{\tau}_{\mu\nu} = \sum_g w_g v_\tau(g) \tfrac12 \nabla\phi_\mu(g) \cdot \nabla\phi_\nu(g)
$$

Both need only the AO gradients on the grid, which the GGA path already
evaluates (`AOGridEvaluation::gradient`, `src/dft/ao_grid.h`).

**Analytic nuclear gradients** need $\partial\tau/\partial R$, which
contracts AO second derivatives with the density. `evaluate_ao_hessian_on_grid`
already exists and is used by the GGA nuclear gradient, so no new AO
machinery is needed.

**Laplacian-dependent** meta-GGAs ($\nabla^2\rho$) are rare in routine use
and need AO second derivatives in the SCF itself. They stay rejected by name.

## Pieces that need a kernel, not just a potential

- **TDDFT and SOSCF** need the second functional derivatives involving τ
  ($v2\tau^2$, $v2\rho\tau$, $v2\sigma\tau$) in the response kernel. The
  analytic fxc Hessian (`src/dft/analytic_hessian.cpp`) is LDA and GGA
  only.
- **Double hybrids** with meta-GGA parts are uncommon; keep rejected.

## Steps

1. **τ and V^τ in the SCF**, RKS and UKS, pure and hybrid (global and
   range-separated), through `xc_grid.cpp`'s existing density and potential
   path, with `is_meta_gga_like()` admitted only for τ-dependent,
   non-Laplacian functionals.
   - Gate: energies against PySCF with the same functional for r2SCAN,
     TPSS, M06-L, M06-2X, r2SCAN0 (hybrid) and ωB97M-family members without
     VV10, on water and a C1 molecule. Set the tolerance by running a GGA
     control (PBE) through the same comparison first, which measures the
     grid floor between the two codes.
   - Grid sensitivity is real for meta-GGAs (SCAN is notorious; r2SCAN was
     designed to fix it). Report the grid-convergence of each gated
     functional, not just one grid.
2. **Analytic gradients** through the AO Hessian. Gate: FD of the energy and
   PySCF gradients; a geometry optimization of water with r2SCAN.
3. **Response kernel** for TDDFT and SOSCF, or an explicit rejection of
   meta-GGA in both until it exists. Gate (if built): PySCF TDA/Casida
   excitations, and SOSCF reaching the DIIS energy.
4. **VV10 nonlocal correlation** (needed for ωB97M-V, B97M-V, ωB97X-V):
   a separate two-grid integral with its own gradient. Scope it only if
   those functionals are wanted.

## Depends on / enables

Independent of spherical DFT and dispersion. r2SCAN-3c-style composite
methods would need spherical DFT, D4 and a special basis as well.
