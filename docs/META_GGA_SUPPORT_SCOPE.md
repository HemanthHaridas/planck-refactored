# Meta-GGA support (r2-SCAN and friends)

**Status:** scope. Nothing here is built. Single-point RKS/UKS only —
analytic gradients, geomopt, frequencies, TDDFT, and SOSCF stay out of scope
(see §5).

---

## 1. What already works, unchanged

The `Custom` functional path in `src/io/io.cpp` (`resolve_libxc_name_or_id`)
already resolves **any** libxc name or numeric ID, including meta-GGAs, and
builds an `XC::Functional` from it (`src/dft/driver.cpp:321-396`,
`DFT::XC::Functional::create`). So

```
exchange mgga_x_r2scan
correlation mgga_c_r2scan
```

already parses and constructs today. `XC::Functional::is_meta_gga_like()`
(`src/dft/base/wrapper.h:344`) already exists and is already checked. The
**only** thing stopping a meta-GGA run is one explicit rejection:

```cpp
// src/dft/xc_grid.cpp:200-206
if (functional.is_meta_gga_like())
{
    return std::unexpected(
        label + " meta-GGA functionals are not supported yet because tau/laplacian terms are not wired");
}
```

No new input syntax is needed. This is a grid/XC-evaluation/Fock-assembly
gap, not a parsing gap.

The other load-bearing piece already in the tree: `evaluate_ao_hessian_on_grid`
(`src/dft/ao_grid.h:278`), built for the SOSCF analytic fxc Hessian
(`docs/SOSCF_DFT.md`). It computes `h_xx..h_zz` — second derivatives of every
AO on every grid point — which is exactly what the Laplacian-dependent piece
of meta-GGA needs. Nothing there has to be duplicated, only reused.

## 2. What a meta-GGA functional needs per grid point

Beyond GGA's `rho` and `sigma = |∇rho|²`, libxc's meta-GGA interface
(`src/external/libxc/install/include/xc.h:456`) is:

```cpp
void xc_mgga_exc_vxc(const xc_func_type *p, size_t np,
    const double *rho, const double *sigma, const double *lapl, const double *tau,
    double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau);
```

Two new per-point scalars in, two new per-point response scalars out:

- **tau** — kinetic energy density, `τ_σ(r) = ½ Σᵢ occ,σ |∇φᵢ(r)|²`, or
  equivalently the density-matrix contraction
  `τ_σ(r) = ½ tr[P_σ · (∇φ)(∇φ)ᵀ]` at each point — same contraction shape
  `evaluate_density_channel` already uses for `rho`/`∇rho`, just with the AO
  gradient dotted into itself instead of into `φ`.
- **lapl** — `∇²rho_σ(r)`, needed only by functionals flagged
  `XC_FLAGS_NEEDS_LAPLACIAN` (`xc.h:63`). **r2-SCAN does not set this flag** —
  libxc computes it internally from tau where needed — so lapl can be wired
  generically (gated on the flag) without being exercised by the functional
  motivating this work.
- **vtau**, **vlapl** — the response terms, consumed at Fock-assembly time
  (§2.3).

### 2.1 `ao_grid.h` — tau (and lapl) on the grid

Add a tau evaluator next to `evaluate_density_channel`, same signature shape,
contracting `grad_x/grad_y/grad_z` against the density matrix instead of
`values`. Add lapl similarly, contracting `h_xx+h_yy+h_zz` (the Hessian trace)
against the density matrix — reusing `evaluate_ao_hessian_on_grid` rather than
adding a new AO derivative evaluator.

Extend `DensityChannelOnGrid` (`src/dft/xc_grid.h:16`) with `tau` and `lapl`
vectors alongside the existing `rho`/`grad_x/y/z`.

### 2.2 `xc_grid.cpp` — pack/unpack and the mgga eval call

- `pack_tau` / `pack_lapl` mirroring `pack_rho`/`pack_sigma`
  (`src/dft/xc_grid.cpp`, same file, same point-major packing convention).
- `XC::Functional::evaluate_mgga_exc_vxc` in `src/dft/base/wrapper.h`, mirroring
  `evaluate_gga_exc_vxc` (`wrapper.h:398-437`) exactly — same chunked
  `#pragma omp parallel for schedule(static)` shape, same per-chunk pointer
  arithmetic, same argument-count validation. Add a `needs_laplacian()` query
  off `XC_FLAGS_NEEDS_LAPLACIAN` so the lapl argument can be passed a real
  buffer only when required and a null/zero one otherwise, without a second
  code path.
- In `validate_functional_compatibility` (`xc_grid.cpp:190`), remove the
  meta-GGA rejection and route `is_meta_gga_like()` into
  `evaluate_functional_on_grid`'s `is_lda_like()/is_gga_like()` dispatch chain
  (`xc_grid.cpp:230-260`) as a third branch calling
  `evaluate_mgga_exc_vxc`. Extend `XCFunctionalGridResult` with `vtau`/`vlapl`
  matrices next to the existing `vrho`/`vsigma`.

### 2.3 `ks_matrix.cpp` — the tau Fock term

GGA's gradient term in `accumulate_local_potential`
(`ks_matrix.cpp:76-90`) adds `φ_μ·(coefficient·∇φ_ν)`-shaped contributions from
`vsigma`. Tau's Fock contribution is a **different** contraction — no `φ` on
either side:

```
F_μν += Σ_p w_p · ½ · vtau(p) · ∇φ_μ(p)·∇φ_ν(p)
```

This is a symmetric rank-1-per-Cartesian-axis update of `∇φ_μ · ∇φ_ν`
(dot product over x/y/z), structurally a `rankUpdate` over
`ao_grid.grad_{x,y,z}` rather than over `ao_grid.values`, so it slots into the
existing per-thread-partial / fixed-order-reduction pattern
(`ks_matrix.cpp:150-230`) without a new accumulation strategy. Add it as one
more term inside the existing point loop, gated on `xc_grid.vtau.cols() > 0`.

vlapl's Fock term (a basis-Laplacian contraction) is not needed for r2-SCAN
and should be **skipped** unless/until a `NEEDS_LAPLACIAN` functional is
actually wanted — build it generically enough that adding it later doesn't
require touching the tau plumbing, but don't build it now. (Ladder check:
speculative need, skip it.)

## 3. Validation

Same pattern every other functional family in this codebase used at
single-point landing: PySCF-gated regression case,
`water_rks_r2scan_sto3g` or similar, RKS and one UKS case (radical, e.g.
`h_dft_uks_r2scan_sto3g` mirroring the existing B3LYP UKS gate). Tolerance
in line with the existing semilocal gates (`1e-7`-class on the energy).

Grid density: r2-SCAN is known to be numerically sensitive to grid quality
(it's the whole reason r2-SCAN exists over SCAN — smoother interpolation, but
still meta-GGA-sharp near the tau→0 / iso-orbital limits). Use `grid
ultrafine` for the gate to avoid confusing a real defect with the pre-existing,
unrelated grid-radial under-resolution tracked in
`docs/DFT_GRID_CONVERGENCE_SCOPE.md`.

## 4. Hybrids

`is_hybrid()`/`exact_exchange_coefficient()` in `wrapper.h` are already
functional-family-agnostic (they read libxc's `hybrid_type()`/mixing
coefficient off the `xc_func_type`, not off `family()`), so a meta-GGA hybrid
(e.g. r2SCANh, M06-2X) should fall through the existing hybrid Fock-build path
unchanged **once the semilocal tau/vtau plumbing above lands** — no separate
hybrid-meta-GGA work anticipated, but not verified until a hybrid meta-GGA is
actually gated.

## 5. Explicitly out of scope — do not build

- **Analytic KS gradient / geomopt / frequency / geomoptfreq / imaginary-follow
  for meta-GGA.** τ depends on occupied MO gradients
  (`τ = ½Σᵢ|∇φᵢ|²·n_i`), not just `P` and `∇P`, so the nuclear-derivative and
  CPHF/Z-vector response machinery both need a new term class. This is a
  project on the scale of the original GGA analytic gradient, not an
  extension of it. The existing `is_gga_like()` gates in `driver.cpp`,
  `analytic_hessian.cpp`, and `dh_pt2_gradient.cpp` already reject meta-GGA by
  construction (they check `is_gga_like()`, not `!is_lda_like()`) — leave
  them rejecting, do not relax them as a side effect of the single-point work.
- **SOSCF for meta-GGA** (`docs/SOSCF_DFT.md`'s analytic XC Hessian-vector
  product) — same tau-response gap, same reasoning.
- **TDDFT/linear-response for meta-GGA** — the Casida/TDA kernel would need a
  tau-response fxc term; unbuilt, unscoped beyond this note.
- **Double-hybrid meta-GGA** — `dh_pt2_gradient.cpp` gates on `is_gga_like()`
  for the same structural reason; the double-hybrid work already has its own
  open residual (`docs/DH_CORR_GRADIENT_DEFECT_SCOPE.md`) and should not be
  entangled with this one.

## 6. Estimate

Steps in §2.1-2.3 are additive to existing per-point loops in three files
(`ao_grid.h`, `xc_grid.h`/`.cpp`, `ks_matrix.cpp`) plus one new wrapper method
in `wrapper.h`. No new files, no new subsystem. Single-point energy is the
whole deliverable; everything in §5 stays rejected with an explicit message
naming the missing feature, the same convention the rest of the codebase uses
for gated boundaries (e.g. the RMP2-spherical-gradient rejection message).
