# What does the RMP2/UMP2 gradient need to run in a spherical basis?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 4).

## Why it matters

Every standard correlated basis (cc-pVXZ, def2) is spherical. MP2 energies
run spherical; MP2 gradients, geometry optimization and the workflows on top
of them are refused (`src/hf_driver.cpp`, the `reject("MP2 analytic
gradient")` in the spherical gate; boundary cases
`water_rmp2_spherical_{gradient,geomopt}_rejected`).

## The pattern that already works

SCF gradients lift every AO matrix to Cartesian once at the skin,
`M_cart = Cᵀ M_sph C` (`BasisFunctions::to_cartesian`), and reuse the
Cartesian derivative engine unchanged (`spherical_gradient_lift_pattern`
memory; `src/gradient/gradient.cpp`). Rule learned on ROHF: lift a finished
AO quantity once; never build it from separately lifted factors, since
C·Cᵀ ≠ I.

## What the audit has to cover

`src/post_hf/mp2_gradient.cpp` sizes everything with
`calculator._shells.nbasis()` (Cartesian) at about ten sites, while the MP2
kernel's MO coefficients are in the working (spherical) basis. Three classes
of object:

1. **Working-basis objects** — MO coefficients, the relaxed one-particle
   density, the energy-weighted density, the Lagrangian. These should be
   built in the spherical basis (`working_nbasis()`) and lifted once.
2. **Fock-like builds of a trial density** — `build_veff_from_density` and
   `build_veff_from_spin_densities` call `_compute_2e_fock{,_uhf}` with the
   Cartesian `nbasis()`. Either pass the working basis (the SCF Fock build
   already handles spherical) or lift the density in and project the result
   out. Use whatever the SCF does.
3. **The separable and non-separable two-particle density** contracted with
   derivative ERIs at the three `compute_eri_deriv_dispatch` sites. These
   are AO-index objects: build in spherical from MO quantities, lift with the
   4-index analogue of `to_cartesian` (C on each index), or back-transform
   the MO-basis Γ with lifted coefficients `C_cart = Cᵀ C_mo` directly. The
   second is cheaper and is what the KS stability code does
   (`ks_orbital_hessian.cpp`).

The response side (`rhf_response.cpp`) already uses `working_nbasis()`;
`uhf_response.cpp:87` still passes `nbasis()`.

## Steps

1. RMP2: make the Cartesian path pass a spherical-run-equivalent test first:
   Cartesian gradient bitwise unchanged after the refactor (lifts are
   identity there).
2. RMP2 spherical: gate against PySCF `cart=False` analytic `mp2.Gradients`
   on water/6-31G* and C1 H2O2/6-31G*, plus an energy FD on one of them.
3. UMP2 spherical: same, water⁺ doublet and one C1 radical.
4. Lift the gate in `hf_driver.cpp`; convert the two rejection cases into
   positive geomopt/frequency cases (PySCF `geometric_solver` E_opt).
5. RI-MP2 gradient in spherical: separate, after 1–4; the RI 3-centre
   factors have their own basis convention.

## Not in this scope

Frozen-core MP2 gradients (refused in Cartesian too:
`build_rmp2_lagrangian: frozen-orbital gradients are not implemented`). That
is the next most-asked MP2 feature and gets its own scope after this.
