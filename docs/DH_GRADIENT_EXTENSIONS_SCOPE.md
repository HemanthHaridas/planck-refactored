# What does the double-hybrid gradient need to run on production inputs?

Scope, opened 2026-11-10. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 14). Architecture is answered in
`docs/DH_IMPLEMENTATION.md` (follow its §9 extension rules) and
`docs/UDH_GRADIENT_ONE_ARCHITECTURE.md`.

## Why it matters

A production DH geometry is B2PLYP-D3 or similar in a spherical triple-zeta
basis with frozen core, usually with RI. Planck's DH gradient runs only
Cartesian, all-electron, conventional (non-RI) PT2, and is qualified on
STO-3G/6-31G fixtures. Three gaps, in order.

## 1. Spherical basis (first)

`validate_spherical_support` (`src/dft/driver.cpp`) refuses any DH workflow
but SinglePoint (`water_rks_b2plyp_gradient_spherical_rejected`). DH single
points already run spherical (PT2 on the spherical ERI tensor), and the KS
gradient lifts its densities with `to_cartesian`. The PT2 half needs the
same audit as `docs/SPHERICAL_MP2_GRADIENT_SCOPE.md`: which objects are in
the working basis, and where they are lifted. The streamed pair density and
Eq. 33 contraction consume AO-index quantities; back-transforming with the
lifted coefficients `Cᵀ C_mo` is the one-mechanism route. Do the MP2 scope
first; it is the same problem on a smaller code.

Gate: RKS B2PLYP/6-31G* spherical, analytic vs fourth-order energy FD (no
PySCF reference exists for DH gradients), ≤ 1e-8 Ha/Bohr; then UKS.

## 2. Frozen-core PT2 in the gradient

The UKS branch refuses `mp2_frozen`. **The RKS branch does not check it**:
verify first whether `calculation gradient` + a DH functional +
`mp2_frozen` is refused somewhere downstream, or silently differentiates an
all-electron Lagrangian against a frozen-core energy. If the latter, refuse
it by name with a regression case immediately; that is a correctness fix, not
part of the feature.

The feature: frozen core adds the core–active orbital-rotation block to the
Lagrangian and Z-vector (the standard frozen-core MP2 gradient term, a
Fock-weighted coupling between frozen and active occupied orbitals). The MP2
gradient refuses frozen core too
(`build_rmp2_lagrangian: frozen-orbital gradients are not implemented`), so
build it once in the MP2 gradient, gate it against PySCF
`mp.MP2(frozen=…).Gradients()`, then reuse the block in the DH Lagrangian.

## 3. RI-DH (O9)

A separate energy backend, the only route that drops the conventional AO
ERI tensor (the remaining N⁴ object), and the home of the UKS amplitude tile
source. RI-MP2 energies and gradients already exist
(`ri_mp2_gradient` memory; validate by FD, PySCF has no DF-MP2 gradient).
Scope it on its own once 1 and 2 land; it is the largest of the three.

## Also: larger-basis qualification

The UKS gradient is qualified only on STO-3G/6-31G fixtures. Add one
cc-pVDZ (spherical, after step 1) open-shell FD case before calling the
feature production-ready.

## Not in this scope

Range-separated, solvent, SCS/DOS, meta-GGA and TDDFT double hybrids
(nice-to-have in Open Work; each needs its own stationary functional).
