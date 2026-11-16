# Can the RHF orbital Hessian be matrix-free, so RHF and UHF share one path?

Answer, 2026-11-16 (commit `9e9c2623`). It also fixed the RI-MP2 gradient:
`docs/RI_MP2_GRADIENT_RESPONSE_SCOPE.md`.

## Answer

**Yes, and nothing in production needed the explicit matrix.** Every
consumer of an HF orbital Hessian only applies it to vectors, so RHF and UHF
now share one matrix-free operator, `Correlation::OrbitalHessian`
(`src/post_hf/orbital_hessian.{h,cpp}`):

| Consumer | Before | Now |
|---|---|---|
| RMP2 gradient Z-vector (`solve_rhf_cphf`) | explicit A from the full AO ERI tensor and an nao⁴ MO transform, pivoted QR | operator + GMRES (orbital-gap diagonal preconditioner, 1e-10 relative) |
| UMP2 gradient Z-vector (`solve_uhf_cphf`) | `UHFOrbitalHessian` + GMRES | the same operator, `OrbitalHessian::uhf` |
| RHF SOSCF `h_op` | explicit A | the operator |
| UHF SOSCF `h_op` | explicit A built column by column, one Fock build per column | the operator |

**One action, one Fock build.** For a trial rotation x (n_virt × n_occ),
with D = C_v x C_oᵀ and D_s = D + Dᵀ:

  (A x)_ai = (ε_a − ε_i) x_ai + [C_vᵀ · 2 G(D_s) · C_o]_ai,  G(D) = J(D) − ½K(D)

which is A_{ai,bj} = (ε_a−ε_i)δ + 4(ai|jb) − (ab|ji) − (aj|bi) contracted
with x. G is what `SCF::two_electron_fock` returns; UHF uses
`two_electron_fock_uhf` on its two blocks.

**One fitted/exact choice.** `SCF::two_electron_fock{,_uhf}` return the RI
Fock build when the SCF ran RI-JK and the exact build otherwise, so every
response consumer follows the operator the SCF actually used. Before, the
choice was made in three places with two rules (`RI::fitted_response` for
CPHF and SOSCF, `mp2_use_ri` for the MP2 gradient's `veff`), which is what
broke the RI-MP2 gradient.

**Deleted:** `build_rhf_cphf_matrix`, `build_rhf_cphf_matrix_ri`,
`build_uhf_cphf_matrix`, `UHFOrbitalHessian`, `RI::fitted_response`, and the
`mp2_use_ri` branches of `build_veff_from_density{,_spin_densities}`.
`PLANCK_DEBUG_RHF_RESPONSE=1` still dumps A, through
`OrbitalHessian::materialize()`, so the MP2 benchmark comparison script
(`tests/benchmarks/mp2/pyscf_reference/rmp2_grad`) keeps working.

## Gates

| Gate | Result |
|---|---|
| `planck-ri-cphf-matrix`: operator vs a dense A from `transform_eri` | exact 4.5e-16; fitted vs exact 7.0e-5 (the fitting error, which must stay visible); UHF on [x, x] vs RHF 0. Before the builders were deleted, the fitted operator equalled `build_rhf_cphf_matrix_ri` to 1.2e-15 |
| Every MP2 gradient/geomopt/freq case, every HF/KS SOSCF and stability case, every RI case (100) | unchanged; dense RMP2/UMP2 still match PySCF at ~1e-8 |
| RI-MP2 gradient vs its own energy FD at 3e-7 | water RI-RMP2 6-31G* 1.3e-7 (was 3.9e-6), water⁺ RI-UMP2 1.2e-7 (was 1.6e-6); spherical 1.1e-7 / 1.5e-7, so spherical RI-MP2 gradients are no longer refused |
| `--suite core` | 174/174 |

## Cost

Before = `b5789db0` (the parent commit), after = `9e9c2623`, same build
tree, 4 threads, interleaved runs, median of 3 wall times for the whole run.
Energies and gradients agree to every printed digit.

| Case | Before | After |
|---|---|---|
| water/6-31G*, RHF SOSCF, conventional | 0.09 s | 0.13 s |
| water/6-31G*, RMP2 gradient, conventional | 0.11 s | 0.12 s |
| 4 H2O/cc-pVDZ, RHF SOSCF, direct | 17.5 s | 2.45 s (7.1×) |
| 4 H2O/cc-pVDZ, RMP2 gradient, direct | 16.8 s | 13.6 s (1.24×) |

The direct cases win because the old RHF path built the full AO ERI tensor
inside a direct run (and UHF SOSCF built A one column at a time). The small
conventional cases lose ≤ 40 ms, because each action is now a direct Fock
build rather than a contraction of the cached tensor. **Decided:** no second
contraction path in `two_electron_fock` to win that back.

## If a result moves later

- **A dense MP2 gradient moves by more than ~1e-9:** check the GMRES
  residual (`solve->true_residuals`); the operator itself equals the dense A
  to 4.5e-16.
- **An RI-JK SOSCF or stability case moves:** the fitted operator comes from
  `two_electron_fock` under `scf_ri_jk`; check the RI cache is primed there
  (the error message says so if not).
- **An RI-MP2 FD gate misses by ~4e-6 again:** a response consumer reaches a
  fitted operator. `git grep build_ri_fock` should show only `scf.cpp`
  (inside `two_electron_fock`) and the RI-JK SCF loop.

## Not answered here

- **The KS orbital Hessians** (`src/dft/ks_orbital_hessian.cpp`) have the
  same shape with an XC term: RKS is diag + 2J + 2f_xc − cK. They need J and
  K separately with a coefficient, which the fused G = J − ½K does not give,
  and they already take the fitted route under RI-JK. Sharing the J/K part
  would be a later step.
- **The SCF loop's own Fock build**, which carries incremental-Fock state.
