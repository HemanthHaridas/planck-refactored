# Which PCM workflows are gated, and against what?

Answer, 2026-11-10 (was a scope opened the same day). Status lives in
`vault/Status/Completion.md`. PCM itself is answered in
`docs/PCM_GEOMETRY.md`.

## Coverage

Every PCM workflow that runs is now covered by a regression case checked
against PySCF 2.13.0 C-PCM (`tests/pyscf/pcm_swig_references.py`, SWIG,
302 Lebedev points, ε 78.3553, Planck's GBS and Bohr), all water / 6-31G*:

| workflow | case | agreement |
|---|---|---|
| UHF gradient | `water_cation_uhf_pcm_swig_gradient_631gd` (earlier) | every component |
| UKS PBE gradient | `water_cation_uks_pbe_pcm_swig_gradient_631gd` (core) | 6e-12 Ha/Bohr |
| spherical RKS PBE gradient | `water_rks_pbe_pcm_swig_gradient_spherical_631gd` | 3.6e-11 |
| UHF geomopt + freq | `water_cation_uhf_pcm_swig_geomoptfreq_631gd` | E 1e-10, freq ≤ 0.8 cm⁻¹ |
| spherical RHF geomopt + freq | `water_rhf_pcm_swig_spherical_geomoptfreq_631gd` | E 7e-10, ≤ 0.3 cm⁻¹ |
| RKS PBE geomopt + freq | `water_rks_pbe_pcm_swig_geomoptfreq_631gd` | every printed digit, ≤ 0.1 cm⁻¹ |
| UKS PBE geomopt + freq | `water_cation_uks_pbe_pcm_swig_geomoptfreq_631gd` | every printed digit, ≤ 0.9 cm⁻¹ |

The cation cases start from SAD (PySCF `minao`); `hcore` has landed the
water cation on a saddle before. Frequencies are compared at the optimized
geometry only, with PySCF's analytic PCM Hessian (which handles UHF and UKS).

Each case also checks the results JSON: `solvation_energy` and the identity
electronic + nuclear + dispersion = total. The gradient cases pin
`solvation_energy` to PySCF's `G_pcm` to 1e-9. The geometry-optimization
cases allow 5e-7, because PySCF's own `gau_tight` geometry leaves about 1e-7
in `G_pcm` (UHF: Planck stops at 5.9e-6 Ha/Bohr and the energies agree to
1e-10).

**Mutation check.** With the PCM gradient term removed from the KS gradient
and from the HF Hessian displacements, all six new cases fail: gradients off
by up to 9e-2, frequencies by up to 570 cm⁻¹, and the UKS optimization does
not converge.

## Defects found and fixed on the way

- **HF frequency runs published the last displaced geometry.** After
  `compute_hessian` the wavefunction, energy, PCM energy and gradient on the
  `Calculator` belonged to the final −h displacement; the results JSON
  reported an energy ~3e-6 Eh off. `planck-dft` re-ran the reference in its
  own wrapper. `compute_hessian` (`src/freq/hessian.cpp`) now rebuilds the
  reference state itself, for both binaries, at the cost of one gradient out
  of 6N, and the DFT wrapper's copy is gone.
- **`planck-dft` geometry optimization wrote `solvation_energy` 0.** The
  optimizer's `Result` carries only the total. The KS SCF now stores the D3
  and PCM energies on the `Calculator` itself, so they always belong to the
  last SCF.
- **The geometry-optimization force threshold was hardcoded** (3e-4
  Ha/Bohr). The total energy is stationary at the minimum, but G_pcm moves
  linearly with the leftover geometry error: 1.5e-6 Eh off PySCF on the UKS
  cation at 3e-4, 1.9e-8 at 1.5e-5. New keyword `geomopt_grad_tol` in
  `%begin_geom` (default unchanged); the four cases above set 1.5e-5
  (`gau_tight`'s max force).

## Still rejected

Imaginary-mode following under PCM (`io.cpp`). The HF loops would carry the
cavity; lift the rejection only together with a case that follows a PCM
transition state to a minimum.
