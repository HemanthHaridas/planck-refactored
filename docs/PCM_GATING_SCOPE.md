# Which PCM workflows run ungated, and what gates them?

Scope, opened 2026-11-10. Tests only, no new code expected. It expires when
the cases land; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 10). PCM itself is answered in
`docs/PCM_GEOMETRY.md`.

## The gap

The PCM geometry work gated RHF and RKS geometry workflows and every
single-point/gradient combination it touched. These run through the same
code (`Gradient::pcm_gradient`, the per-geometry cavity rebuild) with no case
of their own:

| workflow | binary | gated today |
|---|---|---|
| UHF gradient | hartree-fock | yes (`water_cation_uhf_pcm_swig_gradient_631gd`) |
| UHF geomopt / freq | hartree-fock | no |
| UKS gradient | planck-dft | no |
| UKS geomopt / freq | planck-dft | no |
| RKS frequencies | planck-dft | no (geomopt only: `water_rks_pbe_pcm_swig_geomopt_631gd`) |
| spherical gradient | hartree-fock | RHF only (`h2o2_rhf_pcm_swig_gradient_spherical_631gd`) |
| spherical geomopt / freq | both | no |
| spherical KS gradient | planck-dft | no |

## Cases

All 6-31G*, SWIG C-PCM water, references from PySCF 2.13.0
(`tests/pyscf/.venv`), the same scripts the PCM geometry gates use. Set
`conv_tol_grad` on the PySCF side (`pyscf_conv_tol_grad` memory).

1. `water_cation_uks_pbe_pcm_swig_gradient_631gd` — every component.
2. `water_cation_uhf_pcm_swig_geomoptfreq_631gd` — E_opt (geometric
   gau_tight) and frequencies (PySCF analytic PCM Hessian, ≤ 0.5 cm⁻¹).
3. `water_rks_pbe_pcm_swig_freq_631gd` — frequencies.
4. `water_rhf_pcm_swig_spherical_geomoptfreq_631gd` and one spherical RKS
   gradient.
5. `water_cation_uks_pbe_pcm_swig_geomopt_631gd`.

Mutation check on each new case, as the original gates did: remove the
cavity from the loop (or the PCM gradient term) and confirm the case fails.
A case that passes both ways is not a gate.

Keep the runtime down: one or two of these in `core`, the rest `extended`.

## Imaginary-mode following under PCM

Rejected in `io.cpp`. The HF loops would carry the cavity. Lift the
rejection only together with a case (a PCM transition state whose
imaginary mode is followed to a minimum); otherwise leave it rejected. Low
priority.
