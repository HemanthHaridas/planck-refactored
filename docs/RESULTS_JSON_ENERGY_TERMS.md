# Which energy terms does the results JSON report separately?

Answer, 2026-11-10 (was a scope opened the same day). Status lives in
`vault/Status/Completion.md`.

## The defect

`dump_results_json` (`src/io/results_json.h`) wrote
`electronic_energy = _total_energy − _nuclear_repulsion`. For `planck-dft`
with `dispersion d3*` the total includes D3, so `electronic_energy` silently
contained the dispersion energy. A `python/planck.py` user adding their own
correction, or comparing with another code's electronic energy, got the wrong
number.

## What the JSON reports now

| field | meaning |
|---|---|
| `electronic_energy` | total − nuclear repulsion − D3. **Includes** the PCM solvation energy, which is part of the SCF functional. |
| `dispersion_energy` | D3 energy, 0 when off |
| `solvation_energy` | PCM solvation energy G_pcm, 0 when off |
| `scf_total_energy`, `total_energy` | unchanged |

So `electronic_energy + nuclear_repulsion + dispersion_energy ==
scf_total_energy` for every run. `electronic_energy` changes meaning only for
D3 runs; PCM runs keep the old value and gain the separate field.

## Where the values come from

Two `Calculator` fields, `_dispersion_energy` and `_solvation_energy`
(`src/base/types.h`), so both binaries write the same shape from the shared
serializer:

- `planck-dft` copies both from its driver `Result` next to the existing
  `_total_energy` copy before writing the JSON.
- `hartree-fock` sets `_solvation_energy` in the RHF and UHF iterations,
  where `_total_energy` comes from (`src/scf/scf.cpp`). HF has no D3. ROHF
  rejects PCM.

## Gate

The regression runner gained `json_checks` (`tests/run_regressions.py`): the
case runs the binary with `--json` and checks a field
(`{"field", "expected", "atol"}`) or that fields add up to another
(`{"sum": [...], "equals", "atol"}`).

- `water_rks_b3lyp_d3bj_sto3g`: `dispersion_energy` equals the PySCF D3
  reference (1e-11), `solvation_energy` is 0, the sum identity holds.
- `water_rhf_pcm_swig_631gd`: `solvation_energy` equals PySCF's `G_pcm`
  −0.0115156533 to every printed digit, `dispersion_energy` is 0, the sum
  identity holds.

Shifting either expected value by 1e-6 fails both cases. Under the old
definition, the D3 case's sum identity is off by the D3 energy (5.7e-4).
