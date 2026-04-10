# Test Infrastructure

This directory now separates lightweight regression inputs from heavier benchmark/reference corpora.

## Layout

- `inputs/regression/`
  - Executable-facing regression inputs grouped by capability:
  - `hf/` for closed-shell HF symmetry/engine cases
  - `open_shell/` for UHF, ROHF, UMP2, and UHF-gradient coverage
  - `post_hf/` for RMP2, CASSCF, and RASSCF cases
  - `geometry/` for RHF/RMP2 gradient, optimization, and frequency workflows
  - `dft/` for RKS and UKS DFT workflows
  - `checkpoint/` for restart/full-guess fixtures
- `benchmarks/`
  - `casscf/archive/` for older exploratory CASSCF inputs and logs
  - `casscf/pyscf_reference/` for the PySCF-backed CASSCF reference corpus
  - `scf/engine_symmetry/` for engine/symmetry/SAD benchmark matrices
- `pyscf/`
  - External PySCF references for validating selected CASSCF energies
- `run_regressions.py`
  - Manifest-driven binary runner for `hartree-fock` and `planck-dft`
- `regression_cases.json`
  - Source of truth for regression cases, tolerances, and executable selection

## What The Regression Suite Covers

- RHF + RMP2 validation and expected-failure behavior
- SCF engine and symmetry smoke cases
- UHF, ROHF, UMP2, and UHF analytic-gradient smoke coverage
- RHF and RMP2 gradient / optfreq workflows
- checkpoint restart via `guess full`
- selected CASSCF / RASSCF user-visible workflows
- RKS and UKS DFT single-point coverage through `planck-dft`

## Supported Checks

The runner supports:

- required / forbidden output substrings
- expected process exit code
- exact-ish numerical checks with absolute tolerances
- inequality checks between extracted metrics
- expected string metrics such as point group
- expected counts such as the number of printed gradient atom lines
- per-case executable selection via the `executable` field

Extracted metrics currently include:

- `rhf_total_energy`
- `mp2_corr_energy`
- `mp2_total_energy`
- `rccsd_total_energy`
- `rccsdt_total_energy`
- `casscf_corr_energy`
- `casscf_total_energy`
- `dft_total_energy`
- `gradient_max`
- `gradient_rms`
- `point_group`
- `gradient_atom_lines`
- `scf_converged_iterations`

## Running

From the repository root:

```bash
cmake -S . -B build
cmake --build build --target hartree-fock planck-dft -j4
python3 tests/run_regressions.py --build-dir build --suite smoke
python3 tests/run_regressions.py --build-dir build --suite core
python3 tests/run_regressions.py --build-dir build --suite extended
```

Or via CMake / CTest:

```bash
cmake --build build --target regression-smoke
cmake --build build --target regression-core
cmake --build build --target regression-extended
ctest --test-dir build --output-on-failure
```

## Notes

- `tests/benchmarks/` is intentionally broader than the manifest. It stores slower or more exploratory inputs, reference logs, and timing fixtures.
- The checkpoint restart fixture is a true same-stem restart case: the `.hfinp`, `.hfchk`, and `.log` files are kept together under `tests/inputs/regression/checkpoint/`.
