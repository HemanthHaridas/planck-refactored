# Regression Test Infrastructure

This directory contains a manifest-driven regression suite for the current
Planck executable. The goal is to test user-visible chemistry workflows at the
program boundary, not just individual helper functions.

## What it covers

- RHF + RMP2 single-point validation cases
- expected-failure behavior for unsupported edge cases
- CASSCF regression cases tied to the recent active-space fixes
- gradient smoke tests for the current RHF and RMP2 gradient paths

## Files

- `run_regressions.py`
  - Python runner that executes `hartree-fock`, parses the textual output, and
    enforces manifest-defined checks.
- `regression_cases.json`
  - Source of truth for test cases, tolerances, tags, and expected failures.
- `inputs/`
  - Dedicated test-only inputs that exercise workflows not already present in
    the project input corpus.

## Supported checks

The runner currently supports:

- required / forbidden output substrings
- expected process exit code
- exact-ish numerical checks with absolute tolerances
- inequality checks between extracted metrics
- expected string metrics such as point group
- expected counts such as the number of printed gradient atom lines

Extracted metrics currently include:

- `rhf_total_energy`
- `mp2_corr_energy`
- `mp2_total_energy`
- `casscf_corr_energy`
- `casscf_total_energy`
- `gradient_max`
- `gradient_rms`
- `point_group`
- `gradient_atom_lines`
- `scf_converged_iterations`

## Running the suite

From the repository root:

```bash
cmake -S . -B build
cmake --build build --target hartree-fock -j4
python3 tests/run_regressions.py --build-dir build --suite smoke
python3 tests/run_regressions.py --build-dir build --suite core
python3 tests/run_regressions.py --build-dir build --suite extended
```

You can also use the CMake targets:

```bash
cmake --build build --target regression-smoke
cmake --build build --target regression-core
cmake --build build --target regression-extended
```

Or via CTest:

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R planck-regression-core --output-on-failure
```

## Adding a new regression case

1. Add or reuse an input file.
2. Run the executable manually and capture the stable quantities worth checking.
3. Add a new object to `regression_cases.json` with:
   - a unique `id`
   - an `input`
   - one or more `tags`
   - the expected exit code
   - required strings and numerical checks
4. Prefer checking physically meaningful invariants in addition to exact totals.
   Examples:
   - `mp2_total_energy < rhf_total_energy`
   - `casscf_total_energy <= rhf_total_energy`
   - expected gradient lines count matches atom count

## Design notes

- This suite intentionally tests the compiled binary end-to-end, including
  parsing, symmetry handling, SCF, post-HF drivers, logging, and checkpoint
  interactions.
- The `he_rmp2_no_virtual_expected_failure` case is intentionally kept as a
  regression test for current behavior: RHF succeeds, then RMP2 exits with a
  clear diagnostic because there are no occupied/virtual excitations.
- Gradient tests are smoke tests rather than high-precision references because
  the current `RMP2` gradient path is still central-difference based.
