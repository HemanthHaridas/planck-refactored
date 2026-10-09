#!/usr/bin/env bash
# Materialize the consumed-evidence bank for the V4 grid search. This is
# calibration recovery only; it does not select or promote a new grid policy.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export BASIS_PATH="${BASIS_PATH:-$repo_root/basis-sets}"

python3 tests/g4_3_v4_materialize_development.py \
    --request tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v4_development_request.json \
    --executable build/planck-udh-validation
