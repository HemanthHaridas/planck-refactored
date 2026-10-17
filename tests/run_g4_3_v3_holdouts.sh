#!/usr/bin/env bash
# Qualify the frozen, test-only G4.3 v3 candidate on its three fresh holdouts.
# The runner persists raw reference, residual, and direct-grid evidence; it
# does not promote a production grid policy.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

artifact="tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03"
export BASIS_PATH="${BASIS_PATH:-$repo_root/basis-sets}"

python3 tests/g4_3_holdout_audit.py \
    --request tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_holdout_request.json \
    --executable build/planck-udh-validation \
    --finalist "$artifact/g4-3-v3-combined-frontier.json" \
    --v3-direct-builder \
    --artifact-dir "$artifact/holdouts_v3" \
    --output "$artifact/holdouts_v3/ledger.json"
