#!/usr/bin/env bash
# Run the frozen V4 blind fixed-density qualification suite after rebuilding.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
artifact="tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03/v4_holdout"
mkdir -p "$artifact"
exec python3 tests/g4_3_holdout_audit.py \
  --request tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v4_holdout_request.json \
  --executable build/planck-udh-validation \
  --finalist tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03/g4-3-v4-combined-frontier.json \
  --v4-direct-builder --artifact-dir "$artifact" --output "$artifact/ledger.json"
