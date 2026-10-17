#!/usr/bin/env bash
# Enumerate the V4 candidate against all frozen and consumed evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
artifact="tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03"

python3 tests/g4_3_v4_search.py \
    --selection-reports "$artifact"/selection/*.json \
    --development-request tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v4_development_request.json \
    --output "$artifact/g4-3-v4-combined-frontier.json"
