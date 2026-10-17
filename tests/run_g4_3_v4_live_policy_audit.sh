#!/usr/bin/env bash
# Live stationary SCF/PT2 covariance checks for the frozen V4 test policy.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
for fixture in h2o2_cation_c1 ho2_asymmetric water_cation_631g; do
  python3 tests/udh_grid_policy_audit.py --fixture "$fixture" --radial-points 135 \
    --policies legacy g4-3-v4-test-only
done
