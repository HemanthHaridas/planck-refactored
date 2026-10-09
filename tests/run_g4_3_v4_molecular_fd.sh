#!/usr/bin/env bash
# Full-coordinate UKS/DH energy-FD checks under the frozen V4 test policy.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
for fixture in h2o2_cation_c1_b2plyp_sto3g ho2_asymmetric_b2plyp_sto3g water_cation_asymmetric_b2plyp_sto3g; do
  python3 tests/udh_molecular_audit.py --fixture "$fixture" --grid-policy g4-3-v4-test-only --radial-points 135 --steps 1e-4 2e-4
done
