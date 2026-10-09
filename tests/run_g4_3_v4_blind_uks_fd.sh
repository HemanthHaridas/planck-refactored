#!/usr/bin/env bash
# KS-gradient FD checks for the two frozen blind UKS cases under V4.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
for input in tests/inputs/exploratory/dh_gradient/grid_calibration/holdout_v4/water_triplet_uks_pbe_skew_sto3g.hfinp tests/inputs/exploratory/dh_gradient/grid_calibration/holdout_v4/water_cation_uks_pbe0_skew_631g.hfinp; do
  python3 tests/dft_gradient_fd.py "$input" \
    --grid-policy g4-3-v4-test-only --radial-points 135 \
    --scf-tolerance 1e-12 --delta 5e-4 --outer-delta 1e-3 \
    --atol 5e-8 --keep-tmp
done
