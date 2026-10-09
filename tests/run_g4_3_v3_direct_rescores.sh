#!/usr/bin/env bash
# Run the frozen 14-fixture literal-grid/atlas reconciliation for the
# test-only G4.3 v3 candidate. Requires a rebuilt planck-udh-validation
# executable and network access if the signed Boys library is not cached.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

artifact="tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03"
candidate="$artifact/g4-3-v3-combined-frontier.json"
request="tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json"
mkdir -p "$artifact/v3_direct_rescores"

export BASIS_PATH="${BASIS_PATH:-$repo_root/basis-sets}"

while IFS= read -r fixture; do
    base="$(basename "$fixture" .hfinp)"
    if command -v rg >/dev/null 2>&1; then
        reports="$(rg -l --fixed-strings "\"fixture\": \"$fixture\"" "$artifact/selection")"
    else
        reports="$(grep -rlF "\"fixture\": \"$fixture\"" "$artifact/selection")"
    fi
    if [[ "$(printf '%s\n' "$reports" | sed '/^$/d' | wc -l | tr -d ' ')" != "1" ]]; then
        printf 'Expected exactly one residual report for %s, found:\n%s\n' "$fixture" "$reports" >&2
        exit 1
    fi

    python3 tests/g4_3_direct_rescore_run.py \
        --executable build/planck-udh-validation \
        --fixture "$fixture" \
        --residual-report "$reports" \
        --combined-frontier "$candidate" \
        --v3 \
        --report-path "$artifact/v3_direct_rescores/$base.json"
done < <(jq -r '.fixtures[]' "$request")
