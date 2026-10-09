#!/usr/bin/env bash
# Reconcile the test-only V4 literal grid against all consumed evidence.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python3 tests/g4_3_v4_direct_rescores.py "$@"
