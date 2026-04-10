#!/usr/bin/env bash
# sync-memory.sh
# Post-commit hook: regenerate project note files from live doc sources.
# Install: ln -sf ../../scripts/sync-memory.sh .git/hooks/post-commit
#
# Generates:
#   CLAUDE.md                               — aggregated from vault/ notes via vault_to_claude.py
#   notes/validation/CASSCF_Gate_Table.md   — live gate table from docs/CASSCF_STATUS.md
#   notes/roadmap/CASSCF_Remaining_Work.md  — live remaining work from docs/CASSCF_STATUS.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASSCF_STATUS="${REPO_ROOT}/docs/CASSCF_STATUS.md"
VAULT_SCRIPT="${REPO_ROOT}/scripts/vault_to_claude.py"

# ---------------------------------------------------------------------------
# 0. Regenerate CLAUDE.md from vault notes
# ---------------------------------------------------------------------------
if [[ -f "$VAULT_SCRIPT" ]]; then
    python3 "$VAULT_SCRIPT" --quiet
    echo "[sync-memory] CLAUDE.md regenerated from vault."
else
    echo "[sync-memory] WARNING: scripts/vault_to_claude.py not found — CLAUDE.md not regenerated." >&2
fi

if [[ ! -f "$CASSCF_STATUS" ]]; then
    echo "[sync-memory] docs/CASSCF_STATUS.md not found — skipping CASSCF notes." >&2
    exit 0
fi

mkdir -p "${REPO_ROOT}/notes/validation"
mkdir -p "${REPO_ROOT}/notes/roadmap"

# ---------------------------------------------------------------------------
# 1. Gate table note
# ---------------------------------------------------------------------------
GATE_OUT="${REPO_ROOT}/notes/validation/CASSCF_Gate_Table.md"

SUITE_STATUS="$(grep -m1 'Suite status:' "$CASSCF_STATUS" \
    | sed 's/.*\*\*Suite status:\*\* //' | tr -d '\r')"

{
    echo "# CASSCF PySCF Gate Table"
    echo ""
    echo "Source: \`docs/CASSCF_STATUS.md\`  "
    echo "**Suite status:** ${SUITE_STATUS:-unknown}  "
    echo "Last synced: $(date '+%Y-%m-%d %H:%M')"
    echo ""
    awk '
        /^## PySCF Gate Table/  { found=1; next }
        found && /^## /         { exit }
        found && /^\|/          { intable=1; print; next }
        found && intable && /^[^|[:space:]]/  { exit }
    ' "$CASSCF_STATUS"
} > "$GATE_OUT"

# ---------------------------------------------------------------------------
# 2. Remaining work note
# ---------------------------------------------------------------------------
WORK_OUT="${REPO_ROOT}/notes/roadmap/CASSCF_Remaining_Work.md"

{
    echo "# CASSCF Remaining Work"
    echo ""
    echo "Source: \`docs/CASSCF_STATUS.md\`  "
    echo "Last synced: $(date '+%Y-%m-%d %H:%M')"
    echo ""
    awk '
        /^## Remaining Work/    { found=1; print; next }
        found && /^## What Not To Do/ { found=0 }
        found                   { print }
    ' "$CASSCF_STATUS"
    echo ""
    echo "---"
    awk '
        /^## What Not To Do/    { found=1; print; next }
        found && /^## /         { exit }
        found                   { print }
    ' "$CASSCF_STATUS"
} > "$WORK_OUT"

echo "[sync-memory] CASSCF notes updated (suite: ${SUITE_STATUS:-unknown})"
