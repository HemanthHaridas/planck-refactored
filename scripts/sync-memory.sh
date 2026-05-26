#!/usr/bin/env bash
# sync-memory.sh
# Post-commit hook: regenerate project note files from live doc sources.
# Install: ln -sf ../../scripts/sync-memory.sh .git/hooks/post-commit
#
# Generates:
#   CLAUDE.md                               — aggregated from vault/ notes via vault_to_claude.py
#   notes/validation/CASSCF_Gate_Table.md   — live gate table from vault/Status/Completion.md
#   notes/roadmap/CASSCF_Remaining_Work.md  — live remaining work from vault/Status/Open Work.md

set -euo pipefail

# Resolve repo root via git rather than $(dirname $0)/.., because git invokes
# this script through the .git/hooks/post-commit symlink — $0 there is
# ".git/hooks/post-commit", so dirname/.. yields ".git/" instead of the real
# repo root. git rev-parse always returns the worktree's top-level.
REPO_ROOT="$(git rev-parse --show-toplevel)"
CASSCF_COMPLETION="${REPO_ROOT}/vault/Status/Completion.md"
CASSCF_OPEN_WORK="${REPO_ROOT}/vault/Status/Open Work.md"
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

if [[ ! -f "$CASSCF_COMPLETION" || ! -f "$CASSCF_OPEN_WORK" ]]; then
    echo "[sync-memory] canonical status docs not found — skipping CASSCF notes." >&2
    exit 0
fi

mkdir -p "${REPO_ROOT}/notes/validation"
mkdir -p "${REPO_ROOT}/notes/roadmap"

# ---------------------------------------------------------------------------
# 1. Gate table note
# ---------------------------------------------------------------------------
GATE_OUT="${REPO_ROOT}/notes/validation/CASSCF_Gate_Table.md"

SUITE_STATUS="$(grep -m1 'Suite status:' "$CASSCF_COMPLETION" \
    | sed 's/.*Suite status: \*\*//' | sed 's/\*\*.*$//' | tr -d '\r')"

{
    echo "# CASSCF PySCF Gate Table"
    echo ""
    echo "Source: \`vault/Status/Completion.md\`  "
    echo "**Suite status:** ${SUITE_STATUS:-unknown}  "
    echo "Last synced: $(date '+%Y-%m-%d %H:%M')"
    echo ""
    awk '
        /^## CASSCF PySCF Gate Table/  { found=1; next }
        found && /^## /         { exit }
        found && /^\|/          { intable=1; print; next }
        found && intable && /^[^|[:space:]]/  { exit }
    ' "$CASSCF_COMPLETION"
} > "$GATE_OUT"

# ---------------------------------------------------------------------------
# 2. Remaining work note
# ---------------------------------------------------------------------------
WORK_OUT="${REPO_ROOT}/notes/roadmap/CASSCF_Remaining_Work.md"

{
    echo "# CASSCF Remaining Work"
    echo ""
    echo "Source: \`vault/Status/Open Work.md\`  "
    echo "Last synced: $(date '+%Y-%m-%d %H:%M')"
    echo ""
    awk '
        /^## CASSCF/            { found=1; next }
        found && /^## /         { exit }
        found                   { print }
    ' "$CASSCF_OPEN_WORK"
} > "$WORK_OUT"

echo "[sync-memory] CASSCF notes updated (suite: ${SUITE_STATUS:-unknown})"
