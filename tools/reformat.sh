#!/usr/bin/env bash
#
# reformat.sh — Apply Allman/BSD brace style + 4-space indentation to all C++
#               source files in src/ and tests/, excluding external dependencies.
#
# USAGE
# -----
#   From the repository root:
#
#     bash tools/reformat.sh          # reformat in-place
#     bash tools/reformat.sh --check  # dry-run: report files that would change
#                                     # (exits non-zero if any file differs)
#
# REQUIREMENTS
# ------------
#   clang-format must be on PATH.  Any version >= 10 is sufficient.
#   On macOS with Homebrew LLVM:  brew install llvm
#   On Ubuntu/Debian:             sudo apt install clang-format
#
# STYLE SETTINGS (enforced via .clang-format in the repo root)
# -------------------------------------------------------------
#   BasedOnStyle          : LLVM
#   BreakBeforeBraces     : Allman   (opening braces on their own line)
#   IndentWidth           : 4        (4-space indentation throughout)
#   NamespaceIndentation  : All      (namespace contents indented 4 spaces)
#   UseTab                : Never
#   ColumnLimit           : 0        (no line-length wrapping)
#
# FILES TOUCHED
# -------------
#   All *.cpp and *.h files under src/ and tests/, excluding src/external/.
#
# FILES NOT TOUCHED
# -----------------
#   Anything under src/external/ (third-party code; has its own style).
#   Non-C++ files (.json, .py, .sh, CMakeLists.txt, etc.).

set -euo pipefail

# ── Locate repo root ─────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Parse arguments ──────────────────────────────────────────────────────────

CHECK_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --check) CHECK_ONLY=1 ;;
        *) echo "Unknown argument: ${arg}"; echo "Usage: $0 [--check]"; exit 1 ;;
    esac
done

# ── Verify clang-format is available ─────────────────────────────────────────

if ! command -v clang-format &>/dev/null; then
    echo "error: clang-format not found on PATH."
    echo "  macOS:  brew install llvm && export PATH=\"\$(brew --prefix llvm)/bin:\$PATH\""
    echo "  Ubuntu: sudo apt install clang-format"
    exit 1
fi

# ── Verify .clang-format exists ──────────────────────────────────────────────

if [[ ! -f "${REPO_ROOT}/.clang-format" ]]; then
    echo "error: .clang-format not found at repo root (${REPO_ROOT})."
    exit 1
fi

# ── Collect files ────────────────────────────────────────────────────────────

FILES=()
while IFS= read -r f; do
    FILES+=("${f}")
done < <(find src tests \( -name '*.cpp' -o -name '*.h' \) ! -path 'src/external/*' | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No C++ files found."
    exit 0
fi

# ── Reformat or check ────────────────────────────────────────────────────────

CHANGED=()

if [[ ${CHECK_ONLY} -eq 1 ]]; then
    echo "Checking ${#FILES[@]} files for style conformance..."
    for f in "${FILES[@]}"; do
        if ! clang-format --dry-run --Werror "${f}" &>/dev/null; then
            CHANGED+=("${f}")
        fi
    done

    if [[ ${#CHANGED[@]} -gt 0 ]]; then
        echo ""
        echo "The following files do not conform to the house style:"
        for f in "${CHANGED[@]}"; do
            echo "  ${f}"
        done
        echo ""
        echo "${#CHANGED[@]} file(s) need reformatting. Run without --check to fix."
        exit 1
    else
        echo "All files conform to the house style."
    fi
else
    echo "Reformatting ${#FILES[@]} files..."
    for f in "${FILES[@]}"; do
        BEFORE="$(clang-format --output-replacements-xml "${f}")"
        clang-format -i "${f}"
        if echo "${BEFORE}" | grep -q '<replacement '; then
            CHANGED+=("${f}")
        fi
    done

    if [[ ${#CHANGED[@]} -gt 0 ]]; then
        echo "Reformatted ${#CHANGED[@]} file(s):"
        for f in "${CHANGED[@]}"; do
            echo "  ${f}"
        done
    else
        echo "All files were already conformant — nothing changed."
    fi
fi
