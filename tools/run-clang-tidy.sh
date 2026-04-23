#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: tools/run-clang-tidy.sh [options] [path ...] [-- extra clang-tidy args]

Run clang-tidy over translation units in the given paths using the project's
compile_commands.json database. Vendored sources under src/external and the
PySCF fixture tree under tests/pyscf are skipped.

Options:
  --build-dir DIR       Build directory containing compile_commands.json.
  --header-filter REGEX Header filter passed to clang-tidy.
  --checks CHECKS       Checks selection passed to clang-tidy.
  --jobs N              Parallel clang-tidy jobs. Default: available CPUs.
  --fix                 Apply suggested fixes in place.
  --help                Show this help text.

Examples:
  tools/run-clang-tidy.sh
  tools/run-clang-tidy.sh src/post_hf tests
  tools/run-clang-tidy.sh --build-dir build --checks='-*,bugprone-*'
  tools/run-clang-tidy.sh src -- --warnings-as-errors='*'
EOF
}

find_compile_commands_dir() {
    local candidates=(
        "${repo_root}/build"
        "${PWD}/build"
        "${repo_root}"
        "${PWD}"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/compile_commands.json" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    candidate="$(find "${repo_root}" -maxdepth 3 -name compile_commands.json -print | head -n 1 || true)"
    if [[ -n "${candidate}" ]]; then
        dirname "${candidate}"
        return 0
    fi

    return 1
}

cpu_count() {
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || true
        return 0
    fi
    if command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN 2>/dev/null || true
        return 0
    fi
    printf '4\n'
}

build_dir=""
header_filter="^${repo_root}/(src|tests)/"
checks=""
jobs="$(cpu_count)"
fix=0
declare -a scopes=()
declare -a extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            build_dir="${2:?missing value for --build-dir}"
            shift 2
            ;;
        --header-filter)
            header_filter="${2:?missing value for --header-filter}"
            shift 2
            ;;
        --checks)
            checks="${2:?missing value for --checks}"
            shift 2
            ;;
        --jobs)
            jobs="${2:?missing value for --jobs}"
            shift 2
            ;;
        --fix)
            fix=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        *)
            scopes+=("$1")
            shift
            ;;
    esac
done

if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "error: clang-tidy is not installed or not on PATH" >&2
    exit 1
fi

if [[ -z "${build_dir}" ]]; then
    if ! build_dir="$(find_compile_commands_dir)"; then
        echo "error: could not find compile_commands.json; configure the project first or pass --build-dir" >&2
        exit 1
    fi
fi

if [[ ! -f "${build_dir}/compile_commands.json" ]]; then
    echo "error: ${build_dir}/compile_commands.json does not exist" >&2
    exit 1
fi

if [[ "${jobs}" =~ ^[0-9]+$ ]] && [[ "${jobs}" -lt 1 ]]; then
    echo "error: --jobs must be at least 1" >&2
    exit 1
fi

if [[ "${#scopes[@]}" -eq 0 ]]; then
    scopes=("src" "tests")
fi

tmpfile="$(mktemp)"
cleanup() {
    rm -f "${tmpfile}"
}
trap cleanup EXIT

declare -a normalized_scopes=()
for scope in "${scopes[@]}"; do
    if [[ "${scope}" = /* ]]; then
        normalized_scopes+=("${scope}")
    else
        normalized_scopes+=("${repo_root}/${scope}")
    fi
done

for scope in "${normalized_scopes[@]}"; do
    if [[ ! -d "${scope}" ]]; then
        echo "warning: skipping missing directory ${scope}" >&2
        continue
    fi

    find "${scope}" \
        -path "${repo_root}/src/external" -prune -o \
        -path "${repo_root}/tests/pyscf" -prune -o \
        \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) \
        -print0 >> "${tmpfile}"
done

if [[ ! -s "${tmpfile}" ]]; then
    echo "error: no source files found under: ${scopes[*]}" >&2
    exit 1
fi

declare -a clang_tidy_cmd=(
    clang-tidy
    "-p=${build_dir}"
    "-header-filter=${header_filter}"
)

if [[ -n "${checks}" ]]; then
    clang_tidy_cmd+=("-checks=${checks}")
fi

if [[ "${fix}" -eq 1 ]]; then
    clang_tidy_cmd+=("-fix" "-fix-notes" "-fix-errors")
fi

if [[ "${#extra_args[@]}" -gt 0 ]]; then
    clang_tidy_cmd+=("${extra_args[@]}")
fi

echo "Using compile database: ${build_dir}/compile_commands.json"
echo "Scanning paths: ${scopes[*]}"
echo "Running ${jobs} clang-tidy job(s)"

xargs -0 -n 1 -P "${jobs}" "${clang_tidy_cmd[@]}" < "${tmpfile}"
