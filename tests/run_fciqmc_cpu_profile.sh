#!/bin/sh
# Capture a macOS CPU sample and Planck's phase ledger for one fixed-seed
# FCIQMC trajectory.  This is an observation tool: the temporary input only
# lowers verbosity and enables fciqmc_profile; it does not change a stochastic
# parameter in the source fixture.
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
binary=${1:-"$repo/build/hartree-fock"}
fixture=${2:-"$repo/tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp"}
threads=${3:-1}
seconds=${4:-30}
out_dir=${5:-"${TMPDIR:-/tmp}/fciqmc-cpu-profile"}

if [ ! -x "$binary" ]; then
    echo "FCIQMC CPU profile: executable is missing or not executable: $binary" >&2
    exit 2
fi
if [ ! -f "$fixture" ]; then
    echo "FCIQMC CPU profile: fixture is missing: $fixture" >&2
    exit 2
fi
if ! command -v sample >/dev/null 2>&1; then
    echo "FCIQMC CPU profile: macOS 'sample' is required" >&2
    exit 2
fi
case "$threads" in
    ''|*[!0-9]*|0)
        echo "FCIQMC CPU profile: thread count must be a positive integer" >&2
        exit 2
        ;;
esac
case "$seconds" in
    ''|*[!0-9]*|0)
        echo "FCIQMC CPU profile: sample duration must be a positive integer" >&2
        exit 2
        ;;
esac

mkdir -p "$out_dir"
temporary_input=$(mktemp "${TMPDIR:-/tmp}/planck-fciqmc-cpu-profile.XXXXXX.hfinp")
trap 'rm -f "$temporary_input"' EXIT HUP INT TERM

python3 - "$fixture" "$temporary_input" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
if 'fciqmc_profile' in source:
    raise SystemExit('fixture already declares fciqmc_profile; refuse to override it')
if 'fciqmc_seed' not in source:
    raise SystemExit('fixture has no fciqmc_seed insertion point')

source = source.replace('verbosity   verbose', 'verbosity   normal')
source = source.replace('fciqmc_seed', 'fciqmc_profile         true\n    fciqmc_seed', 1)
Path(sys.argv[2]).write_text(source)
PY

name=$(basename "$fixture" .hfinp)
prefix="$out_dir/${name}.threads-${threads}"
log="$prefix.log"
profile="$prefix.sample.txt"

printf 'FCIQMC CPU profile: fixture=%s threads=%s seconds=%s log=%s sample=%s\n' \
    "$fixture" "$threads" "$seconds" "$log" "$profile"
OMP_NUM_THREADS="$threads" "$binary" "$temporary_input" >"$log" 2>&1 &
pid=$!

# Give the program enough time to pass input setup without guessing at a
# solver iteration.  `sample` observes the real long-running phase thereafter.
sleep 1
sample_status=0
if sample "$pid" "$seconds" -file "$profile"; then
    :
else
    sample_status=$?
fi
solver_status=0
if wait "$pid"; then
    :
else
    solver_status=$?
fi

if [ "$solver_status" -ne 0 ]; then
    echo "FCIQMC CPU profile: calculation failed; see $log" >&2
    exit "$solver_status"
fi
if ! grep 'FCIQMC_PROFILE' "$log"; then
    echo "FCIQMC CPU profile: no profile records were emitted; see $log" >&2
    exit 1
fi
if [ "$sample_status" -ne 0 ]; then
    echo "FCIQMC CPU profile: sample failed; calculation completed; see $log" >&2
    exit "$sample_status"
fi

printf 'FCIQMC CPU profile: completed; inspect %s and %s\n' "$log" "$profile"
