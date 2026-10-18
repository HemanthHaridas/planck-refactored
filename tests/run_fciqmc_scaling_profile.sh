#!/bin/sh
# Run one reproducible FCIQMC scaling measurement without modifying its source
# fixture.  The generated input only changes verbosity to normal and enables the
# passive `fciqmc_profile` ledger; all stochastic parameters remain untouched.
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
binary=${1:-"$repo/build/hartree-fock"}
fixture=${2:-"$repo/tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp"}
threads=${3:-1}
out_dir=${4:-"$repo/tests/artifacts/fciqmc_scaling"}

if [ ! -x "$binary" ]; then
    echo "FCIQMC scaling: executable is missing or not executable: $binary" >&2
    exit 2
fi
if [ ! -f "$fixture" ]; then
    echo "FCIQMC scaling: fixture is missing: $fixture" >&2
    exit 2
fi
case "$threads" in
    ''|*[!0-9]*|0)
        echo "FCIQMC scaling: thread count must be a positive integer" >&2
        exit 2
        ;;
esac

mkdir -p "$out_dir"
temporary_input=$(mktemp "${TMPDIR:-/tmp}/planck-fciqmc-profile.XXXXXX.hfinp")
trap 'rm -f "$temporary_input"' EXIT HUP INT TERM

# Use Python rather than platform-specific sed -i syntax.  The input parser's
# normal boolean spelling is used so the emitted file is independently readable.
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
log="$out_dir/${name}.threads-${threads}.log"
printf 'FCIQMC scaling: fixture=%s threads=%s log=%s\n' "$fixture" "$threads" "$log"
OMP_NUM_THREADS="$threads" "$binary" "$temporary_input" >"$log" 2>&1

if ! grep 'FCIQMC_PROFILE' "$log"; then
    echo "FCIQMC scaling: no profile records were emitted; see $log" >&2
    exit 1
fi
