#!/bin/sh
# Measure any fixed-seed FCIQMC production candidate on the representative
# throughput fixture. The filename is retained because E1 introduced it.
#
# Each repetition runs the requested thread counts (default: 1 and 4). Separate output
# directories retain all logs, so medians and run-to-run spread can be audited
# without accidentally overwriting a prior measurement.
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
binary=${1:-"$repo/build/hartree-fock"}
fixture=${2:-"$repo/tests/inputs/exploratory/fciqmc/validation/hf_pop_50000.hfinp"}
out_root=${3:-"/private/tmp/fciqmc-e1"}
repetitions=${4:-3}
thread_counts=${5:-"1 4"}
runner="$repo/tests/run_fciqmc_scaling_profile.sh"

if [ ! -x "$runner" ]; then
    echo "FCIQMC profile ladder: missing runner: $runner" >&2
    exit 2
fi
if [ ! -x "$binary" ]; then
    echo "FCIQMC profile ladder: executable is missing or not executable: $binary" >&2
    exit 2
fi
if [ ! -f "$fixture" ]; then
    echo "FCIQMC profile ladder: fixture is missing: $fixture" >&2
    exit 2
fi
case "$repetitions" in
    ''|*[!0-9]*|0)
        echo "FCIQMC profile ladder: repetitions must be a positive integer" >&2
        exit 2
        ;;
esac
case "$thread_counts" in
    *[!0-9\ ]*) echo "FCIQMC profile ladder: expected space-separated thread counts" >&2; exit 2 ;;
esac
set -- $thread_counts
if [ "$#" -eq 0 ]; then
    echo "FCIQMC profile ladder: no thread counts supplied" >&2
    exit 2
fi
for threads do
    if [ "$threads" -lt 1 ]; then
        echo "FCIQMC profile ladder: thread counts must be positive" >&2
        exit 2
    fi
done

# An ISO-like directory component is sortable and avoids colliding with a
# previous measurement.  The requested root itself remains user-controlled.
stamp=$(date '+%Y%m%d-%H%M%S')
out_dir="$out_root/$stamp"
mkdir -p "$out_dir"

echo "FCIQMC profile ladder: repetitions=$repetitions output=$out_dir"
echo "FCIQMC profile ladder: thread counts $*"

run=1
while [ "$run" -le "$repetitions" ]; do
    for threads do
        "$runner" "$binary" "$fixture" "$threads" \
            "$out_dir/run-$run/threads-$threads"
    done
    run=$((run + 1))
done

echo "FCIQMC profile ladder complete: $out_dir"
echo "Extract records with:"
echo "  grep 'FCIQMC_PROFILE' $out_dir/run-*/*/*.log"
