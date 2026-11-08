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
temporary_input=$(mktemp "${TMPDIR:-/tmp}/planck-fciqmc-profile-XXXXXXXX")
trap 'rm -f "$temporary_input"' EXIT HUP INT TERM

python3 "$repo/tests/fciqmc_profile_input.py" "$fixture" "$temporary_input"

name=$(basename "$fixture" .hfinp)
log="$out_dir/${name}.threads-${threads}.log"
cp "$temporary_input" "$out_dir/${name}.threads-${threads}.hfinp"
printf 'FCIQMC scaling: fixture=%s threads=%s log=%s\n' "$fixture" "$threads" "$log"
export BASIS_PATH="${BASIS_PATH:-$repo/basis-sets}"
OMP_NUM_THREADS="$threads" python3 - "$binary" "$temporary_input" "$log" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

binary, generated_input, log = map(Path, sys.argv[1:])
binary = binary.resolve()
record = {
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256(generated_input.read_bytes()).hexdigest(),
    "system": platform.system(),
    "machine": platform.machine(),
    "logical_cpus": os.cpu_count(),
    "basis_path": os.environ["BASIS_PATH"],
    "thread_env": {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    "load_before": os.getloadavg(),
}
with log.open("w") as out:
    start = time.monotonic()
    solver = subprocess.run([binary, generated_input], stdout=out, stderr=subprocess.STDOUT)
    record["process_wall_s"] = time.monotonic() - start
# In this fresh interpreter the solver is the only child. Darwin reports
# bytes and Linux reports KiB. Keep unsupported systems' RSS unspecified.
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
rss_scale = {"Darwin": 1, "Linux": 1024}.get(record["system"])
record.update(exit_code=solver.returncode, user_s=usage.ru_utime, system_s=usage.ru_stime,
              peak_rss_bytes=usage.ru_maxrss * rss_scale if rss_scale else None,
              load_after=os.getloadavg())
log.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
sys.exit(solver.returncode if solver.returncode >= 0 else 128 - solver.returncode)
PY

if ! grep 'FCIQMC_PROFILE' "$log"; then
    echo "FCIQMC scaling: no profile records were emitted; see $log" >&2
    exit 1
fi
