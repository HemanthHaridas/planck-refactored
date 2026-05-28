#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/build/benchmark-calculation-types}"
REPS="${REPS:-3}"
OMP_THREADS="${OMP_THREADS:-1}"

mkdir -p "$OUT_DIR"

declare -a CASES=(
  "energy|$ROOT_DIR/build/hartree-fock|$ROOT_DIR/tests/inputs/regression/spherical/water_rhf_spherical_direct_631gd.hfinp"
  "gradient|$ROOT_DIR/build/hartree-fock|$ROOT_DIR/tests/inputs/regression/geometry/water_rhf_gradient.hfinp"
  "geomopt|$ROOT_DIR/build/hartree-fock|$ROOT_DIR/tests/inputs/regression/checkpoint/water_geomopt.hfinp"
  "freq|$ROOT_DIR/build/hartree-fock|$ROOT_DIR/tests/inputs/regression/geometry/water_freq_symmetry.hfinp"
  "optfreq|$ROOT_DIR/build/hartree-fock|$ROOT_DIR/tests/inputs/regression/geometry/h2_optfreq.hfinp"
  "imagfollow|$ROOT_DIR/build/hartree-fock|$ROOT_DIR/tests/inputs/benchmarks/calculation_types/linear_water_imagfollow.hfinp"
  "linear_response|$ROOT_DIR/build/planck-dft|$ROOT_DIR/tests/inputs/regression/dft/water_triplet_uks_tddft_pbe_sto3g.hfinp"
)

median_from_file() {
  python3 - "$1" <<'PY'
import pathlib
import statistics
import sys

values = []
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if line:
        values.append(float(line))

if not values:
    raise SystemExit("no timing values found")

print(f"{statistics.median(values):.6f}")
PY
}

extract_wall_time() {
  python3 - "$1" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
patterns = [
    re.compile(r"Wall Time\s*:\s*(?:[0-9:\-\s]+\()?([0-9]+\.[0-9]+)\s*s(?:econds)?\)?"),
    re.compile(r"Wall Time\s*:\s*([0-9]+\.[0-9]+)\s*s(?:econds)?"),
]
for pattern in patterns:
    match = pattern.search(text)
    if match:
        print(match.group(1))
        raise SystemExit(0)
raise SystemExit("wall time not found")
PY
}

summarize_hotspots() {
  python3 - "$1" <<'PY'
import pathlib
import re
import sys

sample_path = pathlib.Path(sys.argv[1])
if not sample_path.exists():
    raise SystemExit(0)

lines = sample_path.read_text(errors="ignore").splitlines()
start = None
for idx, line in enumerate(lines):
    if line.startswith("Sort by top of stack"):
        start = idx + 1
        break

if start is None:
    raise SystemExit(0)

entries = []
for line in lines[start:]:
    if not line.strip():
        if entries:
            break
        continue
    if "Binary images" in line:
        break
    m = re.match(r"\s*(\d+)\s+(.*)", line)
    if not m:
        continue
    count = int(m.group(1))
    symbol = m.group(2).strip()
    if "(in libsystem_" in symbol or "(in libgomp" in symbol:
        continue
    entries.append((count, symbol))
    if len(entries) == 5:
        break

for count, symbol in entries:
    print(f"{count}\t{symbol}")
PY
}

printf "mode\tmedian_wall_s\treps\tbinary\tinput\n" > "$OUT_DIR/summary.tsv"

for case in "${CASES[@]}"; do
  IFS="|" read -r mode binary input <<<"$case"
  times_file="$OUT_DIR/${mode}.times"
  sample_file="$OUT_DIR/${mode}.sample.txt"
  hotspot_file="$OUT_DIR/${mode}.hotspots.tsv"
  : > "$times_file"

  echo "==> $mode"
  for rep in $(seq 1 "$REPS"); do
    run_log="$OUT_DIR/${mode}.run${rep}.log"
    env OMP_NUM_THREADS="$OMP_THREADS" "$binary" "$input" > "$run_log" 2>&1
    extract_wall_time "$run_log" >> "$times_file"
  done

  median="$(median_from_file "$times_file")"
  printf "%s\t%s\t%s\t%s\t%s\n" "$mode" "$median" "$REPS" "$binary" "$input" >> "$OUT_DIR/summary.tsv"

  env OMP_NUM_THREADS="$OMP_THREADS" "$binary" "$input" > "$OUT_DIR/${mode}.profile.log" 2>&1 &
  pid=$!
  sample "$pid" 1 10 -mayDie -file "$sample_file" >/dev/null 2>&1 || true
  wait "$pid"
  summarize_hotspots "$sample_file" > "$hotspot_file"
done

cat "$OUT_DIR/summary.tsv"
