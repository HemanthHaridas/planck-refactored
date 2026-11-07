#!/usr/bin/env python3
"""PCM cavity smoothness gate (docs/PCM_GEOMETRY_SCOPE.md step 1).

Stretches the first H of the input along its O-H bond (0.90..1.10 A in
0.002 A steps) and asserts the energy has no jumps. A hard-pruned cavity
(points appearing and vanishing as spheres cross) puts a spike into the
fourth difference of E(r); a smooth one does not.

  python3 tests/pcm_smoothness_scan.py <input.hfinp> [--build-dir build]
"""
import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENERGY = re.compile(r"Total Energy\s+([-+0-9Ee.]+)")
H_LINE = re.compile(r"^H\s+(\S+)\s+(\S+)\s+(\S+)\s*$", re.M)

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
args = parser.parse_args()
exe = (args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir) / "hartree-fock"
text = args.input.read_text()
hx0, hy0, hz0 = map(float, H_LINE.search(text).groups())
length = math.sqrt(hx0 * hx0 + hy0 * hy0 + hz0 * hz0)  # O sits at the origin


def energy(r):
    s = r / length
    stretched = H_LINE.sub(f"H {hx0 * s:.10f} {hy0 * s:.10f} {hz0 * s:.10f}", text, count=1)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "scan.hfinp").write_text(stretched)
        out = subprocess.run([str(exe), "scan.hfinp"], cwd=d, text=True,
                             capture_output=True, check=True).stdout
    return float(ENERGY.findall(out)[-1])


h = 0.002
es = [energy(0.90 + i * h) for i in range(101)]
d2 = [es[i - 1] - 2 * es[i] + es[i + 1] for i in range(1, len(es) - 1)]
d3 = [d2[i + 1] - d2[i] for i in range(len(d2) - 1)]
d4 = [abs(d3[i + 1] - d3[i]) for i in range(len(d3) - 1)]
# Measured on the SWIG cavity: d3 = E''' h^3 falls smoothly from -1.7e-7 to
# -6e-8 and |d4| stays ~1e-9 (near the 1e-10 print floor). A jump of size dE
# in the energy puts a spike of ~6 dE into d4, so 5e-9 catches jumps >~1e-9.
print(f"max |d2| = {max(map(abs, d2)):.3e}  max |d3| = {max(map(abs, d3)):.3e}  "
      f"max |d4| = {max(d4):.3e}")
assert max(d4) < 5e-9, "PCM energy has a jump along the O-H stretch"
print("PASS")
