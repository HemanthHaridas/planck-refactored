#!/usr/bin/env python3
"""PCM cavity smoothness gate (docs/PCM_GEOMETRY_SCOPE.md step 1).

Scans one O-H bond of water (RHF/STO-3G, SWIG C-PCM, 110 points per sphere)
and asserts the energy has no jumps: the second difference
E(r-h) - 2E(r) + E(r+h) stays within a bound set by the smooth curvature.
A hard-pruned cavity (points appearing and vanishing as spheres cross) gives
isolated second differences orders of magnitude above that bound.

  python3 tests/pcm_smoothness_scan.py [path/to/hartree-fock]
"""
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build/hartree-fock"
TEMPLATE = """%begin_control
    basis       sto-3g
    calculation energy
    verbosity   normal
    basis_type  cartesian
%end_control
%begin_scf
    scf_type    rhf
    guess       hcore
    max_cycles  100
    tol_energy  1.0e-11
    tol_density 1.0e-11
%end_scf
%begin_pcm
    model                   pcm
    solvent                 water
    surface_points_per_atom 110
%end_pcm
%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom
%begin_coords
3
0   1
O   0.000000  0.000000  0.000000
H   {hx:.10f}  0.000000  {hz:.10f}
H  -0.758602  0.000000  0.504284
%end_coords
"""
ENERGY = re.compile(r"Total Energy\s+([-+0-9Ee.]+)")


def energy(r):
    # Stretch H1 along its own O-H direction to length r (Angstrom).
    d = math.hypot(0.758602, 0.504284)
    hx, hz = 0.758602 * r / d, 0.504284 * r / d
    with tempfile.NamedTemporaryFile("w", suffix=".hfinp", dir=tempfile.gettempdir(), delete=False) as f:
        f.write(TEMPLATE.format(hx=hx, hz=hz))
    out = subprocess.run([str(EXE), f.name], cwd=Path(f.name).parent, text=True,
                         capture_output=True, check=True).stdout
    return float(ENERGY.findall(out)[-1])


h = 0.002
rs = [0.90 + i * h for i in range(101)]  # 0.90 .. 1.10 A
es = [energy(r) for r in rs]
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
