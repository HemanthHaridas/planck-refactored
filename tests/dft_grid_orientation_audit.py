#!/usr/bin/env python3
"""Rotational covariance of ordinary KS gradients under each grid policy.

Measurement 1 of docs/DFT_DEFAULT_GRID_SCOPE.md. Runs each fixture at the
input orientation and at three rigid rotations, rotates the gradient back, and
reports max |R^T g(R x) - g(x)| (Ha/Bohr). A covariant quadrature gives ~0.
"""
import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path

H2O2 = """O 0.000000 0.700000 -0.050000
O 0.020000 -0.710000 0.040000
H 0.880000 0.900000 0.420000
H -0.650000 -0.950000 0.780000"""
BENZENE = """C 0.000000 1.395000 0.000000
C 1.208105 0.697500 0.000000
C 1.208105 -0.697500 0.000000
C 0.000000 -1.395000 0.000000
C -1.208105 -0.697500 0.000000
C -1.208105 0.697500 0.000000
H 0.000000 2.485000 0.000000
H 2.152073 1.242500 0.000000
H 2.152073 -1.242500 0.000000
H 0.000000 -2.485000 0.000000
H -2.152073 -1.242500 0.000000
H -2.152073 1.242500 0.000000"""
def water_chain(n, spacing):
    return "\n".join(f"O 0.0 0.0 {z}\nH 0.757 0.586 {z}\nH -0.757 0.586 {z}"
                     for z in (spacing * k for k in range(n)))


WATER4 = water_chain(4, 3.0)

# (name, coords, charge, mult, scf_type, exchange, correlation)
CASES = {
    'h2o2_pbe': (H2O2, 0, 1, 'rhf', 'pbe', 'pbe'),
    'h2o2_b3lyp': (H2O2, 0, 1, 'rhf', 'b3lyp', 'lyp'),
    'h2o2_hse06': (H2O2, 0, 1, 'rhf', 'hse06', 'pbe'),
    'h2o2_cation_uks_pbe': (H2O2, 1, 2, 'uhf', 'pbe', 'pbe'),
    'benzene_b3lyp': (BENZENE, 0, 1, 'rhf', 'b3lyp', 'lyp'),
    'water4_b3lyp': (WATER4, 0, 1, 'rhf', 'b3lyp', 'lyp'),
    # Atom-count vs neighbour-overlap probes for the error growth.
    'water1_b3lyp': (water_chain(1, 3.0), 0, 1, 'rhf', 'b3lyp', 'lyp'),
    'water2_b3lyp': (water_chain(2, 3.0), 0, 1, 'rhf', 'b3lyp', 'lyp'),
    'water4_6a_b3lyp': (water_chain(4, 6.0), 0, 1, 'rhf', 'b3lyp', 'lyp'),
}

TEMPLATE = """%begin_control
    basis       {basis}
    calculation {calc}
    verbosity   normal
    basis_type  cartesian
%end_control
%begin_scf
    scf_type    {scf}
    guess       sad
    max_cycles  200
    tol_energy  1.0e-10
    tol_density 1.0e-10
    save_checkpoint .false.
%end_scf
%begin_dft
    grid        {grid}
    grid_policy {policy}
{extra}    exchange    {x}
    correlation {c}
    print_grid_summary .false.
%end_dft
%begin_geom
    coord_type  cartesian
    coord_units angstrom
    use_symm    .false.
%end_geom
%begin_coords
{n}
{charge} {mult}
{coords}
%end_coords
"""


def axis_angle(axis, angle):
    n = math.sqrt(sum(a * a for a in axis))
    a, b, c = (v / n for v in axis)
    co, si = math.cos(angle), math.sin(angle)
    u = (a, b, c)
    k = ((0., -c, b), (c, 0., -a), (-b, a, 0.))
    return [[co * (i == j) + (1 - co) * u[i] * u[j] + si * k[i][j] for j in range(3)] for i in range(3)]


ROTATIONS = [axis_angle((1, 2, 3), .731), axis_angle((-2, 1, 5), 1.913), axis_angle((3, -1, 1), 2.618)]


def mat_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def mat_t_vec(m, v):
    return [sum(m[j][i] * v[j] for j in range(3)) for i in range(3)]


def run(exe, case, policy, grid, basis, rot, user_extra='', energy_only=False):
    coords, charge, mult, scf, x, c = CASES[case]
    lines = []
    for line in coords.splitlines():
        el, *xyz = line.split()
        p = [float(t) for t in xyz]
        if rot is not None:
            p = mat_vec(rot, p)
        lines.append(f"{el} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}")
    extra = '    grid_radial_points 135\n' if policy == 'g4-3-v4' else ''
    extra += ''.join(f'    {l.strip()}\n' for l in user_extra.split(';') if l.strip())  # V4 is qualified only at 135 shells
    text = TEMPLATE.format(calc='energy' if energy_only else 'gradient', extra=extra, basis=basis, scf=scf, grid=grid, policy=policy, x=x, c=c,
                           n=len(lines), charge=charge, mult=mult, coords="\n".join(lines))
    with tempfile.TemporaryDirectory() as d:
        inp = Path(d) / 'in.hfinp'
        inp.write_text(text)
        out = subprocess.run([str(exe), str(inp)], cwd=d, capture_output=True, text=True).stdout
    grad = [[float(t) for t in m] for m in
            re.findall(r'Atom\s+\d+:\s+(\S+)\s+(\S+)\s+(\S+)', out)]
    energy = re.findall(r'(?:DFT|Total) Energy\s*:\s*(-?\d+\.\d+)', out)
    if energy_only:
        grad = [[0.0] * 3 for _ in lines]
    if len(grad) != len(lines):
        raise RuntimeError(f"{case}/{policy}: no gradient parsed\n{out[-2000:]}")
    if rot is not None:
        grad = [mat_t_vec(rot, g) for g in grad]
    return grad, float(energy[-1]) if energy else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable', type=Path, default=Path('build/planck-dft'))
    ap.add_argument('--cases', nargs='+', default=list(CASES), choices=list(CASES))
    ap.add_argument('--policies', nargs='+', default=['legacy', 'g4-3-v4'])
    ap.add_argument('--grid', default='normal')
    ap.add_argument('--basis', default='6-31g*')
    ap.add_argument('--energy-only', action='store_true', help='skip gradients (dg column reads 0)')
    ap.add_argument('--extra', default='', help='extra %%begin_dft lines, e.g. "grid_angular_points 302"')
    args = ap.parse_args()
    exe = args.executable.resolve()
    print(f"{'case':24s} {'policy':10s} {'max|dg| Ha/Bohr':>17s} {'max|dE| Eh':>12s}", flush=True)
    for case in args.cases:
        for policy in args.policies:
            g0, e0 = run(exe, case, policy, args.grid, args.basis, None, args.extra, args.energy_only)
            dg = de = 0.0
            for rot in ROTATIONS:
                g, e = run(exe, case, policy, args.grid, args.basis, rot, args.extra, args.energy_only)
                dg = max(dg, max(abs(a - b) for ga, gb in zip(g, g0) for a, b in zip(ga, gb)))
                de = max(de, abs(e - e0))
            print(f"{case:24s} {policy:10s} {dg:17.3e} {de:12.3e}", flush=True)


if __name__ == '__main__':
    main()
