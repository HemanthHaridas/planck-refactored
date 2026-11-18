#!/usr/bin/env python3
"""Rotational covariance, frequency noise and cost of KS gradients per grid setting.

The measuring tool of docs/DFT_DEFAULT_GRID_SCOPE.md (steps S3-S5). Runs each
fixture at the input orientation and at three rigid rotations and reports, per
grid setting (policy@preset):

  gradient mode (default)  max |R^T g(R x) - g(x)| (Ha/Bohr) and max |dE| (Eh)
  --frequency              max |nu(R x) - nu(x)| over modes (cm^-1)
  --energy-only            max |dE| only

plus grid points, mean wall time per run and the largest peak memory footprint.
A covariant quadrature gives ~0 noise.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
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
# S22 water dimer: H-bonded, O-O 2.91 A.
WATER_DIMER = """O -1.551007 -0.114520 0.000000
H -1.934259 0.762503 0.000000
H -0.599677 0.040712 0.000000
O 1.350625 0.111469 0.000000
H 1.680398 -0.373741 -0.758561
H 1.680398 -0.373741 0.758561"""


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
    'water_dimer_b3lyp': (WATER_DIMER, 0, 1, 'rhf', 'b3lyp', 'lyp'),
    # Atom-count vs neighbour-overlap probes for the error growth.
    'water1_b3lyp': (water_chain(1, 3.0), 0, 1, 'rhf', 'b3lyp', 'lyp'),
    'water2_b3lyp': (water_chain(2, 3.0), 0, 1, 'rhf', 'b3lyp', 'lyp'),
    'water4_6a_b3lyp': (water_chain(4, 6.0), 0, 1, 'rhf', 'b3lyp', 'lyp'),
    # Cost at size (S5).
    'water12_b3lyp': (water_chain(12, 3.0), 0, 1, 'rhf', 'b3lyp', 'lyp'),
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
    print_grid_summary .true.
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


def timed(cmd, cwd, threads):
    """Run cmd; return (stdout, wall seconds, peak footprint in bytes or None)."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    if sys.platform == 'darwin':
        wrapped = ['/usr/bin/time', '-l'] + cmd
    elif Path('/usr/bin/time').exists():
        wrapped = ['/usr/bin/time', '-v'] + cmd
    else:
        wrapped = cmd
    t0 = time.monotonic()
    p = subprocess.run(wrapped, cwd=cwd, capture_output=True, text=True, env=env)
    wall = time.monotonic() - t0
    peak = None
    if m := re.search(r'(\d+)\s+peak memory footprint', p.stderr):        # macOS, bytes
        peak = int(m.group(1))
    elif m := re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', p.stderr):  # GNU, kB
        peak = 1024 * int(m.group(1))
    return p.stdout, wall, peak


def run(exe, case, policy, grid, basis, rot, user_extra='', mode='gradient', threads=8):
    coords, charge, mult, scf, x, c = CASES[case]
    lines = []
    for line in coords.splitlines():
        el, *xyz = line.split()
        p = [float(t) for t in xyz]
        if rot is not None:
            p = mat_vec(rot, p)
        lines.append(f"{el} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}")
    extra = '    grid_radial_points 135\n' if policy == 'g4-3-v4' else ''  # V4 is qualified only at 135 shells
    extra += ''.join(f'    {l.strip()}\n' for l in user_extra.split(';') if l.strip())
    calc = {'gradient': 'gradient', 'energy': 'energy', 'frequency': 'frequency'}[mode]
    text = TEMPLATE.format(calc=calc, extra=extra, basis=basis, scf=scf, grid=grid, policy=policy, x=x, c=c,
                           n=len(lines), charge=charge, mult=mult, coords="\n".join(lines))
    with tempfile.TemporaryDirectory() as d:
        inp = Path(d) / 'in.hfinp'
        inp.write_text(text)
        out, wall, peak = timed([str(exe), str(inp)], d, threads)
    energy = re.findall(r'(?:DFT|Total) Energy\s*:\s*(-?\d+\.\d+)', out)
    points = re.search(r'DFT Grid\s*:\s*(\d+) points', out)
    r = {'energy': float(energy[-1]) if energy else float('nan'),
         'points': int(points.group(1)) if points else None, 'wall': wall, 'peak': peak}
    if mode == 'gradient':
        grad = [[float(t) for t in m] for m in re.findall(r'Atom\s+\d+:\s+(\S+)\s+(\S+)\s+(\S+)', out)]
        if len(grad) != len(lines):
            raise RuntimeError(f"{case}/{policy}@{grid}: no gradient parsed\n{out[-2000:]}")
        r['grad'] = [mat_t_vec(rot, g) for g in grad] if rot is not None else grad
    elif mode == 'frequency':
        table = out.split('Frequency (cm⁻¹)', 1)
        freqs = []
        if len(table) == 2:
            for m in re.finditer(r'^\[INF\]\s+\d+\s+(\d+\.\d+)(i?)', table[1], re.M):
                freqs.append(-float(m.group(1)) if m.group(2) else float(m.group(1)))
        if not freqs:
            raise RuntimeError(f"{case}/{policy}@{grid}: no frequencies parsed\n{out[-2000:]}")
        r['freqs'] = freqs
    if not energy:
        raise RuntimeError(f"{case}/{policy}@{grid}: no energy parsed\n{out[-2000:]}")
    return r


def measure(exe, case, policy, grid, basis, extra, mode, threads):
    """Input orientation plus the three rotations; noise against the input orientation."""
    ref = run(exe, case, policy, grid, basis, None, extra, mode, threads)
    runs = [ref] + [run(exe, case, policy, grid, basis, rot, extra, mode, threads) for rot in ROTATIONS]
    row = {'case': case, 'policy': policy, 'grid': grid, 'mode': mode,
           'natoms': len(CASES[case][0].splitlines()), 'points': ref['points'],
           'wall_mean_s': sum(r['wall'] for r in runs) / len(runs),
           'peak_max_bytes': max((r['peak'] or 0) for r in runs) or None,
           'max_dE': max(abs(r['energy'] - ref['energy']) for r in runs)}
    if mode == 'gradient':
        row['max_dg'] = max(abs(a - b) for r in runs for ga, gb in zip(r['grad'], ref['grad']) for a, b in zip(ga, gb))
    if mode == 'frequency':
        row['max_dnu'] = max(abs(a - b) for r in runs for a, b in zip(r['freqs'], ref['freqs']))
        row['freqs'] = ref['freqs']
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--executable', type=Path, default=Path('build/planck-dft'))
    ap.add_argument('--cases', nargs='+', default=[c for c in CASES if c != 'water12_b3lyp'], choices=list(CASES))
    ap.add_argument('--policies', nargs='+', default=['legacy', 'g4-3-v4'])
    ap.add_argument('--grid', default='normal')
    ap.add_argument('--settings', nargs='+', metavar='POLICY@GRID',
                    help='explicit grid settings; overrides --policies/--grid')
    ap.add_argument('--basis', default='6-31g*')
    ap.add_argument('--threads', type=int, default=8)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--energy-only', action='store_true', help='skip gradients (max|dg| not reported)')
    group.add_argument('--frequency', action='store_true', help='frequencies instead of gradients')
    ap.add_argument('--extra', default='', help='extra %%begin_dft lines, ";"-separated, e.g. "grid_angular_points 302"')
    ap.add_argument('--json', type=Path, help='also write every row to this JSON file')
    args = ap.parse_args()

    exe = args.executable.resolve()
    settings = ([s.split('@', 1) for s in args.settings] if args.settings
                else [[p, args.grid] for p in args.policies])
    mode = 'frequency' if args.frequency else 'energy' if args.energy_only else 'gradient'
    noise = {'gradient': 'max|dg| Ha/Bohr', 'energy': '', 'frequency': 'max|dnu| cm-1'}[mode]

    print(f"{'case':22s} {'policy@grid':28s} {noise:>16s} {'max|dE| Eh':>11s} {'points':>9s} "
          f"{'pts/atom':>8s} {'wall s':>8s} {'peak GB':>8s}", flush=True)
    rows = []
    for case in args.cases:
        for policy, grid in settings:
            row = measure(exe, case, policy, grid, args.basis, args.extra, mode, args.threads)
            rows.append(row)
            col = (f"{row['max_dg']:16.3e}" if mode == 'gradient' else
                   f"{row['max_dnu']:16.3f}" if mode == 'frequency' else f"{'':16s}")
            pts = row['points'] or 0
            peak = f"{row['peak_max_bytes'] / 1e9:8.2f}" if row['peak_max_bytes'] else f"{'n/a':>8s}"
            print(f"{case:22s} {policy + '@' + grid:28s} {col} {row['max_dE']:11.3e} {pts:9d} "
                  f"{pts // row['natoms']:8d} {row['wall_mean_s']:8.2f} {peak}", flush=True)
            if args.json:
                args.json.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
