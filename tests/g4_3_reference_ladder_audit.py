#!/usr/bin/env python3
"""G4.3.1 fixed-density product-reference audit; never a production workflow.

One parent SCF calculation supplies the AO density and two frozen directions.
The U10-only sidecar evaluates those unchanged objects at A=135:2030,
B=165:2030, and C=165:2702, separating radial from angular-reference error.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from udh_molecular_audit import FIXTURES, with_angular_points, with_grid_policy, with_radial_points


ENERGY_LIMIT = 1e-8
CONTRACTION_LIMIT = 1e-7
CONTRACT = 'g4-fixed-xc-reference-ladder-v1'
EXPECTED_ROWS = ('uniform-2030-a', 'uniform-2030-b', 'uniform-2702-c')


def finite_row(row):
    values = (row['xc_energy'], row['alpha_norm'], row['beta_norm'],
              *(item['D_V'] for item in row['directions']),
              *(item['D_fxc_D'] for item in row['directions']))
    return len(row.get('directions', [])) == 2 and all(math.isfinite(value) for value in values)


def delta(row, reference):
    return dict(
        energy=row['xc_energy'] - reference['xc_energy'],
        alpha_norm=row['alpha_norm'] - reference['alpha_norm'],
        beta_norm=row['beta_norm'] - reference['beta_norm'],
        directions=[dict(D_V=item['D_V'] - ref['D_V'],
                         D_fxc_D=item['D_fxc_D'] - ref['D_fxc_D'])
                    for item, ref in zip(row['directions'], reference['directions'])])


def stable(delta_value):
    contractions = [abs(item[key]) for item in delta_value['directions']
                    for key in ('D_V', 'D_fxc_D')]
    return abs(delta_value['energy']) <= ENERGY_LIMIT and max(contractions) <= CONTRACTION_LIMIT


def normalize(ledger, fixture):
    rows = ledger.get('rows', [])
    if ledger.get('contract') != CONTRACT:
        raise ValueError('invalid G4.3 reference-ladder contract')
    if tuple(row.get('name') for row in rows) != EXPECTED_ROWS or not all(finite_row(row) for row in rows):
        raise ValueError('invalid G4.3 reference-ladder rows')
    a, b, c = rows
    radial_delta = delta(b, a)
    angular_delta = delta(c, b)
    return dict(contract='g4-3-reference-ladder-report-v1', fixture=str(fixture),
                reference_ladder=dict(angular_anchor=dict(radial=135, angular=2030),
                                      radial_check=dict(radial=165, angular=2030),
                                      angular_check=dict(radial=165, angular=2702)),
                energy_limit=ENERGY_LIMIT, contraction_limit=CONTRACTION_LIMIT,
                rows=[dict(name=row['name'], points=row['points']) for row in rows],
                radial_delta=radial_delta, angular_delta=angular_delta,
                radial_stable=stable(radial_delta), angular_stable=stable(angular_delta),
                reference_stable=stable(radial_delta) and stable(angular_delta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--fixture', type=Path,
                        default=FIXTURES / 'ho2_asymmetric_b2plyp_sto3g.hfinp')
    parser.add_argument('--parent-radial-points', type=int, default=135)
    parser.add_argument('--parent-angular-points', type=int, default=1454)
    parser.add_argument('--report-path', type=Path,
                        help='optional durable destination for the normalized reference-ladder report')
    args = parser.parse_args()
    if args.parent_radial_points <= 0 or args.parent_angular_points < 6:
        parser.error('parent grid counts must be positive supported controls')

    root = Path(tempfile.mkdtemp(prefix='g4-3-reference-', dir='/private/tmp'))
    rendered = with_grid_policy(
        with_radial_points(with_angular_points(args.fixture.read_text(), args.parent_angular_points),
                           args.parent_radial_points), 'legacy')
    input_path, result_path, ladder_path = root / 'parent.hfinp', root / 'parent.json', root / 'ladder.json'
    input_path.write_text(rendered)
    environment = dict(os.environ)
    environment.setdefault('BASIS_PATH', str(Path(__file__).resolve().parents[1] / 'basis-sets'))
    completed = subprocess.run([str(args.executable.resolve()), str(input_path), '--json', str(result_path),
                                '--fixed-xc-reference-ladder', str(ladder_path)], env=environment)
    if completed.returncode:
        print('Artifacts:', root, flush=True)
        return completed.returncode
    report = normalize(json.loads(ladder_path.read_text()), args.fixture)
    rendered_report = json.dumps(report, indent=2) + '\n'
    (root / 'results.json').write_text(rendered_report)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered_report)
    print('Artifacts:', root)
    print('G4_3_REFERENCE', json.dumps(report, separators=(',', ':')))
    print(('PASS' if report['reference_stable'] else 'FAIL') + ' G4.3 product reference')
    return 0 if report['reference_stable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
