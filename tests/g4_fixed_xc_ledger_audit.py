#!/usr/bin/env python3
"""G4.2 stationary-density XC quadrature audit; never a production workflow.

Runs one parent UKS double-hybrid calculation at R1, then asks the U10-only
front end to evaluate its converged AO density and two frozen ov directions on
all G3 policies plus uniform R1/R2.  It reports candidate residuals against
R2 without treating an XC-energy cancellation as a response pass.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from udh_molecular_audit import FIXTURES, with_angular_points, with_grid_policy, with_radial_points


R2_ENERGY_LIMIT = 1e-8
R2_CONTRACTION_LIMIT = 1e-7
EXPECTED_ROWS = ('legacy', 'legacy-no-light-reduction', 'legacy-unpruned-max',
                 'pyscf-nwchem-2.13', 'uniform-1454-r1', 'uniform-2030-r2')


def finite_row(row):
    values = (row['xc_energy'], row['alpha_norm'], row['beta_norm'],
              *(item['D_V'] for item in row['directions']),
              *(item['D_fxc_D'] for item in row['directions']))
    return len(row['directions']) == 2 and all(math.isfinite(value) for value in values)


def delta(row, reference):
    return dict(
        energy=row['xc_energy'] - reference['xc_energy'],
        alpha_norm=row['alpha_norm'] - reference['alpha_norm'],
        beta_norm=row['beta_norm'] - reference['beta_norm'],
        directions=[dict(D_V=item['D_V'] - ref['D_V'],
                         D_fxc_D=item['D_fxc_D'] - ref['D_fxc_D'])
                    for item, ref in zip(row['directions'], reference['directions'])])


def band_deltas(candidate, reference):
    result = []
    for direction, ref_direction in zip(candidate['directions'], reference['directions']):
        candidate_bands = direction.get('bands', [])
        reference_bands = ref_direction.get('bands', [])
        if len(candidate_bands) != len(reference_bands):
            raise ValueError('incompatible fixed-XC band decomposition')
        items = []
        for band, ref_band in zip(candidate_bands, reference_bands):
            signature = ('atom', 'diagnostic_angular_points')
            if any(band.get(key) != ref_band.get(key) for key in signature):
                raise ValueError('misaligned fixed-XC physical band decomposition')
            items.append(dict(atom=band['atom'], angular_points=band['diagnostic_angular_points'],
                              first_shell=band['first_shell'], last_shell=band['last_shell'],
                              D_V=band['D_V'] - ref_band['D_V'],
                              D_fxc_D=band['D_fxc_D'] - ref_band['D_fxc_D']))
        result.append(items)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--fixture', type=Path,
                        default=FIXTURES / 'ho2_asymmetric_b2plyp_sto3g.hfinp')
    parser.add_argument('--parent-radial-points', type=int, default=135)
    parser.add_argument('--parent-angular-points', type=int, default=1454)
    parser.add_argument('--report-path', type=Path,
                        help='optional durable destination for the normalized results.json report')
    args = parser.parse_args()
    if args.parent_radial_points <= 0 or args.parent_angular_points < 6:
        parser.error('parent grid counts must be positive supported controls')

    root = Path(tempfile.mkdtemp(prefix='g4-fixed-xc-', dir='/private/tmp'))
    rendered = with_grid_policy(
        with_radial_points(with_angular_points(args.fixture.read_text(), args.parent_angular_points),
                           args.parent_radial_points), 'legacy')
    input_path, result_path, ledger_path = root / 'parent.hfinp', root / 'parent.json', root / 'ledger.json'
    input_path.write_text(rendered)
    environment = dict(os.environ)
    environment.setdefault('BASIS_PATH', str(Path(__file__).resolve().parents[1] / 'basis-sets'))
    completed = subprocess.run([str(args.executable.resolve()), str(input_path), '--json', str(result_path),
                                 '--fixed-xc-ledger', str(ledger_path)], env=environment)
    if completed.returncode:
        print('Artifacts:', root, flush=True)
        return completed.returncode
    ledger = json.loads(ledger_path.read_text())
    rows = ledger.get('rows', [])
    names = tuple(row.get('name') for row in rows)
    if ledger.get('contract') != 'g4-fixed-stationary-density-v1' or names != EXPECTED_ROWS or not all(finite_row(row) for row in rows):
        raise ValueError('invalid stationary fixed-XC ledger contract')
    reference = rows[-1]
    report = dict(contract=ledger['contract'], fixture=str(args.fixture), parent_grid=dict(
        radial=args.parent_radial_points, angular=args.parent_angular_points),
        r2_energy_limit=R2_ENERGY_LIMIT, r2_contraction_limit=R2_CONTRACTION_LIMIT,
        rows=[dict(name=row['name'], points=row['points'], r2_delta=delta(row, reference)) for row in rows])
    report['pyscf_r2_band_deltas'] = band_deltas(rows[3], reference)
    r1 = report['rows'][-2]['r2_delta']
    r1_values = [abs(r1['energy']), *(abs(item['D_V']) for item in r1['directions']),
                 *(abs(item['D_fxc_D']) for item in r1['directions'])]
    report['reference_stable'] = (abs(r1['energy']) <= R2_ENERGY_LIMIT and
                                  max(r1_values[1:]) <= R2_CONTRACTION_LIMIT)
    rendered_report = json.dumps(report, indent=2) + '\n'
    (root / 'results.json').write_text(rendered_report)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered_report)
    print('Artifacts:', root)
    print('G4_FIXED_XC', json.dumps(report, separators=(',', ':')))
    print(('PASS' if report['reference_stable'] else 'FAIL') + ' G4 fixed-density reference')
    return 0 if report['reference_stable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
