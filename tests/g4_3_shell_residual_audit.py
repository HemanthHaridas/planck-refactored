#!/usr/bin/env python3
"""G4.3.4 multi-Lebedev fixed-density shell-residual audit; test-only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from g4_3_derivation_contract import load as load_request
from g4_3_shell_atlas_contract import family_for_atomic_number, physical_band
from g4_3_shell_residual_contract import CONTRACT, validate
from udh_molecular_audit import FIXTURES, with_angular_points, with_grid_policy, with_radial_points


RAW_CONTRACT = 'g4-fixed-xc-shell-layers-v1'
BAND_BOUNDARIES = [0.25, 0.5, 1.0, 4.5]
RECONSTRUCTION_TOLERANCE = 1e-9


def _legacy_reconstruction(raw, records, direct):
    selections = raw.get('legacy_selection', [])
    if not selections:
        raise ValueError('raw sidecar lacks the legacy shell selection')
    by_key = {(record['atom'], record['atomic_number'], record['radial_index'],
               record['lebedev_points']): record for record in records}
    selected = []
    for item in selections:
        key = (item['atom'], item['atomic_number'], item['radial_index'], item['lebedev_points'])
        if key not in by_key:
            raise ValueError('legacy selection points outside the uniform shell atlas')
        selected.append(by_key[key])
    if len(selected) != len(selections):
        raise ValueError('legacy shell selection is not unique')
    rows = [row for row in direct.get('rows', []) if row.get('name') == 'legacy']
    if len(rows) != 1:
        raise ValueError('direct fixed-density ledger lacks exactly one legacy row')
    row = rows[0]
    directions = []
    for ordinal, direct_value in enumerate(row['directions']):
        value = dict(D_V=sum(record['direction_records'][ordinal]['spin_channels'][0]['D_V'] +
                             record['direction_records'][ordinal]['spin_channels'][1]['D_V']
                             for record in selected),
                     D_fxc_D=sum(record['direction_records'][ordinal]['spin_channels'][0]['D_fxc_D'] +
                                 record['direction_records'][ordinal]['spin_channels'][1]['D_fxc_D']
                                 for record in selected))
        value['D_V_error'] = value['D_V'] - direct_value['D_V']
        value['D_fxc_D_error'] = value['D_fxc_D'] - direct_value['D_fxc_D']
        directions.append(value)
    result = dict(xc_energy=sum(record['xc_energy'] for record in selected),
                  direct_xc_energy=row['xc_energy'], directions=directions,
                  selected_shells=len(selected))
    result['xc_energy_error'] = result['xc_energy'] - result['direct_xc_energy']
    errors = [abs(result['xc_energy_error'])]
    errors.extend(abs(value[field]) for value in directions for field in ('D_V_error', 'D_fxc_D_error'))
    if max(errors) > RECONSTRUCTION_TOLERANCE:
        raise ValueError('legacy selected-shell reconstruction exceeds tolerance')
    return result


def normalize(raw, fixture, request, direct=None, allow_external_fixture=False):
    if raw.get('contract') != RAW_CONTRACT:
        raise ValueError('invalid multi-Lebedev raw contract')
    layers = raw.get('layers', [])
    expected = set(request['allowed_lebedev_points'])
    sizes = [layer.get('lebedev_points') for layer in layers]
    if len(sizes) != len(expected) or set(sizes) != expected:
        raise ValueError('raw layers do not cover the frozen Lebedev set exactly')
    records = []
    for layer in layers:
        for record in layer.get('records', []):
            converted = dict(record)
            converted['family'] = family_for_atomic_number(converted['atomic_number'], request['families'])
            converted['physical_band'] = physical_band(converted['u'], BAND_BOUNDARIES)
            records.append(converted)
    table = dict(contract=CONTRACT, fixture=str(fixture), selector=request['shell_selector'],
                 reference=request['reference_ladder']['angular_anchor'], directions=request['directions_per_fixture'],
                 band_boundaries=BAND_BOUNDARIES, records=records)
    values = validate(table, request, allow_external_fixture)
    table['residuals'] = values
    if direct is not None:
        table['legacy_reconstruction'] = _legacy_reconstruction(raw, records, direct)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--fixture', type=Path,
                        default=FIXTURES / 'ho2_asymmetric_b2plyp_sto3g.hfinp')
    parser.add_argument('--report-path', type=Path, help='durable normalized residual-table destination')
    parser.add_argument('--allow-external-fixture', action='store_true',
                        help='G4.3.9 only: permit an immutable holdout outside the selection request')
    args = parser.parse_args()
    request_path = Path(__file__).resolve().parent / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
    request = load_request(request_path)
    root = Path(tempfile.mkdtemp(prefix='g4-3-shell-residuals-', dir='/private/tmp'))
    rendered = with_grid_policy(with_radial_points(with_angular_points(args.fixture.read_text(), 1454), 135), 'legacy')
    input_path, result_path, raw_path, ledger_path = (root / 'parent.hfinp', root / 'parent.json',
                                                       root / 'residuals.raw.json', root / 'legacy.ledger.json')
    input_path.write_text(rendered)
    environment = dict(os.environ)
    environment.setdefault('BASIS_PATH', str(Path(__file__).resolve().parents[1] / 'basis-sets'))
    completed = subprocess.run([str(args.executable.resolve()), str(input_path), '--json', str(result_path),
                                '--fixed-xc-shell-residuals', str(raw_path)], env=environment)
    if completed.returncode:
        print('Artifacts:', root, flush=True)
        return completed.returncode
    completed = subprocess.run([str(args.executable.resolve()), str(input_path), '--json', str(result_path),
                                '--fixed-xc-ledger', str(ledger_path)], env=environment)
    if completed.returncode:
        print('Artifacts:', root, flush=True)
        return completed.returncode
    report = normalize(json.loads(raw_path.read_text()), args.fixture, request, json.loads(ledger_path.read_text()),
                       args.allow_external_fixture)
    rendered_report = json.dumps(report, indent=2) + '\n'
    (root / 'results.json').write_text(rendered_report)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered_report)
    print('Artifacts:', root)
    print('G4_3_SHELL_RESIDUALS', json.dumps(dict(records=len(report['records']), residuals=len(report['residuals'])),
                                              separators=(',', ':')))
    print('PASS G4.3 shell residuals')


if __name__ == '__main__':
    raise SystemExit(main())
