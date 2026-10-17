#!/usr/bin/env python3
"""G4.3.3 fixed-density 135:2030 shell-atlas audit; never production code."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from g4_3_derivation_contract import load as load_request
from g4_3_shell_atlas_contract import CONTRACT as ATLAS_CONTRACT, family_for_atomic_number, physical_band, validate
from udh_molecular_audit import FIXTURES, with_angular_points, with_grid_policy, with_radial_points


RAW_CONTRACT = 'g4-fixed-xc-shell-layers-v1'
BAND_BOUNDARIES = [0.25, 0.5, 1.0, 4.5]
# A shell sum changes the deterministic matrix-reduction grouping relative to
# one monolithic assembly.  405 HO2 slices leave a sub-1e-9 finite-difference
# kernel roundoff, still two orders below the G4 response qualification gate.
RECONSTRUCTION_TOLERANCE = 1e-9


def close(left, right):
    return abs(left - right) <= RECONSTRUCTION_TOLERANCE * max(1., abs(left), abs(right))


def normalize(raw, fixture, request):
    if raw.get('contract') != RAW_CONTRACT:
        raise ValueError('invalid raw shell-atlas contract')
    layers = [layer for layer in raw.get('layers', []) if layer.get('lebedev_points') == 2030]
    if len(layers) != 1:
        raise ValueError('raw shell atlas requires exactly one 2030-point layer')
    raw = layers[0]
    directions = raw.get('directions', [])
    if len(directions) != request['directions_per_fixture']:
        raise ValueError('raw shell atlas has wrong direction count')
    records = raw.get('records', [])
    if not records:
        raise ValueError('raw shell atlas has no records')
    atlas_records = []
    energy_sum = 0.
    direction_sums = [dict(D_V=0., D_fxc_D=0.) for _ in directions]
    for record in records:
        converted = dict(record)
        z = converted['atomic_number']
        converted['family'] = family_for_atomic_number(z, request['families'])
        converted['physical_band'] = physical_band(converted['u'], BAND_BOUNDARIES)
        atlas_records.append(converted)
        energy_sum += converted['xc_energy']
        for direction in converted['direction_records']:
            target = direction_sums[direction['index']]
            for channel in direction['spin_channels']:
                target['D_V'] += channel['D_V']
                target['D_fxc_D'] += channel['D_fxc_D']
    if not close(energy_sum, raw['xc_energy']):
        raise ValueError('shell XC energy does not reconstruct the monolithic grid')
    for index, (actual, expected) in enumerate(zip(direction_sums, directions)):
        if not close(actual['D_V'], expected['D_V']) or not close(actual['D_fxc_D'], expected['D_fxc_D']):
            raise ValueError(f'shell direction {index} does not reconstruct the monolithic grid')
    atlas = dict(contract=ATLAS_CONTRACT, fixture=str(fixture), selector=request['shell_selector'],
                 reference=request['reference_ladder']['angular_anchor'], directions=request['directions_per_fixture'],
                 band_boundaries=BAND_BOUNDARIES, records=atlas_records,
                 reconstruction=dict(tolerance=RECONSTRUCTION_TOLERANCE, xc_energy=dict(shell_sum=energy_sum,
                     monolithic=raw['xc_energy']), directions=[dict(index=index, shell_sum=value,
                     monolithic=expected) for index, (value, expected) in enumerate(zip(direction_sums, directions))]))
    return validate(atlas, request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--fixture', type=Path,
                        default=FIXTURES / 'ho2_asymmetric_b2plyp_sto3g.hfinp')
    parser.add_argument('--report-path', type=Path, help='durable normalized shell-atlas destination')
    args = parser.parse_args()
    request_path = Path(__file__).resolve().parent / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
    request = load_request(request_path)
    root = Path(tempfile.mkdtemp(prefix='g4-3-shell-atlas-', dir='/private/tmp'))
    rendered = with_grid_policy(with_radial_points(with_angular_points(args.fixture.read_text(), 1454), 135), 'legacy')
    input_path, result_path, raw_path = root / 'parent.hfinp', root / 'parent.json', root / 'atlas.raw.json'
    input_path.write_text(rendered)
    environment = dict(os.environ)
    environment.setdefault('BASIS_PATH', str(Path(__file__).resolve().parents[1] / 'basis-sets'))
    completed = subprocess.run([str(args.executable.resolve()), str(input_path), '--json', str(result_path),
                                '--fixed-xc-shell-atlas', str(raw_path)], env=environment)
    if completed.returncode:
        print('Artifacts:', root, flush=True)
        return completed.returncode
    atlas = normalize(json.loads(raw_path.read_text()), args.fixture, request)
    rendered_atlas = json.dumps(atlas, indent=2) + '\n'
    (root / 'atlas.json').write_text(rendered_atlas)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered_atlas)
    print('Artifacts:', root)
    print('G4_3_SHELL_ATLAS', json.dumps(atlas['reconstruction'], separators=(',', ':')))
    print('PASS G4.3 shell reconstruction')


if __name__ == '__main__':
    raise SystemExit(main())
