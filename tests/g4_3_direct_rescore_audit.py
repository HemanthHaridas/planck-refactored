#!/usr/bin/env python3
"""Reconcile a literal G4.3.7 physical grid with its shell-atlas prediction."""
import argparse
import json
import math
from pathlib import Path

from g4_3_rule_grammar import select_band


RAW_CONTRACT = 'g4-fixed-xc-g4-3-direct-v1'
FRONTIER_CONTRACT = 'g4-3-combined-frontier-v1'
CONTRACT = 'g4-3-direct-rescore-v1'
TOLERANCE = 1e-9
FINGERPRINT = 'g4-3-l1-finalist-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg'
REVISED_FINGERPRINT = 'g4-3-second-row-band1-floor-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg'
DUAL_FLOOR_FINGERPRINT = 'g4-3-dual-energy-floor-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg'
V3_FINGERPRINT = 'g4-3-v3-constrained-selection-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg'
V4_FINGERPRINT = 'g4-3-v4-robust-development-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg'
RETIRED_SECOND_ROW = {'boundaries': [.25, 1., 2.5, 4.5], 'lebedev_points': [50, 434, 1454, 590, 170]}
REVISED_SECOND_ROW = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 770, 1454, 590, 170]}
DUAL_FLOOR_SECOND_ROW = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [86, 770, 1730, 590, 170]}
UNCHANGED_HYDROGEN_HELIUM = {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 434]}
DUAL_FLOOR_HYDROGEN_HELIUM = {'boundaries': [.4, 1., 2.5, 3.5], 'lebedev_points': [26, 110, 1454, 770, 590]}
V3_HYDROGEN_HELIUM = {'boundaries': [.4, 1., 3.5, 4.5], 'lebedev_points': [26, 146, 1454, 434, 434]}
V3_SECOND_ROW = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [110, 434, 1454, 590, 170]}
UNCHANGED_THIRD_ROW = {'boundaries': [.1, .5, 3.5, 4.5], 'lebedev_points': [26, 146, 770, 230, 26]}
V4_HYDROGEN_HELIUM = {'boundaries': [.25, 1., 2.5, 3.5], 'lebedev_points': [26, 146, 2030, 974, 590]}
V4_SECOND_ROW = {'boundaries': [.4, 1., 2.5, 4.5], 'lebedev_points': [110, 770, 2030, 590, 170]}
V4_THIRD_ROW = {'boundaries': [.4, .8, 2.5, 4.5], 'lebedev_points': [86, 590, 974, 590, 26]}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _family(z):
    if 1 <= z <= 2:
        return 'hydrogen-helium'
    if 3 <= z <= 10:
        return 'second-row'
    if 11 <= z <= 18:
        return 'third-row'
    raise ValueError('G4.3 direct candidate supports atomic numbers 1 through 18 only')


def load_finalist(path):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == FRONTIER_CONTRACT, 'unsupported combined-frontier contract')
    frontier = value.get('frontier', [])
    _require(len(frontier) == 1 and set(frontier[0]) == {'family_rules', 'cost'},
             'G4.3.8 requires exactly one combined finalist')
    rules = frontier[0]['family_rules']
    _require(set(rules) == {'hydrogen-helium', 'second-row', 'third-row'}, 'incomplete finalist family rules')
    return rules


def expected_fingerprint(rules):
    second = rules.get('second-row')
    hydrogen = rules.get('hydrogen-helium')
    if hydrogen == UNCHANGED_HYDROGEN_HELIUM and second == RETIRED_SECOND_ROW:
        _require(rules.get('third-row') == UNCHANGED_THIRD_ROW,
                 'retired direct rescore requires the frozen third-row rule')
        return FINGERPRINT
    if hydrogen == UNCHANGED_HYDROGEN_HELIUM and second == REVISED_SECOND_ROW:
        _require(rules.get('third-row') == UNCHANGED_THIRD_ROW,
                 'revised direct rescore requires the frozen third-row rule')
        return REVISED_FINGERPRINT
    if hydrogen == DUAL_FLOOR_HYDROGEN_HELIUM and second == DUAL_FLOOR_SECOND_ROW:
        _require(rules.get('third-row') == UNCHANGED_THIRD_ROW,
                 'dual-floor direct rescore requires the frozen third-row rule')
        return DUAL_FLOOR_FINGERPRINT
    if hydrogen == V3_HYDROGEN_HELIUM and second == V3_SECOND_ROW:
        _require(rules.get('third-row') == UNCHANGED_THIRD_ROW,
                 'V3 direct rescore requires the frozen third-row rule')
        return V3_FINGERPRINT
    if (hydrogen == V4_HYDROGEN_HELIUM and second == V4_SECOND_ROW and
            rules.get('third-row') == V4_THIRD_ROW):
        return V4_FINGERPRINT
    raise ValueError('direct rescore has no literal builder for this family-rule combination')


def _direct_directions(candidate):
    directions = candidate.get('directions', [])
    _require(len(directions) == 2, 'direct candidate lacks two AO directions')
    return directions


def _record_direction(record, ordinal):
    directions = record.get('direction_records', [])
    _require(len(directions) == 2 and directions[ordinal].get('index') == ordinal,
             'invalid shell direction record')
    channels = directions[ordinal].get('spin_channels', [])
    _require(len(channels) == 2, 'shell direction lacks alpha/beta channels')
    return dict(D_V=sum(channel['D_V'] for channel in channels),
                D_fxc_D=sum(channel['D_fxc_D'] for channel in channels))


def reconcile(raw, residual_table, rules):
    """Return the direct-minus-atlas ledger after proving identical selection."""
    _require(raw.get('contract') == RAW_CONTRACT, 'invalid direct-rescore raw contract')
    _require(raw.get('radial_points') == 135 and raw.get('directions') == 2,
             'direct candidate must use the frozen 135-shell parent contract')
    _require(raw.get('fingerprint') == expected_fingerprint(rules),
             'direct candidate fingerprint differs from the certified builder')
    candidate = raw.get('candidate', {})
    _direct_directions(candidate)
    shells = candidate.get('shells', [])
    _require(shells, 'direct candidate has no shell selection')
    references = {}
    for record in residual_table.get('records', []):
        if record.get('lebedev_points') == 2030:
            key = record['atom'], record['atomic_number'], record['radial_index']
            _require(key not in references, 'duplicate atlas reference shell')
            references[key] = record
    residuals = {(tuple(item['key']), item['lebedev_points']): item
                 for item in residual_table.get('residuals', [])}
    predicted_energy = 0.0
    predicted_directions = [dict(D_V=0.0, D_fxc_D=0.0) for _ in range(2)]
    total_points = 0
    previous_end = 0
    for shell in shells:
        key = shell['atom'], shell['atomic_number'], shell['radial_index']
        reference = references.get(key)
        _require(reference is not None, 'direct shell is absent from the 2030 atlas')
        _require(abs(shell['u'] - reference['u']) <= 1e-14, 'direct shell M4/Bragg coordinate differs from atlas')
        rule = rules[_family(shell['atomic_number'])]
        expected = rule['lebedev_points'][select_band(shell['u'], rule['boundaries'])]
        _require(shell['lebedev_points'] == expected, 'direct shell selector differs from certified finalist')
        _require(shell['begin'] == previous_end and shell['end'] - shell['begin'] == expected,
                 'direct shell offsets do not form the selected physical grid')
        previous_end = shell['end']
        total_points += expected
        residual = None if expected == 2030 else residuals.get((key, expected))
        _require(expected == 2030 or residual is not None, 'selected shell has no atlas residual')
        predicted_energy += reference['xc_energy'] + (0.0 if residual is None else residual['xc_energy'])
        for ordinal in range(2):
            reference_direction = _record_direction(reference, ordinal)
            predicted_directions[ordinal]['D_V'] += reference_direction['D_V'] + (0.0 if residual is None else residual['directions'][ordinal]['D_V'])
            predicted_directions[ordinal]['D_fxc_D'] += reference_direction['D_fxc_D'] + (0.0 if residual is None else residual['directions'][ordinal]['D_fxc_D'])
    _require(previous_end == candidate.get('points') == total_points, 'direct point count does not equal literal shell count')
    direct_directions = _direct_directions(candidate)
    delta = dict(xc_energy=candidate['xc_energy'] - predicted_energy, directions=[])
    for direct, expected in zip(direct_directions, predicted_directions):
        delta['directions'].append(dict(D_V=direct['D_V'] - expected['D_V'],
                                        D_fxc_D=direct['D_fxc_D'] - expected['D_fxc_D']))
    values = [abs(delta['xc_energy'])]
    values.extend(abs(direction[field]) for direction in delta['directions'] for field in ('D_V', 'D_fxc_D'))
    _require(all(math.isfinite(value) for value in values), 'non-finite direct-rescore delta')
    return dict(contract=CONTRACT, fingerprint=raw.get('fingerprint'), points=total_points,
                atlas=dict(xc_energy=predicted_energy, directions=predicted_directions),
                direct=dict(xc_energy=candidate['xc_energy'], directions=direct_directions), delta=delta,
                tolerance=TOLERANCE, max_error=max(values), passed=max(values) <= TOLERANCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--residual-report', type=Path, required=True)
    parser.add_argument('--combined-frontier', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(json.loads(args.raw.read_text()), json.loads(args.residual_report.read_text()),
                       load_finalist(args.combined_frontier))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_DIRECT_RESCORE', json.dumps(dict(points=result['points'], max_error=result['max_error'],
                                                  passed=result['passed']), separators=(',', ':')))
    if not result['passed']:
        raise SystemExit('FAIL G4.3 direct rescore')
    print('PASS G4.3 direct rescore')


if __name__ == '__main__':
    main()
