#!/usr/bin/env python3
"""Validate the immutable G4.3.9 fixed-density holdout request."""
import json
from pathlib import Path


CONTRACT = 'g4-3-fixed-density-holdout-v1'
FRESH_CONTRACT = 'g4-3-fixed-density-holdout-v2'
V3_CONTRACT = 'g4-3-fixed-density-holdout-v3'
V4_CONTRACT = 'g4-3-fixed-density-holdout-v4'
SUPPORTED_CONTRACTS = {CONTRACT, FRESH_CONTRACT, V3_CONTRACT, V4_CONTRACT}
FAMILIES = {'hydrogen-helium', 'second-row', 'third-row'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    path = Path(path)
    value = json.loads(path.read_text())
    _require(value.get('contract') in SUPPORTED_CONTRACTS, 'unsupported G4.3 holdout contract')
    required_keys = {'contract', 'reference', 'limits', 'finalist', 'cases'}
    if value.get('contract') == V4_CONTRACT:
        required_keys.add('fixture_hashes')
    _require(set(value) == required_keys, 'unexpected holdout keys')
    _require(value['reference'] == {'radial_points': 135, 'angular_points': 2030},
             'holdout must retain the frozen 135:2030 reference')
    _require(value['limits'] == {'xc_energy_ha': 1e-8, 'xc_response_contraction': 1e-7},
             'holdout may not change the G4.3 limits')
    root = path.parents[5]
    finalist = root / value['finalist']
    _require(finalist.is_file(), 'holdout finalist is missing')
    cases = value['cases']
    expected_cases = 6 if value['contract'] == V4_CONTRACT else 3
    _require(isinstance(cases, list) and len(cases) == expected_cases,
             f'holdout requires exactly {expected_cases} frozen cases')
    names = set()
    geometry_families = set()
    roles = []
    for case in cases:
        _require(set(case) == {'name', 'input', 'parent', 'families', 'role'}, 'invalid holdout case keys')
        _require(isinstance(case['name'], str) and case['name'] not in names, 'duplicate holdout name')
        names.add(case['name'])
        _require(case['role'] in {'geometry', 'functional-spin'}, 'invalid holdout role')
        roles.append(case['role'])
        _require(isinstance(case['families'], list) and set(case['families']) <= FAMILIES and case['families'],
                 'invalid holdout family coverage')
        input_path, parent_path = root / case['input'], root / case['parent']
        _require(input_path.is_file() and parent_path.is_file(), 'holdout input or parent is missing')
        _require(input_path.read_text() != parent_path.read_text(), 'holdout input is identical to its selection parent')
        if case['role'] == 'geometry':
            geometry_families.update(case['families'])
    expected_geometry = 4 if value['contract'] == V4_CONTRACT else 2
    expected_spin = 2 if value['contract'] == V4_CONTRACT else 1
    _require(roles.count('geometry') == expected_geometry and roles.count('functional-spin') == expected_spin,
             'holdout has an invalid geometry/functional-spin split')
    _require(geometry_families == FAMILIES, 'geometry holdouts must cover every periodic family')
    if value['contract'] == V3_CONTRACT:
        _require(value['finalist'].endswith('g4-3-v3-combined-frontier.json'),
                 'v3 holdout must qualify the v3 combined candidate')
        consumed = [root / 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json',
                    root / 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_holdout_request.json',
                    root / 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_dual_floor_holdout_request.json']
        consumed_text = set()
        for request in consumed:
            prior = json.loads(request.read_text())
            entries = prior['fixtures'] if 'fixtures' in prior else [item['input'] for item in prior['cases']]
            consumed_text.update((root / entry).read_text() for entry in entries)
        for case in cases:
            _require((root / case['input']).read_text() not in consumed_text,
                     'v3 holdout reuses a frozen selection or v1/v2 holdout input')
    if value['contract'] == V4_CONTRACT:
        _require(value['finalist'].endswith('g4-3-v4-combined-frontier.json'),
                 'v4 holdout must qualify the V4 combined candidate')
        import hashlib
        hashes = value['fixture_hashes']
        _require(isinstance(hashes, dict) and set(hashes) == names,
                 'v4 holdout must hash every frozen fixture exactly once')
        for case in cases:
            digest = hashlib.sha256((root / case['input']).read_bytes()).hexdigest()
            _require(hashes[case['name']] == digest, 'v4 holdout fixture hash differs from frozen manifest')
    return value
