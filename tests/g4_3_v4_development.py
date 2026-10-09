#!/usr/bin/env python3
"""Validate the consumed-evidence manifest for the G4.3 V4 development bank."""
import json
from pathlib import Path


CONTRACT = 'g4-3-v4-development-bank-v1'
SOURCES = {
    'v1': 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_holdout_request.json',
    'v2': 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_dual_floor_holdout_request.json',
    'v3': 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_holdout_request.json',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    path = Path(path)
    root = path.parents[5]
    value = json.loads(path.read_text())
    _require(set(value) == {'contract', 'status', 'selection_request', 'cases', 'forbids'},
             'unexpected V4 development-bank keys')
    _require(value['contract'] == CONTRACT and value['status'] == 'development-only',
             'unsupported V4 development-bank contract')
    _require(value['selection_request'] ==
             'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json',
             'V4 development bank must retain the original selection atlas')
    _require(set(value['forbids']) == {'blind-qualification', 'production-policy-promotion',
                                       'per-case-rule-patching'},
             'V4 development-bank scope is broadened')
    expected = {}
    for label, request_path in SOURCES.items():
        request = json.loads((root / request_path).read_text())
        expected.update({case['name']: (label, case['input']) for case in request['cases']})
    cases = value['cases']
    _require(len(cases) == len(expected), 'V4 development bank must retain every consumed holdout')
    names = set()
    for case in cases:
        _require(set(case) == {'name', 'source', 'input', 'residual', 'source_residual'},
                 'invalid V4 development-bank case keys')
        _require(case['name'] not in names, 'duplicate V4 development-bank case')
        names.add(case['name'])
        _require(case['name'] in expected and expected[case['name']] == (case['source'], case['input']),
                 'V4 development case differs from its consumed source request')
        _require((root / case['input']).is_file(), 'V4 development input is missing')
        _require(case['residual'].startswith(
                 'tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03/v4_development/'),
                 'V4 residual destination is outside the durable development bank')
        if case['source_residual'] is not None:
            _require((root / case['source_residual']).is_file(), 'declared source residual is missing')
    _require(names == set(expected), 'V4 development bank omits a consumed holdout')
    return value
