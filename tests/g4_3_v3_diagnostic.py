#!/usr/bin/env python3
"""Validate the diagnostic-only G4.3 v3 successor request."""
import json
from pathlib import Path


CONTRACT = 'g4-3-v3-diagnostic-request-v1'
LIMITS = {'xc_energy_ha': 1e-8, 'xc_response_contraction': 1e-7}
FORBIDS = {'selection-atlas-augmentation', 'production-policy-promotion',
           'v2-holdout-qualification'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    path = Path(path)
    value = json.loads(path.read_text())
    _require(value.get('contract') == CONTRACT, 'unsupported v3 diagnostic contract')
    _require(set(value) == {'contract', 'status', 'selection_atlas', 'failed_holdout',
                            'candidate', 'limits', 'second_row_core_experiment',
                            'hydrogen_helium', 'forbids'}, 'unexpected v3 diagnostic keys')
    _require(value['status'] == 'diagnostic-only', 'v3 request must remain diagnostic-only')
    _require(value['limits'] == LIMITS, 'v3 request may not relax L1 limits')
    root = path.parents[5]
    atlas = root / value['selection_atlas']
    _require(atlas.is_dir() and len(list(atlas.glob('*.json'))) == 14,
             'v3 request requires the unchanged fourteen-fixture selection atlas')
    _require((root / value['candidate']).is_file(), 'v3 diagnostic candidate is missing')
    holdout = value['failed_holdout']
    _require(set(holdout) == {'fixture', 'residual_report', 'direct_report'},
             'invalid failed-holdout keys')
    _require(all((root / holdout[key]).is_file() for key in holdout),
             'v3 request requires preserved failed-holdout evidence')
    core = value['second_row_core_experiment']
    _require(core == {'u_lower': 0.0, 'u_upper': 0.4, 'current_order': 86,
                      'diagnostic_minimum_order': 110},
             'v3 second-row core experiment must preserve the measured 86-to-110 diagnostic')
    hydrogen = value['hydrogen_helium']
    _require(hydrogen == {'analyze_all_five_bands': True, 'forbid_single_band_floor': True},
             'v3 H/He diagnostic must retain coupled five-band analysis')
    _require(set(value['forbids']) == FORBIDS, 'v3 diagnostic scope may not be broadened')
    return value
