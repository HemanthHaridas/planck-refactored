#!/usr/bin/env python3
"""Composable G4.3.9 energy-floor constraint from the consumed UKS/PBE diagnostic."""
import json
from pathlib import Path

from g4_3_rule_grammar import validate_rule


CONTRACT = 'g4-3-dual-energy-floor-v1'
STATUS = 'test-only-revision-after-g4-3-9-uks-pbe-energy-failure'
FLOORS = (
    ('hydrogen-helium', 4, 590),
    ('second-row', 2, 1730),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, request):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == CONTRACT, 'unsupported dual energy-floor contract')
    _require(set(value) == {'contract', 'source_revision', 'floors', 'status'},
             'unexpected dual energy-floor keys')
    _require(value['source_revision'] == 'g4-3-second-row-band1-floor-v1',
             'dual energy floor must compose the first response-floor revision')
    _require(value['status'] == STATUS, 'dual energy floor may not be presented as production policy')
    received = tuple((item.get('family'), item.get('band_index'), item.get('minimum_lebedev_points'))
                     for item in value['floors'])
    _require(received == FLOORS, 'dual energy floors differ from the frozen diagnostic minima')
    allowed = set(request['allowed_lebedev_points'])
    _require(all(level in allowed for _, _, level in FLOORS), 'energy floor is not an allowed Lebedev order')
    return value


def accepts(rule, family, request, floor):
    """Return whether one normal-form family rule meets its applicable energy floor."""
    validate_rule(rule, request)
    _require(floor['contract'] == CONTRACT, 'invalid loaded dual energy-floor contract')
    requirements = [(band, level) for candidate, band, level in FLOORS if candidate == family]
    _require(len(requirements) == 1, 'family is outside the dual energy-floor scope')
    band, level = requirements[0]
    return rule['lebedev_points'][band] >= level


def filter_rules(rules, family, request, floor):
    for rule in rules:
        if accepts(rule, family, request, floor):
            yield rule
