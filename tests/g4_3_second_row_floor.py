#!/usr/bin/env python3
"""Test-only G4.3 revision constraint after the failed second-row holdout."""
import json
from pathlib import Path

from g4_3_rule_grammar import validate_rule


CONTRACT = 'g4-3-second-row-band1-floor-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, request):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == CONTRACT, 'unsupported second-row floor contract')
    _require(set(value) == {'contract', 'source_finalist', 'family', 'band_index',
                            'minimum_lebedev_points', 'status'}, 'unexpected second-row floor keys')
    _require(value['family'] == 'second-row' and value['band_index'] == 1,
             'floor must apply only to second-row physical band 1')
    _require(value['minimum_lebedev_points'] == 770 and
             value['minimum_lebedev_points'] in request['allowed_lebedev_points'],
             'floor must be the frozen 770-point allowed Lebedev order')
    _require(value['status'] == 'test-only-revision-after-g4-3-9-failure',
             'floor may not be presented as a production policy')
    return value


def accepts(rule, request, floor):
    """Return whether a normal grammar rule also satisfies the repair floor."""
    validate_rule(rule, request)
    _require(floor['family'] == 'second-row' and floor['band_index'] == 1,
             'unsupported revision constraint')
    return rule['lebedev_points'][1] >= floor['minimum_lebedev_points']


def filter_rules(rules, request, floor):
    for rule in rules:
        if accepts(rule, request, floor):
            yield rule
