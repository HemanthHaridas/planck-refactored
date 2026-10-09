#!/usr/bin/env python3
"""Apply only the independently supported G4.3 v3 diagnostic constraints."""
import json
from pathlib import Path

from g4_3_rule_grammar import validate_rule


CONTRACT = 'g4-3-v3-diagnostic-constraints-v1'
FORBIDS = {'exact-diagnostic-rule', 'selection-atlas-augmentation',
           'production-policy-promotion', 'v2-holdout-qualification'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, request):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == CONTRACT, 'unsupported v3 constraint contract')
    _require(set(value) == {'contract', 'status', 'hydrogen_helium', 'second_row', 'forbids'},
             'unexpected v3 constraint keys')
    _require(value['status'] == 'test-only', 'v3 constraints must remain test-only')
    _require(set(value['forbids']) == FORBIDS, 'v3 constraints may not broaden diagnostic scope')
    h = value['hydrogen_helium']
    _require(h == {'minimum_third_boundary': 3.5, 'interval': [0.4, 1.0],
                   'minimum_order': 146}, 'invalid v3 H/He constraint')
    second = value['second_row']
    _require(second == {'interval': [0.0, 0.4], 'minimum_order': 110},
             'invalid v3 second-row core constraint')
    _require(3.5 in request['allowed_band_boundaries'], 'v3 H/He boundary is not permitted')
    _require(146 in request['allowed_lebedev_points'] and 110 in request['allowed_lebedev_points'],
             'v3 required Lebedev orders are not permitted')
    return value


def _overlaps(lower, upper, interval):
    return lower < interval[1] and interval[0] < upper


def _interval_floor(rule, interval, minimum_order):
    boundaries, orders = rule['boundaries'], rule['lebedev_points']
    lower = 0.0
    for index, upper in enumerate(boundaries):
        if _overlaps(lower, upper, interval) and orders[index] < minimum_order:
            return False
        lower = upper
    return not _overlaps(lower, float('inf'), interval) or orders[-1] >= minimum_order


def accepts(rule, family, request, constraints):
    validate_rule(rule, request)
    if family == 'hydrogen-helium':
        h = constraints['hydrogen_helium']
        return (rule['boundaries'][2] >= h['minimum_third_boundary'] and
                _interval_floor(rule, h['interval'], h['minimum_order']))
    if family == 'second-row':
        second = constraints['second_row']
        return _interval_floor(rule, second['interval'], second['minimum_order'])
    raise ValueError('v3 constraints apply only to hydrogen-helium and second-row families')


def filter_rules(rules, family, request, constraints):
    return (rule for rule in rules if accepts(rule, family, request, constraints))
