#!/usr/bin/env python3
"""Combine the independently constrained G4.3 v3 family results; test-only.

The consumed v2 holdout is deliberately absent from this operation.  This
adapter accepts only the two v3 selection-atlas frontiers, preserves the
third-row rule from the preceding candidate, and re-certifies the resulting
three-family rule on the original frozen atlas.
"""
import argparse
import json
from pathlib import Path

from g4_3_derivation_contract import load as load_request
from g4_3_revised_combined_frontier import _retired_rules
from g4_3_rule_certificate import certificate, passes
from g4_3_rule_cost import load_reports, rule_cost
from g4_3_v3_constraints import load as load_constraints


FAMILY_CONTRACT = 'g4-3-v3-constrained-family-frontier-v1'
COMBINED_CONTRACT = 'g4-3-combined-frontier-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _family_rule(path, family, request, constraints):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == FAMILY_CONTRACT,
             'unsupported v3 constrained-family frontier')
    _require(value.get('status') == 'selection-atlas-only' and value.get('family') == family,
             'v3 family frontier has the wrong scope or periodic family')
    _require(value.get('constraints') == constraints,
             'v3 family frontier uses a different diagnostic constraint contract')
    rows = value.get('frontier', [])
    _require(len(rows) == 1 and set(rows[0]) ==
             {'max_points_per_atom', 'total_ledger_points', 'rule'},
             'v3 family frontier must have one non-dominated selection rule')
    return rows[0]['rule']


def build(retired_rules, hydrogen_rule, second_row_rule, tables, request):
    rules = dict(retired_rules)
    rules['hydrogen-helium'] = hydrogen_rule
    rules['second-row'] = second_row_rule
    certificates = {family: certificate(rule, family, tables, request)
                    for family, rule in rules.items()}
    _require(all(passes(value, request) for value in certificates.values()),
             'v3 combined rule fails the frozen selection L1 certificate')
    return {'contract': COMBINED_CONTRACT,
            'families': [item['name'] for item in request['families']],
            'frontier': [{'family_rules': rules,
                          'cost': rule_cost(rules, tables, request)}],
            'provenance': ('G4.3 v3 test-only candidate: independently constrained H/He and '
                           'second-row selection-atlas frontiers, with the unchanged third-row '
                           'rule; all three rules re-certified on the frozen selection atlas.'),
            'certificates': certificates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--retired-finalist', type=Path, required=True)
    parser.add_argument('--hydrogen-helium', type=Path, required=True)
    parser.add_argument('--second-row', type=Path, required=True)
    parser.add_argument('--constraints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load_request(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    constraints = load_constraints(args.constraints, request)
    tables = load_reports(args.reports, request)
    result = build(_retired_rules(args.retired_finalist, request),
                   _family_rule(args.hydrogen_helium, 'hydrogen-helium', request, constraints),
                   _family_rule(args.second_row, 'second-row', request, constraints), tables, request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_V3_COMBINED_FRONTIER', json.dumps({
        'cost': result['frontier'][0]['cost'],
        'family_rules': result['frontier'][0]['family_rules']}))


if __name__ == '__main__':
    main()
