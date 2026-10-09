#!/usr/bin/env python3
"""Build one G4.3.9-revised combined finalist without mutating G4.3.7.

The failed HO2 holdout changes only the second-row admissible set.  This
test-only adapter carries the frozen G4.3.7 H/He and third-row rules forward,
re-certifies all three family rules against the regenerated selection atlas,
and recomputes the global point cost directly.  Its output intentionally uses
the normal combined-frontier shape so the independent direct-grid rescorer can
consume it unchanged.
"""
import argparse
import json
from pathlib import Path

from g4_3_derivation_contract import load as load_request
from g4_3_rule_certificate import certificate, passes
from g4_3_rule_cost import load_reports, rule_cost
from g4_3_second_row_floor import accepts, load as load_floor


RETIRED_CONTRACT = 'g4-3-combined-frontier-v1'
CONSTRAINED_CONTRACT = 'g4-3-constrained-family-frontier-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _retired_rules(path, request):
    value = json.loads(Path(path).read_text())
    names = [item['name'] for item in request['families']]
    _require(value.get('contract') == RETIRED_CONTRACT, 'unsupported retired combined-frontier contract')
    _require(value.get('families') == names, 'retired frontier family order differs from frozen request')
    rows = value.get('frontier', [])
    _require(len(rows) == 1 and set(rows[0]) == {'family_rules', 'cost'},
             'retired frontier must contain one literal finalist')
    rules = rows[0]['family_rules']
    _require(set(rules) == set(names), 'retired finalist lacks a periodic-family rule')
    return rules


def _constrained_rule(path, request, floor):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == CONSTRAINED_CONTRACT, 'unsupported constrained-family frontier contract')
    _require(value.get('family') == floor['family'] and value.get('constraint') == floor,
             'constrained frontier does not carry the frozen response floor')
    rows = value.get('frontier', [])
    _require(len(rows) == 1 and set(rows[0]) == {'max_points_per_atom', 'total_ledger_points', 'rule'},
             'constrained frontier must contain one non-dominated rule')
    _require(accepts(rows[0]['rule'], request, floor), 'constrained rule violates the response floor')
    return rows[0]['rule']


def build(retired_rules, constrained_rule, tables, request, floor):
    """Re-certify unchanged rules and construct the revised direct finalist."""
    rules = dict(retired_rules)
    rules[floor['family']] = constrained_rule
    certificates = {family: certificate(rule, family, tables, request) for family, rule in rules.items()}
    _require(all(passes(value, request) for value in certificates.values()),
             'a revised combined rule fails the frozen selection L1 certificate')
    _require(accepts(rules[floor['family']], request, floor), 'revised second-row rule violates response floor')
    return dict(contract=RETIRED_CONTRACT,
                families=[item['name'] for item in request['families']],
                frontier=[dict(family_rules=rules, cost=rule_cost(rules, tables, request))],
                provenance=('G4.3.9 test-only revision: frozen H/He and third-row G4.3.7 rules '
                            're-certified on the regenerated selection atlas; second-row rule is '
                            'the unique 770-point-band-1 constrained frontier member.'),
                certificates=certificates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--retired-finalist', type=Path, required=True)
    parser.add_argument('--constrained-second-row', type=Path, required=True)
    parser.add_argument('--floor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load_request(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    floor = load_floor(args.floor, request)
    tables = load_reports(args.reports, request)
    result = build(_retired_rules(args.retired_finalist, request),
                   _constrained_rule(args.constrained_second_row, request, floor), tables, request, floor)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_REVISED_COMBINED_FRONTIER', json.dumps(dict(
        cost=result['frontier'][0]['cost'], second_row=result['frontier'][0]['family_rules']['second-row'])))


if __name__ == '__main__':
    main()
