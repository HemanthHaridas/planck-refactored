#!/usr/bin/env python3
"""Build one re-certified G4.3.9 dual-floor combined finalist; test-only."""
import argparse
import json
from pathlib import Path

from g4_3_derivation_contract import load as load_request
from g4_3_dual_energy_floor import accepts as accepts_energy, load as load_energy
from g4_3_revised_combined_frontier import _retired_rules
from g4_3_rule_certificate import certificate, passes
from g4_3_rule_cost import load_reports, rule_cost
from g4_3_second_row_floor import accepts as accepts_response, load as load_response


FAMILY_CONTRACT = 'g4-3-dual-floor-family-frontier-v1'
COMBINED_CONTRACT = 'g4-3-combined-frontier-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_family(path, family, request, energy_floor, response_floor):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == FAMILY_CONTRACT and value.get('family') == family,
             'unsupported dual-floor family frontier')
    _require(value.get('energy_constraint') == energy_floor, 'family frontier uses a different energy-floor contract')
    expected_response = response_floor if family == 'second-row' else None
    _require(value.get('response_constraint') == expected_response,
             'family frontier response-floor composition differs')
    rows = value.get('frontier', [])
    _require(len(rows) == 1 and set(rows[0]) == {'max_points_per_atom', 'total_ledger_points', 'rule'},
             'dual-floor family frontier must contain one non-dominated rule')
    rule = rows[0]['rule']
    _require(accepts_energy(rule, family, request, energy_floor), 'family rule violates dual energy floor')
    if family == 'second-row':
        _require(accepts_response(rule, request, response_floor), 'second-row rule violates response floor')
    return rule


def build(hydrogen_rule, second_rule, retired_rules, tables, request, energy_floor, response_floor):
    rules = dict(retired_rules)
    rules['hydrogen-helium'] = hydrogen_rule
    rules['second-row'] = second_rule
    certificates = {family: certificate(rule, family, tables, request) for family, rule in rules.items()}
    _require(all(passes(value, request) for value in certificates.values()),
             'dual-floor combined rule fails frozen selection L1 certificate')
    _require(accepts_energy(hydrogen_rule, 'hydrogen-helium', request, energy_floor),
             'combined H/He rule violates energy floor')
    _require(accepts_energy(second_rule, 'second-row', request, energy_floor) and
             accepts_response(second_rule, request, response_floor),
             'combined second-row rule violates composed floors')
    return dict(contract=COMBINED_CONTRACT,
                families=[item['name'] for item in request['families']],
                frontier=[dict(family_rules=rules, cost=rule_cost(rules, tables, request))],
                provenance=('G4.3.9 test-only dual-floor revision: H/He outer-energy and '
                            'second-row peak-energy floors, composed with the prior second-row '
                            'band-1 response floor; all rules re-certified on the frozen selection atlas.'),
                certificates=certificates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--retired-finalist', type=Path, required=True)
    parser.add_argument('--hydrogen-helium', type=Path, required=True)
    parser.add_argument('--second-row', type=Path, required=True)
    parser.add_argument('--energy-floor', type=Path, required=True)
    parser.add_argument('--response-floor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load_request(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    energy_floor = load_energy(args.energy_floor, request)
    response_floor = load_response(args.response_floor, request)
    tables = load_reports(args.reports, request)
    hhe = load_family(args.hydrogen_helium, 'hydrogen-helium', request, energy_floor, response_floor)
    second = load_family(args.second_row, 'second-row', request, energy_floor, response_floor)
    result = build(hhe, second, _retired_rules(args.retired_finalist, request), tables, request,
                   energy_floor, response_floor)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_DUAL_FLOOR_COMBINED_FRONTIER', json.dumps(dict(cost=result['frontier'][0]['cost'],
                                                                 family_rules=result['frontier'][0]['family_rules'])))


if __name__ == '__main__':
    main()
