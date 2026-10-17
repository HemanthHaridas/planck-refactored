#!/usr/bin/env python3
"""Enumerate one G4.3.9 dual-floor family frontier; test-only."""
import argparse
import json
from itertools import combinations
from pathlib import Path

from g4_3_derivation_contract import load
from g4_3_dual_energy_floor import filter_rules as filter_energy, load as load_energy
from g4_3_rule_certificate import (boundary_cost_weights, compact_frontier_from_representatives,
                                    family_cost_model, prepare_residual_index,
                                    qualifying_for_boundaries, weighted_rule_cost)
from g4_3_rule_cost import load_reports
from g4_3_second_row_floor import filter_rules as filter_response, load as load_response


def constrained_rules(rules, family, request, energy_floor, response_floor=None):
    """Apply the diagnostic energy floor and, for second row, the prior response floor."""
    result = filter_energy(rules, family, request, energy_floor)
    if family == 'second-row':
        if response_floor is None:
            raise ValueError('second-row dual-floor enumeration requires the response-floor contract')
        result = filter_response(result, request, response_floor)
    elif response_floor is not None:
        raise ValueError('the response-floor contract applies only to second-row enumeration')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('hydrogen-helium', 'second-row'), required=True)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--energy-floor', type=Path, required=True)
    parser.add_argument('--response-floor', type=Path,
                        help='required only for second-row, composing the prior 770-point response floor')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    energy_floor = load_energy(args.energy_floor, request)
    response_floor = load_response(args.response_floor, request) if args.response_floor else None
    if args.family == 'second-row' and response_floor is None:
        parser.error('--response-floor is required for second-row')
    if args.family != 'second-row' and response_floor is not None:
        parser.error('--response-floor is valid only for second-row')
    tables = load_reports(args.reports, request)
    index = prepare_residual_index(tables)
    model = family_cost_model(args.family, tables, request)
    representatives = {}
    boundary_sets = list(combinations(request['allowed_band_boundaries'], 4))
    for ordinal, boundaries in enumerate(boundary_sets, start=1):
        weights = boundary_cost_weights(boundaries, model)
        rules = qualifying_for_boundaries(args.family, boundaries, tables, request, index)
        for rule in constrained_rules(rules, args.family, request, energy_floor, response_floor):
            representatives.setdefault(weighted_rule_cost(rule, weights), rule)
        if ordinal % 25 == 0 or ordinal == len(boundary_sets):
            print('G4_3_DUAL_FLOOR_FAMILY_PROGRESS', json.dumps({
                'family': args.family, 'boundaries_done': ordinal,
                'boundaries_total': len(boundary_sets), 'unique_costs': len(representatives)}), flush=True)
    frontier = compact_frontier_from_representatives(representatives)
    result = {'contract': 'g4-3-dual-floor-family-frontier-v1', 'family': args.family,
              'energy_constraint': energy_floor, 'response_constraint': response_floor,
              'frontier': frontier}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_DUAL_FLOOR_FAMILY_FRONTIER', json.dumps({'family': args.family, 'frontier': len(frontier)}))


if __name__ == '__main__':
    main()
