#!/usr/bin/env python3
"""Enumerate one v3-constrained family frontier on the frozen selection atlas."""
import argparse
import json
from itertools import combinations
from pathlib import Path

from g4_3_derivation_contract import load
from g4_3_rule_certificate import (boundary_cost_weights, compact_frontier_from_representatives,
                                    family_cost_model, prepare_residual_index,
                                    qualifying_for_boundaries, weighted_rule_cost)
from g4_3_rule_cost import load_reports
from g4_3_v3_constraints import filter_rules, load as load_constraints


CONTRACT = 'g4-3-v3-constrained-family-frontier-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('hydrogen-helium', 'second-row'), required=True)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--constraints', type=Path, default=Path(
        'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_constraints.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    constraints = load_constraints(args.constraints, request)
    tables = load_reports(args.reports, request)
    index = prepare_residual_index(tables)
    model = family_cost_model(args.family, tables, request)
    representatives = {}
    boundary_sets = list(combinations(request['allowed_band_boundaries'], 4))
    for ordinal, boundaries in enumerate(boundary_sets, start=1):
        weights = boundary_cost_weights(boundaries, model)
        rules = qualifying_for_boundaries(args.family, boundaries, tables, request, index)
        for rule in filter_rules(rules, args.family, request, constraints):
            representatives.setdefault(weighted_rule_cost(rule, weights), rule)
        if ordinal % 25 == 0 or ordinal == len(boundary_sets):
            print('G4_3_V3_SELECTION_PROGRESS', json.dumps({
                'family': args.family, 'boundaries_done': ordinal,
                'boundaries_total': len(boundary_sets), 'unique_costs': len(representatives)}), flush=True)
    result = {'contract': CONTRACT, 'status': 'selection-atlas-only', 'family': args.family,
              'constraints': constraints, 'reports': [str(path) for path in args.reports],
              'frontier': compact_frontier_from_representatives(representatives)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_V3_CONSTRAINED_FAMILY_FRONTIER', json.dumps({
        'family': args.family, 'frontier': len(result['frontier'])}))


if __name__ == '__main__':
    main()
