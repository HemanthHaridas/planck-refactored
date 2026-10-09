#!/usr/bin/env python3
"""Stream a G4.3.9-revised, floor-constrained family frontier; test-only."""
import argparse
import json
from itertools import combinations
from pathlib import Path

from g4_3_derivation_contract import load
from g4_3_rule_certificate import (prepare_residual_index, qualifying_for_boundaries,
                                    family_cost_model, boundary_cost_weights,
                                    weighted_rule_cost, compact_frontier_from_representatives)
from g4_3_rule_cost import load_reports
from g4_3_second_row_floor import filter_rules, load as load_floor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--floor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    floor = load_floor(args.floor, request)
    if args.family != floor['family']:
        parser.error('the constrained frontier applies only to the floor family')
    tables = load_reports(args.reports, request)
    index = prepare_residual_index(tables)
    model = family_cost_model(args.family, tables, request)
    representatives = {}
    boundary_sets = list(combinations(request['allowed_band_boundaries'], 4))
    for ordinal, boundaries in enumerate(boundary_sets, start=1):
        weights = boundary_cost_weights(boundaries, model)
        candidates = filter_rules(qualifying_for_boundaries(args.family, boundaries, tables, request, index),
                                  request, floor)
        for rule in candidates:
            representatives.setdefault(weighted_rule_cost(rule, weights), rule)
        if ordinal % 25 == 0 or ordinal == len(boundary_sets):
            print('G4_3_CONSTRAINED_FAMILY_PROGRESS', json.dumps({
                'family': args.family, 'boundaries_done': ordinal,
                'boundaries_total': len(boundary_sets), 'unique_costs': len(representatives)}), flush=True)
    frontier = compact_frontier_from_representatives(representatives)
    result = {'contract': 'g4-3-constrained-family-frontier-v1', 'family': args.family,
              'constraint': floor, 'frontier': frontier}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_CONSTRAINED_FAMILY_FRONTIER', json.dumps({'family': args.family, 'frontier': len(frontier)}))


if __name__ == '__main__':
    main()
