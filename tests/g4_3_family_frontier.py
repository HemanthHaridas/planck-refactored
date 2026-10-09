#!/usr/bin/env python3
"""Stream a compact G4.3.7 family frontier from fixed residual reports."""
import argparse
import json
from itertools import combinations
from pathlib import Path

from g4_3_derivation_contract import load
from g4_3_rule_certificate import (compact_cost_frontier, family_cost_model,
                                    prepare_residual_index, qualifying_for_boundaries,
                                    boundary_cost_weights, weighted_rule_cost,
                                    compact_frontier_from_representatives)
from g4_3_rule_cost import load_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = load(Path(__file__).resolve().parent / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    tables = load_reports(args.reports, request)
    index = prepare_residual_index(tables)
    model = family_cost_model(args.family, tables, request)
    representatives = {}
    boundary_sets = list(combinations(request['allowed_band_boundaries'], 4))
    for ordinal, boundaries in enumerate(boundary_sets, start=1):
        weights = boundary_cost_weights(boundaries, model)
        for rule in qualifying_for_boundaries(args.family, boundaries, tables, request, index):
            representatives.setdefault(weighted_rule_cost(rule, weights), rule)
        if ordinal % 25 == 0 or ordinal == len(boundary_sets):
            print('G4_3_FAMILY_PROGRESS', json.dumps({'family': args.family, 'boundaries_done': ordinal,
                                                       'boundaries_total': len(boundary_sets),
                                                       'unique_costs': len(representatives)}), flush=True)
    frontier = compact_frontier_from_representatives(representatives)
    result = {'contract': 'g4-3-family-frontier-v1', 'family': args.family, 'frontier': frontier}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_FAMILY_FRONTIER', json.dumps({'family': args.family, 'frontier': len(frontier)}))


if __name__ == '__main__':
    main()
