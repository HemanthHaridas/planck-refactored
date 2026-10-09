#!/usr/bin/env python3
"""Enumerate the diagnostic-only v3 H/He frontier from consumed holdout evidence."""
import argparse
import json
from itertools import combinations
from pathlib import Path

from g4_3_derivation_contract import load as load_derivation
from g4_3_rule_certificate import (boundary_cost_weights,
                                    compact_frontier_from_representatives,
                                    family_cost_model, prepare_residual_index,
                                    qualifying_for_boundaries, weighted_rule_cost)
from g4_3_v3_diagnostic import load as load_diagnostic


CONTRACT = 'g4-3-v3-hydrogen-diagnostic-frontier-v1'


def diagnostic_frontier(table, derivation, boundary_sets):
    """Return non-dominated diagnostic rules; never a selection frontier."""
    index = prepare_residual_index([table])
    model = family_cost_model('hydrogen-helium', [table], derivation)
    representatives = {}
    for boundaries in boundary_sets:
        weights = boundary_cost_weights(boundaries, model)
        for rule in qualifying_for_boundaries('hydrogen-helium', boundaries, [table], derivation, index):
            representatives.setdefault(weighted_rule_cost(rule, weights), rule)
    return compact_frontier_from_representatives(representatives)


def add_boundary(representatives, table, derivation, index, model, boundaries):
    """Accumulate one boundary tuple without rescoring prior tuples."""
    weights = boundary_cost_weights(boundaries, model)
    for rule in qualifying_for_boundaries('hydrogen-helium', boundaries, [table], derivation, index):
        representatives.setdefault(weighted_rule_cost(rule, weights), rule)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=Path(
        'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v3_diagnostic_request.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnostic = load_diagnostic(args.request)
    root = args.request.parents[5]
    derivation = load_derivation(root / 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    table = json.loads((root / diagnostic['failed_holdout']['residual_report']).read_text())
    if table.get('contract') != 'g4-3-shell-residuals-v1':
        raise ValueError('v3 diagnostic residual report has an unsupported contract')
    if table.get('fixture') != diagnostic['failed_holdout']['fixture']:
        raise ValueError('v3 diagnostic residual report belongs to a different fixture')
    boundary_sets = list(combinations(derivation['allowed_band_boundaries'], 4))
    # The periodic progress record keeps the long all-boundary diagnostic
    # observable without ever treating its result as selection evidence.
    index = prepare_residual_index([table])
    model = family_cost_model('hydrogen-helium', [table], derivation)
    representatives = {}
    for ordinal, boundaries in enumerate(boundary_sets, start=1):
        add_boundary(representatives, table, derivation, index, model, boundaries)
        if ordinal % 25 == 0 or ordinal == len(boundary_sets):
            frontier = compact_frontier_from_representatives(representatives)
            print('G4_3_V3_HYDROGEN_PROGRESS', json.dumps({
                'boundaries_done': ordinal, 'boundaries_total': len(boundary_sets),
                'frontier': len(frontier)}), flush=True)
    frontier = compact_frontier_from_representatives(representatives)
    result = {'contract': CONTRACT, 'status': 'diagnostic-only',
              'source_request': str(args.request),
              'failed_holdout': diagnostic['failed_holdout'],
              'boundaries_examined': len(boundary_sets), 'frontier': frontier,
              'forbids': diagnostic['forbids']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_V3_HYDROGEN_FRONTIER', json.dumps({'frontier': len(frontier)}))


if __name__ == '__main__':
    main()
