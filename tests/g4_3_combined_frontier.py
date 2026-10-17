#!/usr/bin/env python3
"""Combine exact G4.3.7 family frontiers into direct, global finalists.

This remains an offline selector: it reads immutable residual reports and
family-frontier artifacts, then proves that their separable cost accounting
equals a direct full-matrix cost evaluation.  It does not construct a grid or
evaluate an XC quantity.
"""
import argparse
import itertools
import json
from pathlib import Path

from g4_3_derivation_contract import load
from g4_3_rule_certificate import family_cost_model, family_rule_cost
from g4_3_rule_cost import load_reports, rule_cost


CONTRACT = 'g4-3-combined-frontier-v1'
FAMILY_CONTRACT = 'g4-3-family-frontier-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_family_frontier(path, family):
    value = json.loads(Path(path).read_text())
    _require(value.get('contract') == FAMILY_CONTRACT, 'unsupported family-frontier contract')
    _require(value.get('family') == family, 'family-frontier path is assigned to the wrong family')
    frontier = value.get('frontier')
    _require(isinstance(frontier, list) and frontier, 'family frontier is empty')
    for row in frontier:
        _require(set(row) == {'max_points_per_atom', 'total_ledger_points', 'rule'},
                 'invalid family frontier row')
    return frontier


def _pareto(rows):
    """Keep one lexical representative for each non-dominated global cost."""
    representatives = {}
    for row in rows:
        cost = row['cost']['max_points_per_atom'], row['cost']['total_ledger_points']
        representatives.setdefault(cost, row)
    best_total = float('inf')
    result = []
    for cost, row in sorted(representatives.items()):
        if cost[1] < best_total:
            result.append(row)
            best_total = cost[1]
    return result


def combine(family_frontiers, tables, request):
    """Make and independently cost every Cartesian product of family finalists."""
    names = [item['name'] for item in request['families']]
    _require(list(family_frontiers) == names, 'family frontiers must follow the frozen family order')
    models = {name: family_cost_model(name, tables, request) for name in names}
    rows = []
    for entries in itertools.product(*(family_frontiers[name] for name in names)):
        family_rules = {name: entry['rule'] for name, entry in zip(names, entries)}
        separable = [family_rule_cost(family_rules[name], models[name]) for name in names]
        for name, entry, cost in zip(names, entries, separable):
            _require((entry['max_points_per_atom'], entry['total_ledger_points']) == cost,
                     f'{name}: frontier cost does not match its literal rule')
        direct = rule_cost(family_rules, tables, request)
        expected = max(cost[0] for cost in separable), sum(cost[1] for cost in separable)
        _require((direct['max_points_per_atom'], direct['total_ledger_points']) == expected,
                 'separable family cost does not equal direct full-matrix cost')
        rows.append({'family_rules': family_rules, 'cost': direct})
    return {'contract': CONTRACT, 'families': names, 'frontier': _pareto(rows),
            'note': ('Fixed-density L1-certified finalist(s) only; G4.3.8 must directly rescore each '
                     'resolved physical grid before a policy can be considered.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--hydrogen-helium', type=Path, required=True)
    parser.add_argument('--second-row', type=Path, required=True)
    parser.add_argument('--third-row', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    tables = load_reports(args.reports, request)
    paths = {'hydrogen-helium': args.hydrogen_helium, 'second-row': args.second_row,
             'third-row': args.third_row}
    frontiers = {name: load_family_frontier(path, name) for name, path in paths.items()}
    result = combine(frontiers, tables, request)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_COMBINED_FRONTIER', json.dumps({'frontier': len(result['frontier']),
                                                 'costs': [row['cost'] for row in result['frontier']]}))


if __name__ == '__main__':
    main()
