#!/usr/bin/env python3
"""Select a V4 grid rule from frozen selection plus consumed development evidence."""
import argparse
import json
from itertools import combinations
from pathlib import Path

from g4_3_derivation_contract import load as load_request
from g4_3_rule_certificate import (band_l1_vectors, boundary_cost_weights, family_cost_model,
                                    prepare_residual_index, weighted_rule_cost)
from g4_3_rule_cost import load_reports, rule_cost
from g4_3_v4_development import load as load_development


CONTRACT = 'g4-3-combined-frontier-v1'
SEARCH_CONTRACT = 'g4-3-v4-search-v1'
ROBUST_RATIO = .5


def _add(left, right):
    return [dict(xc_energy=a['xc_energy'] + b['xc_energy'], directions=[
        dict(D_V=x['D_V'] + y['D_V'], D_fxc_D=x['D_fxc_D'] + y['D_fxc_D'])
        for x, y in zip(a['directions'], b['directions'])]) for a, b in zip(left, right)]


def _zero(tables, directions):
    return [dict(xc_energy=0., directions=[dict(D_V=0., D_fxc_D=0.)
                                            for _ in range(directions)]) for _ in tables]


def normalized_error(rows, request):
    """Maximum channel error normalized by the fixed physical acceptance limit."""
    values = []
    for row in rows:
        values.append(row['xc_energy'] / request['limits']['xc_energy_ha'])
        for direction in row['directions']:
            values.extend((direction['D_V'] / request['limits']['xc_response_contraction'],
                           direction['D_fxc_D'] / request['limits']['xc_response_contraction']))
    return max(values, default=0.)


def _within(rows, request):
    return normalized_error(rows, request) <= 1. + 1e-14


def family_candidate(family, selection, development, request):
    """Choose the least-cost robust family rule using all consumed evidence."""
    evidence = selection + development
    index = prepare_residual_index(evidence)
    model = family_cost_model(family, selection, request)
    best = best_robust = None
    qualified = robust_count = 0
    levels = request['allowed_lebedev_points']
    boundary_sets = list(combinations(request['allowed_band_boundaries'], 4))
    for ordinal, boundaries in enumerate(boundary_sets, start=1):
        vectors = band_l1_vectors(family, boundaries, evidence, request, index)
        zero = _zero(evidence, request['directions_per_fixture'])
        weights = boundary_cost_weights(boundaries, model)
        for peak_index, peak in enumerate(levels):
            for n0_index in range(peak_index + 1):
                r0 = _add(zero, vectors[0, levels[n0_index]])
                if not _within(r0, request):
                    continue
                for n1_index in range(n0_index, peak_index + 1):
                    r2 = _add(_add(r0, vectors[1, levels[n1_index]]), vectors[2, peak])
                    if not _within(r2, request):
                        continue
                    for n3_index in range(peak_index + 1):
                        r3 = _add(r2, vectors[3, levels[n3_index]])
                        if not _within(r3, request):
                            continue
                        for n4_index in range(n3_index + 1):
                            rows = _add(r3, vectors[4, levels[n4_index]])
                            if not _within(rows, request):
                                continue
                            rule = {'boundaries': list(boundaries),
                                    'lebedev_points': [levels[n0_index], levels[n1_index], peak,
                                                       levels[n3_index], levels[n4_index]]}
                            max_points, total_points = weighted_rule_cost(rule, weights)
                            choice = dict(rule=rule, normalized_error=normalized_error(rows, request),
                                          max_points_per_atom=max_points,
                                          total_ledger_points=total_points)
                            quality_key = (choice['normalized_error'], choice['max_points_per_atom'],
                                           choice['total_ledger_points'], choice['rule']['boundaries'],
                                           choice['rule']['lebedev_points'])
                            cost_key = (choice['max_points_per_atom'], choice['total_ledger_points'],
                                        choice['normalized_error'], choice['rule']['boundaries'],
                                        choice['rule']['lebedev_points'])
                            qualified += 1
                            if best is None or quality_key < (best['normalized_error'], best['max_points_per_atom'],
                                                              best['total_ledger_points'], best['rule']['boundaries'],
                                                              best['rule']['lebedev_points']):
                                best = choice
                            if choice['normalized_error'] <= ROBUST_RATIO:
                                robust_count += 1
                                if best_robust is None or cost_key < (best_robust['max_points_per_atom'],
                                                                       best_robust['total_ledger_points'],
                                                                       best_robust['normalized_error'],
                                                                       best_robust['rule']['boundaries'],
                                                                       best_robust['rule']['lebedev_points']):
                                    best_robust = choice
        if ordinal % 25 == 0 or ordinal == len(boundary_sets):
            print('G4_3_V4_PROGRESS', json.dumps(dict(family=family, boundaries_done=ordinal,
                                                       boundaries_total=len(boundary_sets),
                                                       qualified=qualified)), flush=True)
    if best is None:
        raise ValueError(f'no V4 {family} rule satisfies all development evidence')
    return best_robust, dict(qualified=qualified, robust=robust_count,
                             robust_available=best_robust is not None,
                             lowest_cost_robust=best_robust,
                             lowest_error_qualified=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-reports', nargs='+', type=Path, required=True)
    parser.add_argument('--development-request', type=Path, default=Path(
        'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v4_development_request.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = load_request(root / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    selection = load_reports(args.selection_reports, request)
    development_request = load_development(args.development_request)
    repo = args.development_request.parents[5]
    development = [json.loads((repo / case['residual']).read_text()) for case in development_request['cases']]
    selected, diagnostics = {}, {}
    for family in (item['name'] for item in request['families']):
        choice, diagnostics[family] = family_candidate(family, selection, development, request)
        selected[family] = None if choice is None else choice['rule']
    complete = all(rule is not None for rule in selected.values())
    result = dict(contract=CONTRACT if complete else SEARCH_CONTRACT,
                  status='robust-finalist' if complete else 'no-robust-finalist',
                  families=[item['name'] for item in request['families']],
                  frontier=[dict(family_rules=selected, cost=rule_cost(selected, selection, request))] if complete else [],
                  provenance=('G4.3 V4 development-only selection: one shared family rule was searched '
                              'against the immutable fourteen-fixture atlas and all consumed V1--V3 evidence.'),
                  v4_development=dict(request=str(args.development_request),
                                      cost_policy='unconstrained; rank robust survivors by cost',
                                      robust_ratio=ROBUST_RATIO, diagnostics=diagnostics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_V4_SEARCH', json.dumps(dict(status=result['status'],
                                              cost=result['frontier'][0]['cost'] if complete else None,
                                              family_rules=selected, diagnostics=diagnostics)))


if __name__ == '__main__':
    main()
