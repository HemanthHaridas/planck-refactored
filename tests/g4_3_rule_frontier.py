#!/usr/bin/env python3
"""Emit the deterministic, cost-only G4.3.6 frontier from stored reports."""
import argparse
import json
from pathlib import Path

from g4_3_derivation_contract import load
from g4_3_rule_cost import load_reports, rule_cost
from g4_3_rule_grammar import all_2030_rule, pyscf_nwchem_213_specimen, rule_space_size


CONTRACT = 'g4-3-rule-frontier-v1'


def canonical_minimum(request):
    return {'boundaries': list(request['allowed_band_boundaries'][:4]),
            'lebedev_points': [request['allowed_lebedev_points'][0]] * 5}


def build_frontier(tables, request):
    families = [family['name'] for family in request['families']]
    all_2030 = {name: all_2030_rule(request) for name in families}
    minimum = {name: canonical_minimum(request) for name in families}
    pyscf = pyscf_nwchem_213_specimen(request)
    per_family = rule_space_size(request)
    return {
        'contract': CONTRACT,
        'candidate_space': {'families': families, 'rules_per_family': per_family,
                            'family_rule_combinations': per_family ** len(families)},
        'anchors': [
            {'name': 'all-2030', 'eligible': True, 'family_rules': all_2030,
             'cost': rule_cost(all_2030, tables, request)},
            {'name': 'pyscf-nwchem-2.13-level-7-specimen', 'eligible': False,
             'reason': 'fourth-band maximum violates derived candidate shape',
             'family_rules': pyscf,
             'cost': rule_cost(pyscf, tables, request, validate_candidates=False)},
        ],
        'cost_frontier': [
            {'name': 'canonical-all-minimum', 'family_rules': minimum,
             'cost': rule_cost(minimum, tables, request)},
        ],
        'note': 'Cost-only frontier; G4.3.7 L1 scoring determines eligibility.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = load(Path(__file__).resolve().parent / 'inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    result = build_frontier(load_reports(args.reports, request), request)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('G4_3_RULE_FRONTIER', json.dumps({'rules_per_family': result['candidate_space']['rules_per_family'],
                                             'cost_frontier': len(result['cost_frontier'])}))


if __name__ == '__main__':
    main()
