#!/usr/bin/env python3
"""Build a reproducible G4.2 policy-qualification manifest from ledger reports.

Inputs are ``results.json`` files emitted by g4_fixed_xc_ledger_audit.py.
This tool never runs an SCF calculation: it normalizes already-measured
stationary-density ledgers, preserving separate XC-energy and XC-response
qualification gates.
"""
import argparse
import json
import math
from pathlib import Path


CANDIDATES = ('legacy-unpruned-max', 'pyscf-nwchem-2.13')


def response_maximum(delta):
    values = []
    for direction in delta['directions']:
        values.extend((abs(direction['D_V']), abs(direction['D_fxc_D'])))
    return max(values, default=0.0)


def summarize(path):
    report = json.loads(path.read_text())
    if report.get('contract') != 'g4-fixed-stationary-density-v1':
        raise ValueError(f'{path}: unsupported ledger contract')
    if not report.get('reference_stable'):
        raise ValueError(f'{path}: unstable R1/R2 reference')
    rows = {row['name']: row for row in report.get('rows', [])}
    missing = set(CANDIDATES) - rows.keys()
    if missing:
        raise ValueError(f'{path}: missing candidates {sorted(missing)}')
    energy_limit = report['r2_energy_limit']
    contraction_limit = report['r2_contraction_limit']
    policies = {}
    for name in CANDIDATES:
        delta = rows[name]['r2_delta']
        energy = abs(delta['energy'])
        response = response_maximum(delta)
        policies[name] = dict(points=rows[name]['points'], energy_error=energy,
                              response_error=response,
                              energy_pass=energy <= energy_limit,
                              response_pass=response <= contraction_limit,
                              qualified=(energy <= energy_limit and response <= contraction_limit))
    return dict(fixture=report['fixture'], parent_grid=report['parent_grid'],
                energy_limit=energy_limit, contraction_limit=contraction_limit,
                policies=policies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledgers', type=Path, nargs='+', help='G4 results.json files')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = [summarize(path) for path in args.ledgers]
    if len({item['fixture'] for item in cases}) != len(cases):
        parser.error('each fixture may appear only once')
    manifest = dict(contract='g4-grid-calibration-manifest-v1', cases=cases,
                    qualified_policies=[name for name in CANDIDATES if all(
                        case['policies'][name]['qualified'] for case in cases)])
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, separators=(',', ':')))


if __name__ == '__main__':
    main()
