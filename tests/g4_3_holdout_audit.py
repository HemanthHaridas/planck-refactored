#!/usr/bin/env python3
"""Run one or all frozen G4.3.9 fixed-density holdout cases."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from g4_3_direct_rescore_audit import load_finalist
from g4_3_holdout_contract import load
from g4_3_rule_certificate import certificate, passes


CONTRACT = 'g4-3-fixed-density-holdout-ledger-v1'


def _run(command):
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError('holdout sub-run failed: ' + ' '.join(map(str, command)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=Path(
        'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_holdout_request.json'))
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--case', action='append', help='frozen holdout case name; repeat to select a subset')
    parser.add_argument('--finalist', type=Path,
                        help='test-only revised finalist; the immutable request remains the holdout definition')
    parser.add_argument('--revised-direct-builder', action='store_true',
                        help='use the explicit G4.3.9 second-row-floor direct builder')
    parser.add_argument('--dual-floor-direct-builder', action='store_true',
                        help='use the explicit dual-energy-floor direct builder')
    parser.add_argument('--v3-direct-builder', action='store_true',
                        help='use the explicit G4.3 v3 constrained-candidate direct builder')
    parser.add_argument('--v4-direct-builder', action='store_true',
                        help='use the explicit G4.3 v4 robust-candidate direct builder')
    parser.add_argument('--reference-report', type=Path,
                        help='reuse a completed reference-ladder report for the selected single case')
    parser.add_argument('--artifact-dir', type=Path,
                        help='durable directory for per-case reference, residual, and direct reports')
    parser.add_argument('--output', type=Path, help='immutable normalized ledger destination')
    args = parser.parse_args()
    if sum((args.revised_direct_builder, args.dual_floor_direct_builder,
            args.v3_direct_builder, args.v4_direct_builder)) > 1:
        parser.error('direct-builder flags are mutually exclusive')
    request = load(args.request)
    root = args.request.parents[5]
    selected = [case for case in request['cases'] if not args.case or case['name'] in set(args.case)]
    if not selected or args.case and len(selected) != len(set(args.case)):
        parser.error('unknown or empty holdout case selection')
    work = (args.artifact_dir if args.artifact_dir is not None else
            Path(tempfile.mkdtemp(prefix='g4-3-holdout-', dir='/private/tmp')))
    work.mkdir(parents=True, exist_ok=True)
    derivation = root / 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json'
    finalist = args.finalist or root / request['finalist']
    if (args.revised_direct_builder or args.dual_floor_direct_builder or
            args.v3_direct_builder or args.v4_direct_builder) and args.finalist is None:
        parser.error('an explicit finalist is required with a non-retired direct builder')
    if args.reference_report is not None and len(selected) != 1:
        parser.error('--reference-report requires exactly one selected holdout case')
    family_rules = load_finalist(finalist)
    rows = []
    for case in selected:
        prefix = work / case['name']
        fixture = root / case['input']
        reference = prefix.with_suffix('.reference.json')
        residual = prefix.with_suffix('.residual.json')
        direct = prefix.with_suffix('.direct.json')
        if args.reference_report is None:
            _run([sys.executable, str(root / 'tests/g4_3_reference_ladder_audit.py'), '--executable', str(args.executable),
                  '--fixture', str(fixture), '--report-path', str(reference)])
        else:
            reference = args.reference_report
        _run([sys.executable, str(root / 'tests/g4_3_shell_residual_audit.py'), '--executable', str(args.executable),
              '--fixture', str(fixture), '--report-path', str(residual), '--allow-external-fixture'])
        direct_command = [sys.executable, str(root / 'tests/g4_3_direct_rescore_run.py'), '--executable', str(args.executable),
              '--fixture', str(fixture), '--residual-report', str(residual), '--combined-frontier', str(finalist),
              '--report-path', str(direct)]
        if args.revised_direct_builder:
            direct_command.append('--revised')
        elif args.dual_floor_direct_builder:
            direct_command.append('--dual-floor')
        elif args.v3_direct_builder:
            direct_command.append('--v3')
        elif args.v4_direct_builder:
            direct_command.append('--v4')
        _run(direct_command)
        reference_value = json.loads(reference.read_text())
        if reference_value.get('fixture') != str(fixture):
            raise ValueError('reused reference report belongs to a different fixture')
        residual_value = json.loads(residual.read_text())
        direct_value = json.loads(direct.read_text())
        certificates = {family: certificate(family_rules[family], family, [residual_value],
                                             json.loads(derivation.read_text()))
                        for family in case['families']}
        certificates_pass = all(passes(value, json.loads(derivation.read_text())) for value in certificates.values())
        row = dict(name=case['name'], input=case['input'], role=case['role'], families=case['families'],
                   reference_stable=reference_value['reference_stable'],
                   direct_reconciliation_passed=direct_value['passed'],
                   direct_max_error=direct_value['max_error'],
                   l1_certificate_passed=certificates_pass,
                   l1_certificates=certificates)
        row['passed'] = row['reference_stable'] and row['direct_reconciliation_passed'] and certificates_pass
        rows.append(row)
    result = dict(contract=CONTRACT, request_contract=request['contract'], finalist=str(finalist), reference=request['reference'],
                  limits=request['limits'], cases=rows, passed=all(row['passed'] for row in rows))
    destination = args.output or work / 'results.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print('Artifacts:', work)
    print('G4_3_HOLDOUT', json.dumps(dict(cases=len(rows), passed=result['passed']), separators=(',', ':')))
    if not result['passed']:
        raise SystemExit('FAIL G4.3 fixed-density holdout')
    print('PASS G4.3 fixed-density holdout')


if __name__ == '__main__':
    main()
