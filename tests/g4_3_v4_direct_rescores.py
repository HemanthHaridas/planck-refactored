#!/usr/bin/env python3
"""Run literal V4 direct-grid/atlas reconciliation for all consumed evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from g4_3_derivation_contract import load as load_selection_request
from g4_3_v4_development import load as load_development_request


def _selection_cases(root, artifact):
    request = load_selection_request(root / 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_derivation_request.json')
    reports = {}
    for path in sorted((artifact / 'selection').glob('*.json')):
        value = json.loads(path.read_text())
        fixture = value.get('fixture')
        if fixture in reports:
            raise ValueError(f'duplicate selection residual report for {fixture}')
        reports[fixture] = path
    return [(Path(fixture), reports[fixture], fixture) for fixture in request['fixtures']]


def _development_cases(root, request_path):
    request = load_development_request(request_path)
    return [(root / case['input'], root / case['residual'], case['name']) for case in request['cases']]


def _run(root, executable, frontier, fixture, residual, report):
    command = [sys.executable, str(root / 'tests/g4_3_direct_rescore_run.py'),
               '--executable', str(executable), '--fixture', str(fixture),
               '--residual-report', str(residual), '--combined-frontier', str(frontier),
               '--v4', '--report-path', str(report)]
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--artifact-root', type=Path, default=Path(
        'tests/inputs/exploratory/dh_gradient/grid_calibration/artifacts/g4_3_9_revision_2026_10_03'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact = (root / args.artifact_root).resolve()
    executable = (root / args.executable).resolve()
    frontier = artifact / 'g4-3-v4-combined-frontier.json'
    development_request = root / 'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v4_development_request.json'
    selection_out = artifact / 'v4_direct_rescores' / 'selection'
    development_out = artifact / 'v4_direct_rescores' / 'development'
    selection_out.mkdir(parents=True, exist_ok=True)
    development_out.mkdir(parents=True, exist_ok=True)

    selection = _selection_cases(root, artifact)
    development = _development_cases(root, development_request)
    for label, cases, output in (('selection', selection, selection_out),
                                 ('development', development, development_out)):
        for ordinal, (fixture, residual, name) in enumerate(cases, start=1):
            print(f'G4_3_V4_DIRECT_PROGRESS {{"tier":"{label}","case":{ordinal},"total":{len(cases)},"name":"{name}"}}',
                  flush=True)
            _run(root, executable, frontier, fixture, residual, output / f'{Path(name).stem}.json')
    print(f'PASS G4.3 V4 direct reconciliation: {len(selection)} selection + {len(development)} development cases')


if __name__ == '__main__':
    main()
