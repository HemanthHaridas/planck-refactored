#!/usr/bin/env python3
"""Materialize the immutable V4 development residual bank; test-only."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from g4_3_v4_development import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=Path(
        'tests/inputs/exploratory/dh_gradient/grid_calibration/g4_3_v4_development_request.json'))
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    args = parser.parse_args()
    request = load(args.request)
    root = args.request.parents[5]
    for case in request['cases']:
        destination = root / case['residual']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print('G4_3_V4_DEVELOPMENT_REUSE', case['name'])
            continue
        if case['source_residual'] is not None:
            shutil.copyfile(root / case['source_residual'], destination)
            print('G4_3_V4_DEVELOPMENT_COPY', case['name'])
            continue
        command = [sys.executable, str(root / 'tests/g4_3_shell_residual_audit.py'),
                   '--executable', str(args.executable), '--fixture', str(root / case['input']),
                   '--report-path', str(destination), '--allow-external-fixture']
        completed = subprocess.run(command)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        print('G4_3_V4_DEVELOPMENT_REBUILD', case['name'])


if __name__ == '__main__':
    main()
