#!/usr/bin/env python3
"""Run one G4.3.8 literal-grid/atlas reconciliation from a fixed parent."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from g4_3_direct_rescore_audit import reconcile, load_finalist
from udh_molecular_audit import with_angular_points, with_grid_policy, with_radial_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--residual-report', type=Path, required=True)
    parser.add_argument('--combined-frontier', type=Path, required=True)
    parser.add_argument('--report-path', type=Path)
    parser.add_argument('--revised', action='store_true',
                        help='use the explicit G4.3.9 second-row-floor direct builder')
    parser.add_argument('--dual-floor', action='store_true',
                        help='use the explicit G4.3.9 dual-energy-floor direct builder')
    parser.add_argument('--v3', action='store_true',
                        help='use the explicit G4.3 v3 constrained-candidate direct builder')
    parser.add_argument('--v4', action='store_true',
                        help='use the explicit G4.3 v4 robust-candidate direct builder')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='g4-3-direct-rescore-', dir='/private/tmp'))
    parent = with_grid_policy(with_radial_points(with_angular_points(args.fixture.read_text(), 1454), 135), 'legacy')
    input_path, result_path, raw_path = root / 'parent.hfinp', root / 'parent.json', root / 'direct.raw.json'
    input_path.write_text(parent)
    environment = dict(os.environ)
    environment.setdefault('BASIS_PATH', str(Path(__file__).resolve().parents[1] / 'basis-sets'))
    if sum((args.revised, args.dual_floor, args.v3, args.v4)) > 1:
        parser.error('--revised, --dual-floor, --v3, and --v4 are mutually exclusive')
    flag = ('--fixed-xc-g4-3-v4-direct-rescore' if args.v4 else
            '--fixed-xc-g4-3-v3-direct-rescore' if args.v3 else
            '--fixed-xc-g4-3-dual-floor-direct-rescore' if args.dual_floor else
            '--fixed-xc-g4-3-revised-direct-rescore' if args.revised else
            '--fixed-xc-g4-3-direct-rescore')
    completed = subprocess.run([str(args.executable.resolve()), str(input_path), '--json', str(result_path),
                                flag, str(raw_path)], env=environment)
    if completed.returncode:
        print('Artifacts:', root, flush=True)
        return completed.returncode
    result = reconcile(json.loads(raw_path.read_text()), json.loads(args.residual_report.read_text()),
                       load_finalist(args.combined_frontier))
    result['fixture'] = str(args.fixture)
    rendered = json.dumps(result, indent=2) + '\n'
    (root / 'results.json').write_text(rendered)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered)
    print('Artifacts:', root)
    print('G4_3_DIRECT_RESCORE', json.dumps(dict(fixture=result['fixture'], points=result['points'],
                                                  max_error=result['max_error'], passed=result['passed']),
                                           separators=(',', ':')))
    if not result['passed']:
        raise SystemExit('FAIL G4.3 direct rescore')
    print('PASS G4.3 direct rescore')


if __name__ == '__main__':
    main()
