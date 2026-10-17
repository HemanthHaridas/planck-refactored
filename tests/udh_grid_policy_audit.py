#!/usr/bin/env python3
"""G3 live-SCF attribution audit for immutable DFT grid policies.

This is deliberately not a finite-density oracle: each policy gets a fresh,
stationary SCF/PT2 solution.  It isolates the observable impact of the
angular schedule while holding the requested geometry, radial count, SCF
input, functional, and lab-frame rotations fixed.  The separate fixed-density
contraction ledger remains the next G3 layer.
"""
import argparse
import json
import math
from pathlib import Path
import re
import tempfile

from udh_molecular_audit import (Audit, FIXTURES, matmul, maxdiff, parse_geometry,
                                 requested_matrix, with_grid_policy, with_radial_points)
from udh_grid_orientation_audit import rotations

POLICIES = ('legacy', 'legacy-no-light-reduction', 'legacy-unpruned-max', 'pyscf-nwchem-2.13',
            'g4-3-v4', 'g4-3-v4-test-only')
LIMIT = 2e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--fixture', choices=('h2o2_cation_c1', 'ho2_asymmetric', 'water_cation_631g'),
                        default='ho2_asymmetric')
    parser.add_argument('--radial-points', type=int, default=135,
                        help='Fixed per-atom radial count; zero retains the preset table (default: 135).')
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    args = parser.parse_args()
    if args.radial_points < 0:
        parser.error('--radial-points must be nonnegative')
    if not args.policies or args.policies[0] != 'legacy':
        parser.error('--policies must begin with legacy so all deltas share that baseline')
    stem = 'water_cation_asymmetric' if args.fixture == 'water_cation_631g' else args.fixture
    template = (FIXTURES/(stem+'_b2plyp_sto3g.hfinp')).read_text()
    if args.fixture == 'water_cation_631g':
        template = re.sub(r'(basis\s+)sto-3g', r'\g<1>6-31g', template, count=1)
    root = Path(tempfile.mkdtemp(prefix='udh-grid-policy-', dir='/private/tmp'))
    print('Artifacts:', root, flush=True)
    audit = Audit(args.executable.resolve(), Path('build/planck-dft').resolve(), root, [1e-4, 2e-4])
    report = dict(fixture=args.fixture, radial_points=args.radial_points,
                  policies=[], scope='G3 live stationary-policy attribution; not fixed-density attribution')
    baseline = None
    try:
        for policy in args.policies:
            text = with_grid_policy(with_radial_points(template, args.radial_points), policy)
            symbols, xyz, charge, mult = parse_geometry(text)
            center, state, _, _ = audit.run(policy+'-center', text, symbols, xyz, charge, mult, True)
            matrices = {key: requested_matrix(state, key+'_standard') for key in ('ks', 'correction', 'total')}
            row = dict(policy=policy, total_energy=center['total_energy'],
                       scf_energy=center['scf_total_energy'], physical_fock_error=state['physical_fock_error'],
                       scaled_z_residual=state['scaled_z_residual'], rotations=[])
            for index, rotation in enumerate(rotations()):
                moved_xyz = [[v+t for v, t in zip(atom, [.213, -.317, .149])]
                             for atom in matmul(xyz, rotation)]
                moved, spin, _, _ = audit.run(policy+f'-rotation-{index}', text, symbols,
                                               moved_xyz, charge, mult, True)
                got = {key: requested_matrix(spin, key+'_standard') for key in matrices}
                row['rotations'].append(dict(
                    gradient_errors={key: maxdiff(got[key], matmul(matrices[key], rotation)) for key in matrices},
                    total_energy_change=moved['total_energy']-center['total_energy'],
                    physical_fock_error=spin['physical_fock_error'], scaled_z_residual=spin['scaled_z_residual']))
            row['covariance_passed'] = all(
                abs(item['total_energy_change']) <= LIMIT and
                all(value <= LIMIT for value in item['gradient_errors'].values())
                for item in row['rotations'])
            if baseline is None:
                baseline = center, matrices
                row['delta_from_legacy'] = {key: 0.0 for key in ('energy', 'ks', 'correction', 'total')}
            else:
                baseline_energy, baseline_matrices = baseline
                row['delta_from_legacy'] = dict(
                    energy=center['total_energy']-baseline_energy['total_energy'],
                    ks=maxdiff(matrices['ks'], baseline_matrices['ks']),
                    correction=maxdiff(matrices['correction'], baseline_matrices['correction']),
                    total=maxdiff(matrices['total'], baseline_matrices['total']))
            report['policies'].append(row)
            print('POLICY', json.dumps(row), flush=True)
            (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    except (ValueError, OSError, KeyError) as error:
        report['error'] = str(error)
    report['complete'] = 'error' not in report and len(report['policies']) == len(args.policies)
    (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(('PASS' if report['complete'] else 'FAIL')+' live policy attribution', flush=True)
    if 'error' in report:
        print(report['error'], flush=True)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
