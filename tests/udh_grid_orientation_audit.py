#!/usr/bin/env python3
"""No-build angular-only grid convergence audit using Planck's own energies/gradients.

Keeps radial quadrature, SCF controls and the 2e-6 covariance limit fixed.
Uses two rigid orientations and reports KS/PT2/total channels separately.
A passing refinement candidate still needs the full molecular FD suite.
"""
import argparse
import json
import math
from pathlib import Path
import re
import tempfile

from udh_molecular_audit import (Audit, FIXTURES, continuity, matmul, maxdiff, parse_geometry,
                                 requested_matrix, with_angular_points)

LIMIT = 2e-6


def rotations():
    # Original failing rotation plus an independent, non-octahedral orientation.
    a, b, c = (v/math.sqrt(14) for v in (1., 2., 3.))
    angle = .731
    co, si = math.cos(angle), math.sin(angle)
    u = [a, b, c]
    skew = [[0., -c, b], [c, 0., -a], [-b, a, 0.]]
    second = [[co*float(i == j)+(1-co)*u[i]*u[j]+si*skew[i][j]
               for j in range(3)] for i in range(3)]
    return [[[.8, -.6, 0.], [.48, .64, -.6], [.36, .48, .8]], second]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('build/planck-udh-validation'))
    parser.add_argument('--fixture', choices=('h2o2_cation_c1', 'ho2_asymmetric', 'water_cation_631g'),
                        default='h2o2_cation_c1')
    parser.add_argument('--levels', nargs='+', type=int, default=[0, 590, 974, 1454])
    args = parser.parse_args()
    if len(args.levels) < 2 or args.levels != sorted(set(args.levels)):
        parser.error('At least two distinct, increasing angular levels are required (0 = preset)')
    stem = 'water_cation_asymmetric' if args.fixture == 'water_cation_631g' else args.fixture
    template = (FIXTURES/(stem+'_b2plyp_sto3g.hfinp')).read_text()
    if args.fixture == 'water_cation_631g':
        template = re.sub(r'(basis\s+)sto-3g', r'\g<1>6-31g', template, count=1)
    root = Path(tempfile.mkdtemp(prefix='udh-angular-convergence-', dir='/private/tmp'))
    print('Artifacts:', root, flush=True)
    audit = Audit(args.executable.resolve(), Path('build/planck-dft').resolve(), root, [1e-4, 2e-4])
    report = dict(fixture=args.fixture, tolerance=LIMIT, radial_and_scf_unchanged=True,
                  candidate_passed=False, levels=[], scope='angular convergence only; not U11 promotion')
    previous = None
    try:
        for count in args.levels:
            text = with_angular_points(template, count)
            symbols, xyz, charge, mult = parse_geometry(text)
            center, state, _, _ = audit.run(f'angular-{count}-center', text, symbols, xyz, charge, mult, True)
            channels = {k: requested_matrix(state, k+'_standard') for k in ('ks', 'correction', 'total')}
            row = dict(angular_points=count, physical_fock_error=state['physical_fock_error'],
                       scaled_z_residual=state['scaled_z_residual'], rotations=[])
            current = [channels]
            states = [state]
            energies = [center['total_energy']]
            for index, rotation in enumerate(rotations()):
                moved_xyz = [[v+t for v, t in zip(atom, [.213, -.317, .149])]
                             for atom in matmul(xyz, rotation)]
                moved, spin, _, _ = audit.run(f'angular-{count}-rotation-{index}', text, symbols,
                                             moved_xyz, charge, mult, True)
                if any(spin[k] != state[k] for k in ('nalpha', 'nbeta')) or abs(spin['s2']-state['s2']) > .01:
                    raise ValueError('Spin state changed under grid rotation')
                got = {k: requested_matrix(spin, k+'_standard') for k in channels}
                current.append(got); energies.append(moved['total_energy'])
                states.append(spin)
                row['rotations'].append(dict(
                    gradient_errors={k: maxdiff(got[k], matmul(channels[k], rotation)) for k in channels},
                    total_energy_change=moved['total_energy']-center['total_energy'],
                    ks_energy_change=moved['scf_total_energy']-center['scf_total_energy'],
                    physical_fock_error=spin['physical_fock_error'], scaled_z_residual=spin['scaled_z_residual']))
            row['covariance_passed'] = all(
                abs(r['total_energy_change']) <= LIMIT and all(v <= LIMIT for v in r['gradient_errors'].values())
                for r in row['rotations'])
            row['refinement_stable'] = False
            if previous is not None:
                matrices, old_energies, old_states = previous
                # Same geometry, AO basis and frame at each corresponding
                # orientation: the ordinary AO overlap is the exact cross-S.
                row['subspaces_from_previous'] = [continuity(old, current_state | {
                    'cross_overlap_reference_current': old['overlap']})
                    for old, current_state in zip(old_states, states)]
                row['gradient_change_from_previous'] = {
                    k: max(maxdiff(x[k], y[k]) for x, y in zip(current, matrices)) for k in channels}
                row['energy_change_from_previous'] = max(abs(x-y) for x, y in zip(energies, old_energies))
                row['refinement_stable'] = (row['energy_change_from_previous'] <= LIMIT and
                                           all(v <= LIMIT for v in row['gradient_change_from_previous'].values()))
            previous = current, energies, states
            report['levels'].append(row)
            print('ANGULAR_LEVEL', json.dumps(row), flush=True)
            (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        final = report['levels'][-1]
        report['candidate_passed'] = final['covariance_passed'] and final['refinement_stable']
    except (ValueError, OSError, KeyError) as error:
        report['error'] = str(error)
    (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(('PASS' if report['candidate_passed'] else 'FAIL')+' angular refinement candidate', flush=True)
    if 'error' in report:
        print(report['error'], flush=True)
    return 0 if report['candidate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
