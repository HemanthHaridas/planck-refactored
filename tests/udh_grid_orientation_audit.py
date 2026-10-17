#!/usr/bin/env python3
"""No-build angular, radial, or joint grid convergence audit using Planck's own energies/gradients.

Refines one quadrature dimension at a time, with two rigid orientations and
separate KS/PT2/total channels. A passing candidate still needs the full
molecular FD suite.
"""
import argparse
import json
import math
from pathlib import Path
import re
import tempfile

from udh_molecular_audit import (Audit, FIXTURES, continuity, matmul, maxdiff, parse_geometry,
                                 requested_matrix, with_angular_points, with_radial_points, with_grid_policy)

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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--radial-levels', nargs='+', type=int,
                        help='Radial shell counts to compare. Selects radial mode; use --angular-points for its fixed angular rule.')
    mode.add_argument('--joint-levels', nargs='+',
                      help='Strictly refining RADIAL:ANGULAR pairs, e.g. 105:974 135:1454 165:2030.')
    mode.add_argument('--reference-levels', nargs=2, metavar='RADIAL:ANGULAR',
                      help='G4.1 two-level uniform reference ladder, e.g. 135:1454 165:2030. '
                           'Uses independent reference-stability tolerances.')
    parser.add_argument('--angular-points', type=int, default=1454,
                        help='Fixed unpruned angular count in radial mode (default: 1454).')
    parser.add_argument('--reference-energy-tolerance', type=float, default=1e-8,
                        help='Maximum R1-to-R2 total-energy change for --reference-levels (Ha).')
    parser.add_argument('--reference-gradient-tolerance', type=float, default=1e-7,
                        help='Maximum R1-to-R2 KS/PT2/total gradient change for --reference-levels (Ha/Bohr).')
    parser.add_argument('--grid-policy', default='legacy', choices=(
                        'legacy', 'legacy-no-light-reduction', 'legacy-unpruned-max', 'pyscf-nwchem-2.13'),
                        help='Immutable pruning-policy candidate for this audit (default: legacy).')
    args = parser.parse_args()
    level_tokens = args.reference_levels if args.reference_levels is not None else args.joint_levels
    if level_tokens is not None:
        configurations = []
        for item in level_tokens:
            try:
                radial, angular = (int(value) for value in item.split(':'))
            except ValueError:
                parser.error('joint/reference levels must be RADIAL:ANGULAR integer pairs')
            if radial < 0 or angular < 6 or angular > 5810:
                parser.error('joint/reference radial counts must be nonnegative and angular counts must be 6..5810')
            configurations.append((radial, angular))
        if len(configurations) < 2 or any(
                later[0] <= earlier[0] or later[1] <= earlier[1]
                for earlier, later in zip(configurations, configurations[1:])):
            parser.error('joint/reference levels require increasing radial and angular pairs')
        audit_mode = 'reference' if args.reference_levels is not None else 'joint'
    elif args.radial_levels is not None:
        if len(args.radial_levels) < 2 or args.radial_levels != sorted(set(args.radial_levels)):
            parser.error('At least two distinct, increasing radial levels are required')
        configurations = [(count, args.angular_points) for count in args.radial_levels]
        audit_mode = 'radial'
    else:
        if len(args.levels) < 2 or args.levels != sorted(set(args.levels)):
            parser.error('At least two distinct, increasing angular levels are required')
        configurations = [(None, count) for count in args.levels]
        audit_mode = 'angular'
    if audit_mode == 'radial' and (args.angular_points < 6 or args.angular_points > 5810):
        parser.error('--angular-points must be a supported Lebedev size from 6 to 5810')
    if args.reference_energy_tolerance <= 0 or args.reference_gradient_tolerance <= 0:
        parser.error('reference tolerances must be positive')
    stem = 'water_cation_asymmetric' if args.fixture == 'water_cation_631g' else args.fixture
    template = (FIXTURES/(stem+'_b2plyp_sto3g.hfinp')).read_text()
    if args.fixture == 'water_cation_631g':
        template = re.sub(r'(basis\s+)sto-3g', r'\g<1>6-31g', template, count=1)
    root = Path(tempfile.mkdtemp(prefix='udh-'+audit_mode+'-convergence-', dir='/private/tmp'))
    print('Artifacts:', root, flush=True)
    audit = Audit(args.executable.resolve(), Path('build/planck-dft').resolve(), root, [1e-4, 2e-4])
    report = dict(fixture=args.fixture, grid_policy=args.grid_policy, tolerance=LIMIT,
                  fixed_angular_points=args.angular_points if audit_mode == 'radial' else None,
                  radial_and_scf_unchanged=audit_mode == 'angular',
                  angular_and_scf_unchanged=audit_mode == 'radial',
                  reference_ladder=(dict(levels=[f'{r}:{a}' for r, a in configurations],
                                         energy_tolerance=args.reference_energy_tolerance,
                                         gradient_tolerance=args.reference_gradient_tolerance)
                                    if audit_mode == 'reference' else None),
                  candidate_passed=False, levels=[],
                  scope=(('G4.1 uniform-reference stabilization only; not policy selection or U11 promotion')
                         if audit_mode == 'reference' else audit_mode+' convergence only; not U11 promotion'))
    previous = None
    try:
        for radial_count, angular_count in configurations:
            text = (with_radial_points(with_angular_points(template, angular_count), radial_count)
                    if radial_count is not None else with_angular_points(template, angular_count))
            text = with_grid_policy(text, args.grid_policy)
            tag = audit_mode+f'-{radial_count}-{angular_count}'
            symbols, xyz, charge, mult = parse_geometry(text)
            center, state, _, _ = audit.run(tag+'-center', text, symbols, xyz, charge, mult, True)
            channels = {k: requested_matrix(state, k+'_standard') for k in ('ks', 'correction', 'total')}
            row = dict(**({"radial_points": radial_count, "angular_points": angular_count}
                         if radial_count is not None else {"angular_points": angular_count}),
                       physical_fock_error=state['physical_fock_error'],
                       scaled_z_residual=state['scaled_z_residual'], rotations=[])
            current = [channels]
            states = [state]
            energies = [center['total_energy']]
            for index, rotation in enumerate(rotations()):
                moved_xyz = [[v+t for v, t in zip(atom, [.213, -.317, .149])]
                             for atom in matmul(xyz, rotation)]
                moved, spin, _, _ = audit.run(tag+f'-rotation-{index}', text, symbols,
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
            row['reference_stable'] = None
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
                if audit_mode == 'reference':
                    row['reference_stable'] = (
                        row['energy_change_from_previous'] <= args.reference_energy_tolerance and
                        all(v <= args.reference_gradient_tolerance
                            for v in row['gradient_change_from_previous'].values()))
            previous = current, energies, states
            report['levels'].append(row)
            print(audit_mode.upper()+'_LEVEL', json.dumps(row), flush=True)
            (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        final = report['levels'][-1]
        if audit_mode == 'reference':
            report['reference_stable'] = final['reference_stable']
            report['candidate_passed'] = final['covariance_passed'] and final['reference_stable']
        else:
            report['candidate_passed'] = final['covariance_passed'] and final['refinement_stable']
    except (ValueError, OSError, KeyError) as error:
        report['error'] = str(error)
    (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(('PASS' if report['candidate_passed'] else 'FAIL')+' '+audit_mode+' refinement candidate', flush=True)
    if 'error' in report:
        print(report['error'], flush=True)
    return 0 if report['candidate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
