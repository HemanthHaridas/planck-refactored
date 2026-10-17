#!/usr/bin/env python3
"""Parse and validate the immutable G4.3 physical-radius derivation request.

This is deliberately a data-contract utility.  It does not construct a grid,
evaluate XC quantities, or invoke an SCF calculation.
"""
import json
from pathlib import Path


CONTRACT = 'g4-3-physical-radius-derivation-v1'
OBSERVABLES = {
    'xc_energy', 'density_xc_potential', 'density_xc_kernel_density'
}
FORBIDDEN_SELECTOR_DEPENDENCIES = {
    'density', 'molecular_geometry', 'scf_iteration', 'orbital_response'
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(request):
    _require(request.get('contract') == CONTRACT, 'unsupported derivation contract')
    expected_keys = {
        'contract', 'reference_ladder', 'limits', 'observables',
        'directions_per_fixture', 'shell_selector', 'families',
        'allowed_lebedev_points', 'allowed_band_boundaries', 'fixtures',
    }
    _require(set(request) == expected_keys, 'unexpected derivation keys')

    ladder = request.get('reference_ladder')
    _require(isinstance(ladder, dict), 'missing reference_ladder')
    expected_ladder = {
        'angular_anchor': (135, 2030),
        'radial_check': (165, 2030),
        'angular_check': (165, 2702),
    }
    for name, expected in expected_ladder.items():
        level = ladder.get(name)
        _require(isinstance(level, dict), f'missing reference level {name}')
        pair = (level.get('radial_points'), level.get('angular_points'))
        _require(pair == expected, f'{name} must be {expected[0]}:{expected[1]}')

    limits = request.get('limits')
    _require(isinstance(limits, dict), 'missing limits')
    _require(limits.get('xc_energy_ha') == 1e-8, 'invalid XC-energy limit')
    _require(limits.get('xc_response_contraction') == 1e-7,
             'invalid XC-response limit')

    _require(set(request.get('observables', [])) == OBSERVABLES,
             'must request exactly the three G4.3 observables')
    _require(request.get('directions_per_fixture') == 2,
             'each fixture must retain two AO directions')

    selector = request.get('shell_selector')
    _require(isinstance(selector, dict), 'missing shell_selector')
    _require(selector.get('coordinate') == 'm4-radius-over-bragg-radius',
             'selector must use fixed M4 radius over Bragg radius')
    depends_on = set(selector.get('depends_on', []))
    _require(depends_on == {'atomic_number', 'fixed_radial_coordinate'},
             'selector dependencies must be atom number and fixed radial coordinate only')
    forbidden = set(selector.get('forbids', []))
    _require(forbidden == FORBIDDEN_SELECTOR_DEPENDENCIES,
             'selector must explicitly forbid density, geometry, SCF, and response dependencies')
    _require(not (depends_on & FORBIDDEN_SELECTOR_DEPENDENCIES),
             'selector depends on a forbidden live quantity')

    families = request.get('families')
    _require(isinstance(families, list) and len(families) == 3,
             'exactly three periodic families are required')
    covered = []
    names = set()
    for family in families:
        _require(isinstance(family, dict), 'invalid family record')
        name = family.get('name')
        numbers = family.get('atomic_numbers')
        _require(isinstance(name, str) and name not in names, 'duplicate family name')
        _require(isinstance(numbers, list) and all(isinstance(z, int) for z in numbers),
                 f'{name}: invalid atomic-number list')
        names.add(name)
        covered.extend(numbers)
    _require(sorted(covered) == list(range(1, 19)),
             'families must partition atomic numbers 1 through 18')

    angular = request.get('allowed_lebedev_points')
    _require(isinstance(angular, list) and angular == sorted(set(angular)),
             'allowed Lebedev sizes must be sorted and unique')
    _require(angular and angular[-1] == 2030,
             'allowed Lebedev sizes must include the 2030-point anchor')
    supported = {26, 50, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434,
                 590, 770, 974, 1202, 1454, 1730, 2030}
    _require(set(angular) <= supported, 'unsupported Lebedev size')

    boundaries = request.get('allowed_band_boundaries')
    _require(isinstance(boundaries, list) and len(boundaries) >= 4,
             'at least four candidate physical-radius boundaries are required')
    _require(all(isinstance(value, (int, float)) for value in boundaries),
             'physical-radius boundaries must be numeric')
    _require(boundaries == sorted(set(boundaries)),
             'physical-radius boundaries must be strictly increasing')

    fixtures = request.get('fixtures')
    _require(isinstance(fixtures, list) and len(fixtures) == 14,
             'exactly fourteen G4.2 fixtures are required')
    _require(all(isinstance(path, str) and path.endswith('.hfinp') for path in fixtures),
             'fixture paths must name input files')
    _require(len(set(fixtures)) == len(fixtures), 'duplicate fixture')
    return request


def load(path):
    return validate(json.loads(Path(path).read_text()))
