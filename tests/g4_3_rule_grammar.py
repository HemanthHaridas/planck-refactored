#!/usr/bin/env python3
"""Pure G4.3.6 physical-radius five-band rule grammar and lazy enumerator.

This module deliberately does not open an XC grid, run SCF, or score an
atlas.  It freezes the finite candidate language that the next stage will
cost and certify using the checked-in shell-residual evidence.
"""
from itertools import combinations, product


CONTRACT = 'g4-3-five-band-rule-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_rule(rule, request):
    """Validate one immutable family rule from the Section G4.3 grammar."""
    _require(isinstance(rule, dict) and set(rule) == {'boundaries', 'lebedev_points'},
             'rule must contain only boundaries and lebedev_points')
    boundaries = rule['boundaries']
    counts = rule['lebedev_points']
    allowed_boundaries = request['allowed_band_boundaries']
    allowed_counts = request['allowed_lebedev_points']
    _require(isinstance(boundaries, list) and len(boundaries) == 4,
             'rule requires four physical-radius boundaries')
    _require(boundaries == sorted(boundaries) and len(set(boundaries)) == 4,
             'rule boundaries must be strictly increasing')
    _require(all(value in allowed_boundaries for value in boundaries),
             'rule uses an unapproved physical-radius boundary')
    _require(isinstance(counts, list) and len(counts) == 5,
             'rule requires five Lebedev counts')
    _require(all(value in allowed_counts for value in counts),
             'rule uses an unapproved Lebedev count')
    n0, n1, n2, n3, n4 = counts
    _require(n0 <= n1 <= n2 and n3 <= n2 and n4 <= n3,
             'rule violates the single well-resolved valence-band shape')
    return rule


def select_band(u, boundaries):
    """Return the deterministic band index for a fixed M4/Bragg radius."""
    boundaries = list(boundaries)
    _require(len(boundaries) == 4 and boundaries == sorted(boundaries),
             'invalid physical-radius boundaries')
    return sum(u >= boundary for boundary in boundaries)


def enumerate_rules(request):
    """Yield every valid rule in deterministic lexical order, without storing it."""
    boundaries = request['allowed_band_boundaries']
    counts = request['allowed_lebedev_points']
    for cuts in combinations(boundaries, 4):
        for values in product(counts, repeat=5):
            n0, n1, n2, n3, n4 = values
            if n0 <= n1 <= n2 and n3 <= n2 and n4 <= n3:
                yield {'boundaries': list(cuts), 'lebedev_points': list(values)}


def all_2030_rule(request):
    rule = {'boundaries': list(request['allowed_band_boundaries'][:4]),
            'lebedev_points': [2030] * 5}
    return validate_rule(rule, request)


def pyscf_nwchem_213_specimen(request):
    """Literal level-7 PySCF/NWChem schedule, retained as an external specimen.

    It is intentionally *not* a valid derived rule: its central maximum is
    in band three rather than the grammar's band two.  This keeps comparison
    with PySCF reproducible without importing its schedule into the candidate
    language being derived from Planck evidence.
    """
    specimen = {
        'hydrogen-helium': {'boundaries': [.25, .5, 1., 4.5],
                            'lebedev_points': [50, 86, 770, 974, 770]},
        'second-row': {'boundaries': [.1667, .5, .9, 3.5],
                       'lebedev_points': [50, 86, 974, 1202, 974]},
        'third-row': {'boundaries': [.1, .4, .8, 2.5],
                      'lebedev_points': [50, 86, 974, 1202, 974]},
    }
    _require(set(specimen) == {family['name'] for family in request['families']},
             'PySCF specimen does not cover the frozen periodic families')
    return specimen


def rule_space_size(request):
    """Exact finite count of one family rule, without enumerating it."""
    levels = len(request['allowed_lebedev_points'])
    shapes = sum((((index + 1) * (index + 2)) // 2) ** 2 for index in range(levels))
    return len(list(combinations(request['allowed_band_boundaries'], 4))) * shapes
