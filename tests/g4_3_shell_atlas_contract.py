#!/usr/bin/env python3
"""Validate the test-only G4.3 shell-atlas interchange record.

The atlas records fixed-density XC point contributions.  It intentionally
contains no SCF state, geometry-dependent selector, or production grid policy.
"""
import json
import math
from pathlib import Path

from g4_3_derivation_contract import load as load_request


CONTRACT = 'g4-3-shell-atlas-v1'
SPINS = {'alpha', 'beta'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def physical_band(u, boundaries):
    """Return the fixed radius band; equality belongs to the inner band."""
    return sum(u > boundary for boundary in boundaries)


def family_for_atomic_number(atomic_number, families):
    for family in families:
        if atomic_number in family['atomic_numbers']:
            return family['name']
    raise ValueError(f'unsupported atomic number {atomic_number}')


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate(atlas, request, allow_external_fixture=False):
    _require(atlas.get('contract') == CONTRACT, 'unsupported shell-atlas contract')
    fixture = atlas.get('fixture')
    _require(isinstance(fixture, str) and fixture,
             'atlas fixture must be a non-empty path')
    _require(allow_external_fixture or fixture in request['fixtures'],
             'atlas fixture is not in the derivation request')
    _require(atlas.get('selector') == request['shell_selector'],
             'atlas selector differs from the frozen derivation selector')
    _require(atlas.get('reference') == request['reference_ladder']['angular_anchor'],
             'atlas must use the 135:2030 angular anchor')
    _require(atlas.get('directions') == request['directions_per_fixture'],
             'atlas direction count differs from the derivation request')

    boundaries = atlas.get('band_boundaries')
    allowed_boundaries = request['allowed_band_boundaries']
    _require(isinstance(boundaries, list) and len(boundaries) == 4,
             'atlas requires exactly four band boundaries')
    _require(boundaries == sorted(set(boundaries)), 'atlas band boundaries must be strictly increasing')
    _require(set(boundaries) <= set(allowed_boundaries),
             'atlas uses an unapproved physical-radius boundary')

    records = atlas.get('records')
    _require(isinstance(records, list) and records, 'atlas requires at least one record')
    allowed_lebedev = set(request['allowed_lebedev_points'])
    for record in records:
        _require(isinstance(record, dict), 'invalid atlas record')
        _require(isinstance(record.get('atom'), int) and record['atom'] >= 0,
                 'atlas atom index must be non-negative')
        z = record.get('atomic_number')
        _require(isinstance(z, int), 'atlas atomic number must be an integer')
        _require(record.get('family') == family_for_atomic_number(z, request['families']),
                 'atlas family disagrees with atomic number')
        _require(isinstance(record.get('radial_index'), int) and record['radial_index'] >= 0,
                 'atlas radial index must be non-negative')
        _require(_finite(record.get('radius_bohr')) and record['radius_bohr'] > 0.,
                 'atlas radius must be positive and finite')
        u = record.get('u')
        _require(_finite(u) and u > 0., 'atlas scaled radius must be positive and finite')
        _require(record.get('physical_band') == physical_band(u, boundaries),
                 'atlas physical band disagrees with fixed scaled-radius selector')
        _require(record.get('lebedev_points') in allowed_lebedev,
                 'atlas uses an unapproved Lebedev size')
        _require(_finite(record.get('xc_energy')), 'atlas XC energy must be finite')

        directions = record.get('direction_records')
        _require(isinstance(directions, list) and len(directions) == request['directions_per_fixture'],
                 'atlas record must retain every AO direction')
        indices = set()
        for direction in directions:
            _require(isinstance(direction, dict), 'invalid atlas direction record')
            index = direction.get('index')
            _require(isinstance(index, int) and 0 <= index < request['directions_per_fixture'],
                     'invalid atlas direction index')
            _require(index not in indices, 'duplicate atlas direction index')
            indices.add(index)
            channels = direction.get('spin_channels')
            _require(isinstance(channels, list) and channels, 'atlas direction requires spin channels')
            spins = set()
            for channel in channels:
                _require(channel.get('spin') in SPINS and channel['spin'] not in spins,
                         'invalid or duplicate atlas spin channel')
                spins.add(channel['spin'])
                _require(_finite(channel.get('D_V')) and _finite(channel.get('D_fxc_D')),
                         'atlas contractions must be finite')
    return atlas


def load(path, request_path, allow_external_fixture=False):
    return validate(json.loads(Path(path).read_text()), load_request(request_path), allow_external_fixture)
