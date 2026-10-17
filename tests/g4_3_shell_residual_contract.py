#!/usr/bin/env python3
"""Validate and compare G4.3.4 per-shell angular residual records."""
import math

from g4_3_shell_atlas_contract import validate as validate_atlas


CONTRACT = 'g4-3-shell-residuals-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _record_key(record):
    return record['atom'], record['atomic_number'], record['radial_index']


def _direction_totals(record):
    result = []
    for direction in record['direction_records']:
        result.append(dict(D_V=sum(item['D_V'] for item in direction['spin_channels']),
                           D_fxc_D=sum(item['D_fxc_D'] for item in direction['spin_channels'])))
    return result


def residuals(atlas, request, allow_external_fixture=False):
    """Return every lower-order shell's signed residual versus its 2030 shell."""
    validate_atlas(atlas, request, allow_external_fixture)
    reference = {}
    for record in atlas['records']:
        key = _record_key(record)
        if record['lebedev_points'] == 2030:
            _require(key not in reference, 'duplicate 2030-point shell reference')
            reference[key] = record
    _require(reference, 'missing 2030-point shell reference')
    result = []
    for record in atlas['records']:
        if record['lebedev_points'] == 2030:
            continue
        key = _record_key(record)
        _require(key in reference, 'lower-order shell has no 2030-point reference')
        ref = reference[key]
        _require((record['family'], record['u'], record['physical_band']) ==
                 (ref['family'], ref['u'], ref['physical_band']),
                 'lower-order shell identity differs from its reference')
        directions = []
        for value, ref_value in zip(_direction_totals(record), _direction_totals(ref)):
            directions.append(dict(D_V=value['D_V'] - ref_value['D_V'],
                                   D_fxc_D=value['D_fxc_D'] - ref_value['D_fxc_D']))
        result.append(dict(key=key, lebedev_points=record['lebedev_points'],
                           xc_energy=record['xc_energy'] - ref['xc_energy'], directions=directions))
    return result


def validate(table, request, allow_external_fixture=False):
    _require(table.get('contract') == CONTRACT, 'unsupported shell-residual contract')
    atlas = dict(table)
    atlas['contract'] = 'g4-3-shell-atlas-v1'
    values = residuals(atlas, request, allow_external_fixture)
    _require(values, 'shell residual table has no lower-order records')
    return values
