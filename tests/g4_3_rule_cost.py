#!/usr/bin/env python3
"""Offline G4.3.6 point-cost evaluator for fixed shell-atlas reports."""
import json
from pathlib import Path

from g4_3_rule_grammar import select_band, validate_rule


CONTRACT = 'g4-3-rule-cost-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_reports(paths, request):
    """Load exactly one normalized residual table for every frozen fixture."""
    tables = [json.loads(Path(path).read_text()) for path in paths]
    fixtures = [table.get('fixture') for table in tables]
    _require(set(fixtures) == set(request['fixtures']) and len(fixtures) == len(request['fixtures']),
             'reports must cover every frozen fixture exactly once')
    for table in tables:
        _require(table.get('contract') == 'g4-3-shell-residuals-v1',
                 'unsupported shell-residual report')
    return tables


def reference_records(table):
    records = [record for record in table['records'] if record['lebedev_points'] == 2030]
    _require(records, 'shell report lacks 2030-point reference records')
    return records


def rule_cost(family_rules, tables, request, validate_candidates=True):
    """Compute the declared point-cost vector from fixed 2030 shell identities."""
    names = {family['name'] for family in request['families']}
    _require(set(family_rules) == names, 'must provide one rule for every periodic family')
    if validate_candidates:
        for rule in family_rules.values():
            validate_rule(rule, request)
    per_fixture = []
    total = 0
    maximum_atom = 0
    for table in tables:
        atom_points = {}
        for record in reference_records(table):
            rule = family_rules[record['family']]
            count = rule['lebedev_points'][select_band(record['u'], rule['boundaries'])]
            atom_points[record['atom']] = atom_points.get(record['atom'], 0) + count
        fixture_total = sum(atom_points.values())
        total += fixture_total
        maximum_atom = max(maximum_atom, *atom_points.values())
        per_fixture.append({'fixture': table['fixture'], 'total_points': fixture_total,
                            'max_atom_points': max(atom_points.values())})
    return {'contract': CONTRACT, 'max_points_per_atom': maximum_atom,
            'total_ledger_points': total, 'fixtures': per_fixture}


def dominates(left, right):
    """Strict Pareto dominance in the declared two-component cost vector."""
    keys = ('max_points_per_atom', 'total_ledger_points')
    return all(left[key] <= right[key] for key in keys) and any(left[key] < right[key] for key in keys)


def pareto_frontier(rows):
    """Return deterministic non-dominated rows; scoring enters only in G4.3.7."""
    return [row for index, row in enumerate(rows)
            if not any(other_index != index and dominates(other['cost'], row['cost'])
                       for other_index, other in enumerate(rows))]
