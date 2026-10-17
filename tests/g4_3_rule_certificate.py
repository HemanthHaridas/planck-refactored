#!/usr/bin/env python3
"""G4.3.7 separable family-level L1 shell-residual certificate."""
from g4_3_rule_grammar import select_band, validate_rule


CONTRACT = 'g4-3-rule-certificate-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def certificate(rule, family, tables, request):
    """Score one candidate rule with no cancellation across shells.

    Each returned fixture row keeps signed sums as diagnostics and L1 sums as
    the acceptance quantity.  A 2030 selection contributes exact zero.
    """
    validate_rule(rule, request)
    _require(family in {item['name'] for item in request['families']}, 'unknown periodic family')
    rows = []
    for table in tables:
        references = {(record['atom'], record['atomic_number'], record['radial_index']): record
                      for record in table['records'] if record['lebedev_points'] == 2030 and record['family'] == family}
        residuals = {(tuple(item['key']), item['lebedev_points']): item for item in table['residuals']}
        energy_signed = energy_l1 = 0.0
        directions = [dict(signed_D_V=0.0, l1_D_V=0.0, signed_D_fxc_D=0.0, l1_D_fxc_D=0.0)
                      for _ in range(request['directions_per_fixture'])]
        for key, record in references.items():
            selected = rule['lebedev_points'][select_band(record['u'], rule['boundaries'])]
            if selected == 2030:
                continue
            value = residuals.get((key, selected))
            _require(value is not None, 'selected shell has no residual record')
            energy_signed += value['xc_energy']
            energy_l1 += abs(value['xc_energy'])
            for ordinal, direction in enumerate(value['directions']):
                target = directions[ordinal]
                target['signed_D_V'] += direction['D_V']
                target['l1_D_V'] += abs(direction['D_V'])
                target['signed_D_fxc_D'] += direction['D_fxc_D']
                target['l1_D_fxc_D'] += abs(direction['D_fxc_D'])
        _require(abs(energy_signed) <= energy_l1 + 1e-18, 'energy triangle inequality violated')
        for direction in directions:
            _require(abs(direction['signed_D_V']) <= direction['l1_D_V'] + 1e-18,
                     'potential triangle inequality violated')
            _require(abs(direction['signed_D_fxc_D']) <= direction['l1_D_fxc_D'] + 1e-18,
                     'kernel triangle inequality violated')
        rows.append(dict(fixture=table['fixture'], signed_xc_energy=energy_signed, l1_xc_energy=energy_l1,
                         directions=directions))
    return dict(contract=CONTRACT, family=family, rule=rule, fixtures=rows)


def passes(value, request):
    energy_limit = request['limits']['xc_energy_ha']
    response_limit = request['limits']['xc_response_contraction']
    return all(row['l1_xc_energy'] <= energy_limit and
               all(direction['l1_D_V'] <= response_limit and direction['l1_D_fxc_D'] <= response_limit
                   for direction in row['directions']) for row in value['fixtures'])


def qualifying_rules(rules, family, tables, request):
    """Filter an arbitrary lazy rule stream through the full L1 certificate."""
    accepted = []
    for rule in rules:
        value = certificate(rule, family, tables, request)
        if passes(value, request):
            accepted.append(value)
    return accepted


def prepare_residual_index(tables):
    """Index immutable report data once for exhaustive boundary searches."""
    result = []
    for table in tables:
        references = {}
        for record in table['records']:
            if record['lebedev_points'] == 2030:
                references.setdefault(record['family'], []).append(record)
        residuals = {(tuple(item['key']), item['lebedev_points']): item for item in table['residuals']}
        result.append((references, residuals))
    return result


def band_l1_vectors(family, boundaries, tables, request, index=None):
    """Precompute nonnegative L1 contributions by band and Lebedev order.

    Keys are ``(band, lebedev_points)``; every value is a fixture/channel
    vector.  These are additive, so a partial candidate exceeding a limit can
    be rejected safely before its remaining bands are selected.
    """
    _require(family in {item['name'] for item in request['families']}, 'unknown periodic family')
    index = prepare_residual_index(tables) if index is None else index
    result = {}
    for band in range(5):
        for level in request['allowed_lebedev_points']:
            rows = []
            for references, residuals in index:
                refs = [record for record in references.get(family, [])
                        if select_band(record['u'], boundaries) == band]
                value = dict(xc_energy=0.0, directions=[dict(D_V=0.0, D_fxc_D=0.0)
                                                         for _ in range(request['directions_per_fixture'])])
                if level != 2030:
                    for record in refs:
                        key = (record['atom'], record['atomic_number'], record['radial_index'])
                        item = residuals[(key, level)]
                        value['xc_energy'] += abs(item['xc_energy'])
                        for ordinal, direction in enumerate(item['directions']):
                            value['directions'][ordinal]['D_V'] += abs(direction['D_V'])
                            value['directions'][ordinal]['D_fxc_D'] += abs(direction['D_fxc_D'])
                rows.append(value)
            result[band, level] = rows
    return result


def _add_vectors(left, right):
    return [dict(xc_energy=a['xc_energy'] + b['xc_energy'], directions=[
        dict(D_V=x['D_V'] + y['D_V'], D_fxc_D=x['D_fxc_D'] + y['D_fxc_D'])
        for x, y in zip(a['directions'], b['directions'])]) for a, b in zip(left, right)]


def _within_limits(rows, request):
    energy_limit = request['limits']['xc_energy_ha']
    response_limit = request['limits']['xc_response_contraction']
    return all(row['xc_energy'] <= energy_limit and
               all(direction['D_V'] <= response_limit and direction['D_fxc_D'] <= response_limit
                   for direction in row['directions']) for row in rows)


def qualifying_for_boundaries(family, boundaries, tables, request, index=None):
    """Yield L1-qualified rules for one boundary tuple using safe early pruning."""
    vectors = band_l1_vectors(family, boundaries, tables, request, index)
    levels = request['allowed_lebedev_points']
    zero = [dict(xc_energy=0.0, directions=[dict(D_V=0.0, D_fxc_D=0.0)
                                             for _ in range(request['directions_per_fixture'])])
            for _ in tables]
    for peak_index, peak in enumerate(levels):
        for n0_index in range(peak_index + 1):
            partial0 = _add_vectors(zero, vectors[0, levels[n0_index]])
            if not _within_limits(partial0, request):
                continue
            for n1_index in range(n0_index, peak_index + 1):
                partial2 = _add_vectors(_add_vectors(partial0, vectors[1, levels[n1_index]]),
                                        vectors[2, peak])
                if not _within_limits(partial2, request):
                    continue
                for n3_index in range(peak_index + 1):
                    partial3 = _add_vectors(partial2, vectors[3, levels[n3_index]])
                    if not _within_limits(partial3, request):
                        continue
                    for n4_index in range(n3_index + 1):
                        total = _add_vectors(partial3, vectors[4, levels[n4_index]])
                        if _within_limits(total, request):
                            yield {'boundaries': list(boundaries),
                                   'lebedev_points': [levels[n0_index], levels[n1_index], peak,
                                                      levels[n3_index], levels[n4_index]]}


def family_cost_model(family, tables, request):
    """Fixed shell-occupancy weights for the declared family cost vector."""
    atoms = []
    for table in tables:
        for record in table['records']:
            if record['lebedev_points'] == 2030 and record['family'] == family:
                atoms.append((table['fixture'], record['atom'], record))
    return atoms


def family_rule_cost(rule, model):
    atom_weights = {}
    for fixture, atom, record in model:
        key = fixture, atom
        band = select_band(record['u'], rule['boundaries'])
        atom_weights[key] = atom_weights.get(key, 0) + rule['lebedev_points'][band]
    return max(atom_weights.values(), default=0), sum(atom_weights.values())


def boundary_cost_weights(boundaries, model):
    """Group fixed shell occupancies once for all rules sharing boundaries."""
    atoms = {}
    for fixture, atom, record in model:
        key = fixture, atom
        weights = atoms.setdefault(key, [0, 0, 0, 0, 0])
        weights[select_band(record['u'], boundaries)] += 1
    return list(atoms.values())


def weighted_rule_cost(rule, atom_weights):
    counts = rule['lebedev_points']
    values = [sum(weight * count for weight, count in zip(weights, counts)) for weights in atom_weights]
    return max(values, default=0), sum(values)


def compact_cost_frontier(rules, model):
    """Stream rules to one lexical representative per non-dominated cost pair."""
    representatives = {}
    for rule in rules:
        representatives.setdefault(family_rule_cost(rule, model), rule)
    best_total = float('inf')
    frontier = []
    for cost, rule in sorted(representatives.items()):
        if cost[1] < best_total:
            frontier.append({'max_points_per_atom': cost[0], 'total_ledger_points': cost[1], 'rule': rule})
            best_total = cost[1]
    return frontier


def compact_frontier_from_representatives(representatives):
    """Reduce precomputed ``(max_atom,total) -> rule`` entries without rescoring."""
    best_total = float('inf')
    frontier = []
    for cost, rule in sorted(representatives.items()):
        if cost[1] < best_total:
            frontier.append({'max_points_per_atom': cost[0], 'total_ledger_points': cost[1], 'rule': rule})
            best_total = cost[1]
    return frontier
