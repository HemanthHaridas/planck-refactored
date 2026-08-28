"""Evaluate an emitted Planck CC translation unit numerically, from its TEXT.

Every symbolic gate in this suite validates Python objects; this one validates
what the emitter RENDERED. The distinction is not academic — it is how the
`ooov` builder defect (D4) escaped: the rewrite, the specs, the operator reuse
and the per-term algebra were each exact as Python objects, and the emitted C++
still computed a different tensor.

The emitted code is perfectly regular, so each construct maps onto one einsum:

    for <free> { double acc = 0.0; for <summed> acc += C * f(idx) * ...;
                 result(free) += acc; }

and a driver term omits the accumulator:

    for <free> { result(free) += <expr>; }
"""
from __future__ import annotations

import re

import numpy as np


def parse_blocks(src: str, symbol: str):
    """Split one emitted function body into its `// Term N` / `// Definition
    term N` chunks, returning (target_indices, summed_expr) per chunk."""
    m = re.search(r'Tensor\dD %s\(.*?\n\}' % re.escape(symbol), src, re.S)
    if m is None:
        raise KeyError(f"no emitted function named {symbol}")
    body = m.group(0)
    chunks = re.split(r'// (?:Definition term|Term) \d+', body)[1:]
    out = []
    for c in chunks:
        res = re.search(r'result\(([^)]*)\)\s*\+=', c)
        if res is None:
            continue
        tgt = [x.strip() for x in res.group(1).split(',')]
        if 'double acc = 0.0;' in c:
            expr = c.split('double acc = 0.0;')[1].split('acc +=')[1].split(';')[0]
        else:
            expr = c.split('+=')[1].split(';')[0]
        out.append((tgt, expr.strip()))
    return out


def parse_factors(expr: str):
    """`-0.5 * t2({i,j,a,c}) * W_x(b,c)` -> (coeff, [(name, [idx...]), ...])."""
    coeff = 1.0
    lead = re.match(r'\s*(-?\d+(?:\.\d+)?)\s*\*', expr)
    if lead:
        coeff = float(lead.group(1))
        expr = expr[lead.end():]
    elif expr.strip().startswith('-'):
        coeff = -1.0
        expr = expr.strip()[1:]
    facs = []
    for m in re.finditer(r'(\w+(?:\.\w+)?)\(\{?([^)}]*)\}?\)', expr):
        facs.append((m.group(1).split('.')[-1],
                     [x.strip() for x in m.group(2).split(',')]))
    return coeff, facs


def eval_chunks(chunks, shape, lookup):
    """Sum the einsums for one function's chunks into an array of `shape`."""
    total = np.zeros(shape)
    for tgt, expr in chunks:
        coeff, facs = parse_factors(expr)
        subs, arrs = [], []
        for name, idx in facs:
            arr = lookup(name)
            if arr is None:
                raise KeyError(f"unknown factor {name!r} in {expr!r}")
            arrs.append(arr)
            subs.append(''.join(idx))
        total = total + coeff * np.einsum(
            ','.join(subs) + '->' + ''.join(tgt), *arrs)
    return total


def build_emitted_operators(src: str, method: str, no: int, nv: int, lookup):
    """Materialize every `build_W_*` in `src` by evaluating its emitted body."""
    ops: dict[str, np.ndarray] = {}

    def resolve(name):
        v = lookup(name)
        return ops.get(name) if v is None else v

    for m in re.finditer(r'Tensor\dD (build_(W_[A-Za-z0-9_]+)_%s)\(' % method, src):
        symbol, name = m.group(1), m.group(2)
        dims = re.search(
            r'Tensor\dD %s\(.*?Tensor\dD result\(([^)]*)\)' % re.escape(symbol),
            src, re.S).group(1)
        shape = tuple(no if d.strip() == 'no' else nv
                      for d in dims.split(',')[:-1])
        ops[name] = eval_chunks(parse_blocks(src, symbol), shape, resolve)
    return ops
