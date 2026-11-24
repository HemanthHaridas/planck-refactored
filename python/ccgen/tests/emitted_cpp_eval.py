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


# A lowered contraction (`cc_gemm_contract`, R3) is the same einsum as the
# loop nest it replaced: out(labels) += coeff * A(a_labels) * B(b_labels).
# The trailing `pair_half` flag only narrows loop bounds, which this
# evaluator ignores anyway.
GEMM_CALL = re.compile(
    r'cc_gemm_contract\((\w+), "(\w+)", ([-+.\de]+), cc_operand\(([\w.]+)\), "(\w+)", '
    r'cc_operand\(([\w.]+)\), "(\w+)", "\w+", "\w+", "\w+"(?:, true)?\);')


def gemm_write(m):
    """A `GEMM_CALL` match -> (lhs, target indices, accumulation expression)."""
    lhs, out, coeff, a, al, b, bl = m.groups()
    return (lhs, list(out),
            f"{float(coeff)} * {a}({', '.join(al)}) * {b}({', '.join(bl)})")


def gemm_reads(src: str):
    """(operand name, index labels) for every operand of a lowered call."""
    out = []
    for m in GEMM_CALL.finditer(src):
        out.append((m.group(4).split('.')[-1], m.group(5)))
        out.append((m.group(6).split('.')[-1], m.group(7)))
    return out


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
        g = GEMM_CALL.search(c)
        if g is not None:
            _, tgt, expr = gemm_write(g)
            out.append((tgt, expr))
            continue
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


def _dims(text: str, no: int, nv: int) -> tuple[int, ...]:
    """`no, no, nv, 0.0` or `std::vector<int>{no, nv}, 0.0` -> a shape."""
    names = re.findall(r'\b(no|nv)\b', text)
    return tuple(no if d == 'no' else nv for d in names)


def eval_body(body: str, no: int, nv: int, resolve):
    """Evaluate one emitted function body, flat or factored, in order.

    A factored builder (M3.0) writes scratch steps `Xn(...) += ...` before
    `result`, and carries no `// Definition term` markers. Each write is one
    einsum; a scratch tensor is resolved locally once written. Returns the
    `result` array.
    """
    local: dict[str, np.ndarray] = {}
    decl = re.search(r'(?:Tensor\dD|TensorND) result\(([^;]*)\);', body)
    local['result'] = np.zeros(_dims(decl.group(1), no, nv))
    for m in re.finditer(r'(?:Tensor\dD|TensorND) (X\d+)\(([^;]*)\);', body):
        local[m.group(1)] = np.zeros(_dims(m.group(2), no, nv))

    def look(name):
        return local[name] if name in local else resolve(name)

    writes = []
    for w in re.finditer(r'\b(result|X\d+)\(\{?([^)}]*)\}?\)\s*\+=\s*([^;]*);', body):
        lhs, tgt, rhs = w.group(1), [x.strip() for x in w.group(2).split(',')], w.group(3).strip()
        if rhs == 'acc':
            rhs = re.findall(r'acc\s*\+=\s*([^;]*);', body[:w.start()])[-1].strip()
        writes.append((w.start(), lhs, tgt, rhs))
    for g in GEMM_CALL.finditer(body):
        writes.append((g.start(), *gemm_write(g)))
    for _, lhs, tgt, rhs in sorted(writes, key=lambda w: w[0]):
        local[lhs] = local[lhs] + eval_chunks([(tgt, rhs)], local[lhs].shape, look)
    return local['result']


def build_emitted_operators(src: str, method: str, no: int, nv: int, lookup):
    """Materialize every `build_W_*` in `src` (flat or factored body), in
    emission order, which is dependency order."""
    ops: dict[str, np.ndarray] = {}

    def resolve(name):
        v = lookup(name)
        return ops.get(name) if v is None else v

    for m in re.finditer(
            r'(?:Tensor\dD|TensorND) build_(W_[A-Za-z0-9_]+)_%s\(.*?\n\}\n' % method,
            src, re.S):
        ops[m.group(1)] = eval_body(m.group(0), no, nv, resolve)
    return ops
