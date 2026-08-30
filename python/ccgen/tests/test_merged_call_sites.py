"""M5: merging must not change what the emitted residual COMPUTES.

`merge_transposes` (M4, on by default for `--dressing derived`) folds
transpose-equivalent operators onto one shared array. On spatial `ccsdt` that is
288 builders -> 91, and the surviving arrays are then read at MANY distinct index
orders -- `W_t2t2v_oooovv_07fe` is read at 8, `..._16dd` at 12:

    16dd: (i,j,k,l,a,b)  (i,j,k,l,b,a)  (i,k,j,l,a,c)  (j,i,k,l,a,b) ...

`test_emitted_builder_matches_spec` checks each builder against its own spec and
is NOT vacuous under merging (91 checked, 0 bad). But it is definition-only: it
cannot see a CALL SITE that reads a merged array through the wrong permutation,
because the definition is still correct. That is precisely the D4 failure shape
-- every symbolic object exact, the emitted C++ computing a different tensor --
so a definition gate alone is not sufficient once one array serves twelve
readings.

This gate closes it end to end: emit the residual BOTH ways and require the
evaluated arrays to agree. Merging is a pure sharing transform, so any dropped,
inverted, or misapplied call-site permutation moves the residual and shows up
here regardless of which operator or which term carried it.

Uses rank 3, deliberately. `CCGEN_OPERATOR_IDENTITY_AND_REUSE` records that every
`ccsd` merge permutation is a self-inverse two-element swap, so applying one
BACKWARDS is undetectable on that manifold; rank 3 has 3-cycles (see the
`(i,k,j,l,c,a)` reads above), where inverting is observable.
"""
from __future__ import annotations

import re
import unittest

import numpy as np

from ccgen.tests.emitted_cpp_eval import eval_chunks, parse_blocks

from ccgen.tests.test_emitted_builder_matches_spec import _spatial_tensors


NO, NV = 3, 4


def _physical_tensors(no, nv):
    """`_spatial_tensors` plus the amplitude permutation symmetries.

    THIS IS LOAD-BEARING, and it is the opposite of the usual fixture trap. The
    builder gate's fixture is deliberately GENERAL -- it withholds antisymmetry
    from `v` so an invalid ERI relation cannot pass vacuously. But the merge is
    justified by a symmetry the AMPLITUDES really have:

        t2(a,b,i,j) == t2(b,a,j,i)          (and the t3 analogue)

    Two operators that differ only by that swap ARE the same tensor, which is
    why `merge_transposes` may fold them. Under a `t2` without it they are not,
    and this gate reports a defect that does not exist.

    Measured, on the pair the merge plan maps `a049 -> 85b9` at identity:

        t2 random     : max|85b9 - a049| = 9.3e-02   <- false positive
        t2 symmetric  : max|85b9 - a049| = 2.8e-17   <- the truth

    The end-to-end evidence agrees with the second: LiH (62 iterations) and CH4
    (15) are bitwise identical merged vs unmerged, which no real permutation
    defect could survive. So a fixture may only withhold a symmetry the physical
    object LACKS; withholding one it HAS manufactures failures.
    """
    t = _spatial_tensors(no, nv)
    t["t2"] = t["t2"] + np.transpose(t["t2"], (1, 0, 3, 2))
    t3 = t["t3"]
    t["t3"] = (t3
               + np.transpose(t3, (1, 0, 2, 4, 3, 5))
               + np.transpose(t3, (2, 1, 0, 5, 4, 3))
               + np.transpose(t3, (0, 2, 1, 3, 5, 4))
               + np.transpose(t3, (1, 2, 0, 4, 5, 3))
               + np.transpose(t3, (2, 0, 1, 5, 3, 4)))

    # Rank 4 needs t4, and t4 has TWO independent Sz sectors -- the emitted
    # kernels read `t4` and `t4_aaabaaab` as separate amplitude tensors (aaab is
    # not reducible to aabb). Symmetrize each over the simultaneous
    # (vir_k <-> vir_l, occ_k <-> occ_l) swaps, the same paired permutation the
    # t2/t3 blocks carry, so the merge's justifying symmetry is present here too.
    rng = np.random.default_rng(20260829)
    for name in ("t4", "t4_aaabaaab"):
        a = rng.standard_normal((nv,) * 4 + (no,) * 4) * 0.01
        a = a + np.transpose(a, (1, 0, 2, 3, 5, 4, 6, 7))
        a = a + np.transpose(a, (0, 1, 3, 2, 4, 5, 7, 6))
        a = a + np.transpose(a, (2, 3, 0, 1, 6, 7, 4, 5))
        t[name] = a
    return t


def _emit(merge: bool, method: str = "ccsdt") -> str:
    """Emit spatial dressed `ccsdt`, with and without the transpose merge.

    M4 made merging unconditional, so the unmerged arm is produced by patching
    `factorize_equations`'s default for the duration of one call rather than by
    a flag -- the flag was deliberately not kept (no case was found where the
    unmerged form wins, and a knob nothing selects is the accumulation M4 avoided).
    """
    from ccgen.generate import print_cpp_planck
    if merge:
        return print_cpp_planck(method, dressing="derived", spin_adapt=True,
                                force_arbitrary=True)

    from ccgen.optimization import factorize as F
    real = F.factorize_equations

    def unmerged(eqs, **kw):
        kw["merge_transposes"] = False
        return real(eqs, **kw)

    F.factorize_equations = unmerged
    try:
        return print_cpp_planck(method, dressing="derived", spin_adapt=True,
                                force_arbitrary=True)
    finally:
        F.factorize_equations = real


def _part_blocks(src: str, symbol: str, kind: str = 'static void'):
    """Split a `static void ..._partN(...)` chunk into its per-term einsums.

    Two things differ from the shared `parse_blocks`, both consequences of H5's
    chunking: the function returns void and accumulates into a `result` passed
    by reference (so the `Tensor<N>D <symbol>(` header pattern misses), and the
    `// Term N` comments that helper splits on are not emitted inside the parts.
    The BODY shape is unchanged, so split on the `result(...) += ...;` writes
    and walk back to each one's accumulator.
    """
    m = re.search(r'%s %s\(.*?\n\}\n' % (kind, re.escape(symbol)), src, re.S)
    if m is None:
        raise KeyError(f"no emitted part function named {symbol}")
    body = m.group(0)

    out = []
    for w in re.finditer(r'result\(([^)]*)\)\s*\+=\s*([^;]*);', body):
        # Rank >= 7 targets use the runtime-rank BRACED accessor,
        # `result({i, j, k, l, a, b, c, d})`, so the paren capture keeps the
        # braces and yields subscripts like `{i` / `d}`. numpy then rejects the
        # einsum string with a message that names neither the rank nor the
        # brace, which is why this cost a detour -- strip them here.
        tgt = [x.strip().strip('{}') for x in w.group(1).split(',')]
        rhs = w.group(2).strip()
        if rhs == 'acc':
            # accumulator form: take the `acc += <expr>;` just above this write
            head = body[:w.start()]
            acc = re.findall(r'acc\s*\+=\s*([^;]*);', head)
            if not acc:
                continue
            out.append((tgt, acc[-1].strip()))
        else:
            out.append((tgt, rhs))
    return out


def _build_operators(src, method, no, nv, lookup):
    """Materialize every emitted builder, INCLUDING the rank >= 7 ones.

    `build_emitted_operators` keys on `Tensor<N>D build_W_...`, but the emitter
    switches to the runtime-rank type at rank 7 -- `TensorND build_W_...` with
    `TensorND result(std::vector<int>{no, no, ...}, 0.0)`. Those builders are
    therefore invisible to it, and at rank 4 the residual reads plenty of them
    (`W_t3v_ooooovvv_*`, `W_t1t4v_ooooovvv_*`), so the shared helper alone cannot
    evaluate a `ccsdtq` residual at all. Fixed-rank ones still go through the
    shared helper; this adds the TensorND pass on top, ordered by dependency so a
    builder that reads another resolves.
    """
    from ccgen.tests.emitted_cpp_eval import build_emitted_operators

    ops = dict(build_emitted_operators(src, method, no, nv, lookup))

    def resolve(name):
        v = lookup(name)
        return ops.get(name) if v is None else v

    pending = {}
    for m in re.finditer(
            r'TensorND (build_(W_[A-Za-z0-9_]+)_%s)\(' % method, src):
        symbol, name = m.group(1), m.group(2)
        dims = re.search(
            r'TensorND %s\(.*?TensorND result\(std::vector<int>\{([^}]*)\}'
            % re.escape(symbol), src, re.S).group(1)
        shape = tuple(no if d.strip() == 'no' else nv for d in dims.split(','))
        pending[name] = (symbol, shape)

    # Dependency order is not given, so iterate to a fixed point: each pass
    # resolves whatever now has all its inputs. Terminates because the emitted
    # builders form a DAG.
    while pending:
        progressed = False
        for name, (symbol, shape) in list(pending.items()):
            try:
                ops[name] = eval_chunks(_part_blocks(src, symbol, kind='TensorND'),
                                        shape, resolve)
            except KeyError:
                continue
            del pending[name]
            progressed = True
        if not progressed:
            raise AssertionError(
                f"unresolvable TensorND builders: {sorted(pending)[:5]}")
    return ops


def _residuals(src: str, method: str = "ccsdt"):
    """Evaluate the emitted singles, doubles and triples residuals.

    TRIPLES IS NOT OPTIONAL, and leaving it out was this gate's first bug. H5
    splits the triples residual across `_partN` chunks that accumulate into one
    shared `result`, so it is not a single parseable function and the obvious
    gate covers singles+doubles only. But ALL 56 reads of `t2t2v_oooovv` -- the
    family M3 measured at 23.3 % of runtime, merging 38 -> 4, i.e. the whole
    reason to do this -- live in those parts, and none in doubles. Mutation
    testing caught it: perturbing a `t2t2v_oooovv` call site moved a
    singles+doubles-only gate by 2.2e-16, which is to say not at all.

    The parts sum, so evaluating each and adding recovers the residual.
    """
    tensors = _physical_tensors(NO, NV)
    blocks = tensors.pop("_blocks")
    fock = tensors["f"]
    o, w = slice(0, NO), slice(NO, NO + NV)
    env = dict(blocks)
    env.update({"f_oo": fock[o, o], "f_ov": fock[o, w], "f_vv": fock[w, w]})
    # C++ amplitude layout is (occ..., virt...); ccgen's is (vir..., occ...).
    env["t1"] = tensors["t1"].T
    env["t2"] = np.transpose(tensors["t2"], (2, 3, 0, 1))
    env["t3"] = np.transpose(tensors["t3"], (3, 4, 5, 0, 1, 2))
    for name in ("t4", "t4_aaabaaab"):
        if name in tensors:
            env[name] = np.transpose(tensors[name],
                                     (4, 5, 6, 7, 0, 1, 2, 3))

    ops = _build_operators(src, method, NO, NV, env.get)

    def resolve(name):
        return env.get(name, ops.get(name))

    out = {}
    for rank, tag in enumerate(("singles", "doubles", "triples", "quadruples"), 1):
        shape = (NO,) * rank + (NV,) * rank
        whole = f"compute_{method}_{tag}_residual"
        parts = sorted(set(re.findall(re.escape(whole) + r"_part\d+", src)))
        if parts:
            # H5 chunking: the parts accumulate into one shared `result`.
            acc = np.zeros(shape)
            for part in parts:
                acc = acc + eval_chunks(_part_blocks(src, part), shape, resolve)
            out[tag] = acc
        elif re.search(r"Tensor\dD %s\(" % re.escape(whole), src):
            out[tag] = eval_chunks(parse_blocks(src, whole), shape, resolve)
    assert out, f"no residual functions found for {method}"
    return out


class MergedCallSitesTests(unittest.TestCase):
    def _check(self, method):
        try:
            merged = _residuals(_emit(True, method), method)
            plain = _residuals(_emit(False, method), method)
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")

        self.assertEqual(sorted(merged), sorted(plain))
        for tag in merged:
            a, b = merged[tag], plain[tag]
            self.assertEqual(a.shape, b.shape, tag)
            scale = max(float(np.abs(b).max()), 1.0)
            d = float(np.abs(a - b).max())
            self.assertLess(
                d, 1e-10 * scale,
                f"{method} {tag} residual changed under merge_transposes: "
                f"rel={d/scale:.2e}. A merged operator is being read through the "
                "wrong permutation at some call site.")
        return sorted(merged)

    def test_ccsdt_merged_residual_matches_unmerged(self):
        self.assertEqual(self._check("ccsdt"),
                         ["doubles", "singles", "triples"])

    def test_ccsdtq_merged_residual_matches_unmerged(self):
        """Rank 4, where the merge is LARGEST -- 1615 -> 239 builders (6.8x),
        against rank 3's 288 -> 91 (3.2x), with single families as extreme as
        `t2t2v_ooooovvv` at 95 -> 1. Same emit path and the same unconditional
        merge, so rank 4 needs no new mechanism -- but it is a different code
        path with different tensor types, and the accessor work already showed
        rank 3 is not a proxy for rank 4, so it gets its own numeric gate rather
        than an assumption."""
        self.assertEqual(self._check("ccsdtq"),
                         ["doubles", "quadruples", "singles", "triples"])

    def test_the_gate_is_not_vacuous(self):
        """Merging must actually be happening, and the fixture must see it.

        Two ways this gate could pass while checking nothing: the two emits are
        identical (merge not applied), or the residual is zero (no call sites
        evaluated). Both are asserted against.
        """
        try:
            src_m, src_p = _emit(True), _emit(False)
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"ccgen deps unavailable: {exc}")

        import re
        n_m = len(set(re.findall(r"build_(W_[A-Za-z0-9_]+)_ccsdt\(", src_m)))
        n_p = len(set(re.findall(r"build_(W_[A-Za-z0-9_]+)_ccsdt\(", src_p)))
        self.assertLess(n_m, n_p,
                        f"merge emitted no fewer builders ({n_m} vs {n_p}) -- "
                        "the unmerged arm is not actually unmerged")

        # And a merged array must be read at more than one index order, or the
        # permutation property this gate exists for is not present.
        reads = re.findall(r"(W_t2t2v_oooovv_[0-9a-f]+)\(([^)]*)\)", src_m)
        orders = {}
        for name, idx in reads:
            orders.setdefault(name, set()).add(idx.strip())
        self.assertTrue(
            any(len(v) > 1 for v in orders.values()),
            "no merged operator is read at multiple index orders")

        r = _residuals(src_m)
        for tag in ("singles", "doubles", "triples"):
            self.assertGreater(float(np.abs(r[tag]).max()), 1e-8,
                               f"{tag} residual is zero -- gate is vacuous")


if __name__ == "__main__":
    unittest.main()
