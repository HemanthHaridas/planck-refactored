"""Λ step 3b: the emitted Λ kernels compute the Λ residual (docs/CC_LAMBDA_SCOPE.md).

The Python Λ equations are gated against PySCF and by complex step
(test_lambda_{equations,spatial,ucc}.py). This gate checks what the emitter
RENDERED from them: the emitted C++ text, evaluated term by term with numpy
(emitted_cpp_eval), against the Python Λ residual on the same tensors,

    R = Σ_g s_g g(G),   G = ∂L/∂t,   g over the amplitude block's group,

for every residual block: restricted ranks 2-4 (both rank-4 Sz sectors) and
UCC ranks 2-3. The emit runs the
same adaptation, folding, factorizer and GEMM lowering as the T kernels, so a
dropped term, a wrong permutation, a wrong symmetrizer or λ bound to the wrong
argument shows here.

Fixtures carry the symmetry production amplitudes have: every t and λ block is
projected onto its group. UCC integrals are stored as Planck stores them, plain
<pq|rs> per spin block (the emitted same-spin reads form the antisymmetrized
difference themselves), while the Python terms read <pq||rs>.
"""
from __future__ import annotations

import re
import unittest

import numpy as np

from ccgen.tests.emitted_cpp_eval import build_emitted_operators, eval_chunks, parse_blocks
from ccgen.tests.residual_eval import residual_einsum, ucc_residual_einsum
from ccgen.tests.test_emitted_builder_matches_spec import _spatial_tensors
from ccgen.tests.test_merged_call_sites import _apply_symmetrize, _part_blocks

NO, NV = 3, 4
UCC_DIMS = {"noa": 4, "nob": 3, "nva": 3, "nvb": 4}


def _image(x, names, mapping):
    src = "".join(mapping.get(n, n) for n in names)
    return np.einsum(f"{src}->{''.join(names)}", x)


def _symmetrize(x, names, group):
    return sum(s * _image(x, names, m) for m, s in group)


def _to_vir_first(x, n):
    return np.transpose(x, list(range(n, 2 * n)) + list(range(n)))


def _blocks(method, ucc):
    """Per residual key: (rank, amplitude suffix, free names, group, G terms)."""
    from ccgen.generate import generate_cc_equations
    from ccgen.lagrangian import lambda_residual_terms
    from ccgen.optimization.permutation import block_symmetry_group
    from ccgen.spin import spin_adapt_equations, ucc_adapt_equations

    # The production engine: the emit below uses it, and the reference must
    # differentiate the same equations.
    eqs = generate_cc_equations(method, engine="diagram", canonical_fock=True)
    eqs = ucc_adapt_equations(eqs) if ucc else spin_adapt_equations(eqs)
    G = lambda_residual_terms(eqs, "ucc" if ucc else "rcc", occ_first=True)
    out = {}
    for key, terms in G.items():
        n = len(terms[0].free_indices) // 2
        tag = key.split("_", 1)[1] if "_" in key else ""
        k_alpha = tag[:n].count("a") if (tag and not ucc) else None
        out[key] = (n, f"{n}_{tag}" if tag else f"{n}",
                    [i.name for i in terms[0].free_indices],
                    block_symmetry_group(terms[0], ucc, k_alpha), terms)
    return out


def _amplitudes(blocks, extent, rng):
    """t (vir-first, as ccgen) and λ (occ-first, as ccgen and C++) per block,
    each projected onto its block's group; plus the C++ (occ-first) view."""
    py, cpp = {}, {}
    for n, suffix, names, group, _ in blocks.values():
        shape = tuple(extent(name) for name in names)
        for root, scale in (("t", 0.05), ("l", 0.1)):
            x = _symmetrize(rng.standard_normal(shape) * scale, names, group) / len(group)
            cpp[f"{root}{suffix}"] = x
            py[f"{root}{suffix}"] = _to_vir_first(x, n) if root == "t" else x
    return py, cpp


def _emitted(src, method, ucc, blocks, sizes, env):
    ops = build_emitted_operators(src, ("ucc_" if ucc else "") + "lambda_" + method,
                                  sizes, 0, env.get)

    def resolve(name):
        return env.get(name, ops.get(name))

    out = {}
    for key, (n, _, names, _, _) in blocks.items():
        shape = tuple(sizes[_extent_name(nm, key, n, ucc)] for nm in names)
        kernel = f"compute_{'ucc_' if ucc else ''}lambda_{method}_{key}_residual"
        parts = sorted(set(re.findall(re.escape(kernel) + r"_part\d+", src)))
        if parts:
            x = sum(eval_chunks(_part_blocks(src, p), shape, resolve) for p in parts)
        else:
            x = eval_chunks(parse_blocks(src, kernel), shape, resolve)
        out[key] = _apply_symmetrize(src, kernel, x)
    return out


def _extent_name(name, key, n, ucc):
    """The extent of free index *name* of block *key*: no/nv, or for UCC
    noa/nob/nva/nvb from the block tag (first half the virtual spins, second
    half the occupied ones, slot k carrying the k-th of its space)."""
    space = "o" if name in "ijklmno" else "v"
    if not ucc:
        return "n" + space
    tag = key.split("_", 1)[1]
    pool = "ijklmno" if space == "o" else "abcdefgh"
    half = tag[n:] if space == "o" else tag[:n]
    return f"n{space}{half[pool.index(name)]}"


def _rcc_case(method):
    blocks = _blocks(method, False)
    base = _spatial_tensors(NO, NV)
    sizes = {"no": NO, "nv": NV}
    py, cpp = _amplitudes(
        blocks, lambda name: NO if name in "ijklmno" else NV, np.random.default_rng(20261126))
    fock = base["f"]
    o, w = slice(0, NO), slice(NO, NO + NV)
    env = dict(base["_blocks"])
    env.update({"f_oo": fock[o, o], "f_ov": fock[o, w], "f_vv": fock[w, w]}, **cpp)
    tensors = {"v": base["v"], "f": fock, **py}
    reference = {}
    for key, (n, _, names, group, terms) in blocks.items():
        g = sum(residual_einsum(t, NO, NV, tensors) for t in terms)
        reference[key] = _symmetrize(np.transpose(g, list(range(n, 2 * n)) + list(range(n))),
                                     names, group)
    return blocks, sizes, env, reference


def _ucc_case(method):
    blocks = _blocks(method, True)
    rng = np.random.default_rng(20261127)
    d = UCC_DIMS
    na, nb = d["noa"] + d["nva"], d["nob"] + d["nvb"]

    def chem(n1, n2, same):
        c = rng.standard_normal((n1, n1, n2, n2)) * 0.1
        c = c + c.transpose(1, 0, 2, 3)
        c = c + c.transpose(0, 1, 3, 2)
        if same:
            c = c + c.transpose(2, 3, 0, 1)
        return np.einsum("prqs->pqrs", c)        # <pq|rs> = (pr|qs), real orbitals

    g = {"aaaa": chem(na, na, True), "abab": chem(na, nb, False), "bbbb": chem(nb, nb, True)}
    f = {}
    for spin, n, no in (("aa", na, d["noa"]), ("bb", nb, d["nob"])):
        x = rng.standard_normal((n, n))
        x = x + x.T
        x[:no, no:] = x[no:, :no] = 0.0         # canonical reference: f_ov = 0
        f[spin] = x
    sizes = dict(d)

    def extent_of(key):
        n = blocks[key][0]
        return lambda name: sizes[_extent_name(name, key, n, True)]

    py, cpp = {}, {}
    for key in blocks:
        p, c = _amplitudes({key: blocks[key]}, extent_of(key), rng)
        py.update(p)
        cpp.update(c)

    def sl(spin, space):
        no = d["noa"] if spin == "a" else d["nob"]
        return slice(0, no) if space == "o" else slice(no, None)

    env = dict(cpp)
    for tag, arr in g.items():
        for pat in ("".join("ov"[k] for k in p) for p in np.ndindex(2, 2, 2, 2)):
            env[f"v_{tag}_{pat}"] = arr[tuple(sl(s, x) for s, x in zip(tag, pat))]
    for tag, arr in f.items():
        for pat in ("oo", "vv", "ov"):
            env[f"f_{tag}_{pat}"] = arr[tuple(sl(s, x) for s, x in zip(tag, pat))]

    tensors = {"v_aaaa": g["aaaa"] - g["aaaa"].transpose(0, 1, 3, 2),
               "v_abab": g["abab"],
               "v_bbbb": g["bbbb"] - g["bbbb"].transpose(0, 1, 3, 2),
               "f_aa": f["aa"], "f_bb": f["bb"], **py}
    reference = {}
    for key, (n, _, names, group, terms) in blocks.items():
        x = sum(ucc_residual_einsum(t, d, tensors) for t in terms)
        reference[key] = _symmetrize(np.transpose(x, list(range(n, 2 * n)) + list(range(n))),
                                     names, group)
    return blocks, sizes, env, reference


def _emit(method, ucc, dressing="derived"):
    from ccgen.generate import print_cpp_planck
    return print_cpp_planck(method, dressing=dressing, spin_adapt=not ucc, ucc=ucc,
                            force_arbitrary=True, lambda_=True, engine="diagram",
                            canonical_fock=True)


class LambdaEmitTests(unittest.TestCase):
    def _check(self, method, ucc, dressing="derived", mutate=None):
        blocks, sizes, env, want = (_ucc_case if ucc else _rcc_case)(method)
        src = _emit(method, ucc, dressing)
        if mutate:
            src = mutate(src)
        got = _emitted(src, method, ucc, blocks, sizes, env)
        worst = 0.0
        for key in blocks:
            scale = float(np.abs(want[key]).max())
            self.assertGreater(scale, 1e-3, f"{key}: reference is ~0, gate is vacuous")
            worst = max(worst, float(np.abs(got[key] - want[key]).max()) / scale)
        return worst, src

    def test_rcc_ccsd(self):
        worst, src = self._check("ccsd", False)
        self.assertLess(worst, 1e-12)
        self.assertTrue(re.search(r"build_W_l\w*_lambda_ccsd\(", src))
        self.assertIn("const ArbitraryOrderRCCAmplitudes &lambdas", src)

    def test_rcc_ccsd_undressed(self):
        self.assertLess(self._check("ccsd", False, dressing="none")[0], 1e-12)

    def test_rcc_ccsdt(self):
        self.assertLess(self._check("ccsdt", False)[0], 1e-12)

    def test_rcc_ccsdtq(self):
        """Rank 4: two Sz sectors (`quadruples`, aabb, with the spin flip;
        `quadruples_aaabaaab`, without), rank-8 TensorND kernels in parts."""
        self.assertLess(self._check("ccsdtq", False)[0], 1e-12)

    def test_ucc_ccsd(self):
        self.assertLess(self._check("ccsd", True)[0], 1e-12)

    def test_ucc_ccsdt(self):
        self.assertLess(self._check("ccsdt", True)[0], 1e-12)

    def test_gate_fails_on_a_wrong_symmetrizer(self):
        """Dropping the pair image from the doubles symmetrizer (code and the
        comment the evaluator reads) must fail: the gate sees the group."""
        def drop(src):
            return src.replace("// symmetrize ijab: +ijab +jiba", "// symmetrize ijab: +ijab", 1)
        self.assertGreater(self._check("ccsd", False, mutate=drop)[0], 1e-3)


if __name__ == "__main__":
    unittest.main()
