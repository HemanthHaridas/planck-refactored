"""R2/R2b: merging terms that read one operand the same way must not change
the residual.

`merge_common_operand_reads` rewrites sum_n c_n A(P) R_n as A(P) Y with
Y = sum_n c_n R_n. R2 runs it on t2, comparing reads up to the spatial
symmetry t2(a,b,i,j) = t2(b,a,j,i); R2b on v, seeing through intermediates
read once, which turns the ladder into tau = t2 + t1 t1
(docs/CCSD_T_SIZE_SCOPE.md). Emit the residual with and without both
passes and require the evaluated arrays to agree, on a fixture whose t2 has
that symmetry (and only that one).
"""
from __future__ import annotations

import re
import unittest

import numpy as np

from ccgen.tests.test_merged_call_sites import _residuals


def _emit(merge: bool, method: str) -> str:
    from ccgen.generate import print_cpp_planck
    from ccgen.optimization import factorize as F
    real = F.merge_common_operand_reads
    if not merge:
        F.merge_common_operand_reads = (
            lambda terms, target, **kw: (list(terms), []))
    try:
        return print_cpp_planck(method, dressing="derived", spin_adapt=True,
                                force_arbitrary=True)
    finally:
        F.merge_common_operand_reads = real


class MergedAmplitudeReadsTests(unittest.TestCase):
    def _check(self, method):
        merged_src, plain_src = _emit(True, method), _emit(False, method)
        n_merge = len(set(re.findall(r"build_(W_t2merge_\w+?)_%s\(" % method,
                                     merged_src)))
        self.assertGreater(n_merge, 0, "no merged t2 reads emitted")
        if method == "ccsd":
            # In ccsdt the t1 v(vvvv) intermediate is also read by the triples,
            # so it is not single-use and the ladder stays unmerged there.
            self.assertIn("build_W_vmerge_", merged_src,
                          "no merged v reads emitted")
        self.assertNotIn("W_t2merge_", plain_src)
        self.assertNotIn("W_vmerge_", plain_src)
        merged = _residuals(merged_src, method)
        plain = _residuals(plain_src, method)
        self.assertEqual(sorted(merged), sorted(plain))
        for tag in merged:
            a, b = merged[tag], plain[tag]
            scale = max(float(np.abs(b).max()), 1.0)
            self.assertGreater(float(np.abs(b).max()), 1e-8, f"{tag} is zero")
            d = float(np.abs(a - b).max())
            self.assertLess(d, 1e-10 * scale,
                            f"{method} {tag} changed under the t2-read merge: "
                            f"rel={d / scale:.2e}")
        return n_merge

    def test_ccsd(self):
        self._check("ccsd")

    def test_ccsdt(self):
        self._check("ccsdt")


if __name__ == "__main__":
    unittest.main()
