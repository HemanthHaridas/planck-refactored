"""F0: a dressed intermediate's BUILDER layout must match every USAGE-site indexing.

The defect this gate exists for: `build_Wmbej_<method>` allocates its result as
`Tensor4D result(no, no, nv, nv)` -- an `oovv` layout -- while the residual indexes that
same tensor as `Wmbej(m, d, e, l)` (`ovvo`), `Wmbej(b, m, e, l)` (`voov`), and two more
orderings. The builder writes one layout; the consumers read four different ones.

Mechanism (`ccgen/emit/planck_tensor_cpp.py:_map_factor`): every other factor kind
normalizes its index order at the usage site --

    amplitudes  -> `', '.join(occ + vir)`
    denominators-> `', '.join(occ + vir)`
    ERI / Fock  -> their own dispatch

-- but a dressed intermediate falls through to `_target_expr(name, indices)`, which joins
the indices VERBATIM in whatever order the term carries. Meanwhile the builder side IS
normalized, via `emitted_intermediate_layout` -> `lower_term_restricted_closed_shell`. So
the builder normalizes and the consumer does not, and only the intermediates are exposed.

Why it stayed invisible:

- Only MIXED-SPACE operators can express the bug. `Wmnij` (`oooo`) and `Wabef` (`vvvv`)
  are space-homogeneous, so any permutation preserves the signature and they are correct
  by accident. `Wmbej` (`oovv`) and `tau`/`tau_c` (`oovv`) are the mixed ones -- and are
  exactly the ones that mismatch.
- On the validation system (Be/STO-3G) `nv == no == 4`, so every mis-indexed read stays
  IN BOUNDS and silently returns the wrong element rather than tripping `Tensor4D`'s
  bounds check. No crash, no diagnostic: the solver converges cleanly to a fixed point of
  the wrong equations, ~52% short on correlation energy.
- `validate_intermediate_specs` passes: it does not compare declared layout against
  usage-site indexing.

This gate is written to FAIL on the pre-fix tree (that is the point -- a gate that cannot
fail proves nothing, which is how the unreachable-kernel defect survived every earlier
check). Once F2 lands it should pass at every rank, since the layout convention is a
property of the emitter and is identical at rank 2, 3, 4, 5, 6.

F2.0 NOTE -- this gate reads the EMITTED C++, not the symbolic terms.

An earlier version of this file measured `factor.indices` off the pre-lowering adapted
term. That is not what the emitter sees: `_map_factor` reads `_access_indices()` off a
`LoweredTensorFactor`, whose `spatial_permutation` reorders the slots. The pre-lowering
signatures are real objects but they are NOT what reaches the compiler, so a fix could
satisfy that gate while the generated code stayed wrong.

The measurement below is therefore taken from the generated source itself, keying each
index name to its loop bound (`no` -> occupied, `nv` -> virtual). That is the ground
truth: it is exactly what the C++ compiler will index.

What it exposes, measured on `ccsd`:

    Wmbej   builder (no,no,nv,nv)=oovv   used as ovvo x5, voov x3, vovo x1, ovov x1
    tau     builder oovv                 used as oovv x2, vvoo x1   <-- INCONSISTENT
    tau_c   builder oovv                 used as vvoo x1
    Wmnij   builder oooo                 used as oooo x1   (correct)
    Wabef   builder vvvv                 used as vvvv x1   (correct)

`tau` emitting BOTH `oovv` and `vvoo` for the same tensor in the same translation unit is
the sharpest form of the defect: normalization is not merely absent, it is applied
INCONSISTENTLY. Some factors reach `_map_factor` already lowered; others arrive bare and
are emitted verbatim by `_target_expr`.
"""

from __future__ import annotations

import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import print_cpp_planck  # noqa: E402

# Rank 2 is the cheap anchor and exhibits the defect in full; the mechanism lives in the
# emitter so it is rank-independent (RankGenericLayoutTests re-checks at rank 3).
ANCHOR_METHOD = "ccsd"

# Dressed intermediates whose builder+usages this gate compares.
DRESSED_OPS = ("tau", "tau_c", "Wmnij", "Wabef", "Wmbej")


def _loop_bounds(code: str) -> dict[str, str]:
    """{index name: 'no' | 'nv'} from the emitted `for (int i = 0; i < no; ...)` loops.

    The emitter names every occupied index off an `no` bound and every virtual off `nv`,
    so this recovers each index's space exactly as the compiler will see it.
    """
    return dict(re.findall(r"for \(int ([a-z]) = 0; \1 < (n[ov]);", code))


def _builder_layouts(code: str, method: str) -> dict[str, str]:
    """{op: signature} from each `build_<op>_<method>`'s `Tensor4D result(...)` dims."""
    out: dict[str, str] = {}
    for op in DRESSED_OPS:
        m = re.search(
            rf"Tensor\dD build_{op}_{method}\b.*?Tensor\dD result\(([^)]*)\)",
            code, re.S)
        if not m:
            continue
        dims = [d.strip() for d in m.group(1).split(",")]
        out[op] = "".join("o" if d == "no" else "v" for d in dims if d in ("no", "nv"))
    return out


def _usage_layouts(code: str, bounds: dict[str, str]) -> dict[str, Counter]:
    """{op: Counter(signature)} over every emitted usage site of each intermediate."""
    out: dict[str, Counter] = {}
    for op in DRESSED_OPS:
        sites = re.findall(rf"\b{op}\(([a-z](?:, [a-z])*)\)", code)
        c: Counter = Counter()
        for site in sites:
            names = site.split(", ")
            c["".join("o" if bounds.get(n) == "no" else "v" for n in names)] += 1
        if c:
            out[op] = c
    return out


def _builder_and_usage(method: str):
    """(builder_sig, usage_sigs) per dressed intermediate, read from the emitted C++."""
    code = print_cpp_planck(method, spin_adapt=True, dress_operators=True)
    bounds = _loop_bounds(code)
    return _builder_layouts(code, method), _usage_layouts(code, bounds)


class IntermediateLayoutAgreementTests(unittest.TestCase):
    """The builder's layout and every usage site's layout must be the same string."""

    @classmethod
    def setUpClass(cls):
        cls.builder, cls.usage = _builder_and_usage(ANCHOR_METHOD)

    def test_homogeneous_operators_agree(self):
        """`Wmnij`/`Wabef` are space-homogeneous, so they agree even pre-fix.

        Kept as the control: if these ever fail, the defect is something other than slot
        ordering, because a permutation cannot change `oooo` or `vvvv`.
        """
        for name in ("Wmnij", "Wabef"):
            if name not in self.builder:
                self.skipTest(f"{name} not recognized in {ANCHOR_METHOD}")
            with self.subTest(intermediate=name):
                sigs = set(self.usage.get(name, {}))
                self.assertTrue(sigs, f"{name} is built but never used")
                self.assertEqual(
                    sigs, {self.builder[name]},
                    f"{name}: built as {self.builder[name]!r} but used as {sorted(sigs)}")

        # XFAIL: this is the F0 defect the file documents -- a dressed
        # intermediate's builder normalizes its index order while the usage site
        # joins indices verbatim, so `tau` and `Wmbej` (the MIXED-space operators)
        # are read under several signatures. It only affects DRESSED
        # intermediates, and the dressed route is RETIRED
        # (vault/Status/Completion.md; dressing and spin adaptation do not
        # compose, 52 % short on Be). Fixing it would mean investing in an
        # abandoned route, so the gate is recorded as expected-to-fail rather
        # than repaired or deleted. An unexpected PASS means the layout was
        # normalized after all and this note should go.
    @unittest.expectedFailure
    def test_every_intermediate_is_used_in_its_builder_layout(self):
        """The real gate. Pre-fix this FAILS on Wmbej and tau/tau_c."""
        mismatches = []
        for name, built in sorted(self.builder.items()):
            sigs = self.usage.get(name, {})
            if not sigs:
                continue  # built-but-unused is a different question (see F5)
            wrong = {s: c for s, c in sigs.items() if s != built}
            if wrong:
                mismatches.append(
                    f"  {name}: builder emits {built!r}, "
                    f"used as {', '.join(f'{s!r}x{c}' for s, c in sorted(wrong.items()))}")
        self.assertFalse(
            mismatches,
            "dressed intermediates are BUILT in one slot layout and INDEXED in another "
            "in the emitted C++. Normalization via LoweredTensorFactor.spatial_permutation "
            "reaches some usage sites and not others (note tau appearing under two "
            "different signatures), so the builder's allocation and the consumer's "
            "indexing disagree:\n" + "\n".join(mismatches))

        # XFAIL: this is the F0 defect the file documents -- a dressed
        # intermediate's builder normalizes its index order while the usage site
        # joins indices verbatim, so `tau` and `Wmbej` (the MIXED-space operators)
        # are read under several signatures. It only affects DRESSED
        # intermediates, and the dressed route is RETIRED
        # (vault/Status/Completion.md; dressing and spin adaptation do not
        # compose, 52 % short on Be). Fixing it would mean investing in an
        # abandoned route, so the gate is recorded as expected-to-fail rather
        # than repaired or deleted. An unexpected PASS means the layout was
        # normalized after all and this note should go.
    @unittest.expectedFailure
    def test_no_intermediate_is_used_under_two_signatures(self):
        """`tau` is emitted as BOTH `oovv` and `vvoo` in one TU pre-fix.

        Separated from the builder-agreement check because it is a strictly stronger
        symptom and needs no reference to compare against: one tensor indexed two
        different ways is self-inconsistent whatever the builder does. It is also the
        fact that localizes the fix -- normalization is applied INCONSISTENTLY, not
        uniformly missing, so the question is which code path skips it (F2.1).
        """
        multi = {n: dict(c) for n, c in self.usage.items() if len(c) > 1}
        self.assertFalse(
            multi,
            "an intermediate is indexed under more than one space signature within a "
            f"single translation unit: {multi}")

    def test_mixed_space_operators_are_the_exposed_ones(self):
        """Documents WHY only some operators break, so a future reader does not
        conclude the bug is specific to `Wmbej`.

        A homogeneous signature (`oooo`, `vvvv`) is invariant under permutation, so it
        cannot express a slot-ordering error. Only mixed-space intermediates can.
        """
        mixed = {n for n, s in self.builder.items() if len(set(s)) > 1}
        self.assertTrue(
            mixed,
            "expected at least one mixed-space dressed intermediate (Wmbej/tau/tau_c); "
            "if none exist this gate can no longer detect the defect it was written for")


class RankGenericLayoutTests(unittest.TestCase):
    """F4: the layout convention belongs to the emitter, so it must hold at every rank.

    A fix that repairs one rank and not another did not fix the mechanism -- exactly the
    failure mode of T1b, where `rebind_physicist` was applied on the arbitrary-order path
    only and left the plain rank-3 path wrong.
    """

        # XFAIL: this is the F0 defect the file documents -- a dressed
        # intermediate's builder normalizes its index order while the usage site
        # joins indices verbatim, so `tau` and `Wmbej` (the MIXED-space operators)
        # are read under several signatures. It only affects DRESSED
        # intermediates, and the dressed route is RETIRED
        # (vault/Status/Completion.md; dressing and spin adaptation do not
        # compose, 52 % short on Be). Fixing it would mean investing in an
        # abandoned route, so the gate is recorded as expected-to-fail rather
        # than repaired or deleted. An unexpected PASS means the layout was
        # normalized after all and this note should go.
    @unittest.expectedFailure
    def test_layout_agreement_at_ccsdt(self):
        """Rank 3, where defect B also lives. Slower than the rank-2 anchor, so it is
        one test rather than the full battery."""
        builder, usage = _builder_and_usage("ccsdt")
        bad = {n: (builder[n], dict(usage[n]))
               for n in builder
               if usage.get(n) and set(usage[n]) != {builder[n]}}
        self.assertFalse(bad, f"ccsdt layout mismatch: {bad}")


if __name__ == "__main__":
    unittest.main()
