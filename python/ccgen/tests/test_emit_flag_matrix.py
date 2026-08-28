"""V1.2.0: pin `print_cpp_planck`'s output across its flag matrix.

V1.2 refactors a function with four interacting flags (`dress_operators`, `spin_adapt`,
`factorize_tau`, `force_arbitrary`) to remove an early return. This is the net placed
BEFORE that change: it records what every currently-reachable combination emits, so a
"wiring" edit that quietly alters an existing path is caught rather than assumed benign.

Two things are pinned per combination, because either alone is weak:

- the full-text SHA-256, which catches any content change; and
- the byte length, which is what a human reads in a diff and what the scope docs quote.

A length that holds while the hash moves means the emit was reordered or a token swapped --
exactly the kind of change a length-only check would wave through.

WHAT IS DELIBERATELY NOT PINNED. `dress_operators=True` combined with `spin_adapt`,
`factorize_tau`, or `force_arbitrary`: those are unreachable today (the early return fires
first), so there is no status quo to record. `test_dress_operators_currently_ignores_other_
flags` pins the *observable consequence* of that unreachability instead, which is the thing
V1.2.1 changes and V1.2.4 must guard.

WHAT THIS MATRIX CANNOT SEE, and it shipped a regression (U3b.2, 2026-08-24). Every
baseline here is `METHOD = "ccsd"` -- rank 2. A defect keyed to names that only exist at
higher rank is invisible to the whole matrix. Concretely: the spin-adapted rank-4 sector
amplitudes are named `t4_aaabaaab`, and an emitter change that keyed off that suffix broke
spin-adapted `ccsdtq` while every combination above stayed green. Rank coverage for
`spin_adapt` (ranks 2/3/4, both engines) lives in
`test_ucc_emit_flag.test_spin_adapted_emit_is_unchanged`; if you add a flag here whose
behaviour could vary with rank, pin it there too rather than assuming rank 2 is
representative.

Also rank-2-only and worth knowing: these baselines use the DEFAULT engine. `engine="diagram"`
-- what the generator and every UCC gate actually use -- emits different text (2038 differing
lines at rank 2, at identical byte length).
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import print_cpp_planck  # noqa: E402

METHOD = "ccsd"

# (label, kwargs, byte length, sha256) measured on the current tree.
BASELINES = (
    ("bare", {}, 37216,
     "af74826e253415a261f9b57efd4ed906827ef0c70cb9da6989e0f941d3b9f656"),
    ("spin_adapt", {"spin_adapt": True}, 65431,
     "44705c8ad85f951cbebb532a2fe60ea5418feddfbc28b8523ee2768fd12e0fd4"),
    # Moved at V1.3.2 (37413 -> 37433): tau's builder is now method-suffixed
    # (`build_tau_ccsd`) like every other builder -- one naming rule, no exceptions.
    ("factorize_tau", {"factorize_tau": True}, 37433,
     "b8ecaa6711793302be21d9e9e1cdc3fbc154acc2d5f57d98692c9e10a9926495"),
    ("force_arbitrary", {"force_arbitrary": True}, 37175,
     "bf1e083d5759120621369de2fe1572a926fbf96d21d886387d452d81afe6363e"),
    ("spin_adapt+force_arbitrary",
     {"spin_adapt": True, "force_arbitrary": True}, 64265,
     "f3a85400f3178fabb06f1ba674e51a5fb9e963d437b70ffda32f900c98e7ca2f"),
    # Moved at V1.3 (27960 -> 28122): builders that reference a sibling intermediate now
    # bind it (`const auto tau = build_tau(...)`), which is what made the dressed TU
    # compile for the first time. The four undressed baselines above are unchanged, so the
    # fix is confined to sibling-referencing builders.
    # Moved again at V1.3.2 (28122 -> 28187): builders are method-suffixed so two dressed
    # TUs can share the registry's single translation unit.
    # Digest moved again (length UNCHANGED at 28187) when `_dress_operator_equations`
    # began counting definition-site uses: an operator's definition can reference a
    # pseudo-amplitude (`Wmnij`/`Wabef` both read `tau`), so a name can be needed by the
    # emitted code while appearing nowhere in the residual. Without that, a
    # definition-only pseudo-amplitude gets usage_count=0 and is dropped as an orphan
    # while its builder is still called. The only emitted difference is the count in
    # `// Intermediate tau (oovv, usage=N)`: 1 -> 3, a comment. Byte length is identical,
    # which is why `test_byte_lengths_are_unchanged` did not move and this digest did.
    #
    # RE-PINNED 2026-08-26 (D5): the digest moved again, for a CORRECTNESS fix.
    # `lowering/restricted_closed_shell.py` carried the full 8-fold symmetry
    # group of the ANTISYMMETRIZED <pq||rs>, four members of which are false for
    # the spatial blocks these kernels index. Its phase reaches the emitted text
    # directly, so those four produced wrong ERI reads with a bogus sign. Exactly
    # 8 lines changed here, all of that shape, e.g.
    #     -  result(m, j, b, e) += -mo_blocks.ovov(m, b, j, e);
    #     +  result(m, j, b, e) +=  mo_blocks.ovvo(m, b, e, j);
    # Verified against the operator SPEC, not merely observed: for the `Wabef`
    # definition term `-1 t1(b,m) v(a,m,e,f)`, the new emission reproduces it to
    # 0.000e+00 while the old was off by 1.4e+01. Byte length is unchanged (the
    # edits are in place), which is why only this digest moved.
    # See docs/CCGEN_WIRING_THE_DERIVATION_ROUTE.md, D4/D5.
    ("dress_operators", {"dress_operators": True}, 28187,
     "49d24c2724e73e5dd0e3625bb2e21367b17db58f68374e97c4c2fa91ee756f05"),
)


def _emit(**kwargs) -> str:
    return print_cpp_planck(METHOD, **kwargs)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FlagMatrixBaselineTests(unittest.TestCase):
    """Every currently-reachable flag combination keeps its exact output."""

    @classmethod
    def setUpClass(cls):
        cls.emitted = {label: _emit(**kwargs) for label, kwargs, _, _ in BASELINES}

    def test_full_text_hashes_are_unchanged(self):
        """The strong check: any content change moves the digest."""
        for label, _, _, digest in BASELINES:
            with self.subTest(combination=label):
                self.assertEqual(_sha(self.emitted[label]), digest)

    def test_byte_lengths_are_unchanged(self):
        """The readable check, and the figure the scope docs quote. Kept alongside the
        hash so a failure says *how much* moved, not just that something did."""
        for label, _, length, _ in BASELINES:
            with self.subTest(combination=label):
                self.assertEqual(len(self.emitted[label]), length)

    def test_outputs_are_pairwise_distinct(self):
        """Guards against the length checks passing because two flags collapsed onto the
        same output -- a refactor that made a flag a no-op would otherwise look fine."""
        digests = {label: _sha(text) for label, text in self.emitted.items()}
        self.assertEqual(len(set(digests.values())), len(digests),
                         f"two combinations emit identical text: {digests}")

    def test_emission_is_deterministic(self):
        """The hashes below are only meaningful if emit is reproducible in-process."""
        for label, kwargs, _, _ in BASELINES:
            with self.subTest(combination=label):
                self.assertEqual(_sha(_emit(**kwargs)), _sha(self.emitted[label]))


class DressedPathReachabilityTests(unittest.TestCase):
    """The flags are now REACHABLE under dressing (V1.2.1 removed the early return).

    Before V1.2.1 each of these combinations was byte-identical to `dress_operators` alone
    -- the extra flag was silently dropped. That was pinned here as the status quo, and its
    flipping is the observable evidence that V1.2.1 did what it claims. The six baseline
    hashes held across the same change, so the reachability is the *only* behavior that
    moved.

    `factorize_tau` was the hazard: the parent scope records dress x tau as "already
    mutually exclusive", but the exclusion was unreachability rather than a guard, so
    V1.2.1 briefly ACTIVATED tau under dressing. V1.2.4 closed it with an explicit error --
    found because the collision tripped an assertion added in V1.2.1, not by inspection.
    """

    def test_other_flags_now_reach_the_dressed_path(self):
        dressed_only = _emit(dress_operators=True)
        for extra in ({"spin_adapt": True},
                      {"force_arbitrary": True}):
            with self.subTest(extra=extra):
                self.assertNotEqual(_emit(dress_operators=True, **extra), dressed_only)


class MutualExclusionTests(unittest.TestCase):
    """V1.2.4: excluded flag combinations must be explicit, never silent precedence."""

    def test_dress_operators_and_factorize_tau_raise(self):
        """Both materialize tau; running both would build it twice through two paths.
        Raising beats picking a winner -- silent precedence is what disguised this."""
        with self.assertRaises(ValueError) as caught:
            _emit(dress_operators=True, factorize_tau=True)
        self.assertIn("mutually exclusive", str(caught.exception))

    def test_dressing_forces_cse_off_without_failing(self):
        """Mirrors the spin_adapt precedent: forced off rather than an error, so a caller
        passing include_intermediates does not get a failed dressed build."""
        self.assertEqual(_emit(dress_operators=True, include_intermediates=True),
                         _emit(dress_operators=True))

    def test_spin_adapt_still_forces_cse_off(self):
        """Unchanged by V1.2.4 -- pinned so the new branch cannot have altered it."""
        self.assertEqual(_emit(spin_adapt=True, include_intermediates=True),
                         _emit(spin_adapt=True))

    def test_dressed_output_carries_its_builders(self):
        """Anchors what the dressed path is, so a regression in V1.2.1 that emitted an
        undressed TU at the right byte count could not slip through."""
        text = _emit(dress_operators=True)
        for builder in (f"build_tau_{METHOD}", f"build_Wmnij_{METHOD}",
                        f"build_Wabef_{METHOD}", f"build_Wmbej_{METHOD}"):
            with self.subTest(builder=builder):
                self.assertIn(builder, text)


class ComposedDressedEmitTests(unittest.TestCase):
    """V1.2.2 / V1.2.3: the compositions the early return used to make unreachable."""

    def test_dressed_spin_adapted_emits_all_builders(self):
        text = _emit(dress_operators=True, spin_adapt=True)
        for builder in ("build_tau", "build_tau_c",
                        "build_Wmnij", "build_Wabef", "build_Wmbej"):
            with self.subTest(builder=builder):
                self.assertIn(builder, text)

    def test_dressed_specs_are_adapted_not_gcc(self):
        """The miscompile V1.2.2 prevents. Adaptation moves three of the five declared
        layouts (tau vvoo->oovv, tau_c vvoo->oovv, Wmbej ovvo->oovv), so emitting the GCC
        specs beside a spin-adapted residual would declare one layout and build another.

        Asserted at the source rather than by grepping the TU: the emitted C++ names the
        buffer dimensions, not the signature string, so the spec list is where the contract
        actually lives.
        """
        from ccgen.generate import _dress_operator_equations, generate_cc_equations
        from ccgen.spin import adapt_intermediate_spec

        _, specs = _dress_operator_equations(
            generate_cc_equations("ccsd", engine="diagram", canonical_fock=True))
        gcc = {s.name: s.index_space_sig for s in specs}
        adapted = {s.name: adapt_intermediate_spec(s).index_space_sig for s in specs}

        self.assertEqual(gcc["tau"], "vvoo")
        self.assertEqual(adapted["tau"], "oovv")
        self.assertEqual(gcc["Wmbej"], "ovvo")
        self.assertEqual(adapted["Wmbej"], "oovv")
        self.assertNotEqual(gcc, adapted,
                            "if these ever coincide, this test no longer proves anything")

    def test_adapted_specs_pass_the_v11f_validator(self):
        """The wired assertion's own gate. `print_cpp_planck` raises if this fails, so a
        green emit already implies it -- asserted directly so a failure names the cause."""
        from ccgen.generate import _dress_operator_equations, generate_cc_equations
        from ccgen.optimization.intermediates import validate_intermediate_specs
        from ccgen.spin import adapt_intermediate_spec

        _, specs = _dress_operator_equations(
            generate_cc_equations("ccsd", engine="diagram", canonical_fock=True))
        adapted = [adapt_intermediate_spec(s) for s in specs]
        self.assertEqual(validate_intermediate_specs(adapted), [])

    def test_force_arbitrary_threads_through_dressing(self):
        """V1.2.3. Targets the arbitrary-order runtime, which is where the generated
        production path actually executes."""
        plain = _emit(dress_operators=True, spin_adapt=True)
        arbitrary = _emit(dress_operators=True, spin_adapt=True, force_arbitrary=True)
        self.assertNotEqual(plain, arbitrary)
        self.assertIn("generated_arbitrary_", arbitrary)
        for builder in ("build_tau", "build_Wmbej"):
            with self.subTest(builder=builder):
                self.assertIn(builder, arbitrary)


class DressedTranslationUnitCompileTests(unittest.TestCase):
    """V1.3: the dressed TU must be valid C++.

    It was not, and had never been. `build_Wmnij` and `build_Wabef` reference `tau(...)`,
    but `sibling_names` only made such a factor RENDER as a bare identifier -- nothing
    declared it, so the emitted C++ used `tau` with no `tau` in scope. Pre-existing:
    verified by rebuilding at V1.2's parent, where it fails identically. V1.2 made the
    dressed path reachable, which is what exposed it.

    Fixed in `_emit_intermediate_builder` by binding referenced siblings the same way
    `_emit_kernel` already binds the intermediates its residual terms reference
    (`const auto tau = build_tau(...)`). Correct by the existing dependency order --
    `intermediates` is ordered pseudo-specs-first, so a sibling's own builder is declared
    above its consumer.
    """

    def _syntax_check(self, **kwargs):
        import os
        import shutil
        import subprocess
        import tempfile

        cxx = os.environ.get("CXX", "c++")
        if shutil.which(cxx) is None:
            self.skipTest(f"{cxx} not available")
        repo = Path(__file__).resolve().parents[3]
        eigen = repo / "build" / "_deps" / "eigen-src"
        if not eigen.is_dir():
            self.skipTest("Eigen fetch not present (configure the build first)")

        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as fh:
            fh.write(_emit(**kwargs))
            src = fh.name
        try:
            proc = subprocess.run(
                [cxx, "-std=c++23", "-fsyntax-only", "-w",
                 "-I", str(repo / "src"), "-I", str(eigen), src],
                capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(proc.returncode, 0,
                             f"did not compile:\n{proc.stderr[-1500:]}")
        finally:
            os.unlink(src)

    def test_dressed_only_tu_compiles(self):
        self._syntax_check(dress_operators=True)

    def test_dressed_spin_adapted_tu_compiles(self):
        self._syntax_check(dress_operators=True, spin_adapt=True)

    def test_dressed_spin_adapted_arbitrary_tu_compiles(self):
        """The combination the arbitrary-order runtime actually executes."""
        self._syntax_check(dress_operators=True, spin_adapt=True, force_arbitrary=True)

    def test_the_undressed_paths_still_compile(self):
        self._syntax_check()

    def test_sibling_builders_are_bound_before_use(self):
        """The specific defect, asserted at the source rather than only via the compiler:
        every builder that references a sibling must declare it first.

        A compile check alone would regress silently if the emitter later stopped emitting
        the reference at all -- this pins that the binding accompanies the use.
        """
        text = _emit(dress_operators=True)
        # V1.3.2: builders are method-suffixed (`build_Wmnij_ccsd`).
        for consumer in (f"build_Wmnij_{METHOD}", f"build_Wabef_{METHOD}"):
            start = text.index(f"{consumer}(")
            body = text[start:text.index("\n}", start)]
            binding = f"const auto tau = build_tau_{METHOD}("
            with self.subTest(consumer=consumer):
                self.assertIn("tau(", body, "expected this builder to reference tau")
                self.assertIn(binding, body)
                self.assertLess(body.index(binding),
                                body.index("* tau(") if "* tau(" in body
                                else body.index("tau("),
                                "tau must be bound before first use")


class EmittedBuilderOrderTests(unittest.TestCase):
    """V1.4: the emitted builder order must be a valid topological sort.

    Asserted on the emitted TU rather than the spec list, per the scope, so it also catches
    an emit-layer reordering that left the spec list intact. Only meaningful now that V1.3
    emits the sibling bindings -- before that there were no cross-builder references in the
    text to order.
    """

    def _builder_order_and_deps(self, **kwargs):
        import re

        text = _emit(**kwargs)
        defined = [m.group(1)
                   for m in re.finditer(r"^\w[\w:<>, ]*? build_(\w+)\($", text, re.M)]
        deps = {}
        for name in defined:
            start = text.index(f"build_{name}(\n")
            body = text[start:text.index("\n}", start)]
            deps[name] = {m.group(2) for m
                          in re.finditer(r"const auto (\w+) = build_(\w+)\(", body)}
        return defined, deps

    def test_no_builder_references_one_defined_later(self):
        for kwargs in ({"dress_operators": True},
                       {"dress_operators": True, "spin_adapt": True},
                       {"dress_operators": True, "spin_adapt": True,
                        "force_arbitrary": True}):
            defined, deps = self._builder_order_and_deps(**kwargs)
            with self.subTest(flags=tuple(sorted(kwargs))):
                self.assertTrue(defined, "no builders emitted -- gate would be vacuous")
                seen: set[str] = set()
                for name in defined:
                    for dep in deps[name]:
                        self.assertIn(dep, seen,
                                      f"build_{name} references build_{dep} before it is "
                                      f"defined (order: {defined})")
                    seen.add(name)

    def test_tau_precedes_its_consumers_in_the_emitted_text(self):
        """`tau` is matched by suffix, since V1.3.2 names builders `build_tau_<method>`."""
        defined, deps = self._builder_order_and_deps(dress_operators=True)
        tau_names = {n for n in defined if n.startswith("tau")}
        self.assertTrue(tau_names, f"expected a tau builder among {defined}")
        consumers = [n for n, d in deps.items() if d & tau_names]
        self.assertTrue(consumers, "expected tau to have consumers")
        first_tau = min(defined.index(n) for n in tau_names)
        for consumer in consumers:
            with self.subTest(consumer=consumer):
                self.assertLess(first_tau, defined.index(consumer))


class ComposedNumericTests(unittest.TestCase):
    """V1.2.5: the new combination's correctness.

    Byte-identity proves no regression on existing paths and says nothing about the new
    one, and there is no prior output for it to match -- so the gate is numeric. Not a
    symbolic term count: V1.1e spent five sub-steps establishing that a term multiset
    cannot distinguish different algebra from a symmetry-equivalent rewriting.
    """

    def test_dressed_adapted_residual_matches_adapted_raw(self):
        import numpy as np

        from ccgen.generate import _dress_operator_equations, generate_cc_equations
        from ccgen.optimization.dressed_equation import expand_dressed_term
        from ccgen.spin import spin_adapt_equations
        from ccgen.tests.residual_eval import random_tensors, residual_of

        raw = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        dressed, _ = _dress_operator_equations(raw)
        expanded = {m: [x for t in ts for x in expand_dressed_term(t)]
                    for m, ts in dressed.items()}
        adapted_dressed = spin_adapt_equations(expanded)
        adapted_raw = spin_adapt_equations(raw)

        for no, nv, seed in ((2, 3, 0), (3, 4, 11)):
            tensors = random_tensors(no, nv, seed=seed)
            for manifold in ("energy", "singles", "doubles"):
                a = residual_of(adapted_dressed[manifold], no, nv, tensors=tensors)
                b = residual_of(adapted_raw[manifold], no, nv, tensors=tensors)
                scale = max(float(np.abs(b).max()), 1.0)
                with self.subTest(manifold=manifold, no=no, seed=seed):
                    self.assertLess(float(np.abs(a - b).max()) / scale, 1e-12)


if __name__ == "__main__":
    unittest.main()
