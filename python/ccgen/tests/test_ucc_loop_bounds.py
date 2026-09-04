"""U3b.2 -- the emitted UCC kernels must be spin-resolved in bounds, counts and shape.

Covers U3b.2a (loop bounds), U3b.2b (the `const int` orbital-count preamble) and
U3b.2c (result allocation). 2b and 2c are ONE atomic change and the compiler is
what proves it: 2b removes the `const int no/nv` declarations, so a tree with 2b
but not 2c emits a TU that fails with "use of undeclared identifier 'no'" at every
result allocation (16 errors on the CCSD UCC TU). They cannot ship separately.

THE DEFECT THIS CLOSES. Every emitted kernel opened with

    const int no = reference.orbital_partition.n_occ;
    const int nv = reference.orbital_partition.n_virt;

and every loop ran `< no` / `< nv`. `build_ucc_fock_blocks` never sets
`orbital_partition`, so on a UCC state both counts are 0 -- which is why
`correlation ucc2` died at `sector residual shape mismatch at (rank 1, tag aa)`
rather than returning a wrong number. Measured pre-fix on the CCSD UCC TU: 1182
`< no`, 1192 `< nv`, all spin-blind.

WHERE THE SPIN COMES FROM. An `Index` carries only a space (occ/vir); the U1 bridge
drops spin by design, because RCC stores one spatial tensor per rank and has nothing
to route. `ucc_term_index_spins` (U3b.0) recovers it from the FACTORS instead: slot
*k* of `t2_abab` / `v_aaaa` / `f_bb` carries `tag[k]`. It returns {} for RCC terms,
which is what keeps the RCC emit byte-identical -- `_loop_bound` then sees spin=None
everywhere and emits the same `no`/`nv` it always did.

WHY THE ASSERTIONS ARE SHAPED THIS WAY. The obvious check -- "the aaaa kernel must
use only alpha counts" -- is WRONG, and believing it cost a wrong turn here. The
`aaaa` residual legitimately carries beta bounds: terms like `t2_abab * v_abab`
contribute to it through SUMMED beta indices even though all four FREE indices are
alpha (measured: 36 beta bounds in `doubles_aaaa`). What actually holds is the
MIRROR: `singles_aa` and `singles_bb` must be exact a<->b reflections of each other,
as must `doubles_aaaa` and `doubles_bbbb`, and `abab` must be balanced.

That mirror is the assertion with teeth. Under the space-grouped misreading of the
spin tag -- the trap this scope has hit four times, where `aaaa`/`bbbb` agree and
only `abab` differs -- `ucc_term_index_spins` raises on `v_abab` before emission,
so a same-spin-only check would never have been reached anyway. Mutation-tested in
both directions: ignoring the spin map restores all 1182/1192 spin-blind bounds, and
the space-grouped variant raises a slot-mapping conflict.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

from ccgen.generate import print_cpp_planck


def _kernel_bound_counts(src: str) -> dict[str, dict[str, int]]:
    """Per-kernel counts of each spin-resolved loop bound, scoped by line.

    Line-scoped rather than regex-spanned on purpose: a `.*?\\n\\}\\n` span match
    silently captures the wrong region (it stops at the first closing brace) and
    reports counts for a kernel it is not looking at.
    """
    counts: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in src.split("\n"):
        header = re.match(r"^\w[\w:<>,& ]*\s(compute_ucc_\w+)\(", line)
        if header:
            current = header.group(1)
            counts.setdefault(current, {k: 0 for k in ("noa", "nob", "nva", "nvb")})
        if current is not None:
            for name in counts[current]:
                counts[current][name] += len(re.findall(r"< " + name + r"\b", line))
    return counts


class UccLoopBoundsAreSpinResolvedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = print_cpp_planck("ccsd", ucc=True)
        cls.counts = _kernel_bound_counts(cls.src)

    def test_no_spin_blind_bounds_remain(self):
        """Not one `< no` / `< nv` survives in the UCC TU (pre-fix: 1182 / 1192)."""
        self.assertEqual(len(re.findall(r"< no(?![ab])", self.src)), 0)
        self.assertEqual(len(re.findall(r"< nv(?![ab])", self.src)), 0)

    def test_every_bound_is_accounted_for(self):
        """The spin-resolved bounds sum to the pre-fix spin-blind totals.

        Catches a fix that DROPS loops rather than routing them -- zero spin-blind
        bounds is also achievable by emitting nothing.
        """
        total_o = sum(len(re.findall(r"< " + s + r"\b", self.src)) for s in ("noa", "nob"))
        total_v = sum(len(re.findall(r"< " + s + r"\b", self.src)) for s in ("nva", "nvb"))
        self.assertEqual(total_o, 1182)
        self.assertEqual(total_v, 1192)

    def test_same_spin_blocks_are_exact_mirrors(self):
        """`aa`<->`bb` and `aaaa`<->`bbbb` must reflect under a<->b.

        The discriminating assertion: a spin routing that is wrong in any
        asymmetric way breaks the reflection.
        """
        def flip(d):
            return {"noa": d["nob"], "nob": d["noa"], "nva": d["nvb"], "nvb": d["nva"]}

        for a, b in (("singles_aa", "singles_bb"), ("doubles_aaaa", "doubles_bbbb")):
            ka = f"compute_ucc_ccsd_{a}_residual"
            kb = f"compute_ucc_ccsd_{b}_residual"
            self.assertEqual(flip(self.counts[ka]), self.counts[kb], f"{a} <-> {b}")

    def test_mixed_block_is_balanced(self):
        """`abab` touches alpha and beta equally -- the block where a wrong
        hypothesis changes the SHAPE, not merely the values."""
        c = self.counts["compute_ucc_ccsd_doubles_abab_residual"]
        self.assertEqual(c["noa"], c["nob"])
        self.assertEqual(c["nva"], c["nvb"])

    def test_beta_bounds_appear_in_the_aaaa_kernel(self):
        """Pins the fact that made the naive assertion wrong.

        `doubles_aaaa` has four ALPHA free indices but genuinely sums over beta
        (e.g. `t2_abab * v_abab`). If this ever reaches 0, either the manifold
        changed or someone "fixed" the bounds by forcing the target's tag onto
        every index -- which is the spin-blind defect wearing a spin suffix.
        """
        c = self.counts["compute_ucc_ccsd_doubles_aaaa_residual"]
        self.assertGreater(c["nob"], 0)
        self.assertGreater(c["nvb"], 0)


class UccPreambleAndShapeTests(unittest.TestCase):
    """U3b.2b/U3b.2c -- the counts a UCC kernel declares, and the shape it allocates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = print_cpp_planck("ccsd", ucc=True)

    def test_no_orbital_partition_reads_remain(self):
        """`orbital_partition` is DEFAULT on a UCC reference (U3b.1 keeps it so
        deliberately), so any read of it is a zero-valued bound."""
        self.assertEqual(self.src.count("orbital_partition"), 0)

    def test_every_kernel_declares_the_four_counts(self):
        for name in ("noa", "nob", "nva", "nvb"):
            declared = len(re.findall(r"const int " + name + r" =", self.src))
            self.assertEqual(declared, 6, f"{name} declared {declared} times, expected 6")

    def test_mixed_block_allocates_four_distinct_extents(self):
        """The assertion with teeth: `doubles_abab` is the block where a spin-blind
        extent changes the SHAPE rather than only the values."""
        self.assertIn("Tensor4D result(noa, nob, nva, nvb, 0.0);", self.src)

    def test_same_spin_blocks_allocate_their_own_spin(self):
        self.assertIn("Tensor4D result(noa, noa, nva, nva, 0.0);", self.src)
        self.assertIn("Tensor4D result(nob, nob, nvb, nvb, 0.0);", self.src)
        self.assertIn("Tensor2D result(noa, nva, 0.0);", self.src)
        self.assertIn("Tensor2D result(nob, nvb, 0.0);", self.src)

    def test_no_spin_blind_allocation_survives(self):
        self.assertEqual(len(re.findall(r"result\((?:no|nv)[,)]", self.src)), 0)


class UccChunkedKernelTests(unittest.TestCase):
    """The `_partN` preamble is a SECOND emission site that CCSD never reaches.

    At rank 2 no kernel exceeds the 256-term chunk threshold; at rank 3 all four
    triples blocks do (597/486/486/597 terms). A CCSD-only gate therefore passes
    with every `_part` preamble still spin-blind.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = print_cpp_planck("ccsdt", ucc=True)

    def test_chunking_actually_fires(self):
        """Guards the guard: if chunking stops firing, the assertions below go
        vacuous rather than failing."""
        self.assertGreater(len(re.findall(r"_part\d+\(", self.src)), 0)

    def test_chunked_preamble_is_spin_resolved(self):
        self.assertEqual(self.src.count("orbital_partition"), 0)
        self.assertEqual(len(re.findall(r"< n[ov](?![ab])", self.src)), 0)
        self.assertEqual(len(re.findall(r"\(void\)no;", self.src)), 0)

    def test_mixed_triples_sector_allocates_its_own_shape(self):
        """`aabaab` -- two alpha and one beta in each of occ and vir."""
        self.assertIn("Tensor6D result(noa, noa, nob, nva, nva, nvb, 0.0);", self.src)


class RccEmitIsUnchangedTests(unittest.TestCase):
    """The RCC path must not observe U3b.2a at all."""

    def test_rcc_bounds_stay_spin_blind(self):
        for method in ("ccsd", "ccsdt"):
            src = print_cpp_planck(method)
            with self.subTest(method=method):
                self.assertEqual(len(re.findall(r"< n[ov][ab]\b", src)), 0)
                self.assertGreater(len(re.findall(r"< no\b", src)), 0)


if __name__ == "__main__":
    unittest.main()


class UccTranslationUnitCompilesTests(unittest.TestCase):
    """The generated UCC TU must actually COMPILE.

    Every UCC gate before U3b.2 was structural -- they inspected emitted TEXT and
    none of them ever fed it to a compiler, which is why the spin-blind bounds
    survived U3/U4/U5.0-U5.3b and surfaced only at runtime as a shape mismatch.
    This is the check that would have caught it at emit time, and it is what
    established that 2b and 2c are inseparable (2b alone fails here with 16
    "use of undeclared identifier 'no'" errors).

    Skipped rather than failed when no compiler or Eigen tree is present, so it
    stays runnable on a machine that has not configured a build.
    """

    @staticmethod
    def _eigen_include(root):
        for candidate in (
            root / "build-ccgen-test" / "_deps" / "eigen-src",
            root / "build" / "_deps" / "eigen-src",
            root / "install" / "include" / "eigen3",
        ):
            if (candidate / "signature_of_eigen3_matrix_library").exists():
                return candidate
        return None

    def _compile(self, method):
        root = pathlib.Path(__file__).resolve().parents[3]
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("no C++ compiler on PATH")
        eigen = self._eigen_include(root)
        if eigen is None:
            self.skipTest("no configured Eigen tree found")

        self._compile_source(print_cpp_planck(method, ucc=True), f"UCC {method}")

    def _compile_source(self, code, label):
        root = pathlib.Path(__file__).resolve().parents[3]
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("no C++ compiler on PATH")
        eigen = self._eigen_include(root)
        if eigen is None:
            self.skipTest("no configured Eigen tree found")

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tu.cpp"
            path.write_text(code)
            proc = subprocess.run(
                [compiler, "-std=c++23", "-fsyntax-only", "-w",
                 "-I", str(root / "src"), "-I", str(eigen), str(path)],
                capture_output=True, text=True)
        self.assertEqual(
            proc.returncode, 0,
            f"generated TU ({label}) failed to compile:\n{proc.stderr[:4000]}")

    def test_rank2_ucc_tu_compiles(self):
        self._compile("ccsd")

    def test_rank3_ucc_tu_compiles(self):
        """Rank 3 exercises the chunked `_partN` path that rank 2 does not."""
        self._compile("ccsdt")

    def test_spin_adapted_tu_still_compiles(self):
        """The NEIGHBOUR path, and the one this step actually broke.

        The first version of U3b.2a gated the spin map on "did it come back
        empty" rather than on `ucc`, reasoning that only UCC terms carry
        block-tagged factor names. That is FALSE for the spin-adapted RCC path:
        its rank-4 sector amplitudes are named `t4_aaabaaab`, which matches the
        same `_[ab]+` suffix. So a spin-adapted kernel got `noa` / `nvb` loop
        bounds over a preamble declaring only `no` / `nv`, and stopped compiling.

        The RCC SHA-256 pin did NOT catch it -- it emits with `spin_adapt=False`
        and never reaches this path.

        RANK 4, and that is load-bearing rather than thorough. The tagged factor
        names that trigger the defect are the `t4_aaabaaab` sector amplitudes,
        which do not exist below rank 4: measured, rank-3 spin-adapted emits ZERO
        terms with a non-empty spin map, so a rank-3 version of this gate passes
        under the exact mutation it exists to catch. It was written at rank 3
        first and confirmed vacuous by mutation before being moved -- the same
        RHF-degenerate vacuity this scope has hit repeatedly, met here while
        writing the gate for a defect just found.

        Costs a few minutes of rank-4 generation, which is why it is the only
        rank-4 case in this file.
        """
        self._compile_source(
            print_cpp_planck("ccsdtq", spin_adapt=True, engine="diagram"),
            "spin-adapted ccsdtq")
