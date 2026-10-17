"""Fast oracle and routing checks; no compiler or molecular calculation."""
from pathlib import Path
import unittest

from dh_workflow_audit import DEFAULT, matrix_difference, render, require_close, symmetric_hessian
from dh_channel_fd_audit import parse_geometry

ROOT = Path(__file__).resolve().parents[1]


class WorkflowOracleTests(unittest.TestCase):
    def test_bohr_roundtrip_and_modes(self):
        source = DEFAULT.read_text()
        symbols, xyz, charge, mult = parse_geometry(source)
        xyz[1][0] = 1.2345678901234567
        for mode in ("gradient", "energy", "geomopt", "freq", "geomoptfreq"):
            for opt in ("cartesian", "internal"):
                result = render(source, symbols, xyz, charge, mult, mode, opt)
                self.assertEqual(parse_geometry(result), (symbols, xyz, charge, mult))
                self.assertIn(f"calculation {mode}", result)
                self.assertIn("coord_units bohr", result)
                self.assertIn(f"opt_coords {opt}", result)
                self.assertEqual(result.count("opt_coords"), 1)
                self.assertIn("exchange            b2plyp", result)

    def test_hessian_orientation_and_symmetrization(self):
        # Columns of deliberately nonsymmetric raw derivative [[2,4],[6,8]].
        plus, minus = [[2., 6.], [4., 8.]], [[-2., -6.], [-4., -8.]]
        self.assertEqual(symmetric_hessian(plus, minus, 1.), [[2., 5.], [5., 8.]])

    def test_render_preserves_user_scf_limit(self):
        source = DEFAULT.read_text()
        symbols, xyz, charge, mult = parse_geometry(source)
        for cycles in (1, 100, 400):
            template = source.replace("max_cycles  50", f"max_cycles  {cycles}")
            for mode in ("geomopt", "freq", "geomoptfreq"):
                result = render(template, symbols, xyz, charge, mult, mode, "cartesian")
                self.assertIn(f"max_cycles  {cycles}\n", result)

    def test_bad_matrices_and_thresholds_fail_closed(self):
        for a, b in (([], []), ([[1]], [[1, 2]]), ([[float("nan")]], [[0]])):
            with self.assertRaises(ValueError):
                matrix_difference(a, b)
        for error in (1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                require_close(error, 1e-5, "test")
        for h in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                symmetric_hessian([[1]], [[0]], h)


class WorkflowRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = (ROOT / "src/dft/driver.cpp").read_text()

    def test_both_workflows_use_full_gradient_callback(self):
        for name, end in (("run_frequency_analysis(", "run_geometry_optimization("),
                          ("run_geometry_optimization(", "} // namespace")):
            block = self.driver.split(name, 1)[1].split(end, 1)[0]
            self.assertIn("run_analytic_gradient_current_geometry(inner, functionals)", block)
            self.assertNotIn("compute_analytic_ks_gradient(", block)

    def test_new_geometry_gets_matching_pt2_snapshot_and_full_gradient(self):
        block = self.driver.split("run_analytic_gradient_current_geometry(", 1)[1].split(
            "run_initial_single_point(", 1)[0]
        markers = ["prepare_current_geometry(calculator, true)", "run_ks_scf_scaffold(",
                   "if (!result->converged)", "apply_post_ks_double_hybrid_correction(",
                   "functionals.has_double_hybrid_pt2 ? &pt2_result : nullptr",
                   "compute_analytic_dh_gradient(calculator, *prepared, functionals, pt2_result)",
                   "calculator._gradient = *gradient"]
        offsets = [block.index(marker) for marker in markers]
        self.assertEqual(offsets, sorted(offsets))
        # Single-point and callback must share one literal derivative assembly.
        self.assertEqual(self.driver.count("Gradient::build_dh_gradient_driver_contract(contract_inputs,"), 1)

    def test_frequency_restores_state_before_publishing(self):
        block = self.driver.split("run_frequency_analysis(", 1)[1].split("run_geometry_optimization(", 1)[0]
        markers = ["compute_hessian(calculator, gradient_runner)",
                   "set_standard_from_bohr(reference_geometry)", "gradient_runner(calculator)",
                   "store_frequency_result(calculator, *result)"]
        offsets = [block.index(marker) for marker in markers]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("calculator._hessian_step", block)

    def test_geometry_preparation_preserves_user_scf_limit(self):
        block = self.driver.split("prepare_current_geometry(", 1)[1].split(
            "prepare_quadrature_for_calculator(", 1)[0]
        self.assertEqual(block.count("set_max_cycles_auto("), 1)
        self.assertRegex(block,
            r"if \(calculator\._scf\._max_cycles == 0\)\s*"
            r"calculator\._scf\.set_max_cycles_auto\(calculator\._shells\.nbasis\(\)\);")
        # No replacement by a hard-coded limit elsewhere in preparation.
        self.assertNotRegex(block, r"_max_cycles\s*=(?!=)")

    def test_scope_remains_restricted_and_unsolvated(self):
        block = self.driver.split("// Restricted global-double-hybrid Cartesian derivatives", 1)[1].split(
            "apply_post_ks_double_hybrid_correction(", 1)[0]
        self.assertIn("!functionals.has_range_separation", block)
        self.assertIn("calculator._scf._scf != HartreeFock::SCFType::UHF", block)
        self.assertIn("calculator._solvation._model != HartreeFock::SolvationModel::None", block)
        for mode in ("Gradient", "GeomOpt", "Frequency", "GeomOptFrequency"):
            self.assertIn(f"case HartreeFock::CalculationType::{mode}:", block)
        self.assertNotIn("case HartreeFock::CalculationType::LinearResponse:", block)
        self.assertNotIn("case HartreeFock::CalculationType::ImaginaryFollow:", block)

    def test_internal_fallback_keeps_callback(self):
        source = (ROOT / "src/opt/geomopt.cpp").read_text()
        block = source.split("if (nq == 0)", 1)[1].split("// Count each type", 1)[0]
        self.assertIn("return run_geomopt(calc, gradient_runner);", block)


if __name__ == "__main__":
    unittest.main()
