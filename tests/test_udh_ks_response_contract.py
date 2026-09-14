"""No-build API/routing guards. Numerical U2 acceptance is the C++ target."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/udh_ks_response.cpp").read_text()
        cls.header = (ROOT / "src/dft/udh_ks_response.h").read_text()

    def test_explicit_physical_spin_channels(self):
        for field in ("coulomb", "exchange", "xc_from_alpha", "xc_from_beta", "total"):
            self.assertIn(field, self.header)
        self.assertIn("trial.alpha + trial.beta", self.source)
        self.assertIn("-exact_exchange * *ka, -exact_exchange * *kb", self.source)
        self.assertIn("{trial.alpha, zero}", self.source)
        self.assertIn("{zero, trial.beta}", self.source)
        self.assertNotIn("kernel_scale", self.source)

    def test_uses_existing_direct_and_polarized_actions(self):
        for call in ("_compute_2e_j_direct(", "_compute_2e_k_direct(",
                     "compute_analytic_xc_hessian_vector_product_polarized("):
            self.assertEqual(self.source.count(call), 1)
        self.assertIn("const auto fixed = inputs;", self.source)
        self.assertIn("UDHXCResponseFn xc = [fixed]", self.source)
        self.assertNotIn("[&inputs]", self.source)

    def test_checked_shapes_outputs_and_failures(self):
        for marker in ("valid_pair(trial, nbasis)", "valid_matrix(*value, n)",
                       "valid_pair(*value, n)", "valid_pair(out.total, nbasis)",
                       "XC_FLAGS_HAVE_FXC", "XC::Spin::Polarized", "x.is_range_separated()"):
            self.assertIn(marker, self.source)
        self.assertEqual(self.source.count("catch (...)"), 2)
        self.assertIn("if (exact_exchange != 0.0)", self.source)

    def test_no_uks_gradient_enablement(self):
        driver = (ROOT / "src/dft/driver.cpp").read_text()
        guard = driver.split("// Restricted global-double-hybrid Cartesian derivatives", 1)[1].split(
            "apply_post_ks_double_hybrid_correction(", 1)[0]
        self.assertIn("calculator._scf._scf != HartreeFock::SCFType::UHF", guard)
        self.assertNotIn("make_udh_direct_ks_response_operator", driver)

    def test_detached_target_links_actual_primitive(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-ks-response", 1)[1].split("    )", 1)[0]
        for filename in ("tests/udh_ks_response.cpp", "dft/udh_ks_response.cpp",
                         "dft/analytic_hessian.cpp", "dft/xc_grid.cpp", "dft/ks_matrix.cpp"):
            self.assertIn(filename, target)
        oracle = (ROOT / "tests/udh_ks_response.cpp").read_text()
        self.assertIn('#include "dft/ks_matrix.h"', oracle)


if __name__ == "__main__":
    unittest.main()
