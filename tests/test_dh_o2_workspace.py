"""No-build O2 routing/ownership guards; C++ oracles establish numerics."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class O2WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = (ROOT / "src/dft/dh_pt2_gradient.h").read_text()
        cls.source = (ROOT / "src/dft/dh_pt2_gradient.cpp").read_text()
        cls.driver = (ROOT / "src/dft/driver.cpp").read_text()
        cls.xc = (ROOT / "src/dft/analytic_hessian.cpp").read_text()

    def test_large_inputs_have_shared_const_owner(self):
        view = self.header.split("struct DHGradientDriverInputs", 1)[1].split("};", 1)[0]
        self.assertIn("std::shared_ptr<const DHGradientGeometryWorkspace> workspace", view)
        self.assertNotIn("std::vector<double>", view)
        self.assertNotIn("DHEq33NonXCDerivatives non_xc_derivatives", view)
        self.assertIn("std::move(zvector->mo_eri)", self.driver)
        self.assertIn("std::move(*derivatives)", self.driver)

    def test_stationary_inputs_are_constructed_once_and_moved(self):
        production = self.driver.split("compute_analytic_dh_gradient(", 1)[1].split(
            "run_analytic_gradient_current_geometry(", 1)[0]
        self.assertIn("std::move(static_cast<Gradient::DHStationaryProducts &>(*zvector))", production)
        self.assertNotIn("build_dh_pt2_amplitude_density", production)
        self.assertIn("contract_residual > 1e-9", production)
        assembly = self.source.split("assemble_dh_gradient_driver_contract(", 1)[1].split(
            "build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs)", 1)[0]
        for forbidden in ("build_dh_pt2_amplitude_density", "build_dh_eq41_response_density",
                          "build_dh_eq40_amplitude_rhs", "build_dh_lagrangian_rhs"):
            self.assertNotIn(forbidden, assembly)
        self.assertIn("std::move(stationary)", assembly)
        self.assertIn("std::move(*pair_density)", assembly)
        self.assertIn("stale or invalid stationary products", assembly)

    def test_cache_apply_uses_only_trial_fields(self):
        apply = self.xc.split("RKSXCKernel::apply(", 1)[1].split(
            "compute_analytic_xc_hessian_vector_product(", 1)[0]
        self.assertEqual(apply.count("evaluate_density_on_grid("), 1)
        self.assertNotIn("evaluate_gga", apply)
        self.assertNotIn("evaluate_lda", apply)
        self.assertNotIn("gradient_squared", apply)
        prepare = self.xc.split("prepare_rks_xc_kernel(", 1)[1].split("RKSXCKernel::storage_bytes", 1)[0]
        self.assertEqual(prepare.count("evaluate_density_on_grid("), 1)
        self.assertEqual(prepare.count("gradient_squared()"), 1)
        self.assertIn("data->ao=ao", prepare)
        self.assertIn("drop_correlation_if_combined", prepare)

    def test_gamma_reference_is_separate_from_compact_production(self):
        self.assertIn("DHGammaStorage::ReferenceAll", self.source)
        self.assertIn("DHGammaStorage::ContractionOnly", self.source)
        self.assertIn("std::vector<double>().swap(out.separable_raw_ao)", self.source)
        self.assertIn("std::vector<double>().swap(out.nonseparable_raw_ao)", self.source)
        self.assertIn("Gradient::solve_dh_zvector(", self.driver)


if __name__ == "__main__":
    unittest.main()
