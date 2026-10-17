"""No-build O3 routing guards. Numerical acceptance belongs to C++/FD tests."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class O3ZVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = (ROOT / "src/dft/dh_zvector.h").read_text()
        cls.gmres = (ROOT / "src/dft/response_gmres.h").read_text()
        cls.driver = (ROOT / "src/dft/driver.cpp").read_text()

    def test_one_checked_physical_action(self):
        self.assertEqual(self.solver.count("apply_dh_eq27_hessian("), 1)
        self.assertNotIn("build_ks_orbital_hessian_op", self.solver)
        self.assertIn("a*no+i", self.solver)
        self.assertIn("const Eigen::VectorXd b=-pack(rhs_ai)", self.solver)
        self.assertIn("auto final=action(z)", self.solver)
        self.assertIn("fresh final residual exceeds", self.solver)
        self.assertIn("catch(const std::exception &e)", self.solver)

    def test_dense_only_in_explicit_reference_branch(self):
        iterative = self.solver.split("if (options.backend==DHZVectorBackend::MatrixFreeGMRES)", 1)[1].split(
            "else if (options.backend==DHZVectorBackend::DenseReference)", 1)[0]
        self.assertIn("DFT::Response::gmres", iterative)
        self.assertIn("no dense fallback", iterative)
        self.assertNotIn("hessian(n,n)", iterative)
        self.assertNotIn("qr.solve", iterative)
        assembly = self.driver.split("build_dh_eq41_response_and_solve_zvector(\n", 2)[-1].split(
            "std::expected<PreparedSystem", 1)[0]
        self.assertIn("Gradient::solve_dh_zvector", assembly)
        self.assertNotIn("hessian.col(", assembly)
        self.assertNotIn("colPivHouseholderQr", assembly)

    def test_preserves_probe_algorithm_and_settings(self):
        for setting in ("tolerance = 1e-12", "restart = 24", "max_iterations = 256"):
            self.assertIn(setting, self.solver)
        self.assertIn("pass < 2", self.gmres)
        self.assertIn("diagonal.cwiseAbs().cwiseMax(1e-8).cwiseInverse()", self.gmres)
        self.assertIn("std::min({restart,", self.gmres)
        self.assertIn("action_count", self.gmres)
        self.assertIn("krylov_matrix_bytes", self.gmres)
        bridge = (ROOT / "src/dft/dh_probe_gmres.h").read_text()
        self.assertIn("using DFT::Response::gmres", bridge)
        self.assertNotIn("while (", bridge)

    def test_production_defaults_iterative_and_probe_explicitly_dense(self):
        self.assertIn("backend = DHZVectorBackend::MatrixFreeGMRES", self.solver)
        production = self.driver.split("compute_analytic_dh_gradient(", 1)[1].split(
            "run_analytic_gradient_current_geometry(", 1)[0]
        self.assertIn('if (std::getenv("PLANCK_DFT_DH_HESSIAN_PROBE_LOG"))', production)
        self.assertIn("z_options.backend=Gradient::DHZVectorBackend::DenseReference", production)
        self.assertIn("contract_residual > 1e-9", production)


if __name__ == "__main__":
    unittest.main()
