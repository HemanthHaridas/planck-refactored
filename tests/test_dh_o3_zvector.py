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
        self.assertIn("solve_packed_zvector(action,-pack(rhs_ai),diagonal", self.solver)
        self.assertIn("auto final=counted(z)", self.solver)
        self.assertIn("fresh final residual exceeds", self.solver)
        self.assertIn("catch(const std::exception &e)", self.solver)

    def test_one_matrix_free_solver_no_dense_path(self):
        # RKS and UKS share solve_packed_zvector; the dense oracle lives in
        # tests/dh_zvector_oracle.h and production has no route to it.
        core = self.solver.split("solve_packed_zvector(", 1)[1].split("solve_dh_zvector(", 1)[0]
        self.assertIn("DFT::Response::gmres", core)
        self.assertIn("no dense fallback", core)
        for dense in ("DenseReference", "hessian(n,n)", "qr.solve", "colPivHouseholderQr"):
            self.assertNotIn(dense, self.solver)
        udh = (ROOT / "src/dft/udh_zvector.cpp").read_text()
        self.assertIn("solve_packed_zvector(", udh)
        for dense in ("jacobian", "JacobiSVD", "HouseholderQR", "gmres("):
            self.assertNotIn(dense, udh)
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

    def test_production_always_iterative_with_no_env_override(self):
        # No environment variable may switch the production gradient solver.
        production = self.driver.split("compute_analytic_dh_gradient(", 1)[1].split(
            "run_analytic_gradient_current_geometry(", 1)[0]
        self.assertNotIn("getenv", production)
        self.assertNotIn("DenseReference", production)
        self.assertIn("contract_residual > 1e-9", production)


if __name__ == "__main__":
    unittest.main()
