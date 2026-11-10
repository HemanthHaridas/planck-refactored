"""U4 no-build architecture guards; rebuilt C++ oracles establish numerics."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = (ROOT / "src/dft/udh_zvector.h").read_text()
        cls.source = (ROOT / "src/dft/udh_zvector.cpp").read_text()
        cls.oracle = (ROOT / "tests/udh_zvector.cpp").read_text()

    def test_physical_spin_trial_and_visible_channels(self):
        self.assertIn("return raw+raw.transpose()", self.source)
        self.assertNotIn("2.0 *", self.source)
        self.assertIn("in.response.apply_channels(out.delta_density)", self.source)
        for channel in ("orbital", "coulomb", "exchange", "xc_from_alpha", "xc_from_beta"):
            self.assertIn("out." + channel, self.source)

    def test_packing_and_shared_solver_are_explicit(self):
        self.assertIn("v(l.na+a*l.ob+i)=x.beta(a,i)", self.source)
        self.assertIn("v(a*l.oa+i)=x.alpha(a,i)", self.source)
        self.assertIn("solve_packed_zvector(apply,-pack(rhs,l),diagonal", self.source)
        self.assertIn("out.snapshot=in; out.rhs_ai=rhs", self.source)
        for gap in ("in.fock_mo.alpha(l.oa+a,l.oa+a)-in.fock_mo.alpha(i,i)",
                    "in.fock_mo.beta(l.ob+a,l.ob+a)-in.fock_mo.beta(i,i)"):
            self.assertIn(gap, self.source)

    def test_no_dense_jacobian_in_production(self):
        for guard in ("positive-definite AO metric", "canonical orbitals"):
            self.assertIn(guard, self.source)
        for dense in ("jacobian", "JacobiSVD", "HouseholderQR", "reciprocal_condition", "rank"):
            self.assertNotIn(dense, self.source)
            self.assertNotIn(dense, self.header)
        oracle = (ROOT / "tests/dh_zvector_oracle.h").read_text()
        self.assertIn("apply_udh_eq27_hessian(in,unit)", oracle)
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-zvector", 1)[1].split("    )", 1)[0]
        for source in ("tests/udh_zvector.cpp", "dft/udh_zvector.cpp", "dft/udh_pt2_orbital.cpp",
                       "dft/udh_ks_response.cpp", "dft/dh_pt2_gradient.cpp"):
            self.assertIn(source, target)
        self.assertNotIn("dft/driver.cpp", target)
        driver = (ROOT / "src/dft/driver.cpp").read_text()
        self.assertNotIn("solve_udh_zvector", driver)

    def test_fd_reference_uses_moving_fock_not_action(self):
        potential = self.oracle.split("AO jk(", 1)[1].split("UDHKSResponseOperator response", 1)[0]
        self.assertIn("eri(m,k,v,l)", potential)
        self.assertNotIn("apply_udh", potential)
        for evidence in ("fd.transpose().colPivHouseholderQr()", "build_udh_orbital_rhs",
                         "evaluate_udh_pt2_stationary_scalar", "fresh final residual",
                         "apply_dh_eq27_hessian", "closed_shell()", "audit(make(2,0,3))",
                         "audit(make(4,1,3))", "for(double step:{1e-4,3e-5})"):
            self.assertIn(evidence, self.oracle)

    def test_rectangular_solver_does_not_expand_u3_all_active_contract(self):
        self.assertIn('rhs.error().find("invalid all-active matrix shape")', self.oracle)
        self.assertIn("solve_udh_zvector(in,rectangular_rhs)", self.oracle)
        for fixture in ("audit(make(2,1,4))", "audit(make(1,2,4))",
                        "audit(make(2,0,4))", "audit(make(4,1,4))"):
            self.assertIn(fixture, self.oracle)
        u1 = (ROOT / "src/dft/udh_pt2_gradient.cpp").read_text()
        adapter = u1.split("transform_udh_pt2_dprime_to_ao(", 1)[1].split(
            "evaluate_udh_pt2_stationary_scalar(", 1)[0]
        self.assertIn("ca.cols()!=n", adapter)
        self.assertIn("cb.cols()!=n", adapter)


if __name__ == "__main__":
    unittest.main()
