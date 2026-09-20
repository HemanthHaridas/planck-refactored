"""U5 no-build routing/ownership guards; rebuilt scalar tests prove numerics."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/udh_relaxed_density.cpp").read_text()
        cls.header = (ROOT / "src/dft/udh_relaxed_density.h").read_text()
        cls.oracle = (ROOT / "tests/udh_relaxed_density.cpp").read_text()

    def test_single_raw_z_and_half_symmetric_adapter(self):
        self.assertIn("raw.bottomLeftCorner(z.rows(),z.cols())=z", self.source)
        self.assertIn("sym=0.5*raw+0.5*raw.transpose()", self.source)
        self.assertNotIn("topRightCorner(z.cols(),z.rows())=", self.source)
        self.assertNotIn("c_pt2", self.source)
        self.assertNotIn("ground_density", self.source)
        self.assertIn("UDHOrbitalMatrices raw_mo,symmetric_mo", self.header)

    def test_spin_frames_transform_before_sum(self):
        self.assertIn("c.alpha*d.symmetric_mo.alpha*c.alpha.transpose()", self.source)
        self.assertIn("c.beta*d.symmetric_mo.beta*c.beta.transpose()", self.source)
        self.assertIn("out.total=out.spin.alpha+out.spin.beta", self.source)
        for guard in ("positive-definite metric", "C^T S C frame", "nonfinite transformed density",
                      "diagonal_blocks(d,z.cols())", "raw_layout(raw,sym,o)"):
            self.assertIn(guard, self.source)

    def test_scalar_oracle_and_handoff(self):
        scalar = self.oracle.split("double scalar(", 1)[1].split("void fixture(", 1)[0]
        self.assertIn("z(a,i)*c(mu,z.cols()+a)*f(mu,nu)*c(nu,i)", scalar)
        self.assertNotIn("symmetric_mo", scalar)
        self.assertNotIn("transform_udh", scalar)
        for check in ("solve_udh_zvector", "build_dh_relaxed_difference_density",
                      "fixture(0,0,0,0)", "fixture(2,1,2,2)", "1e-3,1e-4"):
            self.assertIn(check, self.oracle)

    def test_detached_target_and_unchanged_driver_guard(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-relaxed-density", 1)[1].split("    )", 1)[0]
        for source in ("tests/udh_relaxed_density.cpp", "dft/udh_relaxed_density.cpp",
                       "dft/udh_zvector.cpp", "dft/dh_pt2_gradient.cpp"):
            self.assertIn(source, target)
        self.assertNotIn("dft/driver.cpp", target)
        driver = (ROOT / "src/dft/driver.cpp").read_text()
        self.assertNotIn("build_udh_relaxed_difference_density", driver)
        self.assertNotIn("transform_udh_relaxed_density_to_ao", driver)
        u1 = (ROOT / "src/dft/udh_pt2_gradient.cpp").read_text()
        self.assertIn("ca.cols()!=n", u1)
        self.assertIn("cb.cols()!=n", u1)


if __name__ == "__main__":
    unittest.main()
