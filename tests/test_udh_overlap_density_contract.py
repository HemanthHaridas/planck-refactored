"""U6 no-build guards; rebuilt scalar/metric FD tests establish numerics."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U6OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/udh_overlap_density.cpp").read_text()
        cls.header = (ROOT / "src/dft/udh_overlap_density.h").read_text()
        cls.oracle = (ROOT / "tests/udh_overlap_density.cpp").read_text()

    def test_pair_gauge_and_separate_nonpair_channels(self):
        for term in ("b.topLeftCorner(o,o)=-0.5*g.topLeftCorner(o,o)",
                     "b.bottomRightCorner(v,v)=-0.5*g.bottomRightCorner(v,v)",
                     "b.topRightCorner(o,v)=-g.topRightCorner(o,v)",
                     "z(o+a,i)=-fm(i,i)*raw(o+a,i)"):
            self.assertIn(term, self.source)
        for member in ("pair_aa,pair_ab,pair_bb", "dprime_orbital,z_orbital", "response_dprime,response_z"):
            self.assertIn(member, self.header)
        self.assertNotIn("c_pt2", self.source)

    def test_physical_response_and_error_propagation(self):
        self.assertIn("response_w(prime_ao)", self.source)
        self.assertIn("response_w(z_ao)", self.source)
        self.assertIn("-coeff.leftCols(o).transpose()*q*coeff.leftCols(o)", self.source)
        for guard in ("canonical(f.alpha", "inconsistent pair sector sum", "nonfinite projected channel"):
            self.assertIn(guard, self.source)
        self.assertIn("transform_udh_relaxed_density_to_ao(d,c,s)", self.source)

    def test_common_metric_scalar_and_joint_xc_adjoint(self):
        scalar = self.oracle.split("std::array<double,13> scalar(", 1)[1].split(
            "std::array<const Orb*,13> channels", 1)[0]
        self.assertIn("x.model.channels(delta)", scalar)
        self.assertNotIn("build_udh_overlap", scalar)
        self.assertNotIn("apply_channels", scalar)
        self.assertIn("channel<2?4+channel:6+spin", scalar)
        for token in ("transport(x.c.alpha,oa,x.s+step*ds)", "transport(x.c.beta,ob,x.s+step*ds)",
                      "closed_shell()", "U6 injected Z-response failure", "std::swap(swapped.c.alpha"):
            self.assertIn(token, self.oracle)

    def test_no_driver_enablement(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-overlap-density", 1)[1].split("    )", 1)[0]
        self.assertIn("dft/udh_overlap_density.cpp", target)
        self.assertIn("tests/udh_overlap_density.cpp", target)
        self.assertNotIn("dft/driver.cpp", target)
        self.assertNotIn("build_udh_overlap_density", (ROOT / "src/dft/driver.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
