"""U9 no-build wiring/scope guards. C++ FD numerics require the user's rebuild."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U9StationaryTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = (ROOT / "tests/udh_stationary_transfer.cpp").read_text()

    def test_noncanonical_four_slot_operator_is_independent(self):
        action = self.oracle.split("V amplitude_action(", 1)[1].split("M amplitude_matrix", 1)[0]
        for term in ("fs(i,k)*at(k,j,a,b)", "ft(j,k)*at(i,k,a,b)",
                     "fs(os+a,os+e)*at(i,j,e,b)", "ft(ot+b,ot+e)*at(i,j,a,e)"):
            self.assertIn(term, action)
        self.assertNotIn("build_udh_", action)
        self.assertIn("base.b.array()/a.diagonal().array()", self.oracle)
        self.assertIn("solve(ap,plus.b-r)", self.oracle)
        self.assertIn("solve(M(a.transpose()),V(scale*base.b))", self.oracle)
        self.assertIn("2*scale*r.dot(response)", self.oracle)

    def test_common_scalar_does_not_consume_stationary_objects(self):
        scalar = self.oracle.split("std::array<double,8> scalar(", 1)[1].split("double total(", 1)[0]
        self.assertIn("-scale*q.dot(amplitude_action", scalar)
        self.assertIn("2*scale*q(k)*s.b(k)", scalar)
        self.assertNotIn("build_udh_", scalar)
        for token in ("build_udh_orbital_rhs", "solve_udh_zvector", "build_udh_relaxed_difference_density",
                      "build_udh_overlap_density", "build_udh_two_particle_density", "build_udh_non_xc_gradient"):
            self.assertIn(token, self.oracle)

    def test_channel_and_response_ledgers(self):
        for token in ("frozen_nonZ_error=", "frozen_Z_error=", "orbital_nonZ=", "orbital_Z=",
                      "amplitude_cancel_error=", "Eq33_error=", "KS_tangent=",
                      '"pair-aa","pair-ab","pair-bb","overlap"', "V::Unit(m,k)",
                      "near(nonz_response+z_response,0", "near(live_fd,sum(af)"):
            self.assertIn(token, self.oracle)
        self.assertIn("transport(r.alpha,oa,x.s+h*d.s)", self.oracle)
        self.assertIn("transport(r.beta,ob,x.s+h*d.s)", self.oracle)

    def test_standalone_scope_and_all_dependencies(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-stationary-transfer", 1)[1].split("    )", 1)[0]
        for source in ("tests/udh_stationary_transfer.cpp", "dft/udh_ks_response.cpp",
                       "dft/udh_zvector.cpp", "dft/udh_two_particle_density.cpp", "dft/udh_overlap_density.cpp"):
            self.assertIn(source, target)
        self.assertNotIn("dft/driver.cpp", target)
        self.assertIn("not molecular acceptance", self.oracle)
        # U8 physical grid operators retain their independently accepted suite;
        # this synthetic geometry callback must not be described as that test.
        self.assertNotIn("build_udh_gga_complete_xc_ii_gradient", self.oracle)


if __name__ == "__main__":
    unittest.main()
