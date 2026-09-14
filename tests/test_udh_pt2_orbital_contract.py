"""U3 no-build boundaries. Numerical validation requires the rebuilt C++ target."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = (ROOT / "src/dft/udh_pt2_orbital.h").read_text()
        cls.source = (ROOT / "src/dft/udh_pt2_orbital.cpp").read_text()
        cls.test = (ROOT / "tests/udh_pt2_orbital.cpp").read_text()

    def test_four_slots_and_spin_sectors(self):
        self.assertIn("std::array<UDHOrbitalMatrices,4> slots", self.header)
        for term in ("g0(p,i)", "g1(p,os+a)", "g2(p,j)", "g3(p,ot+b)"):
            self.assertIn(term, self.source)
        self.assertIn("out.ab=sector(t.ab,g.ab,t,0,1,2*c)", self.source)
        self.assertNotIn("exact_exchange", self.source.split("build_udh_orbital_rhs(", 1)[0])

    def test_checked_u1_u2_composition(self):
        for call in ("validate_udh_pt2_amplitudes(t)", "build_udh_pt2_dprime(t,c)",
                     "transform_udh_pt2_dprime_to_ao", "response.apply_channels"):
            self.assertIn(call, self.source)
        self.assertIn("2.0*coeff.alpha.rightCols(va).transpose()", self.source)
        for field in ("response_j", "response_k", "xc_from_alpha", "xc_from_beta", "fock_connection_ai"):
            self.assertIn(field, self.header)
        self.assertIn("sum(sum(out.pair.total_ai,out.response_ai),out.fock_connection_ai)", self.source)

    def test_oracles_do_not_differentiate_the_new_builder(self):
        scalar = self.test.split("double scalar(", 1)[1].split("void pair_checks(", 1)[0]
        self.assertNotIn("build_udh", scalar)
        self.assertIn("f.model.factors", scalar)
        self.assertIn("four-slot basis FD", self.test)
        self.assertIn("Eq.22 independently reduced external/internal sums", self.test)
        self.assertIn("rhs_checks(f,true)", self.test)
        self.assertIn("mixed_want", self.test)
        self.assertIn("2c (2t-t_exchange):g", self.test)

    def test_target_links_u1_and_u2_without_driver(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-pt2-orbital", 1)[1].split("    )", 1)[0]
        for source in ("tests/udh_pt2_orbital.cpp", "dft/udh_pt2_orbital.cpp",
                       "dft/udh_pt2_gradient.cpp", "dft/udh_ks_response.cpp"):
            self.assertIn(source, target)
        self.assertNotIn("dft/driver.cpp", target)
        driver = (ROOT / "src/dft/driver.cpp").read_text()
        self.assertNotIn("build_udh_orbital_rhs", driver)


if __name__ == "__main__":
    unittest.main()
