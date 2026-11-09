"""U7 no-build guards. Numerical acceptance requires the rebuilt C++ target."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U7TwoParticleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/udh_two_particle_density.cpp").read_text()
        cls.header = (ROOT / "src/dft/udh_two_particle_density.h").read_text()
        cls.oracle = (ROOT / "tests/udh_two_particle_density.cpp").read_text()

    def test_spin_factors_and_correction_boundary(self):
        for token in (
            "out.coulomb_d=g.relaxed_ao.total",
            "out.coulomb_p=g.ground_ao.alpha+g.ground_ao.beta",
            "out.exchange.push_back({g.relaxed_ao.spin.alpha,g.ground_ao.alpha,-ax})",
            "out.exchange.push_back({g.relaxed_ao.spin.beta,g.ground_ao.beta,-ax})",
            "channel(t.aa,oa,oa,va,va,coeff.alpha,coeff.alpha,c)",
            "channel(t.ab,oa,ob,va,vb,coeff.alpha,coeff.beta,2*c)",
            "channel(t.bb,ob,ob,vb,vb,coeff.beta,coeff.beta,c)",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("t_tilde", self.source)
        self.assertNotIn("RMP2", self.source)

    def test_no_retained_dense_tensor_or_second_consumer(self):
        for token in ("UDHERICoefficients", "build_udh_non_xc_gradient", "raw_ao", "eri_ao"):
            self.assertNotIn(token, self.source + self.header)

    def test_checked_common_snapshot(self):
        for token in ("build_udh_pt2_dprime(t,c)", "transform_udh_relaxed_density_to_ao(d,coeff,s)",
                      "relaxed Dprime does not match"):
            self.assertIn(token, self.source)
        self.assertNotIn("const DHRelaxedDifferenceDensity", self.header)

    def test_independent_oracles_and_physical_geometry(self):
        scalar = self.oracle.split("std::array<double,8> scalar(", 1)[1].split("Integrals model(", 1)[0]
        for token in ("build_udh_two_particle", "dense_non_xc", "literal_pair", "symmetric_ao"):
            self.assertNotIn(token, scalar)
        for token in ("literal_pair(x,spin)", "closed_shell()", "sw.t.ab[ti(j,i,b,a",
                      "for(double ax:{0.,.17,.53,1.})", "for(double step:{1e-3,2e-4})",
                      "integrals(molecule(a,k,step))", "rigid-translation channel max",
                      "_compute_2e_j_direct", "_compute_2e_k_direct"):
            self.assertIn(token, self.oracle)

    def test_no_driver_enablement_or_standalone_build(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-two-particle-density", 1)[1].split("    )", 1)[0]
        self.assertIn("dft/udh_two_particle_density.cpp", target)
        self.assertIn("tests/udh_two_particle_density.cpp", target)
        self.assertNotIn("dft/driver.cpp", target)
        self.assertNotIn("build_udh_two_particle_density", (ROOT / "src/dft/driver.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
