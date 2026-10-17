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
            "out.relaxed_ao.total(mu,nu)*p(ka,ta)",
            "-ax*out.relaxed_ao.spin.alpha(mu,ka)*out.ground_ao.alpha(nu,ta)",
            "-ax*out.relaxed_ao.spin.beta(mu,ka)*out.ground_ao.beta(nu,ta)",
            "pair(t.aa,oa,oa,va,va,coeff.alpha,coeff.alpha,c,n4)",
            "pair(t.ab,oa,ob,va,vb,coeff.alpha,coeff.beta,2*c,n4)",
            "pair(t.bb,ob,ob,vb,vb,coeff.beta,coeff.beta,c,n4)",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("t_tilde", self.source)
        self.assertNotIn("RMP2", self.source)

    def test_staged_transform_and_explicit_symmetry(self):
        for token in ("transform_axis(v,shape,3,ct.rightCols(vt))",
                      "transform_axis(v,shape,2,ct.leftCols(ot))",
                      "transform_axis(v,shape,1,cs.rightCols(vs))",
                      "transform_axis(v,shape,0,cs.leftCols(os))"):
            self.assertIn(token, self.source)
        self.assertIn("raw_ao,symmetric_ao", self.header)
        self.assertIn("average(g.raw_ao,p,q,r,s,n)", self.source)
        self.assertIn("valid_eri(dx.eri_ao[x],n,n4)", self.source)

    def test_checked_common_snapshot_and_ao_consumer(self):
        for token in ("build_udh_pt2_dprime(t,c)", "transform_udh_relaxed_density_to_ao(d,coeff,s)",
                      "relaxed Dprime does not match", "nonfinite gradient channel",
                      "out.one_electron+out.overlap+out.two_electron"):
            self.assertIn(token, self.source)
        self.assertIn("const DHEq33NonXCDerivatives &derivatives", self.header)
        self.assertNotIn("const DHRelaxedDifferenceDensity", self.header)

    def test_independent_oracles_and_physical_geometry(self):
        scalar = self.oracle.split("std::array<double,8> scalar(", 1)[1].split("Integrals model(", 1)[0]
        for token in ("build_udh_two_particle", "build_udh_non_xc", "literal_pair", "symmetric_ao"):
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
