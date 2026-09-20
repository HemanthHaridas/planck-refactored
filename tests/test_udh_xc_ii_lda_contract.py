"""U8a no-build guards; numerical acceptance needs the rebuilt C++ oracle."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U8aXCIITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/udh_xc_ii.cpp").read_text()
        cls.header = (ROOT / "src/dft/udh_xc_ii.h").read_text()
        cls.oracle = (ROOT / "tests/udh_xc_ii_lda.cpp").read_text()

    def test_polarized_kernel_and_signed_difference(self):
        for token in ("rho[2*p]=ground->alpha.rho(p)", "rho[2*p+1]=ground->beta.rho(p)",
                      "qa=da*rr[3*p]+db*rr[3*p+1]", "qb=da*rr[3*p+1]+db*rr[3*p+2]",
                      "integrand=va*da+vb*db"):
            self.assertIn(token, self.source)
        self.assertNotIn("difference->alpha.rho.minCoeff()", self.source)
        self.assertNotIn("c_pt2", self.source)
        self.assertNotIn("evaluate_lda_kxc", self.source)

    def test_four_disjoint_geometry_channels(self):
        for token in ("drho_channel(in.ground_density.alpha", "drho_channel(in.relaxed_density.beta",
                      "becke_partition_owner_derivatives(grid,mol,p)", "grid.atomic_weights(p)*(*partition)(a,k)*integrand",
                      "gradient(ground->beta,p,k)", "gradient(difference->alpha,p,k)",
                      "out.p_side_fixed+out.d_side_ao+out.becke_partition+out.point_translation"):
            self.assertIn(token, self.source)
        lda_inputs = self.header.split("struct UDHXCIIInputs", 1)[1].split("struct UDHCompleteXCIIGradient", 1)[0]
        self.assertNotIn("ao_hessian", lda_inputs)

    def test_fail_closed_and_combined_functional(self):
        for token in ("XC_FLAGS_HAVE_FXC", "XC_FLAGS_HAVE_VXC", "XC::Spin::Polarized", "if(!combined)",
                      "negative reference spin density", "inconsistent owner/partition/atomic weight metadata",
                      "nonfinite geometry channel", "partition.error()"):
            self.assertIn(token, self.source)

    def test_independent_scalar_and_each_displacement(self):
        scalar = self.oracle.split("double scalar(", 1)[1].split("std::size_t crossings(", 1)[0]
        self.assertIn("evaluate_xc_on_grid", scalar)
        for token in ("evaluate_lda_fxc", "build_udh_lda", "drho_channel", "partition_owner_derivatives"):
            self.assertNotIn(token, scalar)
        for token in ("kernel_slots(c)", "closed_shell(f)", "Spin{f.d.alpha,zero}", "Spin{zero,f.d.beta}",
                      "scalar(f.grid,basis_ao,f.ao", "scalar(f.grid,f.ao,basis_ao",
                      "scalar(full_grid,f.ao,f.ao", "scalar(f.grid,point_ao,point_ao",
                      "scalar(full_grid,full_ao,full_ao", "check(changed_masks==0)",
                      "lda_xc_teter93", "for(double h:{1e-3,2e-4})"):
            self.assertIn(token, self.oracle)

    def test_detached_target_only(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-xc-ii-lda", 1)[1].split("    )", 1)[0]
        self.assertIn("dft/udh_xc_ii.cpp", target)
        self.assertIn("tests/udh_xc_ii_lda.cpp", target)
        # The independent ordinary-XC-matrix oracle needs its definition,
        # not just the declaration from ks_matrix.h.
        self.assertIn("dft/ks_matrix.cpp", target)
        self.assertNotIn("dft/driver.cpp", target)
        self.assertNotIn("build_udh_lda_complete_xc_ii_gradient", (ROOT / "src/dft/driver.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
