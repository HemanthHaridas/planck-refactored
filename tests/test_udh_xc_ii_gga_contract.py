"""U8b no-build guards. Rebuilt C++ oracles establish numerical acceptance."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U8bXCIITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/udh_xc_ii_gga.cpp").read_text()
        cls.point = (ROOT / "src/dft/udh_xc_ii_gga_detail.h").read_text()
        cls.oracle = (ROOT / "tests/udh_xc_ii_gga.cpp").read_text()

    def test_five_variable_chain_rule(self):
        for token in ("u.col(0).dot(v.col(1))+u.col(1).dot(v.col(0))", "q=h*sd",
                      "2*fy(2)*v.col(0)+fy(3)*v.col(1)", "2*fy(4)*v.col(1)+fy(3)*v.col(0)",
                      "out.d_gradient.col(0)=2*fy(2)*u.col(0)+fy(3)*u.col(1)"):
            self.assertIn(token, self.point)
        self.assertIn("rs[3*r+s]", self.point)
        self.assertIn("h(r+2,s+2)=h(s+2,r+2)=ss[k++]", self.point)
        self.assertNotIn("kxc", self.source)
        self.assertNotIn("c_pt2", self.source)

    def test_hessians_and_four_channels(self):
        for token in ("&hess.h_xx,&hess.h_xy,&hess.h_xz,&hess.h_yy,&hess.h_yz,&hess.h_zz",
                      "dg_axis_spin(*pd[spin]", "dg_axis_spin(*dd[spin]",
                      "q.p_gradient.col(spin).dot(spatial_column(*pd[spin]",
                      "q.d_gradient.col(spin).dot(spatial_column(*dd[spin]",
                      "grid.atomic_weights(p)*(*partition)(a,k)*q.scalar",
                      "out.p_side_fixed+out.d_side_ao+out.becke_partition+out.point_translation"):
            self.assertIn(token, self.source)

    def test_independent_local_and_molecular_oracles(self):
        scalar = self.oracle.split("double scalar(", 1)[1].split("std::size_t crossings(", 1)[0]
        for token in ("build_udh_gga", "evaluate_gga_fxc", "udh_gga_xc_ii_partials", "dg_axis_spin"):
            self.assertNotIn(token, scalar)
        for token in ("point_chain_rule()", "source<5", "target<5", "k<16",
                      "scalar(f.grid,basis_ao,f.ao", "scalar(f.grid,f.ao,basis_ao",
                      "scalar(full_grid,f.ao,f.ao", "scalar(f.grid,point_ao,point_ao",
                      "scalar(full_grid,full_ao,full_ao", "closed_shell(f)",
                      "check(changed_masks==0)", "hyb_gga_xc_b2plyp"):
            self.assertIn(token, self.oracle)

    def test_fail_closed_and_combined(self):
        for token in ("if(!combined)", "nonfinite XC potential/kernel", "partition.error()",
                      "invalid AO values/gradients/Hessians", "nonfinite geometry channel"):
            self.assertIn(token, self.source)
        self.assertIn("bad.ao_hessian=nullptr", self.oracle)
        self.assertIn("h.h_zz.resize(0,0)", self.oracle)

    def test_two_smooth_acceptance_steps(self):
        self.assertIn("for(double h:{1e-3,2e-4,1e-4})", self.oracle)
        self.assertIn("const bool acceptance_step=h<=2e-4", self.oracle)
        self.assertIn("if(acceptance_step) check(changed_masks==0)", self.oracle)
        self.assertIn("coarse diagnostic only", self.oracle)
        self.assertIn("near(fd,ref,ch==4?4e-6:2e-6)", self.oracle)

    def test_detached_link_dependencies(self):
        target = (ROOT / "CMakeLists.txt").read_text().split(
            "add_executable(planck-udh-xc-ii-gga", 1)[1].split("    )", 1)[0]
        for path in ("dft/udh_xc_ii_gga.cpp", "dft/ks_matrix.cpp", "tests/udh_xc_ii_gga.cpp"):
            self.assertIn(path, target)
        self.assertNotIn("dft/driver.cpp", target)
        self.assertNotIn("build_udh_gga_complete_xc_ii_gradient", (ROOT / "src/dft/driver.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
