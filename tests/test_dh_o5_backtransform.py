"""No-build O5 routing guards; numerical acceptance needs rebuilt C++ tests."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class O5BacktransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/dh_pt2_gradient.cpp").read_text()

    def test_four_successive_axes_and_single_pair_weight(self):
        primitive = self.source.split("build_dh_eq47_nonseparable_raw_ao(", 1)[1].split(
            "std::expected<DHEq46_47TwoParticleDensity", 1)[0]
        self.assertEqual(primitive.count("transform_pair_axis("), 4)
        for axis, block in ((3, "rightCols(nv)"), (2, "leftCols(no)"),
                            (1, "rightCols(nv)"), (0, "leftCols(no)")):
            self.assertIn(f"tensor,shape,{axis},mo_coeff.{block},column_tile", primitive)
        self.assertIn("(1+(i==j))*amplitudes.t_tilde[idx(i,j,a,b,no,nv)]", primitive)
        self.assertNotIn("exact_exchange", primitive)
        self.assertNotIn("c_pt2", primitive)

    def test_gemm_panels_and_batched_last_index(self):
        helper = self.source.split("std::vector<double> transform_pair_axis(", 1)[1].split(
            "namespace DFT::Gradient", 1)[0]
        self.assertIn("Eigen::Map<const RowMatrix>", helper)
        self.assertIn("source.middleRows(begin,width) * coefficients.transpose()", helper)
        self.assertIn("coefficients * source.middleCols(begin,width)", helper)
        self.assertNotIn("t_tilde", helper)

    def test_production_calls_staged_primitive_and_preserves_storage(self):
        builder = self.source.split("build_dh_eq46_47_two_particle_density(", 1)[1].split(
            "std::expected<DHEq33NonXCGradient", 1)[0]
        self.assertEqual(builder.count("build_dh_eq47_nonseparable_raw_ao("), 1)
        self.assertNotIn("for (int i = 0; i < no; ++i)", builder)
        self.assertIn("out.nonseparable_raw_ao=std::move(*nonseparable)", builder)
        self.assertIn("eri_eightfold_symmetrize(out.nonseparable_raw_ao", builder)
        self.assertIn("std::vector<double>().swap(out.nonseparable_raw_ao)", builder)

    def test_scalar_oracle_and_timing_ledger_are_test_only(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        main = (ROOT / "tests/dh_pt2_amplitude_density.cpp").read_text()
        oracle = (ROOT / "tests/dh_o5_backtransform.cpp").read_text()
        self.assertIn("tests/dh_o5_backtransform.cpp", cmake)
        self.assertIn("ok &= test_dh_o5_backtransform()", main)
        self.assertIn("literal_terms=", oracle)
        self.assertIn("staged_macs=", oracle)
        self.assertNotIn("O5 scaling", self.source)


if __name__ == "__main__":
    unittest.main()
