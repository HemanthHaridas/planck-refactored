"""No-build O5 routing guards; numerical acceptance needs rebuilt C++ tests."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class O5BacktransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/dh_pt2_gradient.cpp").read_text()

    def test_four_successive_axes_and_single_pair_weight(self):
        penultimate = self.source.split("dh_eq47_penultimate(const DHPT2AmplitudeDensity", 1)[1].split(
            "for_each_dh_eq47_nonseparable_raw_ao_tile(", 1)[0]
        self.assertEqual(penultimate.count("transform_pair_axis("), 3)
        for axis, block in ((3, "rightCols(nv)"), (2, "leftCols(no)"), (1, "rightCols(nv)")):
            self.assertIn(f"tensor,shape,{axis},mo_coeff.{block},column_tile", penultimate)
        self.assertIn("(1+(i==j))*tile.values", penultimate)
        self.assertNotIn("exact_exchange", penultimate)
        self.assertNotIn("c_pt2", penultimate)
        self.assertIn("target.noalias()=mo_coeff.middleRows(begin,width).leftCols(no)*source", self.source)

    def test_block_stream_axes_and_single_pair_weight(self):
        channel = self.source.split("DHPairChannel make_dh_eq47_pair_channel(", 1)[1].split(
            "for_each_dh_eq47_nonseparable_raw_ao_block(", 1)[0]
        self.assertEqual(channel.count("(1+((i0+il)==j))*amplitudes.tilde(i0+il,j,a,b)"), 1)
        stream = self.source.split("for_each_dh_pair_ao_block(\n        int nao", 1)[1].split(
            "build_dh_eq47_nonseparable_raw_ao(", 1)[0]
        self.assertEqual(stream.count("transform_pair_axis("), 3)
        for token in ("ct.middleRows(tau0,ntile).rightCols(vt),tau_tile",
                      "ct.middleRows(kappa0,ktile).leftCols(ot),kappa_tile",
                      "cs.middleRows(nu0,vtile).rightCols(vs),nu_tile",
                      "result.noalias() += cs.leftCols(os).middleCols(i0,itile)*source"):
            self.assertIn(token, stream)
        self.assertNotIn("exact_exchange", stream)
        self.assertNotIn("c_pt2", stream)

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
        self.assertEqual(builder.count("if (storage == DHGammaStorage::ReferenceAll)"), 2)

    def test_scalar_oracle_and_timing_ledger_are_test_only(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        main = (ROOT / "tests/dh_pt2_amplitude_density.cpp").read_text()
        oracle = (ROOT / "tests/dh_o5_backtransform.cpp").read_text()
        self.assertIn("tests/dh_o5_backtransform.cpp", cmake)
        self.assertIn('delegated_gate("DH O5 backtransform",test_dh_o5_backtransform())', main)
        self.assertIn("literal_terms=", oracle)
        self.assertIn("staged_macs=", oracle)
        self.assertNotIn("O5 scaling", self.source)


if __name__ == "__main__":
    unittest.main()
