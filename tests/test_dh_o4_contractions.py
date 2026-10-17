"""No-build O4 routing guards; C++ scalar/FD oracles establish numerics."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class O4ContractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/dh_pt2_gradient.cpp").read_text()
        cls.header = (ROOT / "src/dft/dh_pt2_gradient.h").read_text()

    def test_four_slot_tiled_products_without_direction_scan(self):
        pair = self.source.split("build_dh_pair_coefficient_gradient(", 1)[1].split(
            "std::expected<DHEq40AmplitudeRHS", 1)[0]
        self.assertIn("slot<4", pair)
        self.assertIn("if (s!=slot) count*=dimensions[s]", pair)
        self.assertIn("out.slots[slot]=tiled_product", pair)
        self.assertIn("out.noalias() += a.leftCols(width) * b.topRows(width)", self.source)
        rhs = self.source.split("build_dh_eq40_amplitude_rhs(", 1)[1].split(
            "build_dh_lagrangian_rhs(", 1)[0]
        self.assertEqual(rhs.count("build_dh_pair_coefficient_gradient("), 1)
        self.assertNotIn("if (k == i)", rhs)
        self.assertNotIn("if (c == a)", rhs)
        self.assertIn("if (!include_legacy_internal) continue", rhs)
        self.assertIn("bool include_legacy_internal = false", self.header)

    def test_pair_overlap_reuses_stationary_coefficients(self):
        assembly = self.source.split("assemble_dh_gradient_driver_contract(", 1)[1].split(
            "build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs)", 1)[0]
        self.assertIn("out.eq40_amplitude_rhs.pair_coefficients, nocc, nvirt", assembly)
        self.assertNotIn("build_dh_pair_coefficient_gradient(", assembly)
        self.assertIn("pair_coefficients.total.bottomLeftCorner(nvirt,nocc)", assembly)
        self.assertIn("std::move(stationary)", assembly)

    def test_density_preserves_orientation_and_unordered_pair_domain(self):
        density = self.source.split("build_dh_pt2_amplitude_density(", 1)[1].split(
            "std::expected<DHPairCoefficientGradient", 1)[0]
        self.assertIn("result.t2[idx(k,j,b,a,no,nv)]", density)
        self.assertIn("for (int j = i; j < no; ++j)", density)
        self.assertIn("Eigen::RowMajor", density)
        self.assertIn("tilde * raw.transpose()", density)
        self.assertIn("tilde.transpose() * raw", density)

    def test_numerical_oracles_are_wired(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        main = (ROOT / "tests/dh_pt2_amplitude_density.cpp").read_text()
        self.assertIn("tests/dh_o4_contractions.cpp", cmake)
        self.assertIn("ok &= test_dh_o4_contractions()", main)


if __name__ == "__main__":
    unittest.main()
