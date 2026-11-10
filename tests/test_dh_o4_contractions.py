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
        # Two overloads: the dense-MO-ERI reference and the typed production
        # route. Each takes the four-slot gradient once, with no direction scan.
        overloads = [body.split("std::expected<", 1)[0] for body in
                     self.source.split("build_dh_eq40_amplitude_rhs(\n")[1:]]
        self.assertEqual(len(overloads), 2)
        reference, typed = overloads
        for rhs in overloads:
            self.assertEqual(rhs.count("build_dh_pair_coefficient_gradient("), 1)
            self.assertNotIn("if (k == i)", rhs)
            self.assertNotIn("if (c == a)", rhs)
        self.assertIn("const std::vector<double> &mo_eri", reference)
        self.assertIn("if (!include_legacy_internal) continue", reference)
        self.assertIn("const DHMolecularERIBlocks &eri_blocks", typed)
        self.assertIn("|| include_legacy_internal)", typed)
        self.assertIn("bool include_legacy_internal = false", self.header)

    def test_pair_overlap_reuses_stationary_coefficients(self):
        assembly = self.source.split("assemble_dh_gradient_driver_contract(", 1)[1].split(
            "build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs, DHStationaryProducts", 1)[0]
        self.assertIn("out.eq40_amplitude_rhs.pair_coefficients, nocc, nvirt", assembly)
        self.assertNotIn("build_dh_pair_coefficient_gradient(", assembly)
        self.assertIn("pair_coefficients.total.bottomLeftCorner(nvirt,nocc)", assembly)
        self.assertIn("std::move(stationary)", assembly)

    def test_density_preserves_orientation_and_unordered_pair_domain(self):
        density = self.source.split("build_dh_pt2_amplitude_density(", 1)[1].split(
            "std::expected<DHPairCoefficientGradient", 1)[0]
        # Eq. 37: tilde [a,b] at (i,k) against the raw [b,a] panel at (k,j).
        self.assertIn("out.read_pair_tile(i*nv,k*nv,nv,nv)", density)
        self.assertIn("result,k*nv,j*nv,nv,nv)", density)
        self.assertIn("left->values[static_cast<std::size_t>(a)*nv+b]*\n"
                      "                    right->values[static_cast<std::size_t>(b)*nv+a]", density)
        # Eq. 38 over the unordered domain i<=j: tilde t^T plus tilde^T t.
        self.assertIn("for (int j = i; j < no; ++j)", density)
        self.assertIn("left->values[static_cast<std::size_t>(a)*nv+k]*\n"
                      "                                right->values[static_cast<std::size_t>(b)*nv+k]", density)
        self.assertIn("left->values[static_cast<std::size_t>(k)*nv+a]*\n"
                      "                                right->values[static_cast<std::size_t>(k)*nv+b]", density)
        self.assertNotIn("t2_at(", density.split("Eq. (37)", 1)[1])

    def test_numerical_oracles_are_wired(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        main = (ROOT / "tests/dh_pt2_amplitude_density.cpp").read_text()
        self.assertIn("tests/dh_o4_contractions.cpp", cmake)
        self.assertRegex(main, r"ok &= .*test_dh_o4_contractions\(\)")


if __name__ == "__main__":
    unittest.main()
