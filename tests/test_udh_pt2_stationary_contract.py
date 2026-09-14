"""No-build U1 boundary guards; numerical acceptance belongs to the C++ target."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class U1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/dft/udh_pt2_gradient.cpp").read_text()
        cls.header = (ROOT / "src/dft/udh_pt2_gradient.h").read_text()

    def test_detached_amplitudes_have_no_canonical_state(self):
        raw = self.header.split("struct UDHPT2Amplitudes", 1)[1].split("};", 1)[0]
        for name in ("nocc_alpha", "nocc_beta", "nvirt_alpha", "nvirt_beta", "aa, ab, bb"):
            self.assertIn(name, raw)
        for name in ("energy", "denominator", "UMP2Result", "converged"):
            self.assertNotIn(name, raw)

    def test_off_shell_uses_full_fock_without_canonical_gate(self):
        scalar = self.source.split("evaluate_udh_pt2_stationary_scalar(", 1)[1].split(
            "build_udh_pt2_stationary_contract(", 1)[0]
        self.assertIn("build_udh_pt2_dprime(t,c)", scalar)
        self.assertIn("alpha_mo.cwiseProduct(fa).sum()", scalar)
        self.assertIn("beta_mo.cwiseProduct(fb).sum()", scalar)
        for forbidden in ("asDiagonal", "eigenvalues", "denominator", "build_udh_pt2_energy_contract"):
            self.assertNotIn(forbidden, scalar)
        handoff = self.source.split("build_udh_pt2_stationary_contract(", 1)[1]
        self.assertIn("build_udh_pt2_energy_contract(r,g,s,options,c,scope)", handoff)

    def test_independent_oracle_is_in_detached_target(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        target = cmake.split("add_executable(planck-udh-pt2-stationary", 1)[1].split("    )", 1)[0]
        self.assertIn("tests/udh_pt2_stationary.cpp", target)
        self.assertIn("dft/udh_pt2_gradient.cpp", target)
        oracle = (ROOT / "tests/udh_pt2_stationary.cpp").read_text()
        residual = oracle.split("residual_oracle(const SpinOrbital", 1)[1].split(
            "UDHPT2StationaryScalar eval", 1)[0]
        self.assertNotIn("dprime", residual.lower())
        self.assertNotIn("build_udh", residual)
        self.assertIn("amplitude_checks(x,true)", oracle)
        self.assertIn("amplitude_checks(off,false)", oracle)

    def test_production_uks_guard_unchanged(self):
        driver = (ROOT / "src/dft/driver.cpp").read_text()
        guard = driver.split("// Restricted global-double-hybrid Cartesian derivatives", 1)[1].split(
            "apply_post_ks_double_hybrid_correction(", 1)[0]
        self.assertIn("calculator._scf._scf != HartreeFock::SCFType::UHF", guard)
        self.assertNotIn("build_udh_pt2_stationary_contract", driver)
        self.assertNotIn("evaluate_udh_pt2_stationary_scalar", driver)


if __name__ == "__main__":
    unittest.main()
