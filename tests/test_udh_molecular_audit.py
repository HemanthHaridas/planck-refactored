"""U10/U11 no-build runner and integration-boundary checks."""
import math
from pathlib import Path
import unittest

from udh_molecular_audit import (bohr_separation, centered_fourth_order, continuity,
                                 eigenvalues_symmetric, matmul, maxdiff, requested_matrix, transpose)

ROOT = Path(__file__).resolve().parents[1]


class MolecularMathTests(unittest.TestCase):
    def test_bohr_denominator_uses_executable_coordinates(self):
        plus = dict(coordinates_requested=[[0.2, 0.3, 0.4]])
        minus = dict(coordinates_requested=[[0.198, 0.3, 0.4]])
        self.assertAlmostEqual(bohr_separation(plus, minus, 0, 0), .002)
        with self.assertRaises(ValueError):
            bohr_separation(minus, plus, 0, 0)

    def test_fourth_order_oracle_cancels_cubic_without_analytic_input(self):
        f = lambda x: 2*x + 3*x*x + 7*x**3
        for h, wide in ((.01, .02), (.01, .0200001)):
            fourth, second = centered_fourth_order((f(h), f(-h)), (f(wide), f(-wide)), 2*h, 2*wide)
            self.assertAlmostEqual(fourth, 2., places=12)
            self.assertAlmostEqual(second, 2+7*h*h, places=12)
        # Quintic remainder decays as h^4 rather than being fitted away.
        error = []
        for h in (.1, .05):
            fourth, _ = centered_fourth_order((h**5, -h**5), ((2*h)**5, -(2*h)**5), 2*h, 4*h)
            self.assertAlmostEqual(fourth, -4*h**4, places=14)
            error.append(fourth)
        self.assertAlmostEqual(error[0]/error[1], 16.)
        for radii in ((0., 1.), (1., 1.), (2., 1.), (1., math.nan)):
            with self.assertRaises(ValueError):
                centered_fourth_order((1., -1.), (2., -2.), *radii)

    def test_jacobi_spectrum(self):
        values = sorted(eigenvalues_symmetric([[2., 1.], [1., 2.]]))
        self.assertAlmostEqual(values[0], 1.)
        self.assertAlmostEqual(values[1], 3.)
        self.assertEqual(eigenvalues_symmetric([[.4]]), [.4])

    def state(self):
        return dict(converged=True, nalpha=2, nbeta=0, s2=2.,
                    occupied_alpha=[[1., 0.], [0., 1.], [0., 0.]], occupied_beta=[[], [], []],
                    cross_overlap_reference_current=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def test_subspace_phase_and_internal_rotation_invariance(self):
        center = self.state(); endpoint = self.state()
        endpoint["occupied_alpha"] = [[.8, -.6], [.6, .8], [0., 0.]]
        self.assertAlmostEqual(continuity(center, endpoint)["alpha"], 1.)
        self.assertEqual(continuity(center, endpoint)["beta"], 1.)

    def test_subspace_switch_and_spin_switch_fail(self):
        center = self.state()
        for key, value in (("occupied_alpha", [[1., 0.], [0., 0.], [0., 1.]]),
                           ("nbeta", 1), ("s2", 2.2), ("s2", math.nan), ("converged", False)):
            endpoint = self.state(); endpoint[key] = value
            with self.assertRaises(ValueError):
                continuity(center, endpoint)

    def test_explicit_frame_mapping(self):
        standard = [[0., 0., 0.], [1., 0., 0.], [.3, 1., .4]]
        rotation = [[.8, -.6, 0.], [.48, .64, -.6], [.36, .48, .8]]
        requested = [[x+t for x, t in zip(row, [.2, .3, -.4])] for row in matmul(standard, rotation)]
        gradient = [[.3, .4, -.1], [.5, -.6, .3], [.1, .7, -.2]]
        state = dict(coordinates_standard=standard, coordinates_requested=requested, gradient=gradient)
        self.assertLess(maxdiff(requested_matrix(state, "gradient"), matmul(gradient, rotation)), 1e-14)
        self.assertLess(maxdiff(matmul(rotation, transpose(rotation)), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), 1e-14)


class U10WiringTests(unittest.TestCase):
    def test_same_snapshot_all_paper_builders_and_no_callback_escape(self):
        source = (ROOT/"src/dft/udh_driver.cpp").read_text()
        for token in ("evaluate_current_density_and_xc", "assemble_current_ks_potential",
                      "physical_fock_error>1e-8", "build_udh_pt2_stationary_contract",
                      "make_udh_direct_ks_response_operator", "build_udh_orbital_rhs", "solve_udh_zvector",
                      "build_udh_relaxed_difference_density", "build_udh_overlap_density",
                      "build_udh_two_particle_density", "build_dh_eq33_derivative_integrals",
                      "build_udh_non_xc_gradient", "build_udh_gga_complete_xc_ii_gradient",
                      "build_udh_lda_complete_xc_ii_gradient", "scaled_z_residual>1e-10"):
            self.assertIn(token, source)
        header = (ROOT/"src/dft/udh_driver.h").read_text()
        self.assertNotIn("UDHKSResponseOperator response", header)
        self.assertNotIn("getenv", source)
        self.assertNotIn("XC_FAMILY_HYB_MGGA", source)
        self.assertIn("scope.meta_gga=exchange.is_meta_gga_like() || (!combined && correlation.is_meta_gga_like())", source)

    def test_test_only_gate_and_unmodified_normal_cli(self):
        driver = (ROOT/"src/dft/driver.cpp").read_text()
        self.assertIn("options.udh_gradient_validation && functionals.has_double_hybrid_pt2", driver)
        self.assertIn("calculator._calculation==HartreeFock::CalculationType::Gradient", driver)
        # 039bd2df promoted UKS DH gradients; the U11 rejection must stay gone.
        self.assertNotIn("UKS double-hybrid gradients await U11 molecular acceptance", driver)
        self.assertIn("correction->total=*ks+correction->correction", driver)
        self.assertIn("calculator,correction->total,requested_gradient_frame_bohr", driver)
        self.assertNotIn("udh_gradient_validation", (ROOT/"src/dft/main.cpp").read_text())
        self.assertNotIn("udh_gradient_validation", (ROOT/"src/mpi/main.cpp").read_text())

    def test_physical_residual_enters_both_dh_scf_handoffs(self):
        driver = (ROOT/"src/dft/driver.cpp").read_text()
        self.assertIn("std::abs(x_functional.perturbative_correlation_coefficient()) > 1e-14", driver)
        self.assertIn("if (use_diis || requested_level_shift > 0.0 || require_pt2_stationarity)", driver)
        block = driver.split("if (requested_level_shift > 0.0 || require_pt2_stationarity)", 1)[1].split(
            "const double iter_time", 1)[0]
        self.assertIn("metrics.diis_error = diis_error", block)
        self.assertIn("std::max(da.cwiseAbs().maxCoeff(), db.cwiseAbs().maxCoeff())", block)
        self.assertIn("if(canonical_error>1e-9)", driver)
        self.assertIn("iter <= max_iter", driver)

    def test_uks_soscf_is_deferred_only_while_shift_is_active(self):
        driver = (ROOT/"src/dft/driver.cpp").read_text()
        self.assertIn("!sao_active_uks && !prepared.pcm && active_level_shift == 0.0", driver)
        self.assertNotIn("!sao_active_uks && !prepared.pcm && requested_level_shift == 0.0", driver)
        self.assertIn("const bool do_diis_uks = use_diis && !soscf_active;", driver)
        self.assertNotIn("const bool do_diis_uks = use_diis && !soscf_active && !unshifted_polish;", driver)

    def test_uks_uses_coupled_diis_without_relaxing_spin_residual(self):
        driver = (ROOT/"src/dft/driver.cpp").read_text()
        self.assertIn("HartreeFock::UHFDIISState diis", driver)
        self.assertIn("diis.push(fock_alpha_iteration, fock_beta_iteration, error_alpha, error_beta)", driver)
        self.assertNotIn("diis_alpha", driver)
        self.assertNotIn("diis_beta", driver)
        self.assertIn("std::max(error_alpha.norm(), error_beta.norm()) / static_cast<double>(nbasis)", driver)
        shared = (ROOT/"src/base/types.h").read_text()
        self.assertIn("G /= gram_scale", shared)
        self.assertIn("errors[i]->array() / error_scale", shared)

    def test_full_suite_and_unchanged_input_settings(self):
        runner = (ROOT/"tests/udh_molecular_audit.py").read_text()
        for token in ("[1e-4, 2e-4]", "range(len(symbols))", "range(3)", "cross_overlap_reference_current",
                      '"ks_error"', '"pt2_error"', "5e-8", "6-31g", "closed_shell", "rejection",
                      "h2o2_c1_b2plyp_gradient_fd.hfinp"):
            self.assertIn(token, runner)
        self.assertNotIn("max_cycles", runner)
        self.assertNotIn("tol_density", runner)
        self.assertIn("PCM is currently implemented only for single-point energy calculations", runner)
        main = (ROOT/"tests/udh_gradient_main.cpp").read_text()
        self.assertIn("_compute_cross_overlap(ref._shells,c._shells)", main)
        self.assertIn("dump_results_json(c,json)", main)
        self.assertIn("--scale", main)


if __name__ == "__main__":
    unittest.main()
