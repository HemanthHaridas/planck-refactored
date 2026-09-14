"""Fast U0 fixture and audit-parser checks; no compiler or Planck process."""
import math
import unittest

from dh_channel_fd_audit import parse_geometry, render_gradient_input
from udh_u0_molecular_audit import FIXTURES, check_energy, check_rejection


class U0Fixtures(unittest.TestCase):
    def test_spin_dimensions_and_settings(self):
        expected = {
            "water_cation_asymmetric": (5, 4, 7),
            "ho2_asymmetric": (9, 8, 11),
            "water_triplet_asymmetric": (6, 4, 7),
            "h2plus_zero_beta": (1, 0, 2),
            "h2o2_cation_c1": (9, 8, 12),
        }
        fixtures = list(FIXTURES.glob("*.hfinp"))
        self.assertEqual(len(fixtures), len(expected))
        for path in fixtures:
            text = path.read_text()
            symbols, xyz, charge, mult = parse_geometry(text)
            electrons = sum({"O": 8, "H": 1}[s] for s in symbols) - charge
            nao = sum({"O": 5, "H": 1}[s] for s in symbols)
            key = path.stem.removesuffix("_b2plyp_sto3g")
            self.assertEqual((electrons + mult - 1) % 2, 0)
            self.assertEqual(((electrons + mult - 1)//2, (electrons - mult + 1)//2, nao), expected[key])
            for value in ("scf_type    uhf", "calculation energy", "grid                ultrafine",
                          "exchange            b2plyp", "use_symm    .false.", "tol_density 1.0e-11"):
                self.assertIn(value, text)
            self.assertTrue(all(math.isfinite(v) for row in xyz for v in row))
            gradient = render_gradient_input(text, symbols, xyz, charge, mult)
            self.assertIn("calculation gradient", gradient)
            self.assertEqual(parse_geometry(gradient), (symbols, xyz, charge, mult))
            if key == "h2o2_cation_c1":
                # This fixture's contract is that the downstream post-KS (PT2)
                # step receives the UNSHIFTED Fock and density -- not that any
                # particular convergence aid is used to get there. Two routes
                # satisfy it, and exactly one must be configured:
                #
                #   level_shift: shifts a COPY (fock_alpha_iteration) for the
                #     DIIS/diagonalization step only, then removes the shift and
                #     reconverges the physical Fock before handing off. The
                #     marker sequence check in udh_u0_molecular_audit.check_energy
                #     enforces that removal actually happened, in order.
                #
                #   SOSCF: unshifted BY CONSTRUCTION -- driver.cpp gates SOSCF on
                #     `requested_level_shift == 0.0`, so no shift matrix is ever
                #     built and the stored fock/density are physical throughout.
                #     Nothing to remove, hence no markers to check.
                #
                # They are mutually exclusive in the driver (a shifted iteration
                # must not be mixed with the unshifted SOSCF Hessian), so assert
                # exactly one is present rather than pinning the level_shift one.
                shifted = "level_shift 0.3" in text
                soscf = ("scf_soscf_cycles" in text or "scf_soscf_start" in text
                         or "scf_soscf_auto_stall" in text)
                self.assertNotEqual(
                    shifted, soscf,
                    "h2o2_cation_c1 must use exactly one unshifted-handoff route: "
                    "a level shift that is removed before post-KS, or SOSCF "
                    "(unshifted by construction). They cannot be combined.")

    def test_actual_c1_geometry(self):
        path = FIXTURES / "h2o2_cation_c1_b2plyp_sto3g.hfinp"
        _, xyz, _, _ = parse_geometry(path.read_text())
        v = [[xyz[i][q]-xyz[0][q] for q in range(3)] for i in (1, 2, 3)]
        triple = sum(v[0][q]*(v[1][(q+1)%3]*v[2][(q+2)%3]-v[1][(q+2)%3]*v[2][(q+1)%3]) for q in range(3))
        self.assertGreater(abs(triple), 0.1)
        def distance(i, j):
            return math.dist(xyz[i], xyz[j])
        # Noncoplanar + no nonidentity species-preserving distance automorphism.
        for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)):
            self.assertGreater(max(abs(distance(i,j)-distance(perm[i],perm[j]))
                                   for i in range(4) for j in range(i)), 1e-3)

    def test_energy_and_rejection_parser(self):
        data = dict(total_energy=-10, has_correlation=True)
        log = ("Converged : true\nPT2 coefficient = 0.270000; bare MP2-like correction = -0.1000000000 Eh; "
               "scaled contribution = -0.0270000000 Eh")
        self.assertEqual(check_energy(log, data)["scaled"], -0.027)
        with self.assertRaises(ValueError):
            check_energy(log, data, zero_beta=True)
        for corrupt in (log.replace("0.027", "0.054"), log.replace("true", "false"), ""):
            with self.assertRaises(ValueError):
                check_energy(corrupt, data)
        message = "Gradient currently supports only single-point energies for range-separated and double-hybrid functionals"
        check_rejection(1, message)
        for code, output in ((0, message), (1, "unrelated SCF failure")):
            with self.assertRaises(ValueError):
                check_rejection(code, output)

    def test_level_shift_requires_unshifted_handoff_before_pt2(self):
        data = dict(total_energy=-10, has_correlation=True)
        energy = ("Converged : true\nPT2 coefficient = 0.270000; bare MP2-like correction = -0.1000000000 Eh; "
                  "scaled contribution = -0.0270000000 Eh")
        markers = ["Applying 0.300000 Eh to the iteration virtual spaces; physical energy is unchanged\n",
                   "Shift removed; reconverging physical Fock before post-KS handoff\n",
                   "Unshifted canonical orbitals verified for post-KS handoff\n"]
        self.assertEqual(check_energy("".join(markers) + energy, data, level_shift=0.3)["scf_level_shift"], 0.3)
        for output in (energy, markers[0] + energy, "".join(reversed(markers)) + energy,
                       energy + "".join(markers), "".join(markers).replace("0.300000", "0.100000") + energy):
            with self.assertRaises(ValueError):
                check_energy(output, data, level_shift=0.3)


if __name__ == "__main__":
    unittest.main()
