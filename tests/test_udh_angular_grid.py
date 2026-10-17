"""No-build angular-control and grid-orientation audit contracts."""
from pathlib import Path
import re
import unittest

from udh_molecular_audit import (FIXTURES, matmul, maxdiff, transpose,
                                 with_angular_points, with_radial_points, with_grid_policy)
from udh_grid_orientation_audit import rotations

ROOT = Path(__file__).resolve().parents[1]


class AngularControlTests(unittest.TestCase):
    def test_input_override_changes_only_dft_angular_control(self):
        original = (FIXTURES/'h2o2_cation_c1_b2plyp_sto3g.hfinp').read_text()
        changed = with_angular_points(original, 974)
        strip = lambda x: re.sub(r'^    grid_angular_points \d+\n', '', x, flags=re.M)
        self.assertEqual(strip(changed), original)
        self.assertIn('grid                ultrafine', changed)
        self.assertIn('max_cycles  200', changed)
        self.assertEqual(with_angular_points(original, None), original)
        replaced = with_angular_points(changed, 1454)
        self.assertEqual(replaced.count('grid_angular_points'), 1)
        self.assertIn('grid_angular_points 1454', replaced)
        for n in (-1, 1, 5811):
            with self.assertRaises(ValueError):
                with_angular_points(original, n)
        with self.assertRaises(ValueError):
            with_angular_points('no dft section', 974)

    def test_radial_override_changes_only_dft_radial_control(self):
        original = (FIXTURES/'h2o2_cation_c1_b2plyp_sto3g.hfinp').read_text()
        changed = with_radial_points(original, 120)
        strip = lambda x: re.sub(r'^    grid_radial_points \d+\n', '', x, flags=re.M)
        self.assertEqual(strip(changed), original)
        self.assertEqual(with_radial_points(changed, 96).count('grid_radial_points'), 1)
        self.assertIn('grid_radial_points 96', with_radial_points(changed, 96))
        self.assertEqual(with_radial_points(original, None), original)
        for n in (-1, 1.5):
            with self.assertRaises(ValueError):
                with_radial_points(original, n)

    def test_grid_policy_changes_only_dft_policy_control(self):
        original = (FIXTURES/'h2o2_cation_c1_b2plyp_sto3g.hfinp').read_text()
        changed = with_grid_policy(original, 'pyscf-nwchem-2.13')
        strip = lambda x: re.sub(r'^    grid_policy [^\n]+\n', '', x, flags=re.M)
        self.assertEqual(strip(changed), original)
        self.assertEqual(with_grid_policy(changed, 'legacy-unpruned-max').count('grid_policy'), 1)
        with self.assertRaises(ValueError):
            with_grid_policy(original, 'unknown')

    def test_two_proper_rotations(self):
        identity = [[float(i == j) for j in range(3)] for i in range(3)]
        for r in rotations():
            self.assertLess(maxdiff(matmul(r, transpose(r)), identity), 1e-14)
            determinant = sum(r[0][i]*(r[1][(i+1)%3]*r[2][(i+2)%3]-r[1][(i+2)%3]*r[2][(i+1)%3]) for i in range(3))
            self.assertAlmostEqual(determinant, 1.)

    def test_every_driver_grid_rebuild_uses_resolved_spec(self):
        source = (ROOT/'src/dft/driver.cpp').read_text()
        self.assertEqual(source.count('auto grid_spec = resolve_calculator_grid_spec(calculator);'), 4)
        self.assertEqual(source.count('auto molecular_grid = MakeMolecularGrid(calculator._molecule, *grid_spec);'), 4)
        self.assertIn('prepared.grid_spec = *grid_spec;', source)
        self.assertIn('prepared.grid_spec.fingerprint()', source)
        header = (ROOT/'src/base/types.h').read_text()
        self.assertIn('int _grid_angular_points = 0', header)
        self.assertIn('int _grid_radial_points = 0', header)
        parser = (ROOT/'src/io/io.cpp').read_text()
        self.assertIn('consumed != value.size()', parser)
        self.assertIn('dft._grid_angular_points = points', parser)
        self.assertIn('dft._grid_radial_points = points', parser)
        self.assertIn('dft._grid_policy = value', parser)

    def test_unpruned_path_keeps_radial_and_partition_rules(self):
        source = (ROOT/'src/dft/base/grid.h').read_text()
        self.assertIn('struct GridRequest', source)
        self.assertIn('struct ResolvedGridSpec', source)
        self.assertIn('legacy-orca-index-v1', source)
        self.assertIn('spec.request.angular_points', source)
        self.assertIn('spec.radial_override', source)
        self.assertIn('const auto nr = radial_point_count(Z, spec)', source)
        self.assertIn('detail::becke_partition_weight(', source)
        audit = (ROOT/'tests/udh_grid_orientation_audit.py').read_text()
        self.assertIn('LIMIT = 2e-6', audit)
        self.assertIn("'--joint-levels'", audit)
        self.assertIn("'--reference-levels'", audit)
        self.assertIn('reference_energy_tolerance', audit)
        self.assertIn('reference_gradient_tolerance', audit)
        self.assertIn("'reference_stable'", audit)
        self.assertIn("RADIAL:ANGULAR", audit)
        self.assertIn("with_radial_points(with_angular_points", audit)
        audit_helpers = (ROOT/'tests/udh_molecular_audit.py').read_text()
        self.assertIn("pyscf-nwchem-2.13", audit_helpers)
        self.assertIn("with_grid_policy", audit_helpers)
        self.assertNotIn('max_cycles', audit)
        self.assertNotIn('tol_density', audit)


if __name__ == '__main__':
    unittest.main()
