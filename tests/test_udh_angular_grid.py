"""No-build angular-control and grid-orientation audit contracts."""
from pathlib import Path
import re
import unittest

from udh_molecular_audit import FIXTURES, matmul, maxdiff, transpose, with_angular_points
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

    def test_two_proper_rotations(self):
        identity = [[float(i == j) for j in range(3)] for i in range(3)]
        for r in rotations():
            self.assertLess(maxdiff(matmul(r, transpose(r)), identity), 1e-14)
            determinant = sum(r[0][i]*(r[1][(i+1)%3]*r[2][(i+2)%3]-r[1][(i+2)%3]*r[2][(i+1)%3]) for i in range(3))
            self.assertAlmostEqual(determinant, 1.)

    def test_every_driver_grid_rebuild_uses_override(self):
        source = (ROOT/'src/dft/driver.cpp').read_text()
        calls = source.split('auto molecular_grid = MakeMolecularGrid(')[1:]
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertIn('calculator._dft._grid_angular_points', call.split(';', 1)[0])
        header = (ROOT/'src/base/types.h').read_text()
        self.assertIn('int _grid_angular_points = 0', header)
        parser = (ROOT/'src/io/io.cpp').read_text()
        self.assertIn('consumed != value.size()', parser)
        self.assertIn('dft._grid_angular_points = points', parser)

    def test_unpruned_path_keeps_radial_and_partition_rules(self):
        source = (ROOT/'src/dft/base/grid.h').read_text()
        block = source.split('angular_shell_sizes(', 1)[1].split('auto scheme', 1)[0]
        self.assertIn('LEBEDEV_ORDER', block)
        self.assertEqual(block.count('angular_points,'), 4)
        self.assertIn('const auto nr = radial_point_count(Z, level)', source)
        self.assertIn('detail::becke_partition_weight(', source)
        audit = (ROOT/'tests/udh_grid_orientation_audit.py').read_text()
        self.assertIn('LIMIT = 2e-6', audit)
        self.assertNotIn('max_cycles', audit)
        self.assertNotIn('tol_density', audit)


if __name__ == '__main__':
    unittest.main()
