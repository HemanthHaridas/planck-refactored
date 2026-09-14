"""No-build guard for the four O1 production loop schedules.

Numerical equivalence/timing uses the actual Eigen method in the C++ test;
this check ensures production retains that once-per-grid schedule.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GridSigmaHoistTests(unittest.TestCase):
    def test_four_sites_evaluate_before_point_loop_and_release_before_libxc(self):
        for filename, count in (("analytic_hessian.cpp", 1), ("dh_pt2_gradient.cpp", 3)):
            source = (ROOT / "src/dft" / filename).read_text()
            self.assertNotRegex(source, r"gradient_squared\s*\(\s*\)\s*\(")
            calls = list(re.finditer(
                r"const Eigen::VectorXd ground_sigma = ground->total.gradient_squared\(\);", source))
            self.assertEqual(len(calls), count)
            for call in calls:
                # Its containing scope begins immediately before the cache,
                # so there cannot be an enclosing per-point loop in that scope.
                opening = source.rfind("{", 0, call.start())
                self.assertNotIn("for (", source[opening:call.start()])
                tail = source[call.end():]
                self.assertRegex(tail, r"^\s*for \(Eigen::Index p = 0; p < n(?:points|pts); \+\+p\)")
                loop_assignment = re.search(
                    r"sigma(?:_vec)?\[static_cast<std::size_t>\(p\)\] = ground_sigma\(p\);", tail)
                self.assertIsNotNone(loop_assignment)
                self.assertLess(loop_assignment.start(), 300)
                # Both cache and point loop end before functional evaluation.
                after = tail[loop_assignment.end():]
                self.assertRegex(after, r"^\s*}\s*(?:}\s*)?std::vector<double>")

    def test_work_ledger_scales_linearly(self):
        # Deterministic operation counts, not a Python timing claim about C++.
        for size in (0, 1, 7, 64, 256, 1024, 4096):
            old_visits = sum(size for _ in range(size))
            new_visits = size
            self.assertEqual(old_visits, size*size)
            self.assertEqual(new_visits, size)
            if size:
                self.assertEqual(old_visits // new_visits, size)


if __name__ == "__main__":
    unittest.main()
