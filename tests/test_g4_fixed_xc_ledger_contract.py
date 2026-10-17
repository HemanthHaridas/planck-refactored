"""G4.2 must keep stationary-density grid attribution test-only and explicit."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class G4FixedXCLedgerContractTests(unittest.TestCase):
    def test_test_frontend_owns_the_stationary_ledger(self):
        source = (ROOT / 'tests' / 'udh_gradient_main.cpp').read_text()
        self.assertIn('"--fixed-xc-ledger"', source)
        self.assertIn('"--fixed-xc-reference-ladder"', source)
        self.assertIn('"--fixed-xc-shell-atlas"', source)
        self.assertIn('"--fixed-xc-shell-residuals"', source)
        self.assertIn('g4-fixed-stationary-density-v1', source)
        self.assertIn('g4-fixed-xc-reference-ladder-v1', source)
        self.assertIn('initialize_boys_library', source)
        self.assertIn('Boys loader initialization failed', source)
        self.assertIn('c._info._scf.alpha.density', source)
        self.assertIn('c._info._scf.beta.density', source)
        self.assertIn('const std::array<std::pair<M,M>,2> directions', source)
        self.assertIn('"uniform-1454-r1"', source)
        self.assertIn('"uniform-2030-r2"', source)
        self.assertIn('D_fxc_D', source)
        self.assertIn('\\"owners\\"', source)
        self.assertIn('owner_ranges', source)
        self.assertIn('band_diagnostic', source)
        self.assertIn('first_shell', source)
        self.assertIn('angular_points', source)
        self.assertIn('diagnostic_angular_points', source)

    def test_production_cli_does_not_expose_the_test_option(self):
        self.assertNotIn('--fixed-xc-ledger', (ROOT / 'src' / 'io' / 'io.cpp').read_text())

    def test_matrix_runner_requires_both_independent_directions(self):
        source = (ROOT / 'tests' / 'g4_fixed_xc_ledger_audit.py').read_text()
        self.assertIn("'uniform-1454-r1'", source)
        self.assertIn("'uniform-2030-r2'", source)
        self.assertIn("len(row['directions']) == 2", source)
        self.assertIn('D_fxc_D', source)
        self.assertIn('R2_CONTRACTION_LIMIT', source)
        self.assertIn('band_deltas', source)
        self.assertIn('pyscf_r2_band_deltas', source)
        self.assertIn("'--report-path'", source)
        self.assertIn('args.report_path.write_text', source)

    def test_reference_ladder_remains_test_only(self):
        source = (ROOT / 'tests' / 'udh_gradient_main.cpp').read_text()
        self.assertIn('uniform-2030-a', source)
        self.assertIn('uniform-2030-b', source)
        self.assertIn('uniform-2702-c', source)
        self.assertIn('g4-fixed-xc-shell-layers-v1', source)
        self.assertIn('write_fixed_xc_shell_layer(out,c,nalpha,nbeta,2030)', source)
        self.assertIn('grid.points.col(3).segment(lo,angular_points)', source)
        self.assertNotIn('--fixed-xc-reference-ladder',
                         (ROOT / 'src' / 'io' / 'io.cpp').read_text())


if __name__ == '__main__':
    unittest.main()
