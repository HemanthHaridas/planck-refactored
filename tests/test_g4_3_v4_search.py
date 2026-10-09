#!/usr/bin/env python3
import importlib.util
import unittest


SPEC = importlib.util.spec_from_file_location('search', __file__.replace('test_', ''))
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)


class G43V4SearchTests(unittest.TestCase):
    def test_normalized_error_rejects_one_over_limit_channel_without_cancellation(self):
        request = {'limits': {'xc_energy_ha': 1e-8, 'xc_response_contraction': 1e-7}}
        rows = [dict(xc_energy=2e-9, directions=[dict(D_V=2e-8, D_fxc_D=1.1e-7),
                                                 dict(D_V=0., D_fxc_D=0.)])]
        self.assertAlmostEqual(search.normalized_error(rows, request), 1.1)
        self.assertFalse(search._within(rows, request))

    def test_robust_ratio_is_stricter_than_physical_limit(self):
        request = {'limits': {'xc_energy_ha': 1e-8, 'xc_response_contraction': 1e-7}}
        rows = [dict(xc_energy=6e-9, directions=[dict(D_V=0., D_fxc_D=0.),
                                                 dict(D_V=0., D_fxc_D=0.)])]
        self.assertTrue(search._within(rows, request))
        self.assertGreater(search.normalized_error(rows, request), search.ROBUST_RATIO)


if __name__ == '__main__':
    unittest.main()
