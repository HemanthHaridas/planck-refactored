"""W3.2: `dressing="derived"` routes the factorizer through production.

The derivation route had an emit bridge and a value gate but **no production
caller** -- it was deferred in its own commit and never revisited. This wires it
into `print_cpp_planck` alongside the recognition route, on ONE dressing axis.

The load-bearing property is that going through production changes nothing:
`print_cpp_planck(dressing="derived")` must equal what the standalone bridge
`emit_factorized_from_equations` already produced. If those ever diverge, the
production path has grown behaviour the value gate never covered.

The two routes deliberately run at DIFFERENT points in the composition:
`recognized` dresses before spin-adaptation (its hand-seeded specs declare GCC
layouts, which `adapt_intermediate_spec` must then transform), while `derived`
factorizes after (it derives operators FROM whatever manifold reaches it, so its
specs are already in the adapted layout). Same emit path, different source.
"""
import unittest


class DressingDerivedTests(unittest.TestCase):
    def test_production_matches_the_standalone_bridge(self):
        """The whole point of W3.2: a caller, with no behaviour change."""
        from ccgen.generate import generate_cc_equations, print_cpp_planck
        from ccgen.optimization.factorize import emit_factorized_from_equations

        via_production = print_cpp_planck("ccsd", dressing="derived")
        eqs = generate_cc_equations("ccsd", engine="diagram", canonical_fock=True)
        via_bridge = emit_factorized_from_equations("ccsd", eqs)
        self.assertEqual(via_production, via_bridge)

    def test_none_is_the_undressed_default(self):
        from ccgen.generate import print_cpp_planck
        self.assertEqual(print_cpp_planck("ccsd"),
                         print_cpp_planck("ccsd", dressing="none"))

    def test_legacy_boolean_is_recognized(self):
        from ccgen.generate import print_cpp_planck
        self.assertEqual(print_cpp_planck("ccsd", dress_operators=True),
                         print_cpp_planck("ccsd", dressing="recognized"))

    def test_the_three_routes_are_distinct(self):
        """Guards against a route being accepted and silently doing nothing."""
        from ccgen.generate import print_cpp_planck
        tus = {d: print_cpp_planck("ccsd", dressing=d)
               for d in ("none", "recognized", "derived")}
        self.assertEqual(len(set(tus.values())), 3, "two routes emit the same TU")

    def test_derived_composes_with_spin_adapt(self):
        """Production emits spatial kernels; derived must survive adaptation."""
        from ccgen.generate import print_cpp_planck
        tu = print_cpp_planck("ccsd", dressing="derived", spin_adapt=True)
        self.assertIn("build_", tu)

    def test_interaction_points_match_recognized(self):
        from ccgen.generate import print_cpp_planck

        with self.assertRaises(ValueError):
            print_cpp_planck("ccsd", dressing="derived", factorize_tau=True)
        # CSE forced off, and engine/canonical_fock overridden, exactly as for
        # `recognized` -- these are properties of DRESSING, not of the route.
        self.assertEqual(
            print_cpp_planck("ccsd", dressing="derived", include_intermediates=True),
            print_cpp_planck("ccsd", dressing="derived"))
        self.assertEqual(
            print_cpp_planck("ccsd", dressing="derived", engine="wick",
                             canonical_fock=False),
            print_cpp_planck("ccsd", dressing="derived"))

    def test_axis_is_validated(self):
        from ccgen.generate import print_cpp_planck
        with self.assertRaises(ValueError):
            print_cpp_planck("ccsd", dressing="bogus")
        with self.assertRaises(ValueError):
            print_cpp_planck("ccsd", dressing="derived", dress_operators=True)


if __name__ == "__main__":
    unittest.main()
