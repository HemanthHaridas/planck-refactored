#!/usr/bin/env python3
"""Gate: Planck's radial grid is identical to PySCF's.

Planck's M4 quadrature formula already matched PySCF to ~1e-15, but two inputs
to it did not:

  1. the M4 scaling parameter. Planck used a Bragg-Slater-like RADIUS table
     where PySCF uses the Treutler-Ahlrichs XI parameters -- different
     quantities, differing for 28 of the first 36 elements. H, C and O happen
     to coincide, which is why water test cases never exposed it.
  2. the radial point count. Planck used an ORCA-style heuristic
     `(15*int_acc - 40) + radial_row_factor*row` with a fixed row factor of 5,
     giving 44 points for a second-row atom where PySCF's RAD_GRIDS lookup
     gives 75 at its default level -- the cause of Planck's grid failing to
     converge (2.9e-5 Eh still moving at "ultrafine" vs PySCF's 1.4e-7).

The Becke partition's atomic-size adjustment is also checked: it consumed the
same table, while PySCF uses BRAGG radii and a different formula.

Requires the pyscf venv at tests/pyscf/.venv. Skips cleanly without it.
"""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "tests/pyscf/.venv/bin/python3"
RADIAL_H = ROOT / "src/dft/base/radial.h"
EIGEN = ROOT / "build/_deps/eigen-src"

# Spans every period, including elements whose xi and radius differ most.
ELEMENTS = [1, 2, 3, 6, 7, 8, 9, 11, 14, 16, 17, 20, 26, 30, 35, 36, 47, 53, 79]
LEVELS = [1, 3, 5, 7]          # Coarse / Normal / Fine / UltraFine
PAIRS = [1, 6, 7, 8, 11, 16, 17, 26, 35]

DUMPER = r'''
#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>
#include "dft/base/radial.h"
int main(int argc, char **argv) {
    if (argv[1][0] == 'r') {
        int Z = atoi(argv[2]), lvl = atoi(argv[3]);
        int n = DFT::pyscf_radial_count(Z, lvl);
        auto g = DFT::MakeTreutlerAhlrichsGrid(n, DFT::treutler_xi(Z));
        std::printf("%d %.17g\n", n, DFT::treutler_xi(Z));
        for (int i = 0; i < n; ++i)
            std::printf("%.17g %.17g\n", g(i, 0), g(i, 1));
    } else {
        int Zi = atoi(argv[2]), Zj = atoi(argv[3]);
        const double ri = std::sqrt(DFT::bragg_radius(Zi));
        const double rj = std::sqrt(DFT::bragg_radius(Zj));
        double a = 0.25 * (rj / ri - ri / rj);
        std::printf("%.17g\n", a < -0.5 ? -0.5 : (a > 0.5 ? 0.5 : a));
    }
}
'''


def _build(tmp):
    src = Path(tmp) / "dump.cpp"
    src.write_text(DUMPER)
    exe = Path(tmp) / "dump"
    for cxx in ("g++-15", "g++", "c++"):
        r = subprocess.run([cxx, "-std=c++23", f"-I{ROOT/'src'}", f"-I{EIGEN}",
                            str(src), "-o", str(exe)], capture_output=True, text=True)
        if r.returncode == 0:
            return exe
    raise unittest.SkipTest(f"could not compile the dumper: {r.stderr[-400:]}")


@unittest.skipUnless(VENV.exists(), "pyscf venv not present")
@unittest.skipUnless(EIGEN.exists(), "eigen sources not present (configure the build first)")
class TestRadialGridMatchesPyscf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        cls.exe = _build(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _pyscf(self, script):
        r = subprocess.run([str(VENV), "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return r.stdout

    def test_radial_points_weights_and_counts(self):
        script = f'''
import json
from pyscf.dft import radi, gen_grid
out = {{}}
for Z in {ELEMENTS}:
    for lvl in {LEVELS}:
        n = int(gen_grid._default_rad(Z, lvl))
        r, dr = radi.treutler_ahlrichs(n, Z)
        out[f"{{Z}}_{{lvl}}"] = [n, radi._treutler_ahlrichs_xi[Z],
                                 list(r), list((dr * r**2))]
print(json.dumps(out))
'''
        import json
        ref = json.loads(self._pyscf(script))
        worst_r = worst_w = 0.0
        for Z in ELEMENTS:
            for lvl in LEVELS:
                n_p, xi_p, r_p, w_p = ref[f"{Z}_{lvl}"]
                out = subprocess.run([str(self.exe), "r", str(Z), str(lvl)],
                                     capture_output=True, text=True).stdout.split("\n")
                n_c, xi_c = out[0].split()
                self.assertEqual(int(n_c), n_p, f"radial count differs for Z={Z} level={lvl}")
                self.assertAlmostEqual(float(xi_c), xi_p, places=12, msg=f"xi differs for Z={Z}")
                rows = [[float(v) for v in l.split()] for l in out[1:n_p + 1]]
                # Planck emits descending r; PySCF ascending.
                r_c = [row[0] for row in rows][::-1]
                w_c = [row[1] for row in rows][::-1]
                for a, b in zip(r_c, r_p):
                    worst_r = max(worst_r, abs(a - b) / abs(b))
                for a, b in zip(w_c, w_p):
                    worst_w = max(worst_w, abs(a - b) / abs(b))
        self.assertLess(worst_r, 1e-10, f"radial positions differ by {worst_r:.3e}")
        self.assertLess(worst_w, 1e-10, f"radial weights differ by {worst_w:.3e}")

    def test_becke_atomic_size_adjustment(self):
        script = f'''
import json, numpy
from pyscf.dft import radi
b = numpy.array(radi.BRAGG_RADII)
out = {{}}
for Zi in {PAIRS}:
    for Zj in {PAIRS}:
        rad = numpy.sqrt(b[[Zi, Zj]])
        rr = rad.reshape(-1, 1) * (1. / rad)
        a = numpy.clip(.25 * (rr.T - rr), -.5, .5)
        out[f"{{Zi}}_{{Zj}}"] = float(a[0, 1])
print(json.dumps(out))
'''
        import json
        ref = json.loads(self._pyscf(script))
        worst = 0.0
        for Zi in PAIRS:
            for Zj in PAIRS:
                got = float(subprocess.run([str(self.exe), "b", str(Zi), str(Zj)],
                                           capture_output=True, text=True).stdout)
                worst = max(worst, abs(got - ref[f"{Zi}_{Zj}"]))
        self.assertLess(worst, 1e-10, f"Becke size adjustment differs by {worst:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
