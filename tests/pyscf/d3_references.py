"""References for the D3 dispersion gates (docs/DISPERSION_D3.md).

Two independent checks per input in tests/inputs/regression/dft/*d3*.hfinp:
  * the D3 term alone, from pyscf-dispersion's own simple-dftd3 build (a
    different version and binary from Planck's), energy and gradient;
  * PySCF's DFT-D3 total (mf.disp), cart=True, Planck's GBS, its CODATA 2018
    Bohr and its unpruned 150 x 590 grid; plus the analytic gradient for
    `calculation gradient` inputs.
Needs pyscf-dispersion (pip) and a shared simple-dftd3 v1.0.0 build
(cmake -DBUILD_SHARED_LIBS=ON -DWITH_API=ON, gfortran):
  DFTD3_LIBRARY=<prefix>/lib/libs-dftd3.dylib \
    tests/pyscf/.venv/bin/python tests/pyscf/d3_references.py
"""
import pathlib
import re

import numpy as np

# pyscf-dispersion 1.0.0 is unusable as shipped on macOS arm64: it calls
# np.ctypeslib.load_library without the loader_path numpy 2 requires, and its
# wheel bundles an x86_64 libs-dftd3. Load a shared simple-dftd3 built
# separately (v1.0.0, the release it wraps) from $DFTD3_LIBRARY instead.
# pyscf.scf.dispersion also imports the D4 wrapper, never called here: stub it.
import ctypes  # noqa: E402
import os  # noqa: E402
from unittest import mock  # noqa: E402

_load = np.ctypeslib.load_library
_OVERRIDE = {"libs-dftd3": lambda: ctypes.CDLL(os.environ["DFTD3_LIBRARY"]),
             "libdftd4": mock.MagicMock}
np.ctypeslib.load_library = lambda name, loader_path=None: (
    _OVERRIDE[name]() if loader_path is None and name in _OVERRIDE else _load(name, loader_path))

from pyscf import dft, gto  # noqa: E402
from pyscf.dispersion import dftd3  # noqa: E402
from pyscf.gto.basis import parse_gaussian  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/dft"
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)
# Planck exchange keyword -> (PySCF xc, simple-dftd3 parameter set). Planck
# derives "pbeh" for PBE0, an alias v1.0.0 lacks; same parameters as "pbe0".
XC = {"b3lyp": ("b3lypg", "b3lyp"), "pbe": ("pbe", "pbe"), "pbe0": ("pbe0", "pbe0"),
      "hyb_gga_xc_b2plyp": ("b2plyp", "b2plyp")}


def field(text, key):
    return re.search(rf"^\s*{key}\s+(\S+)", text, re.M).group(1)


def run(path):
    text = path.read_text()
    basis, exchange = field(text, "basis"), field(text, "exchange")
    variant, calc = field(text, "dispersion"), field(text, "calculation")
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {line.split()[0] for line in coords[2:]}
    # A numeric unit is the Bohr radius in Angstrom: Planck's, not PySCF's.
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=True, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    pyscf_xc, d3_name = XC[exchange]
    version, atm = variant.replace("-atm", ""), variant.endswith("-atm")

    d3 = dftd3.DFTD3Dispersion(mol, xc=d3_name, version=version, atm=atm).get_dispersion(grad=True)
    # The wrapper ignores dftd3 errors: an unknown name reads as exactly zero.
    assert d3["energy"] != 0.0, f"{path.stem}: simple-dftd3 rejected '{d3_name}'"
    print(f"{path.stem}")
    print(f"  D3 alone ({d3_name}, {variant})  E = {d3['energy']:.14f}")
    print("  D3 gradient:", np.array2string(d3["gradient"], precision=12, separator=", "))

    if exchange == "hyb_gga_xc_b2plyp":
        return  # PySCF RKS has no PT2 half; the D3 term above is the gate.
    def make(m):
        mf = (dft.RKS if mult == 1 else dft.UKS)(m)
        mf.xc = pyscf_xc
        mf.disp = variant.replace("-atm", "atm")
        mf.grids.atom_grid = (150, 590)
        mf.grids.prune = None
        mf.init_guess = "1e"
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8  # default sqrt(conv_tol) leaves a 3e-8 gradient gap
        return mf

    mf = make(mol)
    e = mf.kernel()
    assert mf.converged, path.stem
    print(f"  PySCF DFT-D3 total E = {e:.10f}")
    if calc == "gradient":
        g = mf.nuc_grad_method()
        g.grid_response = True
        print("  PySCF gradient:", np.array2string(g.kernel(), precision=10, separator=", "))
    if calc == "geomopt":
        from pyscf.geomopt.geometric_solver import optimize
        mol_eq = optimize(make(mol), convergence_set="gau_tight")
        print(f"  PySCF geometric gau_tight E_opt = {make(mol_eq).kernel():.10f}")


for name in ("water_rks_b3lyp_d3bj_sto3g", "water_cation_uks_b3lyp_d3bj_sto3g",
             "water_dimer_rks_pbe_d3bj_gradient_sto3g", "water_dimer_rks_pbe_d3bj_geomopt_sto3g",
             "benzene_rks_pbe0_d3zero_sto3g",
             "h2o2_rks_b2plyp_d3bj_atm_sto3g"):
    run(CASES / f"{name}.hfinp")
