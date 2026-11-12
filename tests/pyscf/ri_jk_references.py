"""References for the RI-JK SCF gates (docs/RI_JK_ROUTINE_SCOPE.md step 0).

PySCF density_fit on each tests/inputs/regression/ri/*_rijk_* input: the
orbital basis and the auxiliary basis (mp2_ri_basis) both read from Planck's
GBS files, Cartesian, Planck's CODATA 2018 Bohr. Also prints the exact
(non-fitted) energy, so the fitting error is on record.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_jk_references.py [AUX_BASIS]
AUX_BASIS overrides each input's mp2_ri_basis (e.g. def2-universal-JKFIT).
"""
import pathlib
import re
import sys

from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)


def field(text, key):
    m = re.search(rf"^\s*{key}\s+(\S+)", text, re.M)
    return m.group(1) if m else None


def run(path):
    text = path.read_text()
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    elements = {line.split()[0] for line in coords[2:]}
    gbs = str(ROOT / "basis-sets" / field(text, "basis"))
    aux = str(ROOT / "basis-sets" / (sys.argv[1] if len(sys.argv) > 1 else field(text, "mp2_ri_basis")))
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=True, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    energies = []
    for fitted in (True, False):
        mf = (scf.RHF if field(text, "scf_type") == "rhf" else scf.UHF)(mol)
        if fitted:
            mf = mf.density_fit(auxbasis={e: parse_gaussian.load(aux, e) for e in elements})
            assert mf.with_df.auxmol is None or mf.with_df.auxmol.cart
        mf.init_guess = "1e"  # Planck guess hcore
        mf.conv_tol = 1e-12
        mf.kernel()
        # From 1e, H2O2+ UHF lands on a saddle 0.023 Eh up; Planck's hcore
        # run reaches the stable minimum. Follow internal instabilities so
        # both codes are compared at the minimum.
        if isinstance(mf, scf.uhf.UHF):
            for _ in range(5):
                mo, _, stable, _ = mf.stability(return_status=True)
                if stable:
                    break
                mf.kernel(mf.make_rdm1(mo, mf.mo_occ))
            assert stable, path.stem
        energies.append(mf.e_tot)
        assert mf.converged, path.stem
    print(f"{path.stem:34s} E_RI = {energies[0]:.10f}  E_exact = {energies[1]:.10f}  "
          f"fit error = {energies[0] - energies[1]:.3e}")


for p in sorted((ROOT / "tests/inputs/regression/ri").glob("*_rijk_*.hfinp")):
    run(p)
