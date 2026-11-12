"""References for the RI-JK SCF gates (docs/RI_JK_ROUTINE_SCOPE.md).

PySCF density fitting on each tests/inputs/regression/ri/*_rijk_* input: the
orbital basis and the auxiliary basis (mp2_ri_basis) both read from Planck's
GBS files, Planck's CODATA 2018 Bohr. Also prints the exact (non-fitted)
energy, so the fitting error is on record.

The fitted auxiliary functions follow basis_type in both codes (spherical
aux for a spherical orbital basis; docs/RI_SPHERICAL_AUX.md), so plain
density_fit is the reference for every input. ECPs (def2, Rb onward) are
read from the same GBS file with def2_ecp_references.load_ecp.
  tests/pyscf/.venv/bin/python tests/pyscf/ri_jk_references.py [AUX_BASIS]
AUX_BASIS overrides each input's mp2_ri_basis (e.g. def2-universal-JKFIT).
"""
import pathlib
import re
import sys

from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from def2_ecp_references import load_ecp  # noqa: E402

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
    cart = field(text, "basis_type") != "spherical"
    ecp = {e: load_ecp(gbs, e) for e in elements}
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=cart, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements},
                ecp={e: v for e, v in ecp.items() if v})
    energies = []
    for fitted in (True, False):
        mf = {"rhf": scf.RHF, "uhf": scf.UHF, "rohf": scf.ROHF}[field(text, "scf_type")](mol)
        aux_basis = {e: parse_gaussian.load(aux, e) for e in elements}
        if fitted:
            mf = mf.density_fit(auxbasis=aux_basis)
            assert mf.with_df.auxmol is None or mf.with_df.auxmol.cart == cart
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


if __name__ == "__main__":
    for p in sorted((ROOT / "tests/inputs/regression/ri").glob("*_rijk_*.hfinp")):
        run(p)
