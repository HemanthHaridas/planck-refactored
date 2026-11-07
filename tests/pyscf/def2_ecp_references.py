"""PySCF references for the def2 ECP gates (docs/DEF2_ECP.md step 4).

Basis and ECP both come from Planck's own GBS file. PySCF's Gaussian94 reader
skips ECP blocks, so `load_ecp` converts the block to PySCF's internal format.

    python def2_ecp_references.py            # spherical RHF energies
    python def2_ecp_references.py DUMP_DIR CASE
        # element-by-element check of Planck's Cartesian V_ne + U_ECP, dumped
        # with PLANCK_DUMP_AO_INTEGRALS=DUMP_DIR, against int1e_nuc + ECPscalar.
"""
import pathlib
import re
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/inputs/regression/hf"
# Planck uses CODATA 2018 (src/base/tables.h); PySCF defaults to 2010.
BOHR_ANGSTROM = 0.529177210903


def load_ecp(gbs, element):
    """[n_core, [[l, by_power], ...]] with l = -1 for the local channel."""
    lines = pathlib.Path(gbs).read_text().splitlines()
    head = re.compile(rf"^{element.upper()}-ECP\s+(\d+)\s+(\d+)")
    start = next((i for i, s in enumerate(lines) if head.match(s.strip().upper())), None)
    if start is None:
        return None
    L, n_core = map(int, head.match(lines[start].strip().upper()).groups())
    channels, i = [], start + 1
    for c in range(L + 1):
        count = int(lines[i + 1])
        by_power = [[] for _ in range(7)]
        for s in lines[i + 2:i + 2 + count]:
            n, a, d = s.split()
            by_power[int(n)].append([float(a), float(d)])
        channels.append([-1 if c == 0 else c - 1, by_power])
        i += 2 + count
    return [n_core, channels]


def read_case(path):
    text = path.read_text()
    basis = re.search(r"^\s*basis\s+(\S+)", text, re.M).group(1)
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].split("\n")
    charge, mult = map(int, coords[2].split())
    atoms = [line for line in coords[3:] if line.strip()]
    return basis, "\n".join(atoms), charge, mult


def build(case, cart):
    basis, atoms, charge, mult = read_case(case)
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {line.split()[0] for line in atoms.splitlines()}
    ecp = {e: load_ecp(gbs, e) for e in elements}
    return gto.M(atom=atoms, unit=BOHR_ANGSTROM, cart=cart, verbose=0,
                 charge=charge, spin=mult - 1,
                 basis={e: parse_gaussian.load(gbs, e) for e in elements},
                 ecp={e: v for e, v in ecp.items() if v})


def read_dump(path):
    rows, cols, *vals = path.read_text().split()
    return np.array(vals, float).reshape(int(rows), int(cols))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--fcidump":
        # The FCIDUMP describes the valence-only Hamiltonian: its reference
        # determinant, and an RHF solved on it, must both give PySCF's ECP energy.
        from pyscf.tools import fcidump
        path = pathlib.Path(sys.argv[2])
        case = CASES / f"{path.stem}.hfinp"
        mf_ref = scf.RHF(build(case, cart=False))
        mf_ref.conv_tol = 1e-12
        e_ref = mf_ref.kernel()
        d = fcidump.read(str(path))
        h, g, n = d["H1"], d["H2"], d["NELEC"] // 2
        from pyscf import ao2mo
        g = ao2mo.restore(1, g, d["NORB"])
        e_det = (d["ECORE"] + 2 * np.trace(h[:n, :n])
                 + 2 * np.einsum("iijj", g[:n, :n, :n, :n]) - np.einsum("ijji", g[:n, :n, :n, :n]))
        mf = fcidump.to_scf(str(path))
        mf.conv_tol = 1e-12
        e_scf = mf.kernel()
        print(f"NELEC {d['NELEC']}  PySCF ECP RHF {e_ref:.10f}  HOMO {mf_ref.mo_energy[n - 1]:.6f}")
        print(f"FCIDUMP reference determinant {e_det:.10f}  (diff {e_det - e_ref:+.1e})")
        print(f"FCIDUMP RHF                   {e_scf:.10f}  (diff {e_scf - e_ref:+.1e})")
        sys.exit()

    if len(sys.argv) == 3:
        dump, case = pathlib.Path(sys.argv[1]), CASES / f"{sys.argv[2]}.hfinp"
        mol = build(case, cart=True)
        S = mol.intor("int1e_ovlp")
        V = mol.intor("int1e_nuc") + mol.intor("ECPscalar")
        S_p, V_p = read_dump(dump / "Scart.txt"), read_dump(dump / "Vcart.txt")
        # PySCF and Planck normalize Cartesian components differently; compare in
        # the per-function-normalized form, which both must agree on.
        d, d_p = 1 / np.sqrt(np.diag(S)), 1 / np.sqrt(np.diag(S_p))
        print(f"overlap   max |dS| = {np.abs(d[:, None] * S * d - d_p[:, None] * S_p * d_p).max():.2e}")
        print(f"V_ne+ECP  max |dV| = {np.abs(d[:, None] * V * d - d_p[:, None] * V_p * d_p).max():.2e}"
              f"  (max |V| {np.abs(V).max():.1f})")
        sys.exit()

    for case in sorted(CASES.glob("*def2*ecp*.hfinp")):
        mol = build(case, cart=False)
        mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        mf.conv_tol = 1e-12
        e = mf.kernel()
        assert mf.converged, case.stem
        print(f"{case.stem:40s} nao={mol.nao:4d}  E = {e:.10f}")


if __name__ == "__main__":
    main()
