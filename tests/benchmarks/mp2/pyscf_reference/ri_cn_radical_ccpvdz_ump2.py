#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from pyscf import gto, scf
from pyscf.mp import dfump2_native


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    helper_path = repo_root / "scripts" / "export_ri_3c_pyscf.py"
    spec = importlib.util.spec_from_file_location("planck_pyscf_basis_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)

    mol = gto.Mole()
    mol.verbose = 0
    mol.unit = "Angstrom"
    mol.spin = 1
    mol.atom = """
C     0.000000    0.000000    0.000000
N     0.000000    0.000000    1.171800
"""
    mol.basis = helper._load_basis_map(repo_root / "basis-sets" / "cc-pVDZ", ["C", "N"])
    mol.cart = True
    mol.build()

    t0 = time.perf_counter()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.level_shift = 0.3
    mf.kernel()
    scf_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    auxbasis = helper._load_basis_map(repo_root / "basis-sets" / "cc-pVDZ-RIFIT", ["C", "N"])
    pt = dfump2_native.DFUMP2(mf, auxbasis=auxbasis)
    ecorr = pt.calculate_energy()
    mp2_time = time.perf_counter() - t1

    print(f"PYSCF_UHF_TOTAL_ENERGY {mf.e_tot:.14f}")
    print(f"PYSCF_MP2_CORR_ENERGY {ecorr:.14f}")
    print(f"PYSCF_MP2_TOTAL_ENERGY {pt.e_tot:.14f}")
    print(f"PYSCF_SCF_TIME_SEC {scf_time:.6f}")
    print(f"PYSCF_RI_MP2_TIME_SEC {mp2_time:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
