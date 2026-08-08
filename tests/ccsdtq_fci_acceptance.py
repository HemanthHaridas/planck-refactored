#!/usr/bin/env python3
"""B5 acceptance: Be CCSDTQ (generated, spin-adapted) == FCI.

Be has 4 electrons, so CCSDTQ is exact (CCSDTQ == FCI). This runs the built
`hartree-fock` binary on a Be/STO-3G `correlation cc4` input and asserts the
Total RCCSDTQ Energy equals the FCI reference to --atol. It is the end-to-end
gate for the multi-Sz-sector generated CC path (Gap B): the reference AND the
second t4 sector (t4_aaabaaab) must both be driven, or T4 falls ~4e-6 short.

PREREQUISITE: the binary must be built spin-adapted:
    cmake .. -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON
    cmake --build . --target hartree-fock
Without -DPLANCK_CC_SPIN_ADAPT=ON the generated kernels carry the spin-orbital
0.25*t2*oovv defect and the energy is ~4x wrong / dives below FCI; this script
detects that and reports it distinctly from a small numeric miss.

Reference (Be/STO-3G, PySCF FCI): total = -14.4036551081 (E_corr -0.0517746319).

Usage:
    ccsdtq_fci_acceptance.py [--binary PATH] [--input PATH] [--atol 1e-8]
Exit 0 on pass, 1 on fail (or build/run error).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# PySCF FCI reference for Be/STO-3G (== exact CCSDTQ for 4 electrons).
FCI_TOTAL = -14.4036551081
FCI_ECORR = -0.0517746319
# The historical defect drove E_corr ~5x too big (total ~-14.60, ~0.2 Eh BELOW
# FCI). A total this far below FCI means the build is NOT spin-adapted.
DEFECT_TOTAL_CEILING = -14.45  # below this => defect, not a numeric miss

REPO_ROOT = Path(__file__).resolve().parents[1]

TOTAL_RE = re.compile(r"Total RCCSDTQ Energy\s+([-+0-9Ee.]+)")
CORR_RE = re.compile(r"Correlation Energy\s+([-+0-9Ee.]+)")


def find_binary(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"[FAIL] binary not found: {p}")
        return p
    for cand in (REPO_ROOT / "build" / "hartree-fock",
                 REPO_ROOT / "build" / "bin" / "hartree-fock"):
        if cand.is_file():
            return cand
    sys.exit("[FAIL] hartree-fock binary not found under build/; pass --binary. "
             "Build it with -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=None, help="path to hartree-fock")
    default_input = (REPO_ROOT / "tests" / "inputs" / "regression"
                     / "post_hf" / "be_rccsdtq_sto3g.hfinp")
    ap.add_argument("--input", default=str(default_input), help="Be cc4 .hfinp")
    ap.add_argument("--atol", type=float, default=1e-8,
                    help="max |CCSDTQ total - FCI| (default 1e-8)")
    args = ap.parse_args()

    binary = find_binary(args.binary)
    inp = Path(args.input)
    if not inp.is_file():
        sys.exit(f"[FAIL] input not found: {inp}")

    print(f"[INFO] binary: {binary}")
    print(f"[INFO] input : {inp}")
    proc = subprocess.run([str(binary), str(inp)],
                          capture_output=True, text=True, timeout=1800)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out[-3000:])
        print(f"[FAIL] hartree-fock exited {proc.returncode}")
        return 1

    total_m = TOTAL_RE.search(out)
    corr_m = CORR_RE.search(out)
    if not total_m:
        print(out[-3000:])
        print("[FAIL] could not parse 'Total RCCSDTQ Energy' from output "
              "(did the run converge?)")
        return 1

    total = float(total_m.group(1))
    corr = float(corr_m.group(1)) if corr_m else float("nan")
    gap = abs(total - FCI_TOTAL)

    print(f"[INFO] CCSDTQ total  = {total:.10f}  (E_corr {corr:.10f})")
    print(f"[INFO] FCI    total  = {FCI_TOTAL:.10f}  (E_corr {FCI_ECORR:.10f})")
    print(f"[INFO] |gap to FCI|  = {gap:.3e}   (atol {args.atol:.1e})")

    if total < DEFECT_TOTAL_CEILING:
        print(f"[FAIL] total {total:.6f} is far BELOW FCI — the spin-orbital "
              "0.25 defect. Rebuild with -DPLANCK_CC_SPIN_ADAPT=ON.")
        return 1
    if gap <= args.atol:
        print("[PASS] Be CCSDTQ == FCI: the generated multi-sector CC path is "
              "correct end-to-end.")
        return 0
    if gap <= 1e-5:
        print(f"[FAIL] near miss ({gap:.3e}) — likely the t4 second sector "
              "(t4_aaabaaab) is not being driven (Gap B4/B1 not active in this "
              "build); expected T4 contribution ~-4.4e-6.")
        return 1
    print(f"[FAIL] CCSDTQ total off by {gap:.3e}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
