#!/usr/bin/env python3
"""B5 acceptance: Be CCSDTQ (generated, spin-adapted) == FCI.

Be has 4 electrons, so CCSDTQ is exact (CCSDTQ == FCI). This runs the built
`hartree-fock` binary on a Be/STO-3G `correlation cc4` input and asserts the
Total RCCSDTQ Energy equals the FCI reference to --atol. It is the end-to-end
gate for the generated spin-adapted / multi-Sz-sector CC path.

--atol defaults to 1e-7, matching the be_rccsdtq_sto3g regression gate. The
converged generated result sits ~6e-8 from the PySCF FCI reference (a residual-
equation micro-discrepancy well under the gate; the solver itself converges to
rms(res) ~1e-13, so this is not a convergence-tolerance issue). Pass a tighter
--atol only to study that gap, not as the acceptance bar.

PREREQUISITE: the binary must be built spin-adapted:
    cmake .. -DPLANCK_CC_MAXORDER=4 -DPLANCK_CC_SPIN_ADAPT=ON
    cmake --build . --target hartree-fock
Without spin-adaptation (or with a broken ERI convention / emit) the generated
kernels are grossly wrong and the total dives ~0.2 Eh below FCI. This script
flags that magnitude distinctly, but does NOT attempt to name the specific cause
of a smaller miss -- earlier heuristic guesses ("0.25 defect", "undriven t4
sector") misdiagnosed every real failure and were removed.

Reference (Be/STO-3G, PySCF FCI): total = -14.4036551081 (E_corr -0.0517746319).

Usage:
    ccsdtq_fci_acceptance.py [--binary PATH] [--input PATH] [--atol 1e-7]
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
# The gross historical defect (physicist/chemists ERI mismatch + antisymmetric
# emit perms) drove E_corr ~5x too big: total ~-14.60, ~0.2 Eh BELOW FCI. A total
# this far below FCI means the generated kernels are grossly wrong, not a numeric
# miss. This is the only heuristic branch kept, and it triggers only for a defect
# of that magnitude -- it does NOT fire on the healthy converged result.
DEFECT_TOTAL_CEILING = -14.45  # below this => gross defect, not a numeric miss

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
    ap.add_argument("--atol", type=float, default=1e-7,
                    help="max |CCSDTQ total - FCI| (default 1e-7, matching the "
                         "be_rccsdtq_sto3g regression gate)")
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

    if gap <= args.atol:
        print("[PASS] Be CCSDTQ == FCI: the generated multi-sector CC path is "
              "correct end-to-end.")
        return 0
    # Only one diagnostic branch survives, and it keys on MAGNITUDE alone -- the
    # gross ~5x defect drives the total ~0.2 Eh below FCI. Anything milder is just
    # reported as an off-by amount; do NOT guess a specific cause (earlier versions
    # blamed the spin-orbital 0.25 defect or an undriven t4 sector for ANY miss,
    # and misdiagnosed every real failure -- the cause is not recoverable from the
    # total alone).
    if total < DEFECT_TOTAL_CEILING:
        print(f"[FAIL] total {total:.6f} is ~0.2 Eh below FCI — a gross defect in "
              "the generated kernels (not a numeric miss). See the CC kernel "
              "codegen/convention path.")
        return 1
    print(f"[FAIL] CCSDTQ total off by {gap:.3e} (> atol {args.atol:.1e}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
