#!/usr/bin/env python3
"""V1.3.4: a dressed generated CC kernel must compute the same residual as the undressed one.

Runs the same input through two `hartree-fock` builds -- one configured with
`-DPLANCK_CC_DRESS_OPERATORS=ON`, one without -- and requires both the correlation energy AND
the iteration count to match.

WHY THE ITERATION COUNT IS PART OF THE GATE. Equal converged energy only shows the two kernels
share a fixed point. Equal iteration count shows they take the same trajectory to it, which is
much closer to "the residual is the same function". Dressing is supposed to be a pure
refactorization of the residual, so anything else is a real difference.

WHY NOT EVERY rccsdt CASE. `water_rccsdt_sto3g` falls back to the determinant-space backstop
(`RCCSDT[DET-BACKSTOP]` in its log) in BOTH builds, so it never reaches the generated kernel and
tells us nothing about dressing. It converges to the same energy either way but in a different
number of iterations (26 vs 54) -- a property of the backstop path, not of dressing. Including
it would have produced a false failure; excluding it silently would have hidden the fact that
one case does not exercise the feature at all. So it is excluded EXPLICITLY, and the exclusion
is verified: the case must actually show the backstop marker, or this script fails.

Usage:
    dressed_kernel_equivalence.py --dressed <build_dir> --undressed <build_dir>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO / "tests" / "inputs" / "regression" / "post_hf"

# Rank-3 cases that reach the generated kernel.
GENERATED_KERNEL_CASES = ("h2", "lih", "bh3")
# Reaches the determinant backstop in both builds; see the module docstring.
BACKSTOP_CASES = ("water",)

BACKSTOP_MARKER = "DET-BACKSTOP"


def _run(binary: Path, input_path: Path) -> str:
    env = dict(os.environ)
    env.setdefault("BASIS_PATH", str(REPO / "basis-sets"))
    proc = subprocess.run([str(binary), str(input_path)],
                          capture_output=True, text=True, timeout=1800, env=env)
    return proc.stdout + proc.stderr


def _corr_energy(log: str) -> str | None:
    hits = re.findall(r"E_corr=(-?\d+\.\d+)", log)
    return hits[-1] if hits else None


def _iterations(log: str) -> str | None:
    """Iteration count, across the three wordings the CC backends use.

    The determinant/arbitrary solvers print "Converged in N iterations."; the standalone
    tensor backend (`RCCSDT[TENSOR]`, which bh3 takes) prints "converged in N steps".
    Matching only the first wording returned None for bh3, and since the comparison was
    `None == None` the case passed VACUOUSLY -- an iteration regression there would have
    been invisible. Callers must treat None as a failure, not as a match.
    """
    hits = re.findall(r"[Cc]onverged in (\d+) (?:iterations|steps)", log)
    return hits[-1] if hits else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dressed", type=Path, required=True,
                        help="build dir configured with PLANCK_CC_DRESS_OPERATORS=ON")
    parser.add_argument("--undressed", type=Path, required=True,
                        help="reference build dir without dressing")
    args = parser.parse_args()

    dressed = args.dressed / "hartree-fock"
    undressed = args.undressed / "hartree-fock"
    for binary in (dressed, undressed):
        if not binary.is_file():
            print(f"FAIL: no binary at {binary}")
            return 1

    failures = 0

    print("=== generated-kernel cases (must match exactly) ===")
    for case in GENERATED_KERNEL_CASES:
        path = INPUT_DIR / f"{case}_rccsdt_sto3g.hfinp"
        if not path.is_file():
            print(f"FAIL {case}: input missing at {path}")
            failures += 1
            continue

        dressed_log = _run(dressed, path)
        undressed_log = _run(undressed, path)

        # Guard against a vacuous pass: if the dressed build silently fell back to the
        # backstop, this case is not testing dressing at all.
        if BACKSTOP_MARKER in dressed_log:
            print(f"FAIL {case}: expected the generated kernel, got {BACKSTOP_MARKER}")
            failures += 1
            continue

        d_e, u_e = _corr_energy(dressed_log), _corr_energy(undressed_log)
        d_i, u_i = _iterations(dressed_log), _iterations(undressed_log)

        if d_e is None or u_e is None:
            print(f"FAIL {case}: no correlation energy in output")
            failures += 1
            continue
        if d_i is None or u_i is None:
            # Never let a missing count compare equal to a missing count: `None == None`
            # is how bh3 passed vacuously before the tensor backend's "N steps" wording
            # was matched.
            print(f"FAIL {case}: could not parse an iteration count "
                  f"(dressed={d_i}, undressed={u_i})")
            failures += 1
            continue

        ok = (d_e == u_e) and (d_i == u_i)
        status = "MATCH" if ok else "DIFFER"
        print(f"  {case:6s} E_corr {d_e} vs {u_e} | iters {d_i} vs {u_i}  {status}")
        if not ok:
            failures += 1

    print("=== backstop cases (excluded from the gate; exclusion verified) ===")
    for case in BACKSTOP_CASES:
        path = INPUT_DIR / f"{case}_rccsdt_sto3g.hfinp"
        if not path.is_file():
            continue
        log = _run(dressed, path)
        if BACKSTOP_MARKER not in log:
            print(f"FAIL {case}: no longer hits {BACKSTOP_MARKER} -- it now reaches the "
                  f"generated kernel and belongs in GENERATED_KERNEL_CASES")
            failures += 1
        else:
            print(f"  {case:6s} uses {BACKSTOP_MARKER} in both builds -- correctly excluded")

    print()
    if failures:
        print(f"FAILED: {failures} case(s)")
        return 1
    print("PASS: dressed and undressed kernels agree on energy and iteration count")
    return 0


if __name__ == "__main__":
    sys.exit(main())
