#!/usr/bin/env python3
"""Three convergence checks for FCIQMC against deterministic FCI.

These are DIAGNOSTICS, not regression gates. Each sweeps one knob and prints a
table; reading the table is the test. They are separate from the suite because
each sweep is minutes-to-hours, which no CI budget absorbs.

  1. production length  -- error bars must SHRINK while central values do not drift
  2. walker population  -- both estimators must approach FCI with no trend in N
  3. coefficient ratios -- <N_I>/<N_0> against C_I/C_0, MAGNITUDES AND SIGNS

Test 3 is the strongest of the three. The energy is one scalar contracted over
the whole vector, so phase errors in spawning, death/cloning and annihilation can
cancel inside it; the ratios cannot hide them.

Usage:
    tests/fciqmc_validation.py --binary build-full/hartree-fock \
        --exact <FCI total energy> --test all
"""
import argparse
import os
import math
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
INPUTS = HERE / "inputs" / "exploratory" / "fciqmc" / "validation"

SHIFT = re.compile(r"Shift energy\s+([-+0-9Ee.]+)\s+\+/-\s+([-+0-9Ee.]+)")
PROJ = re.compile(r"Projected energy\s+([-+0-9Ee.]+)\s+\+/-\s+([-+0-9Ee.]+)")
RATIO = re.compile(
    r"det\s+(0x[0-9a-f]+)/(0x[0-9a-f]+)\s+(?:C_I/C_0|<N_I>/<N_0>)\s+([-+0-9.]+)")


def run(binary, inp):
    # BASIS_PATH must be set or the binary cannot find its basis sets. The
    # compiled-in default carries a spurious "install/" segment and never
    # resolves (see vault/Status/Open Work.md), so every run in this repo
    # depends on the environment. Default it from the repo layout rather than
    # inheriting whatever happens to be exported.
    env = dict(os.environ)
    env.setdefault("BASIS_PATH", str(HERE.parent / "basis-sets"))
    out = subprocess.run([str(binary), str(inp)], capture_output=True, text=True,
                         env=env)
    if out.returncode != 0:
        sys.exit(f"FAILED: {inp}\n{out.stdout[-3000:]}\n{out.stderr[-2000:]}")
    return out.stdout


def energies(text):
    s = SHIFT.search(text)
    p = PROJ.search(text)
    return (
        (float(s.group(1)), float(s.group(2))) if s else None,
        (float(p.group(1)), float(p.group(2))) if p else None,
    )


def ratios(text):
    """Determinant -> ratio, from the LAST dump in the output."""
    blocks = text.split("Dominant determinants")
    if len(blocks) < 2:
        return {}
    return {(a, b): float(v) for a, b, v in RATIO.findall(blocks[-1])}


def sigma(value, err, exact):
    return abs(value - exact) / err if err > 0 else float("inf")


# The sweep value is the TRAILING number in the stem, not the first one: the
# fixture name itself may contain a digit (e.g. "c2_pop_5000"), so a leading-number match
# keys every file to the same value and silently scrambles the table.
STEM_VALUE = re.compile(r"(\d+)$")


def sweep_value(path):
    m = STEM_VALUE.search(path.stem)
    if not m:
        sys.exit(f"cannot read a sweep value from {path.name}")
    return int(m.group(1))


def sweep(binary, pattern, knob, exact):
    files = sorted(INPUTS.glob(pattern), key=sweep_value)
    print(f"{knob:>10}  {'shift':>15} {'+/-':>10} {'sig':>6}   "
          f"{'projected':>15} {'+/-':>10} {'sig':>6}")
    rows = []
    for f in files:
        n = sweep_value(f)
        s, p = energies(run(binary, f))
        rows.append((n, s, p))
        line = f"{n:>10}  "
        line += (f"{s[0]:>15.8f} {s[1]:>10.2e} {sigma(*s, exact):>6.2f}   "
                 if s else f"{'--':>15} {'--':>10} {'--':>6}   ")
        line += (f"{p[0]:>15.8f} {p[1]:>10.2e} {sigma(*p, exact):>6.2f}"
                 if p else f"{'--':>15} {'--':>10} {'--':>6}")
        print(line, flush=True)
    return rows


def verdict(rows, knob):
    """Error bars should shrink; central values should not trend."""
    ok = True
    errs = [r[1][1] for r in rows if r[1]]
    if len(errs) >= 2 and errs[-1] >= errs[0]:
        print(f"  [WARN] shift error bar did not shrink with {knob} "
              f"({errs[0]:.2e} -> {errs[-1]:.2e})")
        ok = False
    # A systematic drift shows as the last point sitting outside the first
    # point's error bar in a consistent direction. With four points this is a
    # smell test, not a fit -- say so rather than over-claiming.
    vals = [(r[0], r[1][0], r[1][1]) for r in rows if r[1]]
    if len(vals) >= 2:
        drift = vals[-1][1] - vals[0][1]
        scale = max(vals[0][2], vals[-1][2])
        if scale > 0 and abs(drift) > 3 * scale:
            print(f"  [WARN] shift central value drifted {drift:+.2e} "
                  f"across the {knob} sweep, beyond 3x the error bar")
            ok = False
    print(f"  {'PASS' if ok else 'CHECK'}: {knob} sweep")
    return ok


def compare_ratios(binary, exact_energy):
    fci = ratios(run(binary, INPUTS / "hf_fci_ref.hfinp"))
    qmc = ratios(run(binary, INPUTS / "hf_pop_100000.hfinp"))
    if not fci or not qmc:
        sys.exit("no coefficient dump found -- is verbosity set to verbose?")

    shared = [d for d in fci if d in qmc]
    if not shared:
        sys.exit("no determinants in common -- the two paths disagree on the "
                 "reference determinant, which is a defect, not a tolerance issue")

    # RENORMALISE TO A COMMON ANCHOR before comparing.
    #
    # The two codes pick their own reference, and on a degenerate ground state
    # they can pick DIFFERENT members of the same degenerate pair -- which flips
    # every sign globally and looks like total disagreement. Measured on C2:
    # 12 of 15 "sign mismatches" that were entirely an artifact of FCI anchoring
    # on 0x3f/0x6f while FCIQMC anchored on its partner 0x6f/0x3f. Renormalising
    # both to the same determinant reduced it to 0 of 15.
    anchor = max(shared, key=lambda d: abs(qmc[d]))
    fci_scale, qmc_scale = fci[anchor], qmc[anchor]
    if fci_scale == 0.0 or qmc_scale == 0.0:
        sys.exit("common anchor carries zero weight in one of the dumps")

    print(f"  common anchor {anchor[0]}/{anchor[1]}\n")
    print(f"{'alpha':>18} {'beta':>18} {'C_I/C_0':>12} {'<N_I>/<N_0>':>14} "
          f"{'diff':>10}  sign")
    worst = 0.0
    bad_sign = 0
    for d in sorted(shared, key=lambda k: -abs(fci[k])):
        c, n = fci[d] / fci_scale, qmc[d] / qmc_scale
        agree = (c > 0) == (n > 0)
        bad_sign += not agree
        worst = max(worst, abs(c - n))
        print(f"{d[0]:>18} {d[1]:>18} {c:>12.6f} {n:>14.6f} "
              f"{abs(c-n):>10.2e}  {'ok' if agree else 'MISMATCH'}")

    print(f"\n  {len(shared)} determinants in common "
          f"({len(fci)} FCI / {len(qmc)} occupied)")
    print(f"  worst |C_I/C_0 - <N_I>/<N_0>| = {worst:.3e}")
    if bad_sign:
        print(f"  [FAIL] {bad_sign} SIGN MISMATCHES -- spawning phases or "
              f"annihilation are wrong")
        return False
    print("  PASS: all shared determinants agree in sign")
    return True


def main():
    # Line-buffered: each sweep point takes minutes, and a block-buffered
    # stdout shows nothing until the very end, which is indistinguishable
    # from a hung run when watching a log.
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build-full/hartree-fock")
    ap.add_argument("--exact", type=float, required=True,
                    help="deterministic FCI total energy for the fixture, from "
                         "`correlation fci` on c2_fci_ref.hfinp. REQUIRED rather "
                         "than defaulted: a stale default silently turns every "
                         "sigma in the tables into a comparison against the "
                         "wrong number.")
    ap.add_argument("--test", default="all",
                    choices=["all", "length", "population", "ratios"])
    a = ap.parse_args()
    binary = pathlib.Path(a.binary).resolve()
    if not binary.exists():
        sys.exit(f"no such binary: {binary}")

    ok = True
    if a.test in ("all", "length"):
        print("\n=== 1. production-length convergence "
              f"(exact FCI {a.exact:.10f}) ===")
        ok &= verdict(sweep(binary, "hf_len_*.hfinp", "steps", a.exact), "length")
    if a.test in ("all", "population"):
        print("\n=== 2. walker-population convergence ===")
        ok &= verdict(sweep(binary, "hf_pop_*.hfinp", "walkers", a.exact),
                      "population")
    if a.test in ("all", "ratios"):
        print("\n=== 3. coefficient ratios vs deterministic FCI ===")
        ok &= compare_ratios(binary, a.exact)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
