"""U0 energy smoke and unchanged UKS-gradient rejection; never builds Planck.

Not a DH derivative validation or an independent aa/ab/bb reconstruction:
that algebra is tested by planck-udh-pt2-energy-contract. Retains all normal
output and generated rejection inputs. No production DH debug flags.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

FIXTURES = Path(__file__).resolve().parent / "inputs/exploratory/dh_gradient/uks_u0"
NUMBER = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"


def check_energy(output, data, zero_beta=False):
    if not re.search(r"Converged\s*:\s*true", output):
        raise ValueError("UKS did not converge")
    if not math.isfinite(data["total_energy"]) or not data.get("has_correlation"):
        raise ValueError("Missing finite correlated total energy")
    match = re.search(rf"PT2 coefficient = ({NUMBER}); bare MP2-like correction = ({NUMBER}) Eh; "
                      rf"scaled contribution = ({NUMBER}) Eh", output)
    if not match:
        raise ValueError("Missing normal DH energy summary")
    c, bare, scaled = map(float, match.groups())
    # Normal summaries round to 10 decimal places, not FD precision.
    if abs(c - 0.27) > 1e-12 or abs(scaled - c * bare) > 1e-10:
        raise ValueError("B2PLYP single correlation scaling mismatch")
    if zero_beta and (abs(bare) > 1e-12 or abs(scaled) > 1e-12):
        raise ValueError("One-electron PT2 correction is not zero")
    return dict(total_energy=data["total_energy"], c_pt2=c, bare=bare, scaled=scaled)


def check_rejection(returncode, output):
    if returncode == 0 or "supports only single-point energies for range-separated and double-hybrid functionals" not in output:
        raise ValueError("UKS DH gradient did not hit the expected production scope rejection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/planck-dft"))
    args = parser.parse_args()
    executable = str(args.executable.resolve())
    root = Path(tempfile.mkdtemp(prefix="udh-u0-molecular-"))
    print(f"Artifacts: {root}", flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PLANCK_DFT_DH_")}
    results = []
    for fixture in sorted(FIXTURES.glob("*.hfinp")):
        row = dict(fixture=fixture.name, passed=False)
        try:
            for mode in ("energy", "gradient"):
                stem = fixture.stem + "-" + mode
                inp, log, js = (root / (stem + suffix) for suffix in (".hfinp", ".log", ".json"))
                inp.write_text(re.sub(r"(calculation\s+)energy", rf"\g<1>{mode}", fixture.read_text(), count=1))
                proc = subprocess.run([executable, str(inp), "--json", str(js)],
                                      env=env, capture_output=True, text=True)
                output = proc.stdout + proc.stderr
                log.write_text(output)
                if mode == "energy":
                    if proc.returncode != 0:
                        raise ValueError(f"Energy failed: {log}")
                    row.update(check_energy(output, json.loads(js.read_text()), "zero_beta" in fixture.name))
                else:
                    check_rejection(proc.returncode, output)
                    row["gradient_rejected"] = True
            row["passed"] = True
        except (ValueError, OSError, KeyError) as error:
            row["error"] = str(error)
        results.append(row)
        print(json.dumps(row), flush=True)
    passed = bool(results) and all(row["passed"] for row in results)
    (root / "results.json").write_text(json.dumps(dict(passed=passed, results=results), indent=2) + "\n")
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
