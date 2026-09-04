"""Python front end for the Planck quantum-chemistry binaries.

Runs an .hfinp input through hartree-fock or planck-dft and returns the
results as a dict. Results come from the binary's own --json dump (a stable
machine contract), NOT from scraping the human log, so log-format changes
cannot break this.

    import planck
    r = planck.run("water.hfinp")
    r["total_energy"]      # -76.02...  (correlated total if post-HF ran, else SCF)
    r["gradient"]          # [[gx,gy,gz], ...] Ha/Bohr, or None if no gradient
    r["log"]               # full stdout+stderr text, for anything not in the JSON

Schema of the returned dict mirrors src/io/results_json.h:
    natoms, charge, multiplicity, atomic_numbers,
    coordinates_bohr, electronic_energy, nuclear_repulsion,
    scf_total_energy, total_energy, has_correlation,
    gradient (only for gradient workflows),
    dipole_au ([x,y,z]) and quadrupole_au (3x3), when a multipole report ran,
    plus: log, returncode.
"""

import json
import subprocess
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HF = _ROOT / "build" / "hartree-fock"
_DFT = _ROOT / "build" / "planck-dft"


def run(input_path, binary=None, timeout=None):
    """Run an .hfinp file and return a results dict.

    binary: None -> auto (planck-dft if the input has %begin_dft, else
    hartree-fock), or an explicit path/name.
    """
    input_path = Path(input_path)
    if binary is None:
        binary = _DFT if "%begin_dft" in input_path.read_text() else _HF
    exe = Path(binary)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        json_path = tf.name
    try:
        proc = subprocess.run(
            [str(exe), str(input_path), "--json", json_path],
            capture_output=True, text=True, timeout=timeout,
        )
        log = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"{exe.name} exited {proc.returncode}:\n{log}")

        with open(json_path) as f:
            r = json.load(f)
    finally:
        Path(json_path).unlink(missing_ok=True)

    r.setdefault("gradient", None)
    r.setdefault("dipole_au", None)
    r.setdefault("quadrupole_au", None)
    r["log"] = log
    r["returncode"] = proc.returncode
    return r


def run_many(input_paths, binary=None, timeout=None, workers=None):
    """Run several inputs in parallel (process pool). Returns a list aligned to
    input_paths; a failed run yields the RuntimeError instead of a dict so one
    bad geometry can't sink the batch."""
    from concurrent.futures import ProcessPoolExecutor

    def _one(p):
        try:
            return run(p, binary=binary, timeout=timeout)
        except Exception as e:  # ponytail: return the error, don't raise — batch survives
            return e

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, [str(p) for p in input_paths]))


if __name__ == "__main__":
    import sys
    from pprint import pprint

    if len(sys.argv) > 1:
        r = run(sys.argv[1])
        r.pop("log")  # too noisy to print
        pprint(r)
    else:
        print("usage: python planck.py <input.hfinp>")
