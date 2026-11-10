"""Check molecular FCIQMC reproducibility beneath the printed energy precision."""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def measure(binary, path, source, threads):
    path.write_text(source)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="false")
    env.setdefault("BASIS_PATH", str(Path(__file__).resolve().parents[1] / "basis-sets"))
    run = subprocess.run(
        [binary, path], cwd=path.parent, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=60,
    )
    if run.returncode:
        raise RuntimeError(run.stdout)
    hamiltonian = re.search(r"phase=hamiltonian id=([a-f0-9]{16})", run.stdout)
    result = re.search(r"phase=result id=([a-f0-9]{16})", run.stdout)
    metric = re.search(r"mo_orthogonality_max_error=(\S+)", run.stdout)
    if not (hamiltonian and result and metric):
        raise RuntimeError(f"Missing profile data at {threads} threads:\n{run.stdout}")
    error = float(metric[1])
    if not math.isfinite(error) or error > 1e-10:
        raise ValueError(f"Orbitals not orthonormal in the working basis: {error}")
    return hamiltonian[1], result[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    args = parser.parse_args()
    binary = (args.build_dir / "hartree-fock").resolve()
    source = args.input.read_text().replace("%begin_scf\n", "%begin_scf\n    fciqmc_profile true\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.hfinp"
        baseline = measure(binary, path, source, 1)
        for threads in (2, 4, 8):
            actual = measure(binary, path, source, threads)
            if actual != baseline:
                raise ValueError(f"{threads} threads: {actual} versus one thread: {baseline}")
        changed, count = re.subn(r"(?m)^(\s*fciqmc_seed\s+)(\d+)$",
                                 lambda m: m[1] + str(int(m[2]) + 1), source)
        if count != 1:
            raise ValueError("Expected exactly one explicit seed")
        other = measure(binary, path, changed, 1)
        if other[0] != baseline[0] or other[1] == baseline[1]:
            raise ValueError(f"Seed control did not isolate the trajectory: {baseline}, {other}")
        print(f"Thread identities match: hamiltonian={baseline[0]} result={baseline[1]} (1/2/4/8)")


if __name__ == "__main__":
    main()
