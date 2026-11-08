"""Check a fixed-input FCIQMC thread ladder and print its timing and memory table."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median


def read_run(directory, threads):
    logs = list(directory.glob("*.log"))
    if len(logs) != 1:
        raise ValueError(f"{directory}: expected one solver log, found {len(logs)}")
    log = logs[0]
    meta = json.loads(log.with_suffix(".json").read_text())
    if meta["exit_code"] != 0 or int(meta["thread_env"]["OMP_NUM_THREADS"]) != threads:
        raise ValueError(f"{log}: unsuccessful run or wrong thread count")
    input_hash = hashlib.sha256(log.with_suffix(".hfinp").read_bytes()).hexdigest()
    if input_hash != meta["input_sha256"]:
        raise ValueError(f"{log}: generated input differs from the measured input")

    output = log.read_text()
    omp = re.search(r"^\s*OpenMP\s+[^\n]*?, (\d+) threads?\b", output, re.MULTILINE)
    if (int(omp[1]) if omp else 1) != threads:
        raise ValueError(f"{log}: solver's OpenMP configuration differs from the requested count")
    phases = {}
    for line in output.splitlines():
        if "FCIQMC_PROFILE:" in line:
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            phase = fields["phase"]
            if phase in phases:
                raise ValueError(f"{log}: duplicate {phase} record")
            phases[phase] = fields
    wall = re.search(r"Wall Time\s*:\s*([\d.]+) s", output)
    if wall is None:
        raise ValueError(f"{log}: missing final wall time")
    driver_seconds = float(wall[1])
    # The launcher encloses the entire executable. A larger driver interval
    # indicates a clock discontinuity (the log is rounded to milliseconds).
    if driver_seconds > meta["process_wall_s"] + 0.001:
        raise ValueError(f"{log}: driver wall {driver_seconds:g}s exceeds "
                         f"launcher wall {meta['process_wall_s']:g}s")

    # Performance comparisons require identical inputs, sampled results and
    # work. Compare integer counters across both phases, including rejections.
    identity = {
        "binary": meta["binary_sha256"], "input": input_hash,
        "hamiltonian": phases["hamiltonian"]["id"], "result": phases["result"]["id"],
        "system": meta["system"], "machine": meta["machine"], "logical_cpus": meta["logical_cpus"],
        "thread_env": {k: v for k, v in meta["thread_env"].items() if k != "OMP_NUM_THREADS"},
    }
    for name in ("equilibration", "sampling"):
        for key in ("steps", "parents", "attempts", "invalid", "zero_h", "initiator_rejected",
                    "rounded_zero", "records_raw", "records_folded", "canonical_emitted"):
            identity[f"{name}.{key}"] = int(phases[name][key])
    sample = phases["sampling"]
    peak_bytes = meta["peak_rss_bytes"]
    values = {
        "wall": driver_seconds, "equil": float(phases["equilibration"]["phase_wall_s"]),
        "sample": float(sample["phase_wall_s"]), "worker": float(sample["worker_wall_s"]),
        "merge": float(sample["merge_s"]), "prefill": float(sample["prefill_s"]),
        "rss": peak_bytes / 1024**2 if peak_bytes is not None else None,
    }
    if any(value is not None and (not math.isfinite(value) or value < 0)
           for value in values.values()):
        raise ValueError("invalid timing or RSS measurement")
    return identity, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threads", nargs="+", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or min(args.threads) < 1 or len(set(args.threads)) != len(args.threads):
        parser.error("expected positive repetitions and distinct positive thread counts")

    baseline = None
    by_threads = {}
    for threads in args.threads:
        runs = []
        for rep in range(1, args.repetitions + 1):
            directory = args.directory / f"run-{rep}" / f"threads-{threads}"
            try:
                identity, run = read_run(directory, threads)
            except (OSError, KeyError, ValueError) as error:
                parser.exit(1, f"{directory}: {error}\n")
            if baseline is None:
                baseline = identity
            elif baseline != identity:
                changed = [key for key in baseline if baseline[key] != identity[key]]
                parser.exit(1, f"{directory}: differing {', '.join(changed)}\n")
            runs.append(run)
        by_threads[threads] = runs

    print(f"Hamiltonian {baseline['hamiltonian']}   Result {baseline['result']}")
    print("Medians in seconds except peak RSS (MiB). Wall range spans all repetitions.")
    print("threads   wall    min    max   equil   sample   worker   merge   prefill   RSS")
    for threads, runs in by_threads.items():
        walls = [r["wall"] for r in runs]
        times = [median(r[key] for r in runs) for key in ("equil", "sample", "worker", "merge", "prefill")]
        rss = [r["rss"] for r in runs]
        memory = "n/a" if None in rss else f"{median(rss):6.1f}"
        print(f"{threads:7d} " + " ".join(f"{n:7.2f}" for n in
              (median(walls), min(walls), max(walls), *times)) + f" {memory}")


if __name__ == "__main__":
    main()
