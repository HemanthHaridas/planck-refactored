#!/usr/bin/env python3
"""Calibrate the HGP / Rys auto-dispatch rule from measured per-bucket
timings produced by ``planck-auto-dispatch-benchmark``.

The original plan in docs/AUTO_DISPATCH_PLAN.md proposed fitting a
parametric cost model. Once we collected the data, the picture turned
out to be much sharper than that:

  - Rys wins, unanimously across every (molecule, basis) case, only at
    L_AB + L_CD <= 1 — i.e. (0,0), (0,1), (1,0).
  - HGP wins at every other bucket, by factors of 2.5x to 12x.
  - The crossover is clean. No bucket is borderline.

So the calibration script's job is no longer parametric fitting — it is
verifying that the rule

    pick_rys(L_AB, L_CD) ≡ (L_AB + L_CD <= 1)

is unanimous across the dataset, and recording the per-bucket evidence
the rule rests on. The runtime predicate compiles to a single integer
compare; no table needed.

Inputs
------
docs/auto_dispatch_timings.csv  — written by planck-auto-dispatch-benchmark.

Outputs
-------
docs/auto_dispatch_fit.json     — rule + per-bucket median timings + sanity stats.
docs/auto_dispatch_curves.svg   — per-(molecule, basis) ms/quartet vs. L_AB+L_CD.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "docs" / "auto_dispatch_timings.csv"
DEFAULT_FIT = REPO_ROOT / "docs" / "auto_dispatch_fit.json"
DEFAULT_SVG = REPO_ROOT / "docs" / "auto_dispatch_curves.svg"


@dataclass(frozen=True)
class BucketRow:
    molecule: str
    basis: str
    engine: str
    L_AB: int
    L_CD: int
    count: int
    total_ms: float
    ms_per_quartet: float


def parse_csv(path: Path) -> list[BucketRow]:
    rows: list[BucketRow] = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith("molecule,"):
                continue
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) != 8:
                raise ValueError(f"unexpected row: {line!r}")
            mol, basis, engine, lab, lcd, count, total, per = parts
            rows.append(
                BucketRow(
                    molecule=mol,
                    basis=basis,
                    engine=engine,
                    L_AB=int(lab),
                    L_CD=int(lcd),
                    count=int(count),
                    total_ms=float(total),
                    ms_per_quartet=float(per),
                )
            )
    return rows


def per_bucket(rows: list[BucketRow]) -> dict:
    """(molecule, basis, L_AB, L_CD) → {engine: ms/quartet}."""
    idx: dict = {}
    for r in rows:
        key = (r.molecule, r.basis, r.L_AB, r.L_CD)
        idx.setdefault(key, {})[r.engine] = r.ms_per_quartet
    return idx


def rule_picks_rys(L_AB: int, L_CD: int) -> bool:
    """The fitted dispatch rule."""
    return (L_AB + L_CD) <= 1


def verify_rule(index: dict) -> dict:
    """For every bucket with both HGP and Rys timings, check whether the
    rule picks the empirically faster engine. Aggregate stats per
    (molecule, basis) and overall.
    """
    overall_disagreements = 0
    overall_buckets = 0
    overall_overhead: list[float] = []
    per_case: dict = {}
    bucket_rows = []
    for (mol, basis, L_AB, L_CD), times in sorted(index.items()):
        hgp = times.get("hgp", 0.0)
        rys = times.get("rys", 0.0)
        if hgp <= 0 or rys <= 0:
            continue
        rule_pick = "rys" if rule_picks_rys(L_AB, L_CD) else "hgp"
        rule_time = rys if rule_pick == "rys" else hgp
        best_time = min(hgp, rys)
        best_engine = "rys" if rys < hgp else "hgp"
        agree = rule_pick == best_engine
        overhead = rule_time / best_time - 1.0
        overall_buckets += 1
        if not agree:
            overall_disagreements += 1
        overall_overhead.append(overhead)
        per_case.setdefault((mol, basis), {"n": 0, "disagree": 0, "overhead": []})
        per_case[(mol, basis)]["n"] += 1
        per_case[(mol, basis)]["disagree"] += 0 if agree else 1
        per_case[(mol, basis)]["overhead"].append(overhead)
        bucket_rows.append(
            {
                "molecule": mol,
                "basis": basis,
                "L_AB": L_AB,
                "L_CD": L_CD,
                "hgp_ms_per_q": hgp,
                "rys_ms_per_q": rys,
                "best_engine": best_engine,
                "rule_pick": rule_pick,
                "overhead_vs_best": overhead,
            }
        )
    per_case_summary = {
        f"{mol}/{basis}": {
            "n_buckets": d["n"],
            "n_disagreements": d["disagree"],
            "mean_overhead": float(st.mean(d["overhead"])) if d["overhead"] else 0.0,
            "max_overhead": max(d["overhead"]) if d["overhead"] else 0.0,
        }
        for (mol, basis), d in per_case.items()
    }
    return {
        "rule": "pick rys iff (L_AB + L_CD) <= 1; else pick hgp",
        "rule_in_code": "static inline bool pick_rys(int L_AB, int L_CD) { return (L_AB + L_CD) <= 1; }",
        "n_buckets_total": overall_buckets,
        "n_disagreements_total": overall_disagreements,
        "mean_overhead_vs_per_bucket_winner": (
            float(st.mean(overall_overhead)) if overall_overhead else 0.0
        ),
        "max_overhead_vs_per_bucket_winner": (
            max(overall_overhead) if overall_overhead else 0.0
        ),
        "per_case": per_case_summary,
        "buckets": bucket_rows,
    }


def aggregate_median_per_bucket(index: dict) -> dict:
    """(L_AB, L_CD) → {engine: median ms/quartet across cases}."""
    agg: dict = {}
    raw: dict = {}
    for (mol, basis, L_AB, L_CD), times in index.items():
        bucket = raw.setdefault((L_AB, L_CD), {"hgp": [], "rys": [], "os": []})
        for k in ("hgp", "rys", "os"):
            if k in times and times[k] > 0:
                bucket[k].append(times[k])
    for k, vs in raw.items():
        agg[k] = {eng: (st.median(samples) if samples else None) for eng, samples in vs.items()}
        agg[k]["n_cases"] = len(vs["hgp"])
    return agg


def render_curves(index: dict, out_path: Path) -> None:
    if plt is None:
        return
    cases = sorted({(m, b) for (m, b, _, _) in index.keys()})
    n = len(cases)
    if n == 0:
        return
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax_i, (mol, basis) in enumerate(cases):
        ax = axes[ax_i // cols][ax_i % cols]
        scatter = defaultdict(lambda: {"x": [], "y": []})
        for (m, b, L_AB, L_CD), times in index.items():
            if (m, b) != (mol, basis):
                continue
            L = L_AB + L_CD
            for engine in ("os", "hgp", "rys"):
                if engine in times and times[engine] > 0:
                    scatter[engine]["x"].append(L)
                    scatter[engine]["y"].append(times[engine])
        style = {
            "os": ("o", "#888", "OS"),
            "hgp": ("s", "#1f77b4", "HGP"),
            "rys": ("^", "#d62728", "Rys"),
        }
        for engine, (marker, color, label) in style.items():
            if scatter[engine]["x"]:
                ax.scatter(
                    scatter[engine]["x"],
                    scatter[engine]["y"],
                    marker=marker,
                    color=color,
                    s=24,
                    alpha=0.7,
                    label=label,
                )
        # The rule's crossover sits between L=1 (Rys) and L=2 (HGP).
        ax.axvspan(-0.5, 1.5, color="#d62728", alpha=0.07, label="rule: Rys")
        ax.axvspan(1.5, 100, color="#1f77b4", alpha=0.05, label="rule: HGP")
        ax.set_title(f"{mol} / {basis}")
        ax.set_xlabel("L_AB + L_CD")
        ax.set_ylabel("ms / quartet")
        ax.set_yscale("log")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
        # Cap x-axis at the data range.
        all_x = [
            L for (m, b, L_AB, L_CD) in index.keys() if (m, b) == (mol, basis)
            for L in (L_AB + L_CD,)
        ]
        if all_x:
            ax.set_xlim(-0.5, max(all_x) + 0.5)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper left")
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)
    fig.suptitle("HGP vs. Rys per-bucket ms/quartet — auto-dispatch rule shaded")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    ap.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"missing input: {args.csv}", file=sys.stderr)
        return 1

    rows = parse_csv(args.csv)
    if not rows:
        print("no rows in CSV", file=sys.stderr)
        return 1

    index = per_bucket(rows)
    verification = verify_rule(index)
    cross_case_medians = aggregate_median_per_bucket(index)

    # Render cross-case median table as a string keyed by "L_AB,L_CD" so
    # the JSON dictionary stays valid (tuple keys are not allowed there).
    medians_serializable = {
        f"{lab},{lcd}": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
        for (lab, lcd), d in cross_case_medians.items()
    }

    fit_summary = {
        "input": str(args.csv.relative_to(REPO_ROOT)),
        "rule": verification["rule"],
        "rule_in_code": verification["rule_in_code"],
        "engines_in_auto_menu": ["hgp", "rys"],
        "n_buckets_total": verification["n_buckets_total"],
        "n_disagreements_total": verification["n_disagreements_total"],
        "mean_overhead_vs_per_bucket_winner": verification["mean_overhead_vs_per_bucket_winner"],
        "max_overhead_vs_per_bucket_winner": verification["max_overhead_vs_per_bucket_winner"],
        "per_case": verification["per_case"],
        "cross_case_medians": medians_serializable,
        "buckets": verification["buckets"],
    }
    args.fit.parent.mkdir(parents=True, exist_ok=True)
    args.fit.write_text(json.dumps(fit_summary, indent=2) + "\n")

    print(f"wrote {args.fit.relative_to(REPO_ROOT)}")
    print(f"  rule: {verification['rule']}")
    print(f"  buckets evaluated: {verification['n_buckets_total']}")
    print(f"  disagreements: {verification['n_disagreements_total']}")
    print(f"  mean overhead vs. per-bucket winner: {verification['mean_overhead_vs_per_bucket_winner']:.2%}")
    print(f"  max overhead vs. per-bucket winner:  {verification['max_overhead_vs_per_bucket_winner']:.2%}")

    if plt is None:
        print("matplotlib not available; skipping SVG render", file=sys.stderr)
    else:
        render_curves(index, args.svg)
        print(f"wrote {args.svg.relative_to(REPO_ROOT)}")

    return 0 if verification["n_disagreements_total"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
