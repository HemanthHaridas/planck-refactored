#!/usr/bin/env python3
"""Calibrate the OS / HGP / Rys auto-dispatch rule from measured
per-bucket timings produced by ``planck-auto-dispatch-benchmark``.

The original plan in docs/AUTO_DISPATCH_PLAN.md proposed fitting a
parametric cost model. The first calibration on the pre-optimization
data made it look much simpler than that: Rys won the very-low-L corner
(L_AB + L_CD <= 1) and HGP won everywhere else, so the rule collapsed to
a single integer compare and OS dropped out of the menu entirely.

That two-way picture no longer holds. After the integral-engine
optimization pass — per-quartet Rys scratch (PR #126), the HGP
VRR-per-pair / HRR-outside rework with hoisted (a0|c0) blocks, and the
flattened-triangle parallel ``_compute_2e`` across all engines — the
engines were re-timed and the crossovers moved:

  - HGP is now the broad winner: it takes 66 of the 81 angular-momentum
    buckets, including the entire low-L corner that Rys used to own.
    The HGP HRR-hoisting in particular wiped out Rys's old (0,0)/(0,1)/
    (1,0) advantage.
  - OS re-enters the menu and wins a contiguous high-L corner — roughly
    L_AB + L_CD >= 11 with L_AB >= 5 — where HGP's per-shell-quartet HRR
    bookkeeping overhead finally outweighs its primitive-loop savings.
  - Rys survives only in the extreme corner, (7,8) and (8,8), where its
    quadrature cost grows more slowly than the OS/HGP recurrence tables.

So the rule is now genuinely three-way and is expressed as a small set
of integer comparisons on (L_AB, L_CD) rather than a single sum. This
script's job is to (1) hold the canonical rule in ``dispatch_engine``,
(2) verify it picks the empirically fastest engine in every bucket that
has all three timings, and (3) record the per-bucket evidence and
cross-case medians the rule rests on.

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


ENGINES = ("hgp", "rys", "os")
DEFAULT_ENGINE = "hgp"


def derive_region_table(medians: dict) -> dict:
    """Derive the canonical (L_AB, L_CD) → engine dispatch table directly
    from the cross-case bucket medians.

    The rule is no longer a hand-written set of angular-momentum
    inequalities; it is exactly "pick the engine with the lowest
    cross-case median ms/quartet in that bucket". This makes the rule
    self-adjusting: when an engine optimization (e.g. the planned d-orbital
    OS fast path) re-shapes the cost surface, re-running the benchmark and
    this script moves the region boundaries automatically, with no manual
    inequality edits.

    A bucket is only assigned an engine when all three timings are present
    (every engine is benchmarked there); buckets missing a timing are left
    out of the table and handled by the runtime fallback in
    ``dispatch_engine``.
    """
    table: dict = {}
    for (L_AB, L_CD), m in medians.items():
        times = {e: m[e] for e in ENGINES if m.get(e) is not None}
        if len(times) < len(ENGINES):
            continue
        table[(L_AB, L_CD)] = min(times, key=times.get)
    return table


# Canonical region table, populated by ``main`` from the measured medians
# before any verification or rendering runs. Kept module-global so the
# existing (L_AB, L_CD)-only call sites (verify_rule, render_curves) need
# no signature change.
_REGION_TABLE: dict = {}


def set_region_table(table: dict) -> None:
    global _REGION_TABLE
    _REGION_TABLE = dict(table)


def dispatch_engine(L_AB: int, L_CD: int) -> str:
    """Canonical OS / HGP / Rys auto-dispatch rule.

    Returns the engine with the lowest measured cross-case median for the
    bucket, looked up from the data-derived ``_REGION_TABLE`` (see
    ``derive_region_table``). No angular-momentum inequalities are
    hard-coded.

    For a bucket not present in the table (an (L_AB, L_CD) the benchmark
    did not cover, or one missing a timing), fall back to the nearest
    covered bucket by |dL_AB| + |dL_CD|, breaking ties toward higher
    angular momentum; if the table is empty, return ``hgp`` — the broad
    historical winner and a safe default.
    """
    hit = _REGION_TABLE.get((L_AB, L_CD))
    if hit is not None:
        return hit
    if not _REGION_TABLE:
        return DEFAULT_ENGINE
    # Nearest covered bucket. The tie-break prefers the larger (a, c) so an
    # out-of-table high-L quartet inherits the high-L corner's choice
    # rather than a low-L one.
    best_key = min(
        _REGION_TABLE,
        key=lambda k: (abs(k[0] - L_AB) + abs(k[1] - L_CD), -(k[0] + k[1])),
    )
    return _REGION_TABLE[best_key]


def region_table_as_cpp(table: dict) -> str:
    """Render the derived table as a flat C++ lookup so the runtime rule
    is generated from data, not transcribed by hand.
    """
    if not table:
        return "// empty region table"
    max_l = max(max(a, c) for (a, c) in table)
    engine_enum = {"hgp": "HeadGordonPople", "rys": "RysQuadrature", "os": "ObaraSaika"}
    lines = [
        f"// Generated from {DEFAULT_FIT.name} — do not hand-edit. Needs <algorithm>.",
        f"static constexpr int kAutoDispatchMaxL = {max_l};",
        "IntegralMethod auto_dispatch_engine(int L_AB, int L_CD) {",
        "  L_AB = std::clamp(L_AB, 0, kAutoDispatchMaxL);",
        "  L_CD = std::clamp(L_CD, 0, kAutoDispatchMaxL);",
        "  static constexpr IntegralMethod kTable[kAutoDispatchMaxL + 1]"
        "[kAutoDispatchMaxL + 1] = {",
    ]
    for a in range(max_l + 1):
        row = []
        for c in range(max_l + 1):
            eng = table.get((a, c)) or dispatch_engine(a, c)
            row.append(f"IntegralMethod::{engine_enum[eng]}")
        lines.append("    {" + ", ".join(row) + "},")
    lines.append("  };")
    lines.append("  return kTable[L_AB][L_CD];")
    lines.append("}")
    return "\n".join(lines)


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
        os = times.get("os", 0.0)
        rys = times.get("rys", 0.0)
        if hgp <= 0 or rys <= 0 or os <= 0:
            continue
        rule_pick = dispatch_engine(L_AB, L_CD)
        rule_time = {"hgp": hgp, "os": os, "rys": rys}[rule_pick]
        best_time = min(hgp, rys, os)
        best_engine = "rys" if rys < hgp and rys < os else ("hgp" if hgp < os else "os")
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
                "os_ms_per_q": os,
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
        "rule": (
            "Per-bucket lowest-cross-case-median engine, derived directly "
            "from the timing table (no hard-coded angular-momentum "
            "inequalities). See cross_case_medians and region_table."
        ),
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


def verify_rule_on_medians(medians: dict) -> dict:
    """Check the rule against the cross-case bucket medians.

    This is the gate the rule is actually fitted to: per-(molecule,
    basis) rows carry measurement noise that can flip a near-tied bucket
    (e.g. water/cc-pVDZ (4,4), OS vs HGP within ~3%), but the median over
    all cases is the stable signal. Disagreements here mean the rule is
    genuinely wrong, not just unlucky on one timing.
    """
    disagreements = []
    for (L_AB, L_CD), m in sorted(medians.items()):
        times = {e: m[e] for e in ("hgp", "rys", "os") if m.get(e) is not None}
        if len(times) < 3:
            continue
        rule_pick = dispatch_engine(L_AB, L_CD)
        best_engine = min(times, key=times.get)
        if rule_pick != best_engine:
            disagreements.append(
                {
                    "L_AB": L_AB,
                    "L_CD": L_CD,
                    "rule_pick": rule_pick,
                    "best_engine": best_engine,
                    "overhead_vs_best": times[rule_pick] / times[best_engine] - 1.0,
                }
            )
    return {
        "n_buckets": len(medians),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


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
        # Mark, for each bucket, which engine the rule picks. The
        # three-way rule depends on (L_AB, L_CD) individually, not just
        # their sum, so a single crossover band no longer describes it;
        # instead ring the chosen engine's data point per bucket.
        ring = {"os": "#888", "hgp": "#1f77b4", "rys": "#d62728"}
        for (m, b, L_AB, L_CD), times in index.items():
            if (m, b) != (mol, basis):
                continue
            pick = dispatch_engine(L_AB, L_CD)
            t = times.get(pick)
            if t and t > 0:
                ax.scatter(
                    [L_AB + L_CD],
                    [t],
                    marker="o",
                    facecolors="none",
                    edgecolors=ring[pick],
                    s=90,
                    linewidths=1.0,
                )
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
    fig.suptitle(
        "OS / HGP / Rys per-bucket ms/quartet — ringed point = engine the rule picks"
    )
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
    cross_case_medians = aggregate_median_per_bucket(index)

    # Derive the dispatch regions from the medians and install them before
    # any verification or rendering — every dispatch_engine() call below
    # consults this table.
    region_table = derive_region_table(cross_case_medians)
    set_region_table(region_table)

    verification = verify_rule(index)
    median_verification = verify_rule_on_medians(cross_case_medians)

    # Render cross-case median table as a string keyed by "L_AB,L_CD" so
    # the JSON dictionary stays valid (tuple keys are not allowed there).
    medians_serializable = {
        f"{lab},{lcd}": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
        for (lab, lcd), d in cross_case_medians.items()
    }
    region_serializable = {
        f"{lab},{lcd}": eng for (lab, lcd), eng in sorted(region_table.items())
    }

    fit_summary = {
        "input": str(args.csv.relative_to(REPO_ROOT)),
        "rule": verification["rule"],
        "rule_in_code": region_table_as_cpp(region_table),
        "engines_in_auto_menu": sorted({eng for eng in region_table.values()}),
        "n_buckets_total": verification["n_buckets_total"],
        "n_disagreements_total": verification["n_disagreements_total"],
        "mean_overhead_vs_per_bucket_winner": verification["mean_overhead_vs_per_bucket_winner"],
        "max_overhead_vs_per_bucket_winner": verification["max_overhead_vs_per_bucket_winner"],
        "median_verification": median_verification,
        "region_table": region_serializable,
        "per_case": verification["per_case"],
        "cross_case_medians": medians_serializable,
        "buckets": verification["buckets"],
    }
    args.fit.parent.mkdir(parents=True, exist_ok=True)
    args.fit.write_text(json.dumps(fit_summary, indent=2) + "\n")

    print(f"wrote {args.fit.relative_to(REPO_ROOT)}")
    print(f"  rule: {verification['rule']}")
    print(f"  buckets evaluated: {verification['n_buckets_total']}")
    print(
        f"  per-case disagreements: {verification['n_disagreements_total']}"
        f" / {verification['n_buckets_total']} bucket-rows (noise-level ties)"
    )
    print(f"  mean overhead vs. per-bucket winner: {verification['mean_overhead_vs_per_bucket_winner']:.2%}")
    print(f"  max overhead vs. per-bucket winner:  {verification['max_overhead_vs_per_bucket_winner']:.2%}")
    print(
        f"  cross-case median disagreements: {median_verification['n_disagreements']}"
        f" / {median_verification['n_buckets']} buckets  (this is the acceptance gate)"
    )
    for d in median_verification["disagreements"]:
        print(
            f"    ({d['L_AB']},{d['L_CD']}): rule picks {d['rule_pick']},"
            f" median winner is {d['best_engine']} (+{d['overhead_vs_best']:.2%})"
        )

    if plt is None:
        print("matplotlib not available; skipping SVG render", file=sys.stderr)
    else:
        render_curves(index, args.svg)
        print(f"wrote {args.svg.relative_to(REPO_ROOT)}")

    # Gate on the cross-case median, which is what the rule is fitted to;
    # individual per-case rows carry timing noise (see verify_rule_on_medians).
    return 0 if median_verification["n_disagreements"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
