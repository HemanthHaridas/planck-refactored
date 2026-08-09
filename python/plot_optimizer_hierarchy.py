#!/usr/bin/env python3
"""Summarize the factorized-intermediate optimizer across the CC hierarchy.

Five figures, one SVG, over CCSDT / CCSDTQ / CCSDTQP (ranks 3 / 4 / 5):

  1. Operators vs excitation rank        (how many build_W the emitter produces)
  2. Maximum savings vs rank             (the top operator's (uses-1)*build_flops)
  3. Largest footprint vs rank           (the biggest materialized tensor, bytes)
  4. Reuse per operator vs rank          (usage_count: max and mean)
  5. Coverage of top-k operators         (cumulative savings fraction vs k)

Data is computed from ccgen itself (diagram engine, canonical Fock) and cached to
JSON next to the script, because the cc5 (rank-5) manifold is slow to generate
(~90 s) and factor (~40 s). Delete the cache to recompute.

    python plot_optimizer_hierarchy.py [--out FILE.svg] [--sizes O V] [--force]

ponytail: matplotlib SVG, no web stack. Okabe-Ito categorical palette (colorblind-
safe by construction) for the per-rank coverage curves; single-series panels use
one ink color + direct end labels, no legend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CACHE = Path(__file__).with_name(".optimizer_hierarchy_cache.json")

# rank -> (method name, manifolds that carry derived operators)
METHODS = {
    3: ("ccsdt", ("doubles", "triples")),
    4: ("ccsdtq", ("doubles", "triples", "quadruples")),
    5: ("cc5", ("doubles", "triples", "quadruples", "quintuples")),
}
RANK_LABEL = {3: "CCSDT", 4: "CCSDTQ", 5: "CCSDTQP"}

# Okabe-Ito: 8 colorblind-safe hues, assigned in fixed order (never cycled).
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
INK = "#222222"
MUTED = "#888888"


def collect(n_occ: int, n_vir: int) -> dict:
    """Compute the per-rank optimizer metrics from ccgen. Slow (cc5 ~2 min)."""
    from ccgen.generate import generate_cc_equations
    from ccgen.optimization.factorize import (
        manifold_operators, operator_savings, operator_bytes,
    )

    data = {}
    for rank, (method, manifolds) in METHODS.items():
        sys.stderr.write(f"[{RANK_LABEL[rank]}] generating…\n")
        eqs = generate_cc_equations(method, engine="diagram", canonical_fock=True)
        terms = [t for m in manifolds if m in eqs for t in eqs[m]]
        sys.stderr.write(f"[{RANK_LABEL[rank]}] factoring {len(terms)} terms…\n")
        ops = manifold_operators(terms, include_reuse=False)
        savings = sorted((operator_savings(o, n_occ, n_vir) for o in ops),
                         reverse=True)
        data[rank] = {
            "n_ops": len(ops),
            "savings_desc": savings,
            "max_savings": savings[0] if savings else 0,
            "max_bytes": max((operator_bytes(o, n_occ, n_vir) for o in ops),
                             default=0),
            "uses": [o.usage_count for o in ops],
        }
    return data


def load(n_occ: int, n_vir: int, force: bool) -> dict:
    key = f"{n_occ}x{n_vir}"
    if not force and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            # JSON keys are strings; restore int ranks
            return {int(r): v for r, v in cached["data"].items()}
    data = collect(n_occ, n_vir)
    CACHE.write_text(json.dumps({"key": key,
                                 "data": {str(r): v for r, v in data.items()}}))
    return data


def make_figure(data: dict, out: Path, n_occ: int, n_vir: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none",
    })
    ranks = sorted(data)
    xlabels = [RANK_LABEL[r] for r in ranks]
    GB = 1e9

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Factorized-intermediate optimizer across the CC hierarchy  "
        f"(O={n_occ}, V={n_vir})",
        fontsize=13, fontweight="bold", y=0.98)

    def style(ax, title, ylabel, logy=False):
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", color=INK)
        ax.set_ylabel(ylabel)
        ax.set_xticks(ranks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("excitation rank")
        if logy:
            ax.set_yscale("log")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def endlabels(ax, xs, ys, fmt):
        # first point labels up-right (clears the y-axis on log panels), the
        # rest centered above the marker.
        for i, (x, y) in enumerate(zip(xs, ys)):
            ha, off = ("center", (0, 8)) if i else ("left", (6, 8))
            ax.annotate(fmt(y), (x, y), textcoords="offset points",
                        xytext=off, ha=ha, fontsize=9, color=INK)

    # 1. Operators vs rank
    ax = axes[0][0]
    y = [data[r]["n_ops"] for r in ranks]
    ax.plot(ranks, y, "-o", color=OKABE_ITO[0], lw=2, ms=7)
    endlabels(ax, ranks, y, lambda v: f"{v}")
    style(ax, "1. Operators vs rank", "distinct build_W emitted")
    ax.set_ylim(0, max(y) * 1.18)

    # 2. Max savings vs rank (orders of magnitude -> log)
    ax = axes[0][1]
    y = [data[r]["max_savings"] for r in ranks]
    y = [float(v) for v in y]
    ax.plot(ranks, y, "-o", color=OKABE_ITO[1], lw=2, ms=7)
    endlabels(ax, ranks, y, lambda v: f"{v:.1e}")
    style(ax, "2. Maximum savings vs rank", "top operator (uses-1)·flops", logy=True)
    ax.set_ylim(min(y) / 4, max(y) * 4)  # headroom so the low label clears the axis

    # 3. Largest footprint vs rank (log GB)
    ax = axes[0][2]
    y = [data[r]["max_bytes"] / GB for r in ranks]
    ax.plot(ranks, y, "-o", color=OKABE_ITO[2], lw=2, ms=7)
    endlabels(ax, ranks, y, lambda v: f"{v:.0e} GB")
    style(ax, "3. Largest footprint vs rank", "biggest operator tensor (GB)", logy=True)
    ax.set_ylim(min(y) / 4, max(y) * 4)

    # 4. Reuse per operator vs rank (max + mean)
    ax = axes[1][0]
    ymax = [max(data[r]["uses"]) for r in ranks]
    ymean = [sum(data[r]["uses"]) / len(data[r]["uses"]) for r in ranks]
    ax.plot(ranks, ymax, "-o", color=OKABE_ITO[3], lw=2, ms=7, label="max")
    ax.plot(ranks, ymean, "--s", color=OKABE_ITO[4], lw=2, ms=6, label="mean")
    endlabels(ax, ranks, ymax, lambda v: f"{v}")
    style(ax, "4. Reuse per operator vs rank", "usage_count")
    ax.set_ylim(0, max(ymax) * 1.18)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    # 5. Coverage of top-k operators (cumulative savings fraction)
    ax = axes[1][1]
    # anchor each rank's label at a distinct k on its own curve so the three
    # near-identical knees don't stack; stagger vertically as a backstop.
    label_k = {3: 8, 4: 18, 5: 30}
    label_dy = {3: -14, 4: -24, 5: -34}
    for i, r in enumerate(ranks):
        sv = data[r]["savings_desc"]
        total = sum(sv) or 1
        cum, run = [], 0
        for s in sv:
            run += s
            cum.append(run / total)
        ks = list(range(1, len(cum) + 1))
        ax.plot(ks, cum, "-", color=OKABE_ITO[i], lw=2, label=RANK_LABEL[r])
        knee = next((k for k, c in zip(ks, cum) if c >= 0.99), ks[-1])
        kx = min(label_k.get(r, knee), ks[-1])
        cy = dict(zip(ks, cum))[kx]
        ax.annotate(f"{RANK_LABEL[r]} (99% @ k={knee})", (kx, cy),
                    textcoords="offset points", xytext=(4, label_dy.get(r, -18)),
                    fontsize=8.5, color=OKABE_ITO[i], fontweight="bold")
    ax.axhline(0.99, color=MUTED, lw=0.8, ls=":")
    ax.annotate("99%", (max(len(data[r]["savings_desc"]) for r in ranks), 0.99),
                textcoords="offset points", xytext=(-24, 4),
                fontsize=8, color=MUTED)
    ax.set_title("5. Coverage of top-k operators", fontsize=11,
                 fontweight="bold", loc="left", color=INK)
    ax.set_xlabel("k (operators kept, savings-ranked)")
    ax.set_ylabel("cumulative savings fraction")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(1, max(len(data[r]["savings_desc"]) for r in ranks))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 6. caption panel (no plot) — what the figures say
    ax = axes[1][2]
    ax.axis("off")
    lines = [
        "Reading the hierarchy",
        "",
        "• Operators grow modestly with rank; each rank adds",
        "  only its own V·Tn family (rank-locality theorem).",
        "• Savings and footprint both explode by ORDERS of",
        "  magnitude per rank — the high-rank blocks (o^a v^b)",
        "  dominate FLOPs and memory alike.",
        "• A few operators carry almost all savings (panel 5):",
        "  the top-k knee stays small even as the set grows,",
        "  which is why a memory budget can inline the tail",
        "  nearly for free.",
        "• The savings/footprint tension (memory investigation)",
        "  widens with rank: the biggest-savings operators are",
        "  also the biggest tensors.",
    ]
    ax.text(0.0, 0.98, lines[0], fontsize=11, fontweight="bold",
            va="top", color=INK, transform=ax.transAxes)
    ax.text(0.0, 0.90, "\n".join(lines[1:]), fontsize=9.5, va="top",
            color=INK, transform=ax.transAxes, linespacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, format="svg", bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, format="png", dpi=130, bbox_inches="tight")
    sys.stderr.write(f"wrote {out} and {png}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("optimizer_hierarchy.svg"))
    ap.add_argument("--sizes", type=int, nargs=2, metavar=("O", "V"),
                    default=(30, 100), help="occupied/virtual sizes (default 30 100)")
    ap.add_argument("--force", action="store_true", help="recompute, ignore cache")
    args = ap.parse_args()
    data = load(args.sizes[0], args.sizes[1], args.force)
    make_figure(data, args.out, args.sizes[0], args.sizes[1])


if __name__ == "__main__":
    main()
