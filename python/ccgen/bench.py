"""Performance benchmarking for ccgen equation generation.

Measures generation time for CCD, CCSD, and CCSDT with and without
optimizations, and reports term counts and speedups.

Usage::

    python -m ccgen.bench
    python -m ccgen.bench --methods ccsd ccsdt
    python -m ccgen.bench --methods ccsd --timing --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Sequence

from .generate import generate_cc_equations
from .canonicalize import collect_fock_diagonals


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    method: str
    generation_time_s: float
    term_counts: dict[str, int]
    denom_collected_counts: dict[str, int] | None = None
    denom_collection_time_s: float | None = None


@dataclass
class BenchmarkSuite:
    """Results from a complete benchmark sweep."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["ccgen Performance Benchmark", "=" * 60]

        for r in self.results:
            lines.append("")
            lines.append(f"Method: {r.method.upper()}")
            lines.append(f"  Generation time: {r.generation_time_s:.3f}s")
            lines.append("  Term counts:")
            for manifold, count in r.term_counts.items():
                line = f"    {manifold:>12s}: {count:>6d}"
                if (r.denom_collected_counts
                        and manifold in r.denom_collected_counts):
                    dc = r.denom_collected_counts[manifold]
                    reduction = count - dc
                    pct = 100 * reduction / count if count else 0
                    line += f"  → {dc:>6d} after denom collection"
                    line += f" (-{reduction}, {pct:.1f}%)"
                lines.append(line)
            if r.denom_collection_time_s is not None:
                lines.append(
                    f"  Denom collection time: "
                    f"{r.denom_collection_time_s:.3f}s"
                )

        # Speedup comparison if multiple methods
        if len(self.results) >= 2:
            lines.append("")
            lines.append("Relative generation times:")
            base = self.results[0]
            for r in self.results[1:]:
                ratio = r.generation_time_s / base.generation_time_s
                lines.append(
                    f"  {r.method.upper()} / {base.method.upper()}"
                    f" = {ratio:.1f}x"
                )

        total = sum(r.generation_time_s for r in self.results)
        lines.append("")
        lines.append(f"Total benchmark time: {total:.2f}s")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            [asdict(r) for r in self.results],
            indent=2,
        )


def benchmark_method(
    method: str,
    include_denom: bool = True,
) -> BenchmarkResult:
    """Benchmark equation generation for a single CC method."""
    t0 = time.monotonic()
    eqs = generate_cc_equations(method)
    gen_time = time.monotonic() - t0

    term_counts = {k: len(v) for k, v in eqs.items()}

    denom_counts = None
    denom_time = None
    if include_denom:
        t0 = time.monotonic()
        denom_counts = {}
        for manifold, terms in eqs.items():
            if manifold in ("energy", "reference"):
                denom_counts[manifold] = len(terms)
            else:
                collected = collect_fock_diagonals(terms)
                denom_counts[manifold] = len(collected)
        denom_time = time.monotonic() - t0

    return BenchmarkResult(
        method=method,
        generation_time_s=gen_time,
        term_counts=term_counts,
        denom_collected_counts=denom_counts,
        denom_collection_time_s=denom_time,
    )


def run_benchmark(
    methods: Sequence[str] = ("ccd", "ccsd"),
    include_denom: bool = True,
) -> BenchmarkSuite:
    """Run benchmarks for a list of CC methods."""
    suite = BenchmarkSuite()
    for method in methods:
        result = benchmark_method(method, include_denom=include_denom)
        suite.results.append(result)
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ccgen equation generation",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ccd", "ccsd"],
        help="CC methods to benchmark (default: ccd ccsd)",
    )
    parser.add_argument(
        "--no-denom",
        action="store_true",
        help="Skip denominator collection benchmarks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()
    suite = run_benchmark(
        methods=args.methods,
        include_denom=not args.no_denom,
    )

    if args.json:
        print(suite.to_json())
    else:
        print(suite.summary())


if __name__ == "__main__":
    main()
