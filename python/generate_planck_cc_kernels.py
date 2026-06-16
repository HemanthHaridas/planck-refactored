#!/usr/bin/env python3
"""Generate Planck-compatible coupled-cluster kernel source files."""

from __future__ import annotations

import argparse
from pathlib import Path

from ccgen.generate import print_cpp_planck


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Planck-compatible CC kernel translation units.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated .cpp files will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ccsd", "ccsdt"],
        help="CC methods to emit (default: ccsd ccsdt).",
    )
    parser.add_argument(
        "--include-intermediates",
        action="store_true",
        help="Also emit supported intermediate builders.",
    )
    parser.add_argument(
        "--intermediate-threshold",
        type=int,
        default=5,
        help="Min usage count for extracted intermediates.",
    )
    parser.add_argument(
        "--intermediate-memory-budget-mb",
        type=int,
        default=None,
        help="Optional cumulative memory budget in MB for emitted intermediates.",
    )
    parser.add_argument(
        "--intermediate-peak-memory-budget-mb",
        type=int,
        default=None,
        help="Optional per-target live memory budget in MB for emitted intermediates.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        code = print_cpp_planck(
            method.lower(),
            include_intermediates=args.include_intermediates,
            intermediate_threshold=args.intermediate_threshold,
            intermediate_memory_budget_bytes=(
                None
                if args.intermediate_memory_budget_mb is None
                else args.intermediate_memory_budget_mb * 1024 * 1024
            ),
            intermediate_peak_memory_budget_bytes=(
                None
                if args.intermediate_peak_memory_budget_mb is None
                else args.intermediate_peak_memory_budget_mb * 1024 * 1024
            ),
        )
        out_path = output_dir / f"{method.lower()}_planck_generated.cpp"
        out_path.write_text(code + "\n", encoding="utf-8")
        print(out_path)


if __name__ == "__main__":
    main()
