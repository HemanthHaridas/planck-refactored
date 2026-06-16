"""Command-line entry point for ccgen."""

from __future__ import annotations

import argparse
import json
import sys

from .generate import (
    generate_cc_equations,
    print_cpp,
    print_cpp_blas,
    print_cpp_optimized,
    print_cpp_planck,
    print_einsum,
    print_equations,
    print_equations_full,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccgen",
        description="Generate coupled-cluster equations and emitted kernels.",
    )
    parser.add_argument(
        "method",
        help="Coupled-cluster method string, e.g. ccd, ccsd, ccsdt, ccsdtq.",
    )
    parser.add_argument(
        "--format",
        choices=(
            "counts",
            "pretty",
            "pretty-full",
            "einsum",
            "cpp",
            "cpp-optimized",
            "cpp-blas",
            "cpp-planck",
        ),
        default="counts",
        help="Output format.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional subset of manifolds to generate, e.g. energy singles doubles.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Optional worker count for chunked projection/canonicalization.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory for per-manifold equation checkpoints.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print pipeline timing and term-count diagnostics to stderr.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="For format=counts, emit JSON instead of plain text.",
    )
    parser.add_argument(
        "--tile-occ",
        type=int,
        default=16,
        help="Tile size for occupied loops in optimized C++ emitters.",
    )
    parser.add_argument(
        "--tile-vir",
        type=int,
        default=16,
        help="Tile size for virtual loops in optimized C++ emitters.",
    )
    parser.add_argument(
        "--no-openmp",
        action="store_true",
        help="Disable OpenMP in optimized C++ emitters.",
    )
    parser.add_argument(
        "--no-blas",
        action="store_true",
        help="Disable BLAS lowering in the BLAS emitter.",
    )
    parser.add_argument(
        "--opt-einsum",
        action="store_true",
        help="Use opt_einsum in einsum emission if installed.",
    )
    parser.add_argument(
        "--include-intermediates",
        action="store_true",
        help="Emit extracted intermediates where supported.",
    )
    parser.add_argument(
        "--intermediate-threshold",
        type=int,
        default=5,
        help="Minimum usage count before extracting an intermediate.",
    )
    parser.add_argument(
        "--intermediate-memory-budget-mb",
        type=int,
        default=None,
        help="Optional cumulative memory budget in MB for materialized intermediates.",
    )
    parser.add_argument(
        "--intermediate-peak-memory-budget-mb",
        type=int,
        default=None,
        help="Optional per-target live memory budget in MB for materialized intermediates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    common_kwargs = {
        "targets": args.targets,
        "debug": args.debug,
        "parallel_workers": args.parallel_workers,
        "cache_dir": args.cache_dir,
    }
    intermediate_budget = (
        None
        if args.intermediate_memory_budget_mb is None
        else args.intermediate_memory_budget_mb * 1024 * 1024
    )
    intermediate_peak_budget = (
        None
        if args.intermediate_peak_memory_budget_mb is None
        else args.intermediate_peak_memory_budget_mb * 1024 * 1024
    )

    if args.format == "counts":
        eqs = generate_cc_equations(args.method, **common_kwargs)
        counts = {name: len(terms) for name, terms in eqs.items()}
        if args.json:
            print(json.dumps(counts, indent=2, sort_keys=True))
        else:
            for name, count in counts.items():
                print(f"{name}: {count}")
        return 0

    if args.format == "pretty":
        print(print_equations(args.method, **common_kwargs))
        return 0

    if args.format == "pretty-full":
        print(print_equations_full(
            args.method,
            include_intermediates=args.include_intermediates,
            intermediate_threshold=args.intermediate_threshold,
            intermediate_memory_budget_bytes=intermediate_budget,
            intermediate_peak_memory_budget_bytes=intermediate_peak_budget,
            **common_kwargs,
        ))
        return 0

    if args.format == "einsum":
        print(print_einsum(
            args.method,
            use_opt_einsum=args.opt_einsum,
            **common_kwargs,
        ))
        return 0

    if args.format == "cpp":
        print(print_cpp(args.method, **common_kwargs))
        return 0

    if args.format == "cpp-optimized":
        print(print_cpp_optimized(
            args.method,
            tile_occ=args.tile_occ,
            tile_vir=args.tile_vir,
            use_openmp=not args.no_openmp,
            **common_kwargs,
        ))
        return 0

    if args.format == "cpp-blas":
        print(print_cpp_blas(
            args.method,
            tile_occ=args.tile_occ,
            tile_vir=args.tile_vir,
            use_openmp=not args.no_openmp,
            use_blas=not args.no_blas,
            **common_kwargs,
        ))
        return 0

    if args.format == "cpp-planck":
        print(print_cpp_planck(
            args.method,
            include_intermediates=args.include_intermediates,
            intermediate_threshold=args.intermediate_threshold,
            intermediate_memory_budget_bytes=intermediate_budget,
            intermediate_peak_memory_budget_bytes=intermediate_peak_budget,
            **common_kwargs,
        ))
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
