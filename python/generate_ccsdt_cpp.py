#!/usr/bin/env python3
"""
Sample script: generate coupled-cluster residual equations and emit code.

Usage:
    python generate_ccsdt_cpp.py ccsdt                  # CCSDT C++ to stdout
    python generate_ccsdt_cpp.py ccsd -o ccsd.cpp        # CCSD C++ to file
    python generate_ccsdt_cpp.py ccsdt --pretty          # human-readable form
    python generate_ccsdt_cpp.py ccd --einsum            # numpy einsum form
    python generate_ccsdt_cpp.py ccsd --ir               # tensor contraction IR
"""

from __future__ import annotations

import argparse
import sys

from ccgen.generate import (
    generate_cc_contractions,
    print_equations,
    print_einsum,
    print_cpp,
)
from ccgen.tensor_ir import BackendTerm, IndexContraction


def _format_contraction(c: IndexContraction) -> str:
    kind = "free" if c.is_free else "summed"
    slots = ", ".join(
        f"{s.tensor_name}[axis={s.axis}]" for s in c.slots
    )
    return f"  {c.index.name}({c.index.space}, {kind}): {slots}"


def format_ir(equations: dict[str, list[BackendTerm]]) -> str:
    """Format lowered contraction IR as a readable dump."""
    blocks: list[str] = []
    for target, terms in equations.items():
        lines = [f"# {target} ({len(terms)} terms)"]
        for i, t in enumerate(terms, 1):
            lhs_idx = ",".join(idx.name for idx in t.lhs_indices)
            rhs = " * ".join(
                f"{f.name}({','.join(idx.name for idx in f.indices)})"
                for f in t.rhs_factors
            )
            lines.append(f"  Term {i}: {t.lhs_name}({lhs_idx}) += "
                         f"{t.coefficient} {rhs}")
            lines.append(f"    connected: {t.connected}")
            lines.append(f"    free:   [{', '.join(idx.name for idx in t.free_indices)}]")
            lines.append(f"    summed: [{', '.join(idx.name for idx in t.summed_indices)}]")
            lines.append(f"    contractions:")
            for c in t.contractions:
                lines.append(f"    {_format_contraction(c)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate coupled-cluster equations and emit C++ code.",
    )
    parser.add_argument(
        "method",
        type=str,
        help="CC method to generate (e.g. ccd, ccsd, ccsdt).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Write output to file instead of stdout.",
    )
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument(
        "--cpp",
        action="store_true",
        default=True,
        help="Emit C++ loop nests (default).",
    )
    fmt.add_argument(
        "--pretty",
        action="store_true",
        help="Emit human-readable symbolic equations.",
    )
    fmt.add_argument(
        "--einsum",
        action="store_true",
        help="Emit numpy einsum code.",
    )
    fmt.add_argument(
        "--ir",
        action="store_true",
        help="Emit tensor contraction IR (intermediate representation).",
    )
    args = parser.parse_args()

    method = args.method.lower()

    print(f"Generating {method.upper()} equations...", file=sys.stderr)

    if args.pretty:
        result = print_equations(method)
    elif args.einsum:
        result = print_einsum(method)
    elif args.ir:
        result = format_ir(generate_cc_contractions(method))
    else:
        result = print_cpp(method)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
            f.write("\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
