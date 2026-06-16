#!/usr/bin/env python3
"""
Generate coupled-cluster residual equations and emit code.

Usage:
    python generate_ccsdt_cpp.py ccsdt                       # naive C++ to stdout
    python generate_ccsdt_cpp.py ccsd -o ccsd.cpp             # CCSD C++ to file
    python generate_ccsdt_cpp.py ccsdt --pretty               # human-readable form
    python generate_ccsdt_cpp.py ccd --einsum                 # numpy einsum form
    python generate_ccsdt_cpp.py ccsd --ir                    # tensor contraction IR
    python generate_ccsdt_cpp.py ccsd --ir-ex                 # extended IR (BLAS hints)

    # Optimized kernels (Phase 3):
    python generate_ccsdt_cpp.py ccsd --tiled                 # tiled + OpenMP C++
    python generate_ccsdt_cpp.py ccsd --blas                  # BLAS/GEMM-lowered C++
    python generate_ccsdt_cpp.py ccsd --pretty-full           # with intermediates + legend
    python generate_ccsdt_cpp.py ccsd --einsum --opt-einsum   # opt_einsum contraction paths

    # Pipeline options (combinable with any emitter):
    python generate_ccsdt_cpp.py ccsd --tiled --collect-denominators --debug
    python generate_ccsdt_cpp.py ccsdt --blas --tile-occ 32 --tile-vir 24 --no-openmp
"""

from __future__ import annotations

import argparse
import sys

from ccgen.generate import (
    generate_cc_contractions,
    generate_cc_contractions_ex,
    print_equations,
    print_equations_full,
    print_einsum,
    print_cpp,
    print_cpp_planck,
    print_cpp_optimized,
    print_cpp_blas,
    last_stats,
)
from ccgen.tensor_ir import BackendTerm, BackendTermEx, IndexContraction


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


def format_ir_ex(equations: dict[str, list[BackendTermEx]]) -> str:
    """Format extended contraction IR with BLAS hints and FLOP estimates."""
    blocks: list[str] = []
    for target, terms in equations.items():
        total_flops = sum(t.estimated_flops for t in terms)
        gemm_count = sum(1 for t in terms if t.blas_hint is not None)
        lines = [
            f"# {target} ({len(terms)} terms, ~{total_flops} FLOPs, "
            f"{gemm_count} GEMM-eligible)"
        ]
        for i, t in enumerate(terms, 1):
            lhs_idx = ",".join(idx.name for idx in t.lhs_indices)
            rhs = " * ".join(
                f"{f.name}({','.join(idx.name for idx in f.indices)})"
                for f in t.rhs_factors
            )
            lines.append(f"  Term {i}: {t.lhs_name}({lhs_idx}) += "
                         f"{t.coefficient} {rhs}")
            lines.append(f"    FLOPs: {t.estimated_flops}")
            if t.blas_hint:
                h = t.blas_hint
                k_str = ",".join(idx.name for idx in h.contraction_indices)
                m_str = ",".join(idx.name for idx in h.m_indices)
                n_str = ",".join(idx.name for idx in h.n_indices)
                lines.append(
                    f"    BLAS: {h.pattern}  "
                    f"A={h.a_tensor} B={h.b_tensor}  "
                    f"m=[{m_str}] n=[{n_str}] k=[{k_str}]"
                )
            if t.computation_order:
                lines.append(
                    f"    contraction order: {t.computation_order}"
                )
            if t.memory_layout:
                lines.append(
                    f"    memory layout: {t.memory_layout}"
                )
            if t.blocking_hint:
                lines.append(
                    f"    blocking: {t.blocking_hint}"
                )
            if t.reuse_key:
                lines.append(
                    f"    reuse key: {t.reuse_key}"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate coupled-cluster equations and emit code.",
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

    # ── Emission format ───────────────────────────────────────────
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument(
        "--cpp",
        action="store_true",
        default=True,
        help="Emit naive C++ loop nests (default).",
    )
    fmt.add_argument(
        "--tiled",
        action="store_true",
        help="Emit tiled C++ loop nests with OpenMP.",
    )
    fmt.add_argument(
        "--blas",
        action="store_true",
        help="Emit BLAS/GEMM-lowered C++ (with tiled fallback).",
    )
    fmt.add_argument(
        "--planck",
        action="store_true",
        help="Emit Planck-compatible C++ tensor kernels.",
    )
    fmt.add_argument(
        "--pretty",
        action="store_true",
        help="Emit human-readable symbolic equations.",
    )
    fmt.add_argument(
        "--pretty-full",
        action="store_true",
        help="Emit equations with intermediates, legend, and stats.",
    )
    fmt.add_argument(
        "--einsum",
        action="store_true",
        help="Emit numpy einsum code.",
    )
    fmt.add_argument(
        "--ir",
        action="store_true",
        help="Emit basic tensor contraction IR.",
    )
    fmt.add_argument(
        "--ir-ex",
        action="store_true",
        help="Emit extended IR with BLAS hints and FLOP estimates.",
    )

    # ── Tiling / OpenMP options ───────────────────────────────────
    parser.add_argument(
        "--tile-occ",
        type=int,
        default=16,
        help="Tile size for occupied indices (default: 16).",
    )
    parser.add_argument(
        "--tile-vir",
        type=int,
        default=16,
        help="Tile size for virtual indices (default: 16).",
    )
    parser.add_argument(
        "--no-openmp",
        action="store_true",
        help="Disable OpenMP pragmas in tiled/BLAS output.",
    )

    # ── Pipeline optimization flags ───────────────────────────────
    parser.add_argument(
        "--collect-denominators",
        action="store_true",
        help="Collect diagonal Fock terms into orbital energy denominators.",
    )
    parser.add_argument(
        "--permutation-grouping",
        action="store_true",
        help="Merge terms related by index permutations.",
    )
    parser.add_argument(
        "--exploit-symmetry",
        action="store_true",
        help="Exploit implicit antisymmetry (experimental).",
    )
    parser.add_argument(
        "--opt-einsum",
        action="store_true",
        help="Use opt_einsum for contraction path optimization (einsum mode).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print pipeline statistics to stderr.",
    )

    # ── Pretty-full options ───────────────────────────────────────
    parser.add_argument(
        "--intermediate-threshold",
        type=int,
        default=5,
        help="Min usage count to extract an intermediate (default: 5).",
    )

    args = parser.parse_args()

    method = args.method.lower()
    use_openmp = not args.no_openmp

    # Common kwargs passed through to generate_cc_equations
    gen_kwargs = dict(
        collect_denominators=args.collect_denominators,
        permutation_grouping=args.permutation_grouping,
        exploit_symmetry=args.exploit_symmetry,
        debug=args.debug,
    )

    print(f"Generating {method.upper()} equations...", file=sys.stderr)

    if args.planck:
        result = print_cpp_planck(method, **gen_kwargs)
    elif args.pretty:
        result = print_equations(method, **gen_kwargs)
    elif args.pretty_full:
        result = print_equations_full(
            method,
            intermediate_threshold=args.intermediate_threshold,
            **gen_kwargs,
        )
    elif args.einsum:
        result = print_einsum(
            method,
            use_opt_einsum=args.opt_einsum,
            **gen_kwargs,
        )
    elif args.ir:
        result = format_ir(generate_cc_contractions(method))
    elif args.ir_ex:
        result = format_ir_ex(
            generate_cc_contractions_ex(
                method,
                tile_occ=args.tile_occ,
                tile_vir=args.tile_vir,
                **gen_kwargs,
            )
        )
    elif args.tiled:
        result = print_cpp_optimized(
            method,
            tile_occ=args.tile_occ,
            tile_vir=args.tile_vir,
            use_openmp=use_openmp,
            **gen_kwargs,
        )
    elif args.blas:
        result = print_cpp_blas(
            method,
            tile_occ=args.tile_occ,
            tile_vir=args.tile_vir,
            use_openmp=use_openmp,
            **gen_kwargs,
        )
    else:
        result = print_cpp(method, **gen_kwargs)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
            f.write("\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result)

    # Print pipeline stats if debug mode was used
    from ccgen.generate import last_stats as final_stats
    if args.debug and final_stats is not None:
        print("\n" + final_stats.summary(), file=sys.stderr)


if __name__ == "__main__":
    main()
