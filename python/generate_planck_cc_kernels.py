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
        "--factorize-tau",
        action="store_true",
        help="Collapse validated t2 + t1t1 pairs into the tau pseudo-amplitude "
             "(experimental; default off).",
    )
    parser.add_argument(
        "--engine",
        choices=["diagram", "wick"],
        default="diagram",
        help="Equation-generation engine. 'diagram' is canonical-by-construction "
             "and ~200x faster at high rank (CCSDTQ ~3s vs ~600s), residual-equal "
             "to 'wick'. Default: diagram.",
    )
    parser.add_argument(
        "--intermediate-threshold",
        type=int,
        default=5,
        help="Min usage count for extracted intermediates.",
    )
    parser.add_argument(
        "--arbitrary-lower-ranks",
        action="store_true",
        help="Additionally emit each rank<4 method (ccsd/ccsdt) in "
             "arbitrary-order (spatial ArbitraryOrderRCCAmplitudes) form as "
             "<method>_arbitrary_planck_generated.cpp, so the generated runtime "
             "and .ccamp restart can consume them as spatial seed sources. The "
             "plain <method>_planck_generated.cpp (tensor_backend types) is still "
             "emitted unchanged.",
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

    def emit(method: str, *, force_arbitrary: bool) -> str:
        # The arbitrary companion is co-included with the ccsdtq TU in the
        # kernel registry; its shape-named intermediate builders (build_W_oo_3,
        # ...) carry no method suffix and would collide there. Emit it WITHOUT
        # intermediates so the residual is self-contained (no build_W_* symbols).
        include_intermediates = args.include_intermediates and not force_arbitrary
        return print_cpp_planck(
            method.lower(),
            engine=args.engine,
            include_intermediates=include_intermediates,
            factorize_tau=args.factorize_tau,
            force_arbitrary=force_arbitrary,
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

    for method in args.methods:
        code = emit(method, force_arbitrary=False)
        out_path = output_dir / f"{method.lower()}_planck_generated.cpp"
        out_path.write_text(code + "\n", encoding="utf-8")
        print(out_path)

        # Lower-rank arbitrary-order companion TUs (spatial RCC amplitude type),
        # for cross-rank .ccamp restart / the generated runtime. Rank >= 4 methods
        # are already arbitrary-order, so no companion is needed there.
        from ccgen.cluster import parse_cc_level  # local: avoid import cost when unused
        if args.arbitrary_lower_ranks and max(parse_cc_level(method.lower()), default=0) < 4:
            arb_code = emit(method, force_arbitrary=True)
            arb_path = output_dir / f"{method.lower()}_arbitrary_planck_generated.cpp"
            arb_path.write_text(arb_code + "\n", encoding="utf-8")
            print(arb_path)


if __name__ == "__main__":
    main()
