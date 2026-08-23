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
        "--spin-adapt",
        action="store_true",
        help="Emit genuine spatial (restricted) RCC kernels via the R1.0 "
             "spin-adaptation instead of raw spin-orbital algebra bound to "
             "spatial storage. Without this the emitted energy carries the "
             "0.25*t2*oovv defect (spin-orbital 1/4 with no spin sum), which "
             "drives the correlation energy ~4x wrong. Applies to BOTH the "
             "tensor-backend and arbitrary-order TUs this script emits; it does "
             "NOT touch the warm-start .inc (emitted elsewhere, correct on a "
             "spin-orbital reference). Default off for byte-compatibility with "
             "the historical (defective) emit.",
    )
    parser.add_argument(
        "--ucc",
        action="store_true",
        help="Emit UNRESTRICTED (spin-block resolved) CC kernels: one residual "
             "per stored block (doubles_aaaa / doubles_abab / doubles_bbbb) "
             "instead of one per rank. Mutually exclusive with --spin-adapt, "
             "which collapses the same blocks into a single spatial tensor -- "
             "the opposite direction. Every UCC target is block-tagged, so the "
             "emitted bundle carries no per-rank reference residual: an "
             "all-sectors bundle, which the runtime accepts as of U4.0. Requires "
             "a UHF reference at run time. Default off, so the default build is "
             "byte-identical.",
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
        "--dress-operators",
        action="store_true",
        help="Rewrite the residual to reference the recognized CC intermediates "
             "(Wmnij/Wabef/Wmbej + tau/tau_c) and emit their build_<name> "
             "functions. Requires the diagram engine + canonical Fock (both "
             "forced). Mutually exclusive with --factorize-tau (dressing already "
             "recognizes tau) and forces intermediates off (CSE and dressing "
             "share the same emission channel). Adds ~9s at rank 3 and ~62s at "
             "rank 4 to generation.",
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

    if args.dress_operators and args.factorize_tau:
        parser.error(
            "--dress-operators and --factorize-tau are mutually exclusive: dressing "
            "already recognizes tau/tau_c, so factorize_tau would materialize it twice")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit(method: str, *, force_arbitrary: bool) -> str:
        # The arbitrary companion is co-included with the ccsdtq TU in the
        # kernel registry; its shape-named intermediate builders (build_W_oo_3,
        # ...) carry no method suffix and would collide there. Emit it WITHOUT
        # intermediates so the residual is self-contained (no build_W_* symbols).
        include_intermediates = args.include_intermediates and not force_arbitrary
        # Dressing DOES apply to the arbitrary-order TUs, and must: those are the ones the
        # kernel registry actually executes (rank >= 4, or rank 3 under
        # -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON). The plain per-method TUs are compiled but
        # their residual kernels have no caller -- `generated_kernel_registry.cpp` says so
        # outright: "rank 2 and 3 use the hand-written backends".
        #
        # V1.3.1 suppressed dressing here, because two dressed arbitrary-order TUs shared
        # unsuffixed builder names and collided when co-included in the registry (5
        # redefinitions). V1.3.2 removed that collision by method-suffixing every builder
        # (`build_tau_ccsdtq`), so the suppression now only prevents dressing from reaching
        # the one path that runs. Gated by test_dressed_tu_coinclusion.
        dress_operators = args.dress_operators
        # `dress_operators` supersedes tau collapse and forces CSE off inside
        # print_cpp_planck; passing factorize_tau alongside it raises, so drop it here
        # (the CLI already rejects the combination outright).
        return print_cpp_planck(
            method.lower(),
            engine=args.engine,
            include_intermediates=include_intermediates,
            factorize_tau=args.factorize_tau and not dress_operators,
            dress_operators=dress_operators,
            spin_adapt=args.spin_adapt,
            ucc=args.ucc,
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

    # U5.0: a UCC run writes its own file. Without this it would OVERWRITE the RCC
    # TU for the same method -- the two are separate translation units, not
    # variants of one, and both are meant to be linkable into the same binary.
    suffix = "_ucc_planck_generated.cpp" if args.ucc else "_planck_generated.cpp"

    for method in args.methods:
        code = emit(method, force_arbitrary=False)
        out_path = output_dir / f"{method.lower()}{suffix}"
        out_path.write_text(code + "\n", encoding="utf-8")
        print(out_path)

        # Lower-rank arbitrary-order companion TUs (spatial RCC amplitude type),
        # for cross-rank .ccamp restart / the generated runtime. Rank >= 4 methods
        # are already arbitrary-order, so no companion is needed there.
        from ccgen.cluster import parse_cc_level  # local: avoid import cost when unused
        if args.arbitrary_lower_ranks and max(parse_cc_level(method.lower()), default=0) < 4:
            arb_code = emit(method, force_arbitrary=True)
            arb_path = output_dir / (
                f"{method.lower()}_arbitrary"
                + ("_ucc" if args.ucc else "")
                + "_planck_generated.cpp")
            arb_path.write_text(arb_code + "\n", encoding="utf-8")
            print(arb_path)


if __name__ == "__main__":
    main()
