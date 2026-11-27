#!/usr/bin/env python3
"""Generate Planck-compatible coupled-cluster kernel source files."""

from __future__ import annotations

import argparse
import sys
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
        "--ccsd-t",
        action="store_true",
        help="Write only ccsd_t_planck_generated.cpp, the restricted CCSD(T) W and "
             "V builders (docs/CCSD_T_SCOPE.md), and ignore every other option.",
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
    # W3.1: ONE dressing axis with a value, not a second boolean. ccgen has two
    # routes that derive dressed operators -- recognition (hand-seeded
    # fingerprints, retired) and derivation (from each term's contraction tree) --
    # and they are alternatives, never combined. A `--derive-operators` boolean
    # would make that a fourth dressing-ish axis on a function whose three
    # existing ones already need prose to explain, and every pairwise combination
    # becomes a question with no meaningful answer.
    parser.add_argument(
        "--dressing",
        choices=("none", "recognized", "derived"),
        default=None,
        help="Which dressed-operator route to emit. 'none' (default) is the "
             "undressed emit. 'recognized' rewrites the residual to reference the "
             "recognized CC intermediates (Wmnij/Wabef/Wmbej + tau/tau_c) and "
             "emits their build_<name> functions; it requires the diagram engine "
             "+ canonical Fock (both forced), is mutually exclusive with "
             "--factorize-tau (dressing already recognizes tau), forces "
             "intermediates off (CSE and dressing share the same emission "
             "channel), and adds ~9s at rank 3 / ~62s at rank 4. 'derived' is the "
             "factorizer route (not yet wired -- W3.2).",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        action="store_true",
        help="Emit the Λ (left-hand) residual kernels instead of the T ones, as "
             "<method>[_ucc]_lambda_planck_generated.cpp (docs/CC_LAMBDA_SCOPE.md, "
             "3b). Needs --spin-adapt or --ucc.",
    )
    parser.add_argument(
        "--dress-operators",
        action="store_true",
        help="Deprecated alias for --dressing recognized. Kept so existing "
             "invocations keep emitting the same kernels.",
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
    parser.add_argument(
        "--shards",
        nargs="*",
        default=[],
        metavar="METHOD=N",
        help="Write each method's TU as <stem>.h plus N sources <stem>_s<k>.cpp "
             "(default 1). The planck_cc_generated library compiles them separately, "
             "so no single compile holds a whole rank-4 file (ccgen/emit/shard.py).",
    )
    args = parser.parse_args()
    shard_counts = {}
    for item in args.shards:
        name, _, n = item.partition("=")
        if not n.isdigit() or int(n) < 1:
            parser.error(f"--shards expects METHOD=N with N >= 1, got {item!r}")
        shard_counts[name.lower()] = int(n)

    # Resolve the alias onto the one axis before anything reads it, so there is a
    # single source of truth for "which dressing".
    if args.dress_operators and args.dressing is not None:
        parser.error(
            "--dress-operators is an alias for --dressing recognized; pass one or "
            "the other, not both")
    dressing = args.dressing
    if dressing is None:
        dressing = "recognized" if args.dress_operators else "none"
    if args.dress_operators:
        print("[ccgen] --dress-operators is deprecated; use --dressing recognized",
              file=sys.stderr)
    if dressing != "none" and args.factorize_tau:
        parser.error(
            f"--dressing {dressing} and --factorize-tau are mutually exclusive: dressing "
            "already recognizes tau/tau_c, so factorize_tau would materialize it twice")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_tu(stem: str, method: str, code: str) -> None:
        """<stem>.h and <stem>_s<k>.cpp; prints each path and the largest shard."""
        from ccgen.emit.shard import shard_translation_unit
        n = shard_counts.get(method.lower(), 1)
        header, shards = shard_translation_unit(code, n, f"{stem}.h")
        (output_dir / f"{stem}.h").write_text(header, encoding="utf-8")
        for k, text in enumerate(shards):
            path = output_dir / f"{stem}_s{k}.cpp"
            path.write_text(text, encoding="utf-8")
            print(path)
        biggest = max(len(t) for t in shards) / 1e6
        # ponytail: shard counts are a CMake table (_planck_cc_shards); this
        # warning is how a new rank finds out its count is too low.
        if biggest > 4.0:
            print(f"[ccgen] warning: {stem} largest shard {biggest:.1f} MB; raise its "
                  "count in CMakeLists.txt _planck_cc_shards", file=sys.stderr)

    if args.ccsd_t:
        from ccgen.methods.ccsd_t import emit_planck_ccsd_t_translation_unit
        out_path = output_dir / "ccsd_t_planck_generated.cpp"
        out_path.write_text(emit_planck_ccsd_t_translation_unit() + "\n", encoding="utf-8")
        print(out_path)
        return

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
        # W3.2: pass the AXIS, not a boolean -- print_cpp_planck routes
        # `recognized` and `derived` to different operator sources but the same
        # downstream emit, so the caller does not need to know which.
        # Dressing supersedes tau collapse and forces CSE off inside
        # print_cpp_planck; passing factorize_tau alongside it raises, so drop it here
        # (the CLI already rejects the combination outright).
        return print_cpp_planck(
            method.lower(),
            engine=args.engine,
            include_intermediates=include_intermediates,
            factorize_tau=args.factorize_tau and dressing == "none",
            dressing=dressing,
            spin_adapt=args.spin_adapt,
            ucc=args.ucc,
            force_arbitrary=force_arbitrary,
            lambda_=args.lambda_,
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
    suffix = "_ucc_planck_generated" if args.ucc else "_planck_generated"
    if args.lambda_:
        # One file per method, always arbitrary-order: the Λ bundle has no
        # plain tensor_backend form.
        for method in args.methods:
            write_tu(f"{method.lower()}{'_ucc' if args.ucc else ''}_lambda_planck_generated",
                     method, emit(method, force_arbitrary=True))
        return

    for method in args.methods:
        write_tu(f"{method.lower()}{suffix}", method, emit(method, force_arbitrary=False))

        # Lower-rank arbitrary-order companion TUs (spatial RCC amplitude type),
        # for cross-rank .ccamp restart / the generated runtime. Rank >= 4 methods
        # are already arbitrary-order, so no companion is needed there.
        from ccgen.cluster import parse_cc_level  # local: avoid import cost when unused
        if args.arbitrary_lower_ranks and max(parse_cc_level(method.lower()), default=0) < 4:
            write_tu(f"{method.lower()}_arbitrary{'_ucc' if args.ucc else ''}_planck_generated",
                     method, emit(method, force_arbitrary=True))


if __name__ == "__main__":
    main()
