"""Top-level CC equation generation API."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .hamiltonian import build_hamiltonian
from .cluster import build_cluster, parse_cc_level
from .algebra import bch_expand
from .project import project, AlgebraTerm, manifold_name
from .canonicalize import canonicalize_term, merge_like_terms, collect_fock_diagonals
from .emit.pretty import format_equations, format_equations_with_intermediates
from .emit.einsum import format_equations_einsum
from .emit.cpp_loops import (
    emit_translation_unit,
    emit_optimized_translation_unit,
    emit_blas_translation_unit,
)
from .emit.planck_tensor_cpp import emit_planck_translation_unit
from .lowering import lower_equations_restricted_closed_shell, RestrictedClosedShellTerm
from .tensor_ir import lower_equations, lower_equations_ex, BackendTerm, BackendTermEx


@dataclass
class PipelineStats:
    """Term count and timing statistics from a generation run."""

    method: str = ""
    bch_terms: int = 0
    bch_time_s: float = 0.0
    manifolds: dict[str, dict[str, int | float]] = field(
        default_factory=dict,
    )

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline statistics."""
        lines = [
            f"{self.method.upper()} generation:",
            f"  After BCH expansion:     {self.bch_terms:>6d} terms"
            f"  ({self.bch_time_s:.2f}s)",
        ]
        for name, stats in self.manifolds.items():
            lines.append(f"  {name}:")
            lines.append(
                f"    After Wick projection: {stats['after_projection']:>6d} terms"
                f"  ({stats['projection_time_s']:.2f}s)"
            )
            if stats.get("after_connected_filter", None) is not None:
                lines.append(
                    f"    After connected filter:{stats['after_connected_filter']:>6d} terms"
                )
            lines.append(
                f"    After canonicalization: {stats['after_canonicalization']:>6d} terms"
                f"  ({stats['canonicalization_time_s']:.2f}s)"
            )
            lines.append(
                f"    After merge_like_terms:{stats['after_merge']:>6d} terms"
            )
            if stats.get("after_denom_collection") is not None:
                lines.append(
                    f"    After denom collection:{stats['after_denom_collection']:>6d} terms"
                )
            if stats.get("after_perm_grouping") is not None:
                lines.append(
                    f"    After perm grouping:   {stats['after_perm_grouping']:>6d} terms"
                )
            if stats.get("after_symmetry") is not None:
                lines.append(
                    f"    After symmetry exploit:{stats['after_symmetry']:>6d} terms"
                )
        return "\n".join(lines)


# Module-level storage for the most recent pipeline stats.
last_stats: PipelineStats | None = None


def targets_for_method(method: str) -> list[str]:
    """Derive the default projection targets for a CC method.

    Returns ``["energy"]`` plus the manifold name for each
    excitation rank present in *method*.
    """
    ranks = parse_cc_level(method)
    return ["energy"] + [manifold_name(r) for r in ranks]


def generate_cc_equations(
    method: str,
    targets: list[str] | None = None,
    connected_only: bool = True,
    collect_denominators: bool = False,
    permutation_grouping: bool = False,
    exploit_symmetry: bool = False,
    debug: bool = False,
) -> dict[str, list[AlgebraTerm]]:
    """Generate coupled-cluster equations for a given method.

    Parameters
    ----------
    collect_denominators : bool
        If True, apply orbital energy denominator collection as a
        post-canonicalization pass.  Diagonal Fock terms like
        ``-f(i,i)*t2 + f(a,a)*t2`` are collected into denominator
        tensors ``D(i,a)*t2``.
    permutation_grouping : bool
        If True, apply permutation-based term grouping to merge
        terms related by index permutations (Phase 3).
    exploit_symmetry : bool
        If True, apply implicit antisymmetry exploitation to reduce
        term count (Phase 3, experimental).
    debug : bool
        If True, print term counts and timing at each pipeline stage
        to stderr.  Statistics are also stored in ``ccgen.generate.last_stats``.
    """
    global last_stats

    method = method.lower()
    if targets is None:
        targets = targets_for_method(method)

    H = build_hamiltonian()
    T = build_cluster(method)

    t0 = time.monotonic()
    Hbar = bch_expand(H, T, max_order=4)
    bch_time = time.monotonic() - t0

    stats = PipelineStats(
        method=method,
        bch_terms=len(Hbar.terms),
        bch_time_s=bch_time,
    )

    equations: dict[str, list[AlgebraTerm]] = {}

    for manifold in targets:
        mstats: dict[str, int | float] = {}

        t0 = time.monotonic()
        raw = project(Hbar, manifold)
        mstats["projection_time_s"] = time.monotonic() - t0
        mstats["after_projection"] = len(raw)

        if connected_only and manifold not in ("energy", "reference"):
            raw = [t for t in raw if t.connected]
            mstats["after_connected_filter"] = len(raw)
        else:
            mstats["after_connected_filter"] = None  # type: ignore[assignment]

        t0 = time.monotonic()
        canonicalized = [canonicalize_term(t) for t in raw]
        mstats["canonicalization_time_s"] = time.monotonic() - t0
        mstats["after_canonicalization"] = len(canonicalized)

        canonical = merge_like_terms(canonicalized)
        mstats["after_merge"] = len(canonical)

        if collect_denominators and manifold not in ("energy", "reference"):
            canonical = collect_fock_diagonals(canonical)
            mstats["after_denom_collection"] = len(canonical)

        if permutation_grouping and manifold not in ("energy", "reference"):
            from .optimization.permutation import apply_permutation_grouping
            canonical = apply_permutation_grouping(canonical)
            mstats["after_perm_grouping"] = len(canonical)

        if exploit_symmetry and manifold not in ("energy", "reference"):
            from .optimization.symmetry import exploit_antisymmetry
            canonical = exploit_antisymmetry(canonical)
            mstats["after_symmetry"] = len(canonical)

        stats.manifolds[manifold] = mstats
        equations[manifold] = canonical

    last_stats = stats

    if debug:
        print(stats.summary(), file=sys.stderr)

    return equations


def print_equations(method: str, **kwargs: Any) -> str:
    """Generate and format equations as readable text."""
    eqs = generate_cc_equations(method, **kwargs)
    return format_equations(eqs)


def generate_cc_contractions(
    method: str,
    targets: list[str] | None = None,
    connected_only: bool = True,
) -> dict[str, list[BackendTerm]]:
    """Generate coupled-cluster equations lowered to contraction IR."""
    eqs = generate_cc_equations(
        method,
        targets=targets,
        connected_only=connected_only,
    )
    return lower_equations(eqs)


def generate_cc_equations_lowered(
    method: str,
    orbital_model: str = "restricted_closed_shell",
    **kwargs: Any,
) -> dict[str, list[RestrictedClosedShellTerm]]:
    """Generate equations lowered to a backend-oriented orbital model.

    The current lowering target is ``restricted_closed_shell``, which preserves
    the original algebra terms and annotates each tensor factor with explicit
    spatial-orbital layout metadata for restricted backends.
    """
    eqs = generate_cc_equations(method, **kwargs)
    if orbital_model != "restricted_closed_shell":
        raise ValueError(
            "Unsupported orbital model "
            f"{orbital_model!r}; expected 'restricted_closed_shell'"
        )
    return lower_equations_restricted_closed_shell(eqs)


def generate_cc_contractions_ex(
    method: str,
    targets: list[str] | None = None,
    connected_only: bool = True,
    detect_blas: bool = True,
    tile_occ: int = 16,
    tile_vir: int = 16,
    **kwargs: Any,
) -> dict[str, list[BackendTermEx]]:
    """Generate equations lowered to extended contraction IR with hints.

    Returns ``BackendTermEx`` objects with BLAS hints, FLOP estimates,
    optimal contraction ordering, and blocking hints.
    """
    eqs = generate_cc_equations(
        method,
        targets=targets,
        connected_only=connected_only,
        **kwargs,
    )
    return lower_equations_ex(
        eqs,
        detect_blas=detect_blas,
        tile_occ=tile_occ,
        tile_vir=tile_vir,
    )


def print_einsum(
    method: str,
    use_opt_einsum: bool = False,
    **kwargs: Any,
) -> str:
    """Generate and format equations as numpy einsum code.

    Parameters
    ----------
    use_opt_einsum : bool
        If True and opt_einsum is installed, emit optimized
        contraction paths via ``opt_einsum.contract()``.
    """
    eqs = generate_cc_equations(method, **kwargs)
    return format_equations_einsum(eqs, use_opt_einsum=use_opt_einsum)


def print_cpp(method: str, **kwargs: Any) -> str:
    """Generate and format equations as C++ loop nests."""
    eqs = generate_cc_equations(method, **kwargs)
    return emit_translation_unit(eqs)


def print_cpp_optimized(
    method: str,
    tile_occ: int = 16,
    tile_vir: int = 16,
    use_openmp: bool = True,
    **kwargs: Any,
) -> str:
    """Generate and format equations as tiled C++ loop nests with OpenMP."""
    eqs = generate_cc_equations(method, **kwargs)
    return emit_optimized_translation_unit(
        eqs,
        tile_occ=tile_occ,
        tile_vir=tile_vir,
        use_openmp=use_openmp,
    )


def print_cpp_blas(
    method: str,
    tile_occ: int = 16,
    tile_vir: int = 16,
    use_openmp: bool = True,
    use_blas: bool = True,
    **kwargs: Any,
) -> str:
    """Generate equations as C++ with BLAS/GEMM lowering (Phase 3).

    Detects GEMM patterns in two-factor contractions and emits
    CBLAS dgemm calls.  Falls back to tiled loops for non-GEMM terms.
    """
    eqs = generate_cc_equations(method, **kwargs)
    return emit_blas_translation_unit(
        eqs,
        tile_occ=tile_occ,
        tile_vir=tile_vir,
        use_openmp=use_openmp,
        use_blas=use_blas,
    )


def print_cpp_planck(
    method: str,
    include_intermediates: bool = False,
    intermediate_threshold: int = 5,
    **kwargs: Any,
) -> str:
    """Generate Planck-compatible C++ tensor kernels."""
    eqs = generate_cc_equations(method, **kwargs)
    intermediates = None
    if include_intermediates:
        from .optimization.intermediates import (
            annotate_layout_hints,
            detect_intermediates,
            rewrite_equations,
        )
        detected = detect_intermediates(
            eqs,
            threshold=intermediate_threshold,
        )
        supported = [
            spec for spec in detected
            if spec.rank >= 0
        ]
        intermediates = annotate_layout_hints(supported) if supported else None
        if intermediates:
            eqs = rewrite_equations(eqs, list(intermediates))
    return emit_planck_translation_unit(
        method,
        eqs,
        intermediates=intermediates,
    )


def print_equations_full(
    method: str,
    include_intermediates: bool = True,
    intermediate_threshold: int = 5,
    include_legend: bool = True,
    include_stats: bool = True,
    annotate_layout: bool = False,
    **kwargs: Any,
) -> str:
    """Generate and format equations with full Phase 3 pretty printing.

    Includes: intermediate definitions, index/tensor legend,
    section headers, and summary statistics.
    """
    eqs = generate_cc_equations(method, **kwargs)

    intermediates = None
    if include_intermediates:
        from .optimization.intermediates import (
            detect_intermediates,
            annotate_layout_hints,
        )
        intms = detect_intermediates(eqs, threshold=intermediate_threshold)
        if annotate_layout:
            intms = annotate_layout_hints(intms)
        intermediates = intms if intms else None

    return format_equations_with_intermediates(
        eqs,
        intermediates=intermediates,
        include_legend=include_legend,
        include_stats=include_stats,
    )
