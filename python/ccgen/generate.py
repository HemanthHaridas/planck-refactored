"""Top-level CC equation generation API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .hamiltonian import build_hamiltonian
from .cluster import build_cluster, build_tn, canonicalize_cc_level, parse_cc_level
from .algebra import bch_expand_from_prefix_levels, bch_levels, bch_result_from_levels
from .expr import Expr
from .project import (
    iter_projected_terms_from_terms,
    bucket_terms_by_manifold,
    AlgebraTerm,
    manifold_name,
)
from .canonicalize import (
    canonicalize_term,
    merge_exact_term_into_buckets,
    merge_term_into_buckets,
    term_is_zero_before_canonicalization,
    collect_fock_diagonals,
)
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
    bch_cache_hit: bool = False
    bch_reused_from: str | None = None
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
        if self.bch_cache_hit:
            lines.append("  BCH cache:              exact cache hit")
        elif self.bch_reused_from is not None:
            lines.append(f"  BCH prefix reuse:       {self.bch_reused_from.upper()}")
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
            if stats.get("after_precanonical_prune") is not None:
                lines.append(
                    f"    After pre-canonical prune:{stats['after_precanonical_prune']:>6d} terms"
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

_PARALLEL_ENV_VAR = "CCGEN_PARALLEL_WORKERS"
_MIN_CHUNK_TERMS = 8
_CACHE_VERSION = 1
_BCH_CACHE_VERSION = 1
_BCH_MAX_ORDER = 4


def targets_for_method(method: str) -> list[str]:
    """Derive the default projection targets for a CC method.

    Returns ``["energy"]`` plus the manifold name for each
    excitation rank present in *method*.
    """
    ranks = parse_cc_level(method)
    return ["energy"] + [manifold_name(r) for r in ranks]


def _resolve_parallel_workers(parallel_workers: int | None) -> int:
    if parallel_workers is not None:
        return max(1, parallel_workers)

    env_value = os.environ.get(_PARALLEL_ENV_VAR)
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            return 1
    return 1


def _chunk_term_bucket(
    terms: tuple[object, ...],
    workers: int,
) -> list[tuple[object, ...]]:
    # Tiny workloads are cheaper to run serially than to pickle, dispatch, and
    # merge across worker processes.  Once a bucket is large enough, split it
    # into reasonably even chunks so each worker gets a similar amount of work.
    if workers <= 1 or len(terms) < workers * _MIN_CHUNK_TERMS:
        return [terms]

    chunk_size = max(_MIN_CHUNK_TERMS, (len(terms) + workers - 1) // workers)
    return [
        terms[offset:offset + chunk_size]
        for offset in range(0, len(terms), chunk_size)
    ]


def _process_term_chunk(
    terms: tuple[object, ...],
    manifold: str,
    connected_only: bool,
) -> tuple[dict[str, int | float], tuple[AlgebraTerm, ...]]:
    raw_count = 0
    filtered_count = 0
    canonicalized_count = 0
    projection_time = 0.0
    canonicalization_time = 0.0
    raw_buckets: dict[tuple[object, ...], AlgebraTerm] = {}
    raw_order: list[tuple[object, ...]] = []
    buckets: dict[tuple[object, ...], AlgebraTerm] = {}
    order: list[tuple[object, ...]] = []

    # Projection is streamed term-by-term so we never materialize the full
    # projected list for a large BCH expansion.  This keeps memory usage closer
    # to "current chunk" size instead of "all terms for the manifold".
    projected_terms = iter(iter_projected_terms_from_terms(terms, manifold))

    while True:
        t1 = time.monotonic()
        try:
            projected = next(projected_terms)
        except StopIteration:
            break
        projection_time += time.monotonic() - t1
        raw_count += 1
        if connected_only and manifold not in ("energy", "reference"):
            if not projected.connected:
                continue
            filtered_count += 1

        # Many projected terms can be recognized as zero before the more
        # expensive canonicalization step.  Pruning them here keeps the later
        # merge/canonicalize passes focused on genuinely distinct algebra.
        if term_is_zero_before_canonicalization(projected):
            continue
        merge_exact_term_into_buckets(projected, raw_buckets, raw_order)

    raw_terms = [
        raw_buckets[sig] for sig in raw_order if raw_buckets[sig].coeff != 0
    ]

    for raw_term in raw_terms:
        # Canonicalization normalizes dummy-index naming and tensor ordering so
        # algebraically equivalent terms acquire the same signature and can be
        # merged into a single bucket.
        t1 = time.monotonic()
        canonical = canonicalize_term(raw_term)
        canonicalization_time += time.monotonic() - t1
        canonicalized_count += 1
        merge_term_into_buckets(canonical, buckets, order)

    merged_terms = tuple(
        buckets[sig] for sig in order if buckets[sig].coeff != 0
    )
    stats = {
        "after_projection": raw_count,
        "after_connected_filter": filtered_count,
        "after_precanonical_prune": len(raw_terms),
        "after_canonicalization": canonicalized_count,
        "projection_time_s": projection_time,
        "canonicalization_time_s": canonicalization_time,
    }
    return stats, merged_terms


def _generate_manifold_equation(
    manifold: str,
    terms: tuple[object, ...],
    connected_only: bool,
    collect_denominators: bool,
    permutation_grouping: bool,
    exploit_symmetry: bool,
    parallel_workers: int,
) -> tuple[str, dict[str, int | float], list[AlgebraTerm]]:
    mstats: dict[str, int | float] = {}
    buckets: dict[tuple[object, ...], AlgebraTerm] = {}
    order: list[tuple[object, ...]] = []

    # First parallelization layer: split one manifold's BCH term bucket into
    # chunks.  This is the right level when a single target equation dominates
    # the runtime.
    chunk_terms = _chunk_term_bucket(terms, parallel_workers)
    if len(chunk_terms) == 1:
        chunk_results = [
            _process_term_chunk(
                chunk_terms[0],
                manifold,
                connected_only,
            )
        ]
    else:
        try:
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                chunk_results = list(executor.map(
                    _process_term_chunk,
                    chunk_terms,
                    [manifold] * len(chunk_terms),
                    [connected_only] * len(chunk_terms),
                ))
        except (OSError, PermissionError):
            # Multiprocessing is opportunistic.  If the host disallows process
            # creation or file-descriptor passing, we transparently fall back
            # to the serial path instead of failing the generation request.
            chunk_results = [
                _process_term_chunk(chunk, manifold, connected_only)
                for chunk in chunk_terms
            ]

    projection_time = 0.0
    canonicalization_time = 0.0
    raw_count = 0
    filtered_count = 0
    precanonical_count = 0
    canonicalized_count = 0

    for chunk_stats, chunk_canonical in chunk_results:
        projection_time += float(chunk_stats["projection_time_s"])
        canonicalization_time += float(chunk_stats["canonicalization_time_s"])
        raw_count += int(chunk_stats["after_projection"])
        filtered_count += int(chunk_stats["after_connected_filter"])
        precanonical_count += int(chunk_stats["after_precanonical_prune"])
        canonicalized_count += int(chunk_stats["after_canonicalization"])
        for term in chunk_canonical:
            merge_term_into_buckets(term, buckets, order)

    mstats["projection_time_s"] = projection_time
    mstats["after_projection"] = raw_count

    if connected_only and manifold not in ("energy", "reference"):
        mstats["after_connected_filter"] = filtered_count
    else:
        mstats["after_connected_filter"] = None  # type: ignore[assignment]

    mstats["after_precanonical_prune"] = precanonical_count
    mstats["canonicalization_time_s"] = canonicalization_time
    mstats["after_canonicalization"] = canonicalized_count

    canonical = [
        buckets[sig] for sig in order if buckets[sig].coeff != 0
    ]
    mstats["after_merge"] = len(canonical)

    # The remaining passes are optional algebra simplifications layered after
    # the canonical term list has been established.
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

    return manifold, mstats, canonical


def _cache_config(
    method: str,
    targets: tuple[str, ...],
    connected_only: bool,
    collect_denominators: bool,
    permutation_grouping: bool,
    exploit_symmetry: bool,
) -> dict[str, object]:
    return {
        "version": _CACHE_VERSION,
        "method": method,
        "targets": list(targets),
        "connected_only": connected_only,
        "collect_denominators": collect_denominators,
        "permutation_grouping": permutation_grouping,
        "exploit_symmetry": exploit_symmetry,
    }


def _cache_root(
    cache_dir: str,
    method: str,
) -> Path:
    return Path(cache_dir).expanduser().resolve() / method


def _bch_cache_config(method: str) -> dict[str, object]:
    return {
        "version": _BCH_CACHE_VERSION,
        "method": method,
        "ranks": parse_cc_level(method),
        "max_order": _BCH_MAX_ORDER,
    }


def _bch_config_path(root: Path) -> Path:
    return root / "bch_config.json"


def _bch_cache_file(root: Path) -> Path:
    return root / "bch_levels.pkl"


def _bch_cache_ready(root: Path, config: dict[str, object]) -> bool:
    config_path = _bch_config_path(root)
    if not config_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return True
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return existing == config


def _load_cached_bch_levels(root: Path) -> tuple[Expr, ...] | None:
    path = _bch_cache_file(root)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _store_cached_bch_levels(root: Path, levels: tuple[Expr, ...]) -> None:
    if not _bch_cache_ready(root, _bch_cache_config(root.name)):
        return
    path = _bch_cache_file(root)
    with path.open("wb") as handle:
        pickle.dump(levels, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _find_prefix_bch_cache(
    cache_dir: str,
    method: str,
) -> tuple[str, tuple[object, ...]] | None:
    root = Path(cache_dir).expanduser().resolve()
    if not root.exists():
        return None

    method_ranks = tuple(parse_cc_level(method))
    best_method: str | None = None
    best_levels: tuple[Expr, ...] | None = None
    best_len = -1

    for child in root.iterdir():
        if not child.is_dir():
            continue
        config_path = _bch_config_path(child)
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        candidate_method = str(config.get("method", child.name))
        candidate_ranks = tuple(config.get("ranks", []))
        if not candidate_ranks or len(candidate_ranks) >= len(method_ranks):
            continue
        if not set(candidate_ranks).issubset(method_ranks):
            continue
        # Reuse is only valid when the cached method is a strict prefix in the
        # excitation hierarchy, e.g. CCSD -> CCSDT.  Among valid candidates we
        # prefer the longest prefix to minimize the amount of new BCH work.
        levels = _load_cached_bch_levels(child)
        if levels is None:
            continue
        if len(candidate_ranks) > best_len:
            best_method = candidate_method
            best_levels = levels
            best_len = len(candidate_ranks)

    if best_method is None or best_levels is None:
        return None
    return best_method, best_levels


def _cache_ready(
    root: Path,
    config: dict[str, object],
) -> bool:
    config_path = root / "config.json"
    if not config_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        return True
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return existing == config


def _cache_file(root: Path, manifold: str) -> Path:
    return root / f"{manifold}.pkl"


def _load_cached_manifold(
    root: Path,
    manifold: str,
) -> tuple[dict[str, int | float], list[AlgebraTerm]] | None:
    path = _cache_file(root, manifold)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _store_cached_manifold(
    root: Path,
    manifold: str,
    mstats: dict[str, int | float],
    canonical: list[AlgebraTerm],
) -> None:
    path = _cache_file(root, manifold)
    with path.open("wb") as handle:
        pickle.dump((mstats, canonical), handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_cc_equations(
    method: str,
    targets: list[str] | None = None,
    connected_only: bool = True,
    collect_denominators: bool = False,
    permutation_grouping: bool = False,
    exploit_symmetry: bool = False,
    debug: bool = False,
    parallel_workers: int | None = None,
    cache_dir: str | None = None,
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

    method = canonicalize_cc_level(method.lower())
    if targets is None:
        targets = targets_for_method(method)

    # Build the symbolic Hamiltonian H and cluster operator T once, then reuse
    # them throughout the pipeline.  Everything below transforms or projects
    # the resulting BCH-expanded similarity-transformed Hamiltonian.
    H = build_hamiltonian()
    T = build_cluster(method)

    cache_root_path: Path | None = None
    bch_root_path: Path | None = None
    if cache_dir is not None:
        cache_root_path = _cache_root(cache_dir, method)
        bch_root_path = _cache_root(cache_dir, method)

    t0 = time.monotonic()
    bch_cache_hit = False
    bch_reused_from: str | None = None
    bch_levels_cached: tuple[object, ...] | None = None
    if bch_root_path is not None and _bch_cache_ready(
        bch_root_path,
        _bch_cache_config(method),
    ):
        bch_levels_cached = _load_cached_bch_levels(bch_root_path)

    if bch_levels_cached is not None:
        levels = bch_levels_cached
        Hbar = bch_result_from_levels(levels)
        bch_cache_hit = True
    else:
        prefix_cached: tuple[str, tuple[object, ...]] | None = None
        if cache_dir is not None:
            prefix_cached = _find_prefix_bch_cache(cache_dir, method)
        if prefix_cached is not None:
            prefix_method, prefix_levels = prefix_cached
            prefix_ranks = set(parse_cc_level(prefix_method))
            delta_ranks = [
                rank for rank in parse_cc_level(method)
                if rank not in prefix_ranks
            ]
            T_prefix = build_cluster(prefix_method)
            T_delta = build_tn(delta_ranks[0])
            for rank in delta_ranks[1:]:
                T_delta = T_delta + build_tn(rank)
            # Reconstruct Hbar by continuing from the cached prefix BCH levels
            # rather than restarting from H.  For higher-rank methods this can
            # avoid repeating the most expensive lower-order commutators.
            Hbar, levels = bch_expand_from_prefix_levels(
                prefix_levels,
                T_prefix,
                T_delta,
                max_order=_BCH_MAX_ORDER,
            )
            bch_reused_from = prefix_method
        else:
            levels = bch_levels(H, T, max_order=_BCH_MAX_ORDER)
            Hbar = bch_result_from_levels(levels)

        if bch_root_path is not None:
            _store_cached_bch_levels(bch_root_path, levels)

    bch_time = time.monotonic() - t0

    stats = PipelineStats(
        method=method,
        bch_terms=len(Hbar.terms),
        bch_time_s=bch_time,
        bch_cache_hit=bch_cache_hit,
        bch_reused_from=bch_reused_from,
    )

    equations: dict[str, list[AlgebraTerm]] = {}
    target_tuple = tuple(targets)
    term_buckets = bucket_terms_by_manifold(Hbar, target_tuple)
    resolved_workers = _resolve_parallel_workers(parallel_workers)
    if cache_dir is not None:
        config = _cache_config(
            method,
            target_tuple,
            connected_only,
            collect_denominators,
            permutation_grouping,
            exploit_symmetry,
        )
        if not _cache_ready(cache_root_path, config):
            cache_root_path = None

    pending: list[str] = []
    for manifold in target_tuple:
        if cache_root_path is None:
            pending.append(manifold)
            continue
        cached = _load_cached_manifold(cache_root_path, manifold)
        if cached is None:
            pending.append(manifold)
            continue
        mstats, canonical = cached
        stats.manifolds[manifold] = mstats
        equations[manifold] = canonical

    if pending:
        manifold_results: list[tuple[str, dict[str, int | float], list[AlgebraTerm]]] = []
        # Second parallelization layer: distribute different manifolds across
        # workers.  When we take this route, each worker handles its assigned
        # manifold serially to avoid nested process pools.
        use_manifold_parallel = resolved_workers > 1 and len(pending) > 1
        if use_manifold_parallel:
            try:
                with ProcessPoolExecutor(max_workers=min(resolved_workers, len(pending))) as executor:
                    manifold_results = list(executor.map(
                        _generate_manifold_equation,
                        pending,
                        [term_buckets[name] for name in pending],
                        [connected_only] * len(pending),
                        [collect_denominators] * len(pending),
                        [permutation_grouping] * len(pending),
                        [exploit_symmetry] * len(pending),
                        [1] * len(pending),
                    ))
            except (OSError, PermissionError):
                # If manifold-level fan-out is unavailable, retry locally and
                # still allow intra-manifold chunking via resolved_workers.
                manifold_results = [
                    _generate_manifold_equation(
                        manifold,
                        term_buckets[manifold],
                        connected_only,
                        collect_denominators,
                        permutation_grouping,
                        exploit_symmetry,
                        resolved_workers,
                    )
                    for manifold in pending
                ]
        else:
            manifold_results = [
                _generate_manifold_equation(
                    manifold,
                    term_buckets[manifold],
                    connected_only,
                    collect_denominators,
                    permutation_grouping,
                    exploit_symmetry,
                    resolved_workers,
                )
                for manifold in pending
            ]

        for manifold, mstats, canonical in manifold_results:
            stats.manifolds[manifold] = mstats
            equations[manifold] = canonical
            if cache_root_path is not None:
                _store_cached_manifold(cache_root_path, manifold, mstats, canonical)

    # Preserve the caller-requested target order even if cache hits and parallel
    # execution completed in a different sequence.
    stats.manifolds = {
        manifold: stats.manifolds[manifold]
        for manifold in target_tuple
    }
    equations = {
        manifold: equations[manifold]
        for manifold in target_tuple
    }

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
    intermediate_memory_budget_bytes: int | None = None,
    intermediate_peak_memory_budget_bytes: int | None = None,
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
            memory_budget_bytes=intermediate_memory_budget_bytes,
            peak_memory_budget_bytes=intermediate_peak_memory_budget_bytes,
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
    intermediate_memory_budget_bytes: int | None = None,
    intermediate_peak_memory_budget_bytes: int | None = None,
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
        intms = detect_intermediates(
            eqs,
            threshold=intermediate_threshold,
            memory_budget_bytes=intermediate_memory_budget_bytes,
            peak_memory_budget_bytes=intermediate_peak_memory_budget_bytes,
        )
        if annotate_layout:
            intms = annotate_layout_hints(intms)
        intermediates = intms if intms else None

    return format_equations_with_intermediates(
        eqs,
        intermediates=intermediates,
        include_legend=include_legend,
        include_stats=include_stats,
    )
