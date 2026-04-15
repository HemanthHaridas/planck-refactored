"""Top-level CC equation generation API."""

from __future__ import annotations

from typing import Any

from .hamiltonian import build_hamiltonian
from .cluster import build_cluster, parse_cc_level
from .algebra import bch_expand
from .project import project, AlgebraTerm, manifold_name
from .canonicalize import canonicalize_term, merge_like_terms
from .emit.pretty import format_equations
from .emit.einsum import format_equations_einsum
from .emit.cpp_loops import emit_translation_unit
from .tensor_ir import lower_equations, BackendTerm


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
) -> dict[str, list[AlgebraTerm]]:
    """Generate coupled-cluster equations for a given method."""
    method = method.lower()
    if targets is None:
        targets = targets_for_method(method)

    H = build_hamiltonian()
    T = build_cluster(method)

    Hbar = bch_expand(H, T, max_order=4)

    equations: dict[str, list[AlgebraTerm]] = {}

    for manifold in targets:
        raw = project(Hbar, manifold)

        if connected_only and manifold not in ("energy", "reference"):
            raw = [t for t in raw if t.connected]

        canonical = merge_like_terms(
            [canonicalize_term(t) for t in raw]
        )
        equations[manifold] = canonical

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


def print_einsum(method: str, **kwargs: Any) -> str:
    """Generate and format equations as numpy einsum code."""
    eqs = generate_cc_equations(method, **kwargs)
    return format_equations_einsum(eqs)


def print_cpp(method: str, **kwargs: Any) -> str:
    """Generate and format equations as C++ loop nests."""
    eqs = generate_cc_equations(method, **kwargs)
    return emit_translation_unit(eqs)
