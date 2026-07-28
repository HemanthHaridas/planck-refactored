"""Spin-adaptation layer (S0): the index model + single-term spin labeling.

ccgen derives every equation in spin-orbital (GCC) form -- `indices.Index` has a
space (occ/vir/gen) but NO spin. Mapping those equations to restricted (RCC) or
unrestricted (UCC) spatial-orbital form is *spin integration*: each spin-orbital
index p is a (spatial index, spin sigma in {a,b}) pair, and the spin-orbital
equation is summed over spin.

DESIGN DECISION (S0): the spin layer is kept ISOLATED from the GCC path. It does
NOT add a spin field to `Index` (which is baked into every canonicalize / wick /
diagram hash and equality -- perturbing it would risk the validated GCC path).
Instead it wraps a spatial `Index` in a lightweight `SpinIndex` and operates on
`AlgebraTerm`s produced by generation. Generation, canonicalization, and the
diagram engine are untouched.

S0 delivers only the model + the single-term spin *labeling* (assign a spin to
every index consistently along shared lines, enumerate the summed-index spin
cases). The coefficient algebra -- UCC block coefficients (S1) and the RCC
alpha=beta collapse (S2) -- is later work. So the S0 gate is STRUCTURAL: labels
are consistent along shared names and the summed indices enumerate the right
spin cases; it does not yet assert collapsed coefficients.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .indices import Index

SPINS = ("a", "b")  # alpha, beta


@dataclass(frozen=True)
class SpinIndex:
    """A spatial index plus a spin label. Wraps a GCC `Index` without touching
    it -- the spatial identity (`base`) is the original occ/vir index; `spin` is
    "a" or "b". Two spin-orbital lines are the same iff same base name AND same
    spin."""

    base: Index
    spin: str

    def __post_init__(self) -> None:
        if self.spin not in SPINS:
            raise ValueError(f"spin must be one of {SPINS}, got {self.spin!r}")

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def space(self) -> str:
        return self.base.space

    def __repr__(self) -> str:
        return f"{self.base.name}{self.spin}"


def _distinct_names(term) -> tuple[list[Index], list[Index]]:
    """(free, summed) index objects of a term, de-duplicated by name -- a
    repeated NAME is one physical line and gets one spin."""
    seen: dict[str, Index] = {}
    free: list[Index] = []
    summed: list[Index] = []
    free_names = {i.name for i in term.free_indices}
    for fac in term.factors:
        for idx in fac.indices:
            if idx.name in seen:
                continue
            seen[idx.name] = idx
            (free if idx.name in free_names else summed).append(idx)
    return free, summed


def spin_label_cases(term, external_spins: dict[str, str]):
    """Enumerate the spin-labeled forms of one GCC term for a fixed choice of
    EXTERNAL (free-index) spins (S0).

    Every distinct index NAME in the term is one spin-orbital line and carries a
    single spin. Free names take their spin from `external_spins`; each summed
    name is enumerated over both spins (the spin sum). Returns a list of
    ``{index_name: SpinIndex}`` maps, one per summed-spin assignment -- i.e.
    ``2 ** (number of distinct summed names)`` cases.

    This is labeling only: it does NOT yet produce spin-blocked tensors or
    integrate coefficients (S1/S2). It is the structural primitive both build on.
    """
    free, summed = _distinct_names(term)
    missing = [i.name for i in free if i.name not in external_spins]
    if missing:
        raise ValueError(f"external_spins missing free indices: {missing}")

    base_by_name = {i.name: i for i in free + summed}
    summed_names = [i.name for i in summed]

    cases = []
    for combo in itertools.product(SPINS, repeat=len(summed_names)):
        assignment = dict(zip(summed_names, combo))
        label = {}
        for name, base in base_by_name.items():
            spin = external_spins[name] if name in external_spins else assignment[name]
            label[name] = SpinIndex(base, spin)
        cases.append(label)
    return cases


# ── S1.0 the UCC block model: spin conservation per line ──────────────
#
# A spin-orbital tensor block is nonzero iff spin is CONSERVED along every
# excitation / interaction LINE. ccgen orders each tensor's indices as n virtual
# slots followed by n occupied slots (t1: [v,o]; t2: [v,v,o,o]; t3: [v,v,v,o,o,o];
# f: [p,q]; v = <pq||rs>: [v,v,o,o] but the physicist antisym lines pair p-r, q-s).
# So for a rank-2n tensor the lines pair slot k with slot k+n, and the block
# exists iff spin(slot k) == spin(slot k+n) for every k in 0..n-1.
#
# This one rule covers every tensor kind ccgen emits -- there is no per-tensor
# lookup table. It is the spin-conservation content of the UCC block structure
# (t1 -> a,b; t2 -> aa,bb,ab; f -> a,b; v -> aaaa,bbbb,abab,baba), derived from
# physics, matching PySCF's case-tagged UCC blocks.


def _line_pairs(factor) -> list[tuple[int, int]]:
    """Slot pairs (k, k+n) for a rank-2n tensor -- the excitation/interaction
    lines along which spin is conserved. Raises for odd rank (no line pairing)."""
    r = len(factor.indices)
    if r % 2 != 0:
        raise ValueError(f"{factor.name}: odd rank {r} has no line pairing")
    n = r // 2
    return [(k, k + n) for k in range(n)]


def block_exists(factor, label: dict) -> bool:
    """Whether the spin-labeled *factor* is a nonzero UCC block: spin is
    conserved along every line (slot k and slot k+n carry the same spin).
    ``label`` maps index name -> SpinIndex (from :func:`spin_label_cases`)."""
    spins = [label[i.name].spin for i in factor.indices]
    return all(spins[a] == spins[b] for a, b in _line_pairs(factor))


def resolve_block(factor, label: dict) -> tuple[str, bool]:
    """Resolve the spin-labeled *factor* to a (block_tag, exists) pair (S1.1).

    ``block_tag`` is the per-slot spin string in the factor's own index order
    (e.g. ``t2`` slots [v,v,o,o] with spins a,b,a,b -> "abab"); ``exists`` is
    :func:`block_exists`. A forbidden block still gets a tag (for diagnostics)
    but ``exists=False`` and S1.2 will drop it. Coefficient integration and the
    physicist->chemist naming are S1.2/S1.4, not here."""
    tag = "".join(label[i.name].spin for i in factor.indices)
    return tag, block_exists(factor, label)


# ── S1.2 single-term UCC integration ──────────────────────────────────
#
# For a fixed EXTERNAL block (a spin assignment to the free indices), a GCC term
# contributes the SUM over its summed-index spin cases (S0) of the cases where
# EVERY factor is a valid block (S1.1). Each surviving case is one spatial term:
# same GCC coefficient, factors tagged with their spin block, spatial indices.
#
# The coefficient here is the RAW GCC coefficient. Whether the spatial block
# combinatorics multiply it by a factor (e.g. the GCC 1/2 on a summed
# spin-orbital antisym pair becoming 1 when the surviving spatial case has
# distinct-spin summed indices) is deliberately NOT decided here -- it is the
# thing the numeric gates settle: the spin-orbital identity (S1.2 test) and PySCF
# `uccsd.update_amps` (S1.3). S1.2 emits the surviving cases faithfully; the gates
# tell us if the block model already makes the coefficients come out right.


@dataclass(frozen=True)
class SpinFactor:
    """One factor of a spin-integrated (spatial) term: the tensor name, its spin
    block tag (per-slot spins in the factor's index order), and the SpinIndex per
    slot (spatial index + spin)."""

    name: str
    block: str
    indices: tuple


@dataclass(frozen=True)
class SpinTerm:
    """A single spin-integrated spatial term: coefficient + tagged factors, for a
    named external block (the spin assignment of the free indices)."""

    coeff: object
    external_block: tuple
    factors: tuple


def ucc_integrate_term(term, external_spins: dict):
    """Integrate one GCC ``AlgebraTerm`` into the UCC ``SpinTerm``s that
    contribute to the given external block (S1.2).

    Enumerates the summed-spin cases (:func:`spin_label_cases`), keeps only cases
    where every factor is a valid block (:func:`block_exists`), and emits one
    ``SpinTerm`` per surviving case carrying the GCC coefficient. The external
    block is the sorted (free-name, spin) tuple; each factor is tagged with its
    resolved block."""
    ext_block = tuple(sorted(external_spins.items()))
    out = []
    for label in spin_label_cases(term, external_spins):
        if not all(block_exists(f, label) for f in term.factors):
            continue
        factors = tuple(
            SpinFactor(
                name=f.name,
                block=resolve_block(f, label)[0],
                indices=tuple(label[i.name] for i in f.indices),
            )
            for f in term.factors
        )
        out.append(SpinTerm(coeff=term.coeff, external_block=ext_block, factors=factors))
    return out
