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
