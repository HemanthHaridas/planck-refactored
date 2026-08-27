"""Planck-specific C++ emitter for coupled-cluster tensor kernels.

The generic ``cpp_loops`` backend emits abstract ``F/V/R`` accessors. This
backend targets the concrete tensor objects already present in Planck's CC
implementation:

- ``CanonicalRHFCCReference`` for Fock blocks
- ``TensorCCBlockCache`` for ERI blocks
- ``DenominatorCache`` for orbital-energy denominators
- ``RCCSDAmplitudes`` / ``RCCSDTAmplitudes`` for cluster amplitudes
"""

from __future__ import annotations

from fractions import Fraction
import itertools
import re
from typing import Sequence, TYPE_CHECKING

from ..cluster import parse_cc_level
from ..indices import Index
from ..lowering import (
    LoweredTensorFactor,
    RestrictedClosedShellTerm,
    lower_equations_restricted_closed_shell,
    lower_term_restricted_closed_shell,
)
from ..project import AlgebraTerm
from ..spin import ucc_term_index_spins
from ..tensors import SPATIAL_ERI_SYMMETRIES, Tensor

if TYPE_CHECKING:
    from ..optimization.intermediates import IntermediateSpec


_CANONICAL_ERI_BLOCKS: dict[str, tuple[str, str, str, str]] = {
    "oooo": ("o", "o", "o", "o"),
    "ooov": ("o", "o", "o", "v"),
    "oovv": ("o", "o", "v", "v"),
    "ovov": ("o", "v", "o", "v"),
    "ovvo": ("o", "v", "v", "o"),
    "ovvv": ("o", "v", "v", "v"),
    "vvvv": ("v", "v", "v", "v"),
}

# These are the symmetries _map_eri_tensor may use to route an abstract `v`
# factor to a canonical mo_blocks block. `mo_blocks` holds the NON-antisymmetrized
# spatial physicist integral <pq|rs> (built by build_tensor_cc_block_cache;
# rebound to physicist for the generated-kernel path in
# generated_arbitrary_prepare.cpp). Its only genuine index symmetries for real
# orbitals are the four +1 relations below: identity, particle swap <qp|sr>,
# bra<->ket <rs|pq>, and their product <sr|qp>. They cover all 16 four-index o/v
# patterns (verified), so no coverage is lost.
#
# The four antisymmetric single-swap relations <qp|rs> = -<pq|rs> and
# <pq|sr> = -<pq|rs> hold ONLY for the antisymmetrized <pq||rs> the spin-ORBITAL
# equations use -- NOT for these spatial blocks. Emitting them produced reads like
# `oovv(l,k,c,d)` with a bogus sign, which was the residual-emit defect: the
# energy kernel (identity-perm oovv reads only) was exact while the doubles/
# quadruples residuals were wrong. Do NOT re-add the -1 perms.
# Shared with the lowering path (D5): both resolve SPATIAL blocks, and keeping
# two copies is what let one of them carry the invalid -1 relations.
_ERI_SYMMETRY_PERMUTATIONS = SPATIAL_ERI_SYMMETRIES


def eri_permutation_preserves_block(block_tag: str, perm: tuple[int, ...]) -> bool:
    """U3.0 -- is `perm` a valid symmetry of the UCC spin block `block_tag`?

    The four permutations above are symmetries of the SPATIAL physicist integral
    `<pq|rs>`, and applying them is sound as long as there is one such integral.
    Under UCC there are three (`v_aaaa`, `v_abab`, `v_bbbb`), and a permutation
    that reorders the indices also reorders their SPINS. It is therefore usable on
    a block only when it maps that block's spin string to itself; otherwise it
    relates the block to a DIFFERENT one and using it silently reads the wrong
    integral with permuted indices.

    Measured on the current CCSD UCC manifold, with `_ERI_SYMMETRY_PERMUTATIONS`
    applied spin-blindly as it is today:

        identity          (0,1,2,3)   abab -> abab   valid      92 reads
        particle <qp|sr>  (1,0,3,2)   abab -> baba   INVALID    24 reads
        bra<->ket <rs|pq> (2,3,0,1)   abab -> abab   valid      13 reads
        product <sr|qp>   (3,2,1,0)   abab -> baba   INVALID    13 reads

    i.e. 37 of 142 mixed-block reads currently use a symmetry that holds only for
    `baba`. Same-spin blocks are unaffected: every permutation of `aaaa` is `aaaa`,
    which is why this never surfaced on the RCC path (one block, all four valid).

    Verified numerically on random real orbitals rather than by tag algebra alone:
    `baba == abab.transpose(1,0,3,2)`, `abab` IS invariant under `(2,3,0,1)`, and
    `abab` is NOT invariant under `(1,0,3,2)`.

    `baba` is deliberately NOT a stored block -- it is `abab` under the particle
    swap, so storing it would cost ~33% more ERI memory to avoid one explicit swap
    at the point of use.
    """
    if len(perm) != len(block_tag):
        raise ValueError(
            f"eri_permutation_preserves_block: permutation of length {len(perm)} "
            f"cannot apply to block tag {block_tag!r} of length {len(block_tag)}")
    return "".join(block_tag[i] for i in perm) == block_tag


def eri_permutations_for_block(
    block_tag: str | None,
) -> tuple[tuple[tuple[int, int, int, int], int], ...]:
    """The subset of `_ERI_SYMMETRY_PERMUTATIONS` usable on `block_tag`.

    `block_tag=None` is the RCC path (a bare `v` with no spin resolution), which
    keeps all four -- there is a single spatial tensor and nothing to leave.
    """
    if block_tag is None:
        return _ERI_SYMMETRY_PERMUTATIONS
    return tuple(
        (perm, sign) for perm, sign in _ERI_SYMMETRY_PERMUTATIONS
        if eri_permutation_preserves_block(block_tag, perm))


def _space_char(idx: Index) -> str:
    if idx.space == "occ":
        return "o"
    if idx.space == "vir":
        return "v"
    return "g"


def _loop_bound(idx: Index, spin: str | None = None) -> str:
    """The C++ loop bound for one index.

    U3b.2a: under UCC the bound is spin-resolved -- an alpha-occupied index runs
    to ``noa`` and a beta-occupied one to ``nob``, and under UHF those differ.
    ``spin`` is ``'a'``/``'b'`` (from ``ucc_term_index_spins``) or ``None`` on the
    RCC path, where one (n_occ, n_virt) pair covers every index and the emitted
    text must stay byte-identical.
    """
    suffix = spin if spin in ("a", "b") else ""
    if idx.space == "occ":
        return "no" + suffix
    if idx.space == "vir":
        return "nv" + suffix
    return "n"


def _orbital_count_preamble(ucc: bool) -> list[str]:
    """The `const int` orbital-count declarations at the top of a kernel.

    U3b.2b: RCC reads the single (n_occ, n_virt) pair off `orbital_partition`.
    UCC cannot -- `build_ucc_fock_blocks` deliberately leaves `orbital_partition`
    DEFAULT, because it holds one pair and an unrestricted partition needs four
    (filling it with either spin's counts is the plausible-wrong-answer shortcut
    U3b.1 explicitly rejects). So UCC declares the four spin-resolved counts that
    U3b.2a's loop bounds and U3b.2c's allocations refer to.

    The four public members are read directly rather than through
    `occupied_count('a')` / `virtual_count('b')`. Those return
    `std::expected<int, std::string>` and would need an unwrap at every kernel
    preamble for an error path that cannot fire: the emitter writes the spin
    literal itself, so a non-'a'/'b' spin is unreachable by construction. The
    accessors stay the checked entry point for callers whose spin is data.
    """
    if not ucc:
        return [
            "    const int no = reference.orbital_partition.n_occ;",
            "    const int nv = reference.orbital_partition.n_virt;",
        ]
    return [
        "    const int noa = reference.n_occ_alpha;",
        "    const int nob = reference.n_occ_beta;",
        "    const int nva = reference.n_virt_alpha;",
        "    const int nvb = reference.n_virt_beta;",
    ]


def _coeff_literal(coeff: Fraction) -> str:
    value = float(coeff)
    if value == int(value):
        integer = int(value)
        if integer == 1:
            return ""
        if integer == -1:
            return "-"
        return f"{integer} * "
    return f"{value} * "


def _tensor_type(rank: int) -> str:
    if rank == 0:
        return "double"
    if rank == 2:
        return "Tensor2D"
    if rank == 4:
        return "Tensor4D"
    if rank == 6:
        return "Tensor6D"
    if rank > 0:
        return "TensorND"
    raise ValueError(f"Unsupported tensor rank {rank} for Planck emitter")


def _is_supported_tensor_rank(rank: int) -> bool:
    return rank >= 0


def _dims_expr(
    indices: Sequence[Index],
    result_type: str | None = None,
    spins: dict[str, str] | None = None,
) -> str:
    """The extents for a result allocation.

    U3b.2c: under UCC each extent is spin-resolved, so `doubles_abab` allocates
    `(noa, nob, nva, nvb)` rather than four copies of one pair. `spins` maps index
    name -> 'a'/'b' (from `ucc_term_index_spins`); absent or unmapped indices fall
    back to the RCC `no`/`nv`, which is what keeps that emit byte-identical.

    Folded in with U3b.2b rather than shipped separately: 2b removes the
    `const int no/nv` declarations, so leaving the allocations spin-blind emits a
    TU that does not compile ("use of undeclared identifier 'no'", 16 errors on
    the CCSD UCC TU). The two are one atomic change.
    """
    result_type = result_type or _tensor_type(len(indices))
    spins = spins or {}
    dims: list[str] = []
    for idx in indices:
        suffix = spins.get(idx.name, "")
        if suffix not in ("a", "b"):
            suffix = ""
        if idx.space == "occ":
            dims.append("no" + suffix)
        elif idx.space == "vir":
            dims.append("nv" + suffix)
        else:
            raise ValueError(
                "Planck emitter only supports occupied/virtual tensors, "
                f"got index {idx!r}"
            )
    if result_type == "TensorND":
        return "std::vector<int>{" + ", ".join(dims) + "}, 0.0"
    return ", ".join(dims) + ", 0.0"


def _inverse_permutation(perm: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(perm)
    for i, value in enumerate(perm):
        inverse[value] = i
    return tuple(inverse)


def _source_tensor(factor: Tensor | LoweredTensorFactor) -> Tensor:
    if isinstance(factor, LoweredTensorFactor):
        return factor.source
    return factor


def _access_indices(factor: Tensor | LoweredTensorFactor) -> tuple[Index, ...]:
    if isinstance(factor, LoweredTensorFactor):
        return factor.spatial_indices
    return _source_tensor(factor).indices


def _block_needs_explicit_exchange(block_tag: str | None) -> bool:
    """Does a `v` read on this spin block need an explicit exchange term?

    U5.4. THE CONVENTION, stated in one place because having it implicit on both
    sides is what let the two halves disagree:

        every array in the UCC block cache stores the PLAIN <pq|rs>.

    The UCC algebra means the ANTISYMMETRIZED <pq||rs> when it writes `v_aaaa` --
    `ucc_integrate_term_antisym` is documented as integrating "for REAL
    ANTISYMMETRIC tensors", and the energy manifold carries ONE same-spin term at
    coefficient 1/4 with no exchange partner, which only balances for <pq||rs>.
    So the exchange has to appear somewhere. It appears HERE, in the emitted text,
    rather than being folded into the stored array.

    WHY THIS SIDE, and not by antisymmetrizing the cache:

    1. `<pq||rs> = <pq|rs> - <pq|sr>` requires slots 2 and 3 to be interchangeable,
       which holds only when they carry the SAME spin. For a mixed block the
       exchange partner is a different tensor SHAPE -- `oovv_abab` is
       (noa, nob, nva, nvb) and its partner would be (noa, nob, nvb, nva), which
       the array cannot even hold. So antisymmetrizing the cache is not one
       transform over a vocabulary; it is a conditional applied to a subset,
       leaving one accessor with two meanings and no marker saying which.
    2. `ucc_blocks.cpp` already states the rule this would break: the four
       antisymmetric single-swap relations "hold only for the ANTISYMMETRIZED
       <pq||rs> ... do not add them here either". `kEriSymmetries` is built on
       that premise and would be invalid for exactly the blocks that changed.
    3. Three landed C++ gates assert on the plain blocks -- U3.4's MP2 limit
       (which writes the exchange by hand as `gab - gba`), U5.2c's UMP2 energy,
       and the structural rebind gate. Antisymmetrizing under them keeps them
       passing while silently changing what they mean.

    The 2-index Fock blocks are unaffected: they are plain on both sides already.
    Mixed 4-index blocks are unaffected because the algebra gives them no exchange
    partner -- `abab` appears at coefficient 1, not 1/4.

    Returns False on the RCC path (`block_tag is None`), where a single spatial
    ERI tensor serves every read and the closed-shell equations already carry
    their own exchange structure.
    """
    if block_tag is None:
        return False
    return len(set(block_tag)) == 1


def _resolve_eri_block_name(
    spaces: tuple[str, ...],
    block_tag: str | None,
) -> str | None:
    """The stored block one o/v pattern resolves to, or None if unreachable.

    Shares the search with `_resolve_eri_read` so the bound views and the emitted
    reads cannot disagree about which arrays a kernel touches.
    """
    permutations = eri_permutations_for_block(block_tag)
    for block_name, block_spaces in _canonical_eri_blocks_for(block_tag).items():
        for perm, _sign in permutations:
            if tuple(block_spaces[i] for i in perm) == tuple(spaces):
                return block_name
    return None


def _resolve_eri_read(
    spaces: tuple[str, ...],
    names: Sequence[str],
    block_tag: str | None,
) -> tuple[int, str] | None:
    """Resolve one o/v pattern to `(sign, "array(args)")`, or None if unreachable.

    The block search, factored out of `_map_eri_tensor` so the antisymmetrization
    partner can go through exactly the same routing as the direct read rather
    than reusing the direct read's array and argument order.
    """
    permutations = eri_permutations_for_block(block_tag)
    for block_name, block_spaces in _canonical_eri_blocks_for(block_tag).items():
        for perm, sign in permutations:
            if tuple(block_spaces[i] for i in perm) != tuple(spaces):
                continue
            inverse = _inverse_permutation(perm)
            reordered = [names[i] for i in inverse]
            return sign, f"{_eri_read(block_tag, block_name)}({', '.join(reordered)})"
    return None


def _map_eri_tensor(
    tensor: Tensor | LoweredTensorFactor,
    block_tag: str | None = None,
) -> tuple[int, str]:
    """Route an abstract `v` factor to a stored mo_blocks array.

    `block_tag` (U3.2) is the UCC spin block (`"abab"`), or None on the RCC path
    where a single spatial ERI tensor serves every read. It does two things, and
    doing only the first is worse than doing neither:

    1. Routes to the block's OWN array (`v_abab` -> `v_abab.oovv`), rather than
       collapsing all three spin blocks onto `mo_blocks.oovv`.
    2. Restricts the symmetry search to the permutations that are actually
       symmetries of that block (U3.0). Two of the four map `abab` to `baba`, so
       using them would read the right array with permuted indices -- a quieter
       wrong answer than the name collapse, and one that fixing (1) alone would
       have introduced.
    """
    if isinstance(tensor, LoweredTensorFactor):
        if "g" in tensor.spatial_block:
            raise NotImplementedError(
                "General-space ERI blocks are not supported in Planck output: "
                f"{tensor.source!r}"
            )
        return (
            tensor.phase,
            f"{_eri_read(block_tag, tensor.spatial_block)}("
            f"{', '.join(idx.name for idx in tensor.spatial_indices)})",
        )

    tensor_obj = _source_tensor(tensor)
    spaces = tuple(_space_char(idx) for idx in tensor_obj.indices)
    if "g" in spaces:
        raise NotImplementedError(
            "General-space ERI blocks are not supported in Planck output: "
            f"{tensor_obj!r}"
        )

    names = [idx.name for idx in tensor_obj.indices]
    direct = _resolve_eri_read(spaces, names, block_tag)
    if direct is not None:
        sign, direct_expr = direct
        if not _block_needs_explicit_exchange(block_tag):
            return sign, direct_expr

        # U5.4 / R5: `v_aaaa` means the ANTISYMMETRIZED <pq||rs>, but the stored
        # array holds the plain <pq|rs>, so the exchange is emitted here.
        #
        # `<pq||rs> = <pq|rs> - <pq|sr>` swaps the two KET slots of the PHYSICIST
        # integral. The partner is then re-resolved through the SAME block search
        # as the direct read, because the ket-swapped PATTERN is generally a
        # DIFFERENT stored block reached by a DIFFERENT permutation.
        #
        # R4: the original fix swapped the last two ARGUMENT POSITIONS of the
        # direct read instead. That coincides with the ket swap only when it
        # stays inside one space (oooo/oovv/vvvv) and is wrong on
        # ooov/ovov/ovvv, where it crosses occ/vir and indexes the wrong array --
        # e.g. <ic|ka> (ovov) has ket-swapped partner <ic|ak> (ovvo), which the
        # old code read out of `ovov`. Half the emitted exchange pairs were
        # affected.
        ex_spaces = (spaces[0], spaces[1], spaces[3], spaces[2])
        ex_names = [names[0], names[1], names[3], names[2]]
        partner = _resolve_eri_read(ex_spaces, ex_names, block_tag)
        if partner is None:
            raise NotImplementedError(
                f"No Planck ERI block mapping for the exchange partner pattern "
                f"{''.join(ex_spaces)} in spin block {block_tag!r} "
                f"(direct pattern {''.join(spaces)})")
        partner_sign, partner_expr = partner
        # The partner carries its own permutation sign; fold it against the
        # direct read's so the emitted subtraction has the right relative sign.
        joiner = "-" if partner_sign * sign > 0 else "+"
        return sign, f"({direct_expr} {joiner} {partner_expr})"

    raise NotImplementedError(
        f"No Planck ERI block mapping available for pattern "
        f"{''.join(spaces)}"
        + (f" in spin block {block_tag!r}" if block_tag else "")
        + f" in {tensor_obj!r}"
    )


def _fock_read(block_tag: str | None, space: str, *names: str) -> str:
    """The C++ expression reading one Fock element.

    RCC reads the reference member directly (`reference.f_ov(i, a)`). UCC reads a
    per-spin view bound once per kernel (`f_aa_ov(i, a)`), the same shape as the
    spin-blocked ERI reads and the amplitude sector views. Must agree with
    `_fock_view_bindings`, which declares these names.
    """
    args = ", ".join(names)
    if block_tag is None:
        return f"reference.f_{space}({args})"
    return f"f_{block_tag}_{space}({args})"


def _fock_blocks_used(terms) -> list[tuple[str, str]]:
    """Distinct (space, spin tag) Fock blocks referenced across `terms`.

    `vo` is normalized to `ov` here exactly as `_map_factor` does, so the bound
    set and the reads cannot disagree about whether a `vo` read needs its own
    view (it does not -- the Fock is symmetric).
    """
    used: set[tuple[str, str]] = set()
    for term in terms:
        for factor in term.factors:
            obj = _source_tensor(factor)
            m = re.fullmatch(r"f_([ab]+)", obj.name)
            if not m:
                continue
            spaces = tuple(_space_char(idx) for idx in obj.indices)
            if spaces == ("o", "o"):
                space = "oo"
            elif spaces == ("v", "v"):
                space = "vv"
            else:
                space = "ov"
            used.add((space, m.group(1)))
    return sorted(used)


def _fock_view_bindings(terms, indent: int = 4) -> list[str]:
    """C++ lines binding one spin-resolved Fock view per (space, tag) used.

    Empty on the RCC path, where every `f` is a bare `reference.f_<space>` read.
    """
    pad = " " * indent
    return [
        f'{pad}const auto &f_{tag}_{space} = '
        f'*reference.spin_block("{space}", "{tag}").value();'
        for space, tag in _fock_blocks_used(terms)
    ]


def _canonical_eri_blocks_for(block_tag: str | None) -> dict[str, tuple[str, ...]]:
    """The stored canonical blocks available to `block_tag`.

    U3.2. RCC uses the seven `_CANONICAL_ERI_BLOCKS`, whose 8-fold orbit reaches
    all 16 o/v patterns. A UCC block cannot: two of the four permutations are not
    its symmetries (U3.0), so its orbits are smaller and more of them are needed
    to cover the same patterns. Measured on the CCSD UCC manifold, a mixed block
    reaches only 11 of 16 patterns from the seven, and four of the five it misses
    (`oovo`, `vooo`, `vovo`, `vovv`) are patterns the residuals actually read.

    So the block set is DERIVED from the tag's own symmetry group rather than
    shared: every four-index pattern is offered as a candidate, and the search in
    `_map_eri_tensor` picks the first whose orbit contains the requested pattern.
    That yields 6 stored arrays for a same-spin tag and 10 for a mixed one -- the
    same counts `build_ucc_spin_block_cache` (U3.1) is driven with, and the same
    counts `test_ucc_eri_symmetry` pins. The three sets must agree; they are one
    fact on three sides of the codegen boundary.
    """
    if block_tag is None:
        return _CANONICAL_ERI_BLOCKS

    permutations = eri_permutations_for_block(block_tag)
    blocks: dict[str, tuple[str, ...]] = {}
    covered: set[tuple[str, ...]] = set()
    # Deterministic order, and canonical members chosen occupied-first so the
    # names stay recognizable (`oovv`, not `vvoo`).
    for pattern in sorted(itertools.product("ov", repeat=4)):
        spaces = tuple(pattern)
        if spaces in covered:
            continue
        orbit = {tuple(spaces[i] for i in perm) for perm, _ in permutations}
        covered |= orbit
        blocks["".join(spaces)] = spaces
    return blocks


def _eri_read(block_tag: str | None, space: str) -> str:
    """The C++ expression naming the stored ERI array for (space, spin block).

    RCC reads the cache member directly (`mo_blocks.oovv`). UCC reads a per-block
    view bound once at the top of the kernel (`v_abab_oovv`), mirroring how the
    arbitrary-order amplitudes bind `t<rank>_<tag>` -- same mechanism, not a
    parallel one. Must agree exactly with `_eri_view_bindings`, which declares
    these names; they are the two halves of one convention.
    """
    if block_tag is None:
        return f"mo_blocks.{space}"
    return f"v_{block_tag}_{space}"


def _map_factor(
    tensor: Tensor | LoweredTensorFactor,
    intermediate_names: frozenset[str] | None = None,
    arbitrary_amplitudes: bool = False,
) -> tuple[int, str]:
    tensor_obj = _source_tensor(tensor)
    indices = _access_indices(tensor)
    # `t<rank>` is the reference Sz sector; `t<rank>_<tag>` (R3.1.3d) is a higher
    # independent Sz sector of the same amplitude (t4_aaabaaab), stored/read as a
    # distinct block. Both are rank-`\d+` amplitudes; the tag routes the read to
    # the sector's own tensor.
    amplitude_match = re.fullmatch(r"t(\d+)(?:_([ab]+))?", tensor_obj.name)

    # v/f may carry a UCC spin-block suffix (`v_abab`, `f_aa`). RCC emits bare
    # `v`/`f`, so this is a strict superset: `re.fullmatch(r"v(?:_([ab]+))?", "v")`
    # matches with a None tag and takes exactly the old path.
    #
    # U3.2: the tag is CARRIED, not stripped and discarded. F2.0b's comment here
    # said it "does not change which space block it is, so the mapping below is
    # unchanged once the suffix is stripped" -- right about the SPACE block
    # (`oovv` stays `oovv`) and wrong about the ARRAY: under UHF <aa|aa>, <ab|ab>
    # and <bb|bb> are three different integrals. It also selects which symmetries
    # may be used to reach the block, since two of the four are not symmetries of
    # a mixed block (U3.0). Both go through `_map_eri_tensor`.
    integral_match = re.fullmatch(r"([vf])(?:_([ab]+))?", tensor_obj.name)
    integral_root = integral_match.group(1) if integral_match else None
    integral_tag = integral_match.group(2) if integral_match else None

    if integral_root == "f":
        # U3.3: same tag routing as the ERIs, and simpler by nature. The Fock is
        # two-index, so BOTH slots carry the same spin -- there is no mixed block,
        # no permutation-validity question, and nothing to enumerate. Collapsing
        # `vo` onto `ov` stays correct because the Fock is symmetric, and that
        # reorder is spin-safe precisely because a two-index tag cannot mix spins
        # the way <ab|ab> does.
        left, right = indices
        if left.space == "occ" and right.space == "occ":
            return 1, _fock_read(integral_tag, "oo", left.name, right.name)
        if left.space == "vir" and right.space == "vir":
            return 1, _fock_read(integral_tag, "vv", left.name, right.name)
        occ = next((idx for idx in indices if idx.space == "occ"), None)
        vir = next((idx for idx in indices if idx.space == "vir"), None)
        if occ is not None and vir is not None:
            return 1, _fock_read(integral_tag, "ov", occ.name, vir.name)
        raise NotImplementedError(f"Unsupported Fock block for {tensor_obj!r}")

    if integral_root == "v":
        return _map_eri_tensor(tensor, integral_tag)

    if amplitude_match is not None:
        excitation_rank = int(amplitude_match.group(1))
        sector_tag = amplitude_match.group(2)      # None for the reference sector
        occ = [idx.name for idx in indices if idx.space == "occ"]
        vir = [idx.name for idx in indices if idx.space == "vir"]
        if len(occ) != excitation_rank or len(vir) != excitation_rank:
            raise ValueError(f"Invalid amplitude tensor layout for {tensor_obj!r}")
        # The arbitrary-order runtime type (ArbitraryOrderRCCAmplitudes, rank ≥ 4
        # methods) exposes only `.tensor(rank)` — returning std::expected<view> —
        # with no t1/t2/t3 members. The view is bound ONCE per kernel/builder as a
        # local `t<rank>` (see _amplitude_view_bindings); the factor then indexes
        # that local with an initializer-list `t<rank>({...})`. The rank ≤ 3
        # tensor_backend types (RCCSD/RCCSDTAmplitudes) have direct t1/t2/t3(...)
        # accessors instead.
        if arbitrary_amplitudes:
            # R3.1.3d: a higher Sz sector reads from its own bound view
            # `t<rank>_<tag>` (sector_tag), the reference from `t<rank>`.
            view = (f"t{excitation_rank}_{sector_tag}" if sector_tag
                    else f"t{excitation_rank}")
            return 1, f"{view}({{{', '.join(occ + vir)}}})"
        if excitation_rank == 1:
            return 1, f"amplitudes.t1({occ[0]}, {vir[0]})"
        if excitation_rank == 2:
            return 1, f"amplitudes.t2({', '.join(occ + vir)})"
        if excitation_rank == 3:
            return 1, f"amplitudes.t3({', '.join(occ + vir)})"
        return 1, (
            f"amplitudes.tensor({excitation_rank})"
            f"({{{', '.join(occ + vir)}}})"
        )

    if tensor_obj.name == "D":
        occ = [idx.name for idx in indices if idx.space == "occ"]
        vir = [idx.name for idx in indices if idx.space == "vir"]
        rank = len(indices)
        excitation_rank = rank // 2
        if rank == 2 and len(occ) == 1 and len(vir) == 1:
            return 1, f"denominators.d1({occ[0]}, {vir[0]})"
        if rank == 4 and len(occ) == 2 and len(vir) == 2:
            return 1, f"denominators.d2({', '.join(occ + vir)})"
        if rank == 6 and len(occ) == 3 and len(vir) == 3:
            return 1, f"denominators.d3({', '.join(occ + vir)})"
        if rank > 0 and len(occ) == excitation_rank and len(vir) == excitation_rank:
            return 1, (
                f"denominators.tensor({excitation_rank})"
                f"({{{', '.join(occ + vir)}}})"
            )
        raise ValueError(f"Invalid denominator tensor layout for {tensor_obj!r}")

    if tensor_obj.name == "delta":
        left, right = indices
        return 1, f"(({left.name} == {right.name}) ? 1.0 : 0.0)"

    # Extracted intermediates -- CSE "W_*", the tau/tau_c pseudo-amplitudes, and
    # (D7.3) the recognized dressed operators (Wmnij/Wabef/Wmbej/Fae/...) -- are
    # materialized locals built once per kernel; reference them by name.  Any
    # factor named in ``intermediate_names`` (the specs passed to the kernel)
    # resolves as such a local; the W_/tau prefixes stay as a fallback for the
    # CSE path that does not thread the name set.
    if (
        tensor_obj.name.startswith("W_")
        or tensor_obj.name == "tau"
        or (intermediate_names is not None and tensor_obj.name in intermediate_names)
    ):
        return 1, _target_expr(tensor_obj.name, indices)

    raise NotImplementedError(f"Unsupported tensor factor {tensor_obj!r}")


def emit_planck_term(
    term: AlgebraTerm | RestrictedClosedShellTerm,
    lhs: str = "result",
    indent: int = 4,
    intermediate_names: frozenset[str] | None = None,
    arbitrary_amplitudes: bool = False,
    ucc: bool = False,
) -> str:
    """Emit a single algebraic term using Planck tensor accessors.

    ``intermediate_names`` lists the materialized intermediates in scope (the
    kernel's specs) so their factors resolve as local references (D7.3).
    ``arbitrary_amplitudes`` routes t-amplitude accessors through
    ``.tensor(rank)(...)`` for the arbitrary-order runtime type (rank ≥ 4)."""
    # U3b.2a: the per-index spin map, for spin-resolved loop bounds. An `Index`
    # carries only a space (occ/vir) -- the U1 bridge drops spin by design -- so
    # the spin has to come from the FACTORS, where slot k of `t2_abab` / `v_aaaa`
    # carries `tag[k]`.
    #
    # GATED ON `ucc`, NOT on the map coming back empty. An earlier version relied
    # on "non-UCC terms have no tagged factors", which is FALSE for the
    # SPIN-ADAPTED path: its rank-4 sector amplitudes are named `t4_aaabaaab`,
    # which matches the same `_[ab]+` suffix. That emitted `noa`/`nvb` bounds into
    # a spin-adapted kernel whose preamble declares only `no`/`nv`, and the TU
    # stopped compiling (caught by
    # `test_spin_adapted_ccsdtq_compiles_with_sector_accessor`, not by the RCC
    # SHA-256 pin -- that pin emits with spin_adapt=False and never sees this
    # path).
    index_spins = ucc_term_index_spins(term) if ucc else {}

    pad = " " * indent
    lines: list[str] = []

    if isinstance(term, RestrictedClosedShellTerm):
        free = list(term.canonical_free_indices)
        summed = list(term.canonical_summed_indices)
        factors: Sequence[Tensor | LoweredTensorFactor] = term.factors
    else:
        free = list(term.free_indices)
        summed = list(term.summed_indices)
        factors = term.factors

    for idx in free:
        lines.append(
            f"{pad}for (int {idx.name} = 0; {idx.name} < {_loop_bound(idx, index_spins.get(idx.name))}; ++{idx.name})"
        )

    lines.append(f"{pad}{{")

    sign = 1
    factor_exprs: list[str] = []
    for factor in factors:
        factor_sign, factor_expr = _map_factor(
            factor, intermediate_names, arbitrary_amplitudes)
        sign *= factor_sign
        factor_exprs.append(factor_expr)

    coeff = term.coeff * sign
    product = " * ".join(factor_exprs) if factor_exprs else "1.0"

    target = _target_expr(lhs, free)

    if summed:
        lines.append(f"{pad}    double acc = 0.0;")
        for idx in summed:
            lines.append(
                f"{pad}    for (int {idx.name} = 0; {idx.name} < {_loop_bound(idx, index_spins.get(idx.name))}; ++{idx.name})"
            )
        lines.append(f"{pad}        acc += {_coeff_literal(coeff)}{product};")
        lines.append(f"{pad}    {target} += acc;")
    else:
        lines.append(f"{pad}    {target} += {_coeff_literal(coeff)}{product};")

    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _amplitude_type(method: str, force_arbitrary: bool = False) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if force_arbitrary or max_rank >= 4:
        return "ArbitraryOrderRCCAmplitudes"
    if max_rank >= 3:
        return "RCCSDTAmplitudes"
    return "RCCSDAmplitudes"


def _amplitude_ranks_used(terms) -> list[tuple[int, str | None]]:
    """Distinct t-amplitude `(rank, sector_tag)` pairs referenced across `terms`,
    sorted. `sector_tag` is None for the reference Sz sector (`t<rank>`) and the
    tag string for a higher sector (`t<rank>_<tag>`, R3.1.3d). Drives the
    per-kernel view bindings for the arbitrary-order runtime."""
    used: set[tuple[int, str | None]] = set()
    for term in terms:
        for factor in term.factors:
            m = re.fullmatch(r"t(\d+)(?:_([ab]+))?", _source_tensor(factor).name)
            if m:
                used.add((int(m.group(1)), m.group(2)))
    return sorted(used, key=lambda rt: (rt[0], rt[1] or ""))


def _amplitude_view_bindings(terms, indent: int = 4) -> list[str]:
    """C++ lines binding one amplitude view per (rank, sector) used — the
    arbitrary-order runtime returns std::expected from its accessors, so the view
    is unwrapped once here and indexed in the loops. `.value()` is safe: the
    solver constructs every rank ≤ max before calling. The reference sector binds
    `t<rank> = amplitudes.tensor(rank)`; a higher Sz sector (R3.1.3d) binds
    `t<rank>_<tag> = amplitudes.sector_tensor(rank, "<tag>")`."""
    pad = " " * indent
    lines: list[str] = []
    for rank, tag in _amplitude_ranks_used(terms):
        if tag is None:
            lines.append(f"{pad}const auto t{rank} = amplitudes.tensor({rank}).value();")
        else:
            lines.append(
                f'{pad}const auto t{rank}_{tag} = '
                f'amplitudes.sector_tensor({rank}, "{tag}").value();')
    return lines


def _eri_blocks_used(terms) -> list[tuple[str, str]]:
    """Distinct (space pattern, spin tag) ERI blocks referenced across `terms`.

    U3.2. Each becomes one bound view at the top of the kernel. The SPACE pattern
    is the canonical block the symmetry search lands on -- not the factor's raw
    index pattern -- so this resolves each factor exactly the way `_map_eri_tensor`
    will, rather than re-deriving it and risking the two disagreeing.
    """
    used: set[tuple[str, str]] = set()
    for term in terms:
        for factor in term.factors:
            obj = _source_tensor(factor)
            m = re.fullmatch(r"v_([ab]+)", obj.name)
            if not m:
                continue
            tag = m.group(1)
            spaces = tuple(_space_char(idx) for idx in obj.indices)
            patterns = [spaces]
            if _block_needs_explicit_exchange(tag):
                # R5: a same-spin read also emits its KET-SWAPPED partner, which
                # generally lives in a DIFFERENT stored block (ovov -> ovvo), so
                # that block needs a bound view too. Missing it is a compile
                # error, not a wrong number -- which is how the omission was
                # caught, and why this must stay in step with `_map_eri_tensor`.
                patterns.append((spaces[0], spaces[1], spaces[3], spaces[2]))
            for pattern in patterns:
                block = _resolve_eri_block_name(pattern, tag)
                if block is not None:
                    used.add((block, tag))
    return sorted(used)


def _eri_view_bindings(terms, indent: int = 4) -> list[str]:
    """C++ lines binding one spin-blocked ERI view per (space, tag) used.

    Mirrors `_amplitude_view_bindings` deliberately: the UCC block cache returns
    `std::expected<const Tensor4D *>` from `spin_block`, so the pointer is
    unwrapped once per kernel and the loops index it, exactly as the
    arbitrary-order amplitudes bind `t<rank>_<tag>` once. Same mechanism, not a
    parallel one.

    Emits nothing on the RCC path, where every `v` is a bare `mo_blocks.<block>`
    read and no view is needed -- which is what keeps the RCC emit byte-identical.
    """
    pad = " " * indent
    lines: list[str] = []
    for space, tag in _eri_blocks_used(terms):
        lines.append(
            f'{pad}const auto &v_{tag}_{space} = '
            f'*mo_blocks.spin_block("{space}", "{tag}").value();')
    return lines


def _denominator_type(method: str, force_arbitrary: bool = False) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if force_arbitrary or max_rank >= 4:
        return "ArbitraryOrderDenominatorCache"
    return "DenominatorCache"


def _target_expr(lhs: str, indices: Sequence[Index]) -> str:
    if not indices:
        return lhs
    if len(indices) in (2, 4, 6):
        return f"{lhs}({', '.join(idx.name for idx in indices)})"
    return f"{lhs}({{{', '.join(idx.name for idx in indices)}}})"


def _kernel_name(method: str, target: str, ucc: bool = False) -> str:
    """The C++ name of one emitted kernel.

    U5.0: `ucc=True` prefixes the symbol so a UCC translation unit does not
    collide with the RCC one for the same method. Measured before this landed,
    at a rank where both sides emit a bundle: `compute_ccsdt_energy` AND
    `make_generated_ccsdt_kernels` were defined by both. (Rank 2 shows only the
    first, because RCC emits no bundle below the arbitrary floor -- so measuring
    there alone understates the collision.)

    The prefix goes on the emitted NAME only, never on `method` itself:
    `method` is also parsed for the excitation rank (`parse_cc_level`), which
    requires a 'cc' prefix and rejects `ucc_ccsd`. Overloading one string for
    both roles is what makes this look like a one-line change and then fail at
    the rank parse.
    """
    prefix = "ucc_" if ucc else ""
    if target == "energy":
        return f"compute_{prefix}{method}_energy"
    return f"compute_{prefix}{method}_{target}_residual"


def _target_rank(
    target: str,
    terms: Sequence[AlgebraTerm],
    lowered_terms: Sequence[RestrictedClosedShellTerm] | None = None,
) -> int:
    if target == "energy":
        return 0
    if lowered_terms:
        return len(lowered_terms[0].canonical_free_indices) // 2
    if terms:
        return len(terms[0].free_indices) // 2
    raise ValueError(f"Cannot infer excitation rank for target {target!r}")


def _builder_symbol(method: str, name: str) -> str:
    """The C++ symbol for intermediate ``name``'s builder in ``method``'s TU (V1.3.2).

    Method-suffixed (``build_tau_ccsdt``), because the kernel registry ``#include``s several
    generated TUs into ONE translation unit. Unsuffixed builders collide there: in
    arbitrary-order form every method's builders take ``ArbitraryOrderRCCAmplitudes``, so the
    signatures are identical and co-inclusion is a REDEFINITION, not an overload -- measured, 5
    errors, one per dressed builder (gated by ``test_dressed_tu_coinclusion``). In the plain
    per-method TUs the amplitude type differs, so those happen to overload cleanly; the suffix
    makes both cases correct by construction instead of relying on that accident.

    Chosen over restricting dressing to one rank and enforcing it: the collision is a property of
    the NAMING SCHEME, not of how many ranks are enabled, so a scope restriction would leave the
    trap armed for whoever enables a second dressed rank. Same reasoning as V1.1e.2 -- fix the
    mechanism, not each caller.

    One function for all four emission sites (the definition plus three call sites), so a
    definition and its calls cannot drift apart.
    """
    return f"build_{name}_{method.lower()}"


def _emit_kernel(
    method: str,
    target: str,
    terms: Sequence[AlgebraTerm],
    lowered_terms: Sequence[RestrictedClosedShellTerm] | None = None,
    intermediates: Sequence[IntermediateSpec] | None = None,
    free_indices: Sequence[Index] | None = None,
    force_arbitrary: bool = False,
    ucc: bool = False,
) -> str:
    lowered_terms = tuple(lowered_terms or ())
    if free_indices is None:
        if lowered_terms:
            free_indices = lowered_terms[0].canonical_free_indices
        else:
            free_indices = terms[0].free_indices if terms else ()
    free_indices = tuple(free_indices)
    result_rank = len(free_indices)
    result_type = _tensor_type(result_rank)
    amplitude_type = _amplitude_type(method, force_arbitrary)
    denominator_type = _denominator_type(method, force_arbitrary)
    intermediate_map = {spec.name: spec for spec in intermediates or ()}

    lines: list[str] = []
    lines.append(f"{result_type} {_kernel_name(method, target, ucc)}(")
    lines.append("    const CanonicalRHFCCReference &reference,")
    lines.append("    const TensorCCBlockCache &mo_blocks,")
    lines.append(f"    const {denominator_type} &denominators,")
    lines.append(f"    const {amplitude_type} &amplitudes)")
    lines.append("{")
    lines.extend(_orbital_count_preamble(ucc))
    # U3b.2c: the result's extents are per-spin under UCC. The spins come from the
    # kernel's own terms -- every term of a block-tagged target carries the same
    # free-index spins (measured across the CCSD UCC manifold: 352 terms, zero
    # disagreements with the target's tag), so the first term is representative.
    result_spins = ucc_term_index_spins(terms[0]) if (ucc and terms) else None
    if result_rank == 0:
        lines.append("    double result = 0.0;")
    else:
        lines.append(
            f"    {result_type} result("
            f"{_dims_expr(free_indices, result_type, result_spins)});"
        )
    required_intermediates: list[IntermediateSpec] = []
    seen_intermediates: set[str] = set()
    for term in terms:
        for factor in term.factors:
            if factor.name in intermediate_map and factor.name not in seen_intermediates:
                seen_intermediates.add(factor.name)
                required_intermediates.append(intermediate_map[factor.name])
    if required_intermediates:
        lines.append("")
        lines.append("    // Build reused intermediates once for this kernel")
        for spec in required_intermediates:
            lines.append(
                f"    const auto {spec.name} = {_builder_symbol(method, spec.name)}("
                "reference, mo_blocks, denominators, amplitudes);"
            )
    lines.append("")
    lines.append(f"    // {target} kernel ({len(terms)} terms)")
    emitted_terms: Sequence[AlgebraTerm | RestrictedClosedShellTerm]
    emitted_terms = lowered_terms if lowered_terms else terms
    intermediate_names = frozenset(intermediate_map)
    arbitrary = amplitude_type == "ArbitraryOrderRCCAmplitudes"

    # Large kernels (the spin-adapted CCSDTQ quadruples residuals are ~5000
    # statements each) are super-linear to optimize as one function -- an -O3
    # compile of the containing TU takes 40+ min. Split them into `_partN`
    # sub-functions each accumulating a slice of the terms into `result` (passed
    # by reference); the compiler then optimizes N bounded functions in ~linear
    # total time. Small kernels stay inline (byte-identical). The intermediate
    # builds and amplitude-view bindings are re-emitted per part -- cheap, local,
    # and keeps each part self-contained. `result_rank == 0` (energy) stays inline
    # (a scalar accumulator can't be passed the same way and is always tiny).
    if result_rank > 0 and len(emitted_terms) > _KERNEL_CHUNK_TERMS:
        return _emit_chunked_kernel(
            method, target, lines, emitted_terms, result_type, free_indices,
            denominator_type, amplitude_type, required_intermediates,
            intermediate_names, arbitrary, ucc)

    if arbitrary:
        bindings = (_amplitude_view_bindings(emitted_terms)
                    + _eri_view_bindings(emitted_terms)
                    + _fock_view_bindings(emitted_terms))
        if bindings:
            lines.append("")
            lines.extend(bindings)
    for i, term in enumerate(emitted_terms, start=1):
        lines.append(f"    // Term {i}")
        lines.append(emit_planck_term(
            term, lhs="result", indent=4,
            intermediate_names=intermediate_names,
            arbitrary_amplitudes=arbitrary, ucc=ucc))
        lines.append("")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


# Terms-per-function threshold above which a residual kernel is split into
# `_partN` sub-functions (see _emit_kernel). Smaller = smaller functions = faster
# optimizer, with diminishing returns; measured g++-15 -O1 on the CCSDTQ registry
# object: 512→176s, 256→135s. 256 is the chosen balance (fewer, but still bounded,
# functions). The largest kernel (6871-term sector) splits into ~27 parts.
_KERNEL_CHUNK_TERMS = 256


def _emit_chunked_kernel(
    method, target, header_lines, emitted_terms, result_type, free_indices,
    denominator_type, amplitude_type, required_intermediates,
    intermediate_names, arbitrary, ucc=False,
):
    """Emit a large residual kernel as `_partN` sub-functions accumulating into a
    by-reference `result`, plus the main kernel that allocates `result`, calls
    each part, and returns it. Keeps per-function body size bounded so any -O
    level compiles in ~linear time. See _emit_kernel for why."""
    kernel = _kernel_name(method, target, ucc)
    n_parts = (len(emitted_terms) + _KERNEL_CHUNK_TERMS - 1) // _KERNEL_CHUNK_TERMS

    out: list[str] = []
    for p in range(n_parts):
        chunk = emitted_terms[p * _KERNEL_CHUNK_TERMS:(p + 1) * _KERNEL_CHUNK_TERMS]
        out.append(f"static void {kernel}_part{p}(")
        out.append(f"    {result_type} &result,")
        out.append("    const CanonicalRHFCCReference &reference,")
        out.append("    const TensorCCBlockCache &mo_blocks,")
        out.append(f"    const {denominator_type} &denominators,")
        out.append(f"    const {amplitude_type} &amplitudes)")
        out.append("{")
        out.extend(_orbital_count_preamble(ucc))
        if ucc:
            out.append("    (void)noa; (void)nob; (void)nva; (void)nvb;")
        else:
            out.append("    (void)no; (void)nv;")
        if required_intermediates:
            for spec in required_intermediates:
                out.append(
                    f"    const auto {spec.name} = {_builder_symbol(method, spec.name)}("
                    "reference, mo_blocks, denominators, amplitudes);")
        if arbitrary:
            bindings = (_amplitude_view_bindings(chunk)
                        + _eri_view_bindings(chunk)
                        + _fock_view_bindings(chunk))
            out.extend(bindings)
        for i, term in enumerate(chunk, start=1):
            out.append(emit_planck_term(
                term, lhs="result", indent=4,
                intermediate_names=intermediate_names,
                arbitrary_amplitudes=arbitrary, ucc=ucc))
        out.append("}")
        out.append("")

    # main kernel: allocate result, call the parts, return it. header_lines holds
    # the signature + result allocation already; strip its trailing setup we don't
    # reuse (the parts do their own no/nv + intermediates) by appending calls.
    body = list(header_lines)
    for p in range(n_parts):
        body.append(
            f"    {kernel}_part{p}(result, reference, mo_blocks, denominators, amplitudes);")
    body.append("    return result;")
    body.append("}")
    return "\n".join(out + body)


def _emit_intermediate_builder(
    method: str,
    spec: IntermediateSpec,
    sibling_names: frozenset[str] | None = None,
    factor_body: bool = False,
    stride_order: bool = False,
    force_arbitrary: bool = False,
    sibling_specs: dict[str, IntermediateSpec] | None = None,
) -> str:
    result_type = _tensor_type(spec.rank)
    amplitude_type = _amplitude_type(method, force_arbitrary)
    denominator_type = _denominator_type(method, force_arbitrary)
    lowered_definition_terms = tuple(
        lower_term_restricted_closed_shell(term, "reference")
        for term in spec.definition_terms
    )
    if lowered_definition_terms:
        builder_indices = lowered_definition_terms[0].canonical_free_indices
    else:
        builder_indices = spec.indices

    lines: list[str] = []
    lines.append(f"{result_type} {_builder_symbol(method, spec.name)}(")
    lines.append("    const CanonicalRHFCCReference &reference,")
    lines.append("    const TensorCCBlockCache &mo_blocks,")
    lines.append(f"    const {denominator_type} &denominators,")
    lines.append(f"    const {amplitude_type} &amplitudes)")
    lines.append("{")
    lines.append("    const int no = reference.orbital_partition.n_occ;")
    lines.append("    const int nv = reference.orbital_partition.n_virt;")
    lines.append(
        f"    {result_type} result({_dims_expr(builder_indices, result_type)});"
    )
    lines.append("")
    lines.append(
        f"    // Intermediate {spec.name} ({spec.index_space_sig}, usage={spec.usage_count})"
    )

    # V1.3: bind the SIBLING intermediates this definition references, the same way
    # `_emit_kernel` binds the ones its residual terms reference. `sibling_names` only
    # makes such a factor RENDER as a bare identifier; without a binding the emitted C++
    # said `tau(i, j, e, f)` with no `tau` in scope, so the dressed TU never compiled
    # (Wmnij and Wabef both reference tau). Emitted before the amplitude-view bindings
    # and the body so the declaration precedes every use.
    #
    # Ordering is the caller's responsibility: `intermediates` is dependency-ordered
    # (pseudo-amplitude specs first, then the operators referencing them), so a sibling's
    # own `build_` is already declared above this function. `sibling_specs` is passed as
    # an ordered mapping rather than recomputed here to keep that single source of truth.
    if sibling_specs:
        referenced: list[str] = []
        seen: set[str] = set()
        for term in lowered_definition_terms or spec.definition_terms:
            for factor in term.factors:
                name = getattr(factor, "name", None)
                if (name in sibling_specs and name != spec.name
                        and name not in seen):
                    seen.add(name)
                    referenced.append(name)
        if referenced:
            for name in referenced:
                lines.append(
                    f"    const auto {name} = {_builder_symbol(method, name)}("
                    "reference, mo_blocks, denominators, amplitudes);"
                )
            lines.append("")

    # M3.0: emit the builder body as a factored contraction tree (scratch-step
    # pairwise contractions) instead of one flat n-ary nest, when it factors
    # below the flat cost. Cuts the builder's own FLOP scaling (e.g.
    # W_t1t2v_oooovv o^5v^3 -> o^5v^2) at ~0.3x scratch memory. Falls back to the
    # flat lowered emit for single-step (<=2-factor) definitions.
    from ..optimization.factorize import factored_builder_steps
    steps = factored_builder_steps(spec, stride_order=stride_order)
    arbitrary = amplitude_type == "ArbitraryOrderRCCAmplitudes"
    if arbitrary:
        # bind amplitude views once; the factored steps and the flat definition
        # both reference the same leaf amplitudes, so bind from the definition.
        binding_terms = ([s for _, s in steps] if (factor_body and len(steps) > 1)
                         else lowered_definition_terms)
        bindings = (_amplitude_view_bindings(binding_terms)
                    + _eri_view_bindings(binding_terms)
                    + _fock_view_bindings(binding_terms))
        if bindings:
            lines.extend(bindings)
            lines.append("")
    if factor_body and len(steps) > 1:
        scratch_names = frozenset(
            lhs for lhs, _ in steps if lhs != "result")
        names_in_scope = (sibling_names or frozenset()) | scratch_names
        for lhs, step in steps:
            if lhs != "result":
                stype = _tensor_type(len(step.free_indices))
                lines.append(
                    f"    {stype} {lhs}({_dims_expr(step.free_indices, stype)});")
            lines.append(emit_planck_term(
                step, lhs=lhs, indent=4, intermediate_names=names_in_scope,
                arbitrary_amplitudes=arbitrary))
            lines.append("")
    else:
        for i, term in enumerate(lowered_definition_terms, start=1):
            lines.append(f"    // Definition term {i}")
            lines.append(emit_planck_term(
                term, lhs="result", indent=4, intermediate_names=sibling_names,
                arbitrary_amplitudes=arbitrary))
            lines.append("")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


def _emit_arbitrary_order_kernel_bundle(
    method: str,
    equations: dict[str, list[AlgebraTerm]],
    lowered_equations: dict[str, list[RestrictedClosedShellTerm]],
    force_arbitrary: bool = False,
    ucc: bool = False,
) -> str:
    max_rank = max(parse_cc_level(method), default=0)
    if max_rank < 4 and not force_arbitrary:
        return ""

    # Split targets into the per-rank REFERENCE residuals (residuals_by_rank) and
    # the higher-Sz SECTOR residuals (`quadruples_aaabaaab`, R3.1.3d), keyed
    # (rank, tag). The sector kernels are emitted as their own functions; B3
    # registers them in the bundle's `sector_residuals` + `sector_tags` so B1
    # allocates the sector amplitude block and B4 evaluates/updates it.
    ranked_targets: list[tuple[int, str]] = []
    sector_targets: list[tuple[int, str, str]] = []   # (rank, tag, target)
    for target, terms in equations.items():
        if target == "energy":
            continue
        rank = _target_rank(target, terms, lowered_equations.get(target))
        m = re.search(r"_([ab]+)$", target)
        if m:
            sector_targets.append((rank, m.group(1), target))
        else:
            ranked_targets.append((rank, target))
    ranked_targets.sort()
    sector_targets.sort()

    def _residual_lambda(target: str, indent: str) -> list[str]:
        return [
            f"{indent}[](",
            f"{indent}    const CanonicalRHFCCReference &reference,",
            f"{indent}    const TensorCCBlockCache &mo_blocks,",
            f"{indent}    const ArbitraryOrderDenominatorCache &denominators,",
            f"{indent}    const ArbitraryOrderRCCAmplitudes &amplitudes) -> TensorND",
            f"{indent}{{",
            f"{indent}    return to_tensor_nd("
            f"{_kernel_name(method, target, ucc)}(reference, mo_blocks, denominators, amplitudes));",
            f"{indent}}}",
        ]

    lines: list[str] = []
    factory = f"make_generated_{'ucc_' if ucc else ''}{method}_kernels"
    lines.append(f"GeneratedArbitraryOrderKernels {factory}()")
    lines.append("{")
    lines.append("    GeneratedArbitraryOrderKernels kernels;")
    lines.append(f"    kernels.max_excitation_rank = {max_rank};")
    lines.append(f"    kernels.energy = {_kernel_name(method, 'energy', ucc)};")
    lines.append(f"    kernels.residuals_by_rank.reserve({max_rank});")
    for rank, target in ranked_targets:
        lines.append("    kernels.residuals_by_rank.push_back(")
        lines.extend(_residual_lambda(target, "        "))
        lines.append("        );")
    for rank, tag, target in sector_targets:
        lines.append(f'    kernels.sector_tags.push_back({{{rank}, "{tag}"}});')
        lines.append("    kernels.sector_residuals.push_back(")
        lines.append(f'        {{{rank}, "{tag}",')
        lines.extend(_residual_lambda(target, "         "))
        lines.append("        });")
    lines.append("    return kernels;")
    lines.append("}")
    return "\n".join(lines)


def emit_planck_translation_unit(
    method: str,
    equations: dict[str, list[AlgebraTerm]],
    intermediates: Sequence[IntermediateSpec] | None = None,
    lowered_equations: dict[str, list[RestrictedClosedShellTerm]] | None = None,
    factor_builder_bodies: bool = False,
    force_arbitrary: bool = False,
    spin_adapted: bool = False,
    ucc: bool = False,
) -> str:
    """Emit a Planck-compatible C++ translation unit.

    ``factor_builder_bodies`` (M3.0): emit each intermediate's `build_W` as a
    factored contraction tree (scratch-step pairwise contractions) instead of one
    flat n-ary loop nest, cutting the builder's own FLOP scaling. Default off ->
    byte-identical flat emit.

    ``spin_adapted`` (R3.1.3d): the ``equations`` are ALREADY spatial RCC
    ``AlgebraTerm``s from ``spin_adapt_equations`` -- the genuine spatial
    contraction (2*(direct)-(exchange)), the whole point of the spin adaptation.
    Skip the relabel-only ``lower_equations_restricted_closed_shell`` (which would
    both re-apply the defect it was meant to replace AND crash on the multi-Sz
    target names like ``quadruples_aaabaaab``); emit the AlgebraTerms directly."""
    method = method.lower()
    if spin_adapted:
        lowered_equations = {}                  # emit AlgebraTerms directly
    else:
        lowered_equations = lowered_equations or lower_equations_restricted_closed_shell(
            equations
        )
    lines: list[str] = []
    lines.append("// Auto-generated by ccgen")
    lines.append(f"// Planck tensor kernels for {method.upper()}")
    lines.append("")
    lines.append('#include "post_hf/cc/tensor_backend.h"')
    if force_arbitrary or max(parse_cc_level(method), default=0) >= 4:
        lines.append('#include "post_hf/cc/generated_arbitrary_runtime.h"')
    lines.append("")
    lines.append("namespace HartreeFock::Correlation::CC")
    lines.append("{")
    lines.append("")

    if intermediates:
        sibling_names = frozenset(spec.name for spec in intermediates)
        # Ordered by construction: `intermediates` is dependency-ordered, so a sibling
        # referenced by a later spec already has its `build_` emitted above it (V1.3).
        sibling_spec_map = {spec.name: spec for spec in intermediates}
        for spec in intermediates:
            if not _is_supported_tensor_rank(spec.rank):
                continue
            lines.append(_emit_intermediate_builder(
                method, spec, sibling_names, factor_body=factor_builder_bodies,
                stride_order=factor_builder_bodies, force_arbitrary=force_arbitrary,
                sibling_specs=sibling_spec_map))
            lines.append("")

    for target, terms in equations.items():
        lines.append(
            _emit_kernel(
                method,
                target,
                terms,
                lowered_terms=lowered_equations.get(target),
                intermediates=intermediates,
                force_arbitrary=force_arbitrary,
                ucc=ucc,
            )
        )
        lines.append("")

    bundle_code = _emit_arbitrary_order_kernel_bundle(
        method,
        equations,
        lowered_equations,
        force_arbitrary=force_arbitrary,
        ucc=ucc,
    )
    if bundle_code:
        lines.append(bundle_code)
        lines.append("")

    lines.append("} // namespace HartreeFock::Correlation::CC")
    return "\n".join(lines)
