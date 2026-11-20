"""CCSD(T) perturbative-triples generation (docs/CCSD_T_SCOPE.md step 1).

Not a residual: (T) needs two triples-manifold projections of a bare operator
product, no BCH expansion,

    W_ijk^abc = <Phi_ijk^abc | (V_N T2)_C | 0>     connected
    V_ijk^abc = <Phi_ijk^abc | V_N T1 | 0>         disconnected only

and the scalar  E(T) = (1/36) sum (W + V) W / D_ijk^abc  in spin orbitals,
with a canonical reference (f_ov = 0) so F_N contributes only through D.
"""

from __future__ import annotations

from ..algebra import multiply
from ..cluster import build_tn
from ..generate import _generate_manifold_equation
from ..hamiltonian import build_two_body_operator
from ..project import AlgebraTerm


def _project_triples(product, connected_only: bool) -> list[AlgebraTerm]:
    # ponytail: no bucket_terms_by_manifold pre-filter. Its block-connectivity
    # test assumes BCH commutator terms (connected by construction) and drops
    # the disconnected V_N T1 product outright; two bare products need no
    # pre-pruning.
    _, _, canonical = _generate_manifold_equation(
        "triples", tuple(product.terms), connected_only,
        collect_denominators=False, permutation_grouping=False,
        exploit_symmetry=False, parallel_workers=1,
    )
    return canonical


def build_ccsd_t_equations() -> dict[str, list[AlgebraTerm]]:
    """Return ``{"W": [...], "V": [...]}`` triples-manifold term lists."""
    vn = build_two_body_operator()
    w = _project_triples(multiply(vn, build_tn(2)), connected_only=True)
    # V_N T1 cannot reach triples through a connected product (it tops out at
    # doubles), so the unfiltered projection is the disconnected term alone.
    # Assert it: a connected term here would mean the projector or the
    # connectivity flag is wrong, and it would silently enter E(T).
    v_all = _project_triples(multiply(vn, build_tn(1)), connected_only=False)
    assert all(not t.connected for t in v_all), "connected term in V_N T1 triples"
    assert all({f.name for f in t.factors} == {"t1", "v"} for t in v_all)
    return {"W": w, "V": v_all}


def build_ccsd_t_restricted() -> dict[str, list[AlgebraTerm]]:
    """Closed-shell (T): W and V lowered to spatial orbitals on the aab|aab
    block (a, b, i, j alpha; c, k beta), in spatial t1, t2 (abab) and the
    non-antisymmetrized <pq|rs>.

    That one block carries the whole energy. Closed shell gives aaa = bbb and
    aab = abb, and the all-alpha block is half the full antisymmetrizer over the
    three virtuals applied to the aab block (exact, measured to 4e-14 for both
    W and V):

        X_aaa(abc, ijk) = 1/2 sum_P sign(P) X_aab(P(abc), ijk)

    so with S_blk = sum (W + V) W / D over a block,

        E(T) = (2 S_aaa + 18 S_aab) / 36 = S_aaa / 18 + S_aab / 2.
    """
    from ..spin import spin_adapt_equations

    eqs = build_ccsd_t_equations()
    for terms in eqs.values():
        # spin_adapt_equations assigns the block by slot position over these
        # names; a renamed external would silently land on another block.
        assert {i.name for i in terms[0].free_indices} == set("abcijk")
    return spin_adapt_equations(eqs)
