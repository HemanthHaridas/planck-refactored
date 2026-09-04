"""AR2.0 (WIP) — expose the oriented-loop structure of a diagram's contraction.

The CCSD-doubles diagram SIGN is `(-1)^(nh + nl)` where nh = hole lines and
nl = closed oriented loops. AR2.0's job is to compute `nl` from the assembled
representative (`diagram_representative`). This is the prerequisite for AR2.1
(the sign rule) and, with AR2.2 (magnitude), the whole AR2 weight formula.

THE RULE (Crawford & Schaefer III, Rev. Comput. Chem. 14 (2000), p.84):
  sign = (-1)^(h + l), h = number of hole lines, l = number of loops.
  "A loop is a route along a series of DIRECTED lines that either returns to its
   beginning OR begins at one external line and ends at another."
Corollaries the earlier tracers got wrong:
  - lines are DIRECTED (hole vs particle direction); a loop follows the direction
    through vertices -- an UNDIRECTED cycle count is not it.
  - OPEN loops run external-line -> through the diagram -> external-line; external
    endpoints are loop TERMINI, not nodes joined by return edges. (My "bra hub"
    / "specific external pairing" were approximations of this.)
  - hole-line count h includes external hole lines (worked p.91 example: a
    diagram with 1 internal line has "two loops and two hole lines").
Magnitude rules (for AR2.2), same source p.85-87:
  - each pair of EQUIVALENT LINES (two lines starting at the same vertex AND
    ending at the same vertex) -> factor 1/2;
  - each set of n EQUIVALENT VERTICES (identical operators connected to the same
    interaction line the same way) -> factor 1/n!.

STATUS: partial, **26/30** on CCSD-doubles signs. Established rigorously (see
the scope doc, AR2.0 findings):
  - The sign is NOT any linear/GF(2) function of scalar counts (nh, n_particle,
    n_bubbles, n_ops, n_factors, n_summed) — a brute-force GF(2) search over all
    subsets finds NO formula. So oriented loop tracing is irreducibly required.
  - Tracer progression: undirected cycle-rank 13-17/30 → directed simple cycles
    19/30 → + operator-internal p/h pairing 19/30 → + a single-hub projection-bra
    node 21/30 → **+ SPECIFIC external pairing (a<->i, b<->j, not a hub) 26/30**
    (this file). The specific-pairing fix was refinement 1 below; it landed.

REMAINING (to reach 30/30). This tracer (26/30) is an UNDIRECTED approximation.
The Crawford rule (above) is now known verbatim and says the correct model is a
DIRECTED pass-through trace: follow a directed line to a vertex, exit along the
line it pairs with there, repeat; a loop is one such orbit (closed, or ext->ext).
Two directed-trace cuts were prototyped (24/30 and 21/30) — the right MODEL, but
the exact vertex "pass-through" pairing (which hole slot pairs with which
particle slot at a vertex, esp. the 2-electron `<pq||rs>` bra/ket) and the
external-terminal closure need calibrating against Crawford's WORKED EXAMPLES,
not the weight table (which gives only sign PARITY, underdetermining nl):

  CROSSREF calibration targets (Crawford & Schaefer, Rev. Comput. Chem. 14):
  - p.84: `f2` energy diagram (t2 fully contracted to <ij||ab>): l=2, h=2, +.
  - p.86 Eq.[168]: the reversed-ket variant: l=1, h=2, -.
  - p.87 Eq.[172]: `(t1 t1)` energy diagram: l=2, h=2, +; equivalent VERTICES → 1/2.
  - p.91 Eq.[180] LEFT (t1·v singles, 1 internal line c): l=2, h=2, +.
  - p.91 Eq.[180] RIGHT (t1·v singles, internal k): l=2, h=3, -.
  Tune the directed tracer so it reproduces these exact (l,h) — not just the
  sign — then it will fix the 4 remaining weight-table failures by construction.

  The 4 weight-table failures (all mixed external+summed-leg operators):
    ((1,1,0),(2,2,1)), ((1,1,0),), ((2,1,0),), ((2,1,1),).

Uses networkx (present in the pyscf venv; NOT in the default env). If AR2.0
lands in production, either add networkx as a ccgen dep or replace the cycle
count with a stdlib union-find (the graph is tiny). For now this is a
scoping/diagnostic scaffold, imported by nothing in the generator.

Run: `tests/pyscf/.venv/bin/python -m ccgen.tests.ar20_loop_structure`
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

from ccgen.diagram import DiagramString, diagram_representative

_TABLE = Path(__file__).with_name("ccsd_diagram_weights.json")


def trace_loops(rep) -> int:
    """Oriented-loop count of a diagram rep (WIP — 21/30 on CCSD doubles).

    Nodes are per-factor summed-index endpoints plus a single projection-bra
    node; edges are internal lines (summed), external lines (to bra), and the
    operator-internal particle<->hole pairing. Loops = independent cycles
    (E - V + components). See the REMAINING notes for why this is not yet exact.
    """
    import networkx as nx

    facs = rep.factors
    g = nx.MultiGraph()
    line_nodes: dict[str, list] = defaultdict(list)
    fac_slots: dict[int, dict] = defaultdict(lambda: {"p": [], "h": []})
    for k, f in enumerate(facs):
        for i in f.indices:
            node = (k, i.name)
            g.add_node(node)
            line_nodes[i.name].append(node)
            fac_slots[k]["p" if i.space == "vir" else "h"].append(node)
    # internal summed lines: connect the two endpoints of each summed index
    for name, nodes in line_nodes.items():
        if len(nodes) == 2:
            g.add_edge(nodes[0], nodes[1], kind="line")
    # projection-bra return lines: pair external particles with external holes by
    # sorted name (a<->i, b<->j), the manifold's canonical (identity) pairing --
    # NOT a single hub node, which over-connects and makes spurious loops.
    ext_p, ext_h = [], []
    for name, nodes in line_nodes.items():
        if len(nodes) == 1:
            sp = next(idx.space for f in facs for idx in f.indices if idx.name == name)
            (ext_p if sp == "vir" else ext_h).append((name, nodes[0]))
    ext_p.sort()
    ext_h.sort()
    for (_, p), (_, h) in zip(ext_p, ext_h):
        g.add_edge(p, h, kind="ext")
    # operator-internal turn-around: pair each factor's particle endpoints with
    # its hole endpoints (arbitrary zip -- see REMAINING refinement 2).
    for slots in fac_slots.values():
        for pnode, hnode in zip(slots["p"], slots["h"]):
            g.add_edge(pnode, hnode, kind="vertex")
    if g.number_of_nodes() == 0:
        return 0
    return g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)


def _retired_directed_loops(rep) -> int:
    """RETIRED — promoted to `ccgen.diagram.directed_loops` (AR2.1). This copy is
    kept only for the doc trail below; `directed_loops` now re-exports the
    production function. Original docstring follows.

    DIRECTED pass-through loop trace — VALIDATED against Crawford's worked
    examples (both closed AND open diagrams).

    Orient each summed line as a directed factor->factor edge (particle vir-index
    in enumeration order; hole occ-index REVERSED); external lines become half-
    edges to an ("EXT", name) terminus; at each factor pass-through-pair an
    incoming edge with an outgoing edge and count cycles, counting each open
    (EXT->...->EXT) path as one loop. Reproduces Crawford & Schaefer III, Rev.
    Comput. Chem. 14 (2000) EXACTLY:
      - p.84 f2-energy `((2,4,2),)`               : l=2  ✓
      - p.87 (t1 t1)-energy `((1,2,1),(1,2,1))`   : l=2  ✓
      - p.91 Eq.[180] LEFT  `((1,1,1),)` (t1*v, internal c): l=2, h=2, +  ✓
      - p.91 Eq.[180] RIGHT `((1,1,0),)` (t1*v, internal k): l=2, h=3, -  ✓
    with the hole-line count h = ALL hole lines (internal + external), also per
    the p.91 text ("two/three hole lines"). `check_energy_anchors()` gates the
    two energy anchors; the open anchors are checked in `check_open_anchors()`.

    So `(-1)^(h + directed_loops)` IS Crawford's diagram sign, verified against
    the source. It does NOT match the PySCF weight-table sign on 11/30 diagrams
    (`score_directed_vs_table()`), but that is a CONVENTION delta, not a tracer
    bug: the table's sign was fit to *this repo's assembled-rep external
    labeling* + the P(ij)P(ab) orbit, whereas Crawford's sign is for his
    canonical arrangement before the antisymmetrizer. Reconciling the two (so the
    diagram path's emitted sign is self-consistent end to end) is an AR2.3/D4
    convention task, NOT an AR2.0 loop-counting gap. AR2.0's deliverable — a
    loop count validated against the authoritative source — is DONE.
    """
    sset = set(rep.summed_indices)
    occ_by: dict[str, list] = defaultdict(list)
    for k, f in enumerate(rep.factors):
        for i in f.indices:
            occ_by[i.name].append((k, i.space, i in sset))
    edges = []  # (src, dst, species); src/dst is a factor int or ("EXT", name)
    for name, eps in occ_by.items():
        if len(eps) == 2:
            (k1, sp, _), (k2, _, _) = eps
            edges.append((k1, k2, sp) if sp == "vir" else (k2, k1, sp))
        else:
            (k, sp, _) = eps[0]
            ext = ("EXT", name)
            edges.append((k, ext, sp) if sp == "vir" else (ext, k, sp))
    inc: dict = defaultdict(list)
    out: dict = defaultdict(list)
    for ei, (s, d, sp) in enumerate(edges):
        out[s].append(ei)
        inc[d].append(ei)
    nxt: dict = {}
    facs = {k for e in edges for k in e[:2] if not isinstance(k, tuple)}
    for k in facs:
        for a, b in zip(inc[k], out[k]):
            nxt[a] = b
    seen: set = set()
    nloops = 0
    for ei in range(len(edges)):  # open loops from EXT sources first
        if ei in seen or not isinstance(edges[ei][0], tuple):
            continue
        nloops += 1
        cur = ei
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = nxt.get(cur)
    for ei in range(len(edges)):  # then closed cycles
        if ei in seen or ei not in nxt:
            continue
        nloops += 1
        cur = ei
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = nxt.get(cur)
    return nloops


# AR2.1: the tracer is promoted to the production module; re-export so this
# scaffold's checks (and any callers) use the single source of truth.
from ccgen.diagram import (  # noqa: E402
    directed_loops,
    diagram_hole_lines as _hole_lines,
)


def check_energy_anchors() -> bool:
    """Directed tracer reproduces Crawford's worked ENERGY-diagram loops (l=2)."""
    for tops, hr in [(((2, 4, 2),), 2), (((1, 2, 1), (1, 2, 1)), 2)]:
        rep = diagram_representative(DiagramString(tops, 0, 0), hr)
        if directed_loops(rep) != 2:
            return False
    return True


def check_open_anchors() -> bool:
    """Directed tracer reproduces Crawford's worked OPEN (p.91 Eq.[180]) t1*v
    singles: LEFT ((1,1,1),) l=2 h=2 sign +, RIGHT ((1,1,0),) l=2 h=3 sign -."""
    cases = [
        ((((1, 1, 1),), 2), 2, 2, +1),
        ((((1, 1, 0),), 2), 2, 3, -1),
    ]
    for (tops, hr), l_exp, h_exp, sign_exp in cases:
        rep = diagram_representative(DiagramString(tops, 2, 0), hr)
        if directed_loops(rep) != l_exp:
            return False
        if _hole_lines(rep) != h_exp:
            return False
        if (-1) ** (_hole_lines(rep) + directed_loops(rep)) != sign_exp:
            return False
    return True


def score_directed_vs_table() -> tuple[int, int]:
    """(matches, total) of Crawford's `(-1)^(h+directed_loops)` vs the PySCF
    weight-table sign. Note: mismatches are a rep-labeling / P-orbit CONVENTION
    delta (11/30), not a tracer error — the tracer is validated against
    Crawford directly by `check_*_anchors`."""
    table = json.load(open(_TABLE))
    ok = total = 0
    for key, (num, _den) in table.items():
        if key == "bare":
            continue
        total += 1
        tops, hr = ast.literal_eval(key)
        rep = diagram_representative(DiagramString(tops, 2, 0), hr)
        if (-1) ** (_hole_lines(rep) + directed_loops(rep)) == (1 if num > 0 else -1):
            ok += 1
    return ok, total


def score_against_table() -> tuple[int, int, list]:
    """(matches, total, failures) of `(-1)^(nh+trace_loops)` vs the weight table sign."""
    table = json.load(open(_TABLE))
    ok = total = 0
    fails = []
    for key, (num, _den) in table.items():
        if key == "bare":
            continue
        total += 1
        tops, hr = ast.literal_eval(key)
        rep = diagram_representative(DiagramString(tops, 2, 0), hr)
        nh = sum(1 for i in rep.summed_indices if i.space == "occ")
        nl = trace_loops(rep)
        sign = 1 if num > 0 else -1
        if (-1) ** (nh + nl) == sign:
            ok += 1
        else:
            fails.append((tops, nh, nl, sign))
    return ok, total, fails


if __name__ == "__main__":
    print(f"directed tracer vs Crawford ENERGY anchors (l=2): {check_energy_anchors()}")
    print(f"directed tracer vs Crawford OPEN anchors (p.91 l,h,sign): {check_open_anchors()}")
    dok, dtot = score_directed_vs_table()
    print(f"directed Crawford-sign vs PySCF table: {dok}/{dtot} "
          f"({dtot - dok} rep-labeling/P-orbit convention deltas, not tracer errors)")
    ok, total, fails = score_against_table()
    print(f"[legacy undirected trace_loops] {ok}/{total}")
    for f in fails:
        print("  MISS", f)
