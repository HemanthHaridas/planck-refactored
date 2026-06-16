"""Graph-based connectivity analysis for contracted terms."""

from __future__ import annotations

from typing import Sequence


def _union_find_components(
    n_nodes: int, edges: Sequence[tuple[int, int]]
) -> list[set[int]]:
    """Return connected components via union-find."""
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if a < n_nodes and b < n_nodes:
            union(a, b)

    comps: dict[int, set[int]] = {}
    for i in range(n_nodes):
        r = find(i)
        comps.setdefault(r, set()).add(i)
    return list(comps.values())


def is_connected(
    n_blocks: int,
    block_edges: Sequence[tuple[int, int]],
    ignore_block: int | None = None,
) -> bool:
    """Test whether the contraction graph is connected."""
    if n_blocks <= 1:
        return True

    if ignore_block is not None:
        remap: dict[int, int] = {}
        idx = 0
        for b in range(n_blocks):
            if b == ignore_block:
                continue
            remap[b] = idx
            idx += 1
        n = idx
        edges = [
            (remap[a], remap[b])
            for a, b in block_edges
            if a != ignore_block and b != ignore_block
            and a in remap and b in remap
        ]
    else:
        n = n_blocks
        edges = list(block_edges)

    if n <= 1:
        return True

    comps = _union_find_components(n, edges)
    return len(comps) == 1
