---
name: Shell Pair Indexing
description: Shell pair row-major ordering — never use invert_pair_index, use sp.A._index directly
type: gotcha
priority: critical
include_in_claude: true
tags: [shell-pairs, indexing, gotcha, integrals]
---

# Shell Pair Indexing Gotcha

## Build Order

`build_shellpairs` constructs shell pairs in **row-major upper triangle** order:
```cpp
for (int ia = 0; ia < n_shells; ++ia)
    for (int ib = ia; ib < n_shells; ++ib)  // ib >= ia
        pairs.push_back(ShellPair(shells[ia], shells[ib]));
```

The resulting `vector<ShellPair>` is a flat list in this ordering.

## The Trap

There is (or was) a function `invert_pair_index(k)` that attempts to recover `(ia, ib)` from the flat index `k`. **Do not use it.**

The correct way to find where in the AO matrix to place the results of shell pair `sp` is:
```cpp
int row = sp.A._index;   // position of shell A's first AO in basis
int col = sp.B._index;   // position of shell B's first AO in basis
```

`ContractedView._index` encodes exactly this. Always use `sp.A._index` / `sp.B._index` to determine AO matrix positions.

## Why invert_pair_index Fails

The closed-form index inversion formula for the upper triangle only works when shells are 1-function (S-type). For multi-function shells (P, D, F, ...), the mapping from shell pair flat index to AO matrix position is not recoverable by arithmetic alone — you need to track the cumulative basis function counts, which `_index` already does.
