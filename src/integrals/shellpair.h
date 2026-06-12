#ifndef HF_SHELLPAIR_H
#define HF_SHELLPAIR_H

#include <cstddef>
#include <vector>

#include "base/types.h"

// ─── Per-Cartesian-AO shell pairs (current production granularity) ───────────
//
// build_shellpairs returns one HartreeFock::ShellPair per ordered pair of
// *Cartesian AO components* (upper triangle of _basis_functions), NOT per
// shell. Every integral engine consumes that granularity today: each ShellPair
// is a single (lx,ly,lz)-component on each side, sp.A._index is one AO row, and
// the engine emits a 1x1 contribution. This is the contract H-10 flags for
// reorganization — see build_shell_groups below for the shell-granular layer
// that will eventually replace it.
std::vector<HartreeFock::ShellPair> build_shellpairs(const HartreeFock::Basis &basis);

// ─── Shell-granular layer (H-10 step 1, no-op adapter) ───────────────────────
//
// A ShellGroup describes one *true shell*: its slice of Basis::_basis_functions.
// Because _basis_functions is built shell-by-shell, contiguously, in
// _cartesian_shell_order, a shell's Cartesian AO components are exactly the
// contiguous range [first_ao, first_ao + n_components) and they all share the
// same underlying Shell* (== _basis_functions[first_ao]._shell).
//
// This is the granularity the integral engines should ultimately iterate, so
// the per-primitive-pair OS/Rys/HGP seed work (product center, prefactor, Boys
// args, VRR roots) is computed once per shell quartet and reused across all
// Cartesian-component combinations, instead of being recomputed per component.
//
// Nothing consumes ShellGroup for real integral work yet. expand_shell_groups_
// to_ao_pairs below re-expands the groups back into the exact per-AO ShellPair
// list build_shellpairs already produces, so build_shellpairs can route through
// this layer with bitwise-identical output and every downstream engine stays
// untouched. The migration then converts engines off the expanded list one at a
// time, behind the existing bitwise gates.
struct ShellGroup
{
    const HartreeFock::Shell *shell = nullptr; // owning shell
    std::size_t shell_index = 0;               // position in Basis::_shells
    std::size_t first_ao = 0;                  // first AO component's _index in _basis_functions
    std::size_t n_components = 0;              // (L+1)(L+2)/2 Cartesian components
};

// Partition Basis::_basis_functions into its constituent shells. The returned
// vector is in shell order and its size is basis.nshells(). For each group,
// _basis_functions[first_ao .. first_ao + n_components) are that shell's
// Cartesian AO components.
std::vector<ShellGroup> build_shell_groups(const HartreeFock::Basis &basis);

// Re-expand shell groups into the per-Cartesian-AO ShellPair list, identical
// (order and content) to what the legacy per-AO build_shellpairs produces:
// upper triangle over _basis_functions, outer index >= inner. The ShellPair
// constructor folds in each component's _component_norm, so the expanded pairs
// carry the per-component normalization exactly as before. This is the no-op
// adapter that lets build_shellpairs route through the shell-granular layer
// without changing any engine.
std::vector<HartreeFock::ShellPair> expand_shell_groups_to_ao_pairs(
    const HartreeFock::Basis &basis, const std::vector<ShellGroup> &groups);

// Compute the shell pair index for (i,j) given total number of shells
inline std::size_t pair_index(std::size_t i, std::size_t j)
{
    if (i < j)
        std::swap(i, j); // enforce i >= j
    return i * (i + 1) / 2 + j;
}

#endif // !HF_SHELLPAIR_H
