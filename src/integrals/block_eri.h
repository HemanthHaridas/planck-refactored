// block_eri.h — the shell-quartet block interface of the fused Fock loop.
//
// A block kernel evaluates every Cartesian component of one shell quartet
// from a single primitive pass; the loop reads the components it needs out of
// it (docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md, step D1). The contract:
//
//   bool   prepare(const BlockEriRequest &)   // false: class not handled
//   double value(ca, cb, cc, cd) const        // component (ca,cb|cc,cd)
//   bool   covers(total_L, kernel) const      // prepare() accepts every block
//                                             // of this total L and kernel
//
// prepare() runs at most once per block, at the first component that survives
// the loop's filters, so a fully screened block costs nothing. When it returns
// false the loop evaluates that block with the engine's per-component
// eri_elem instead (the adapter), exactly as before the interface existed.
// One kernel object lives per bra iteration, so per thread.
#pragma once

#include <cstddef>
#include <vector>

#include "base/types.h"

namespace HartreeFock::Integrals
{
    // Block-kernel scratch is grow-only: a smaller block leaves the size alone,
    // so the next larger one neither calls resize nor re-zeroes the regrown
    // tail (2.4-4.7 % of a build when sized per block). Readers index within
    // the current block's extent and never use size().
    inline void grow_to(std::vector<double> &v, std::size_t n)
    {
        if (v.size() < n)
            v.resize(n);
    }

    struct BlockEriRequest
    {
        // Pairs of the FIRST component of A,B and of C,D. Their coeff_product
        // carries that first component's _component_norm; component outputs
        // are rescaled by norm_ratio (exactly 1 for s and p).
        const HartreeFock::ShellPair *ab = nullptr;
        const HartreeFock::ShellPair *cd = nullptr;
        // Component views of A, B, C, D (views[s][c] is component c of shell s).
        const HartreeFock::ContractedView *const *views[4] = {};
        std::size_t n[4] = {};
        int L[4] = {};
        const double *norm_ratio = nullptr; // per AO: component norm / first component's
        HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb;
        double omega = 0.0;                 // screened kernels (scope step C5)
        double prim_eps = 0.0;              // primitive screening (scope step P2); 0 = none
    };

    // The default: no block kernel, every block takes the per-component path.
    // Explicit engines use it, so the cross-engine comparators stay meaningful.
    struct NoBlockKernel
    {
        bool prepare(const BlockEriRequest &) noexcept { return false; }
        bool covers(int, HartreeFock::ERIKernel) const noexcept { return false; }
        double value(std::size_t, std::size_t, std::size_t, std::size_t) const noexcept { return 0.0; }
    };

    // Try First, then Second, for each block: the closed forms for low L,
    // a general block kernel for what they decline.
    template <typename First, typename Second>
    struct ChainBlockKernel
    {
        First first;
        Second second;
        bool use_first = true;

        bool covers(int total_l, HartreeFock::ERIKernel k) const noexcept
        {
            return first.covers(total_l, k) || second.covers(total_l, k);
        }
        bool prepare(const BlockEriRequest &r)
        {
            use_first = first.prepare(r);
            return use_first || second.prepare(r);
        }
        double value(std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd) const
        {
            return use_first ? first.value(ca, cb, cc, cd) : second.value(ca, cb, cc, cd);
        }
    };
}
