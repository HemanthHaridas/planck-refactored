#ifndef HF_POSTHF_CC_MO_BLOCKS_H
#define HF_POSTHF_CC_MO_BLOCKS_H

#include <expected>
#include <string>

#include "integrals/shellpair.h"
#include "post_hf/cc/common.h"

namespace HartreeFock::Correlation::CC
{
    // Canonical RHF MO-integral blocks used by the first CCSD and CCSDT
    // implementations. All tensors use chemists' notation (pq|rs) and the
    // row-major `(p,q,r,s)` layout exposed by Tensor4D.
    struct MOBlockCache
    {
        Tensor4D full; // (pq|rs) over the full spatial MO basis
        Tensor4D oooo; // (ij|kl)
        Tensor4D ooov; // (ij|ka)
        Tensor4D oovv; // (ij|ab)
        Tensor4D ovov; // (ia|jb)
        Tensor4D ovvo; // (ia|bj)
        Tensor4D ovvv; // (ia|bc)
        Tensor4D vvvv; // (ab|cd)
    };

    // The cache is intentionally small and explicit: the teaching solvers can
    // point to a named block rather than a generic tensor slice, and the same
    // transformed integrals can be reused across multiple iterations.
    std::expected<MOBlockCache, std::string> build_mo_block_cache(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RHFReference &reference,
        const std::string &tag);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_MO_BLOCKS_H
