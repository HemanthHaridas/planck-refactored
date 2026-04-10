#ifndef HF_POSTHF_CC_AMPLITUDES_H
#define HF_POSTHF_CC_AMPLITUDES_H

#include <expected>
#include <string>

#include "post_hf/cc/common.h"

namespace HartreeFock::Correlation::CC
{
    // Canonical energy denominators are reused by multiple update schemes.
    // The tensors follow the same occupied/virtual index order as the
    // amplitudes so the update loops read almost like the algebra.
    struct DenominatorCache
    {
        Tensor2D d1; // eps_i - eps_a
        Tensor4D d2; // eps_i + eps_j - eps_a - eps_b
        Tensor6D d3; // eps_i + eps_j + eps_k - eps_a - eps_b - eps_c
    };

    struct RCCSDAmplitudes
    {
        Tensor2D t1; // t_i^a stored as (i,a)
        Tensor4D t2; // t_ij^ab stored as (i,j,a,b)
    };

    struct RCCSDTAmplitudes
    {
        Tensor2D t1; // t_i^a stored as (i,a)
        Tensor4D t2; // t_ij^ab stored as (i,j,a,b)
        Tensor6D t3; // t_ijk^abc stored as (i,j,k,a,b,c)
    };

    // `include_triples=false` is useful for the current teaching code paths that
    // need only singles/doubles denominators and want to avoid an unnecessary
    // O(o^3 v^3) allocation.
    std::expected<DenominatorCache, std::string> build_denominator_cache(
        const RHFReference &reference,
        bool include_triples = true);

    RCCSDAmplitudes make_zero_rccsd_amplitudes(const RHFReference &reference);

    // The dense T3 container is kept for the future tensor-based CCSDT path. The
    // current determinant-space prototype does not allocate it eagerly inside the
    // top-level solver because that would dominate memory before any iterations.
    RCCSDTAmplitudes make_zero_rccsdt_amplitudes(const RHFReference &reference);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_AMPLITUDES_H
