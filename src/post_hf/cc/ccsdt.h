#ifndef HF_POSTHF_CC_CCSDT_H
#define HF_POSTHF_CC_CCSDT_H

#include <expected>
#include <string>

#include "integrals/shellpair.h"
#include "post_hf/cc/amplitudes.h"
#include "post_hf/cc/mo_blocks.h"

namespace HartreeFock::Correlation::CC
{
    // The long-term CCSDT design in this codebase is still tensor-oriented, so
    // the prepared state keeps the same reference/integral/denominator pieces as
    // CCSD. The current solver prototype builds a small determinant-space model
    // on top of this state during `run_rccsdt`.
    struct RCCSDTState
    {
        RHFReference reference;
        MOBlockCache mo_blocks;
        DenominatorCache denominators;
        RCCSDTAmplitudes amplitudes;
    };

    std::expected<RCCSDTState, std::string> prepare_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_CCSDT_H
