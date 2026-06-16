#ifndef HF_POSTHF_CC_CCSD_H
#define HF_POSTHF_CC_CCSD_H

#include <expected>
#include <string>

#include "integrals/shellpair.h"
#include "post_hf/cc/amplitudes.h"
#include "post_hf/cc/diis.h"
#include "post_hf/cc/mo_blocks.h"

namespace HartreeFock::Correlation::CC
{
    // `RCCSDState` holds the reusable canonical-RHF data for the tensor-based
    // solver. Keeping preparation separate from the iteration loop makes it easy
    // to teach which parts are one-time setup and which parts are the actual CC
    // equations.
    struct RCCSDState
    {
        RHFReference reference;
        MOBlockCache mo_blocks;
        DenominatorCache denominators;
        RCCSDAmplitudes amplitudes;
    };

    // The unrestricted teaching implementation currently uses the same
    // determinant-space solver engine as the prototype CCSDT path. Preparation
    // therefore only needs to snapshot the validated UHF reference.
    struct UCCSDState
    {
        UHFReference reference;
    };

    std::expected<RCCSDState, std::string> prepare_rccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_rccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<UCCSDState, std::string> prepare_uccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_uccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_CCSD_H
