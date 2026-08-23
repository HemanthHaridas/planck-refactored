#ifndef HF_POSTHF_CC_CCSDTQ_H
#define HF_POSTHF_CC_CCSDTQ_H

#include <expected>
#include <string>

#include "integrals/shellpair.h"

namespace HartreeFock::Correlation::CC
{
    // U5.3b: the generated UNRESTRICTED CC entry (correlation ucc2/ucc3/ucc4).
    // Rank comes from OptionsSCF::_cc_generated_rank, as on the RCC path.
    // Declared beside run_rccsdtq because the aggregate post_hf/cc.h -- which the
    // driver includes -- already exposes this header.
    std::expected<void, std::string> run_uccgen(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_rccsdtq(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_CCSDTQ_H
