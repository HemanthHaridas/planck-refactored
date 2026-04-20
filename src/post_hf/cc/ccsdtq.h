#ifndef HF_POSTHF_CC_CCSDTQ_H
#define HF_POSTHF_CC_CCSDTQ_H

#include <expected>
#include <string>

#include "integrals/shellpair.h"

namespace HartreeFock::Correlation::CC
{
    std::expected<void, std::string> run_rccsdtq(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_CCSDTQ_H
