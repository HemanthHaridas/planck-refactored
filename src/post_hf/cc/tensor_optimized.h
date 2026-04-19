#ifndef HF_POSTHF_CC_TENSOR_OPTIMIZED_H
#define HF_POSTHF_CC_TENSOR_OPTIMIZED_H

#include <expected>
#include <string>
#include <vector>

#include "integrals/shellpair.h"

namespace HartreeFock
{
    class Calculator;
}

namespace HartreeFock::Correlation::CC
{
    std::expected<void, std::string> run_tensor_optimized_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_TENSOR_OPTIMIZED_H
