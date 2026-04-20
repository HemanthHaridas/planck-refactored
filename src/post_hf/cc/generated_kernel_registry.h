#ifndef HF_POSTHF_CC_GENERATED_KERNEL_REGISTRY_H
#define HF_POSTHF_CC_GENERATED_KERNEL_REGISTRY_H

#include <expected>
#include <string>

#include "post_hf/cc/generated_arbitrary_runtime.h"

namespace HartreeFock::Correlation::CC
{
    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rccsdtq_kernels();
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_GENERATED_KERNEL_REGISTRY_H
