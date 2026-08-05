#ifndef HF_POSTHF_CC_GENERATED_KERNEL_REGISTRY_H
#define HF_POSTHF_CC_GENERATED_KERNEL_REGISTRY_H

#include <expected>
#include <string>

#include "post_hf/cc/generated_arbitrary_runtime.h"

namespace HartreeFock::Correlation::CC
{
    // Rank-parameterized generated-kernel registry. Returns the generated tensor
    // kernels for the requested excitation rank (4=CCSDTQ, 5=CC5, 6=CC6), or an
    // error naming the PLANCK_CC_MAXORDER reconfigure needed when this build did
    // not generate that rank. The runtime solver
    // (prepare_generated_arbitrary_order_state) is already rank-generic, so no
    // per-rank driver path is needed — the ceiling is PLANCK_CC_MAXORDER alone.
    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rcc_kernels(int rank);

    // Rank-4 alias retained for the existing run_rccsdtq call site.
    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rccsdtq_kernels();
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_GENERATED_KERNEL_REGISTRY_H
