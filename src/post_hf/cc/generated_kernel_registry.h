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

    // U5.3: the UNRESTRICTED (spin-block resolved) generated kernels for a rank.
    //
    // A sibling of make_generated_rcc_kernels rather than a flag on it: the two
    // return bundles of different SHAPE. An RCC bundle carries one reference
    // residual per rank in `residuals_by_rank`; a UCC bundle carries none at all
    // and drives every excitation through `sector_residuals` (one per spin block),
    // which is the all-sectors mode U4.0 taught the runtime to accept.
    //
    // Available only when the build emitted the UCC translation units
    // (-DPLANCK_CC_UCC=ON). Without them this returns an error naming the
    // reconfigure, rather than silently falling back to the RCC bundle -- which
    // would run a restricted kernel against an unrestricted reference and produce
    // a plausible wrong number.
    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_ucc_kernels(int rank);

    // Whether this build carries UCC kernels at all. Lets a caller distinguish
    // "not built" from "built but this rank is out of range".
    [[nodiscard]] bool generated_ucc_kernels_available() noexcept;

    // Rank-4 alias retained for the existing run_rccsdtq call site.
    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rccsdtq_kernels();
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_GENERATED_KERNEL_REGISTRY_H
