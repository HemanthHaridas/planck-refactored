#include "post_hf/cc/generated_kernel_registry.h"

#include <format>

// Each generated translation unit defines make_generated_<method>_kernels().
// Included per rank, guarded so a build only compiles the ranks it generated
// (PLANCK_CC_MAXORDER). The rank -> method-name map matches CMakeLists.txt's
// _planck_cc_method_by_rank: 4=ccsdtq, 5=cc5, 6=cc6.
#if PLANCK_CC_MAXORDER >= 4
#include "generated/cc/ccsdtq_planck_generated.cpp"
#endif
#if PLANCK_CC_MAXORDER >= 5
#include "generated/cc/cc5_planck_generated.cpp"
#endif
#if PLANCK_CC_MAXORDER >= 6
#include "generated/cc/cc6_planck_generated.cpp"
#endif

namespace HartreeFock::Correlation::CC
{
    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rcc_kernels(int rank)
    {
        // The generated tensor path begins at CCSDTQ (rank 4); ranks 2/3 have
        // hand-written / tensor backends and never route here.
        if (rank < 4)
            return std::unexpected(std::format(
                "make_generated_rcc_kernels: rank {} has no generated tensor "
                "kernel path (ranks 2/3 use the hand-written backends).",
                rank));

        switch (rank)
        {
#if PLANCK_CC_MAXORDER >= 4
        case 4:
            return make_generated_ccsdtq_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 5
        case 5:
            return make_generated_cc5_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 6
        case 6:
            return make_generated_cc6_kernels();
#endif
        default:
            break;
        }

        // Requested rank is within the supported range but this build did not
        // generate it (PLANCK_CC_MAXORDER too low), or exceeds the max rank.
        return std::unexpected(std::format(
            "Generated rank-{} CC kernels are not available in this build "
            "(PLANCK_CC_MAXORDER={}). Reconfigure with "
            "-DPLANCK_CC_MAXORDER={} (or higher, up to 6) and rebuild.",
            rank, PLANCK_CC_MAXORDER, rank));
    }

    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rccsdtq_kernels()
    {
        // Rank-4 alias, kept for the existing run_rccsdtq call site.
        return make_generated_rcc_kernels(4);
    }
} // namespace HartreeFock::Correlation::CC
