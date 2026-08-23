#include "post_hf/cc/generated_kernel_registry.h"

#include <format>

// Each generated translation unit defines make_generated_<method>_kernels().
// Included per rank, guarded so a build only compiles the ranks it generated
// (PLANCK_CC_MAXORDER). The rank -> method-name map matches CMakeLists.txt's
// _planck_cc_method_by_rank: 4=ccsdtq, 5=cc5, 6=cc6.
//
// The rank-3 arbitrary-order companion (ccsdt_arbitrary_planck_generated.cpp,
// emitted only when -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON) is a SPATIAL
// ArbitraryOrderRCCAmplitudes kernel — a lower-rank seed source for the cc4
// warm-start (Route A). The plain ccsdt TU (RCCSDTAmplitudes, tensor_backend)
// is a different consumer and is NOT included here.
// The rank-3 arbitrary-order companion (emitted only with
// -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON) is a SPATIAL ArbitraryOrderRCCAmplitudes
// kernel — a lower-rank seed source for the cc4 warm-start (Route A). It is
// emitted WITHOUT intermediate builders on purpose: the shape-named builders
// (build_W_oo_3, ...) carry no method suffix, so co-including it with the
// ccsdtq TU would collide on those symbols. The residual is self-contained
// without them. The plain ccsdt TU (RCCSDTAmplitudes, tensor_backend) is a
// different consumer and is NOT included here.
#ifndef PLANCK_CC_ARBITRARY_LOWER_RANKS
#define PLANCK_CC_ARBITRARY_LOWER_RANKS 0
#endif
#if PLANCK_CC_ARBITRARY_LOWER_RANKS
#include "generated/cc/ccsdt_arbitrary_planck_generated.cpp"
#endif
#if PLANCK_CC_MAXORDER >= 4
#include "generated/cc/ccsdtq_planck_generated.cpp"
#endif
#if PLANCK_CC_MAXORDER >= 5
#include "generated/cc/cc5_planck_generated.cpp"
#endif
#if PLANCK_CC_MAXORDER >= 6
#include "generated/cc/cc6_planck_generated.cpp"
#endif

// U5.3: the UCC translation units, emitted only under -DPLANCK_CC_UCC=ON. They
// define make_generated_ucc_<method>_kernels(), distinct from their RCC siblings
// (U5.0) so both can link into one binary.
#ifndef PLANCK_CC_UCC
#define PLANCK_CC_UCC 0
#endif
#if PLANCK_CC_UCC
#include "generated/cc/ccsd_ucc_planck_generated.cpp"
#if PLANCK_CC_MAXORDER >= 3
#include "generated/cc/ccsdt_ucc_planck_generated.cpp"
#endif
#if PLANCK_CC_MAXORDER >= 4
#include "generated/cc/ccsdtq_ucc_planck_generated.cpp"
#endif
#endif

namespace HartreeFock::Correlation::CC
{
    bool generated_ucc_kernels_available() noexcept
    {
        return PLANCK_CC_UCC != 0;
    }

    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_ucc_kernels(int rank)
    {
        if (rank < 2)
            return std::unexpected(std::format(
                "make_generated_ucc_kernels: rank {} is below the smallest UCC "
                "method (rank 2, uccsd).", rank));

#if !PLANCK_CC_UCC
        // Deliberately an ERROR, not a fall-back to the RCC bundle: running a
        // restricted kernel against an unrestricted reference compiles, runs, and
        // returns a plausible wrong number.
        return std::unexpected(std::format(
            "make_generated_ucc_kernels: this build carries no UCC kernels "
            "(rank {} requested). Reconfigure with -DPLANCK_CC_UCC=ON.", rank));
#else
        switch (rank)
        {
        case 2:
            return make_generated_ucc_ccsd_kernels();
#if PLANCK_CC_MAXORDER >= 3
        case 3:
            return make_generated_ucc_ccsdt_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 4
        case 4:
            return make_generated_ucc_ccsdtq_kernels();
#endif
        default:
            return std::unexpected(std::format(
                "make_generated_ucc_kernels: rank {} exceeds this build's "
                "PLANCK_CC_MAXORDER ({}). Reconfigure with a higher value.",
                rank, PLANCK_CC_MAXORDER));
        }
#endif
    }

    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rcc_kernels(int rank)
    {
        // The generated tensor path normally begins at CCSDTQ (rank 4); ranks 2/3
        // use the hand-written backends. When the rank-3 arbitrary companion is
        // built (-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON) rank 3 also routes here as
        // a spatial seed source for the cc4 warm-start.
        constexpr int generated_floor = PLANCK_CC_ARBITRARY_LOWER_RANKS ? 3 : 4;
        if (rank < generated_floor)
            return std::unexpected(std::format(
                "make_generated_rcc_kernels: rank {} has no generated tensor "
                "kernel path (below the generated floor {}).",
                rank, generated_floor));

        switch (rank)
        {
#if PLANCK_CC_ARBITRARY_LOWER_RANKS
        case 3:
            return make_generated_ccsdt_kernels();
#endif
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
