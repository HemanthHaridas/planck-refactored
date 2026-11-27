#include "post_hf/cc/generated_kernel_registry.h"

#include <format>

// Each generated TU defines make_generated_<method>_kernels(). The TUs are
// compiled in shards into the planck_cc_generated library (CMakeLists.txt,
// ccgen/emit/shard.py); this file includes only their headers (declarations),
// so no compile holds every kernel at once. Included per rank, guarded so a
// build only references the ranks it generated (PLANCK_CC_MAXORDER). The rank -> method-name map matches CMakeLists.txt's
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
// Rank 2: spatial RCCSD behind ccsd(t) / cc2, generated in every build
// (docs/CCSD_T_SIZE_SCOPE.md route A).
#include "generated/cc/ccsd_arbitrary_planck_generated.h"
// Λ residual kernels (docs/CC_LAMBDA_SCOPE.md): generated for the same
// ranks as the T kernels beside them.
#include "generated/cc/ccsd_lambda_planck_generated.h"
#if PLANCK_CC_ARBITRARY_LOWER_RANKS
#include "generated/cc/ccsdt_arbitrary_planck_generated.h"
#include "generated/cc/ccsdt_lambda_planck_generated.h"
#endif
#if PLANCK_CC_MAXORDER >= 4
#include "generated/cc/ccsdtq_planck_generated.h"
#include "generated/cc/ccsdtq_lambda_planck_generated.h"
#endif
#if PLANCK_CC_MAXORDER >= 5
#include "generated/cc/cc5_planck_generated.h"
#include "generated/cc/cc5_lambda_planck_generated.h"
#endif
#if PLANCK_CC_MAXORDER >= 6
#include "generated/cc/cc6_planck_generated.h"
#include "generated/cc/cc6_lambda_planck_generated.h"
#endif

// U5.3: the UCC translation units, emitted only under -DPLANCK_CC_UCC=ON. They
// define make_generated_ucc_<method>_kernels(), distinct from their RCC siblings
// (U5.0) so both can link into one binary.
#ifndef PLANCK_CC_UCC
#define PLANCK_CC_UCC 0
#endif
#if PLANCK_CC_UCC
#include "generated/cc/ccsd_ucc_planck_generated.h"
#include "generated/cc/ccsd_ucc_lambda_planck_generated.h"
#if PLANCK_CC_MAXORDER >= 3
#include "generated/cc/ccsdt_ucc_planck_generated.h"
#include "generated/cc/ccsdt_ucc_lambda_planck_generated.h"
#endif
#if PLANCK_CC_MAXORDER >= 4
#include "generated/cc/ccsdtq_ucc_planck_generated.h"
#include "generated/cc/ccsdtq_ucc_lambda_planck_generated.h"
#endif
#if PLANCK_CC_MAXORDER >= 5
#include "generated/cc/cc5_ucc_planck_generated.h"
#include "generated/cc/cc5_ucc_lambda_planck_generated.h"
#endif
#if PLANCK_CC_MAXORDER >= 6
#include "generated/cc/cc6_ucc_planck_generated.h"
#include "generated/cc/cc6_ucc_lambda_planck_generated.h"
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
        // U5.3c: the rank-4 ceiling here was a hand-written limit, not a real
        // one -- the emitter is rank-generic and PLANCK_CC_METHODS is derived
        // from PLANCK_CC_MAXORDER, so the UCC translation units are emitted at
        // ranks 5 and 6 exactly when the RCC ones are.
#if PLANCK_CC_MAXORDER >= 5
        case 5:
            return make_generated_ucc_cc5_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 6
        case 6:
            return make_generated_ucc_cc6_kernels();
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
        // Rank 2 (the CCSD behind ccsd(t)) is generated in every build. Rank 3
        // needs the arbitrary companion (-DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON);
        // without it CCSDT uses the hand-written backends. Ranks >= 4 follow
        // PLANCK_CC_MAXORDER.
        if (rank < 2)
            return std::unexpected(std::format(
                "make_generated_rcc_kernels: rank {} has no generated tensor "
                "kernel path (the generated floor is 2).",
                rank));
        if (rank == 3 && !PLANCK_CC_ARBITRARY_LOWER_RANKS)
            return std::unexpected(
                "Generated rank-3 CC kernels are not available in this build. "
                "Reconfigure with -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON and rebuild.");

        switch (rank)
        {
        case 2:
            return make_generated_ccsd_kernels();
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

    std::expected<GeneratedLambdaKernels, std::string>
    make_generated_lambda_kernels(int rank, bool ucc)
    {
        // Built for exactly the ranks the T kernels are (make_generated_rcc_kernels,
        // make_generated_ucc_kernels); those name the reconfigure when a rank is missing.
        if (ucc)
        {
#if PLANCK_CC_UCC
            switch (rank)
            {
            case 2: return make_generated_ucc_lambda_ccsd_kernels();
#if PLANCK_CC_MAXORDER >= 3
            case 3: return make_generated_ucc_lambda_ccsdt_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 4
            case 4: return make_generated_ucc_lambda_ccsdtq_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 5
            case 5: return make_generated_ucc_lambda_cc5_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 6
            case 6: return make_generated_ucc_lambda_cc6_kernels();
#endif
            default: break;
            }
#endif
        }
        else
        {
            switch (rank)
            {
            case 2: return make_generated_lambda_ccsd_kernels();
#if PLANCK_CC_ARBITRARY_LOWER_RANKS
            case 3: return make_generated_lambda_ccsdt_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 4
            case 4: return make_generated_lambda_ccsdtq_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 5
            case 5: return make_generated_lambda_cc5_kernels();
#endif
#if PLANCK_CC_MAXORDER >= 6
            case 6: return make_generated_lambda_cc6_kernels();
#endif
            default: break;
            }
        }
        if (auto t = ucc ? make_generated_ucc_kernels(rank) : make_generated_rcc_kernels(rank); !t)
            return std::unexpected(t.error());
        return std::unexpected(std::format(
            "Generated {}Λ kernels are not available for rank {} in this build.",
            ucc ? "UCC " : "", rank));
    }

    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rccsdtq_kernels()
    {
        // Rank-4 alias, kept for the existing run_rccgen call site.
        return make_generated_rcc_kernels(4);
    }
} // namespace HartreeFock::Correlation::CC
