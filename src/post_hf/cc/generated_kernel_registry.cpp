#include "post_hf/cc/generated_kernel_registry.h"

#if PLANCK_CC_MAXORDER >= 4
#include "generated/cc/ccsdtq_planck_generated.cpp"
#endif

namespace HartreeFock::Correlation::CC
{
    std::expected<GeneratedArbitraryOrderKernels, std::string>
    make_generated_rccsdtq_kernels()
    {
#if PLANCK_CC_MAXORDER >= 4
        return make_generated_ccsdtq_kernels();
#else
        return std::unexpected(
            "Generated RCCSDTQ kernels are not available in this build. Reconfigure with -DPLANCK_CC_MAXORDER=4 (or higher) and rebuild.");
#endif
    }
} // namespace HartreeFock::Correlation::CC
