// auto_deriv_block_eri.h — the Auto engine's derivative block-kernel chain,
// defined once for the gradient and every test or benchmark that measures it
// (docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md, G3).
//
//   Rys2D derivative  its crossover blocks (rys_deriv_block_eri.h)
//   HGP derivative    everything else up to its limits
//
// Blocks both decline stay on the per-component derivative.
#pragma once

#include "block_eri.h"
#include "hgp_deriv_block_eri.h"
#include "rys_deriv_block_eri.h"

namespace HartreeFock::Integrals
{
    using AutoDerivKernel = ChainBlockKernel<RysDerivBlockKernel, HgpDerivBlockKernel>;

    inline AutoDerivKernel auto_deriv_kernel()
    {
        return AutoDerivKernel{RysDerivBlockKernel{/*crossover=*/true}, HgpDerivBlockKernel{}};
    }
}
