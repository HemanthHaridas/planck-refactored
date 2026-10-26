// auto_block_eri.h — the Auto engine's block-kernel chain (block_eri.h),
// defined once for the engine and every test or benchmark that measures it.
//
//   closed forms     total L <= 3
//   Rys2D block      K = 1 at L >= 4, K <= 3 at L >= 7 (K primitive quartets),
//                    where it measured faster than the HGP block
//                    (docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md, K2.1b b3)
//   HGP block        the rest of 4 <= L <= 12
//
// Above L = 12, and for screened kernels, blocks stay on the per-component
// path.
#pragma once

#include "block_eri.h"
#include "hgp_block_eri.h"
#include "low_l_eri.h"
#include "rys_block_eri.h"

namespace HartreeFock::Integrals
{
    using AutoBlockKernel =
        ChainBlockKernel<ClosedFormBlockKernel, ChainBlockKernel<RysBlockKernel, HgpBlockKernel>>;

    inline AutoBlockKernel auto_block_kernel()
    {
        return AutoBlockKernel{ClosedFormBlockKernel{3},
                               {RysBlockKernel{4, 12, /*crossover=*/true}, HgpBlockKernel{4, 12}}};
    }
}
