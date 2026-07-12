// mpi_env.h — the whole MPI surface the compute layer sees.
//
// Everything here is a no-op unless the translation unit is compiled with
// USE_MPI (only the planck-mpi target is). The serial binaries — hartree-fock,
// planck-dft — get rank 0, size 1, and reductions that touch nothing, so the
// ERI/Fock code carries the same source in all three binaries with zero runtime
// cost off the MPI path. This is the only place <mpi.h> is included below the
// driver; keep it that way so the kernels stay MPI-agnostic.
#pragma once

#include <cstddef>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace HartreeFock::Mpi
{
    // Rank of this process in MPI_COMM_WORLD; 0 in a serial build.
    inline int rank() noexcept
    {
#ifdef USE_MPI
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
#else
        return 0;
#endif
    }

    // Number of ranks; 1 in a serial build.
    inline int size() noexcept
    {
#ifdef USE_MPI
        int s = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        return s;
#else
        return 1;
#endif
    }

    // True when work must actually be split (more than one rank). Lets the
    // kernels skip the stride/reduce bookkeeping entirely in the common
    // single-rank case, keeping the serial path byte-identical.
    inline bool distributed() noexcept { return size() > 1; }

    // In-place sum-reduce of a double buffer across all ranks, leaving the full
    // result replicated on every rank (MPI_Allreduce). No-op when serial.
    //
    // ponytail: Allreduce the whole dense buffer, not a bespoke sparse merge.
    // The buffer is already nb^4 (or nb^2) and replicated by design at this
    // tier; a smarter reduction is a Tier-1-optimization, not a correctness
    // need. Upgrade to reduce-scatter only if the replicated buffer becomes the
    // memory wall.
    inline void allreduce_inplace([[maybe_unused]] double *buf,
                                  [[maybe_unused]] std::size_t n) noexcept
    {
#ifdef USE_MPI
        // MPI_Allreduce takes an int count; nb^4 doubles overflows 2^31 at
        // nb ~= 215 (the exact >6-atom regime this executable exists for), so
        // chunk the buffer rather than trust the whole length to fit an int.
        constexpr std::size_t kChunk = 1u << 28; // 256M doubles = 2 GiB per call
        for (std::size_t off = 0; off < n; off += kChunk)
        {
            const std::size_t len = (n - off < kChunk) ? (n - off) : kChunk;
            MPI_Allreduce(MPI_IN_PLACE, buf + off, static_cast<int>(len),
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif
    }
}
