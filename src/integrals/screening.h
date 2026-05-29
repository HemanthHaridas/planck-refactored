#ifndef HF_SCREENING_H
#define HF_SCREENING_H

#include <Eigen/Core>
#include <vector>

#include "base/types.h"
#include "shellpair.h"

namespace HartreeFock
{
    namespace Screening
    {
        // Schwarz bound Q(i,j) = sqrt((ij|ij)) on every shell-pair AO index,
        // evaluated through the HGP contracted-ERI kernel. The diagonal bound
        // is engine-independent at the value level, so this is correct for
        // any engine that wants a Schwarz table.
        //
        // Profiling step (vault: HGP Schwarz vs Rys Schwarz) showed that
        // _rys_schwarz_table was 4-5x slower than the HGP equivalent on the
        // L >= 2 buckets that dominate the table, and that this overhead was
        // the entire source of the auto-dispatch Fock-build gap vs HGP. The
        // shared helper exists so callers in the auto path (and any other
        // engine that wants the cheap Schwarz) can share one implementation
        // instead of re-deriving slower variants.
        //
        // Output layout: row-major nb*nb double vector. Q[i*nb + j] == Q[j*nb + i].
        std::vector<double> schwarz_table_hgp(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            std::size_t nbasis,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops);
    } // namespace Screening
} // namespace HartreeFock

#endif // HF_SCREENING_H
