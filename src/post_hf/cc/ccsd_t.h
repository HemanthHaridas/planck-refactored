#ifndef HF_POSTHF_CC_CCSD_T_H
#define HF_POSTHF_CC_CCSD_T_H

#include <expected>
#include <string>
#include <vector>

#include "post_hf/cc/amplitudes.h"

namespace HartreeFock
{
    struct Calculator;
    struct ShellPair;
}

namespace HartreeFock::Correlation::CC
{
    // Closed-shell (T) correction from converged spatial RCCSD amplitudes
    // (rank 2: t1 (i,a), t2 (i,j,a,b)), using the ccgen-generated W and V
    // builders on the aab|aab block (docs/CCSD_T_SCOPE.md):
    //
    //   E(T) = S_aaa / 18 + S_aab / 2,   S = sum (W + V) W / D,
    //   X_aaa(abc, ijk) = 1/2 sum_P sign(P) X_aab(P(abc), ijk).
    //
    // Batched over virtual triples a >= b >= c (o^3 scratch per thread, no
    // o^3 v^3 tensor), threaded, independent of the thread count. At verbose
    // output on small systems it also recomputes E(T) with the full-tensor
    // kernels and logs the difference.
    std::expected<double, std::string> compute_rccsd_t_energy(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const ArbitraryOrderRCCAmplitudes &t1_t2);

    struct ArbitraryOrderTensorCCState;

    // Open-shell (T) from a converged generated UCCSD state (ucc2): t1_aa/bb and
    // t2_aaaa/abab/bbbb sectors, spin-blocked physicist ERIs and the per-spin
    // Fock blocks, whose diagonals are the denominators:
    //
    //   E(T) = (S_aaa + S_bbb)/36 + (S_aab + S_abb)/4.
    //
    // Batched over each block's distinct virtual triples, threaded, independent
    // of the thread count; at verbose output on small systems it also logs the
    // difference from the full-tensor kernels.
    std::expected<double, std::string> compute_uccsd_t_energy(
        const HartreeFock::Calculator &calculator,
        const ArbitraryOrderTensorCCState &state);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_CCSD_T_H
