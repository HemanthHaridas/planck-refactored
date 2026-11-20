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
    // Full o^3 v^3 tensors: the oracle for the batched kernel, small systems only.
    std::expected<double, std::string> compute_rccsd_t_energy(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const ArbitraryOrderRCCAmplitudes &t1_t2);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_CCSD_T_H
