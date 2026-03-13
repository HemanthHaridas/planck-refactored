#ifndef HF_RMP2_H
#define HF_RMP2_H

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{
    // Compute RMP2 correlation energy and store in calculator._correlation_energy.
    // Requires a converged RHF wavefunction in calculator._info._scf.alpha.
    // If calculator._eri is populated (from conventional SCF), reuses it directly.
    // Otherwise builds the AO ERI tensor from shell_pairs (direct path).
    std::expected<void, std::string> run_rmp2(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

    // Compute UMP2 correlation energy and store in calculator._correlation_energy.
    // Requires a converged UHF wavefunction in calculator._info._scf.alpha/beta.
    // If calculator._eri is populated (from conventional SCF), reuses it directly.
    // Otherwise builds the AO ERI tensor from shell_pairs (direct path).
    std::expected<void, std::string> run_ump2(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);
}

#endif // HF_RMP2_H
