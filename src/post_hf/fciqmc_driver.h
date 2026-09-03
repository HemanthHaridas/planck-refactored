#ifndef HF_POSTHF_FCIQMC_DRIVER_H
#define HF_POSTHF_FCIQMC_DRIVER_H

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{

    // ── FCIQMC ─────────────────────────────────────────────────────────────────
    //
    // Stochastic sampling of the FCI wavefunction with a population of signed
    // walkers, instead of storing a CI vector. Same determinant space run_fci
    // uses -- the whole MO space, no inactive core -- and the SAME integrals, via
    // the shared build_all_mo_ci_setup.
    //
    // The two paths therefore differ only in how they solve: run_fci diagonalizes,
    // run_fciqmc samples. That is what makes a disagreement between them
    // attributable to sampling rather than to plumbing.
    //
    // Reports both estimators with a blocked error bar. The energy is a MEAN WITH
    // AN UNCERTAINTY, not an exact value -- see docs/FCIQMC_POPULATION_CONTROL.md
    // for why the projected energy is biased at finite population and why the
    // error bar must be blocked rather than naive.
    //
    // Reads from calc._active_space (ci_max_dim) and calc._fciqmc.
    //
    // Writes to:
    //   calc._correlation_energy         — E_FCIQMC - E_reference
    //   calc._correlated_total_energy    — E_FCIQMC
    //   calc._have_correlated_total_energy
    std::expected<void, std::string> run_fciqmc(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

} // namespace HartreeFock::Correlation

#endif
