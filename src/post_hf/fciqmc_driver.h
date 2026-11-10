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
    // walkers. By default it shares the all-MO Hamiltonian with run_fci.
    // Setting nactele and nactorb selects a fixed-orbital CAS with a
    // doubly occupied inactive core. Both estimators include its energy.
    //
    // Reports both estimators with a blocked error bar. The energy is a MEAN WITH
    // AN UNCERTAINTY, not an exact value -- see docs/FCIQMC_POPULATION_CONTROL.md
    // for why the projected energy is biased at finite population and why the
    // error bar must be blocked rather than naive.
    //
    // Requires a converged RHF or ROHF common spatial-orbital reference.
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
