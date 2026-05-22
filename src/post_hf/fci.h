#ifndef HF_POSTHF_FCI_H
#define HF_POSTHF_FCI_H

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{

    // ── Full Configuration Interaction ─────────────────────────────────────────
    //
    // Exact diagonalization of the electronic Hamiltonian in the determinant basis
    // spanned by the entire MO space. This is CASCI with the active space taken to
    // be the whole basis (n_core = 0, n_act = nbasis), so it reuses the shared CI
    // engine in namespace HartreeFock::Correlation::CI without any orbital
    // optimization.
    //
    // Requires a converged RHF reference (the CI engine works in a spatial-orbital
    // / shared alpha-beta MO basis). Single-point only.
    //
    // Reads from calc._active_space:
    //   nroots        — number of CI roots to print (1 = ground state only)
    //   ci_max_dim    — abort if the determinant count exceeds this
    //   target_irrep  — target CI state irrep (empty → totally symmetric)
    //
    // Writes to:
    //   calc._total_energy        — left as the RHF reference energy
    //   calc._correlation_energy  — E_FCI - E_RHF
    //   calc._correlated_total_energy / _have_correlated_total_energy — E_FCI
    //
    std::expected<void, std::string> run_fci(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

} // namespace HartreeFock::Correlation

#endif // HF_POSTHF_FCI_H
