#ifndef HF_POSTHF_CASSCF_H
#define HF_POSTHF_CASSCF_H

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{

    // ── CASSCF ────────────────────────────────────────────────────────────────────
    //
    // State-averaged (or single-state) Complete Active Space SCF.
    //
    // Reads from calc._active_space:
    //   nactele       — number of active electrons
    //   nactorb       — number of active orbitals
    //   nroots        — number of CI roots (1 = single-state)
    //   weights       — SA weights (empty → equal weights)
    //   mcscf_debug_numeric_newton — debug-only numeric Newton fallback toggle
    //   mcscf_max_iter, mcscf_micro_per_macro, tol_mcscf_energy, tol_mcscf_grad
    //   ci_max_dim    — abort if CI space exceeds this
    //   target_irrep  — target CI state irrep (empty → totally symmetric)
    //
    // Writes to:
    //   calc._total_energy         — final CASSCF total energy
    //   calc._cas_nat_occ          — active natural occupation numbers (descending)
    //   calc._cas_mo_coefficients  — converged MO coefficients [nb×nb] in the
    //                                 optimization basis
    //
    std::expected<void, std::string> run_casscf(
        HartreeFock::Calculator&                   calc,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);


    // ── RASSCF ────────────────────────────────────────────────────────────────────
    //
    // Restricted Active Space SCF.  Active space is partitioned as
    //   RAS1 (nras1 orbs, at most max_holes holes)
    //   RAS2 (nras2 orbs, full CI)
    //   RAS3 (nras3 orbs, at most max_elec electrons)
    //
    // nactorb must equal nras1 + nras2 + nras3.
    //
    std::expected<void, std::string> run_rasscf(
        HartreeFock::Calculator&                   calc,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

} // namespace HartreeFock::Correlation

#endif // HF_POSTHF_CASSCF_H
