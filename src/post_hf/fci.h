#ifndef HF_POSTHF_FCI_H
#define HF_POSTHF_FCI_H

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"

#include <Eigen/Core>
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
    // Requires a converged RHF or ROHF reference (the CI engine works in a spatial-orbital
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
    // Everything fixed-orbital CI needs from a converged reference, built once.
    //
    // Extracted so FCIQMC consumes the SAME integral transform run_fci does. If
    // the two built h_eff/ga independently they would drift, and every comparison
    // between the stochastic and deterministic answers would then be ambiguous:
    // a disagreement could be sampling, or it could be plumbing.
    struct FixedOrbitalCISetup
    {
        int n_act = 0;              // spatial orbitals sampled by CI
        int n_core = 0;             // doubly occupied frozen orbitals
        double core_energy = 0.0;   // electronic inactive-core constant
        int n_alpha = 0;
        int n_beta = 0;
        long long ci_dim = 0;       // determinant count, for logging and guards
        Eigen::MatrixXd h_eff;      // active block of the inactive Fock matrix
        std::vector<double> ga;     // transformed two-electron integrals
    };

    struct CIIntegralTimings
    {
        double ao_eri_seconds = 0.0;  // post-SCF lookup or construction
        double mo_eri_seconds = 0.0;  // RI includes fitting and reconstruction
    };

    enum class CIOrbitalSpace { All, Active };

    // Validate the reference, check the packed-orbital and ci_max_dim limits, and
    // build h_eff / ga. `tag` prefixes error messages so the caller's method name
    // appears in them.
    std::expected<FixedOrbitalCISetup, std::string> build_fixed_orbital_ci_setup(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const std::string &tag,
        CIOrbitalSpace space, CIIntegralTimings *timings = nullptr);

    std::expected<void, std::string> run_fci(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

} // namespace HartreeFock::Correlation

#endif // HF_POSTHF_FCI_H
