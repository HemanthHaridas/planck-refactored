#include "post_hf/fci.h"

#include "base/tables.h"
#include "io/logging.h"
#include "post_hf/casscf_internal.h"
#include "post_hf/ci/ci.h"
#include "post_hf/ci/strings.h"
#include "post_hf/integrals.h"

#include <algorithm>
#include <format>

namespace HartreeFock::Correlation
{
    using HartreeFock::LogLevel;
    using HartreeFock::Logger::logging;
    using HartreeFock::Correlation::CASSCFInternal::CIString;
    using HartreeFock::Correlation::CASSCFInternal::RASParams;
    using HartreeFock::Correlation::CI::build_ci_space;
    using HartreeFock::Correlation::CI::build_spin_strings_unfiltered;
    using HartreeFock::Correlation::CI::CIDeterminantSpace;
    using HartreeFock::Correlation::CI::CISolveResult;
    using HartreeFock::Correlation::CI::solve_ci;

    std::expected<void, std::string> run_fci(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        const std::string tag = "FCI";

        // ── Reference and method guards ────────────────────────────────────────
        // FCI here is CASCI over the whole basis; it reuses the spatial-orbital CI
        // engine and so requires a converged RHF reference.
        if (!calc._info._is_converged)
            return std::unexpected(tag + ": requires a converged RHF reference.");
        if (calc._scf._scf != HartreeFock::SCFType::RHF)
            return std::unexpected(tag + ": only RHF reference supported.");

        const int nbasis = static_cast<int>(calc._shells.nbasis());
        if (nbasis <= 0)
            return std::unexpected(tag + ": empty basis.");

        // The whole basis becomes the active space: no inactive core, no virtuals.
        const int n_core = 0;
        const int n_act = nbasis;

        // The packed alpha/beta determinant encoding caps how many spatial
        // orbitals the whole-basis CI can address. Fail loudly rather than
        // silently truncating.
        if (n_act > CASSCFInternal::kMaxPackedSpatialOrbitals)
            return std::unexpected(std::format(
                "{}: basis has {} orbitals, exceeding the packed determinant limit ({}). "
                "FCI over the whole MO space is only available for small bases.",
                tag, n_act, CASSCFInternal::kMaxPackedSpatialOrbitals));

        const int n_total_elec =
            static_cast<int>(calc._molecule.atomic_numbers.cast<int>().sum()) - calc._molecule.charge;
        if (n_total_elec <= 0)
            return std::unexpected(tag + ": non-positive electron count.");

        const int multiplicity = static_cast<int>(calc._molecule.multiplicity);
        const int n_alpha = (n_total_elec + (multiplicity - 1)) / 2;
        const int n_beta = n_total_elec - n_alpha;
        if (n_alpha < 0 || n_beta < 0 || n_alpha > n_act || n_beta > n_act)
            return std::unexpected(std::format(
                "{}: invalid electron/multiplicity combination ({} electrons, multiplicity {}).",
                tag, n_total_elec, multiplicity));
        if ((n_alpha - n_beta) != (multiplicity - 1))
            return std::unexpected(std::format(
                "{}: multiplicity {} is inconsistent with {} electrons.",
                tag, multiplicity, n_total_elec));

        // ── CI dimension guard ─────────────────────────────────────────────────
        auto nchoose = [](int n, int k) -> long long
        {
            if (k > n || k < 0)
                return 0;
            if (k == 0 || k == n)
                return 1;
            long long r = 1;
            for (int i = 0; i < k; ++i)
                r = r * (n - i) / (i + 1);
            return r;
        };
        const long long ci_dim_est = nchoose(n_act, n_alpha) * nchoose(n_act, n_beta);
        if (ci_dim_est > static_cast<long long>(calc._active_space.ci_max_dim))
            return std::unexpected(std::format(
                "{}: CI dimension ({}) exceeds ci_max_dim ({}). "
                "Increase ci_max_dim in [scf] to run a larger FCI.",
                tag, ci_dim_est, calc._active_space.ci_max_dim));

        const int nroots = std::max(1, calc._active_space.nroots);

        // ── MO basis and effective integrals ───────────────────────────────────
        const Eigen::MatrixXd &C = calc._info._scf.alpha.mo_coefficients;
        if (C.rows() != nbasis || C.cols() != nbasis)
            return std::unexpected(tag + ": MO coefficient matrix has wrong size.");

        std::vector<double> eri_local;
        const std::vector<double> &eri =
            ensure_eri(calc, shell_pairs, eri_local, tag + " :");

        // With no inactive core, the effective one-electron integrals are just the
        // core Hamiltonian in the MO basis, and the core energy is zero — so the CI
        // eigenvalue plus nuclear repulsion is the total electronic energy.
        const Eigen::MatrixXd h_eff = C.transpose() * calc._hcore * C;
        const std::vector<double> ga = transform_eri_internal(eri, nbasis, C);

        logging(LogLevel::Info, tag + " :",
                std::format("Full CI over {} orbitals, {} alpha / {} beta electrons  (CI dim = {})",
                            n_act, n_alpha, n_beta, ci_dim_est));
        if (nroots > 1)
            logging(LogLevel::Info, tag + " :",
                    std::format("Solving for {} lowest roots", nroots));

        // ── Build determinant space and solve ──────────────────────────────────
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(n_act, n_alpha, n_beta, a_strs, b_strs);

        RASParams ras; // inactive: full CI, no RAS restriction
        CIDeterminantSpace ci_space = build_ci_space(a_strs, b_strs, ras, h_eff, ga, n_act);
        if (ci_space.dets.empty())
            return std::unexpected(tag + ": empty determinant space.");

        const int nroots_avail = std::min(nroots, static_cast<int>(ci_space.dets.size()));
        CISolveResult ci = solve_ci(ci_space, a_strs, b_strs, h_eff, ga, n_act, nroots_avail);
        if (ci.energies.size() < 1)
            return std::unexpected(tag + ": CI solve returned no roots.");

        // ── Energies ───────────────────────────────────────────────────────────
        const double e_ground = ci.energies(0) + calc._nuclear_repulsion;
        const double e_corr = e_ground - calc._total_energy;

        calc._correlation_energy = e_corr;
        calc._correlated_total_energy = e_ground;
        calc._have_correlated_total_energy = true;

        if (nroots_avail > 1)
        {
            logging(LogLevel::Info, tag + " :", "Root energies (total, Eh):");
            for (int r = 0; r < nroots_avail; ++r)
                logging(LogLevel::Info, tag + " :",
                        std::format("  root {:2d}: {:.10f}",
                                    r, ci.energies(r) + calc._nuclear_repulsion));
        }

        return {};
    }

} // namespace HartreeFock::Correlation
