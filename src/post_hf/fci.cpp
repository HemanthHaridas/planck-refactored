#include "post_hf/fci.h"

#include "base/tables.h"
#include "io/logging.h"
#include "post_hf/casscf_internal.h"
#include "post_hf/ci/ci.h"
#include "post_hf/ci/strings.h"
#include "post_hf/integrals.h"

#include <algorithm>
#include <format>
#include <numeric>

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

    std::expected<AllMOCISetup, std::string> build_all_mo_ci_setup(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const std::string &tag)
    {
        const int nbasis = static_cast<int>(calc._info._scf.alpha.mo_coefficients.rows());
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

        // ── MO basis and effective integrals ───────────────────────────────────
        // For RHF the alpha channel holds the (only) MO set; for ROHF the alpha and
        // beta channels hold the same common spatial orbitals, so reading the alpha
        // channel gives the genuine reference orbitals for both spins.
        const Eigen::MatrixXd &C = calc._info._scf.alpha.mo_coefficients;
        if (C.rows() != nbasis || C.cols() != nbasis)
            return std::unexpected(tag + ": MO coefficient matrix has wrong size.");

        // With no inactive core, the effective one-electron integrals are just the
        // core Hamiltonian in the MO basis, and the core energy is zero — so the CI
        // eigenvalue plus nuclear repulsion is the total electronic energy.
        const Eigen::MatrixXd h_eff = C.transpose() * calc._hcore * C;

        // FCI's only ERI consumer is the all-MO (pq|rs) transform, so it opts
        // into the density-fitted path with a single call swap. RI makes FCI
        // approximate (density-fitting error on the two-electron integrals), so
        // it is strictly opt-in via mp2_use_ri and never the default here.
        std::vector<double> ga;
        if (calc._mp2.use_ri)
        {
            auto ga_ri = transform_eri_internal_ri(calc, C);
            if (!ga_ri)
                return std::unexpected(tag + ": " + ga_ri.error());
            ga = std::move(*ga_ri);
        }
        else
        {
            std::vector<double> eri_local;
            const std::vector<double> &eri =
                ensure_eri(calc, shell_pairs, eri_local, tag + " :");
            ga = transform_eri_internal(eri, nbasis, C);
        }

        AllMOCISetup out;
        out.n_act = n_act;
        out.n_alpha = n_alpha;
        out.n_beta = n_beta;
        out.ci_dim = ci_dim_est;
        out.h_eff = std::move(h_eff);
        out.ga = std::move(ga);
        return out;
    }

    std::expected<void, std::string> run_fci(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        const std::string tag = "FCI";

        // ── Reference and method guards ────────────────────────────────────────
        // FCI here is CASCI over the whole basis; it reuses the spatial-orbital CI
        // engine, which assumes a single common spatial-orbital set for both spins.
        // Both RHF and ROHF satisfy that (ROHF stores one common orbital set for
        // alpha and beta), so either can serve as the reference. The FCI energy is
        // invariant to the orbital choice; only the correlation energy
        // (E_FCI - E_ref) depends on which reference was used.
        if (!calc._info._is_converged)
            return std::unexpected(tag + ": requires a converged RHF or ROHF reference.");
        if (calc._scf._scf != HartreeFock::SCFType::RHF &&
            calc._scf._scf != HartreeFock::SCFType::ROHF)
            return std::unexpected(tag + ": only RHF or ROHF references supported.");

        // Spherical mode: the CI active space is the full MO space, whose dimension
        // is the spherical working basis (working_nbasis()), not the larger Cartesian
        // nbasis(). The cached ERI and MO coefficients are both spherical.
        const int nbasis = static_cast<int>(calc.working_nbasis());
        if (nbasis <= 0)
            return std::unexpected(tag + ": empty basis.");

        // The whole basis becomes the active space: no inactive core, no virtuals.
        const int n_core = 0;
        const int n_act = nbasis;

        auto setup = build_all_mo_ci_setup(calc, shell_pairs, tag);
        if (!setup)
            return std::unexpected(setup.error());
        const int n_alpha = setup->n_alpha;
        const int n_beta = setup->n_beta;
        const long long ci_dim_est = setup->ci_dim;
        const Eigen::MatrixXd &h_eff = setup->h_eff;
        const std::vector<double> &ga = setup->ga;
        const int nroots = std::max(1, calc._active_space.nroots);

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

        // ── Coefficient ratios C_I / C_0 ───────────────────────────────────────
        //
        // The reference for a stochastic method to compare its sampled
        // wavefunction against. Energy agreement is a weaker test: it is one
        // scalar contracted over the whole vector, so errors in spawning phases,
        // death/cloning and annihilation can cancel within it. Ratios expose the
        // vector itself, sign included.
        //
        // Ratios rather than raw coefficients because the FCI eigenvector's
        // overall phase and normalisation are both arbitrary, while a walker
        // population is normalised by its own population and anchored on the
        // reference. C_I/C_0 is the quantity both sides agree on.
        //
        // Printed at Verbose because it is a validation instrument, not part of a
        // normal run's output.
        if (calc._output._verbosity >= Verbosity::Verbose && ci.vectors.cols() > 0)
        {
            // The reference is the largest-weight determinant, not det 0: the
            // enumeration order is not guaranteed to put the RHF determinant
            // first, and a ratio against a near-zero denominator is noise.
            const Eigen::VectorXd v = ci.vectors.col(0);
            int i_ref = 0;
            for (int i = 1; i < v.size(); ++i)
                if (std::abs(v(i)) > std::abs(v(i_ref)))
                    i_ref = i;

            std::vector<int> order(static_cast<std::size_t>(v.size()));
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return std::abs(v(a)) > std::abs(v(b));
            });

            const double c0 = v(i_ref);
            const std::size_t n_show =
                std::min<std::size_t>(order.size(), 20);
            logging(LogLevel::Info, tag + " :",
                    std::format("Dominant determinants (alpha/beta bitstrings, "
                                "C_I/C_0 against reference {:#018x}/{:#018x}):",
                                static_cast<unsigned long long>(a_strs[ci_space.dets[i_ref].first]),
                                static_cast<unsigned long long>(b_strs[ci_space.dets[i_ref].second])));
            for (std::size_t k = 0; k < n_show; ++k)
            {
                const int i = order[k];
                const auto &d = ci_space.dets[i];
                logging(LogLevel::Info, tag + " :",
                        std::format("  det {:#018x}/{:#018x}  C_I/C_0 {:+.6f}",
                                    static_cast<unsigned long long>(a_strs[d.first]),
                                    static_cast<unsigned long long>(b_strs[d.second]),
                                    v(i) / c0));
            }
        }

        return {};
    }

} // namespace HartreeFock::Correlation
