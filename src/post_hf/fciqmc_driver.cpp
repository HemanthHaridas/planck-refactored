#include "post_hf/fciqmc_driver.h"

#include "io/logging.h"
#include "post_hf/casscf_internal.h"
#include "post_hf/ci/ci.h"
#include "post_hf/ci/fciqmc.h"
#include "post_hf/fci.h"

#include <cmath>
#include <format>
#include <vector>

namespace HartreeFock::Correlation
{
    using HartreeFock::LogLevel;
    using HartreeFock::Logger::logging;
    using namespace HartreeFock::Correlation::CI::QMC;

    namespace
    {
        // Adapt the shared all-MO setup to the callbacks the propagator wants.
        //
        // slater_condon_element is the SAME matrix-element routine the
        // deterministic CI uses, so the two paths cannot disagree about the
        // Hamiltonian -- only about how they solve it.
        HamiltonianOps make_ops(const AllMOCISetup &setup)
        {
            const int n_act = setup.n_act;
            const Eigen::MatrixXd *h = &setup.h_eff;
            const std::vector<double> *g = &setup.ga;

            HamiltonianOps ops;
            ops.off_diagonal = [h, g, n_act](const DetKey &i, const DetKey &j) {
                return CI::slater_condon_element(i.alpha, i.beta, j.alpha, j.beta,
                                                 *h, *g, n_act);
            };
            ops.diagonal = [h, g, n_act](const DetKey &i) {
                return CI::slater_condon_element(i.alpha, i.beta, i.alpha, i.beta,
                                                 *h, *g, n_act);
            };
            return ops;
        }

        // The lowest-energy determinant: alpha and beta electrons in the lowest
        // orbitals. Used as both the starting population and the projection
        // reference, matching how the deterministic path orders its determinants.
        DetKey reference_determinant(int n_alpha, int n_beta)
        {
            CIString a = 0, b = 0;
            for (int i = 0; i < n_alpha; ++i)
                a |= (CIString{1} << i);
            for (int i = 0; i < n_beta; ++i)
                b |= (CIString{1} << i);
            return DetKey{a, b};
        }
    } // namespace

    std::expected<void, std::string> run_fciqmc(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        const std::string tag = "FCIQMC";

        auto setup = build_all_mo_ci_setup(calc, shell_pairs, tag);
        if (!setup)
            return std::unexpected(setup.error());

        const auto &opt = calc._fciqmc;
        if (!(opt.target_walkers > 0.0))
            return std::unexpected(tag + ": fciqmc_walkers must be positive.");
        if (!(opt.timestep > 0.0))
            return std::unexpected(tag + ": fciqmc_timestep must be positive.");
        if (opt.sampling_steps < 4)
            return std::unexpected(tag + ": fciqmc_steps must be at least 4.");

        const auto ops = make_ops(*setup);
        const DetKey reference = reference_determinant(setup->n_alpha, setup->n_beta);

        logging(LogLevel::Info, tag + " :",
                std::format("Sampling {} orbitals, {} alpha / {} beta electrons  "
                            "(CI dim = {})",
                            setup->n_act, setup->n_alpha, setup->n_beta, setup->ci_dim));
        // Every parameter that changes the answer is echoed, so a result can be
        // reproduced from its own output alone. The seed especially: F3.5's
        // contract is that the same seed gives the same trajectory bitwise, and
        // that is unusable if the value is not recorded.
        logging(LogLevel::Info, tag + " :",
                std::format("target walkers {:.0f}, dt {:g}, seed {}",
                            opt.target_walkers, opt.timestep, opt.seed));
        logging(LogLevel::Info, tag + " :",
                std::format("zeta {:g}, xi {:g}, interval {}, granularity {:g}, "
                            "initiator {:g}",
                            opt.shift_damping, opt.shift_restoring,
                            opt.shift_interval, opt.walker_granularity,
                            opt.initiator_threshold));
        logging(LogLevel::Info, tag + " :",
                std::format("{} equilibration + {} sampling steps, {} spawn "
                            "attempts per walker",
                            opt.equilibration_steps, opt.sampling_steps,
                            opt.spawn_attempts));

        // ── Propagate ──────────────────────────────────────────────────────────
        WalkerPopulation pop;
        pop.add(reference, opt.target_walkers);

        ShiftController ctl;
        ctl.shift = ops.diagonal(reference);
        ctl.target_population = opt.target_walkers;
        ctl.zeta = opt.shift_damping;
        ctl.xi = opt.shift_restoring;
        ctl.interval = opt.shift_interval;

        RandomSource rng(opt.seed);
        std::vector<double> shift_samples;
        std::vector<double> projected_samples;
        const int total_steps = opt.equilibration_steps + opt.sampling_steps;

        for (int step = 0; step < total_steps; ++step)
        {
            pop = propagate_stochastic(pop, setup->n_act, ops, opt.timestep,
                                       ctl.shift, rng, opt.spawn_attempts,
                                       opt.walker_granularity,
                                       opt.initiator_threshold);
            pop.compress(1e-12);

            const double n = ordered_l1_norm(pop);
            // A collapsed or diverged population carries no energy. Reporting a
            // number from one would be the most misleading possible output, so
            // fail loudly instead.
            if (!std::isfinite(n) || n <= 0.0)
                return std::unexpected(std::format(
                    "{}: walker population collapsed at step {}. Reduce "
                    "fciqmc_timestep or raise fciqmc_walkers.", tag, step));
            if (n > opt.target_walkers * 1e6)
                return std::unexpected(std::format(
                    "{}: walker population diverged at step {} ({:.3g} walkers "
                    "against a target of {:.0f}). The timestep is almost certainly "
                    "above the stability bound.", tag, step, n, opt.target_walkers));

            ctl.update(n, opt.timestep);

            if (step >= opt.equilibration_steps)
            {
                shift_samples.push_back(ctl.shift);
                const auto pe = projected_energy(pop, reference, setup->n_act, ops);
                if (pe.valid)
                    projected_samples.push_back(pe.energy);
            }
        }

        if (shift_samples.size() < 4)
            return std::unexpected(tag + ": too few samples to average.");

        // ── Estimators ─────────────────────────────────────────────────────────
        auto mean_of = [](const std::vector<double> &v) {
            double s = 0.0;
            for (double x : v)
                s += x;
            return v.empty() ? 0.0 : s / static_cast<double>(v.size());
        };

        const double e_shift = mean_of(shift_samples) + calc._nuclear_repulsion;
        const double shift_err = blocked_standard_error(shift_samples);

        logging(LogLevel::Info, tag + " :",
                std::format("Shift energy     {:.10f} +/- {:.2e}  ({} samples)",
                            e_shift, shift_err, shift_samples.size()));

        if (!projected_samples.empty())
        {
            const double e_proj = mean_of(projected_samples) + calc._nuclear_repulsion;
            const double proj_err = blocked_standard_error(projected_samples);
            logging(LogLevel::Info, tag + " :",
                    std::format("Projected energy {:.10f} +/- {:.2e}  ({} samples)",
                                e_proj, proj_err, projected_samples.size()));
            // The two share no arithmetic, so a large gap between them is a
            // symptom worth surfacing rather than averaging away.
            const double gap = std::abs(e_proj - e_shift);
            const double tol = 5.0 * std::max(shift_err, proj_err);
            if (std::isfinite(tol) && gap > tol)
                logging(LogLevel::Warning, tag + " :",
                        std::format("shift and projected energies differ by {:.3e}, "
                                    "beyond 5 sigma ({:.3e}) -- the run may not be "
                                    "equilibrated", gap, tol));
        }

        // The shift energy is the reported value: it is the estimator population
        // control produces directly, and unlike the projected energy it carries no
        // finite-population ratio bias.
        calc._correlation_energy = e_shift - calc._total_energy;
        calc._correlated_total_energy = e_shift;
        calc._have_correlated_total_energy = true;
        return {};
    }

} // namespace HartreeFock::Correlation
