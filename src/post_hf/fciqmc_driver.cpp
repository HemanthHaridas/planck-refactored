#include "post_hf/fciqmc_driver.h"

#include "io/logging.h"
#include "post_hf/casscf_internal.h"
#include "post_hf/ci/ci.h"
#include "post_hf/ci/fciqmc.h"
#include "post_hf/fci.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
        double proj_numerator_sum = 0.0;
        double proj_denominator_sum = 0.0;
        int proj_n = 0;
        double ref_weight_sum = 0.0;
        double ref_weight_min = std::numeric_limits<double>::infinity();
        int ref_weight_n = 0;
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
                // THE REFERENCE WEIGHT THRESHOLD IS NOT 1e-12 HERE.
                //
                // projected_energy defaults to rejecting only c_0 == 0, which is
                // enough to avoid dividing by zero but NOT enough for the ratio to
                // be statistically meaningful. At c_0 = 1 walker, 1/c_0 swings by
                // a factor of several between consecutive steps and E[1/X] !=
                // 1/E[X] -- measured on N2/STO-3G, the projected energy came out
                // at -99.19 +/- 6.63 against an exact -107.65 while the shift
                // energy was correct to 0.14 sigma.
                //
                // Requiring several walkers on the reference makes the estimator
                // usable or honestly absent. A run whose reference is chronically
                // underpopulated reports no projected energy at all, which is the
                // correct outcome: the shift energy is still valid, and a missing
                // number is better than a wrong one.
                const double min_ref = 3.0 * std::max(1.0, opt.walker_granularity);
                const auto pe = projected_energy(pop, reference, setup->n_act, ops,
                                                 min_ref);
                if (pe.valid)
                {
                    // RATIO OF SUMS, not mean of ratios. Accumulate the numerator
                    // and denominator separately and divide once at the end:
                    //
                    //   E = H_00 + (sum_t sum_j H_0j c_j) / (sum_t c_0)
                    //
                    // Averaging the per-step ratios instead is the E[A/B] vs
                    // E[A]/E[B] error this project has now hit three times (F2.5's
                    // acceptance-rate correction, F3.4's documented bias, and
                    // here). The per-step ratio distribution is heavy-tailed --
                    // rare small c_0 produce huge 1/c_0 spikes -- so its mean is
                    // set by outliers. Measured on N2/STO-3G: changing only the
                    // reference-weight threshold moved the mean-of-ratios by
                    // 7.5 Eh (-99.19 to -106.64) at identical configuration and
                    // seed, which no well-behaved estimator does.
                    proj_numerator_sum += pe.numerator;
                    proj_denominator_sum += pe.reference_weight;
                    ++proj_n;

                    // The per-step ratios are still collected, but ONLY to give
                    // the ratio-of-sums an error bar via their scatter. They are
                    // not the estimator.
                    projected_samples.push_back(pe.energy);
                    ref_weight_sum += std::abs(pe.reference_weight);
                    ref_weight_min = std::min(ref_weight_min, std::abs(pe.reference_weight));
                    ++ref_weight_n;
                }
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

        // THE SIGN OF THE DENOMINATOR IS A TIMESTEP-STABILITY DIAGNOSTIC, and it
        // is cheaper and sharper than either energy.
        //
        // The reference determinant should hold a stable-signed weight. If
        // dt*|H_ii - S| > 2 for it, the diagonal factor (1 - dt*(H_ii - S)) falls
        // below -1 and the weight FLIPS SIGN every step -- the instability F4.3
        // gates. The signed sum then nearly cancels while |c_0| stays large.
        //
        // Measured on N2/STO-3G: at dt = 0.010 the mean |c_0| was 91.75 while the
        // mean signed c_0 was -7.50, and the projected energy came out 2.7 sigma
        // from exact. At dt = 0.001 the denominator was cleanly positive and the
        // projected energy landed within 0.6 sigma.
        //
        // THE SHIFT ENERGY DOES NOT NOTICE. It responds to the TOTAL population,
        // which is dominated by well-behaved determinants -- at dt = 0.010 it read
        // 0.14 sigma from exact while the dynamics were unstable. That asymmetry
        // is the whole reason two independent estimators are worth their cost.
        if (proj_n > 0)
        {
            const double mean_signed = proj_denominator_sum / proj_n;
            logging(LogLevel::Info, tag + " :",
                    std::format("projected numerator sum {:.6e}, denominator sum "
                                "{:.6e}, mean signed c_0 {:.2f}",
                                proj_numerator_sum, proj_denominator_sum,
                                mean_signed));

            // Compare the signed mean against the magnitude mean: they should be
            // close. A large gap means the sign is oscillating.
            const double mean_magnitude =
                (ref_weight_n > 0) ? ref_weight_sum / ref_weight_n : 0.0;
            if (mean_magnitude > 0.0
                && std::abs(mean_signed) < 0.5 * mean_magnitude)
                logging(LogLevel::Warning, tag + " :",
                        std::format("the reference determinant is SIGN-UNSTABLE "
                                    "(mean |c_0| {:.2f} but mean signed c_0 {:.2f}). "
                                    "The timestep is above the stability bound for "
                                    "this determinant; reduce fciqmc_timestep. The "
                                    "shift energy may still look converged.",
                                    mean_magnitude, mean_signed));
        }

        if (ref_weight_n > 0)
            logging(LogLevel::Info, tag + " :",
                    std::format("reference weight: mean {:.2f}, min {:.2f} walkers",
                                ref_weight_sum / ref_weight_n, ref_weight_min));

        // Report the projected energy only if it was usable often enough to mean
        // something. A handful of samples out of thousands is not an estimator.
        const bool projected_usable =
            projected_samples.size() >= static_cast<std::size_t>(shift_samples.size() / 4);
        if (!projected_samples.empty() && !projected_usable)
            logging(LogLevel::Warning, tag + " :",
                    std::format("projected energy suppressed: the reference held "
                                "enough walkers on only {} of {} steps. Raise "
                                "fciqmc_walkers.",
                                projected_samples.size(), shift_samples.size()));

        if (projected_usable)
        {
            const double e_proj = ops.diagonal(reference)
                                  + proj_numerator_sum / proj_denominator_sum
                                  + calc._nuclear_repulsion;
            // Error bar from a blocking analysis of the per-step ratios. This
            // OVERSTATES the uncertainty on the ratio-of-sums (which averages the
            // tails away), so it is conservative -- an overestimate fails a gate
            // loudly, an underestimate passes one silently.
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
