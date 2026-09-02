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
#include <algorithm>
#include <unordered_map>
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

        // The reference determinant: the one with the LOWEST DIAGONAL ENERGY.
        //
        // NOT simply "occupy the lowest-index orbitals". That assumption was wrong
        // and it was wrong on the gated fixture: on N2/STO-3G the Aufbau
        // determinant is 0xbf (orbitals [0,1,2,3,4,5,7]) while the lowest-index
        // form gives 0x7f ([0,1,2,3,4,5,6]). MO 6 lies ABOVE MO 7 in the converged
        // SCF ordering, so index order is not energy order.
        //
        // The consequences were silent rather than loud, which is why it survived:
        //   - the projected energy E = H_00 + (sum_j H_0j c_j)/c_0 anchors on this
        //     determinant, so a weakly-occupied choice inflates its variance (N2's
        //     projected error bar ran ~20x the shift's) and biases the ratio;
        //   - <N_I>/<N_0> normalises by it, and the true reference carried 14.2x
        //     the weight of the one being used as the unit.
        // The SHIFT energy never touches it, which is exactly why the shift looked
        // healthy throughout and hid the defect.
        //
        // Minimising ops.diagonal rather than reading SCF MO energies is
        // deliberate: it uses the SAME slater_condon_element the propagator uses,
        // so the reference cannot disagree with the Hamiltonian being sampled. The
        // search is over single orbital swaps from the Aufbau guess (occupied ->
        // virtual, one at a time, repeated until no swap lowers the diagonal),
        // which is a hill-climb rather than an exhaustive scan -- exhaustive is
        // C(n,k)^2 determinants and this runs once per calculation.
        //
        // ponytail: hill-climb, not exhaustive. A determinant that is lowest only
        // via a simultaneous multi-orbital swap would be missed; if that ever
        // matters, seed from the SCF occupation instead of searching from Aufbau.
        DetKey reference_determinant(int n_alpha, int n_beta, int n_act,
                                     const HamiltonianOps &ops)
        {
            auto aufbau = [](int n) {
                CIString s = 0;
                for (int i = 0; i < n; ++i)
                    s |= (CIString{1} << i);
                return s;
            };
            DetKey best{aufbau(n_alpha), aufbau(n_beta)};
            double best_e = ops.diagonal(best);

            // Repeat until a full sweep finds no improvement: one pass fixes a
            // single misordered pair, but several orbitals can be out of order.
            bool improved = true;
            while (improved)
            {
                improved = false;
                for (int spin = 0; spin < 2; ++spin)
                {
                    CIString &str = (spin == 0) ? best.alpha : best.beta;
                    for (int occ = 0; occ < n_act; ++occ)
                    {
                        if (!(str >> occ & 1))
                            continue;
                        for (int vir = 0; vir < n_act; ++vir)
                        {
                            if (str >> vir & 1)
                                continue;
                            const CIString saved = str;
                            str = (str & ~(CIString{1} << occ)) | (CIString{1} << vir);
                            const double e = ops.diagonal(best);
                            if (e < best_e - 1e-12)
                            {
                                best_e = e;
                                improved = true;
                            }
                            else
                            {
                                str = saved;
                            }
                        }
                    }
                }
            }
            return best;
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
        DetKey reference = reference_determinant(setup->n_alpha, setup->n_beta,
                                                 setup->n_act, ops);
        // Print it. The wrong reference produced no error and no warning -- only a
        // wide projected-energy error bar that read as ordinary noise -- so the
        // determinant that everything is normalised against is now visible in the
        // output and directly comparable with the FCI dump's reference line.
        logging(LogLevel::Info, tag + " :",
                std::format("reference determinant {:#018x}/{:#018x}, "
                            "diagonal {:.10f} Eh",
                            static_cast<unsigned long long>(reference.alpha),
                            static_cast<unsigned long long>(reference.beta),
                            ops.diagonal(reference)));

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
        // Running sum of SIGNED weight per determinant over the sampling phase.
        //
        // Signed, and summed over steps rather than snapshotted: an instantaneous
        // population is dominated by shot noise (many determinants hold a single
        // walker whose sign flips step to step), while <N_I> converges to the
        // wavefunction. The sign is the whole point -- a magnitude-only average
        // would agree with a broken sampler that got every phase wrong.
        std::unordered_map<DetKey, double, DetKeyHash> signed_population;
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

            // RE-ANCHOR THE PROJECTION REFERENCE ONCE, AT THE END OF EQUILIBRATION.
            //
            // The projected energy E = H_00 + (sum_j H_0j c_j)/c_0 assumes the
            // reference carries a dominant share of the wavefunction. On a
            // DEGENERATE ground state that assumption decays with time: any
            // mixture of degenerate eigenstates is itself an eigenstate at the
            // same energy, so the imaginary-time dynamics apply NO restoring
            // force within the degenerate manifold and the population random-walks
            // between the partners.
            //
            // Measured on C2/STO-3G, whose ground state is doubly degenerate (FCI
            // roots 0 and 1 both -74.6406501646, partners 0x3f/0x6f and 0x6f/0x3f
            // at +/-1.000000). Holding everything else fixed and varying only the
            // equilibration length, the partner-to-anchor ratio and the projected
            // energy degrade together:
            //
            //   equil    partner/anchor    E_proj          sigma vs exact
            //   20000    -0.861            -74.6172886     +2.62
            //   40000    -1.674            -74.6413697     -0.06
            //   60000    -3.833            -74.7503958     -5.57
            //
            // By 60000 the anchor holds a quarter of the partner's weight, the
            // numerator still samples the whole manifold, and the ratio inflates
            // NEGATIVELY -- reporting an energy 5.6 sigma BELOW the variational
            // minimum. Nothing is unstable: the sign is steady, the population is
            // controlled, the reference holds 743 walkers. The run is fine and the
            // ESTIMATOR is measuring the wrong thing.
            //
            // The shift energy is immune (-1.05/-0.14/-0.49 sigma across the same
            // three runs) because it responds to total population growth, which is
            // indifferent to how weight is distributed inside the manifold. That
            // asymmetry is the mirror image of the N2 sign-instability finding,
            // where the projected energy caught what the shift could not -- neither
            // estimator dominates, which is why both are computed.
            //
            // Re-anchoring on the largest-weight determinant is what the
            // deterministic FCI coefficient dump already does, for the same reason.
            // Done ONCE rather than per step: a reference that moves during
            // sampling would change what the accumulated ratio-of-sums means
            // partway through, which is worse than a slightly suboptimal anchor.
            if (step == opt.equilibration_steps)
            {
                const DetKey seeded = reference;
                double best_w = std::abs(pop.weight_at(reference));
                DetKey best = reference;
                // Deterministic scan: the population is a hash map, so ties must
                // break on the bitstrings or the anchor depends on iteration order
                // and the run stops being reproducible at fixed seed.
                for (const auto &[det, w] : pop)
                {
                    const double aw = std::abs(w);
                    if (aw > best_w
                        || (aw == best_w
                            && (det.alpha < best.alpha
                                || (det.alpha == best.alpha && det.beta < best.beta))))
                    {
                        best_w = aw;
                        best = det;
                    }
                }

                const double seeded_w = std::abs(pop.weight_at(seeded));
                // Warn whenever the seeded reference has been overtaken by a
                // margin, whether or not re-anchoring fixes it -- a large drift
                // says the state is degenerate or near-degenerate, which the user
                // should know regardless.
                if (seeded_w > 0.0 && best_w > 2.0 * seeded_w)
                    logging(LogLevel::Warning, tag + " :",
                            std::format("REFERENCE DRIFT: determinant "
                                        "{:#018x}/{:#018x} now carries {:.2f} walkers "
                                        "against the seeded reference's {:.2f} "
                                        "({:.1f}x). This is the signature of a "
                                        "degenerate or near-degenerate ground state, "
                                        "in which the population drifts freely "
                                        "between partners. Re-anchoring the "
                                        "projection; the shift energy is unaffected.",
                                        static_cast<unsigned long long>(best.alpha),
                                        static_cast<unsigned long long>(best.beta),
                                        best_w, seeded_w, best_w / seeded_w));

                if (!(best == reference))
                {
                    reference = best;
                    logging(LogLevel::Info, tag + " :",
                            std::format("projection re-anchored to {:#018x}/{:#018x} "
                                        "({:.2f} walkers) after equilibration",
                                        static_cast<unsigned long long>(best.alpha),
                                        static_cast<unsigned long long>(best.beta),
                                        best_w));
                }
            }

            if (step >= opt.equilibration_steps)
            {
                shift_samples.push_back(ctl.shift);
                for (const auto &[det, w] : pop)
                    signed_population[det] += w;
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

        // ── Coefficient ratios <N_I>/<N_0> ─────────────────────────────────────
        //
        // The sampled wavefunction, in the form that can be compared directly
        // against deterministic FCI's C_I/C_0 (printed by `correlation fci` on the
        // same input at `verbosity verbose`). This is a stronger check than energy
        // agreement: the energy is one scalar contracted over the whole vector, so
        // errors in spawning phases, death/cloning and annihilation can cancel
        // inside it. Ratios expose the vector, sign included.
        //
        // Ratios rather than raw populations because a walker population is
        // normalised by its own target and an FCI eigenvector by its norm; only
        // the ratio against the reference is common to both.
        if (calc._output._verbosity >= Verbosity::Verbose && !signed_population.empty())
        {
            std::vector<std::pair<DetKey, double>> rows(signed_population.begin(),
                                                        signed_population.end());
            // Sort by magnitude, breaking ties on the bitstrings. The tie-break is
            // load-bearing, not cosmetic: `signed_population` is a hash map, so
            // without a total order the printed list depends on iteration order.
            std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
                if (std::abs(a.second) != std::abs(b.second))
                    return std::abs(a.second) > std::abs(b.second);
                if (a.first.alpha != b.first.alpha)
                    return a.first.alpha < b.first.alpha;
                return a.first.beta < b.first.beta;
            });

            const double n0 = signed_population.count(reference)
                                  ? signed_population.at(reference)
                                  : 0.0;
            if (n0 == 0.0)
                logging(LogLevel::Warning, tag + " :",
                        "reference determinant carries no accumulated weight; "
                        "coefficient ratios are unavailable.");
            else
            {
                logging(LogLevel::Info, tag + " :",
                        std::format("Dominant determinants (alpha/beta bitstrings, "
                                    "<N_I>/<N_0> against reference {:#018x}/{:#018x}, "
                                    "{} occupied):",
                                    static_cast<unsigned long long>(reference.alpha),
                                    static_cast<unsigned long long>(reference.beta),
                                    rows.size()));
                const std::size_t n_show = std::min<std::size_t>(rows.size(), 20);
                for (std::size_t k = 0; k < n_show; ++k)
                    logging(LogLevel::Info, tag + " :",
                            std::format("  det {:#018x}/{:#018x}  <N_I>/<N_0> {:+.6f}",
                                        static_cast<unsigned long long>(rows[k].first.alpha),
                                        static_cast<unsigned long long>(rows[k].first.beta),
                                        rows[k].second / n0));
            }
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
