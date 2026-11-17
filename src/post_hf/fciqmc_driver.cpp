#include "post_hf/fciqmc_driver.h"

#include "io/logging.h"
#include "post_hf/casscf_internal.h"
#include "post_hf/ci/ci.h"
#include "post_hf/ci/fciqmc.h"
#include "post_hf/ci/spawn_accumulator.h"
#include "post_hf/fci.h"

#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <format>
#include <iostream>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

namespace HartreeFock::Correlation
{
    using HartreeFock::LogLevel;
    using HartreeFock::Logger::logging;
    using namespace HartreeFock::Correlation::CI::QMC;

    namespace
    {
        // Process high-water mark through this point. The units of ru_maxrss
        // are bytes on Darwin and KiB on Linux.
        std::int64_t peak_rss_bytes()
        {
#if defined(__APPLE__) || defined(__linux__)
            struct rusage usage{};
            if (getrusage(RUSAGE_SELF, &usage) == 0)
            {
#ifdef __APPLE__
                return usage.ru_maxrss;
#else
                return static_cast<std::int64_t>(usage.ru_maxrss) * 1024;
#endif
            }
#endif
            return -1;
        }

        // Diagnostic only. FNV-1a on little-endian words, including all bits
        // of every double. An identity for comparing runs, not a physics oracle.
        struct RunFingerprint
        {
            std::uint64_t value = 14695981039346656037ULL;
            void word(std::uint64_t x)
            {
                for (int i = 0; i < 8; ++i, x >>= 8)
                    value = (value ^ (x & 255)) * 1099511628211ULL;
            }
            void real(double x) { word(std::bit_cast<std::uint64_t>(x)); }
        };

        void log_hamiltonian_identity(const FixedOrbitalCISetup &setup,
                                      const HartreeFock::Calculator &calc, double build_seconds)
        {
            RunFingerprint id;
            id.word(setup.n_act);
            id.word(setup.n_alpha);
            id.word(setup.n_beta);
            id.real(calc._nuclear_repulsion + setup.core_energy);
            for (int i = 0; i < setup.n_act; ++i)
                for (int j = 0; j < setup.n_act; ++j)
                    id.real(setup.h_eff(i, j));
            for (double g : setup.ga)
                id.real(g);
            const auto &C = calc._info._scf.alpha.mo_coefficients;
            const Eigen::MatrixXd metric = C.transpose() * calc._overlap * C;
            const double error = (metric - Eigen::MatrixXd::Identity(
                metric.rows(), metric.cols())).cwiseAbs().maxCoeff();
            logging(LogLevel::Info, "FCIQMC_PROFILE:",
                    std::format("phase=hamiltonian id={:016x} nuclear_e={:.12f} "
                                "mo_orthogonality_max_error={:.3e} build_s={:.6f}",
                                id.value, calc._nuclear_repulsion, error, build_seconds));
        }

        struct FCIQMCDriverWork
        {
            std::uint64_t steps = 0;
            double prefill_seconds = 0.0;
            double compress_seconds = 0.0;
            double estimator_seconds = 0.0;
        };

        void log_profile(const std::string &phase,
                         const FCIQMCDriverWork &driver,
                         const FCIQMCPropagationProfile &propagation, double phase_seconds)
        {
            std::uint64_t parents = 0;
            std::uint64_t attempts = 0;
            std::uint64_t invalid = 0;
            std::uint64_t zero_h = 0;
            std::uint64_t initiator = 0;
            std::uint64_t rounded = 0;
            std::uint64_t raw_records = 0;
            std::uint64_t finalized_records = 0;
            std::uint64_t min_bin_parents = propagation.bins.front().parents, max_bin_parents = 0;
            double max_bin_seconds = 0.0;
            FCIQMCParentWeightWork below, at_or_above;
            const auto add_group = [](FCIQMCParentWeightWork &total,
                                      const FCIQMCParentWeightWork &group) {
                total.visits += group.visits;
                total.absolute_weight += group.absolute_weight;
                total.attempts += group.attempts;
                total.nonzero_spawns += group.nonzero_spawns;
            };
            for (const auto &bin : propagation.bins)
            {
                add_group(below, bin.below_granularity);
                add_group(at_or_above, bin.at_or_above_granularity);
                parents += bin.parents;
                min_bin_parents = std::min(min_bin_parents, bin.parents);
                max_bin_parents = std::max(max_bin_parents, bin.parents);
                attempts += bin.spawn_attempts;
                invalid += bin.invalid_excitations;
                zero_h += bin.zero_hamiltonian;
                initiator += bin.initiator_rejections;
                rounded += bin.rounded_to_zero;
                raw_records += bin.accumulator_records;
                finalized_records += bin.finalized_records;
                max_bin_seconds = std::max(max_bin_seconds, bin.worker_seconds);
            }

            // A deliberately flat, machine-readable line: benchmark scripts can
            // retain the exact raw line while human logs stay compact. Worker wall
            // time includes spawn/death plus the per-bin sort/fold; the sum of
            // individual-bin times is intentionally not reported as a wall time.
            logging(LogLevel::Info, "FCIQMC_PROFILE:",
                    std::format("phase={} steps={} parents={} attempts={} "
                                "invalid={} zero_h={} initiator_rejected={} "
                                "rounded_zero={} records_raw={} records_folded={} "
                                "prefill_s={:.6f} setup_s={:.6f} worker_wall_s={:.6f} "
                                "max_bin_work_s={:.6f} merge_s={:.6f} merge_insert_s={:.6f} "
                                "canonical_reserve_s={:.6f} canonical_frontier_s={:.6f} "
                                "canonical_stream_s={:.6f} canonical_emitted={} "
                                "canonical_pairwise_s={:.6f} "
                                "parent_cleanup_s={:.6f} compress_s={:.6f} "
                                "estimator_s={:.6f} phase_wall_s={:.6f} "
                                "bin_count={} min_bin_parents={} max_bin_parents={}",
                                phase, driver.steps, parents, attempts, invalid,
                                zero_h, initiator, rounded, raw_records,
                                finalized_records, driver.prefill_seconds,
                                propagation.setup_seconds,
                                propagation.worker_wall_seconds,
                                max_bin_seconds, propagation.merge_seconds,
                                propagation.merge_insert_seconds,
                                propagation.canonical_merge_reserve_seconds,
                                propagation.canonical_merge_frontier_seconds,
                                propagation.canonical_merge_stream_seconds,
                                propagation.canonical_merge_emitted,
                                propagation.canonical_pairwise_seconds,
                                propagation.parent_cleanup_seconds,
                                driver.compress_seconds, driver.estimator_seconds, phase_seconds,
                                propagation.bins.size(), min_bin_parents, max_bin_parents));
            const auto log_group = [&](const char *group, const FCIQMCParentWeightWork &work) {
                logging(LogLevel::Info, "FCIQMC_PROFILE:",
                        std::format("phase=parent_weight window={} group={} visits={} "
                                    "abs_weight={:.12e} attempts={} nonzero_spawns={}",
                                    phase, group, work.visits, work.absolute_weight,
                                    work.attempts, work.nonzero_spawns));
            };
            log_group("below_granularity", below);
            log_group("at_or_above_granularity", at_or_above);
        }

        // Adapt the shared fixed-orbital setup to the callbacks the propagator wants.
        //
        // slater_condon_element is the SAME matrix-element routine the
        // deterministic CI uses, so the two paths cannot disagree about the
        // Hamiltonian -- only about how they solve it.
        using DiagonalCache = std::unordered_map<DetKey, double, DetKeyHash>;
        HamiltonianOps make_ops(const FixedOrbitalCISetup &setup,
                                const std::shared_ptr<DiagonalCache> &diag_cache)
        {
            const int n_act = setup.n_act;
            const Eigen::MatrixXd *h = &setup.h_eff;
            const std::vector<double> *g = &setup.ga;

            HamiltonianOps ops;
            ops.off_diagonal = [h, g, n_act](const DetKey &i, const DetKey &j) {
                return CI::slater_condon_element(i.alpha, i.beta, j.alpha, j.beta,
                                                 *h, *g, n_act);
            };
            // MEMOIZED (scope step T4). H_ii is a pure function of the
            // determinant -- `h_eff` and `ga` are built once in
            // `build_fixed_orbital_ci_setup` and never mutated -- but the spawn loop asks
            // for it once per parent per iteration, so the same value is recomputed
            // for the life of the run.
            //
            // MEASURED on the N2/STO-3G gate case: 68 696 226 calls over 1820
            // DISTINCT determinants, a 37 745x reuse factor and a 99.9974 % hit
            // rate on an 85 KB table. The diagonal branch is the expensive one --
            // O(n_act^2) (a p,q double loop plus an h_eff pass) against O(n_act)
            // for a single excitation and O(1) for a double, ~86 % of
            // slater_condon_element's work by operation count, and that function
            // was 53.1 % of self time.
            //
            // The serial prefill relies on a surviving parent's diagonal
            // remaining resident between steps. Any future cache eviction must
            // preserve those entries or arrange to prefill them again before
            // the workers read the cache.
            //
            // `shared_ptr` because HamiltonianOps holds std::function, which must
            // stay copyable; the cache is shared by every copy, which is what we
            // want since they all describe the same Hamiltonian.
            //
            // THREADING NOTE (T2): this map is NOT thread-safe. When the spawn loop
            // is threaded it must be prefilled before the parallel region, or made
            // per-bin -- never a shared mutable map. It cannot change any value
            // either way: a memo is a pure cache of a pure function.
            ops.diagonal = [h, g, n_act, diag_cache](const DetKey &i) {
                if (const auto it = diag_cache->find(i); it != diag_cache->end())
                    return it->second;
                const double v = CI::slater_condon_element(i.alpha, i.beta,
                                                           i.alpha, i.beta,
                                                           *h, *g, n_act);
                diag_cache->emplace(i, v);
                return v;
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
        // The shift energy does not anchor on it, which is why the shift looked
        // healthy and hid the defect -- but it is also the starting population,
        // so a bad choice still cost the shift equilibration time.
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
        const auto &opt = calc._fciqmc;
        using Clock = std::chrono::steady_clock;
        using Seconds = std::chrono::duration<double>;
        const bool had_ao_eri = !calc._eri.empty();
        const auto setup_start = opt.profile ? Clock::now() : Clock::time_point{};
        CIIntegralTimings integral_timings;
        const auto &as = calc._active_space;
        const bool active = as.nactele != 0 || as.nactorb != 0;
        auto setup = build_fixed_orbital_ci_setup(
            calc, shell_pairs, tag, active ? CIOrbitalSpace::Active : CIOrbitalSpace::All,
            opt.profile ? &integral_timings : nullptr);
        if (!setup)
            return std::unexpected(setup.error());
        const double energy_offset = calc._nuclear_repulsion + setup->core_energy;
        if (active)
            logging(LogLevel::Info, tag + " :",
                    std::format("Fixed orbitals: {} frozen core, {} active, core energy {:.12f} Eh",
                                setup->n_core, setup->n_act, setup->core_energy));
        if (opt.profile)
        {
            const double build_seconds = Seconds(Clock::now() - setup_start).count();
            const auto setup_peak_rss = peak_rss_bytes();
            log_hamiltonian_identity(*setup, calc, build_seconds);
            const char *route = calc._mp2.use_ri ? "ri" :
                                (had_ao_eri ? "cached" : "constructed");
            logging(LogLevel::Info, "FCIQMC_PROFILE:",
                    std::format("phase=integrals route={} ao_eri_s={:.6f} mo_eri_s={:.6f} "
                                "setup_peak_rss_bytes={}",
                                route, integral_timings.ao_eri_seconds,
                                integral_timings.mo_eri_seconds, setup_peak_rss));
        }

        if (!(opt.target_walkers > 0.0))
            return std::unexpected(tag + ": fciqmc_walkers must be positive.");
        if (!(opt.timestep > 0.0))
            return std::unexpected(tag + ": fciqmc_timestep must be positive.");
        if (opt.sampling_steps < 4)
            return std::unexpected(tag + ": fciqmc_steps must be at least 4.");

        const auto diag_cache = std::make_shared<DiagonalCache>();
        const auto ops = make_ops(*setup, diag_cache);
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
                            "attempts per {} determinant",
                            opt.equilibration_steps, opt.sampling_steps,
                            opt.spawn_attempts,
                            opt.sample_low_weight_parents ? "selected" : "occupied"));
        if (opt.sample_low_weight_parents)
            logging(LogLevel::Info, tag + " :", "probabilistic spawning from low-weight parents enabled");

        // ── Propagate ──────────────────────────────────────────────────────────
        CanonicalWalkerPopulation pop;
        CanonicalWalkerPopulation previous_parents;
        pop.assign_sorted({{reference, opt.target_walkers}});

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
        int weak_reference_steps = 0;
        const double min_reference_weight = 3.0 * std::max(1.0, opt.walker_granularity);
        double ref_weight_sum = 0.0;
        double ref_weight_min = std::numeric_limits<double>::infinity();
        int ref_weight_n = 0;
        std::vector<double> shift_samples;
        // Per-step numerator sum_j H_0j c_j and denominator c_0, for EVERY
        // sampling step. Their joint series gives the ratio-of-sums error bar.
        std::vector<double> projected_numerators;
        std::vector<double> projected_denominators;
        // Signed weights summed over the last `anchor_window` equilibration
        // steps; the projection reference is chosen from these, not from the
        // single step at which equilibration ends (finding 5).
        // ponytail: hash-map accumulation, O(occupied) per window step. Fine at
        // 1000 steps x 1e5 determinants; merge into the canonical population if
        // the window or population grows.
        const int anchor_window = std::min(1000, opt.equilibration_steps / 4);
        std::unordered_map<DetKey, double, DetKeyHash> anchor_weights;
        // Running sum of SIGNED weight per determinant over the sampling phase.
        //
        // Signed, and summed over steps rather than snapshotted: an instantaneous
        // population is dominated by shot noise (many determinants hold a single
        // walker whose sign flips step to step), while <N_I> converges to the
        // wavefunction. The sign is the whole point -- a magnitude-only average
        // would agree with a broken sampler that got every phase wrong.
        std::unordered_map<DetKey, double, DetKeyHash> signed_population;
        // Accumulating it is an O(n_occupied) map update on EVERY sampling step,
        // and its only consumer is the `<N_I>/<N_0>` dump below, which prints at
        // Verbose. Measured on N2/STO-3G it was a material part of the 9.4 % of
        // runtime spent in this driver loop rather than in the propagator, so it
        // is gated on the same condition the consumer uses -- not on a separate
        // flag, which could drift out of agreement with it.
        const bool want_coefficient_ratios =
            calc._output._verbosity >= Verbosity::Verbose;
        const int total_steps = opt.equilibration_steps + opt.sampling_steps;

        // H2.4: the propagator's per-call scaffolding (64 accumulators + 64
        // parent buckets), built ONCE here instead of ~50,000 times inside
        // propagate_stochastic. Reused across every step; a fresh workspace and
        // a reused one produce the same bytes (SpawnAccumulator/vector reset
        // genuinely, unlike unordered_map::clear).
        SpawnWorkspace spawn_ws;
        FCIQMCDriverWork equilibration_work;
        FCIQMCDriverWork sampling_work;
        FCIQMCPropagationProfile equilibration_profile;
        FCIQMCPropagationProfile sampling_profile;
        std::size_t peak_population_bytes = 0;

        const auto equilibration_start = opt.profile ? Clock::now() : Clock::time_point{};
        auto sampling_start = equilibration_start;
        for (int step = 0; step < total_steps; ++step)
        {
            if (opt.profile && step == opt.equilibration_steps)
            {
                // The macOS CPU sampler waits for this event in a redirected
                // log. Emit it once, including for a zero-equilibration run.
                logging(LogLevel::Info, "FCIQMC_PROFILE:", "phase=sampling_begin");
                std::cout.flush();
                if (step > 0)
                    sampling_start = Clock::now();
            }
            const bool sampling = step >= opt.equilibration_steps;
            FCIQMCDriverWork &step_work = sampling ? sampling_work : equilibration_work;
            FCIQMCPropagationProfile *step_profile = nullptr;
            if (opt.profile)
                step_profile = sampling ? &sampling_profile : &equilibration_profile;
            if (opt.profile)
                ++step_work.steps;
            std::chrono::steady_clock::time_point prefill_start;
            if (opt.profile)
                prefill_start = std::chrono::steady_clock::now();
            spawn_ws.prepare_canonical_parents(pop, previous_parents, ops);
            // Both populations are sorted. Release the old allocation before
            // propagation constructs the next population and its merge buffers.
            previous_parents = {};

            if (opt.profile)
            {
                const auto propagation_start = std::chrono::steady_clock::now();
                step_work.prefill_seconds += std::chrono::duration<double>(
                    propagation_start - prefill_start).count();
            }

            CanonicalWalkerPopulation next_pop;
            propagate_stochastic(pop, setup->n_act, ops, opt.timestep,
                                 ctl.shift, rng, opt.spawn_attempts,
                                 opt.walker_granularity,
                                 opt.initiator_threshold, spawn_ws, next_pop,
                                 step_profile, opt.sample_low_weight_parents);
            if (opt.profile)
                peak_population_bytes = std::max(
                    peak_population_bytes, pop.retained_bytes() + next_pop.retained_bytes());
            previous_parents = std::move(pop);
            pop = std::move(next_pop);

            // H2.8.4-b (fold B): compress + ordered L1 norm in one walk of the
            // new `pop`, instead of `pop.compress(1e-12)` then a second full
            // walk in `ordered_l1_norm(pop)`. The norm is byte-identical to the
            // old two-call form (same survivors, same fixed 64-bin partition,
            // same fixed summation order).
            std::chrono::steady_clock::time_point compress_start;
            if (opt.profile)
                compress_start = std::chrono::steady_clock::now();
            // Occupation rounding of the surviving population (finding 1): one
            // raw draw per step roots the per-determinant streams, independent of
            // the spawn streams propagate_stochastic derived from its own draw.
            // Skipped at zero granularity, which keeps continuous weights and the
            // pre-fix trajectory.
            double n = 0.0;
            if (opt.walker_granularity > 0.0)
            {
                const RandomSource rounding_root(rng.raw64());
                n = pop.compress_with_l1_norm(1e-12, opt.walker_granularity,
                                              &rounding_root).l1_norm;
            }
            else
                n = pop.compress_with_l1_norm(1e-12).l1_norm;
            if (opt.profile)
                step_work.compress_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - compress_start).count();
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

            if (step < opt.equilibration_steps
                && step >= opt.equilibration_steps - anchor_window)
                for (const auto &[det, w] : pop)
                    anchor_weights[det] += w;

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
            //
            // CHOSEN FROM AVERAGED WEIGHTS, WITH A MARGIN (code review finding 5,
            // 2026-11-17). The first version took the largest |w| on this single
            // step and switched on any margin. At the 5-13 walkers per
            // determinant of Cr2 CAS(12,18) that choice was sampling noise: the
            // replacement determinant differed between populations at one seed,
            // and runs that switched kept fewer usable projections. The weights
            // are now signed sums over the last `anchor_window` equilibration
            // steps (|sum| also penalizes a determinant whose sign oscillates),
            // and the anchor moves only when the seeded reference is overtaken
            // by more than 2x -- the same margin as the drift warning.
            if (step == opt.equilibration_steps)
            {
                const DetKey seeded = reference;
                const auto weight_of = [&](const DetKey &det) {
                    if (anchor_window == 0)
                        return std::abs(pop.weight_at(det));
                    const auto it = anchor_weights.find(det);
                    return it == anchor_weights.end()
                               ? 0.0
                               : std::abs(it->second) / anchor_window;
                };
                double best_w = weight_of(reference);
                DetKey best = reference;
                // Ties break on the bitstrings: anchor_weights is a hash map,
                // so without a total order the anchor would depend on iteration
                // order and the run would stop being reproducible at fixed seed.
                const auto consider = [&](const DetKey &det, double aw) {
                    if (aw > best_w
                        || (aw == best_w
                            && (det.alpha < best.alpha
                                || (det.alpha == best.alpha && det.beta < best.beta))))
                    {
                        best_w = aw;
                        best = det;
                    }
                };
                if (anchor_window == 0)
                    for (const auto &[det, w] : pop)
                        consider(det, std::abs(w));
                else
                    for (const auto &[det, w] : anchor_weights)
                        consider(det, std::abs(w) / anchor_window);
                const double seeded_w = weight_of(seeded);
                anchor_weights = {};
                const bool overtaken = best_w > 2.0 * seeded_w;
                // Warn whenever the seeded reference has been overtaken by a
                // margin -- a large drift says the state is degenerate or
                // near-degenerate, which the user should know regardless.
                if (overtaken && seeded_w > 0.0)
                    logging(LogLevel::Warning, tag + " :",
                            std::format("REFERENCE DRIFT: determinant "
                                        "{:#018x}/{:#018x} carries {:.2f} walkers "
                                        "(averaged over {} steps) against the seeded "
                                        "reference's {:.2f} ({:.1f}x). This is the "
                                        "signature of a degenerate or near-degenerate "
                                        "ground state, in which the population drifts "
                                        "freely between partners. Re-anchoring the "
                                        "projection; the shift energy is unaffected.",
                                        static_cast<unsigned long long>(best.alpha),
                                        static_cast<unsigned long long>(best.beta),
                                        best_w, std::max(anchor_window, 1), seeded_w,
                                        best_w / seeded_w));

                if (overtaken)
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
                std::chrono::steady_clock::time_point estimator_start;
                if (opt.profile)
                    estimator_start = std::chrono::steady_clock::now();
                shift_samples.push_back(ctl.shift);
                if (want_coefficient_ratios)
                    for (const auto &[det, w] : pop)
                        signed_population[det] += w;
                // RATIO OF SUMS, not mean of ratios. Accumulate the numerator
                // and denominator separately and divide once at the end:
                //
                //   E = H_00 + (sum_t sum_j H_0j c_j) / (sum_t c_0)
                //
                // Averaging the per-step ratios instead is the E[A/B] vs
                // E[A]/E[B] error this project has now hit three times (F2.5's
                // acceptance-rate correction, F3.4's documented bias, and here).
                // The per-step ratio distribution is heavy-tailed -- rare small
                // c_0 produce huge 1/c_0 spikes -- so its mean is set by
                // outliers. Measured on N2/STO-3G: the mean of ratios was
                // -99.19 +/- 6.63 against an exact -107.65.
                //
                // EVERY STEP ENTERS BOTH SUMS (code review finding 4,
                // 2026-11-17). A |c_0| >= 3 walker threshold once protected the
                // per-step division. The ratio of sums divides once, so the
                // threshold protected nothing and instead conditioned the average
                // on steps where the reference was well occupied -- a selection
                // correlated with the numerator. On Cr2 CAS(12,18) it discarded
                // up to 56 % of steps. The threshold survives only as the
                // usability diagnostic below: a reference that is weak on most
                // steps still suppresses the projected energy.
                const auto pe = projected_energy(pop, reference, setup->n_act, ops, 0.0);
                proj_numerator_sum += pe.numerator;
                proj_denominator_sum += pe.reference_weight;
                ++proj_n;
                projected_numerators.push_back(pe.numerator);
                projected_denominators.push_back(pe.reference_weight);
                if (std::abs(pe.reference_weight) < min_reference_weight)
                    ++weak_reference_steps;
                ref_weight_sum += std::abs(pe.reference_weight);
                ref_weight_min = std::min(ref_weight_min, std::abs(pe.reference_weight));
                ++ref_weight_n;
                if (opt.profile)
                    step_work.estimator_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - estimator_start).count();
            }
        }

        if (opt.profile)
        {
            const auto end = Clock::now();
            log_profile("equilibration", equilibration_work, equilibration_profile,
                        Seconds(sampling_start - equilibration_start).count());
            log_profile("sampling", sampling_work, sampling_profile,
                        Seconds(end - sampling_start).count());
            // Both sets of bins retain capacity across steps. Their final
            // capacities are high-water marks, even though their sizes reset.
            std::size_t accumulator_bytes = 0, parent_bytes = 0;
            for (const auto &bin : spawn_ws.bins)
                accumulator_bytes += bin.retained_bytes();
            for (const auto &parents : spawn_ws.parents)
                parent_bytes += parents.capacity() * sizeof(parents.front());
            logging(LogLevel::Info, "FCIQMC_PROFILE:",
                    std::format("phase=memory population_pair_peak_bytes={} "
                                "accumulator_bytes={} parent_bin_bytes={} merge_tagged_bound_bytes={} "
                                "merge_output_peak_bytes={} "
                                "diagonal_entries={} sample_series_bytes={} coefficient_ratio_entries={}",
                                peak_population_bytes, accumulator_bytes, parent_bytes,
                                std::max(equilibration_profile.peak_merge_tagged_bytes,
                                         sampling_profile.peak_merge_tagged_bytes),
                                std::max(equilibration_profile.peak_merge_output_bytes,
                                         sampling_profile.peak_merge_output_bytes),
                                diag_cache->size(), sizeof(double) *
                                    (shift_samples.capacity() + projected_numerators.capacity()
                                     + projected_denominators.capacity()),
                                signed_population.size()));
            // Sample-series and endpoint check, outside the step timings. The
            // canonical population already iterates in determinant order.
            RunFingerprint result;
            result.word(shift_samples.size());
            for (double v : shift_samples)
                result.real(v);
            result.word(projected_numerators.size());
            for (std::size_t t = 0; t < projected_numerators.size(); ++t)
            {
                result.real(projected_numerators[t]);
                result.real(projected_denominators[t]);
            }
            result.real(proj_numerator_sum);
            result.real(proj_denominator_sum);
            result.word(reference.alpha);
            result.word(reference.beta);
            result.word(pop.size());
            for (const auto &[det, weight] : pop)
            {
                result.word(det.alpha);
                result.word(det.beta);
                result.real(weight);
            }
            logging(LogLevel::Info, "FCIQMC_PROFILE:",
                    std::format("phase=result id={:016x}", result.value));
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

        const double e_shift = mean_of(shift_samples) + energy_offset;
        const auto shift_blocking = blocking_analysis(shift_samples);
        const double shift_err = shift_blocking.error;

        // An unconverged blocking curve means the correlation time is comparable
        // to the sampling window and the error bar is too small (finding 3). The
        // flag follows the number on the same line, so parsers that read
        // "<energy> +/- <error>" are unaffected.
        const auto blocking_flag = [](const BlockingResult &b) {
            return b.converged ? "" : "  [ERROR BAR UNCONVERGED]";
        };
        logging(LogLevel::Info, tag + " :",
                std::format("Shift energy     {:.10f} +/- {:.2e}  ({} samples){}",
                            e_shift, shift_err, shift_samples.size(),
                            blocking_flag(shift_blocking)));
        if (!shift_blocking.converged)
            logging(LogLevel::Warning, tag + " :",
                    "the shift error bar did not reach a blocking plateau: the "
                    "correlation time is comparable to the sampling window, so the "
                    "error is underestimated. Raise fciqmc_steps.");

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

        // Report the projected energy only if the reference was well occupied on
        // most steps. This is a usability diagnostic only: every step is in the
        // sums regardless (finding 4). With the reference weak on most steps the
        // ratio of sums carries an O(1/c_0) bias the error bar does not show.
        const int strong_reference_steps = proj_n - weak_reference_steps;
        const bool projected_usable =
            proj_denominator_sum != 0.0
            && strong_reference_steps >= static_cast<int>(shift_samples.size() / 4);
        if (proj_n > 0 && !projected_usable)
            logging(LogLevel::Warning, tag + " :",
                    std::format("projected energy suppressed: the reference held "
                                "enough walkers on only {} of {} steps. Raise "
                                "fciqmc_walkers.",
                                strong_reference_steps, shift_samples.size()));

        if (projected_usable)
        {
            const double ratio = proj_numerator_sum / proj_denominator_sum;
            const double e_proj = ops.diagonal(reference) + ratio + energy_offset;
            // ERROR BAR OF THE RATIO OF SUMS (finding 2, 2026-11-17). The old bar
            // blocked the per-step ratios num_t / c0_t, a different random
            // variable; on five fresh Cr2 CAS(12,10) seeds it understated the
            // spread across seeds about 4x. The ratio's first-order (delta-method)
            // error is the standard error of the mean of
            //   y_t = (num_t - R c0_t) / mean(c0),
            // whose mean is exactly zero at R = sum num / sum c0, so blocking y_t
            // propagates the numerator-denominator covariance with every
            // autocorrelation the blocking resolves.
            const double mean_c0 = proj_denominator_sum / proj_n;
            std::vector<double> linearized(projected_numerators.size());
            for (std::size_t t = 0; t < linearized.size(); ++t)
                linearized[t] = (projected_numerators[t]
                                 - ratio * projected_denominators[t]) / mean_c0;
            const auto proj_blocking = blocking_analysis(linearized);
            const double proj_err = proj_blocking.error;
            logging(LogLevel::Info, tag + " :",
                    std::format("Projected energy {:.10f} +/- {:.2e}  ({} samples, "
                                "{} with |c_0| < {:g}){}",
                                e_proj, proj_err, proj_n, weak_reference_steps,
                                min_reference_weight, blocking_flag(proj_blocking)));
            if (!proj_blocking.converged)
                logging(LogLevel::Warning, tag + " :",
                        "the projected error bar did not reach a blocking plateau: "
                        "the error is underestimated. Raise fciqmc_steps.");
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
        if (want_coefficient_ratios && !signed_population.empty())
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
