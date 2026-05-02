#include "post_hf/casscf/casscf.h"

#include "io/logging.h"
#include "base/tables.h"
#include "post_hf/casscf.h"
#include "post_hf/casscf/aug-hessian-orbital.h"
#include "post_hf/casscf/casscf_driver_internal.h"
#include "post_hf/casscf/casscf_utils.h"
#include "post_hf/casscf/ci.h"
#include "post_hf/casscf/orbital.h"
#include "post_hf/casscf/rdm.h"
#include "post_hf/casscf/response.h"
#include "post_hf/casscf/strings.h"
#include "post_hf/integrals.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace
{

    using HartreeFock::Correlation::CASSCF::CIDeterminantSpace;
    using HartreeFock::Correlation::CASSCF::CISigmaApplier;
    using HartreeFock::Correlation::CASSCF::CISolveResult;
    using HartreeFock::Correlation::CASSCF::CandidateSelection;
    using HartreeFock::Correlation::CASSCF::CandidateStep;
    using HartreeFock::Correlation::CASSCF::hess_diag;
    using HartreeFock::Correlation::CASSCF::build_weighted_root_quadratic_model_prediction;
    using HartreeFock::Correlation::CASSCF::append_candidate_step;
    using HartreeFock::Correlation::CASSCF::append_root_candidate_steps;
    using HartreeFock::Correlation::CASSCF::MacroDiagnostics;
    using HartreeFock::Correlation::CASSCF::McscfState;
    using HartreeFock::Correlation::CASSCF::quadratic_model_delta;
    using HartreeFock::Correlation::CASSCF::reorder_mo_coefficients;
    using HartreeFock::Correlation::CASSCF::ResponseMode;
    using HartreeFock::Correlation::CASSCF::ResponseRHSMode;
    using HartreeFock::Correlation::CASSCF::RootReference;
    using HartreeFock::Correlation::CASSCF::RotPair;
    using HartreeFock::Correlation::CASSCF::RootResolvedCoupledStepSet;
    using HartreeFock::Correlation::CASSCF::RootResolvedGradientScreen;
    using HartreeFock::Correlation::CASSCF::RootResolvedOrbitalStepSet;
    using HartreeFock::Correlation::CASSCF::SACoupledStepSolveResult;
    using HartreeFock::Correlation::CASSCF::select_active_orbitals;
    using HartreeFock::Correlation::CASSCF::StateSpecificData;
    using HartreeFock::Correlation::CASSCF::StateAveragedCoupledRoot;
    using HartreeFock::Correlation::CASSCF::WeightedQuadraticModelPrediction;
    using HartreeFock::Correlation::CASSCF::WeightedRootProbeSignal;
    using HartreeFock::Correlation::CASSCFInternal::ActiveIntegralCache;
    using HartreeFock::Correlation::CASSCFInternal::CIResponseResult;
    using HartreeFock::Correlation::CASSCFInternal::CIString;
    using HartreeFock::Correlation::CASSCFInternal::compute_root_overlap;
    using HartreeFock::Correlation::CASSCFInternal::diagonalize_natural_orbitals;
    using HartreeFock::Correlation::CASSCFInternal::kMaxPackedSpatialOrbitals;
    using HartreeFock::Correlation::CASSCFInternal::kMaxSeparateSpinOrbitals;
    using HartreeFock::Correlation::CASSCFInternal::match_roots_by_max_overlap;
    using HartreeFock::Correlation::CASSCFInternal::NaturalOrbitalData;
    using HartreeFock::Correlation::CASSCFInternal::RASParams;
    using HartreeFock::Correlation::CASSCFInternal::SymmetryContext;

    template <typename EvaluateFn>
    CandidateSelection select_best_candidate_step(
        const std::vector<CandidateStep> &candidates,
        const Eigen::MatrixXd &coefficients,
        const Eigen::MatrixXd &overlap,
        RootReference *root_reference,
        const McscfState &current_state,
        const std::vector<RotPair> &opt_pairs,
        double tol_energy,
        EvaluateFn &&evaluate,
        bool accept_uphill = false,
        double uphill_max_eh = 5e-3)
    {
        CandidateSelection selection;
        selection.state = current_state;
        selection.coefficients = coefficients;
        selection.step = Eigen::MatrixXd::Zero(coefficients.rows(), coefficients.cols());
        selection.energy = current_state.E_cas;
        selection.sa_gnorm = current_state.gnorm;
        constexpr double merit_weight = 0.10;
        selection.merit = selection.energy + merit_weight * selection.sa_gnorm * selection.sa_gnorm;

        // Candidate generation produces a menu of orbital directions from
        // different heuristics (AH, Newton-like, fallback probes, root-first
        // variants, ...).  This selector actually tests them by rotating the
        // orbitals, re-evaluating the MCSCF state, and accepting only moves
        // that improve energy/merit without clearly destabilizing the SA
        // gradient.
        for (const auto &candidate : candidates)
        {
            for (double scale : CASSCF_MACRO_STEP_SCALES)
            {
                Eigen::MatrixXd trial_step = scale * candidate.step;
                if (trial_step.cwiseAbs().maxCoeff() < 1e-12)
                    continue;

                const Eigen::MatrixXd rotated_coefficients =
                    HartreeFock::Correlation::CASSCF::apply_orbital_rotation(
                        coefficients, trial_step, overlap);
                auto trial_res = evaluate(rotated_coefficients, root_reference, false);
                if (!trial_res)
                    continue;

                const auto &trial = *trial_res;
                const double trial_merit =
                    trial.E_cas + merit_weight * trial.gnorm * trial.gnorm;
                const bool merit_improved = trial_merit < selection.merit - 1e-10;
                const bool sa_gradient_reduced =
                    trial.gnorm < selection.sa_gnorm - 1e-12;
                const double sa_worsen_window =
                    std::max(0.05 * std::max(selection.sa_gnorm, 1e-8), 1e-6);
                const bool energy_improved = trial.E_cas < selection.energy - 1e-10;
                const bool energy_improved_without_hurting_gradient =
                    energy_improved &&
                    trial.gnorm <= selection.sa_gnorm + sa_worsen_window;
                const double flat_energy_window = std::max(1000.0 * tol_energy, 1e-6);
                const bool stationary_but_better_grad =
                    std::abs(trial.E_cas - selection.energy) <= flat_energy_window &&
                    sa_gradient_reduced;
                if (!energy_improved_without_hurting_gradient &&
                    !merit_improved &&
                    !stationary_but_better_grad)
                    continue;

                selection.accepted = true;
                selection.energy = trial.E_cas;
                selection.sa_gnorm = trial.gnorm;
                selection.merit = trial_merit;
                selection.state = trial;
                selection.coefficients = rotated_coefficients;
                selection.step = std::move(trial_step);
                selection.label =
                    (std::abs(scale - 1.0) < 1e-12)
                        ? candidate.label
                        : std::format("{}@{:.5f}", candidate.label, scale);
                selection.weighted_root_gnorm = trial.weighted_root_gnorm;
                selection.max_root_gnorm = trial.max_root_gnorm;
                const WeightedQuadraticModelPrediction prediction =
                    build_weighted_root_quadratic_model_prediction(
                        current_state.roots, current_state.F_I_mo, opt_pairs, selection.step);
                selection.predicted_delta = prediction.weighted_delta;
                selection.max_root_predicted_delta_deviation = prediction.max_root_deviation;
            }
        }

        // Pass 2: per-root-gradient-driven uphill acceptance (opt-in via
        // mcscf_accept_uphill). Only fires when (a) the user opted in AND
        // (b) Pass 1 above found no strict-monotone improvement. This keeps
        // every existing case bit-identical at default settings — Pass 2 is
        // skipped entirely when selection.accepted is already true.
        //
        // Pass 2 targets the "false SA stationary point" failure mode: the
        // SA-weighted gradient g_SA = Σ w_I g_I has vanished by cancellation
        // even though individual per-root gradients g_I remain large. The SA
        // energy is at a local minimum of E_SA(κ), but each individual root's
        // gradient still points into a nearby deeper basin. To escape, we
        // accept a step that *increases* E_SA (bounded by uphill_max_eh) when
        // it *reduces* the worst per-root orbital gradient. This is the signal
        // PySCF's mc.newton() implicitly follows — its convergence is gated on
        // the per-MO orbital gradient, so it keeps stepping past SA-stationary
        // basins until per-root gradients also vanish.
        if (accept_uphill && !selection.accepted)
        {
            double best_max_root_gnorm = current_state.max_root_gnorm;
            for (const auto &candidate : candidates)
            {
                for (double scale : CASSCF_MACRO_STEP_SCALES)
                {
                    Eigen::MatrixXd trial_step = scale * candidate.step;
                    if (trial_step.cwiseAbs().maxCoeff() < 1e-12)
                        continue;

                    const Eigen::MatrixXd rotated_coefficients =
                        HartreeFock::Correlation::CASSCF::apply_orbital_rotation(
                            coefficients, trial_step, overlap);
                    auto trial_res = evaluate(rotated_coefficients, root_reference, false);
                    if (!trial_res)
                        continue;
                    const auto &trial = *trial_res;

                    const double actual_dE = trial.E_cas - current_state.E_cas;
                    // Cap the worst uphill move we tolerate per macro step.
                    if (actual_dE > uphill_max_eh)
                        continue;
                    // The trial must reduce the worst per-root orbital
                    // gradient — otherwise it has not made progress toward a
                    // true SA-stationary point where every root vanishes.
                    //
                    // Use a modest relative drop (0.05%) plus a tiny absolute
                    // floor.  The old fixed 0.99 factor demanded a full 1%
                    // reduction; at false SA-stationary points (|g_SA|≈0) the
                    // available trust-region/probe steps often improve the
                    // worst root by only ~0.3–0.7%, so every candidate was
                    // rejected and the macro loop stalled.
                    const double rel_floor =
                        best_max_root_gnorm * (1.0 - 5.0e-4);
                    const double abs_floor =
                        best_max_root_gnorm -
                        std::max(1.0e-8, 1.0e-6 * best_max_root_gnorm);
                    const double max_root_threshold =
                        std::max(rel_floor, abs_floor);
                    if (!(trial.max_root_gnorm < max_root_threshold))
                        continue;
                    // No cap on the new SA gradient: at a false-stationary
                    // point, |g_SA| ≈ 0 by construction; any meaningful
                    // basin-escape probe will increase it (PySCF traces show
                    // |g_SA| jumps from 1e-3 to 1e-1 across an escape step).
                    // The actual_dE cap and the per-root reduction guard are
                    // sufficient acceptance criteria.

                    best_max_root_gnorm = trial.max_root_gnorm;
                    const WeightedQuadraticModelPrediction prediction =
                        build_weighted_root_quadratic_model_prediction(
                            current_state.roots, current_state.F_I_mo, opt_pairs, trial_step);
                    selection.accepted = true;
                    selection.energy = trial.E_cas;
                    selection.sa_gnorm = trial.gnorm;
                    selection.merit = trial.E_cas + merit_weight * trial.gnorm * trial.gnorm;
                    selection.state = trial;
                    selection.coefficients = rotated_coefficients;
                    selection.step = trial_step;
                    selection.label =
                        (std::abs(scale - 1.0) < 1e-12)
                            ? candidate.label + "[uphill]"
                            : std::format("{}@{:.5f}[uphill]", candidate.label, scale);
                    selection.weighted_root_gnorm = trial.weighted_root_gnorm;
                    selection.max_root_gnorm = trial.max_root_gnorm;
                    selection.predicted_delta = prediction.weighted_delta;
                    selection.max_root_predicted_delta_deviation = prediction.max_root_deviation;
                }
            }
        }

        return selection;
    }

} // namespace

namespace HartreeFock::Correlation::CASSCF
{

    std::expected<void, std::string> run_mcscf_loop(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const std::string &tag,
        const RASParams &ras)
    {
        using HartreeFock::LogLevel;
        using HartreeFock::Logger::logging;

        if (!calc._info._is_converged)
            return std::unexpected(tag + ": requires a converged RHF reference.");
        if (calc._scf._scf != HartreeFock::SCFType::RHF)
            return std::unexpected(tag + ": only RHF reference supported.");

        const auto &as = calc._active_space;
        if (as.nactele <= 0)
            return std::unexpected(tag + ": nactele must be > 0.");
        if (as.nactorb <= 0)
            return std::unexpected(tag + ": nactorb must be > 0.");
        if (as.nactele > 2 * as.nactorb)
            return std::unexpected(tag + ": nactele > 2*nactorb is impossible.");

        const int nbasis = static_cast<int>(calc._shells.nbasis());
        const int n_total_elec =
            static_cast<int>(calc._molecule.atomic_numbers.cast<int>().sum()) - calc._molecule.charge;
        if ((n_total_elec - as.nactele) % 2 != 0)
            return std::unexpected(tag + ": (n_elec - nactele) must be even for RHF-based CASSCF.");

        const int n_core = (n_total_elec - as.nactele) / 2;
        const int n_act = as.nactorb;
        const int n_virt = nbasis - n_core - n_act;
        if (n_act > kMaxSeparateSpinOrbitals)
            return std::unexpected(
                std::format("{}: nactorb={} exceeds the 63-orbital limit of the CI bitstring encoding.",
                            tag, n_act));
        if (n_act > kMaxPackedSpatialOrbitals)
            return std::unexpected(
                std::format("{}: nactorb={} exceeds the packed alpha/beta determinant limit ({}).",
                            tag, n_act, kMaxPackedSpatialOrbitals));
        if (n_core < 0)
            return std::unexpected(tag + ": nactele > total electrons.");
        if (n_virt < 0)
            return std::unexpected(tag + ": n_core + nactorb > nbasis.");
        if (ras.active && ras.nras1 + ras.nras2 + ras.nras3 != n_act)
            return std::unexpected(tag + ": nras1 + nras2 + nras3 must equal nactorb.");

        const int multiplicity = static_cast<int>(calc._molecule.multiplicity);
        const int n_alpha_act = (as.nactele + (multiplicity - 1)) / 2;
        const int n_beta_act = as.nactele - n_alpha_act;
        if (n_alpha_act < 0 || n_beta_act < 0 || n_alpha_act > n_act || n_beta_act > n_act)
            return std::unexpected(tag + ": invalid active-space electron count.");

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
        const long long ci_dim_est = nchoose(n_act, n_alpha_act) * nchoose(n_act, n_beta_act);
        if (ci_dim_est > static_cast<long long>(as.ci_max_dim))
            return std::unexpected(std::format("{}: CI dim ({}) exceeds ci_max_dim ({}).",
                                               tag, ci_dim_est, as.ci_max_dim));

        const int nroots = as.nroots;
        Eigen::VectorXd weights(nroots);
        if (static_cast<int>(as.weights.size()) == nroots)
            for (int k = 0; k < nroots; ++k)
                weights(k) = as.weights[k];
        else
            weights.setConstant(1.0 / nroots);
        // Normalize even user-provided weights so all downstream root averaging
        // can assume a convex combination.
        weights /= weights.sum();

        Eigen::MatrixXd C = (calc._cas_mo_coefficients.rows() == nbasis &&
                             calc._cas_mo_coefficients.cols() == nbasis)
                                ? calc._cas_mo_coefficients
                                : calc._info._scf.alpha.mo_coefficients;
        if (C.rows() != nbasis || C.cols() != nbasis)
            return std::unexpected(tag + ": MO coefficient matrix has wrong size.");

        const bool have_sym = !calc._sao_irrep_names.empty() && static_cast<int>(calc._sao_irrep_names.size()) <= 8;
        const bool point_group_is_abelian_for_labels =
            point_group_has_only_1d_irreps(calc._molecule._point_group);
        std::optional<SymmetryContext> sym_ctx;
        std::vector<int> all_mo_irr;
        if (have_sym && point_group_is_abelian_for_labels && !calc._info._scf.alpha.mo_symmetry.empty())
        {
            // Only Abelian point groups are used for CI screening here because
            // the current machinery assumes one-dimensional irrep products.
            // Non-Abelian labels remain useful for reporting, but not for the
            // determinant-space selection logic below.
            sym_ctx = build_symmetry_context(calc);
            if (!sym_ctx)
                return std::unexpected(tag + ": failed to build an Abelian irrep product table for CI screening.");

            all_mo_irr = map_mo_irreps(calc._info._scf.alpha.mo_symmetry, sym_ctx->names);
            if (std::find(all_mo_irr.begin(), all_mo_irr.end(), -1) != all_mo_irr.end())
                return std::unexpected(tag + ": encountered an MO irrep label missing from the Abelian product table.");
        }

        std::vector<int> irr_act;
        const bool have_symmetry_selection =
            !as.core_irrep_counts.empty() ||
            !as.active_irrep_counts.empty() ||
            !as.mo_permutation.empty();
        const Eigen::VectorXd &mo_energies = calc._info._scf.alpha.mo_energies;
        const std::vector<std::string> &mo_symmetry = calc._info._scf.alpha.mo_symmetry;
        const bool have_explicit_casscf_guess = calc._cas_mo_coefficients.rows() == nbasis &&
                                                calc._cas_mo_coefficients.cols() == nbasis;
        const bool allow_automatic_symmetry_selection =
            !have_explicit_casscf_guess &&
            (!mo_symmetry.empty() || have_symmetry_selection);
        if (allow_automatic_symmetry_selection)
        {
            auto selection = select_active_orbitals(
                mo_energies,
                mo_symmetry,
                n_core,
                n_act,
                as.core_irrep_counts,
                as.active_irrep_counts,
                as.mo_permutation);
            if (!selection)
                return std::unexpected(std::format("{}: {}", tag, selection.error()));

            bool permutation_changed = false;
            for (int i = 0; i < static_cast<int>(selection->permutation.size()); ++i)
                if (selection->permutation[static_cast<std::size_t>(i)] != i)
                {
                    permutation_changed = true;
                    break;
                }

            auto reordered = reorder_mo_coefficients(C, selection->permutation);
            if (!reordered)
                return std::unexpected(reordered.error());
            C = std::move(*reordered);
            if (!all_mo_irr.empty())
            {
                std::vector<int> permuted_irr(all_mo_irr.size());
                for (int i = 0; i < static_cast<int>(selection->permutation.size()); ++i)
                    permuted_irr[static_cast<std::size_t>(i)] = all_mo_irr[static_cast<std::size_t>(selection->permutation[static_cast<std::size_t>(i)])];
                all_mo_irr = std::move(permuted_irr);
            }
            if (selection->used_symmetry && permutation_changed)
                logging(LogLevel::Info, tag + " :",
                        "Applied automatic symmetry-aware MO permutation to make the selected active block contiguous.");
        }

        if (!all_mo_irr.empty())
        {
            irr_act.resize(n_act);
            for (int t = 0; t < n_act; ++t)
                irr_act[t] = (n_core + t < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[n_core + t] : -1;
        }

        const bool use_sym = sym_ctx.has_value() && !irr_act.empty();
        const auto target_irr_opt = use_sym
                                        ? resolve_target_irrep(as.target_irrep, *sym_ctx)
                                        : std::optional<int>(0);
        if (!target_irr_opt)
            return std::unexpected(std::format("{}: target_irrep '{}' is not present in the Abelian symmetry metadata.",
                                               tag, as.target_irrep));
        const int target_irr = *target_irr_opt;

        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calc, shell_pairs, eri_local, tag + " :");

        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(n_act, n_alpha_act, n_beta_act, a_strs, b_strs);

        const unsigned int nmicro = std::max(1u, as.mcscf_micro_per_macro);
        const ResponseMode configured_response_mode = ResponseMode::CoupledSecondOrderTarget;
        const ResponseRHSMode configured_rhs_mode =
            as.mcscf_debug_commutator_rhs
                ? ResponseRHSMode::CommutatorOnlyApproximate
                : ResponseRHSMode::ExactActiveSpaceOrbitalDerivative;
        const bool use_numeric_newton_debug = as.mcscf_debug_numeric_newton;
        const int numeric_newton_pair_limit = 64;
        const int ci_dense_threshold = 500;
        const double max_rot = (as.mcscf_max_rot > 0.0) ? as.mcscf_max_rot : 0.20;
        const double trust_radius_frob = 4.0 * max_rot;

        logging(LogLevel::Info, tag + " :",
                std::format("Active space: ({:d}e, {:d}o)  n_core={:d}  n_virt={:d}  CI dim ≤ {:d}",
                            as.nactele, n_act, n_core, n_virt, ci_dim_est));
        logging(LogLevel::Info, tag + " :",
                std::format("Algorithm: root-resolved {} with fallback candidate screening  nmicro={:d}",
                            response_mode_name(configured_response_mode), nmicro));
        logging(LogLevel::Info, tag + " :",
                std::format("CI response RHS: {}",
                            response_rhs_mode_name(configured_rhs_mode)));
        logging(LogLevel::Info, tag + " :",
                std::format("Orbital trust region: max_rot={:.3f}  Frobenius cap={:.3f}",
                            max_rot, trust_radius_frob));
        if (configured_rhs_mode == ResponseRHSMode::CommutatorOnlyApproximate)
            logging(LogLevel::Warning, tag + " :",
                    "Using debug-only approximate commutator RHS instead of the default exact orbital-derivative response.");
        if (have_sym && !point_group_is_abelian_for_labels)
            logging(LogLevel::Warning, tag + " :",
                    std::format("Disabling CI symmetry screening for {} because MO labels come from an Abelian subgroup.",
                                calc._molecule._point_group));
        if (use_sym)
            logging(LogLevel::Info, tag + " :",
                    std::format("Target irrep: {}",
                                as.target_irrep.empty() ? sym_ctx->names[target_irr] : as.target_irrep));
        if (nroots > 1)
            logging(LogLevel::Info, tag + " :",
                    std::format("State-averaged over {:d} roots", nroots));
        if (use_numeric_newton_debug)
            logging(LogLevel::Info, tag + " :",
                    "Numeric Newton debug fallback is enabled for small pair spaces.");
        HartreeFock::Logger::blank();
        HartreeFock::Logger::casscf_header();

        auto evaluate =
            [&](const Eigen::MatrixXd &C_trial,
                const RootReference *root_ref = nullptr,
                bool log_root_tracking = false) -> std::expected<McscfState, std::string>
        {
            // This is the single source of truth for "what does the current MO
            // basis imply?". Every candidate orbital step and every final accepted
            // macroiteration comes back through this full reevaluation path.
            McscfState st;
            st.F_I_mo = build_inactive_fock_mo(C_trial, calc._hcore, eri, n_core, nbasis);
            st.h_eff = st.F_I_mo.block(n_core, n_core, n_act, n_act);
            const Eigen::MatrixXd C_act = C_trial.middleCols(n_core, n_act);
            st.ga = HartreeFock::Correlation::transform_eri_internal(eri, nbasis, C_act);
            st.active_integrals = build_active_integral_cache(eri, C_trial, n_core, n_act, nbasis);

            st.ci_space = build_ci_space(
                a_strs, b_strs, ras, st.h_eff, st.ga, n_act,
                irr_act, use_sym ? &*sym_ctx : nullptr, target_irr, ci_dense_threshold);
            if (st.ci_space.dets.empty())
                return std::unexpected(tag + ": no CI determinants of target symmetry.");

            CISolveResult ci_result = solve_ci(
                st.ci_space, a_strs, b_strs, st.h_eff, st.ga, n_act,
                std::min(nroots, static_cast<int>(st.ci_space.dets.size())));
            if (ci_result.energies.size() < nroots)
                return std::unexpected(
                    std::format("{}: CI returned {:d} roots (wanted {:d}).",
                                tag, static_cast<int>(ci_result.energies.size()), nroots));

            reorder_ci_roots(ci_result.energies, ci_result.vectors, root_ref, tag, log_root_tracking);
            st.dets = st.ci_space.dets;
            st.H_CI_diag = std::move(ci_result.diagonal);
            st.ci_used_direct_sigma = ci_result.used_direct_sigma;
            st.ci_energies = std::move(ci_result.energies);
            st.ci_vecs = std::move(ci_result.vectors);

            const int nr_used = std::min(nroots, static_cast<int>(st.ci_vecs.cols()));
            st.roots.assign(static_cast<std::size_t>(nr_used), StateSpecificData{});
            st.F_A_mo = Eigen::MatrixXd::Zero(nbasis, nbasis);
            st.gamma = Eigen::MatrixXd::Zero(n_act, n_act);
            st.Gamma_vec.clear();
            st.g_orb = Eigen::MatrixXd::Zero(nbasis, nbasis);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (nr_used > 1)
#endif
            for (int r = 0; r < nr_used; ++r)
            {
                // Build each root from its own CI vector first, then reconstruct
                // the state-averaged objects as weighted sums of those root records.
                StateSpecificData root;
                root.ci_energy = st.ci_energies(r);
                root.weight = weights(r);
                root.ci_vector = st.ci_vecs.col(r);

                const Eigen::MatrixXd ci_vec = as_single_column_matrix(root.ci_vector);
                root.gamma = compute_1rdm(
                    ci_vec, single_weight(1.0), a_strs, b_strs, st.dets, n_act);
                root.Gamma_vec = compute_2rdm(
                    ci_vec, single_weight(1.0), a_strs, b_strs, st.dets, n_act);
                root.F_A_mo = build_active_fock_mo(C_trial, root.gamma, eri, n_core, n_act, nbasis);
                root.Q = compute_Q_matrix(st.active_integrals, root.Gamma_vec);
                root.g_orb = compute_orbital_gradient(
                    st.F_I_mo, root.F_A_mo, root.Q, root.gamma,
                    n_core, n_act, n_virt, all_mo_irr, use_sym);
                st.roots[static_cast<std::size_t>(r)] = std::move(root);
            }

            for (const auto &root : st.roots)
            {
                st.gamma.noalias() += root.weight * root.gamma;
                accumulate_weighted_tensor(st.Gamma_vec, root.Gamma_vec, root.weight);
                st.F_A_mo.noalias() += root.weight * root.F_A_mo;
            }

            st.g_orb = build_weighted_root_orbital_gradient(st.roots, nbasis);
            const RootResolvedGradientScreen gradient_screen =
                build_root_resolved_gradient_screen(st.roots);

            const Eigen::MatrixXd h_mo = C_trial.transpose() * calc._hcore * C_trial;
            const double E_core = compute_core_energy(h_mo, st.F_I_mo, n_core);
            double E_act = 0.0;
            for (const auto &root : st.roots)
                E_act += root.weight * root.ci_energy;
            st.E_cas = calc._nuclear_repulsion + E_core + E_act;
            st.gnorm = st.g_orb.cwiseAbs().maxCoeff();
            st.weighted_root_gnorm = gradient_screen.weighted;
            st.max_root_gnorm = gradient_screen.max_root;
            return st;
        };

        std::vector<RotPair> opt_pairs;
        for (const auto &pair : non_redundant_pairs(n_core, n_act, n_virt))
        {
            if (use_sym && !all_mo_irr.empty())
            {
                const int ip = (pair.p < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[pair.p] : -1;
                const int iq = (pair.q < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[pair.q] : -1;
                if (ip >= 0 && iq >= 0 && ip != iq)
                    continue;
            }
            opt_pairs.push_back(pair);
        }

        auto pack_pairs = [&](const Eigen::MatrixXd &M)
        {
            // Compress the antisymmetric orbital gradient/Hessian to just the
            // symmetry-allowed non-redundant rotation parameters.
            Eigen::VectorXd v = Eigen::VectorXd::Zero(static_cast<int>(opt_pairs.size()));
            for (int k = 0; k < static_cast<int>(opt_pairs.size()); ++k)
                v(k) = M(opt_pairs[k].p, opt_pairs[k].q);
            return v;
        };

        auto unpack_pairs = [&](const Eigen::VectorXd &v)
        {
            Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbasis, nbasis);
            for (int k = 0; k < static_cast<int>(opt_pairs.size()); ++k)
            {
                M(opt_pairs[k].p, opt_pairs[k].q) = v(k);
                M(opt_pairs[k].q, opt_pairs[k].p) = -v(k);
            }
            return M;
        };

        auto build_numeric_newton_step =
            [&](const McscfState &st_cur,
                const Eigen::MatrixXd &C_cur,
                double lm_shift,
                MacroDiagnostics &diag)
        {
            // For small pair spaces, estimate the orbital Hessian numerically from
            // fully reevaluated gradients. This is too expensive for production use
            // in large spaces, but it is invaluable as a debug/trust-region fallback.
            const int npairs = static_cast<int>(opt_pairs.size());
            Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(nbasis, nbasis);
            if (npairs == 0 || npairs > numeric_newton_pair_limit)
                return zero;

            const Eigen::VectorXd g0 = pack_pairs(st_cur.g_orb);
            if (g0.cwiseAbs().maxCoeff() < 1e-10)
                return zero;

            diag.numeric_newton_attempted = true;
            constexpr double fd_step = 5e-4;
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(npairs, npairs);
            const RootReference local_ref{st_cur.ci_energies, st_cur.ci_vecs, true};

            for (int k = 0; k < npairs; ++k)
            {
                Eigen::VectorXd ek = Eigen::VectorXd::Zero(npairs);
                ek(k) = fd_step;
                auto plus_res = evaluate(apply_orbital_rotation(C_cur, unpack_pairs(ek), calc._overlap), &local_ref, false);
                auto minus_res = evaluate(apply_orbital_rotation(C_cur, unpack_pairs(-ek), calc._overlap), &local_ref, false);
                if (!plus_res || !minus_res)
                {
                    diag.numeric_newton_failed = true;
                    return zero;
                }

                H.col(k) = (pack_pairs(plus_res->g_orb) - pack_pairs(minus_res->g_orb)) / (2.0 * fd_step);
                if (!H.col(k).allFinite())
                {
                    diag.numeric_newton_failed = true;
                    return zero;
                }
            }

            H = 0.5 * (H + H.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
            if (eig.info() != Eigen::Success)
            {
                diag.numeric_newton_failed = true;
                return zero;
            }

            Eigen::VectorXd evals = eig.eigenvalues();
            const double floor = std::max(1e-4, lm_shift);
            for (int i = 0; i < evals.size(); ++i)
                evals(i) = std::max(evals(i), floor);

            Eigen::VectorXd step = -eig.eigenvectors() * evals.cwiseInverse().asDiagonal() * eig.eigenvectors().transpose() * g0;
            if (!step.allFinite())
            {
                diag.numeric_newton_failed = true;
                return zero;
            }

            Eigen::MatrixXd kappa = unpack_pairs(step);
            const double max_elem = kappa.cwiseAbs().maxCoeff();
            if (max_elem > max_rot)
                kappa *= max_rot / max_elem;

            const double frob = kappa.norm();
            if (frob > trust_radius_frob)
                kappa *= trust_radius_frob / frob;
            return kappa;
        };

        auto cap_orbital_step = [&](Eigen::MatrixXd kappa)
        {
            const double max_elem = kappa.cwiseAbs().maxCoeff();
            if (max_elem > max_rot)
                kappa *= max_rot / max_elem;

            const double frob = kappa.norm();
            if (frob > trust_radius_frob)
                kappa *= trust_radius_frob / frob;
            return kappa;
        };

        auto build_root_resolved_gradient_fallback_step_set =
            [&](const std::vector<StateSpecificData> &roots)
        {
            RootResolvedOrbitalStepSet steps;
            steps.weighted = Eigen::MatrixXd::Zero(nbasis, nbasis);
            steps.per_root.reserve(roots.size());
            // Mirror the root-first AH path for the fallback direction: build a
            // raw descent proposal per root, then reduce them with the SA weights
            // while still keeping the state-specific candidates available.
            for (const auto &root : roots)
            {
                Eigen::MatrixXd root_step = Eigen::MatrixXd::Zero(nbasis, nbasis);
                if (root.weight == 0.0)
                {
                    steps.per_root.push_back(std::move(root_step));
                    continue;
                }
                for (const auto &pair : opt_pairs)
                {
                    const double step = -root.g_orb(pair.p, pair.q);
                    root_step(pair.p, pair.q) = step;
                    root_step(pair.q, pair.p) = -step;
                }
                root_step = cap_orbital_step(std::move(root_step));
                steps.weighted.noalias() += root.weight * root_step;
                steps.per_root.push_back(std::move(root_step));
            }
            steps.weighted = cap_orbital_step(std::move(steps.weighted));
            return steps;
        };

        auto build_single_pair_probe_step = [&](int pair_index, double signed_magnitude)
        {
            Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);
            if (pair_index < 0 || pair_index >= static_cast<int>(opt_pairs.size()))
                return kappa;

            const auto &pair = opt_pairs[static_cast<std::size_t>(pair_index)];
            kappa(pair.p, pair.q) = signed_magnitude;
            kappa(pair.q, pair.p) = -signed_magnitude;
            return kappa;
        };

        double E_prev = 0.0;
        double prev_sa_gnorm = std::numeric_limits<double>::infinity();
        bool converged = false;
        double level_shift = 0.2;
        int rejected_streak = 0;
        int stagnation_streak = 0;
        RootReference root_reference;

        for (unsigned int macro = 1; macro <= as.mcscf_max_iter; ++macro)
        {
            const auto macro_start = std::chrono::steady_clock::now();
            const RootReference previous_root_reference = root_reference;
            auto res = evaluate(C, root_reference.valid ? &root_reference : nullptr, root_reference.valid);
            if (!res)
                return std::unexpected(res.error());
            auto st_current = std::move(*res);

            MacroDiagnostics diag;
            diag.max_root_delta = compute_max_root_delta(
                previous_root_reference, build_root_energy_vector(st_current.roots));
            root_reference = {
                build_root_energy_vector(st_current.roots),
                build_root_ci_matrix(st_current.roots),
                true};

            const bool e_conv = macro > 1 && std::abs(st_current.E_cas - E_prev) < as.tol_mcscf_energy;
            const bool g_conv = sa_gradient_converged(st_current.gnorm, as.tol_mcscf_grad);
            const bool no_orb_rot = (st_current.gnorm == 0.0);
            // Under mcscf_accept_uphill, additionally require the per-root
            // gradient to be small (within 100x of the SA tolerance) before
            // declaring convergence. This keeps the optimizer iterating past
            // SA-stationary points where individual roots are still large,
            // letting the Pass-2 uphill branch attempt a basin escape.
            const double max_root_gconv_tol = 100.0 * as.tol_mcscf_grad;
            const bool per_root_g_conv =
                !as.mcscf_accept_uphill ||
                st_current.max_root_gnorm <= max_root_gconv_tol;
            if (per_root_g_conv && ((e_conv && g_conv) || (g_conv && no_orb_rot)))
            {
                if (macro == 1)
                {
                    const double macro_time_sec =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - macro_start).count();
                    HartreeFock::Logger::casscf_iteration(
                        0,
                        st_current.E_cas,
                        0.0,
                        st_current.gnorm,
                        st_current.max_root_gnorm,
                        0.0,
                        level_shift,
                        macro_time_sec);
                    logging(LogLevel::Info, tag + " :",
                            std::format(
                                "Macro {:3d}  mode={:<12}  ci_solver={}\n"
                                "             accepted={:<3}  candidate={}  max_root_dE={:.2e}  step_norm={:.2e}\n"
                                "             sa_g={:.2e}  root_screen_g={:.2e}  max_root_g={:.2e}\n"
                                "             predicted_dE={:.2e}  root_model_spread={:.2e}  actual_dE={:.2e}  response_resid={:.2e}\n"
                                "             response_iter={:3d}  level_shift={:.2e}",
                                0,
                                "initial",
                                st_current.ci_used_direct_sigma ? "direct-davidson" : "dense",
                                "no",
                                "already-converged",
                                diag.max_root_delta,
                                0.0,
                                st_current.gnorm,
                                st_current.weighted_root_gnorm,
                                st_current.max_root_gnorm,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0,
                                level_shift));
                }
                converged = true;
                break;
            }

            Eigen::MatrixXd kappa_total = Eigen::MatrixXd::Zero(nbasis, nbasis);
            Eigen::MatrixXd kappa_newton = Eigen::MatrixXd::Zero(nbasis, nbasis);
            const bool build_numeric_newton_candidate =
                use_numeric_newton_debug || static_cast<int>(opt_pairs.size()) <= numeric_newton_pair_limit;
            if (build_numeric_newton_candidate)
            {
                kappa_newton = build_numeric_newton_step(st_current, C, level_shift, diag);
                if (diag.numeric_newton_attempted && diag.numeric_newton_failed)
                    logging(LogLevel::Warning, tag + " :",
                            "Finite-difference Newton fallback produced an inconsistent column and was discarded.");
            }

            const CISigmaApplier ci_apply = [&](const Eigen::VectorXd &vec, Eigen::VectorXd &sigma_vec)
            {
                apply_ci_hamiltonian(
                    st_current.ci_space, a_strs, b_strs,
                    st_current.h_eff, st_current.ga, n_act,
                    vec, sigma_vec);
            };

            auto build_root_resolved_coupled_step_set =
                [&](const std::vector<StateSpecificData> &roots,
                    double level_shift_local)
            {
                RootResolvedCoupledStepSet steps;
                steps.orbital_steps.weighted = Eigen::MatrixXd::Zero(nbasis, nbasis);
                steps.orbital_steps.per_root.reserve(roots.size());

                for (const auto &root : roots)
                {
                    Eigen::MatrixXd root_step = Eigen::MatrixXd::Zero(nbasis, nbasis);
                    if (root.weight == 0.0)
                    {
                        steps.orbital_steps.per_root.push_back(std::move(root_step));
                        continue;
                    }

                    const OrbitalHessianContext root_hessian_ctx{
                        .C = &C,
                        .S = &calc._overlap,
                        .H_core = &calc._hcore,
                        .eri = &eri,
                        .gamma = &root.gamma,
                        .Gamma_vec = &root.Gamma_vec,
                    };
                    const CoupledStepSolveResult result =
                        solve_coupled_orbital_ci_step(
                            configured_rhs_mode,
                            root.g_orb,
                            st_current.F_I_mo,
                            root.F_A_mo,
                            st_current.h_eff,
                            st_current.ga,
                            st_current.ci_space,
                            a_strs,
                            b_strs,
                            st_current.dets,
                            st_current.active_integrals,
                            ci_apply,
                            root.ci_vector,
                            root.ci_energy,
                            st_current.H_CI_diag,
                            nbasis,
                            n_core,
                            n_act,
                            n_virt,
                            level_shift_local,
                            max_rot,
                            all_mo_irr,
                            use_sym,
                            &root_hessian_ctx);

                    steps.converged = steps.converged && result.converged;
                    steps.max_orbital_residual =
                        std::max(steps.max_orbital_residual, result.orbital_residual_max);
                    steps.max_ci_residual =
                        std::max(steps.max_ci_residual, result.ci_residual_norm);
                    steps.max_iterations =
                        std::max(steps.max_iterations, result.iterations);

                    root_step = cap_orbital_step(result.orbital_step);
                    steps.orbital_steps.weighted.noalias() += root.weight * root_step;
                    steps.orbital_steps.per_root.push_back(std::move(root_step));
                }

                steps.orbital_steps.weighted = cap_orbital_step(std::move(steps.orbital_steps.weighted));
                return steps;
            };

            std::vector<StateAveragedCoupledRoot> sa_coupled_roots;
            sa_coupled_roots.reserve(st_current.roots.size());
            for (const auto &root : st_current.roots)
                sa_coupled_roots.push_back({root.weight, root.ci_vector, root.ci_energy});

            const OrbitalHessianContext sa_hessian_ctx{
                .C = &C,
                .S = &calc._overlap,
                .H_core = &calc._hcore,
                .eri = &eri,
                .gamma = &st_current.gamma,
                .Gamma_vec = &st_current.Gamma_vec,
            };

            const SACoupledStepSolveResult sa_coupled_result =
                solve_sa_coupled_orbital_ci_step(
                    configured_rhs_mode,
                    st_current.g_orb,
                    st_current.F_I_mo,
                    st_current.F_A_mo,
                    st_current.h_eff,
                    st_current.ga,
                    st_current.ci_space,
                    a_strs,
                    b_strs,
                    st_current.dets,
                    st_current.active_integrals,
                    ci_apply,
                    sa_coupled_roots,
                    st_current.H_CI_diag,
                    nbasis,
                    n_core,
                    n_act,
                    n_virt,
                    level_shift,
                    max_rot,
                    all_mo_irr,
                    use_sym,
                    &sa_hessian_ctx,
                    1e-6,
                    24,
                    1e-4);

            if (!sa_coupled_result.converged)
                diag.sa_coupled_step_inexact = true;
            diag.max_response_residual =
                std::max(diag.max_response_residual, sa_coupled_result.max_ci_residual_norm);
            diag.max_response_iterations =
                std::max(diag.max_response_iterations, sa_coupled_result.iterations);

            for (unsigned int micro = 0; micro < nmicro; ++micro)
            {
                const RootResolvedOrbitalStepSet kappa_step_set = build_root_resolved_orbital_step_set(
                    st_current.roots, st_current.F_I_mo, nbasis,
                    n_core, n_act, n_virt,
                    level_shift, max_rot, all_mo_irr, use_sym);
                const Eigen::MatrixXd &kappa = kappa_step_set.weighted;
                kappa_total += kappa;

                const int nr_used = static_cast<int>(st_current.roots.size());
                for (int r = 0; r < nr_used; ++r)
                {
                    auto &root = st_current.roots[static_cast<std::size_t>(r)];
                    const Eigen::VectorXd &c0r = root.ci_vector;
                    root.response_rhs_mode = configured_rhs_mode;
                    const CoupledResponseBlocks blocks =
                        build_coupled_response_blocks(
                            configured_rhs_mode,
                            kappa,
                            st_current.F_I_mo,
                            st_current.h_eff,
                            st_current.ga,
                            st_current.ci_space,
                            a_strs,
                            b_strs,
                            st_current.dets,
                            st_current.active_integrals,
                            ci_apply,
                            c0r,
                            root.ci_energy,
                            st_current.H_CI_diag,
                            nbasis,
                            n_core,
                            n_act,
                            n_virt);
                    if (!blocks.ci_response.converged)
                        diag.ci_response_fallback_used = true;

                    root.response_rhs = blocks.ci_rhs;
                    root.c1_response = blocks.ci_response.c1;
                    root.response_residual = blocks.ci_residual;
                    root.Gamma1_vec = blocks.Gamma1_vec;
                    root.Q1 = blocks.Q1;
                    root.g_ci = blocks.orbital_correction;
                    root.g_orb = fep1_gradient_update(
                        root.g_orb, kappa, st_current.F_I_mo, root.F_A_mo,
                        n_core, n_act, n_virt);
                    root.g_orb += root.g_ci;

                    diag.max_response_residual =
                        std::max(diag.max_response_residual, blocks.ci_response.residual_norm);
                    diag.max_response_iterations =
                        std::max(diag.max_response_iterations, blocks.ci_response.iterations);
                    diag.max_response_regularization =
                        std::max(diag.max_response_regularization, blocks.ci_response.max_denominator_regularization);
                }
            }
            const double max_k = kappa_total.cwiseAbs().maxCoeff();
            if (max_k > max_rot)
                kappa_total *= max_rot / max_k;
            const Eigen::MatrixXd &kappa_coupled = sa_coupled_result.orbital_step;
            const RootResolvedOrbitalStepSet kappa_grad_step_set =
                build_root_resolved_gradient_fallback_step_set(st_current.roots);
            const Eigen::MatrixXd &kappa_grad = kappa_grad_step_set.weighted;

            const WeightedRootProbeSignal probe_signal =
                build_weighted_root_probe_signal(st_current.roots, opt_pairs);
            std::vector<CandidateStep> step_candidates;
            const bool coupled_step_reliable =
                sa_coupled_result.converged &&
                std::isfinite(sa_coupled_result.max_ci_residual_norm) &&
                std::isfinite(sa_coupled_result.orbital_residual_max);
            const bool use_numeric_newton_fallback =
                use_numeric_newton_debug ||
                (static_cast<int>(opt_pairs.size()) <= numeric_newton_pair_limit &&
                 (!coupled_step_reliable || stagnation_streak >= 2));
            const bool use_diagonal_fallback = stagnation_streak >= 2;

            // CIAH augmented-Hessian step over the SA-averaged orbital
            // gradient. Uses the same Hessian context the SA coupled solve
            // already built, so the cost over the existing cascade is one
            // extra Davidson loop driven by delta_g_sa_action. We always
            // generate the candidate so the merit selector can compare it
            // against sa-coupled; it only wins when the bordered eigenvalue
            // is well-conditioned and the resulting kappa survives the cap.
            const OrbitalAugHessianStep sa_aug_hess =
                solve_orbital_augmented_hessian_step(
                    st_current.g_orb,
                    st_current.F_I_mo,
                    st_current.F_A_mo,
                    &sa_hessian_ctx,
                    n_core,
                    n_act,
                    n_virt,
                    level_shift,
                    max_rot,
                    all_mo_irr,
                    use_sym);
            const Eigen::MatrixXd &kappa_aug_hess = sa_aug_hess.kappa;

            append_candidate_step(step_candidates, kappa_coupled, "sa-coupled");
            if (sa_aug_hess.kappa.allFinite() &&
                sa_aug_hess.kappa.cwiseAbs().maxCoeff() > 1e-12 &&
                !sa_aug_hess.ah.orbital_only_fallback)
                append_candidate_step(step_candidates, kappa_aug_hess, "sa-aug-hessian");
            if (use_numeric_newton_fallback)
                append_candidate_step(step_candidates, kappa_newton, "numeric-newton");
            if (use_diagonal_fallback)
                append_candidate_step(step_candidates, kappa_total, "sa-diag-fallback");
            append_candidate_step(step_candidates, kappa_grad, "sa-grad-fallback");

            if (stagnation_streak >= 2 && probe_signal.weighted_abs.size() > 0)
            {
                if (nroots > 1)
                {
                    const RootResolvedCoupledStepSet root_resolved_coupled_step_set =
                        build_root_resolved_coupled_step_set(st_current.roots, level_shift);
                    append_root_candidate_steps(
                        step_candidates, root_resolved_coupled_step_set.orbital_steps.per_root, "coupled", false);
                    append_root_candidate_steps(
                        step_candidates, kappa_grad_step_set.per_root, "grad-fallback", false);
                }

                std::vector<int> ranked_pairs(static_cast<std::size_t>(probe_signal.weighted_abs.size()));
                std::iota(ranked_pairs.begin(), ranked_pairs.end(), 0);
                std::partial_sort(
                    ranked_pairs.begin(),
                    ranked_pairs.begin() + std::min<std::size_t>(2, ranked_pairs.size()),
                    ranked_pairs.end(),
                    [&](int lhs, int rhs)
                    {
                        return probe_signal.weighted_abs(lhs) > probe_signal.weighted_abs(rhs);
                    });

                for (std::size_t i = 0; i < std::min<std::size_t>(2, ranked_pairs.size()); ++i)
                {
                    const int k = ranked_pairs[i];
                    if (probe_signal.weighted_abs(k) < 1e-6)
                        break;

                    const double signed_probe =
                        (probe_signal.weighted_signed(k) >= 0.0) ? -max_rot : max_rot;
                    append_candidate_step(
                        step_candidates,
                        build_single_pair_probe_step(k, signed_probe),
                        std::format("probe-pair{:d}-favored", k));
                    append_candidate_step(
                        step_candidates,
                        build_single_pair_probe_step(k, -signed_probe),
                        std::format("probe-pair{:d}-opposite", k));
                }
            }

            const CandidateSelection accepted_candidate =
                select_best_candidate_step(
                    step_candidates,
                    C,
                    calc._overlap,
                    &root_reference,
                    st_current,
                    opt_pairs,
                    as.tol_mcscf_energy,
                    evaluate,
                    as.mcscf_accept_uphill,
                    as.mcscf_uphill_max_eh);

            if (accepted_candidate.accepted)
            {
                diag.step_accepted = true;
                diag.accepted_candidate_label = accepted_candidate.label;
                diag.accepted_sa_gnorm = accepted_candidate.sa_gnorm;
                diag.accepted_weighted_root_gnorm = accepted_candidate.weighted_root_gnorm;
                diag.accepted_max_root_gnorm = accepted_candidate.max_root_gnorm;
                diag.predicted_delta = accepted_candidate.predicted_delta;
                diag.max_root_predicted_delta_deviation =
                    accepted_candidate.max_root_predicted_delta_deviation;
                diag.accepted_step_norm = accepted_candidate.step.norm();
                diag.actual_delta = accepted_candidate.energy - st_current.E_cas;
                C = accepted_candidate.coefficients;
                st_current = std::move(accepted_candidate.state);
                root_reference = {
                    build_root_energy_vector(st_current.roots),
                    build_root_ci_matrix(st_current.roots),
                    true};
                level_shift = std::max(1e-3, level_shift * 0.7);
                rejected_streak = 0;
            }
            else
            {
                level_shift = std::min(20.0, level_shift * 2.0);
                ++rejected_streak;
                logging(LogLevel::Warning, tag + " :",
                        "No orbital step candidate improved the merit function; increasing damping and retrying next macroiteration.");
            }

            const double reported_gnorm = st_current.g_orb.cwiseAbs().maxCoeff();
            st_current.gnorm = reported_gnorm;
            const double reported_screen_gnorm = st_current.weighted_root_gnorm;
            const double reported_max_root_gnorm = st_current.max_root_gnorm;
            const double dE = st_current.E_cas - E_prev;
            E_prev = st_current.E_cas;

            const bool small_energy_change = macro > 1 && std::abs(dE) < std::max(10.0 * as.tol_mcscf_energy, 1e-8);
            const bool little_gradient_progress =
                sa_gradient_progress_flat(reported_gnorm, prev_sa_gnorm);
            const bool accepted_micro_step_plateau =
                diag.step_accepted && diag.accepted_step_norm < 5e-5;
            // Track repeated "flat" iterations separately from hard rejections so
            // we can switch to more exploratory probes before declaring failure.
            if ((!diag.step_accepted && rejected_streak >= 2) || (small_energy_change && little_gradient_progress))
                ++stagnation_streak;
            else
                stagnation_streak = 0;
            prev_sa_gnorm = reported_gnorm;

            if (stagnation_streak >= 2)
            {
                level_shift = std::min(50.0, level_shift * 1.5);
                logging(LogLevel::Warning, tag + " :",
                        std::format("Detected stagnation over {:d} macroiterations; increasing damping to {:.3f}.",
                                    stagnation_streak, level_shift));
            }

            const double macro_time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - macro_start).count();
            HartreeFock::Logger::casscf_iteration(
                macro, st_current.E_cas, dE, reported_gnorm, reported_max_root_gnorm, diag.accepted_step_norm,
                level_shift, macro_time_sec);

            logging(LogLevel::Info, tag + " :",
                    std::format(
                        "Macro {:3d}  mode={:<12}  ci_solver={}\n"
                        "             accepted={:<3}  candidate={}  max_root_dE={:.2e}  step_norm={:.2e}\n"
                        "             sa_g={:.2e}  root_screen_g={:.2e}  max_root_g={:.2e}\n"
                        "             predicted_dE={:.2e}  root_model_spread={:.2e}  actual_dE={:.2e}  response_resid={:.2e}\n"
                        "             response_iter={:3d}  level_shift={:.2e}",
                        macro,
                        response_mode_name(configured_response_mode),
                        st_current.ci_used_direct_sigma ? "direct-davidson" : "dense",
                        diag.step_accepted ? "yes" : "no",
                        diag.accepted_candidate_label,
                        diag.max_root_delta,
                        diag.accepted_step_norm,
                        reported_gnorm,
                        diag.step_accepted ? diag.accepted_weighted_root_gnorm : reported_screen_gnorm,
                        diag.step_accepted ? diag.accepted_max_root_gnorm : reported_max_root_gnorm,
                        diag.predicted_delta,
                        diag.max_root_predicted_delta_deviation,
                        diag.actual_delta,
                        diag.max_response_residual,
                        diag.max_response_iterations,
                        level_shift));
            if (diag.ci_response_fallback_used)
                logging(LogLevel::Warning, tag + " :",
                        "CI response Davidson solve did not fully converge for at least one root; using single-step fallback.");

            if (stagnation_streak >= 2 && small_energy_change && accepted_micro_step_plateau)
            {
                logging(LogLevel::Warning, tag + " :",
                        "Treating the stationary orbital plateau as converged: the CASSCF energy and accepted orbital step are flat, while the weighted and max-root orbital-gradient screens are no longer improving.");
                converged = true;
                break;
            }

            const bool e_conv_post = macro > 1 && std::abs(dE) < as.tol_mcscf_energy;
            const bool g_conv_post = sa_gradient_converged(reported_gnorm, as.tol_mcscf_grad);
            const bool no_orb_rot_post = (reported_gnorm == 0.0);
            const double max_root_gconv_tol_post = 100.0 * as.tol_mcscf_grad;
            const bool per_root_g_conv_post =
                !as.mcscf_accept_uphill ||
                reported_max_root_gnorm <= max_root_gconv_tol_post;
            if (per_root_g_conv_post &&
                ((e_conv_post && g_conv_post) || (g_conv_post && no_orb_rot_post)))
            {
                converged = true;
                break;
            }
        }

        if (!converged)
            return std::unexpected(
                std::format("{}: did not converge in {:d} iterations.", tag, as.mcscf_max_iter));

        HartreeFock::Logger::blank();
        logging(LogLevel::Info, tag + " :", "Converged.");

        auto final_res = evaluate(C, root_reference.valid ? &root_reference : nullptr, false);
        if (!final_res)
            return std::unexpected(final_res.error());
        const auto &fst = *final_res;

        const NaturalOrbitalData natural_orbitals = diagonalize_natural_orbitals(fst.gamma);
        calc._cas_nat_occ = natural_orbitals.occupations;
        calc._cas_mo_coefficients = C;
        calc._total_energy = fst.E_cas;

        if (nroots > 1)
        {
            const Eigen::MatrixXd h_mo = C.transpose() * calc._hcore * C;
            const double E_core = compute_core_energy(h_mo, fst.F_I_mo, n_core);
            const double shared_energy = calc._nuclear_repulsion + E_core;
            calc._cas_root_energies.resize(nroots);
            for (int r = 0; r < nroots; ++r)
                calc._cas_root_energies(r) = shared_energy + fst.roots[r].ci_energy;
        }

        return {};
    }

} // namespace HartreeFock::Correlation::CASSCF

namespace HartreeFock::Correlation
{

    std::expected<void, std::string> run_casscf(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        RASParams ras;
        return CASSCF::run_mcscf_loop(calc, shell_pairs, "CASSCF", ras);
    }

    std::expected<void, std::string> run_rasscf(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        const auto &as = calc._active_space;
        RASParams ras;
        ras.nras1 = as.nras1;
        ras.nras2 = as.nras2;
        ras.nras3 = as.nras3;
        ras.max_holes = as.max_holes;
        ras.max_elec = as.max_elec;
        ras.active = true;
        return CASSCF::run_mcscf_loop(calc, shell_pairs, "RASSCF", ras);
    }

} // namespace HartreeFock::Correlation
