#include "post_hf/casscf/casscf.h"

#include "io/logging.h"
#include "post_hf/casscf.h"
#include "post_hf/casscf/ci.h"
#include "post_hf/casscf/orbital.h"
#include "post_hf/casscf/rdm.h"
#include "post_hf/casscf/response.h"
#include "post_hf/casscf/strings.h"
#include "post_hf/integrals.h"

#include <algorithm>
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
    using HartreeFock::Correlation::CASSCF::ResponseMode;
    using HartreeFock::Correlation::CASSCF::ResponseRHSMode;
    using HartreeFock::Correlation::CASSCF::RotPair;
    using HartreeFock::Correlation::CASSCF::hess_diag;
    using HartreeFock::Correlation::CASSCF::quadratic_model_delta;
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

    // Keep the state-averaged driver rooted in explicit per-root data so the
    // code can defer averaging until the last possible moment.
    struct StateSpecificData
    {
        double ci_energy = 0.0;
        double weight = 0.0;
        Eigen::VectorXd ci_vector;
        Eigen::MatrixXd gamma;
        std::vector<double> Gamma_vec;
        Eigen::MatrixXd F_A_mo;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd g_orb;
        ResponseRHSMode response_rhs_mode = ResponseRHSMode::CommutatorOnlyApproximate;
        Eigen::VectorXd response_rhs;
        Eigen::VectorXd c1_response;
        Eigen::VectorXd response_residual;
        std::vector<double> Gamma1_vec;
        Eigen::MatrixXd Q1;
        Eigen::MatrixXd g_ci;
    };

    // Scratch state returned by each full CASSCF evaluation at a fixed MO basis.
    // This bundles the CI model, active-space intermediates, state-averaged
    // densities, and diagnostics needed by the macro/micro optimizer.
    struct McscfState
    {
        CIDeterminantSpace ci_space;
        Eigen::MatrixXd h_eff;
        std::vector<double> ga;
        Eigen::MatrixXd F_I_mo;
        Eigen::MatrixXd F_A_mo;
        Eigen::MatrixXd gamma;
        Eigen::MatrixXd g_orb;
        std::vector<double> Gamma_vec;
        ActiveIntegralCache active_integrals;
        Eigen::VectorXd H_CI_diag;
        Eigen::VectorXd ci_energies;
        Eigen::MatrixXd ci_vecs;
        std::vector<StateSpecificData> roots;
        std::vector<std::pair<int, int>> dets;
        bool ci_used_direct_sigma = false;
        double E_cas = 0.0;
        double gnorm = 0.0;
        double weighted_root_gnorm = 0.0;
        double max_root_gnorm = 0.0;
    };

    // Root identities are tracked across macroiterations by CI overlap so the
    // state-averaged weights remain attached to the same physical roots.
    struct RootReference
    {
        Eigen::VectorXd energies;
        Eigen::MatrixXd vecs;
        bool valid = false;
    };

    struct MacroDiagnostics
    {
        double max_response_residual = 0.0;
        int max_response_iterations = 0;
        double max_response_regularization = 0.0;
        bool response_fallback_used = false;
        bool numeric_newton_attempted = false;
        bool numeric_newton_failed = false;
        bool step_accepted = false;
        double accepted_step_norm = 0.0;
        double predicted_delta = 0.0;
        double max_root_predicted_delta_deviation = 0.0;
        double actual_delta = 0.0;
        double max_root_delta = 0.0;
        std::string accepted_candidate_label = "none";
        double accepted_weighted_root_gnorm = 0.0;
        double accepted_max_root_gnorm = 0.0;
    };

    struct WeightedQuadraticModelPrediction
    {
        double weighted_delta = 0.0;
        double max_root_deviation = 0.0;
    };

    struct WeightedRootProbeSignal
    {
        Eigen::VectorXd weighted_abs;
        Eigen::VectorXd weighted_signed;
    };

    struct RootResolvedOrbitalStepSet
    {
        Eigen::MatrixXd weighted;
        std::vector<Eigen::MatrixXd> per_root;
    };

    struct RootResolvedGradientScreen
    {
        double weighted = 0.0;
        double max_root = 0.0;
    };

    // Reorder the newly solved CI roots to best match the previous macroiteration.
    // This avoids state flipping when near-degenerate roots cross in energy.
    void reorder_ci_roots(
        Eigen::VectorXd &E,
        Eigen::MatrixXd &V,
        const RootReference *root_ref,
        const std::string &tag,
        bool log_tracking)
    {
        using HartreeFock::LogLevel;
        using HartreeFock::Logger::logging;

        if (root_ref == nullptr || !root_ref->valid)
            return;
        if (root_ref->vecs.rows() != V.rows() || root_ref->vecs.cols() == 0 || V.cols() == 0)
            return;

        const int nmatch = std::min<int>(root_ref->vecs.cols(), V.cols());
        const Eigen::MatrixXd overlaps =
            compute_root_overlap(root_ref->vecs.leftCols(nmatch), V.leftCols(nmatch));
        if (overlaps.size() == 0)
            return;

        const std::vector<int> assignment = match_roots_by_max_overlap(overlaps);
        Eigen::VectorXd E_reordered = E;
        Eigen::MatrixXd V_reordered = V;
        std::vector<bool> used_new(static_cast<std::size_t>(V.cols()), false);
        int swaps = 0;
        double min_overlap = 1.0;

        for (int i = 0; i < nmatch; ++i)
        {
            const int j = assignment[i];
            if (j < 0 || j >= V.cols())
                continue;
            E_reordered(i) = E(j);
            V_reordered.col(i) = V.col(j);
            used_new[static_cast<std::size_t>(j)] = true;
            min_overlap = std::min(min_overlap, std::abs(overlaps(i, j)));
            if (i != j)
                ++swaps;
        }

        int next_slot = nmatch;
        for (int j = 0; j < V.cols() && next_slot < V.cols(); ++j)
        {
            if (used_new[static_cast<std::size_t>(j)])
                continue;
            E_reordered(next_slot) = E(j);
            V_reordered.col(next_slot) = V.col(j);
            ++next_slot;
        }

        E = std::move(E_reordered);
        V = std::move(V_reordered);

        if (log_tracking && nmatch > 0)
        {
            if (swaps > 0)
                logging(LogLevel::Info, tag + " :",
                        std::format("Root tracking reordered {:d} CI roots (min |overlap| = {:.3f}).",
                                    swaps, min_overlap));
            if (min_overlap < 0.7)
                logging(LogLevel::Warning, tag + " :",
                        std::format("Root tracking minimum |overlap| dropped to {:.3f}; state identities may be unstable.",
                                    min_overlap));
        }
    }

    double compute_max_root_delta(const RootReference &previous, const Eigen::VectorXd &current)
    {
        if (!previous.valid || previous.energies.size() == 0 || current.size() == 0)
            return 0.0;
        const int n = std::min<int>(previous.energies.size(), current.size());
        double delta = 0.0;
        for (int i = 0; i < n; ++i)
            delta = std::max(delta, std::abs(current(i) - previous.energies(i)));
        return delta;
    }

    Eigen::MatrixXd as_single_column_matrix(const Eigen::VectorXd &vec)
    {
        Eigen::MatrixXd mat(vec.size(), 1);
        mat.col(0) = vec;
        return mat;
    }

    Eigen::VectorXd single_weight(double weight)
    {
        Eigen::VectorXd weights(1);
        weights(0) = weight;
        return weights;
    }

    void accumulate_weighted_tensor(
        std::vector<double> &destination,
        const std::vector<double> &source,
        double weight)
    {
        // The first contributing root fixes the tensor shape; later roots just add
        // their weighted contribution elementwise.
        if (destination.empty())
            destination.assign(source.size(), 0.0);
        for (std::size_t i = 0; i < source.size(); ++i)
            destination[i] += weight * source[i];
    }

    void accumulate_weighted_matrix(
        Eigen::MatrixXd &destination,
        const Eigen::MatrixXd &source,
        double weight)
    {
        // Mirror the tensor helper for matrix-valued intermediates such as F_A,
        // Q-derived gradients, and other root-resolved orbital data.
        if (destination.size() == 0)
            destination = Eigen::MatrixXd::Zero(source.rows(), source.cols());
        destination.noalias() += weight * source;
    }

    Eigen::MatrixXd build_root_ci_matrix(const std::vector<StateSpecificData> &roots)
    {
        if (roots.empty() || roots.front().ci_vector.size() == 0)
            return Eigen::MatrixXd();

        Eigen::MatrixXd matrix(roots.front().ci_vector.size(), static_cast<int>(roots.size()));
        for (int r = 0; r < static_cast<int>(roots.size()); ++r)
            matrix.col(r) = roots[static_cast<std::size_t>(r)].ci_vector;
        return matrix;
    }

    Eigen::VectorXd build_root_energy_vector(const std::vector<StateSpecificData> &roots)
    {
        Eigen::VectorXd energies(static_cast<int>(roots.size()));
        for (int r = 0; r < static_cast<int>(roots.size()); ++r)
            energies(r) = roots[static_cast<std::size_t>(r)].ci_energy;
        return energies;
    }

    Eigen::MatrixXd build_weighted_root_orbital_gradient(
        const std::vector<StateSpecificData> &roots,
        int nbasis)
    {
        Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(nbasis, nbasis);
        for (const auto &root : roots)
            gradient.noalias() += root.weight * root.g_orb;
        return gradient;
    }

    RootResolvedGradientScreen build_root_resolved_gradient_screen(
        const std::vector<StateSpecificData> &roots)
    {
        RootResolvedGradientScreen screen;
        for (const auto &root : roots)
        {
            if (root.g_orb.size() == 0)
                continue;
            const double root_norm = root.g_orb.cwiseAbs().maxCoeff();
            screen.weighted += root.weight * root_norm;
            screen.max_root = std::max(screen.max_root, root_norm);
        }
        return screen;
    }

    bool root_resolved_gradient_converged(
        double weighted_root_gnorm,
        double max_root_gnorm,
        double tol)
    {
        return weighted_root_gnorm < tol && max_root_gnorm < tol;
    }

    bool root_resolved_gradient_progress_flat(
        double weighted_root_gnorm,
        double prev_weighted_root_gnorm,
        double max_root_gnorm,
        double prev_max_root_gnorm)
    {
        if (!std::isfinite(prev_weighted_root_gnorm) || !std::isfinite(prev_max_root_gnorm))
            return false;

        const double weighted_window =
            std::max(0.05 * std::max(prev_weighted_root_gnorm, 1e-8), 1e-8);
        const double max_root_window =
            std::max(0.05 * std::max(prev_max_root_gnorm, 1e-8), 1e-8);
        return std::abs(weighted_root_gnorm - prev_weighted_root_gnorm) < weighted_window &&
               std::abs(max_root_gnorm - prev_max_root_gnorm) < max_root_window;
    }

    WeightedQuadraticModelPrediction build_weighted_root_quadratic_model_prediction(
        const std::vector<StateSpecificData> &roots,
        const Eigen::MatrixXd &F_I_mo,
        const std::vector<RotPair> &pairs,
        const Eigen::MatrixXd &step)
    {
        WeightedQuadraticModelPrediction prediction;
        if (roots.empty() || pairs.empty())
            return prediction;

        Eigen::VectorXd x(static_cast<int>(pairs.size()));
        for (int k = 0; k < static_cast<int>(pairs.size()); ++k)
            x(k) = step(pairs[static_cast<std::size_t>(k)].p, pairs[static_cast<std::size_t>(k)].q);

        std::vector<double> per_root_delta;
        per_root_delta.reserve(roots.size());

        // Keep the quadratic-model diagnostic root-resolved until the final
        // reduction so the log can report whether the current SA candidate is
        // uniformly favorable or is hiding strong per-root disagreement.
        for (const auto &root : roots)
        {
            if (root.weight == 0.0)
            {
                per_root_delta.push_back(0.0);
                continue;
            }

            const Eigen::MatrixXd F_sum = F_I_mo + root.F_A_mo;
            Eigen::VectorXd g_flat(static_cast<int>(pairs.size()));
            Eigen::VectorXd h_flat(static_cast<int>(pairs.size()));
            for (int k = 0; k < static_cast<int>(pairs.size()); ++k)
            {
                const auto &pair = pairs[static_cast<std::size_t>(k)];
                g_flat(k) = root.g_orb(pair.p, pair.q);
                h_flat(k) = hess_diag(F_sum, pair.p, pair.q);
            }

            const double delta = quadratic_model_delta(g_flat, h_flat, x);
            per_root_delta.push_back(delta);
            prediction.weighted_delta += root.weight * delta;
        }

        for (const double delta : per_root_delta)
            prediction.max_root_deviation =
                std::max(prediction.max_root_deviation, std::abs(delta - prediction.weighted_delta));

        return prediction;
    }

    WeightedRootProbeSignal build_weighted_root_probe_signal(
        const std::vector<StateSpecificData> &roots,
        const std::vector<RotPair> &pairs)
    {
        WeightedRootProbeSignal signal;
        signal.weighted_abs = Eigen::VectorXd::Zero(static_cast<int>(pairs.size()));
        signal.weighted_signed = Eigen::VectorXd::Zero(static_cast<int>(pairs.size()));
        if (roots.empty() || pairs.empty())
            return signal;

        // Use the per-root gradient magnitudes to decide which pair directions
        // deserve individual probe steps, while keeping a weighted signed vote
        // to choose the first probe orientation.
        for (const auto &root : roots)
        {
            if (root.weight == 0.0)
                continue;
            for (int k = 0; k < static_cast<int>(pairs.size()); ++k)
            {
                const auto &pair = pairs[static_cast<std::size_t>(k)];
                const double value = root.g_orb(pair.p, pair.q);
                signal.weighted_abs(k) += root.weight * std::abs(value);
                signal.weighted_signed(k) += root.weight * value;
            }
        }

        return signal;
    }

    RootResolvedOrbitalStepSet build_root_resolved_orbital_step_set(
        const std::vector<StateSpecificData> &roots,
        const Eigen::MatrixXd &F_I_mo,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym)
    {
        RootResolvedOrbitalStepSet steps;
        steps.weighted = Eigen::MatrixXd::Zero(nbasis, nbasis);
        steps.per_root.reserve(roots.size());

        // Build each root's preconditioned orbital step from its own gradient
        // and active Fock contribution before reducing those proposals back to
        // one state-averaged rotation. Keep the per-root steps too so the
        // later candidate screen can still evaluate them directly when the
        // weighted reduction damps an important state-specific direction.
        for (const auto &root : roots)
        {
            Eigen::MatrixXd step = Eigen::MatrixXd::Zero(nbasis, nbasis);
            if (root.weight == 0.0)
            {
                steps.per_root.push_back(std::move(step));
                continue;
            }

            step = HartreeFock::Correlation::CASSCF::augmented_hessian_step(
                root.g_orb, F_I_mo, root.F_A_mo,
                n_core, n_act, n_virt,
                level_shift, max_rot, mo_irreps, use_sym);
            steps.weighted.noalias() += root.weight * step;
            steps.per_root.push_back(std::move(step));
        }
        return steps;
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

        const bool have_sym = !calc._sao_irrep_names.empty() && static_cast<int>(calc._sao_irrep_names.size()) <= 8;
        const bool point_group_is_abelian_for_labels =
            point_group_has_only_1d_irreps(calc._molecule._point_group);
        std::optional<SymmetryContext> sym_ctx;
        std::vector<int> irr_act;
        std::vector<int> all_mo_irr;
        if (have_sym && point_group_is_abelian_for_labels && !calc._info._scf.alpha.mo_symmetry.empty())
        {
            sym_ctx = build_symmetry_context(calc);
            if (!sym_ctx)
                return std::unexpected(tag + ": failed to build an Abelian irrep product table for CI screening.");

            all_mo_irr = map_mo_irreps(calc._info._scf.alpha.mo_symmetry, sym_ctx->names);
            if (std::find(all_mo_irr.begin(), all_mo_irr.end(), -1) != all_mo_irr.end())
                return std::unexpected(tag + ": encountered an MO irrep label missing from the Abelian product table.");

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

        Eigen::MatrixXd C = (calc._cas_mo_coefficients.rows() == nbasis &&
                             calc._cas_mo_coefficients.cols() == nbasis)
                                ? calc._cas_mo_coefficients
                                : calc._info._scf.alpha.mo_coefficients;
        if (C.rows() != nbasis || C.cols() != nbasis)
            return std::unexpected(tag + ": MO coefficient matrix has wrong size.");

        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calc, shell_pairs, eri_local, tag + " :");

        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(n_act, n_alpha_act, n_beta_act, a_strs, b_strs);

        const unsigned int nmicro = std::max(1u, as.mcscf_micro_per_macro);
        const ResponseMode configured_response_mode = ResponseMode::DiagonalResponse;
        const ResponseRHSMode configured_rhs_mode =
            as.mcscf_debug_commutator_rhs
                ? ResponseRHSMode::CommutatorOnlyApproximate
                : ResponseRHSMode::ExactActiveSpaceOrbitalDerivative;
        const bool use_numeric_newton_debug = as.mcscf_debug_numeric_newton;
        const int numeric_newton_pair_limit = 64;
        const int ci_dense_threshold = 500;

        logging(LogLevel::Info, tag + " :",
                std::format("Active space: ({:d}e, {:d}o)  n_core={:d}  n_virt={:d}  CI dim ≤ {:d}",
                            as.nactele, n_act, n_core, n_virt, ci_dim_est));
        logging(LogLevel::Info, tag + " :",
                std::format("Algorithm: approximate macro/micro scaffold with {}  nmicro={:d}",
                            response_mode_name(configured_response_mode), nmicro));
        logging(LogLevel::Info, tag + " :",
                std::format("CI response RHS: {}",
                            response_rhs_mode_name(configured_rhs_mode)));
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
            st.roots.clear();
            st.roots.reserve(static_cast<std::size_t>(nr_used));
            st.F_A_mo = Eigen::MatrixXd::Zero(nbasis, nbasis);
            st.gamma = Eigen::MatrixXd::Zero(n_act, n_act);
            st.Gamma_vec.clear();
            st.g_orb = Eigen::MatrixXd::Zero(nbasis, nbasis);

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

                st.gamma.noalias() += root.weight * root.gamma;
                accumulate_weighted_tensor(st.Gamma_vec, root.Gamma_vec, root.weight);
                st.F_A_mo.noalias() += root.weight * root.F_A_mo;
                st.roots.push_back(std::move(root));
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
            if (max_elem > 0.20)
                kappa *= 0.20 / max_elem;

            const double trust_radius = 0.80;
            const double frob = kappa.norm();
            if (frob > trust_radius)
                kappa *= trust_radius / frob;
            return kappa;
        };

        auto cap_orbital_step = [&](Eigen::MatrixXd kappa)
        {
            const double max_elem = kappa.cwiseAbs().maxCoeff();
            if (max_elem > 0.20)
                kappa *= 0.20 / max_elem;

            const double trust_radius = 0.80;
            const double frob = kappa.norm();
            if (frob > trust_radius)
                kappa *= trust_radius / frob;
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

        auto build_root_resolved_coupled_correction_step_set =
            [&](const McscfState &st_base,
                const std::vector<StateSpecificData> &roots,
                double level_shift_local)
        {
            RootResolvedOrbitalStepSet steps;
            steps.weighted = Eigen::MatrixXd::Zero(nbasis, nbasis);
            steps.per_root.reserve(roots.size());
            // Once the explicit CI response has updated each root's orbital
            // and CI residuals, apply the current block-diagonal preconditioner
            // to both pieces together. This is still only the first coupled-step
            // increment, but it makes the orbital/CI block structure explicit.
            for (const auto &root : roots)
            {
                Eigen::MatrixXd root_step = Eigen::MatrixXd::Zero(nbasis, nbasis);
                if (root.weight == 0.0)
                {
                    steps.per_root.push_back(std::move(root_step));
                    continue;
                }
                const CoupledStepDirection coupled_step =
                    diagonal_preconditioned_coupled_step(
                    root.g_orb, root.response_residual, root.ci_vector, root.ci_energy,
                    st_base.F_I_mo, root.F_A_mo, st_base.H_CI_diag,
                    n_core, n_act, n_virt,
                    level_shift_local, 0.20, all_mo_irr, use_sym);
                root_step = cap_orbital_step(coupled_step.orbital_step);
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
        double prev_screen_gnorm = std::numeric_limits<double>::infinity();
        double prev_max_root_gnorm = std::numeric_limits<double>::infinity();
        bool converged = false;
        double level_shift = 0.2;
        int rejected_streak = 0;
        int stagnation_streak = 0;
        RootReference root_reference;

        for (unsigned int macro = 1; macro <= as.mcscf_max_iter; ++macro)
        {
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
            const bool g_conv =
                root_resolved_gradient_converged(
                    st_current.weighted_root_gnorm,
                    st_current.max_root_gnorm,
                    as.tol_mcscf_grad);
            const bool no_orb_rot = (st_current.max_root_gnorm == 0.0);
            if ((e_conv && g_conv) || (g_conv && no_orb_rot))
            {
                converged = true;
                break;
            }

            Eigen::MatrixXd kappa_total_weighted = Eigen::MatrixXd::Zero(nbasis, nbasis);
            Eigen::MatrixXd kappa_total = Eigen::MatrixXd::Zero(nbasis, nbasis);
            Eigen::MatrixXd kappa_first = Eigen::MatrixXd::Zero(nbasis, nbasis);
            std::vector<Eigen::MatrixXd> kappa_total_roots(
                st_current.roots.size(), Eigen::MatrixXd::Zero(nbasis, nbasis));
            std::vector<Eigen::MatrixXd> kappa_first_roots;
            Eigen::MatrixXd kappa_newton = Eigen::MatrixXd::Zero(nbasis, nbasis);
            const bool use_numeric_newton_fallback =
                use_numeric_newton_debug || static_cast<int>(opt_pairs.size()) <= numeric_newton_pair_limit;
            if (use_numeric_newton_fallback)
            {
                kappa_newton = build_numeric_newton_step(st_current, C, level_shift, diag);
                if (diag.numeric_newton_attempted && diag.numeric_newton_failed)
                    logging(LogLevel::Warning, tag + " :",
                            "Finite-difference Newton fallback produced an inconsistent column and was discarded.");
            }

            for (unsigned int micro = 0; micro < nmicro; ++micro)
            {
                const RootResolvedOrbitalStepSet kappa_step_set = build_root_resolved_orbital_step_set(
                    st_current.roots, st_current.F_I_mo, nbasis,
                    n_core, n_act, n_virt,
                    level_shift, 0.20, all_mo_irr, use_sym);
                const Eigen::MatrixXd &kappa = kappa_step_set.weighted;
                if (micro == 0)
                {
                    kappa_first = kappa;
                    kappa_first_roots = kappa_step_set.per_root;
                }
                kappa_total_weighted += kappa;
                kappa_total += kappa;
                for (int r = 0; r < static_cast<int>(kappa_total_roots.size()); ++r)
                    kappa_total_roots[static_cast<std::size_t>(r)] +=
                        kappa_step_set.per_root[static_cast<std::size_t>(r)];

                // The orbital trial step perturbs the active-space Hamiltonian,
                // which in turn drives a first-order CI response for each root.
                const int nr_used = static_cast<int>(st_current.roots.size());
                const CISigmaApplier ci_apply = [&](const Eigen::VectorXd &vec, Eigen::VectorXd &sigma_vec)
                {
                    apply_ci_hamiltonian(
                        st_current.ci_space, a_strs, b_strs,
                        st_current.h_eff, st_current.ga, n_act,
                        vec, sigma_vec);
                };

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
                        diag.response_fallback_used = true;

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
            if (max_k > 0.20)
                kappa_total *= 0.20 / max_k;
            const RootResolvedOrbitalStepSet kappa_grad_step_set =
                build_root_resolved_gradient_fallback_step_set(st_current.roots);
            const Eigen::MatrixXd &kappa_grad = kappa_grad_step_set.weighted;
            // The diagonal-preconditioned coupled-step scaffold is kept parked
            // until the candidate screen stops being order-sensitive. Enabling
            // it too early perturbs the stagnation rescue path enough to hide
            // whether a trial truly beats the existing AH/gradient candidates.
            const bool use_coupled_stagnation_rescue = false;
            Eigen::MatrixXd kappa_coupled_total = Eigen::MatrixXd::Zero(nbasis, nbasis);
            std::vector<Eigen::MatrixXd> kappa_coupled_total_roots;
            if (use_coupled_stagnation_rescue && stagnation_streak >= 2)
            {
                const RootResolvedOrbitalStepSet kappa_coupled_correction_set =
                    build_root_resolved_coupled_correction_step_set(st_current, st_current.roots, level_shift);
                kappa_coupled_total =
                    cap_orbital_step(kappa_total_weighted + kappa_coupled_correction_set.weighted);
                kappa_coupled_total_roots.reserve(st_current.roots.size());
                for (int r = 0; r < static_cast<int>(st_current.roots.size()); ++r)
                {
                    Eigen::MatrixXd root_step =
                        kappa_total_roots[static_cast<std::size_t>(r)] +
                        kappa_coupled_correction_set.per_root[static_cast<std::size_t>(r)];
                    kappa_coupled_total_roots.push_back(cap_orbital_step(std::move(root_step)));
                }
            }

            const WeightedRootProbeSignal probe_signal =
                build_weighted_root_probe_signal(st_current.roots, opt_pairs);

            bool accepted = false;
            McscfState accepted_state = st_current;
            double best_E = st_current.E_cas;
            double best_screen_g = st_current.weighted_root_gnorm;
            double best_max_root_g = st_current.max_root_gnorm;
            const double merit_weight = 0.10;
            double best_merit = best_E + merit_weight * best_screen_g * best_screen_g;
            Eigen::MatrixXd C_best = C;
            Eigen::MatrixXd best_step = Eigen::MatrixXd::Zero(nbasis, nbasis);
            struct CandidateStep
            {
                Eigen::MatrixXd step;
                std::string label;
            };
            std::vector<CandidateStep> step_candidates;
            auto append_candidate = [&](Eigen::MatrixXd step, const std::string &label)
            {
                if (step.size() == 0 || step.cwiseAbs().maxCoeff() <= 1e-12)
                    return;
                step_candidates.push_back({std::move(step), label});
            };
            auto append_root_candidates =
                [&](const std::vector<Eigen::MatrixXd> &root_steps, const std::string &base_label, bool cap_steps)
            {
                for (int r = 0; r < static_cast<int>(root_steps.size()); ++r)
                {
                    Eigen::MatrixXd step = root_steps[static_cast<std::size_t>(r)];
                    if (cap_steps)
                        step = cap_orbital_step(std::move(step));
                    append_candidate(std::move(step), std::format("root{:d}-{}", r, base_label));
                }
            };
            append_candidate(kappa_newton, "numeric-newton");
            if (stagnation_streak >= 2)
            {
                append_candidate(kappa_first, "sa-ah-first");
                append_candidate(kappa_total, "sa-ah-total");
                append_candidate(kappa_grad, "sa-grad");
                if (use_coupled_stagnation_rescue)
                    append_candidate(kappa_coupled_total, "sa-coupled-total");
                append_root_candidates(kappa_first_roots, "ah-first", false);
                append_root_candidates(kappa_total_roots, "ah-total", true);
                if (use_coupled_stagnation_rescue)
                    append_root_candidates(kappa_coupled_total_roots, "coupled-total", false);
                append_root_candidates(kappa_grad_step_set.per_root, "grad", false);
                if (kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
                {
                    append_candidate(cap_orbital_step(4.0 * kappa_grad), "sa-grad-x4");
                    append_candidate(cap_orbital_step(2.0 * kappa_grad), "sa-grad-x2");
                    append_candidate(kappa_grad, "sa-grad");
                    append_candidate(cap_orbital_step(0.5 * kappa_grad), "sa-grad-x0.5");
                    append_candidate(cap_orbital_step(-4.0 * kappa_grad), "sa-grad-neg-x4");
                    append_candidate(cap_orbital_step(-2.0 * kappa_grad), "sa-grad-neg-x2");
                    append_candidate(-kappa_grad, "sa-grad-neg");
                    append_candidate(cap_orbital_step(-0.5 * kappa_grad), "sa-grad-neg-x0.5");
                }
                // Large virtual spaces can make the full preconditioned gradient
                // step too entangled: a few productive rotations get mixed with many
                // weak directions and the energy screen rejects the whole update.
                // Probe the dominant pair directions individually so the exact
                // CASSCF energy can pick the useful rotations.
                if (probe_signal.weighted_abs.size() > 0)
                {
                    std::vector<int> ranked_pairs(static_cast<std::size_t>(probe_signal.weighted_abs.size()));
                    std::iota(ranked_pairs.begin(), ranked_pairs.end(), 0);
                    std::partial_sort(
                        ranked_pairs.begin(),
                        ranked_pairs.begin() + std::min<std::size_t>(4, ranked_pairs.size()),
                        ranked_pairs.end(),
                        [&](int lhs, int rhs)
                        {
                            return probe_signal.weighted_abs(lhs) > probe_signal.weighted_abs(rhs);
                        });

                    for (std::size_t i = 0; i < std::min<std::size_t>(4, ranked_pairs.size()); ++i)
                    {
                        const int k = ranked_pairs[i];
                        if (probe_signal.weighted_abs(k) < 1e-6)
                            break;

                        const double signed_probe =
                            (probe_signal.weighted_signed(k) >= 0.0) ? -0.20 : 0.20;
                        append_candidate(build_single_pair_probe_step(k, signed_probe),
                                         std::format("probe-pair{:d}-favored", k));
                        append_candidate(build_single_pair_probe_step(k, -signed_probe),
                                         std::format("probe-pair{:d}-opposite", k));
                    }
                }
            }
            else
            {
                append_candidate(kappa_first, "sa-ah-first");
                append_candidate(kappa_total, "sa-ah-total");
                append_candidate(kappa_grad, "sa-grad");
                append_root_candidates(kappa_first_roots, "ah-first", false);
                append_root_candidates(kappa_total_roots, "ah-total", true);
                append_root_candidates(kappa_grad_step_set.per_root, "grad", false);
                if (kappa_newton.cwiseAbs().maxCoeff() > 1e-12 && kappa_total.cwiseAbs().maxCoeff() > 1e-12)
                    append_candidate(0.5 * (kappa_newton + kappa_total), "mix-newton-total");
                if (kappa_first.cwiseAbs().maxCoeff() > 1e-12 && kappa_total.cwiseAbs().maxCoeff() > 1e-12)
                    append_candidate(0.5 * (kappa_first + kappa_total), "mix-first-total");
                if (kappa_total.cwiseAbs().maxCoeff() > 1e-12 && kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
                    append_candidate(0.5 * (kappa_total + kappa_grad), "mix-total-grad");
                if (kappa_first.cwiseAbs().maxCoeff() > 1e-12 && kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
                    append_candidate(0.5 * (kappa_first + kappa_grad), "mix-first-grad");
                if (kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
                {
                    append_candidate(cap_orbital_step(4.0 * kappa_grad), "sa-grad-x4");
                    append_candidate(cap_orbital_step(2.0 * kappa_grad), "sa-grad-x2");
                    append_candidate(kappa_grad, "sa-grad");
                    append_candidate(cap_orbital_step(0.5 * kappa_grad), "sa-grad-x0.5");
                    append_candidate(cap_orbital_step(-4.0 * kappa_grad), "sa-grad-neg-x4");
                    append_candidate(cap_orbital_step(-2.0 * kappa_grad), "sa-grad-neg-x2");
                    append_candidate(-kappa_grad, "sa-grad-neg");
                    append_candidate(cap_orbital_step(-0.5 * kappa_grad), "sa-grad-neg-x0.5");
                }
            }

            for (const auto &candidate : step_candidates)
            {
                for (double scale : {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625})
                {
                    Eigen::MatrixXd kappa_try = scale * candidate.step;
                    if (kappa_try.cwiseAbs().maxCoeff() < 1e-12)
                        continue;

                    // The approximate AH/response model only proposes candidates.
                    // Acceptance is always decided by a fresh full CASSCF evaluation.
                    auto trial_res = evaluate(apply_orbital_rotation(C, kappa_try, calc._overlap), &root_reference, false);
                    if (!trial_res)
                        continue;

                    const auto &trial = *trial_res;
                    const double trial_merit =
                        trial.E_cas + merit_weight * trial.weighted_root_gnorm * trial.weighted_root_gnorm;
                    const bool merit_improved = trial_merit < best_merit - 1e-10;
                    const bool weighted_root_gradient_reduced =
                        trial.weighted_root_gnorm < best_screen_g - 1e-12;
                    const bool max_root_gradient_reduced =
                        trial.max_root_gnorm < best_max_root_g - 1e-12;
                    const double weighted_root_worsen_window =
                        std::max(0.05 * std::max(best_screen_g, 1e-8), 1e-6);
                    const double max_root_worsen_window =
                        std::max(0.05 * std::max(best_max_root_g, 1e-8), 1e-6);
                    const bool energy_improved = trial.E_cas < best_E - 1e-10;
                    const bool energy_improved_without_hurting_gradient =
                        energy_improved &&
                        trial.weighted_root_gnorm <= best_screen_g + weighted_root_worsen_window &&
                        trial.max_root_gnorm <= best_max_root_g + max_root_worsen_window;
                    const double flat_energy_window = std::max(1000.0 * as.tol_mcscf_energy, 1e-6);
                    const bool stationary_but_better_grad =
                        std::abs(trial.E_cas - best_E) <= flat_energy_window &&
                        (weighted_root_gradient_reduced || max_root_gradient_reduced);
                    if (!energy_improved_without_hurting_gradient &&
                        !merit_improved &&
                        !stationary_but_better_grad)
                        continue;

                    accepted = true;
                    best_E = trial.E_cas;
                    best_screen_g = trial.weighted_root_gnorm;
                    best_max_root_g = trial.max_root_gnorm;
                    best_merit = trial_merit;
                    accepted_state = trial;
                    C_best = apply_orbital_rotation(C, kappa_try, calc._overlap);
                    best_step = kappa_try;
                    diag.accepted_candidate_label =
                        (std::abs(scale - 1.0) < 1e-12)
                            ? candidate.label
                            : std::format("{}@{:.5f}", candidate.label, scale);
                    diag.accepted_weighted_root_gnorm = trial.weighted_root_gnorm;
                    diag.accepted_max_root_gnorm = trial.max_root_gnorm;
                    const WeightedQuadraticModelPrediction prediction =
                        build_weighted_root_quadratic_model_prediction(
                            st_current.roots, st_current.F_I_mo, opt_pairs, best_step);
                    diag.predicted_delta = prediction.weighted_delta;
                    diag.max_root_predicted_delta_deviation = prediction.max_root_deviation;
                    break;
                }
                if (accepted)
                    break;
            }

            if (accepted)
            {
                diag.step_accepted = true;
                diag.accepted_step_norm = best_step.norm();
                diag.actual_delta = best_E - st_current.E_cas;
                C = C_best;
                st_current = std::move(accepted_state);
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
                root_resolved_gradient_progress_flat(
                    reported_screen_gnorm,
                    prev_screen_gnorm,
                    reported_max_root_gnorm,
                    prev_max_root_gnorm);
            const bool accepted_micro_step_plateau =
                diag.step_accepted && diag.accepted_step_norm < 5e-5;
            // Track repeated "flat" iterations separately from hard rejections so
            // we can switch to more exploratory probes before declaring failure.
            if ((!diag.step_accepted && rejected_streak >= 2) || (small_energy_change && little_gradient_progress))
                ++stagnation_streak;
            else
                stagnation_streak = 0;
            prev_screen_gnorm = reported_screen_gnorm;
            prev_max_root_gnorm = reported_max_root_gnorm;

            if (stagnation_streak >= 2)
            {
                level_shift = std::min(50.0, level_shift * 1.5);
                logging(LogLevel::Warning, tag + " :",
                        std::format("Detected stagnation over {:d} macroiterations; increasing damping to {:.3f}.",
                                    stagnation_streak, level_shift));
            }

            HartreeFock::Logger::casscf_iteration(
                macro, st_current.E_cas, dE, reported_screen_gnorm, reported_max_root_gnorm, diag.accepted_step_norm,
                level_shift, 0.0);

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
            if (diag.response_fallback_used)
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
            const bool g_conv_post =
                root_resolved_gradient_converged(
                    reported_screen_gnorm,
                    reported_max_root_gnorm,
                    as.tol_mcscf_grad);
            const bool no_orb_rot_post = (reported_max_root_gnorm == 0.0);
            if ((e_conv_post && g_conv_post) || (g_conv_post && no_orb_rot_post))
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
