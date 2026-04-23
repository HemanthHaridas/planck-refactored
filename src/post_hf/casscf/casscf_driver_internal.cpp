#include "post_hf/casscf/casscf_driver_internal.h"

#include "io/logging.h"

#include <algorithm>
#include <cmath>
#include <format>

namespace
{
    using HartreeFock::Correlation::CASSCFInternal::compute_root_overlap;
    using HartreeFock::Correlation::CASSCFInternal::match_roots_by_max_overlap;
} // namespace

namespace HartreeFock::Correlation::CASSCF
{
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
        std::vector<bool> used_candidate_roots(static_cast<std::size_t>(V.cols()), false);
        int swaps = 0;
        double min_overlap = 1.0;

        for (int i = 0; i < nmatch; ++i)
        {
            const int j = assignment[i];
            if (j < 0 || j >= V.cols())
                continue;
            E_reordered(i) = E(j);
            V_reordered.col(i) = V.col(j);
            used_candidate_roots[static_cast<std::size_t>(j)] = true;
            min_overlap = std::min(min_overlap, std::abs(overlaps(i, j)));
            if (i != j)
                ++swaps;
        }

        int next_slot = nmatch;
        for (int j = 0; j < V.cols() && next_slot < V.cols(); ++j)
        {
            if (used_candidate_roots[static_cast<std::size_t>(j)])
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
            {
                logging(
                    LogLevel::Info,
                    tag + " :",
                    std::format(
                        "Root tracking reordered {:d} CI roots (min |overlap| = {:.3f}).",
                        swaps,
                        min_overlap));
            }
            if (min_overlap < 0.7)
            {
                logging(
                    LogLevel::Warning,
                    tag + " :",
                    std::format(
                        "Root tracking minimum |overlap| dropped to {:.3f}; state identities may be unstable.",
                        min_overlap));
            }
        }
    }

    double compute_max_root_delta(
        const RootReference &previous,
        const Eigen::VectorXd &current)
    {
        if (!previous.valid || previous.energies.size() == 0 || current.size() == 0)
            return 0.0;
        const int n = std::min<int>(previous.energies.size(), current.size());
        double delta = 0.0;
        for (int i = 0; i < n; ++i)
            delta = std::max(delta, std::abs(current(i) - previous.energies(i)));
        return delta;
    }

    void accumulate_weighted_tensor(
        std::vector<double> &destination,
        const std::vector<double> &source,
        double weight)
    {
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
        if (destination.size() == 0)
            destination = Eigen::MatrixXd::Zero(source.rows(), source.cols());
        destination.noalias() += weight * source;
    }

    Eigen::MatrixXd build_root_ci_matrix(
        const std::vector<StateSpecificData> &roots)
    {
        if (roots.empty() || roots.front().ci_vector.size() == 0)
            return Eigen::MatrixXd();

        Eigen::MatrixXd matrix(
            roots.front().ci_vector.size(),
            static_cast<int>(roots.size()));
        for (int r = 0; r < static_cast<int>(roots.size()); ++r)
            matrix.col(r) = roots[static_cast<std::size_t>(r)].ci_vector;
        return matrix;
    }

    Eigen::VectorXd build_root_energy_vector(
        const std::vector<StateSpecificData> &roots)
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

    bool sa_gradient_converged(
        double sa_gnorm,
        double tol)
    {
        return sa_gnorm < tol;
    }

    bool sa_gradient_progress_flat(
        double sa_gnorm,
        double prev_sa_gnorm)
    {
        if (!std::isfinite(prev_sa_gnorm))
            return false;

        const double window =
            std::max(0.05 * std::max(prev_sa_gnorm, 1e-8), 1e-8);
        return std::abs(sa_gnorm - prev_sa_gnorm) < window;
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
        {
            prediction.max_root_deviation =
                std::max(prediction.max_root_deviation, std::abs(delta - prediction.weighted_delta));
        }

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

        for (const auto &root : roots)
        {
            Eigen::MatrixXd step = Eigen::MatrixXd::Zero(nbasis, nbasis);
            if (root.weight == 0.0)
            {
                steps.per_root.push_back(std::move(step));
                continue;
            }

            step = augmented_hessian_step(
                root.g_orb,
                F_I_mo,
                root.F_A_mo,
                n_core,
                n_act,
                n_virt,
                level_shift,
                max_rot,
                mo_irreps,
                use_sym);
            steps.weighted.noalias() += root.weight * step;
            steps.per_root.push_back(std::move(step));
        }
        return steps;
    }

    void append_candidate_step(
        std::vector<CandidateStep> &candidates,
        Eigen::MatrixXd step,
        const std::string &label)
    {
        if (step.size() == 0 || step.cwiseAbs().maxCoeff() <= 1e-12)
            return;
        candidates.push_back({std::move(step), label});
    }

    void append_root_candidate_steps(
        std::vector<CandidateStep> &candidates,
        const std::vector<Eigen::MatrixXd> &root_steps,
        const std::string &base_label,
        bool cap_steps)
    {
        const HartreeFock::index_t root_count =
            static_cast<HartreeFock::index_t>(root_steps.size());
        for (HartreeFock::index_t root = 0; root < root_count; ++root)
        {
            Eigen::MatrixXd step = root_steps[static_cast<std::size_t>(root)];
            if (cap_steps)
            {
                const double max_elem = step.cwiseAbs().maxCoeff();
                if (max_elem > 0.20)
                    step *= 0.20 / max_elem;

                const double trust_radius = 0.80;
                const double frob = step.norm();
                if (frob > trust_radius)
                    step *= trust_radius / frob;
            }
            append_candidate_step(
                candidates,
                std::move(step),
                std::format("root{:d}-{}", root, base_label));
        }
    }
} // namespace HartreeFock::Correlation::CASSCF
