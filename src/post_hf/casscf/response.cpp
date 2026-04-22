#include "post_hf/casscf/response.h"

#include "base/tables.h"
#include "post_hf/casscf/casscf_utils.h"
#include "post_hf/casscf/ci.h"
#include "post_hf/casscf/orbital.h"
#include "post_hf/casscf/rdm.h"
#include "post_hf/casscf/strings.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>

namespace
{

    using HartreeFock::Correlation::CASSCF::as_single_column_matrix;
    using HartreeFock::Correlation::CASSCF::build_ci_orbital_gradient_correction;
    using HartreeFock::Correlation::CASSCF::build_det_lookup;
    using HartreeFock::Correlation::CASSCF::build_spin_dets;
    using HartreeFock::Correlation::CASSCF::CIDeterminantSpace;
    using HartreeFock::Correlation::CASSCF::CISigmaApplier;
    using HartreeFock::Correlation::CASSCF::compute_2rdm_bilinear;
    using HartreeFock::Correlation::CASSCF::compute_Q_matrix;
    using HartreeFock::Correlation::CASSCF::count_occupied_below;
    using HartreeFock::Correlation::CASSCF::delta_g_sa_action;
    using HartreeFock::Correlation::CASSCF::hess_diag;
    using HartreeFock::Correlation::CASSCF::hessian_action;
    using HartreeFock::Correlation::CASSCF::non_redundant_pairs;
    using HartreeFock::Correlation::CASSCF::OrbitalHessianContext;
    using HartreeFock::Correlation::CASSCF::ResponseRHSMode;
    using HartreeFock::Correlation::CASSCF::RotPair;
    using HartreeFock::Correlation::CASSCF::single_weight;
    using HartreeFock::Correlation::CASSCF::StateAveragedCoupledRoot;
    using HartreeFock::Correlation::CASSCFInternal::ActiveIntegralCache;
    using HartreeFock::Correlation::CASSCFInternal::apply_response_diag_preconditioner;
    using HartreeFock::Correlation::CASSCFInternal::CIString;
    using HartreeFock::Correlation::CASSCFInternal::project_orthogonal;
    using HartreeFock::Correlation::CASSCFInternal::single_bit_mask;

    std::size_t idx4(int p, int q, int r, int s, int n_act)
    {
        return static_cast<std::size_t>(((p * n_act + q) * n_act + r) * n_act + s);
    }

    Eigen::MatrixXd exact_active_one_body_derivative(
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        int n_core,
        int n_act)
    {
        if (n_act <= 0 ||
            kappa.rows() < n_core + n_act ||
            kappa.cols() < n_core + n_act ||
            F_I_mo.rows() < n_core + n_act ||
            F_I_mo.cols() < n_core + n_act)
            return Eigen::MatrixXd::Zero(std::max(n_act, 0), std::max(n_act, 0));

        // The active-space one-electron Hamiltonian is the active-active block
        // of the inactive Fock matrix in the current MO basis. Rotating the full
        // MO basis changes that block even for core-active and active-virtual
        // directions, so the "exact" RHS must differentiate the full matrix
        // before projecting back to the active block.
        const Eigen::MatrixXd dF = F_I_mo * kappa - kappa * F_I_mo;
        return dF.block(n_core, n_core, n_act, n_act);
    }

    std::vector<double> exact_active_two_body_derivative_active_only(
        const Eigen::MatrixXd &kappa_act,
        const std::vector<double> &ga,
        int n_act)
    {
        std::vector<double> dga(ga.size(), 0.0);
        if (n_act <= 0 || ga.empty())
            return dga;

        for (int p = 0; p < n_act; ++p)
            for (int q = 0; q < n_act; ++q)
                for (int r = 0; r < n_act; ++r)
                    for (int s = 0; s < n_act; ++s)
                    {
                        double value = 0.0;
                        for (int t = 0; t < n_act; ++t)
                        {
                            value += kappa_act(t, p) * ga[idx4(t, q, r, s, n_act)];
                            value += kappa_act(t, q) * ga[idx4(p, t, r, s, n_act)];
                            value += kappa_act(t, r) * ga[idx4(p, q, t, s, n_act)];
                            value += kappa_act(t, s) * ga[idx4(p, q, r, t, n_act)];
                        }
                        dga[idx4(p, q, r, s, n_act)] = value;
                    }
        return dga;
    }

    std::size_t idx4_puvw(int p, int u, int v, int w, int n_act)
    {
        return static_cast<std::size_t>(((p * n_act + u) * n_act + v) * n_act + w);
    }

    double cached_puvw_value(
        const ActiveIntegralCache &active_integrals,
        int p,
        int u,
        int v,
        int w)
    {
        if (!active_integrals.valid ||
            p < 0 ||
            p >= active_integrals.nbasis ||
            u < 0 ||
            u >= active_integrals.nact ||
            v < 0 ||
            v >= active_integrals.nact ||
            w < 0 ||
            w >= active_integrals.nact)
            return 0.0;
        return active_integrals.puvw[idx4_puvw(p, u, v, w, active_integrals.nact)];
    }

    std::vector<double> exact_active_two_body_derivative(
        const Eigen::MatrixXd &kappa,
        const std::vector<double> &ga,
        const ActiveIntegralCache &active_integrals,
        int n_core,
        int n_act)
    {
        if (n_act <= 0)
            return {};

        std::vector<double> dga_active_only;
        if (kappa.rows() >= n_core + n_act && kappa.cols() >= n_core + n_act)
            dga_active_only = exact_active_two_body_derivative_active_only(
                kappa.block(n_core, n_core, n_act, n_act), ga, n_act);
        else
            dga_active_only = std::vector<double>(ga.size(), 0.0);

        const bool have_full_cache =
            active_integrals.valid &&
            active_integrals.nact == n_act &&
            active_integrals.nbasis > 0 &&
            active_integrals.puvw.size() ==
                static_cast<std::size_t>(active_integrals.nbasis) * n_act * n_act * n_act &&
            kappa.rows() >= active_integrals.nbasis &&
            kappa.cols() >= active_integrals.nbasis;

        if (!have_full_cache)
            return dga_active_only;

        std::vector<double> dga = dga_active_only;
        for (int t = 0; t < n_act; ++t)
            for (int u = 0; u < n_act; ++u)
                for (int v = 0; v < n_act; ++v)
                    for (int w = 0; w < n_act; ++w)
                    {
                        const int gt = n_core + t;
                        const int gu = n_core + u;
                        const int gv = n_core + v;
                        const int gw = n_core + w;
                        double value = 0.0;
                        for (int p = 0; p < active_integrals.nbasis; ++p)
                        {
                            if (p >= n_core && p < n_core + n_act)
                                continue;
                            value += kappa(p, gt) * cached_puvw_value(active_integrals, p, u, v, w);
                            value += kappa(p, gu) * cached_puvw_value(active_integrals, p, t, v, w);
                            value += kappa(p, gv) * cached_puvw_value(active_integrals, p, w, t, u);
                            value += kappa(p, gw) * cached_puvw_value(active_integrals, p, v, t, u);
                        }
                        dga[idx4(t, u, v, w, n_act)] += value;
                    }
        return dga;
    }

    Eigen::VectorXd response_residual(
        const HartreeFock::Correlation::CASSCF::CISigmaApplier &apply,
        const Eigen::VectorXd &c1,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &sigma)
    {
        Eigen::VectorXd hc1(c1.size());
        apply(c1, hc1);
        // The residual is the projected linearized response equation:
        // (H - E0) c1 + Q sigma = 0, with Q enforcing orthogonality to c0.
        const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
        return project_orthogonal(rhs - (hc1 - E0 * c1), c0);
    }

    struct CoupledResidualEvaluation
    {
        Eigen::MatrixXd orbital_residual;
        Eigen::VectorXd ci_residual;
        Eigen::MatrixXd orbital_correction;
    };

    struct SACoupledResidualEvaluation
    {
        Eigen::MatrixXd orbital_residual;
        std::vector<Eigen::VectorXd> ci_residuals;
        Eigen::MatrixXd orbital_correction;
    };

    struct OrbitalLinearOperator
    {
        std::vector<RotPair> pairs;
        Eigen::MatrixXd matrix;
        int nbasis = 0;
        bool available = false;
    };

    std::vector<RotPair> symmetry_allowed_pairs(
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<int> &mo_irreps,
        bool use_sym)
    {
        std::vector<RotPair> pairs;
        for (const auto &pair : non_redundant_pairs(n_core, n_act, n_virt))
        {
            if (use_sym && !mo_irreps.empty())
            {
                const int ip = (pair.p < static_cast<int>(mo_irreps.size())) ? mo_irreps[pair.p] : -1;
                const int iq = (pair.q < static_cast<int>(mo_irreps.size())) ? mo_irreps[pair.q] : -1;
                if (ip >= 0 && iq >= 0 && ip != iq)
                    continue;
            }
            pairs.push_back(pair);
        }
        return pairs;
    }

    Eigen::VectorXd pack_orbital_pairs(
        const Eigen::MatrixXd &M,
        const std::vector<RotPair> &pairs)
    {
        const HartreeFock::index_t pair_count = static_cast<HartreeFock::index_t>(pairs.size());
        Eigen::VectorXd packed = Eigen::VectorXd::Zero(pair_count);
        for (HartreeFock::index_t i = 0; i < pair_count; ++i)
            packed(i) = M(pairs[static_cast<std::size_t>(i)].p, pairs[static_cast<std::size_t>(i)].q);
        return packed;
    }

    Eigen::MatrixXd unpack_orbital_pairs(
        const Eigen::VectorXd &packed,
        const std::vector<RotPair> &pairs,
        int nbasis)
    {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbasis, nbasis);
        const HartreeFock::index_t pair_count = static_cast<HartreeFock::index_t>(pairs.size());
        for (HartreeFock::index_t i = 0; i < pair_count && i < packed.size(); ++i)
        {
            const auto &pair = pairs[static_cast<std::size_t>(i)];
            M(pair.p, pair.q) = packed(i);
            M(pair.q, pair.p) = -packed(i);
        }
        return M;
    }

    OrbitalLinearOperator build_orbital_linear_operator(
        const OrbitalHessianContext *orbital_hessian_ctx,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        const std::vector<int> &mo_irreps,
        bool use_sym)
    {
        constexpr int max_dense_pairs = 128;

        OrbitalLinearOperator op;
        op.pairs = symmetry_allowed_pairs(n_core, n_act, n_virt, mo_irreps, use_sym);
        op.nbasis = nbasis;

        if (orbital_hessian_ctx == nullptr || op.pairs.empty() || static_cast<int>(op.pairs.size()) > max_dense_pairs)
            return op;

        const HartreeFock::index_t npairs = static_cast<HartreeFock::index_t>(op.pairs.size());
        op.matrix = Eigen::MatrixXd::Zero(npairs, npairs);
        for (HartreeFock::index_t col = 0; col < npairs; ++col)
        {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(npairs);
            e(col) = 1.0;
            const Eigen::MatrixXd trial = unpack_orbital_pairs(e, op.pairs, nbasis);
            Eigen::MatrixXd action = delta_g_sa_action(
                trial, orbital_hessian_ctx, F_I_mo, F_A_mo,
                n_core, n_act, n_virt, mo_irreps, use_sym);
            action.noalias() += level_shift * trial;
            op.matrix.col(col) = pack_orbital_pairs(action, op.pairs);
        }

        op.matrix = 0.5 * (op.matrix + op.matrix.transpose());
        op.available = op.matrix.allFinite();
        return op;
    }

    Eigen::VectorXd apply_orbital_linear_operator_packed(
        const Eigen::VectorXd &x,
        const OrbitalLinearOperator &op,
        const OrbitalHessianContext *orbital_hessian_ctx,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        const std::vector<int> &mo_irreps,
        bool use_sym)
    {
        if (x.size() == 0 || op.pairs.empty())
            return Eigen::VectorXd::Zero(static_cast<HartreeFock::index_t>(op.pairs.size()));
        if (op.available && op.matrix.rows() == x.size())
            return op.matrix * x;

        const Eigen::MatrixXd trial = unpack_orbital_pairs(x, op.pairs, op.nbasis);
        Eigen::MatrixXd action = delta_g_sa_action(
            trial, orbital_hessian_ctx, F_I_mo, F_A_mo,
            n_core, n_act, n_virt, mo_irreps, use_sym);
        action.noalias() += level_shift * trial;
        return pack_orbital_pairs(action, op.pairs);
    }

    Eigen::VectorXd orbital_pair_preconditioner(
        const std::vector<RotPair> &pairs,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        double level_shift)
    {
        const Eigen::MatrixXd F_sum = F_I_mo + F_A_mo;
        const HartreeFock::index_t pair_count = static_cast<HartreeFock::index_t>(pairs.size());
        Eigen::VectorXd denom = Eigen::VectorXd::Ones(pair_count);
        for (HartreeFock::index_t k = 0; k < pair_count; ++k)
        {
            const auto &pair = pairs[static_cast<std::size_t>(k)];
            denom(k) = hess_diag(F_sum, pair.p, pair.q) + level_shift;
            if (std::abs(denom(k)) < 1e-4)
                denom(k) = (denom(k) >= 0.0) ? 1e-4 : -1e-4;
        }
        return denom;
    }

    Eigen::MatrixXd solve_orbital_action_system(
        const Eigen::MatrixXd &orbital_residual,
        const OrbitalLinearOperator &op,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        const OrbitalHessianContext *orbital_hessian_ctx)
    {
        if (op.pairs.empty())
            return Eigen::MatrixXd::Zero(op.nbasis, op.nbasis);

        if ((!op.available || op.matrix.rows() == 0) && orbital_hessian_ctx == nullptr)
            return HartreeFock::Correlation::CASSCF::diagonal_preconditioned_orbital_step(
                orbital_residual,
                F_I_mo,
                F_A_mo,
                n_core,
                n_act,
                n_virt,
                level_shift,
                max_rot,
                mo_irreps,
                use_sym);

        const Eigen::VectorXd g = pack_orbital_pairs(orbital_residual, op.pairs);
        if (g.size() == 0 || g.cwiseAbs().maxCoeff() < 1e-12)
            return Eigen::MatrixXd::Zero(op.nbasis, op.nbasis);

        auto cap_packed_step = [&](Eigen::VectorXd x)
        {
            if (!x.allFinite() || x.size() == 0)
                return x;
            const double max_x = x.cwiseAbs().maxCoeff();
            if (max_x > max_rot)
                x *= max_rot / max_x;
            return x;
        };

        Eigen::VectorXd step = Eigen::VectorXd::Zero(g.size());
        if (op.available && op.matrix.rows() == g.size())
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(op.matrix);
            if (eig.info() != Eigen::Success)
                return HartreeFock::Correlation::CASSCF::diagonal_preconditioned_orbital_step(
                    orbital_residual,
                    F_I_mo,
                    F_A_mo,
                    n_core,
                    n_act,
                    n_virt,
                    level_shift,
                    max_rot,
                    mo_irreps,
                    use_sym);

            Eigen::VectorXd evals = eig.eigenvalues();
            for (int i = 0; i < evals.size(); ++i)
                evals(i) = std::max(evals(i), 1e-4);
            step =
                -eig.eigenvectors() * evals.cwiseInverse().asDiagonal() * eig.eigenvectors().transpose() * g;
            step = cap_packed_step(std::move(step));
        }
        else
        {
            const Eigen::MatrixXd diagonal_guess =
                HartreeFock::Correlation::CASSCF::diagonal_preconditioned_orbital_step(
                    orbital_residual,
                    F_I_mo,
                    F_A_mo,
                    n_core,
                    n_act,
                    n_virt,
                    level_shift,
                    max_rot,
                    mo_irreps,
                    use_sym);
            step = cap_packed_step(pack_orbital_pairs(diagonal_guess, op.pairs));

            const Eigen::VectorXd denom =
                orbital_pair_preconditioner(op.pairs, F_I_mo, F_A_mo, level_shift);
            auto residual = [&](const Eigen::VectorXd &trial) -> Eigen::VectorXd
            {
                Eigen::VectorXd applied = apply_orbital_linear_operator_packed(
                    trial,
                    op,
                    orbital_hessian_ctx,
                    F_I_mo,
                    F_A_mo,
                    n_core,
                    n_act,
                    n_virt,
                    level_shift,
                    mo_irreps,
                    use_sym);
                return -g - applied;
            };

            Eigen::VectorXd best_r = residual(step);
            double best_norm = best_r.norm();
            if (!std::isfinite(best_norm))
                return diagonal_guess;

            for (int iter = 0; iter < 6; ++iter)
            {
                if (best_norm < 1e-8)
                    break;

                Eigen::VectorXd correction = best_r;
                for (int k = 0; k < correction.size(); ++k)
                    correction(k) /= denom(k);
                if (!correction.allFinite() || correction.cwiseAbs().maxCoeff() < 1e-12)
                    break;

                bool improved = false;
                Eigen::VectorXd best_trial = step;
                Eigen::VectorXd best_trial_r = best_r;
                double best_trial_norm = best_norm;
                for (double scale : CASSCF_PROBE_STEP_SCALES)
                {
                    const Eigen::VectorXd trial = cap_packed_step(step + scale * correction);
                    const Eigen::VectorXd trial_r = residual(trial);
                    const double trial_norm = trial_r.norm();
                    if (!std::isfinite(trial_norm))
                        continue;
                    if (trial_norm < best_trial_norm - 1e-12)
                    {
                        improved = true;
                        best_trial = trial;
                        best_trial_r = trial_r;
                        best_trial_norm = trial_norm;
                    }
                }

                if (!improved)
                    break;

                step = std::move(best_trial);
                best_r = std::move(best_trial_r);
                best_norm = best_trial_norm;
            }
        }

        Eigen::MatrixXd kappa = unpack_orbital_pairs(step, op.pairs, op.nbasis);
        const double max_elem = kappa.cwiseAbs().maxCoeff();
        if (max_elem > max_rot)
            kappa *= max_rot / max_elem;

        const double trust_radius = 0.80;
        const double frob = kappa.norm();
        if (frob > trust_radius)
            kappa *= trust_radius / frob;
        return kappa;
    }

    Eigen::MatrixXd orbital_correction_from_ci_step(
        const Eigen::VectorXd &c1,
        const Eigen::VectorXd &c0,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt)
    {
        if (c1.size() == 0 || c0.size() == 0)
            return Eigen::MatrixXd::Zero(nbasis, nbasis);

        const Eigen::MatrixXd c1_vec = as_single_column_matrix(c1);
        const Eigen::MatrixXd c0_vec = as_single_column_matrix(c0);
        const auto Gamma1_r = compute_2rdm_bilinear(
            c1_vec, c0_vec, single_weight(1.0),
            a_strs, b_strs, dets, n_act);
        const auto Gamma1_rt = compute_2rdm_bilinear(
            c0_vec, c1_vec, single_weight(1.0),
            a_strs, b_strs, dets, n_act);

        std::vector<double> Gamma1_vec(Gamma1_r.size(), 0.0);
        for (std::size_t i = 0; i < Gamma1_r.size(); ++i)
            Gamma1_vec[i] = Gamma1_r[i] + Gamma1_rt[i];

        const Eigen::MatrixXd Q1 = compute_Q_matrix(active_integrals, Gamma1_vec);
        return build_ci_orbital_gradient_correction(Q1, nbasis, n_core, n_act, n_virt);
    }

    CoupledResidualEvaluation evaluate_coupled_residual(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &orbital_gradient,
        const Eigen::MatrixXd &kappa,
        const Eigen::VectorXd &c1,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        const OrbitalHessianContext *orbital_hessian_ctx)
    {
        CoupledResidualEvaluation evaluation;
        evaluation.orbital_correction = orbital_correction_from_ci_step(
            c1, c0, a_strs, b_strs, dets, active_integrals, nbasis, n_core, n_act, n_virt);
        evaluation.orbital_residual =
            orbital_gradient +
            delta_g_sa_action(
                kappa, orbital_hessian_ctx, F_I_mo, F_A_mo,
                n_core, n_act, n_virt, mo_irreps, use_sym) +
            evaluation.orbital_correction;
        const Eigen::VectorXd rhs = build_ci_response_rhs(
            mode, kappa, F_I_mo, h_eff, ga, active_integrals,
            space, a_strs, b_strs, c0, n_core, n_act);
        evaluation.ci_residual = response_residual(apply, c1, c0, E0, rhs);
        return evaluation;
    }

    double max_ci_residual_norm(
        const std::vector<Eigen::VectorXd> &residuals,
        const std::vector<StateAveragedCoupledRoot> &roots)
    {
        double max_norm = 0.0;
        const int n = std::min<int>(residuals.size(), roots.size());
        for (int i = 0; i < n; ++i)
        {
            if (roots[static_cast<std::size_t>(i)].weight == 0.0)
                continue;
            max_norm = std::max(max_norm, residuals[static_cast<std::size_t>(i)].norm());
        }
        return max_norm;
    }

    SACoupledResidualEvaluation evaluate_sa_coupled_residual(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &orbital_gradient,
        const Eigen::MatrixXd &kappa,
        const std::vector<Eigen::VectorXd> &ci_steps,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const std::vector<StateAveragedCoupledRoot> &roots,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        const OrbitalHessianContext *orbital_hessian_ctx)
    {
        SACoupledResidualEvaluation evaluation;
        evaluation.orbital_correction = Eigen::MatrixXd::Zero(nbasis, nbasis);
        evaluation.orbital_residual =
            orbital_gradient +
            delta_g_sa_action(
                kappa, orbital_hessian_ctx, F_I_mo, F_A_mo,
                n_core, n_act, n_virt, mo_irreps, use_sym);
        evaluation.ci_residuals.reserve(roots.size());

        for (int i = 0; i < static_cast<int>(roots.size()); ++i)
        {
            const auto &root = roots[static_cast<std::size_t>(i)];
            const Eigen::VectorXd empty = Eigen::VectorXd::Zero(root.ci_vector.size());
            if (root.weight == 0.0 || root.ci_vector.size() == 0)
            {
                evaluation.ci_residuals.push_back(empty);
                continue;
            }

            const Eigen::VectorXd &c1 =
                (i < static_cast<int>(ci_steps.size())) ? ci_steps[static_cast<std::size_t>(i)] : empty;
            const Eigen::MatrixXd root_orbital_correction = orbital_correction_from_ci_step(
                c1, root.ci_vector, a_strs, b_strs, dets, active_integrals, nbasis, n_core, n_act, n_virt);
            evaluation.orbital_correction.noalias() += root.weight * root_orbital_correction;

            const Eigen::VectorXd rhs = build_ci_response_rhs(
                mode, kappa, F_I_mo, h_eff, ga, active_integrals,
                space, a_strs, b_strs, root.ci_vector, n_core, n_act);
            evaluation.ci_residuals.push_back(
                response_residual(apply, c1, root.ci_vector, root.ci_energy, rhs));
        }

        evaluation.orbital_residual += evaluation.orbital_correction;
        return evaluation;
    }

} // namespace

namespace HartreeFock::Correlation::CASSCF
{

    const char *response_mode_name(ResponseMode mode)
    {
        switch (mode)
        {
        case ResponseMode::ApproximatePrototype:
            return "approximate prototype";
        case ResponseMode::DiagonalResponse:
            return "diagonal-orbital-plus-CI-response approximation";
        case ResponseMode::CoupledSecondOrderTarget:
            return "matrix-free coupled orbital/CI solve";
        }
        return "unknown";
    }

    const char *response_rhs_mode_name(ResponseRHSMode mode)
    {
        switch (mode)
        {
        case ResponseRHSMode::CommutatorOnlyApproximate:
            return "commutator-only approximate RHS";
        case ResponseRHSMode::ExactActiveSpaceOrbitalDerivative:
            return "exact active-space orbital derivative RHS";
        }
        return "unknown";
    }

    Eigen::VectorXd ci_sigma_1body(
        const Eigen::MatrixXd &dh,
        const Eigen::VectorXd &c,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act)
    {
        const int dim = static_cast<int>(c.size());
        const auto sd = build_spin_dets(a_strs, b_strs, dets, n_act);
        const auto lut = build_det_lookup(sd);
        Eigen::VectorXd sigma = Eigen::VectorXd::Zero(dim);

        for (int j = 0; j < dim; ++j)
        {
            const double cJ = c(j);
            if (std::abs(cJ) < 1e-15)
                continue;
            const auto ket = sd[j];
            // Match slater_condon_element(): use the ket->bra convention
            // for a_p^\dagger a_q, i.e. annihilate an occupied q in the ket
            // and create the corresponding bra orbital p with coefficient dh(p, q).
            for (int q_so = 0; q_so < 2 * n_act; ++q_so)
            {
                auto ann = apply_annihilation(ket, q_so);
                if (!ann.valid)
                    continue;
                const int spin_offset = (q_so >= n_act) ? n_act : 0;
                const int q = q_so - spin_offset;
                for (int p = 0; p < n_act; ++p)
                {
                    if (std::abs(dh(p, q)) < 1e-18)
                        continue;
                    auto cre = apply_creation(ann.det, spin_offset + p);
                    if (!cre.valid)
                        continue;
                    auto it = lut.find(cre.det);
                    if (it == lut.end())
                        continue;
                    sigma(it->second) += dh(p, q) * ann.phase * cre.phase * cJ;
                }
            }
        }

        return sigma;
    }

    Eigen::MatrixXd delta_h_eff(
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        int n_core,
        int n_act)
    {
        // This is the current approximate shortcut: keep only the active block
        // of the commutator between the orbital rotation and the inactive Fock.
        Eigen::MatrixXd comm = kappa * F_I_mo - F_I_mo * kappa;
        return comm.block(n_core, n_core, n_act, n_act);
    }

    Eigen::VectorXd build_ci_response_rhs(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const ActiveIntegralCache &active_integrals,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const Eigen::VectorXd &c0,
        int n_core,
        int n_act)
    {
        if (c0.size() == 0 || space.dets.empty())
            return Eigen::VectorXd::Zero(c0.size());

        if (mode == ResponseRHSMode::CommutatorOnlyApproximate)
        {
            const Eigen::MatrixXd dh = delta_h_eff(kappa, F_I_mo, n_core, n_act);
            return ci_sigma_1body(dh, c0, a_strs, b_strs, space.dets, n_act);
        }

        if (kappa.rows() < n_core + n_act ||
            kappa.cols() < n_core + n_act ||
            F_I_mo.rows() < n_core + n_act ||
            F_I_mo.cols() < n_core + n_act)
            return Eigen::VectorXd::Zero(c0.size());

        const Eigen::MatrixXd dh =
            exact_active_one_body_derivative(kappa, F_I_mo, n_core, n_act);
        const std::vector<double> dga =
            exact_active_two_body_derivative(kappa, ga, active_integrals, n_core, n_act);
        const Eigen::MatrixXd dH = build_ci_hamiltonian_dense(a_strs, b_strs, space.dets, dh, dga, n_act);
        return dH * c0;
    }

    CoupledStepSolveResult solve_coupled_orbital_ci_step(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &orbital_gradient,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        const OrbitalHessianContext *orbital_hessian_ctx,
        double tol,
        int max_iter,
        double response_precond_floor)
    {
        CoupledStepSolveResult result;
        result.orbital_step = Eigen::MatrixXd::Zero(nbasis, nbasis);
        result.ci_step = Eigen::VectorXd::Zero(c0.size());
        result.orbital_residual = orbital_gradient;
        result.ci_residual = Eigen::VectorXd::Zero(c0.size());
        result.orbital_correction = Eigen::MatrixXd::Zero(nbasis, nbasis);
        result.orbital_residual_max = orbital_gradient.cwiseAbs().maxCoeff();

        if (c0.size() == 0 || H_diag.size() != c0.size())
            return result;

        auto residual_metric = [](const CoupledResidualEvaluation &evaluation)
        {
            return std::max(
                evaluation.orbital_residual.cwiseAbs().maxCoeff(),
                evaluation.ci_residual.norm());
        };

        const OrbitalLinearOperator orbital_operator = build_orbital_linear_operator(
            orbital_hessian_ctx, F_I_mo, F_A_mo, nbasis, n_core, n_act, n_virt,
            level_shift, mo_irreps, use_sym);

        // Seed the coupled iteration with one damped Newton solve in the
        // pair-compressed orbital space, using the matrix-free Hessian action
        // when available and falling back to the diagonal model otherwise.
        result.orbital_step = solve_orbital_action_system(
            orbital_gradient,
            orbital_operator,
            F_I_mo,
            F_A_mo,
            n_core,
            n_act,
            n_virt,
            level_shift,
            max_rot,
            mo_irreps,
            use_sym,
            orbital_hessian_ctx);
        const CoupledResponseBlocks seed_blocks = build_coupled_response_blocks(
            mode,
            result.orbital_step,
            F_I_mo,
            h_eff,
            ga,
            space,
            a_strs,
            b_strs,
            dets,
            active_integrals,
            apply,
            c0,
            E0,
            H_diag,
            nbasis,
            n_core,
            n_act,
            n_virt);
        result.ci_step = seed_blocks.ci_response.c1;

        CoupledResidualEvaluation current = evaluate_coupled_residual(
            mode, orbital_gradient, result.orbital_step, result.ci_step,
            F_I_mo, F_A_mo, h_eff, ga, space, a_strs, b_strs, dets,
            active_integrals, apply, c0, E0, nbasis, n_core, n_act, n_virt,
            mo_irreps, use_sym, orbital_hessian_ctx);
        double current_metric = residual_metric(current);

        for (int iter = 1; iter <= max_iter; ++iter)
        {
            result.iterations = iter;
            if (!std::isfinite(current_metric))
                break;
            if (current_metric < tol)
            {
                result.converged = true;
                break;
            }

            CoupledStepDirection correction = diagonal_preconditioned_coupled_step(
                current.orbital_residual,
                current.ci_residual,
                c0,
                E0,
                F_I_mo,
                F_A_mo,
                H_diag,
                n_core,
                n_act,
                n_virt,
                level_shift,
                max_rot,
                mo_irreps,
                use_sym,
                response_precond_floor);
            correction.orbital_step = solve_orbital_action_system(
                current.orbital_residual,
                orbital_operator,
                F_I_mo,
                F_A_mo,
                n_core,
                n_act,
                n_virt,
                level_shift,
                max_rot,
                mo_irreps,
                use_sym,
                orbital_hessian_ctx);

            bool accepted_update = false;
            CoupledResidualEvaluation best_evaluation = current;
            Eigen::MatrixXd best_orbital_step = result.orbital_step;
            Eigen::VectorXd best_ci_step = result.ci_step;
            double best_metric = current_metric;

            for (double scale : CASSCF_PROBE_STEP_SCALES)
            {
                Eigen::MatrixXd trial_orbital_step =
                    result.orbital_step + scale * correction.orbital_step;
                const double max_elem = trial_orbital_step.cwiseAbs().maxCoeff();
                if (max_elem > max_rot)
                    trial_orbital_step *= max_rot / max_elem;

                Eigen::VectorXd trial_ci_step =
                    project_orthogonal(result.ci_step + scale * correction.ci_step, c0);
                CoupledResidualEvaluation trial = evaluate_coupled_residual(
                    mode, orbital_gradient, trial_orbital_step, trial_ci_step,
                    F_I_mo, F_A_mo, h_eff, ga, space, a_strs, b_strs, dets,
                    active_integrals, apply, c0, E0, nbasis, n_core, n_act, n_virt,
                    mo_irreps, use_sym, orbital_hessian_ctx);
                const double trial_metric = residual_metric(trial);
                if (!std::isfinite(trial_metric))
                    continue;
                if (trial_metric < best_metric - 1e-12)
                {
                    accepted_update = true;
                    best_metric = trial_metric;
                    best_evaluation = std::move(trial);
                    best_orbital_step = std::move(trial_orbital_step);
                    best_ci_step = std::move(trial_ci_step);
                }
            }

            if (!accepted_update)
                break;

            result.orbital_step = std::move(best_orbital_step);
            result.ci_step = std::move(best_ci_step);
            current = std::move(best_evaluation);
            current_metric = best_metric;
        }

        result.orbital_residual = std::move(current.orbital_residual);
        result.ci_residual = std::move(current.ci_residual);
        result.orbital_correction = std::move(current.orbital_correction);
        result.orbital_residual_max = result.orbital_residual.cwiseAbs().maxCoeff();
        result.ci_residual_norm = result.ci_residual.norm();
        return result;
    }

    SACoupledStepSolveResult solve_sa_coupled_orbital_ci_step(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &orbital_gradient,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const std::vector<StateAveragedCoupledRoot> &roots,
        const Eigen::VectorXd &H_diag,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        const OrbitalHessianContext *orbital_hessian_ctx,
        double tol,
        int max_iter,
        double response_precond_floor)
    {
        SACoupledStepSolveResult result;
        result.orbital_step = Eigen::MatrixXd::Zero(nbasis, nbasis);
        result.ci_steps.reserve(roots.size());
        result.ci_residuals.reserve(roots.size());
        for (const auto &root : roots)
        {
            result.ci_steps.push_back(Eigen::VectorXd::Zero(root.ci_vector.size()));
            result.ci_residuals.push_back(Eigen::VectorXd::Zero(root.ci_vector.size()));
        }
        result.orbital_residual = orbital_gradient;
        result.orbital_correction = Eigen::MatrixXd::Zero(nbasis, nbasis);
        result.orbital_residual_max = orbital_gradient.cwiseAbs().maxCoeff();

        if (roots.empty())
            return result;

        auto residual_metric =
            [&](const SACoupledResidualEvaluation &evaluation)
        {
            return std::max(
                evaluation.orbital_residual.cwiseAbs().maxCoeff(),
                max_ci_residual_norm(evaluation.ci_residuals, roots));
        };

        const OrbitalLinearOperator orbital_operator = build_orbital_linear_operator(
            orbital_hessian_ctx, F_I_mo, F_A_mo, nbasis, n_core, n_act, n_virt,
            level_shift, mo_irreps, use_sym);

        result.orbital_step = solve_orbital_action_system(
            orbital_gradient,
            orbital_operator,
            F_I_mo,
            F_A_mo,
            n_core,
            n_act,
            n_virt,
            level_shift,
            max_rot,
            mo_irreps,
            use_sym,
            orbital_hessian_ctx);
        for (int i = 0; i < static_cast<int>(roots.size()); ++i)
        {
            const auto &root = roots[static_cast<std::size_t>(i)];
            if (root.weight == 0.0 || root.ci_vector.size() == 0 || H_diag.size() != root.ci_vector.size())
                continue;

            const CoupledResponseBlocks seed_blocks = build_coupled_response_blocks(
                mode,
                result.orbital_step,
                F_I_mo,
                h_eff,
                ga,
                space,
                a_strs,
                b_strs,
                dets,
                active_integrals,
                apply,
                root.ci_vector,
                root.ci_energy,
                H_diag,
                nbasis,
                n_core,
                n_act,
                n_virt);
            result.ci_steps[static_cast<std::size_t>(i)] = seed_blocks.ci_response.c1;
        }

        SACoupledResidualEvaluation current = evaluate_sa_coupled_residual(
            mode, orbital_gradient, result.orbital_step, result.ci_steps,
            F_I_mo, F_A_mo, h_eff, ga, space, a_strs, b_strs, dets,
            active_integrals, apply, roots, nbasis, n_core, n_act, n_virt,
            mo_irreps, use_sym, orbital_hessian_ctx);
        double current_metric = residual_metric(current);

        for (int iter = 1; iter <= max_iter; ++iter)
        {
            result.iterations = iter;
            if (!std::isfinite(current_metric))
                break;
            if (current_metric < tol)
            {
                result.converged = true;
                break;
            }

            Eigen::MatrixXd orbital_correction = solve_orbital_action_system(
                current.orbital_residual,
                orbital_operator,
                F_I_mo,
                F_A_mo,
                n_core,
                n_act,
                n_virt,
                level_shift,
                max_rot,
                mo_irreps,
                use_sym,
                orbital_hessian_ctx);
            std::vector<Eigen::VectorXd> ci_corrections;
            ci_corrections.reserve(roots.size());
            for (int i = 0; i < static_cast<int>(roots.size()); ++i)
            {
                const auto &root = roots[static_cast<std::size_t>(i)];
                if (root.weight == 0.0 || root.ci_vector.size() == 0 ||
                    H_diag.size() != root.ci_vector.size() ||
                    i >= static_cast<int>(current.ci_residuals.size()))
                {
                    ci_corrections.push_back(Eigen::VectorXd::Zero(root.ci_vector.size()));
                    continue;
                }

                double max_regularization = 0.0;
                Eigen::VectorXd correction = apply_response_diag_preconditioner(
                    -current.ci_residuals[static_cast<std::size_t>(i)],
                    H_diag,
                    root.ci_energy,
                    response_precond_floor,
                    max_regularization);
                ci_corrections.push_back(project_orthogonal(correction, root.ci_vector));
            }

            bool accepted_update = false;
            SACoupledResidualEvaluation best_evaluation = current;
            Eigen::MatrixXd best_orbital_step = result.orbital_step;
            std::vector<Eigen::VectorXd> best_ci_steps = result.ci_steps;
            double best_metric = current_metric;

            for (double scale : CASSCF_PROBE_STEP_SCALES)
            {
                Eigen::MatrixXd trial_orbital_step =
                    result.orbital_step + scale * orbital_correction;
                const double max_elem = trial_orbital_step.cwiseAbs().maxCoeff();
                if (max_elem > max_rot)
                    trial_orbital_step *= max_rot / max_elem;

                std::vector<Eigen::VectorXd> trial_ci_steps;
                trial_ci_steps.reserve(roots.size());
                for (int i = 0; i < static_cast<int>(roots.size()); ++i)
                {
                    const auto &root = roots[static_cast<std::size_t>(i)];
                    if (root.weight == 0.0 || root.ci_vector.size() == 0)
                    {
                        trial_ci_steps.push_back(Eigen::VectorXd::Zero(root.ci_vector.size()));
                        continue;
                    }

                    trial_ci_steps.push_back(project_orthogonal(
                        result.ci_steps[static_cast<std::size_t>(i)] +
                            scale * ci_corrections[static_cast<std::size_t>(i)],
                        root.ci_vector));
                }

                SACoupledResidualEvaluation trial = evaluate_sa_coupled_residual(
                    mode, orbital_gradient, trial_orbital_step, trial_ci_steps,
                    F_I_mo, F_A_mo, h_eff, ga, space, a_strs, b_strs, dets,
                    active_integrals, apply, roots, nbasis, n_core, n_act, n_virt,
                    mo_irreps, use_sym, orbital_hessian_ctx);
                const double trial_metric = residual_metric(trial);
                if (!std::isfinite(trial_metric))
                    continue;
                if (trial_metric < best_metric - 1e-12)
                {
                    accepted_update = true;
                    best_metric = trial_metric;
                    best_evaluation = std::move(trial);
                    best_orbital_step = std::move(trial_orbital_step);
                    best_ci_steps = std::move(trial_ci_steps);
                }
            }

            if (!accepted_update)
                break;

            result.orbital_step = std::move(best_orbital_step);
            result.ci_steps = std::move(best_ci_steps);
            current = std::move(best_evaluation);
            current_metric = best_metric;
        }

        result.orbital_residual = std::move(current.orbital_residual);
        result.ci_residuals = std::move(current.ci_residuals);
        result.orbital_correction = std::move(current.orbital_correction);
        result.orbital_residual_max = result.orbital_residual.cwiseAbs().maxCoeff();
        result.max_ci_residual_norm = max_ci_residual_norm(result.ci_residuals, roots);
        return result;
    }

    CoupledResponseBlocks build_coupled_response_blocks(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double tol,
        int max_iter,
        double precond_floor)
    {
        CoupledResponseBlocks blocks;
        blocks.ci_rhs = build_ci_response_rhs(
            mode,
            kappa,
            F_I_mo,
            h_eff,
            ga,
            active_integrals,
            space,
            a_strs,
            b_strs,
            c0,
            n_core,
            n_act);

        blocks.ci_response =
            solve_ci_response_davidson(apply, c0, E0, H_diag, blocks.ci_rhs, tol, max_iter, precond_floor);
        if (!blocks.ci_response.converged)
            blocks.ci_response = solve_ci_response_single_step(
                apply, c0, E0, H_diag, blocks.ci_rhs, precond_floor, tol);

        const Eigen::MatrixXd c1_vec = as_single_column_matrix(blocks.ci_response.c1);
        const Eigen::MatrixXd c0_vec = as_single_column_matrix(c0);
        const auto Gamma1_r = compute_2rdm_bilinear(
            c1_vec, c0_vec, single_weight(1.0),
            a_strs, b_strs, dets, n_act);
        const auto Gamma1_rt = compute_2rdm_bilinear(
            c0_vec, c1_vec, single_weight(1.0),
            a_strs, b_strs, dets, n_act);
        blocks.Gamma1_vec.resize(Gamma1_r.size(), 0.0);
        for (std::size_t i = 0; i < Gamma1_r.size(); ++i)
            blocks.Gamma1_vec[i] = Gamma1_r[i] + Gamma1_rt[i];
        blocks.Q1 = compute_Q_matrix(active_integrals, blocks.Gamma1_vec);
        blocks.orbital_correction =
            build_ci_orbital_gradient_correction(blocks.Q1, nbasis, n_core, n_act, n_virt);
        blocks.ci_residual = response_residual(
            apply, blocks.ci_response.c1, c0, E0, blocks.ci_rhs);
        return blocks;
    }

    CIResponseResult solve_ci_response_single_step(
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        const Eigen::VectorXd &sigma,
        double precond_floor,
        double tol)
    {
        CIResponseResult result;
        result.c1 = Eigen::VectorXd::Zero(c0.size());

        const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
        result.c1 = apply_response_diag_preconditioner(
            rhs, H_diag, E0, precond_floor, result.max_denominator_regularization);
        result.c1 = project_orthogonal(result.c1, c0);
        result.residual_norm = response_residual(apply, result.c1, c0, E0, sigma).norm();
        result.iterations = 1;
        result.converged = std::isfinite(result.residual_norm) && result.residual_norm < tol;
        return result;
    }

    CIResponseResult solve_ci_response_davidson(
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        const Eigen::VectorXd &sigma,
        double tol,
        int max_iter,
        double precond_floor,
        int max_subspace)
    {
        CIResponseResult result;
        result.c1 = Eigen::VectorXd::Zero(c0.size());

        if (c0.size() == 0 || H_diag.size() != c0.size() || sigma.size() != c0.size())
            return result;
        if (max_subspace < 1)
            max_subspace = 1;

        const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
        const double rhs_norm = rhs.norm();
        result.residual_norm = rhs_norm;
        if (!std::isfinite(rhs_norm))
            return result;
        if (rhs_norm < tol)
        {
            result.converged = true;
            return result;
        }

        // Keep the best finite iterate even if the subspace has to restart or the
        // linear solve stalls before the requested tolerance is reached.
        Eigen::VectorXd best_c1 = Eigen::VectorXd::Zero(c0.size());
        double best_residual_norm = rhs_norm;

        auto record_best = [&](const Eigen::VectorXd &c1, const Eigen::VectorXd &residual)
        {
            const double residual_norm = residual.norm();
            if (!std::isfinite(residual_norm))
                return;
            if (residual_norm < best_residual_norm)
            {
                best_residual_norm = residual_norm;
                best_c1 = c1;
            }
        };

        Eigen::VectorXd guess = apply_response_diag_preconditioner(
            rhs, H_diag, E0, precond_floor, result.max_denominator_regularization);
        guess = project_orthogonal(guess, c0);
        const double guess_norm = guess.norm();
        if (!(guess_norm > 1e-14))
            return result;

        record_best(guess, response_residual(apply, guess, c0, E0, sigma));

        Eigen::MatrixXd V(guess.size(), 1);
        V.col(0) = guess / guess_norm;

        for (int iter = 1; iter <= max_iter; ++iter)
        {
            const int m = static_cast<int>(V.cols());
            Eigen::MatrixXd AV(V.rows(), m);
            for (int k = 0; k < m; ++k)
            {
                Eigen::VectorXd sigma_vec(V.rows());
                apply(V.col(k), sigma_vec);
                AV.col(k) = sigma_vec - E0 * V.col(k);
            }

            const Eigen::MatrixXd M = V.transpose() * AV;
            const Eigen::VectorXd b = V.transpose() * rhs;
            const Eigen::VectorXd y = M.colPivHouseholderQr().solve(b);
            if (!y.allFinite())
                break;

            result.c1 = project_orthogonal(V * y, c0);
            Eigen::VectorXd residual = response_residual(apply, result.c1, c0, E0, sigma);
            result.residual_norm = residual.norm();
            result.iterations = iter;
            record_best(result.c1, residual);
            if (!std::isfinite(result.residual_norm))
                break;
            if (result.residual_norm < tol)
            {
                result.converged = true;
                return result;
            }

            Eigen::VectorXd correction = apply_response_diag_preconditioner(
                residual, H_diag, E0, precond_floor, result.max_denominator_regularization);
            correction = project_orthogonal(correction, c0);
            for (int k = 0; k < V.cols(); ++k)
                correction -= V.col(k).dot(correction) * V.col(k);

            const double corr_norm = correction.norm();
            if (!(corr_norm > 1e-14))
                break;

            if (m >= max_subspace)
            {
                Eigen::MatrixXd restart(c0.size(), 0);

                auto append_restart_vector = [&](const Eigen::VectorXd &v)
                {
                    // Rebuild the subspace from the best estimate and the newest
                    // correction, then re-orthogonalize both against c0 and the
                    // restarted basis.
                    Eigen::VectorXd orth = project_orthogonal(v, c0);
                    for (int k = 0; k < restart.cols(); ++k)
                        orth -= restart.col(k).dot(orth) * restart.col(k);
                    const double norm = orth.norm();
                    if (norm > 1e-14)
                    {
                        restart.conservativeResize(Eigen::NoChange, restart.cols() + 1);
                        restart.col(restart.cols() - 1) = orth / norm;
                    }
                };

                append_restart_vector(best_c1);
                append_restart_vector(correction);

                if (restart.cols() == 0)
                    break;

                V = std::move(restart);
                continue;
            }

            V.conservativeResize(Eigen::NoChange, m + 1);
            V.col(m) = correction / corr_norm;
        }

        result.c1 = std::move(best_c1);
        result.residual_norm = best_residual_norm;
        result.converged = false;
        return result;
    }

} // namespace HartreeFock::Correlation::CASSCF
