#include "post_hf/casscf/response.h"

#include "post_hf/casscf/ci.h"
#include "post_hf/casscf/strings.h"

#include <Eigen/QR>

namespace
{

    using HartreeFock::Correlation::CASSCF::build_det_lookup;
    using HartreeFock::Correlation::CASSCF::build_spin_dets;
    using HartreeFock::Correlation::CASSCF::count_occupied_below;
    using HartreeFock::Correlation::CASSCFInternal::apply_response_diag_preconditioner;
    using HartreeFock::Correlation::CASSCFInternal::project_orthogonal;
    using HartreeFock::Correlation::CASSCFInternal::single_bit_mask;

    struct FermionOpResult
    {
        HartreeFock::Correlation::CASSCFInternal::CIString det = 0;
        double phase = 0.0;
        bool valid = false;
    };

    // These mirror the string-layer operators so the response code uses the same
    // determinant phase convention as the sigma and RDM builders.
    inline FermionOpResult apply_annihilation(HartreeFock::Correlation::CASSCFInternal::CIString det, int orb)
    {
        const auto bit = single_bit_mask(orb);
        if (!(det & bit))
            return {};
        return {det ^ bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
    }

    inline FermionOpResult apply_creation(HartreeFock::Correlation::CASSCFInternal::CIString det, int orb)
    {
        const auto bit = single_bit_mask(orb);
        if (det & bit)
            return {};
        return {det | bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
    }

    std::size_t idx4(int p, int q, int r, int s, int n_act)
    {
        return static_cast<std::size_t>(((p * n_act + q) * n_act + r) * n_act + s);
    }

    Eigen::MatrixXd exact_active_one_body_derivative(
        const Eigen::MatrixXd &kappa_act,
        const Eigen::MatrixXd &h_eff)
    {
        // Match the same MO-rotation convention used by the orbital update path:
        // h' = U^T h U, so the first-order derivative is kappa^T h + h kappa.
        return kappa_act.transpose() * h_eff + h_eff * kappa_act;
    }

    std::vector<double> exact_active_two_body_derivative(
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
            return "coupled second-order target (not implemented)";
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
            h_eff.rows() < n_act ||
            h_eff.cols() < n_act)
            return Eigen::VectorXd::Zero(c0.size());

        const Eigen::MatrixXd kappa_act = kappa.block(n_core, n_core, n_act, n_act);
        const Eigen::MatrixXd dh = exact_active_one_body_derivative(kappa_act, h_eff);
        const std::vector<double> dga = exact_active_two_body_derivative(kappa_act, ga, n_act);
        const Eigen::MatrixXd dH = build_ci_hamiltonian_dense(a_strs, b_strs, space.dets, dh, dga, n_act);
        return dH * c0;
    }

    CIResponseResult solve_ci_response_single_step(
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        const Eigen::VectorXd &sigma,
        double precond_floor)
    {
        CIResponseResult result;
        result.c1 = Eigen::VectorXd::Zero(c0.size());

        const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
        result.c1 = apply_response_diag_preconditioner(
            rhs, H_diag, E0, precond_floor, result.max_denominator_regularization);
        result.c1 = project_orthogonal(result.c1, c0);
        result.residual_norm = response_residual(apply, result.c1, c0, E0, sigma).norm();
        result.iterations = 1;
        result.converged = false;
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
