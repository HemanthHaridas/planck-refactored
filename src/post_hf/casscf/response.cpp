#include "post_hf/casscf/response.h"

#include "post_hf/casscf/strings.h"

#include <Eigen/QR>

namespace
{

using HartreeFock::Correlation::CASSCFInternal::apply_response_diag_preconditioner;
using HartreeFock::Correlation::CASSCFInternal::project_orthogonal;
using HartreeFock::Correlation::CASSCFInternal::single_bit_mask;
using HartreeFock::Correlation::CASSCF::build_det_lookup;
using HartreeFock::Correlation::CASSCF::build_spin_dets;
using HartreeFock::Correlation::CASSCF::count_occupied_below;

struct FermionOpResult
{
    HartreeFock::Correlation::CASSCFInternal::CIString det = 0;
    double phase = 0.0;
    bool valid = false;
};

inline FermionOpResult apply_annihilation(HartreeFock::Correlation::CASSCFInternal::CIString det, int orb)
{
    const auto bit = single_bit_mask(orb);
    if (!(det & bit)) return {};
    return {det ^ bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
}

inline FermionOpResult apply_creation(HartreeFock::Correlation::CASSCFInternal::CIString det, int orb)
{
    const auto bit = single_bit_mask(orb);
    if (det & bit) return {};
    return {det | bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
}

Eigen::VectorXd response_residual(
    const HartreeFock::Correlation::CASSCF::CISigmaApplier& apply,
    const Eigen::VectorXd& c1,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& sigma)
{
    Eigen::VectorXd hc1(c1.size());
    apply(c1, hc1);
    const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
    return project_orthogonal(rhs - (hc1 - E0 * c1), c0);
}

} // namespace

namespace HartreeFock::Correlation::CASSCF
{

const char* response_mode_name(ResponseMode mode)
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

Eigen::VectorXd ci_sigma_1body(
    const Eigen::MatrixXd& dh,
    const Eigen::VectorXd& c,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    const int dim = static_cast<int>(c.size());
    const auto sd = build_spin_dets(a_strs, b_strs, dets, n_act);
    const auto lut = build_det_lookup(sd);
    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(dim);

    for (int j = 0; j < dim; ++j)
    {
        const double cJ = c(j);
        if (std::abs(cJ) < 1e-15) continue;
        const auto ket = sd[j];
        // Match slater_condon_element(): use the ket->bra convention
        // for a_p^\dagger a_q, i.e. annihilate an occupied q in the ket
        // and create the corresponding bra orbital p with coefficient dh(p, q).
        for (int q_so = 0; q_so < 2 * n_act; ++q_so)
        {
            auto ann = apply_annihilation(ket, q_so);
            if (!ann.valid) continue;
            const int spin_offset = (q_so >= n_act) ? n_act : 0;
            const int q = q_so - spin_offset;
            for (int p = 0; p < n_act; ++p)
            {
                if (std::abs(dh(p, q)) < 1e-18) continue;
                auto cre = apply_creation(ann.det, spin_offset + p);
                if (!cre.valid) continue;
                auto it = lut.find(cre.det);
                if (it == lut.end()) continue;
                sigma(it->second) += dh(p, q) * ann.phase * cre.phase * cJ;
            }
        }
    }

    return sigma;
}

Eigen::MatrixXd delta_h_eff(
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& F_I_mo,
    int n_core,
    int n_act)
{
    Eigen::MatrixXd comm = kappa * F_I_mo - F_I_mo * kappa;
    return comm.block(n_core, n_core, n_act, n_act);
}

CIResponseResult solve_ci_response_single_step(
    const CISigmaApplier& apply,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& H_diag,
    const Eigen::VectorXd& sigma,
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
    const CISigmaApplier& apply,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& H_diag,
    const Eigen::VectorXd& sigma,
    double tol,
    int max_iter,
    double precond_floor)
{
    CIResponseResult result;
    result.c1 = Eigen::VectorXd::Zero(c0.size());

    if (c0.size() == 0 || H_diag.size() != c0.size() || sigma.size() != c0.size())
        return result;

    const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
    result.residual_norm = rhs.norm();
    if (!std::isfinite(result.residual_norm))
        return result;
    if (result.residual_norm < tol)
    {
        result.converged = true;
        return result;
    }

    Eigen::VectorXd guess = apply_response_diag_preconditioner(
        rhs, H_diag, E0, precond_floor, result.max_denominator_regularization);
    guess = project_orthogonal(guess, c0);
    const double guess_norm = guess.norm();
    if (!(guess_norm > 1e-14))
        return result;

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
        for (int k = 0; k < V.cols(); ++k)
            correction -= V.col(k).dot(correction) * V.col(k);

        const double corr_norm = correction.norm();
        if (!(corr_norm > 1e-14))
            break;

        V.conservativeResize(Eigen::NoChange, m + 1);
        V.col(m) = correction / corr_norm;
    }

    return result;
}

} // namespace HartreeFock::Correlation::CASSCF
