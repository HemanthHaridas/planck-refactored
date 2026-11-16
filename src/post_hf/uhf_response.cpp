#include "uhf_response.h"

#include <format>
#include <limits>

#include "dft/response_gmres.h"
#include "post_hf/orbital_hessian.h"

namespace HartreeFock::Correlation
{
    std::expected<UHFCphfSolution, std::string> solve_uhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &coeff_alpha,
        const Eigen::MatrixXd &coeff_beta,
        const Eigen::VectorXd &energy_alpha,
        const Eigen::VectorXd &energy_beta,
        int nocc_alpha,
        int nocc_beta,
        const Eigen::MatrixXd &rhs_alpha,
        const Eigen::MatrixXd &rhs_beta)
    {
        if (!calculator._info._is_converged)
            return std::unexpected("solve_uhf_cphf: SCF not converged.");

        const int nva = static_cast<int>(coeff_alpha.cols()) - nocc_alpha;
        const int nvb = static_cast<int>(coeff_beta.cols()) - nocc_beta;
        if (nocc_alpha <= 0 || nocc_beta < 0 || nva <= 0 || nvb <= 0)
            return std::unexpected("solve_uhf_cphf: invalid occupied/virtual dimensions.");
        const int nova = nva * nocc_alpha;
        const int novb = nvb * nocc_beta;

        Eigen::VectorXd rhs(nova + novb);
        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < nocc_alpha; ++i)
                rhs(a * nocc_alpha + i) = -rhs_alpha(a, i);
        for (int a = 0; a < nvb; ++a)
            for (int i = 0; i < nocc_beta; ++i)
                rhs(nova + a * nocc_beta + i) = -rhs_beta(a, i);

        // Matrix-free: GMRES with the orbital-gap diagonal as preconditioner,
        // one Fock build per action instead of one per Hessian column.
        const OrbitalHessian hessian = OrbitalHessian::uhf(
            calculator, shell_pairs, coeff_alpha, coeff_beta,
            energy_alpha, energy_beta, nocc_alpha, nocc_beta);
        const DFT::Response::Action action =
            [&hessian](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
        { return hessian(x); };
        // 1e-10 of max|rhs|: on 2 H2O+ / 6-31G* (n=541) this is 33 actions and
        // agrees with the dense QR to 1.2e-8 of max|z|; the dense reference's
        // own screening noise floors that at ~5e-10, and the printed gradient
        // is identical either way.
        auto solve = DFT::Response::gmres(action, rhs, hessian.diagonal(), 1e-10);
        if (!solve)
            return std::unexpected("solve_uhf_cphf: " + solve.error());
        if (!solve->converged)
            return std::unexpected(std::format(
                "solve_uhf_cphf: GMRES did not converge (iterations={}, actions={}, residual={:.3e}).",
                solve->iterations, solve->action_count, solve->true_residuals.back()));
        const Eigen::VectorXd &sol = solve->x;
        UHFCphfSolution out;
        out.alpha = Eigen::MatrixXd::Zero(nva, nocc_alpha);
        out.beta = Eigen::MatrixXd::Zero(nvb, nocc_beta);
        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < nocc_alpha; ++i)
                out.alpha(a, i) = sol(a * nocc_alpha + i);
        for (int a = 0; a < nvb; ++a)
            for (int i = 0; i < nocc_beta; ++i)
                out.beta(a, i) = sol(nova + a * nocc_beta + i);
        return out;
    }
} // namespace HartreeFock::Correlation
