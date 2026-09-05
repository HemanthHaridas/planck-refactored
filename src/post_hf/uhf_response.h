#ifndef HF_POSTHF_UHF_RESPONSE_H
#define HF_POSTHF_UHF_RESPONSE_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{
    struct UHFCphfSolution
    {
        Eigen::MatrixXd alpha;
        Eigen::MatrixXd beta;
    };

    // Coupled alpha/beta orbital-Hessian matrix (docs/SOSCF_UHF_DFT_SCOPE.md,
    // U1). Same dense layout solve_uhf_cphf built inline; split out so SOSCF
    // can read it directly, mirroring build_rhf_cphf_matrix / solve_rhf_cphf.
    std::expected<Eigen::MatrixXd, std::string> build_uhf_cphf_matrix(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &coeff_alpha,
        const Eigen::MatrixXd &coeff_beta,
        const Eigen::VectorXd &energy_alpha,
        const Eigen::VectorXd &energy_beta,
        int nocc_alpha,
        int nocc_beta);

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
        const Eigen::MatrixXd &rhs_beta);
} // namespace HartreeFock::Correlation

#endif
