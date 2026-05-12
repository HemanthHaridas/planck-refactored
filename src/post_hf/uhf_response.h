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
