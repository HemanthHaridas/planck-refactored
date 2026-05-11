#ifndef HF_POSTHF_RHF_RESPONSE_H
#define HF_POSTHF_RHF_RESPONSE_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{
    std::expected<Eigen::MatrixXd, std::string> build_rhf_cphf_matrix(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &rhs);
} // namespace HartreeFock::Correlation

#endif
