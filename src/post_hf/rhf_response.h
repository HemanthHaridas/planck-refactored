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
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &mo_coeff,
        const Eigen::VectorXd &mo_energy);

    // RI-fitted RHF CPHF orbital Hessian (Step RG3.1). Same A as
    // build_rhf_cphf_matrix but assembled from the RI 3-center factors, no nao⁴.
    // build_rhf_cphf_matrix routes here under _mp2.use_ri; exposed for the gate.
    std::expected<Eigen::MatrixXd, std::string> build_rhf_cphf_matrix_ri(
        HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &mo_coeff,
        const Eigen::VectorXd &mo_energy,
        int n_occ,
        int n_virt);

    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &mo_coeff,
        const Eigen::VectorXd &mo_energy,
        const Eigen::MatrixXd &rhs);
} // namespace HartreeFock::Correlation

#endif
