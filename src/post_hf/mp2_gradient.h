#ifndef HF_POSTHF_MP2_GRADIENT_H
#define HF_POSTHF_MP2_GRADIENT_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"
#include "post_hf/mp2.h"

namespace HartreeFock::Correlation
{
    struct RMP2GradientIntermediates
    {
        Eigen::MatrixXd electronic_gradient;
        Eigen::MatrixXd P_mo;
        Eigen::MatrixXd P_ao;
        Eigen::MatrixXd W_ao;
        Eigen::MatrixXd P_total_ao;
        Eigen::MatrixXd P_gamma_ao;
        Eigen::MatrixXd im1_ao;
        Eigen::MatrixXd zeta_ao;
        Eigen::MatrixXd vhf_s1occ_ao;
        std::vector<double> Gamma_pair_ao;
    };

    struct UMP2GradientIntermediates
    {
        Eigen::MatrixXd electronic_gradient;
        Eigen::MatrixXd P_alpha_ao;
        Eigen::MatrixXd P_beta_ao;
        Eigen::MatrixXd P_alpha_corr_ao;
        Eigen::MatrixXd P_beta_corr_ao;
        Eigen::MatrixXd P_total_ao;
        Eigen::MatrixXd W_ao;
        std::vector<double> Gamma_pair_ao;
    };

    // Public gradient-builder surface now consumes an explicit MP2 kernel
    // result instead of re-running MP2 internally.
    std::expected<RMP2GradientIntermediates, std::string> build_rmp2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result);

    std::expected<UMP2GradientIntermediates, std::string> build_ump2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UMP2Result &result);
} // namespace HartreeFock::Correlation

#endif
