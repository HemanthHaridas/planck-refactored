#ifndef HF_POSTHF_CASSCF_RESPONSE_H
#define HF_POSTHF_CASSCF_RESPONSE_H

#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <functional>

namespace HartreeFock::Correlation::CASSCF
{

using HartreeFock::Correlation::CASSCFInternal::CIResponseResult;
using HartreeFock::Correlation::CASSCFInternal::CIString;

enum class ResponseMode
{
    ApproximateDressedGradient,
    IterativeResponse,
    TargetFullSecondOrder,
};

const char* response_mode_name(ResponseMode mode);

using CISigmaApplier = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;

Eigen::VectorXd ci_sigma_1body(
    const Eigen::MatrixXd& dh,
    const Eigen::VectorXd& c,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

Eigen::MatrixXd delta_h_eff(
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& F_I_mo,
    int n_core,
    int n_act);

CIResponseResult solve_ci_response_single_step(
    const CISigmaApplier& apply,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& H_diag,
    const Eigen::VectorXd& sigma,
    double precond_floor = 1e-4);

CIResponseResult solve_ci_response_davidson(
    const CISigmaApplier& apply,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& H_diag,
    const Eigen::VectorXd& sigma,
    double tol = 1e-8,
    int max_iter = 32,
    double precond_floor = 1e-4);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_RESPONSE_H
