#ifndef HF_POSTHF_CASSCF_RESPONSE_H
#define HF_POSTHF_CASSCF_RESPONSE_H

#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <functional>

namespace HartreeFock::Correlation::CASSCF
{

    using HartreeFock::Correlation::CASSCFInternal::CIResponseResult;
    using HartreeFock::Correlation::CASSCFInternal::CIString;

    // The response mode flag is a small policy switch used by the CASSCF driver to
    // describe which CI-response approximation is currently in play.
    enum class ResponseMode
    {
        ApproximatePrototype,
        DiagonalResponse,
        CoupledSecondOrderTarget,
    };

    const char *response_mode_name(ResponseMode mode);

    // A sigma application callback hides whether the response operator is coming
    // from a dense matrix or from an iterative Hamiltonian application.
    using CISigmaApplier = std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &)>;

    // Applies the same ket->bra convention as slater_condon_element():
    // sum_pq dh(p,q) a_p^dagger a_q.
    // This is the active-space 1-body response operator used to build the CI RHS.
    Eigen::VectorXd ci_sigma_1body(
        const Eigen::MatrixXd &dh,
        const Eigen::VectorXd &c,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

    // Orbital perturbations enter the CI response through the active-block
    // commutator with the inactive Fock matrix.
    Eigen::MatrixXd delta_h_eff(
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        int n_core,
        int n_act);

    // One preconditioned response step without iterative subspace growth.
    CIResponseResult solve_ci_response_single_step(
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        const Eigen::VectorXd &sigma,
        double precond_floor = 1e-4);

    // Davidson-style solver for the first-order CI response. It balances restart
    // stability against the need to preserve the best finite iterate.
    CIResponseResult solve_ci_response_davidson(
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        const Eigen::VectorXd &sigma,
        double tol = 1e-8,
        int max_iter = 32,
        double precond_floor = 1e-4,
        int max_subspace = 16);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_RESPONSE_H
