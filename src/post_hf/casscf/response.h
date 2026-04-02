#ifndef HF_POSTHF_CASSCF_RESPONSE_H
#define HF_POSTHF_CASSCF_RESPONSE_H

#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <functional>

namespace HartreeFock::Correlation::CASSCF
{

    using HartreeFock::Correlation::CASSCFInternal::CIResponseResult;
    using HartreeFock::Correlation::CASSCFInternal::CIString;
    struct CIDeterminantSpace;

    // The response mode flag is a small policy switch used by the CASSCF driver to
    // describe which CI-response approximation is currently in play.
    enum class ResponseMode
    {
        ApproximatePrototype,
        DiagonalResponse,
        CoupledSecondOrderTarget,
    };

    const char *response_mode_name(ResponseMode mode);

    // CI-response RHS policy. The current production path is still approximate
    // by default for single-root runs, while the SA plumbing can request an
    // explicit orbital-derivative build that includes the full active
    // Hamiltonian response to a trial orbital rotation.
    enum class ResponseRHSMode
    {
        CommutatorOnlyApproximate,
        ExactActiveSpaceOrbitalDerivative,
    };

    const char *response_rhs_mode_name(ResponseRHSMode mode);

    struct ResponseRHSExactContext
    {
        const Eigen::MatrixXd *C = nullptr;
        const Eigen::MatrixXd *overlap = nullptr;
        const Eigen::MatrixXd *H_core = nullptr;
        const std::vector<double> *eri = nullptr;
        int nbasis = 0;
        double fd_step = 1e-4;
    };

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

    // Orbital perturbations enter the approximate CI response through the
    // active-block commutator with the inactive Fock matrix.
    Eigen::MatrixXd delta_h_eff(
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        int n_core,
        int n_act);

    // Build the CI-response right-hand side using either the legacy
    // commutator-only shortcut or an explicit finite-difference orbital
    // derivative of the active-space Hamiltonian action.
    Eigen::VectorXd build_ci_response_rhs(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const Eigen::VectorXd &c0,
        int n_core,
        int n_act,
        const ResponseRHSExactContext *exact_context = nullptr);

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
