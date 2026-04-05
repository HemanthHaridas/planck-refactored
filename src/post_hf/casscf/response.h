#ifndef HF_POSTHF_CASSCF_RESPONSE_H
#define HF_POSTHF_CASSCF_RESPONSE_H

#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <functional>
#include <utility>
#include <vector>

namespace HartreeFock::Correlation::CASSCF
{

    using HartreeFock::Correlation::CASSCFInternal::CIResponseResult;
    using HartreeFock::Correlation::CASSCFInternal::CIString;
    using HartreeFock::Correlation::CASSCFInternal::ActiveIntegralCache;
    struct CIDeterminantSpace;

    // Collect the explicit coupled response blocks for a single root after an
    // orbital trial step has been chosen. These are the building blocks for a
    // true coupled orbital/CI correction:
    // OO is still handled elsewhere, CO is `ci_rhs`, CC is `ci_response`, and
    // OC is the orbital correction reconstructed from the first-order CI state.
    struct CoupledResponseBlocks
    {
        Eigen::VectorXd ci_rhs;
        CIResponseResult ci_response;
        Eigen::VectorXd ci_residual;
        std::vector<double> Gamma1_vec;
        Eigen::MatrixXd Q1;
        Eigen::MatrixXd orbital_correction;
    };

    // Result of the block-iterative coupled orbital/CI linear solve used to
    // build a production trial direction. The current implementation keeps the
    // orbital and CI blocks explicit while using the diagonal orbital Hessian
    // and response diagonal only as preconditioners for the coupled residual.
    struct CoupledStepSolveResult
    {
        Eigen::MatrixXd orbital_step;
        Eigen::VectorXd ci_step;
        Eigen::MatrixXd orbital_residual;
        Eigen::VectorXd ci_residual;
        Eigen::MatrixXd orbital_correction;
        int iterations = 0;
        double orbital_residual_max = 0.0;
        double ci_residual_norm = 0.0;
        bool converged = false;
    };

    // Root-specific data for the shared-kappa state-averaged coupled solve.
    // All roots see the same orbital step, but each carries its own CI vector,
    // energy, and SA weight inside the coupled residual.
    struct StateAveragedCoupledRoot
    {
        double weight = 0.0;
        Eigen::VectorXd ci_vector;
        double ci_energy = 0.0;
    };

    // Result of the shared-kappa SA coupled solve. The orbital block is one
    // common step, while the CI corrections/residuals remain root-resolved.
    struct SACoupledStepSolveResult
    {
        Eigen::MatrixXd orbital_step;
        std::vector<Eigen::VectorXd> ci_steps;
        Eigen::MatrixXd orbital_residual;
        std::vector<Eigen::VectorXd> ci_residuals;
        Eigen::MatrixXd orbital_correction;
        int iterations = 0;
        double orbital_residual_max = 0.0;
        double max_ci_residual_norm = 0.0;
        bool converged = false;
    };

    // The response mode flag is a small policy switch used by the CASSCF driver to
    // describe which CI-response approximation is currently in play.
    enum class ResponseMode
    {
        ApproximatePrototype,
        DiagonalResponse,
        CoupledSecondOrderTarget,
    };

    const char *response_mode_name(ResponseMode mode);

    // CI-response RHS policy. The approximate path keeps the commutator
    // shortcut, while the exact path differentiates the active-space
    // Hamiltonian analytically with respect to a trial orbital rotation.
    enum class ResponseRHSMode
    {
        CommutatorOnlyApproximate,
        ExactActiveSpaceOrbitalDerivative,
    };

    const char *response_rhs_mode_name(ResponseRHSMode mode);

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

    // Approximate orbital response: active-block commutator with the inactive
    // Fock matrix only.
    Eigen::MatrixXd delta_h_eff(
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        int n_core,
        int n_act);

    // Build the CI-response right-hand side using either the current
    // commutator-only shortcut or the exact active-space orbital derivative.
    Eigen::VectorXd build_ci_response_rhs(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const Eigen::VectorXd &c0,
        int n_core,
        int n_act);

    // Solve the first-order CI block induced by an orbital trial step and
    // return the explicit CO/CC/OC pieces needed by later coupled-step
    // builders. This packages the currently inline driver logic into one
    // reusable block object.
    CoupledResponseBlocks build_coupled_response_blocks(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double tol = 1e-8,
        int max_iter = 64,
        double precond_floor = 1e-4);

    // Solve the linearized coupled orbital/CI stationarity system with a
    // matrix-free block iteration. The OO and CC blocks use diagonal
    // preconditioners, while the OC/CO couplings are applied explicitly through
    // the CI-response RHS and the orbital correction reconstructed from c1.
    CoupledStepSolveResult solve_coupled_orbital_ci_step(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &orbital_gradient,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::VectorXd &H_diag,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        double tol = 1e-6,
        int max_iter = 8,
        double response_precond_floor = 1e-4);

    // Solve the SA stationarity system with one shared orbital rotation kappa
    // and one CI-response vector per root.
    SACoupledStepSolveResult solve_sa_coupled_orbital_ci_step(
        ResponseRHSMode mode,
        const Eigen::MatrixXd &orbital_gradient,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const ActiveIntegralCache &active_integrals,
        const CISigmaApplier &apply,
        const std::vector<StateAveragedCoupledRoot> &roots,
        const Eigen::VectorXd &H_diag,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        double tol = 1e-6,
        int max_iter = 8,
        double response_precond_floor = 1e-4);

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
