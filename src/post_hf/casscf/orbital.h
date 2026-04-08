#ifndef HF_POSTHF_CASSCF_ORBITAL_H
#define HF_POSTHF_CASSCF_ORBITAL_H

#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <vector>

namespace HartreeFock::Correlation::CASSCF
{

    using HartreeFock::Correlation::CASSCFInternal::ActiveIntegralCache;

    // Rotation pairs are stored as an antisymmetric generator index (p,q) for the
    // non-redundant orbital blocks only.
    struct RotPair
    {
        int p = 0;
        int q = 0;
    };

    // Explicit coupled-step correction with separate orbital and CI blocks.
    // The current implementation still uses block-diagonal preconditioning, but
    // packaging the two pieces together gives the driver a real coupled-step
    // object instead of loose scalar/vector scratch variables.
    struct CoupledStepDirection
    {
        Eigen::MatrixXd orbital_step;
        Eigen::VectorXd ci_step;
    };

    // Matrix-free OO Hessian actions need the fixed-CI orbital-gradient model:
    // current MO coefficients, AO overlap, one-/two-electron integrals, and the
    // state-specific (or state-averaged) active-space densities held fixed while
    // differentiating the orbital gradient.
    struct OrbitalHessianContext
    {
        const Eigen::MatrixXd *C = nullptr;
        const Eigen::MatrixXd *S = nullptr;
        const Eigen::MatrixXd *H_core = nullptr;
        const std::vector<double> *eri = nullptr;
        const Eigen::MatrixXd *gamma = nullptr;
        const std::vector<double> *Gamma_vec = nullptr;
        double fd_step = 5e-4;
    };

    // Cache the active-space integral transform and reuse it across all response
    // contractions in a macroiteration.
    ActiveIntegralCache build_active_integral_cache(
        const std::vector<double> &eri,
        const Eigen::MatrixXd &C,
        int n_core,
        int n_act,
        int nbasis);

    // Contract the active 2-RDM against the cached mixed-basis integrals to build
    // the Q matrix that enters the orbital gradient.
    Eigen::MatrixXd compute_Q_matrix(
        const ActiveIntegralCache &cache,
        const std::vector<double> &Gamma);

    // Build the inactive/core contribution to the Fock matrix in the current MO
    // basis. This is the closed-shell part driven entirely by the doubly occupied
    // orbitals.
    Eigen::MatrixXd build_inactive_fock_mo(
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &H_core,
        const std::vector<double> &eri,
        int n_core,
        int nbasis);

    // Build the active-space contribution from the 1-RDM so the generalized Fock
    // matrix can include the coupling between core, active, and virtual blocks.
    Eigen::MatrixXd build_active_fock_mo(
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &gamma,
        const std::vector<double> &eri,
        int n_core,
        int n_act,
        int nbasis);

    // Compute the total electronic energy contribution from the occupied core and
    // inactive Fock blocks.
    double compute_core_energy(
        const Eigen::MatrixXd &h_mo,
        const Eigen::MatrixXd &F_I_mo,
        int n_core);

    // Assemble the generalized orbital gradient for non-redundant rotations.
    // `use_sym` lets the caller zero forbidden blocks after the raw gradient is
    // formed.
    Eigen::MatrixXd compute_orbital_gradient(
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::MatrixXd &Q,
        const Eigen::MatrixXd &gamma,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Rebuild the fixed-CI orbital gradient at a given MO basis, keeping the
    // supplied active 1-/2-RDMs frozen while recomputing the MO-basis integral
    // intermediates.
    Eigen::MatrixXd fixed_ci_orbital_gradient(
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &H_core,
        const std::vector<double> &eri,
        const Eigen::MatrixXd &gamma,
        const std::vector<double> &Gamma_vec,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Feed the first-order CI response back into the orbital stationarity
    // equations. Only inactive/active and active/virtual blocks survive; within
    // a subspace the rotation remains gauge redundant.
    Eigen::MatrixXd build_ci_orbital_gradient_correction(
        const Eigen::MatrixXd &Q1,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt);

    // Diagonal orbital-Hessian estimate for a pair of orbitals.
    double hess_diag(const Eigen::MatrixXd &F_sum, int p, int q);

    // Apply the diagonal Hessian model to an arbitrary antisymmetric trial
    // direction.
    Eigen::MatrixXd hessian_action(
        const Eigen::MatrixXd &R,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt);

    // State-averaged orbital Hessian action δg_SA[R] evaluated at fixed CI by
    // finite-differencing the true orbital gradient under a small orbital
    // rotation. Falls back to the diagonal model if the context is incomplete.
    Eigen::MatrixXd delta_g_sa_action(
        const Eigen::MatrixXd &R,
        const OrbitalHessianContext *context,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Backward-compatible alias for the milestone-3 orbital Hessian action.
    Eigen::MatrixXd matrix_free_hessian_action(
        const Eigen::MatrixXd &R,
        const OrbitalHessianContext *context,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Update the orbital gradient with the first-order response of the diagonal
    // Hessian model to the current rotation step.
    Eigen::MatrixXd fep1_gradient_update(
        const Eigen::MatrixXd &G,
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt);

    // Quadratic model used to score competing orbital steps before a full CASSCF
    // reevaluation decides whether to accept them.
    double quadratic_model_delta(
        const Eigen::VectorXd &g_flat,
        const Eigen::VectorXd &h_flat,
        const Eigen::VectorXd &x);

    // Enumerate the unique core-active, core-virtual, and active-virtual rotation
    // pairs. Rotations within a block are gauge freedom and omitted.
    std::vector<RotPair> non_redundant_pairs(int n_core, int n_act, int n_virt);

    // Augmented-Hessian style orbital step builder with diagonal preconditioning,
    // symmetry masking, and a step-length cap.
    Eigen::MatrixXd augmented_hessian_step(
        const Eigen::MatrixXd &G,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Apply only the diagonal orbital-Hessian preconditioner to the current
    // orbital residual. This is the lightweight building block used by the
    // coupled-step scaffold after the explicit CI response has already updated
    // the residual, so the diagonal model acts as a preconditioner rather than
    // as the full step model.
    Eigen::MatrixXd diagonal_preconditioned_orbital_step(
        const Eigen::MatrixXd &G,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Apply the block-diagonal orbital/CI preconditioner to the current coupled
    // residual. This is the first concrete step object that carries both the
    // orbital and CI corrections together, even though the production solver
    // still treats it as an experimental fallback rather than the default path.
    CoupledStepDirection diagonal_preconditioned_coupled_step(
        const Eigen::MatrixXd &orbital_residual,
        const Eigen::VectorXd &ci_residual,
        const Eigen::VectorXd &c0,
        double E0,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const Eigen::VectorXd &H_diag,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        double response_precond_floor = 1e-4);

    // Apply the antisymmetric orbital rotation to the coefficient matrix and then
    // restore orthonormality in the current AO metric.
    Eigen::MatrixXd apply_orbital_rotation(
        const Eigen::MatrixXd &C_old,
        const Eigen::MatrixXd &kappa,
        const Eigen::MatrixXd &S);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_ORBITAL_H
