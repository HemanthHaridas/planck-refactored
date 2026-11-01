#ifndef HF_POSTHF_CASSCF_ORBITAL_H
#define HF_POSTHF_CASSCF_ORBITAL_H

#include "post_hf/casscf_internal.h"
#include "base/types.h"

#include <Eigen/Core>

#include <atomic>
#include <memory>
#include <vector>

namespace HartreeFock::Correlation::CASSCF
{

    // ponytail: temporary P0 probe for docs/CASSCF_PERFORMANCE_SCOPE.md; printed
    // per macro under PLANCK_CASSCF_CALL_PROBE. Delete once the scope is answered.
    struct CallProbe
    {
        std::atomic<long> evaluate{0};
        std::atomic<long> fixed_ci_gradient{0};
        std::atomic<long> active_cache{0};
        std::atomic<long> active_transform{0};
        std::atomic<long> fock{0};
    };
    inline CallProbe call_probe;

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
    // Per-macro data for the analytic fixed-CI orbital Hessian action: the
    // current Q matrix and Y[p,a,t,x], the 2-RDM contracted with the
    // (pa|vw) and (pu|aw) integrals that carry the active legs' response.
    // H[mu,nu,lam,i] = (mu nu|lam i) for i over the first nocc MOs (core and
    // active). One pass over the nb^4 tensor builds it; then G[X Y^T + Y X^T]
    // with X = C_occ, which covers the inactive and active Fock densities and
    // every density a Hessian product contracts, and the mixed (p u|v w)-type
    // blocks all cost O(nb^3) contractions of H.
    struct OccupiedHalfTransform
    {
        std::vector<double> H;
        int nbasis = 0;
        int nocc = 0;
    };

    OccupiedHalfTransform build_occupied_half_transform(
        const std::vector<double> &eri,
        const Eigen::MatrixXd &C,
        int nocc);

    // G[X Y^T + Y X^T] in the AO basis, X = the half transform's occupied
    // columns of C, Y (nb x nocc) arbitrary. O(nb^3 nocc).
    Eigen::MatrixXd occupied_two_electron_ao(
        const OccupiedHalfTransform &half,
        const Eigen::MatrixXd &Y);

    // out[a1,a2,a3,w] = (a1 a2|a3 n_core+w) with a_k over the columns of C_k and
    // w over the active MOs. lam_first contracts the C3 leg before the C2 leg:
    // contract the leg with fewer columns first, or the cost is O(nb^4 n_act).
    std::vector<double> transform_occupied_half(
        const OccupiedHalfTransform &half,
        const Eigen::MatrixXd &C1,
        const Eigen::MatrixXd &C2,
        const Eigen::MatrixXd &C3,
        int n_core,
        int n_act,
        bool lam_first);

    struct AnalyticOrbitalHessian
    {
        Eigen::MatrixXd Q;
        std::vector<double> Y;
        // The half transform at the same C (shared with the McscfState it came
        // from).
        std::shared_ptr<const OccupiedHalfTransform> half;
        int nbasis = 0;
        int nact = 0;
    };

    // Builds the half transform at C itself (one pass over eri).
    AnalyticOrbitalHessian build_analytic_orbital_hessian(
        const std::vector<double> &eri,
        const Eigen::MatrixXd &C,
        const std::vector<double> &Gamma_vec,
        int n_core,
        int n_act);

    // Reuses a half transform already built at C.
    AnalyticOrbitalHessian build_analytic_orbital_hessian(
        std::shared_ptr<const OccupiedHalfTransform> half,
        const Eigen::MatrixXd &C,
        const std::vector<double> &Gamma_vec,
        int n_core,
        int n_act);

    struct OrbitalHessianContext
    {
        const Eigen::MatrixXd *C = nullptr;
        const Eigen::MatrixXd *S = nullptr;
        const Eigen::MatrixXd *H_core = nullptr;
        const std::vector<double> *eri = nullptr;
        const Eigen::MatrixXd *gamma = nullptr;
        const std::vector<double> *Gamma_vec = nullptr;
        // When non-null and RI-enabled, the FD Hessian differentiates the RI
        // orbital gradient (so the Hessian is consistent with the RI energy /
        // gradient it curves). Left null in the dense path.
        HartreeFock::Calculator *ri_calc = nullptr;
        double fd_step = 5e-4;
        // When set, delta_g_sa_action is analytic; otherwise it is the central
        // finite difference of fixed_ci_orbital_gradient (kept as the oracle).
        const AnalyticOrbitalHessian *analytic = nullptr;
    };

    // C^T G[D] C for an AO density D, with G the closed-shell two-electron
    // operator of the context's integrals (dense or RI).
    Eigen::MatrixXd fock_response_mo(
        const Eigen::MatrixXd &D_ao,
        const OrbitalHessianContext &ctx);

    // C^T G[X Y^T + Y X^T] C with X = the first Y.cols() columns of C (the
    // core and active orbitals). With the analytic context's H this is
    // O(nb^3 nocc) instead of a pass over the nb^4 tensor; otherwise it forms
    // the density and calls fock_response_mo.
    Eigen::MatrixXd occupied_fock_response_mo(
        const Eigen::MatrixXd &Y,
        const OrbitalHessianContext &ctx);

    // Cache the active-space integral transform and reuse it across all response
    // contractions in a macroiteration.
    // When ri_calc is non-null and RI is enabled on it, the ERI-dependent Fock /
    // transform pieces are built from the density-fitted RI path instead of the
    // dense eri tensor (the eri argument is then ignored for those pieces).
    ActiveIntegralCache build_active_integral_cache(
        const std::vector<double> &eri,
        const Eigen::MatrixXd &C,
        int n_core,
        int n_act,
        int nbasis,
        HartreeFock::Calculator *ri_calc = nullptr);

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
        int nbasis,
        const HartreeFock::Calculator *ri_calc = nullptr);

    // Build the active-space contribution from the 1-RDM so the generalized Fock
    // matrix can include the coupling between core, active, and virtual blocks.
    Eigen::MatrixXd build_active_fock_mo(
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &gamma,
        const std::vector<double> &eri,
        int n_core,
        int n_act,
        int nbasis,
        const HartreeFock::Calculator *ri_calc = nullptr);

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
        bool use_sym,
        HartreeFock::Calculator *ri_calc = nullptr);

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
