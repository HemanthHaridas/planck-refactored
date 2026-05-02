#ifndef HF_POSTHF_CASSCF_AUG_HESSIAN_H
#define HF_POSTHF_CASSCF_AUG_HESSIAN_H

#include <Eigen/Core>

#include <functional>

namespace HartreeFock::Correlation::CASSCF
{
    // Generic Co-Iterative Augmented Hessian (CIAH) solver, modeled after
    // pyscf/soscf/ciah.py:davidson_cc. The augmented system
    //
    //     [ 0   g^T ] [ 1 ]       [ 1 ]
    //     [ g    H  ] [ x ]  = w  [ x ]
    //
    // is never assembled. Instead a Krylov subspace is built from the trial
    // vector x and the Hessian-vector product H_op(x), and the lowest
    // eigenpair of the bordered subspace matrix is extracted. The orbital
    // step recovered from the eigenvector is x / v[0]; v[0] is the bordered
    // first component (the implicit "1" in the augmented system).
    //
    // The callbacks give the caller full control over how the gradient and
    // Hessian-vector product are computed without coupling this solver to
    // the CASSCF data structures.
    using AugHessianHopFn      = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    using AugHessianGradFn     = std::function<Eigen::VectorXd()>;
    using AugHessianPrecondFn  = std::function<Eigen::VectorXd(const Eigen::VectorXd &, double)>;

    // Tunables follow the PySCF defaults documented in
    // pyscf/mcscf/mc1step.py:707-733. The early-exit threshold ah_start_tol
    // and minimum cycle count ah_start_cycle keep the inner micro-iteration
    // cheap when the Krylov subspace is producing usable steps.
    struct AugHessianOptions
    {
        double ah_conv_tol         = 1e-12;
        double ah_start_tol        = 2.5;
        int    ah_start_cycle      = 1;
        int    ah_max_cycle        = 30;
        double ah_lindep           = 1e-14;
        double ah_level_shift      = 0.0;
        // Mode-selection guard from ciah._regular_step (line 295). Eigenvectors
        // with |v[0]| below this threshold imply the orbital step blows up and
        // are skipped in favor of the next eigenvector.
        double v0_min              = 0.1;
        // Below this AH eigenvalue the bordered system is dropped and only the
        // pure orbital block is solved (ciah lines 301-306).
        double orbital_only_floor  = 1e-4;
        // Maximum Krylov subspace size before restart. Bounded by ah_max_cycle.
        int    max_subspace        = 30;
    };

    struct AugHessianResult
    {
        // Orbital step (or joint orb+CI step in the joint Newton extension)
        // packed in the same layout the caller used to define h_op.
        Eigen::VectorXd x;
        // AH eigenvalue selected from the subspace.
        double          eigenvalue     = 0.0;
        // First component of the bordered eigenvector. The orbital scaling is
        // 1/v0; small |v0| means the step would explode.
        double          v0             = 0.0;
        // Final residual norm of (H + g v0 - w v0) x.
        double          residual_norm  = 0.0;
        // Number of Davidson iterations actually run.
        int             iterations     = 0;
        // Whether the convergence test was satisfied (vs. early-exit / restart).
        bool            converged      = false;
        // True when the orbital-only fallback fired because the AH eigenvalue
        // dropped below opts.orbital_only_floor.
        bool            orbital_only_fallback = false;
    };

    // Solve the bordered AH eigenproblem with a CIAH-style Davidson inner
    // loop. h_op must compute H*x for any trial vector x of size g.size().
    // g_op is invoked once per outer iteration so the caller can re-evaluate
    // the gradient if it depends on accumulated rotations (the keyframe path
    // in mc1step.py:rotate_orb_cc). precond applies a preconditioner with
    // the current Ritz value as the level-shift target; if it is null, an
    // identity preconditioner is used.
    AugHessianResult solve_augmented_hessian(
        const AugHessianHopFn     &h_op,
        const AugHessianGradFn    &g_op,
        const AugHessianPrecondFn &precond,
        const Eigen::VectorXd     &x0,
        const AugHessianOptions   &opts);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_AUG_HESSIAN_H
