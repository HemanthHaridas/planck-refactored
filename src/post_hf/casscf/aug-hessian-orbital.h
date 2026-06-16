#ifndef HF_POSTHF_CASSCF_AUG_HESSIAN_ORBITAL_H
#define HF_POSTHF_CASSCF_AUG_HESSIAN_ORBITAL_H

#include "post_hf/casscf/aug-hessian.h"
#include "post_hf/casscf/orbital.h"

#include <Eigen/Core>

#include <vector>

namespace HartreeFock::Correlation::CASSCF
{
    // Adapter layer that packages the existing CASSCF orbital gradient and
    // matrix-free Hessian action (delta_g_sa_action / hess_diag) into the
    // generic AugHessianHopFn / AugHessianGradFn / AugHessianPrecondFn
    // callbacks consumed by solve_augmented_hessian.
    //
    // Nothing here mutates global state or replaces the existing diagonal
    // augmented_hessian_step. The intent is for the macro driver to call
    // solve_orbital_augmented_hessian_step as one more candidate alongside
    // sa-coupled / sa-grad-fallback / numeric-newton.

    // Pack the upper triangle of an antisymmetric kappa matrix into the flat
    // pair vector used by solve_augmented_hessian. The layout matches
    // non_redundant_pairs(): one entry per (p, q) with p < q in different
    // orbital blocks. Symmetry-forbidden pairs are zeroed in place so the
    // packed vector contains only physically variable rotations.
    Eigen::VectorXd pack_rotation_matrix(
        const Eigen::MatrixXd &kappa,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Inverse of pack_rotation_matrix: scatter a flat pair vector into the
    // antisymmetric (nbasis x nbasis) representation expected by
    // delta_g_sa_action and apply_orbital_rotation.
    Eigen::MatrixXd unpack_rotation_vector(
        const Eigen::VectorXd &x,
        const std::vector<RotPair> &pairs,
        int nbasis);

    // Build a Hessian-vector callback that wraps delta_g_sa_action. The
    // returned closure captures every input by reference, so the caller is
    // responsible for keeping F_I_mo / F_A_mo / context / pairs / mo_irreps
    // alive for the entire CIAH solve.
    AugHessianHopFn make_orbital_hessian_action(
        const OrbitalHessianContext *context,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Build a gradient callback that returns the packed (signed) g vector
    // from a fixed orbital gradient matrix. The closure takes a copy of
    // g_orb so the caller can free the source matrix without invalidating
    // the AH solver.
    AugHessianGradFn make_orbital_gradient(
        const Eigen::MatrixXd &g_orb,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    // Build the diagonal Hessian preconditioner used inside CIAH Davidson.
    // For each pair (p, q) the denominator is (h_pp_qq + level_shift) - e
    // where e is the Ritz value passed in by the AH solver. denom_floor
    // protects against tiny denominators in the same way the existing
    // diagonal augmented_hessian_step does.
    AugHessianPrecondFn make_orbital_preconditioner(
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        double level_shift,
        double denom_floor = 1e-4);

    // Top-level entry point. Builds the three callbacks, runs the CIAH
    // Davidson, and returns the orbital step as an antisymmetric matrix
    // capped at max_rot in element-wise infinity norm. Result.x carries the
    // packed pair vector and the AH metadata so the macro driver can decide
    // whether to accept the step.
    struct OrbitalAugHessianStep
    {
        Eigen::MatrixXd kappa;
        AugHessianResult ah;
    };

    OrbitalAugHessianStep solve_orbital_augmented_hessian_step(
        const Eigen::MatrixXd &g_orb,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const OrbitalHessianContext *context,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        const AugHessianOptions &opts = AugHessianOptions{});

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_AUG_HESSIAN_ORBITAL_H
