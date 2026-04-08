#ifndef HF_POSTHF_CASSCF_CI_H
#define HF_POSTHF_CASSCF_CI_H

#include "post_hf/casscf/strings.h"

#include <Eigen/Core>

#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace HartreeFock::Correlation::CASSCF
{

    // The determinant space keeps the ordering, diagonal, and optional dense
    // Hamiltonian together so the solver can choose between exact and iterative CI.
    struct CIDeterminantSpace
    {
        std::vector<std::pair<int, int>> dets;
        Eigen::VectorXd diagonal;
        std::optional<Eigen::MatrixXd> dense_hamiltonian;
        std::vector<CIString> spin_dets;
        std::unordered_map<CIString, int> det_lookup;
    };

    // CI solve results always carry the eigenvectors alongside the energies and
    // the working diagonal used by any response or restart logic.
    struct CISolveResult
    {
        Eigen::VectorXd energies;
        Eigen::MatrixXd vectors;
        Eigen::VectorXd diagonal;
        std::optional<Eigen::MatrixXd> dense_hamiltonian;
        bool used_direct_sigma = false;
    };

    // Matrix element of the active-space Hamiltonian using the ket->bra
    // convention for one-body excitations: sum_pq h(p,q) a_p^dagger a_q.
    // The packed alpha/beta determinant pair is the only state representation used
    // at this layer.
    double slater_condon_element(
        CIString bra_a,
        CIString bra_b,
        CIString ket_a,
        CIString ket_b,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        int n_act);

    // Build the filtered determinant list in the same order used by all CI arrays
    // and lookup tables.
    std::vector<std::pair<int, int>> build_ci_determinant_list(
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const RASParams &ras,
        const std::vector<int> &irr_act,
        const SymmetryContext *sym_ctx,
        int target_irr);

    // Materialize the full CI Hamiltonian when the determinant space is small
    // enough that the dense solve path is cheaper than iterative sigma application.
    Eigen::MatrixXd build_ci_hamiltonian_dense(
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        int n_act);

    // Assemble the determinant space, diagonal, packed determinant keys, and
    // optional dense matrix in one place so the solve path can defer the strategy
    // choice until after screening.
    CIDeterminantSpace build_ci_space(
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const RASParams &ras,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        int n_act,
        const std::vector<int> &irr_act = {},
        const SymmetryContext *sym_ctx = nullptr,
        int target_irr = 0,
        int dense_threshold = 500);

    // Apply H to a CI vector using either the cached dense matrix or the packed
    // Slater-Condon traversal depending on the determinant-space size.
    void apply_ci_hamiltonian(
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        int n_act,
        const Eigen::VectorXd &c,
        Eigen::VectorXd &sigma);

    // Build only the diagonal part of the CI Hamiltonian in the same determinant
    // order used by the full solve routines.
    Eigen::VectorXd build_ci_diagonal(
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        int n_act);

    // Dense CI solve used when the determinant space is small enough to diagonalize
    // directly.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_ci_dense(
        const Eigen::MatrixXd &H,
        int nroots,
        double tol = 1e-10);

    // Convenience wrapper for direct dense/iterative solve selection when the full
    // Hamiltonian matrix is already available.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_ci(
        const Eigen::MatrixXd &H,
        int nroots,
        double tol = 1e-10,
        int dense_threshold = 500);

    // Iterative CI solve that applies the Hamiltonian through a sigma callback.
    // This is the large-space fallback when explicitly forming H would be too big.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_ci(
        int dim,
        int nroots,
        const Eigen::VectorXd &diag,
        const std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &)> &sigma_apply,
        double tol = 1e-10,
        int max_iter = 1000,
        int dense_threshold = 500);

    // Top-level CI solve wrapper that carries the cached diagonal/dense matrix
    // decision through to the result object.
    CISolveResult solve_ci(
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        int n_act,
        int nroots,
        double tol = 1e-10,
        int dense_threshold = 500,
        int max_iter = 1000);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_CI_H
