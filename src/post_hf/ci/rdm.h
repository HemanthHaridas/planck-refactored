#ifndef HF_POSTHF_CI_RDM_H
#define HF_POSTHF_CI_RDM_H

#include "post_hf/ci/strings.h"

#include <Eigen/Core>

namespace HartreeFock::Correlation::CI
{

    // The RDM builders all take the same determinant ordering and spin-string
    // lists so the weighted state-averaged contractions stay consistent.
    Eigen::MatrixXd compute_1rdm(
        const Eigen::MatrixXd &ci_vecs,
        const Eigen::VectorXd &weights,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

    // Build the full active-space 2-RDM in packed pqrs order.
    std::vector<double> compute_2rdm(
        const Eigen::MatrixXd &ci_vecs,
        const Eigen::VectorXd &weights,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

    // Bilinear form used for CI response terms and transition-like contractions.
    // The bra and ket CI spaces must share the same determinant ordering.
    std::vector<double> compute_2rdm_bilinear(
        const Eigen::MatrixXd &bra_vecs,
        const Eigen::MatrixXd &ket_vecs,
        const Eigen::VectorXd &weights,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

    // Reference implementations traverse the spin-orbital operator algebra more
    // explicitly and are mainly used to validate the production path.
    Eigen::MatrixXd compute_1rdm_reference(
        const Eigen::MatrixXd &ci_vecs,
        const Eigen::VectorXd &weights,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

    // Reference 2-RDM builder with explicit spin-orbital traversal.
    std::vector<double> compute_2rdm_reference(
        const Eigen::MatrixXd &ci_vecs,
        const Eigen::VectorXd &weights,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

    // Reference bilinear 2-RDM contraction used by tests to check response
    // symmetry and root-by-root reductions.
    std::vector<double> compute_2rdm_bilinear_reference(
        const Eigen::MatrixXd &bra_vecs,
        const Eigen::MatrixXd &ket_vecs,
        const Eigen::VectorXd &weights,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

} // namespace HartreeFock::Correlation::CI

// Backward-compatibility alias: see strings.h. CASSCF code references these CI
// symbols as CASSCF::<name>; re-export so existing call sites compile unchanged.
namespace HartreeFock::Correlation::CASSCF
{
    using namespace HartreeFock::Correlation::CI;
}

#endif // HF_POSTHF_CI_RDM_H
