#ifndef HF_POSTHF_CASSCF_RDM_H
#define HF_POSTHF_CASSCF_RDM_H

#include "post_hf/casscf/strings.h"

#include <Eigen/Core>

namespace HartreeFock::Correlation::CASSCF
{

Eigen::MatrixXd compute_1rdm(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

std::vector<double> compute_2rdm(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

std::vector<double> compute_2rdm_bilinear(
    const Eigen::MatrixXd& bra_vecs,
    const Eigen::MatrixXd& ket_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

Eigen::MatrixXd compute_1rdm_reference(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

std::vector<double> compute_2rdm_reference(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

std::vector<double> compute_2rdm_bilinear_reference(
    const Eigen::MatrixXd& bra_vecs,
    const Eigen::MatrixXd& ket_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_RDM_H
