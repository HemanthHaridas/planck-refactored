#ifndef HF_POSTHF_CASSCF_ORBITAL_H
#define HF_POSTHF_CASSCF_ORBITAL_H

#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <vector>

namespace HartreeFock::Correlation::CASSCF
{

using HartreeFock::Correlation::CASSCFInternal::ActiveIntegralCache;

struct RotPair
{
    int p = 0;
    int q = 0;
};

ActiveIntegralCache build_active_integral_cache(
    const std::vector<double>& eri,
    const Eigen::MatrixXd& C,
    int n_core,
    int n_act,
    int nbasis);

Eigen::MatrixXd compute_Q_matrix(
    const ActiveIntegralCache& cache,
    const std::vector<double>& Gamma);

Eigen::MatrixXd build_inactive_fock_mo(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& H_core,
    const std::vector<double>& eri,
    int n_core,
    int nbasis);

Eigen::MatrixXd build_active_fock_mo(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& gamma,
    const std::vector<double>& eri,
    int n_core,
    int n_act,
    int nbasis);

double compute_core_energy(
    const Eigen::MatrixXd& h_mo,
    const Eigen::MatrixXd& F_I_mo,
    int n_core);

Eigen::MatrixXd compute_orbital_gradient(
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& gamma,
    int n_core,
    int n_act,
    int n_virt,
    const std::vector<int>& mo_irreps,
    bool use_sym);

double hess_diag(const Eigen::MatrixXd& F_sum, int p, int q);

Eigen::MatrixXd hessian_action(
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core,
    int n_act,
    int n_virt);

Eigen::MatrixXd fep1_gradient_update(
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core,
    int n_act,
    int n_virt);

double quadratic_model_delta(
    const Eigen::VectorXd& g_flat,
    const Eigen::VectorXd& h_flat,
    const Eigen::VectorXd& x);

std::vector<RotPair> non_redundant_pairs(int n_core, int n_act, int n_virt);

Eigen::MatrixXd augmented_hessian_step(
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core,
    int n_act,
    int n_virt,
    double level_shift,
    double max_rot,
    const std::vector<int>& mo_irreps,
    bool use_sym);

Eigen::MatrixXd apply_orbital_rotation(
    const Eigen::MatrixXd& C_old,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& S);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_ORBITAL_H
