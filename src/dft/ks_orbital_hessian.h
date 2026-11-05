#ifndef DFT_KS_ORBITAL_HESSIAN_H
#define DFT_KS_ORBITAL_HESSIAN_H

#include <expected>
#include <functional>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "base/types.h"
#include "base/wrapper.h"
#include "base/grid.h"
#include "ao_grid.h"

namespace DFT::Driver
{
    // Shared RKS SOSCF orbital Hessian; retained independently of the removed
    // double-hybrid derivative path.
    struct KsOrbitalHessianInputs
    {
        const std::vector<HartreeFock::ShellPair> *shell_pairs = nullptr;
        const MolecularGrid *molecular_grid = nullptr;
        const AOGridEvaluation *ao_grid = nullptr;
        Eigen::MatrixXd density, C_occ, C_virt;
        Eigen::VectorXd eps;
        const XC::Functional *x_functional = nullptr;
        const XC::Functional *c_functional = nullptr;
        HartreeFock::IntegralMethod engine = HartreeFock::IntegralMethod::ObaraSaika;
        double tol_eri = 1e-10;
        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr;
        double full_range_exchange_coefficient = 0.0;
        double short_range_exchange_coefficient = 0.0;
        double range_separation_omega = 0.0;
        double kernel_scale = 2.0;
    };

    [[nodiscard]] std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
    build_ks_orbital_hessian_op(const KsOrbitalHessianInputs &in);

    // UKS spin-conserving orbital Hessian (A + B) acting on packed rotations
    // [alpha (a*n_occ_a + i) | beta (a*n_occ_b + i)]: orbital-energy
    // differences plus J from the total trial density, the polarized XC kernel
    // and -(c_fr K + c_sr K_SR) per spin, with dP_s = C_v x_s C_o^T + transpose.
    // d2E/dkappa2 = 2 * this (docs/SOSCF_DFT.md). Shared by UKS SOSCF and
    // UKS stability analysis. `in` must outlive the returned operator.
    struct UksOrbitalHessianInputs
    {
        const std::vector<HartreeFock::ShellPair> *shell_pairs = nullptr;
        const MolecularGrid *molecular_grid = nullptr;
        const AOGridEvaluation *ao_grid = nullptr;
        Eigen::MatrixXd alpha_density, beta_density;
        Eigen::MatrixXd Ca_occ, Ca_virt, Cb_occ, Cb_virt;
        Eigen::VectorXd eps_a, eps_b;
        const XC::Functional *x_functional = nullptr;
        const XC::Functional *c_functional = nullptr;
        HartreeFock::IntegralMethod engine = HartreeFock::IntegralMethod::ObaraSaika;
        double tol_eri = 1e-10;
        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr;
        double full_range_exchange_coefficient = 0.0;
        double short_range_exchange_coefficient = 0.0;
        double range_separation_omega = 0.0;
    };

    // The XC kernel is prepared once here (the ground densities are fixed), so
    // each product costs one trial-density grid pass, not a Libxc evaluation.
    [[nodiscard]] std::expected<std::function<Eigen::VectorXd(const Eigen::VectorXd &)>, std::string>
    build_uks_orbital_hessian_op(const UksOrbitalHessianInputs &in);
}
#endif
