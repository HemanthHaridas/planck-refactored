#ifndef DFT_KS_ORBITAL_HESSIAN_H
#define DFT_KS_ORBITAL_HESSIAN_H

#include <functional>
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
}
#endif
