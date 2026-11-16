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
    // Both builders take the density and MO coefficients in the SCF's working
    // basis. When `basis` is spherical they lift them to Cartesian once (density
    // C^T P C, MOs C^T C_mo), so every trial density, J/K/XC response and the
    // MO packing run in the Cartesian integral/grid basis with no further
    // transform: (C^T C_mo)^T M_cart (C^T C_mo) = C_mo^T to_working(M_cart) C_mo.
    // `basis` null means Cartesian. The operators keep their own copy of the
    // inputs; only the pointed-to objects must outlive them.

    // Shared RKS SOSCF orbital Hessian.
    struct KsOrbitalHessianInputs
    {
        const HartreeFock::Basis *basis = nullptr;
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
        // RI-JK (docs/RI_JK_ROUTINE_SCOPE.md step 14): non-null means the J and
        // K responses come from the fitted operator, built in the working basis
        // from the working MOs, as the RI-JK SCF does. Needs a primed RI cache;
        // full-range exchange only (range separation is refused under RI).
        const HartreeFock::Calculator *ri = nullptr;
    };

    [[nodiscard]] std::expected<std::function<Eigen::VectorXd(const Eigen::VectorXd &)>, std::string>
    build_ks_orbital_hessian_op(const KsOrbitalHessianInputs &in);

    // UKS spin-conserving orbital Hessian (A + B) acting on packed rotations
    // [alpha (a*n_occ_a + i) | beta (a*n_occ_b + i)]: orbital-energy
    // differences plus J from the total trial density, the polarized XC kernel
    // and -(c_fr K + c_sr K_SR) per spin, with dP_s = C_v x_s C_o^T + transpose.
    // d2E/dkappa2 = 2 * this (docs/SOSCF_DFT.md). Shared by UKS SOSCF and
    // UKS stability analysis.
    struct UksOrbitalHessianInputs
    {
        const HartreeFock::Basis *basis = nullptr;
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
        const HartreeFock::Calculator *ri = nullptr; // as KsOrbitalHessianInputs::ri
    };

    // The XC kernel is prepared once here (the ground densities are fixed), so
    // each product costs one trial-density grid pass, not a Libxc evaluation.
    [[nodiscard]] std::expected<std::function<Eigen::VectorXd(const Eigen::VectorXd &)>, std::string>
    build_uks_orbital_hessian_op(const UksOrbitalHessianInputs &in);
}
#endif
