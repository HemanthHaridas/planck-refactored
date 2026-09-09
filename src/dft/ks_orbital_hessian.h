#ifndef DFT_KS_ORBITAL_HESSIAN_H
#define DFT_KS_ORBITAL_HESSIAN_H

#include <functional>
#include <vector>

#include <Eigen/Dense>

#include "base/types.h"
#include "base/wrapper.h"
#include "base/grid.h"
#include "ao_grid.h"

// The KS orbital Hessian as a matrix-free operator, in CPHF (a,i) ordering
// idx(a,i) = a*n_occ + i -- the SAME ordering pack_hessian_vector_product_cphf_order
// and HartreeFock::Correlation::build_rhf_cphf_matrix use.
//
//   h_op(x) = diag_term(x)  +  s * (J_packed(x) + xc_packed(x) + K_packed(x))
//
// This is exactly the RKS SOSCF h_op (src/dft/driver.cpp, docs/SOSCF_DFT.md
// invariant 2/3), lifted out of the SCF loop so the analytic double-hybrid
// gradient's Z-vector solve can use the same operator. Restricted / RKS only;
// UKS is a later step (polarized fxc + per-spin packing).
//
//   s = 2 for RKS (docs/SOSCF_DFT.md invariant 2).
//   J_packed  = pack(delta_J[dP]),   dP = C_virt*x_mat*C_occ^T + h.c.
//   xc_packed = pack(compute_analytic_xc_hessian_vector_product(...))
//   K_packed  = pack(-0.5 * (c_fr*delta_K_Coulomb + c_sr*delta_K_ShortRange)),
//               only when c_fr != 0 or c_sr != 0 (hybrids).
namespace DFT::Driver
{
    struct KsOrbitalHessianInputs
    {
        const std::vector<HartreeFock::ShellPair> *shell_pairs = nullptr;
        const MolecularGrid *molecular_grid = nullptr;
        const AOGridEvaluation *ao_grid = nullptr;

        // Converged KS total density (AO), and the occupied/virtual MO blocks
        // + orbital energies of the reference the Z-vector rotates about.
        Eigen::MatrixXd density;
        Eigen::MatrixXd C_occ;
        Eigen::MatrixXd C_virt;
        Eigen::VectorXd eps; // full length n_occ + n_virt

        const XC::Functional *x_functional = nullptr;
        const XC::Functional *c_functional = nullptr;

        HartreeFock::IntegralMethod engine = HartreeFock::IntegralMethod::ObaraSaika;
        double tol_eri = 1e-10;
        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr;

        // Hybrid exact-exchange coefficients (0 for a pure functional).
        double full_range_exchange_coefficient = 0.0;
        double short_range_exchange_coefficient = 0.0;
        double range_separation_omega = 0.0;

        double kernel_scale = 2.0; // s; 2 for RKS
    };

    // Returns h_op. The returned closure holds a reference to `in`, so `in`
    // must outlive it. On an internal XC-HVP failure the closure returns a
    // zero vector (same degradation the SOSCF branch uses).
    [[nodiscard]] std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
    build_ks_orbital_hessian_op(const KsOrbitalHessianInputs &in);

} // namespace DFT::Driver

#endif // DFT_KS_ORBITAL_HESSIAN_H
