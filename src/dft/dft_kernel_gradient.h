#ifndef DFT_KERNEL_GRADIENT_H
#define DFT_KERNEL_GRADIENT_H

#include <expected>
#include <string>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/grid.h"
#include "base/types.h"
#include "base/wrapper.h"

namespace DFT::Gradient
{
    // N3.5.7 (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md): the FULL
    // geometry derivative of
    //
    //   Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>
    //          = integral w * { vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D) } dr
    //
    // (<mu|V_xc|nu> = the SCF operator's XC part, Eq. 10, FIRST functional
    // derivatives; D = relaxed PT2 difference density). Since Phi_XC is a
    // grid integral, d/dR is a true geometry derivative with three pieces
    // -- Python/PySCF-validated to rel 3e-9:
    //
    //   XC_I   basis-function derivative of rho_D (drho_channel(D),
    //          dg_axis_spin(D)) against the FIRST XC derivatives:
    //            w [ vrho*rho_D^(x) + 2*vsigma*(grad_rho_P . grad_rho_D^(x)) ]
    //          -- ~85% of d/dR{Phi_XC}
    //   XC_II  rho_P inside V_xc[rho_P] responds (drho_channel(P),
    //          dg_axis_spin(P)) against the SECOND XC derivatives:
    //            w [ (v2rho2*rx + 2*v2rhosigma*(g.gx))*rho_D
    //              + 2*(v2rhosigma*rx + 2*v2sigma2*(g.gx))*(g.grad_rho_D)
    //              + 2*vsigma*(gx.grad_rho_D) ]
    //          -- ~15%
    //   XC_III grid quadrature moving frame: Becke partition weight
    //          response + point translation (owner atom), integrand
    //            I_p = vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D)
    //          -- ~0.1%, load-bearing for translational invariance
    //
    // rx = drho_channel(P), gx_a = dg_axis_spin(P), g = grad_rho_P.
    // All first / second derivatives at the GROUND density; rho_P >= 1e-8
    // screened; combined-XC guard on the _c arrays.
    //
    // *** NOT WIRED INTO THE DRIVER *** -- see S5 in the scope doc. Eq. 33's
    // XC term is the XC_II piece ONLY (basis-only rho_P^(x), no moving grid,
    // no rho_D^(x)); XC_I blows the end-to-end B2PLYP gradient up (~3.5e-3),
    // and XC_II alone overshoots the ~1.9e-4 residual ~2x with a
    // per-component sign structure -- unresolved (likely a closed-shell
    // spin factor: v2rho2_aa vs v2rho2_unpol differ by v2rho2_ab/2 for
    // correlation). This routine and its FD gate are kept as the validated
    // building block for the eventual fix.
    //
    // `ground_density_restricted` is the SCF (KS) density P;
    // `relaxed_density_restricted` is the relaxed PT2 difference density D
    // (dm1_corr_relaxed_ao). Both are the full closed-shell (RKS) matrices.
    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string>
    compute_dh_xc_pt2_gradient(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const AOGridHessian &hess,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_restricted,
        const Eigen::Ref<const Eigen::MatrixXd> &relaxed_density_restricted,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

} // namespace DFT::Gradient

#endif
