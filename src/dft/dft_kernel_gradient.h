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
    // N3.5.7 (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md): the XC
    // contribution to the double-hybrid PT2 gradient -- Eq. 33
    // (Neese/Schwabe/Grimme, JCP 126, 124115, 2007). It is the geometry
    // derivative of
    //
    //   Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>
    //          = integral w * { (df/drho)*rho_D + 2*(df/dgamma)*(grad_rho_P . grad_rho_D) } dr
    //
    // where <mu|V_xc|nu> is the XC part of the SCF operator (Eq. 10, FIRST
    // functional derivatives) and D is the RELAXED PT2 difference density.
    // The paper's own text (p.6): "a response-type term arises which
    // requires the evaluation of the SECOND functional derivative of the XC
    // functional" -- there is NO third derivative here (the "f^{...zeta}"
    // notation in Eq. 33 is d^2 f / d rho_sigma d zeta, i.e. v2*, matching
    // Eq. 41's own response operator). So the S1 kxc wrapper is not needed
    // for this term (it stays as a correct utility for future use).
    //
    // The response part (rho_P, grad_rho_P move via their basis-function
    // derivative at a FIXED spatial point; rho_D, grad_rho_D frozen):
    //
    //   LDA:  integrand(A,q) = rho_P^(x) * v2rho2 * rho_D
    //   GGA:  integrand(A,q) =
    //             [ v2rho2*rx + 2*v2rhosigma*(g.gx) ] * rho_D
    //           + 2*[ v2rhosigma*rx + 2*v2sigma2*(g.gx) ] * (g.grad_rho_D)
    //           + 2*vsigma * (gx . grad_rho_D)
    //   with rx = rho_P^(x) (drho_channel), gx = grad_rho_P^(x)
    //   (dg_axis_spin), g = grad_rho_P (SCF gradient, frozen here).
    //
    // Plain grid integrals of a basis-derivative integrand -- NO moving-grid
    // (Becke-weight / point-translation) correction. The first attempt
    // (deleted at S0) scattered AO-pair derivatives against P and made the
    // FD residual worse. Term 1/Term 2 alone are NOT translationally
    // invariant; sum_A grad_A = 0 holds only for the full E_PT2^x (S4).
    //
    // `ground_density_restricted` is the SCF (KS) density P;
    // `relaxed_density_restricted` is the relaxed PT2 difference density D
    // (dm1_corr_relaxed_ao). Both are the full closed-shell (RKS) matrices.
    // Same drop_correlation_if_combined guard as
    // compute_analytic_xc_hessian_vector_product (combined XC libxc entries
    // must not double-count the correlation kernel).
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
