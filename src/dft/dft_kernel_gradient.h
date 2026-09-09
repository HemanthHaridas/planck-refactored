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
    // (Neese/Schwabe/Grimme, JCP 126, 124115, 2007). It is a response-type
    // term: the SCF operator already carries the first XC-potential
    // derivative, and E_PT2 is not stationary w.r.t. the SCF density, so
    // differentiating it pulls down the SECOND (Term 2) and THIRD (Term 1)
    // functional derivatives of the XC functional, contracted with the
    // RELAXED PT2 difference density.
    //
    // Both terms are DIRECT per-point scalar grid integrals -- NOT an
    // AO-pair scatter (the first attempt, deleted at S0, mirrored
    // compute_xc_nuclear_gradient_rks's vtmp/vxc1 machinery against P and
    // made the FD residual worse).
    //
    //   Term 1 (closed-shell LDA, this routine's only branch for now):
    //     integral  rho_P^(x)(r) * [ v3rho3(rho_P; r) * rho_D(r) ]  dr
    //   with rho_P^(x)(r) the BASIS-function derivative of the density at a
    //   FIXED spatial point (Eq. 15), v3rho3 = f^{rho rho rho} at the
    //   GROUND (SCF) density, and rho_D(r) from the relaxed difference
    //   density -- both frozen per-point scalars.
    //
    // rho_P^(x) is the integrand itself, so this is a plain grid integral,
    // NOT d/dR of an integral -- there is NO moving-grid (Becke-weight /
    // point-translation) correction (those belong to E_xc^x, where E_xc is
    // the thing being differentiated). Only the basis-function-derivative
    // piece, drho_channel, at the current geometry.
    //
    //   Term 1 GGA (v3rho2sigma/v3rhosigma2/v3sigma3, plus the gamma(P)
    //   branch weighted by grad_rho_P^(x) via dg_axis_spin) and Term 2
    //   (f^(2) v2sigma contracting grad_rho_P^(x) with grad_rho_D) are S3
    //   -- the GGA branch here returns an explicit error until then.
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
