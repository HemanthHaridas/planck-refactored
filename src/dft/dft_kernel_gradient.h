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
    // *** NOT WIRED INTO PRODUCTION *** -- see S5 / N3.5.7.8 in the scope
    // doc. Eq. 33's XC term is the XC_II piece ONLY (basis-only rho_P^(x),
    // no moving grid, no rho_D^(x)); XC_I blows the end-to-end B2PLYP
    // gradient up (~3.5e-3). XC_II supplies one component of the residual
    // almost exactly but not the z structure. Measured (N3.5.7.8), so do
    // not re-litigate: XC_III is ~1e-7 on real water, not just on the
    // synthetic He2 fixture; the closed-shell spin factor is NOT the
    // problem (Sec. II derivation cross-checked against the polarized
    // R^XC to 1e-17); and no single scale factor closes the gap (per-
    // component ratios -5.29 / 1.11 / -0.52), so the c_pt2 = 0.27 "fit"
    // was a max-norm coincidence. The driver reaches this routine only
    // via the PLANCK_DFT_DH_XC_PARTS probe hook.
    //
    // `ground_density_restricted` is the SCF (KS) density P;
    // `relaxed_density_restricted` is the relaxed PT2 difference density D
    // (dm1_corr_relaxed_ao). Both are the full closed-shell (RKS) matrices.
    //
    // `parts` selects which of the three pieces to accumulate. Default = all
    // three (the full d/dR{Phi_XC} the FD gate verifies). Eq. 33's XC term is
    // XC_II alone; the selector exists so an end-to-end probe can isolate the
    // pieces without env vars or edit-rebuild cycles.
    enum XcPart : unsigned
    {
        kXcI = 1u,
        kXcII = 2u,
        kXcIII = 4u,
        // XC_II's point-translation companion: the owner-atom scatter that
        // makes the XC_II piece translationally invariant on its own.
        kXcIIt = 8u,
        kXcAll = kXcI | kXcII | kXcIII,
    };

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
        const XC::Functional &correlation_functional,
        unsigned parts = kXcAll);

} // namespace DFT::Gradient

#endif
