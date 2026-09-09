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
    // Eq. 33's XC term is the XC_II piece ONLY (basis-only rho_P^(x), no
    // moving grid, no rho_D^(x)). N3.5.7.10 IDENTIFIED it against an exact
    // FD target on a low-symmetry C1 H2O2 fixture: XC_II scores cos 0.946 at
    // coefficient 1.078 and removes 69% of the missing term. Use kXcII.
    //
    // Measured, so do not re-litigate: XC_I is not in the answer (cos -0.54);
    // XC_III is ~1e-7 on real molecules, not just the synthetic He2 fixture;
    // the relaxed density D is correct (cos 0.93 vs the unrelaxed 0.75); and
    // the closed-shell spin factor is fine (Sec. II vs polarized R^XC, 1e-17).
    //
    // Every pre-N3.5.7.10 conclusion here was drawn on water/C2v, which has
    // only 3 independent gradient components -- fewer than the number of
    // candidate terms, so any three of them span the target exactly and no
    // decomposition is identifiable. Score candidates on the C1 fixture
    // (tests/inputs/exploratory/dh_gradient/h2o2_c1_b2plyp_gradient_fd.hfinp),
    // by cos in the translation-free subspace, never by max-norm on water.
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
