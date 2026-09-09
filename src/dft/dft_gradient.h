#ifndef DFT_GRADIENT_H
#define DFT_GRADIENT_H

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/grid.h"
#include "base/types.h"
#include "basis/basis.h"
#include "xc_grid.h"

namespace DFT::Gradient
{

    // d(Becke partition weight of grid point `ip`'s owner atom)/dR, for
    // every nuclear Cartesian -- the core moving-grid correction analytic
    // DFT gradients need. Shared by compute_xc_nuclear_gradient_{rks,uks}
    // and (N3.5.7, docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md)
    // compute_dh_xc_pt2_gradient -- exported here (was file-local to
    // dft_gradient.cpp) rather than duplicated a third time, since this is
    // real shared Becke-weight-derivative algebra.
    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string> becke_partition_owner_derivatives(
        const MolecularGrid &grid,
        const HartreeFock::Molecule &mol,
        Eigen::Index ip);

    // Per-atom lists of basis-function indices (flat AO list -> shells ->
    // atoms). The moving-grid derivative terms are organized atom-by-atom;
    // this recovers that structure. Shared with compute_dh_xc_pt2_gradient
    // (N3.5.7) -- a single wrapper over what was three file-local steps.
    [[nodiscard]] std::expected<std::vector<std::vector<int>>, std::string>
    atom_bf_lists(const HartreeFock::Molecule &mol, const HartreeFock::Basis &basis);

    // Basis-function-derivative piece of d(rho_P(r))/dR_{atom_A,q} at grid
    // point `ip`: sum over mu on atom A of -2 P_sym(mu,nu) phi_nu (d phi_mu/dq).
    // (The point-translation and Becke-weight pieces are separate, added by
    // the caller.) `P_sym` must already be symmetrized. Shared with
    // compute_dh_xc_pt2_gradient (N3.5.7); it consumes rho_P^(x) as a scalar
    // grid integrand rather than scattering it AO-pair-wise like
    // compute_xc_nuclear_gradient_rks does.
    [[nodiscard]] double drho_channel(
        const Eigen::MatrixXd &P_sym,
        const AOGridEvaluation &ao,
        Eigen::Index ip,
        int atom_A,
        int q,
        const std::vector<std::vector<int>> &atoms_bf);

    // Basis-function-derivative of the `axis_g` component of grad_rho_P(r)
    // w.r.t. nuclear Cartesian (atom_A, q) at grid point `ip` -- i.e.
    // [grad_rho_P^(x)]_{axis_g}. Needs AO Hessians. `P_sym` symmetrized.
    // Shared with compute_dh_xc_pt2_gradient's GGA branch (N3.5.7 S3).
    [[nodiscard]] double dg_axis_spin(
        const Eigen::MatrixXd &P_sym,
        const AOGridEvaluation &ao,
        const AOGridHessian &hess,
        Eigen::Index ip,
        int axis_g,
        int atom_A,
        int q,
        const std::vector<std::vector<int>> &atoms_bf);

    // Nuclear XC contribution ∂E_xc/∂R for semilocal and global-hybrid KS-DFT,
    // including the moving-grid (Becke partition + point-translation) response.
    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string>
    compute_xc_nuclear_gradient_rks(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const AOGridHessian &hess,
        const XCGridEvaluation &xc,
        const Eigen::Ref<const Eigen::MatrixXd> &density_restricted);

    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string>
    compute_xc_nuclear_gradient_uks(
        const HartreeFock::Molecule &mol,
        const HartreeFock::Basis &basis,
        const MolecularGrid &grid,
        const AOGridEvaluation &ao,
        const AOGridHessian &hess,
        const XCGridEvaluation &xc,
        const Eigen::Ref<const Eigen::MatrixXd> &density_alpha,
        const Eigen::Ref<const Eigen::MatrixXd> &density_beta);

} // namespace DFT::Gradient

#endif
