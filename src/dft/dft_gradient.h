#ifndef DFT_GRADIENT_H
#define DFT_GRADIENT_H

#include <expected>
#include <string>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/grid.h"
#include "base/types.h"
#include "xc_grid.h"

namespace DFT::Gradient
{

    // d(Becke partition weight of grid point `ip`'s owner atom)/dR, for
    // every nuclear Cartesian -- the core moving-grid correction analytic
    // DFT gradients need. Shared by compute_xc_nuclear_gradient_{rks,uks}
    // and (N3.5.7.3, docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md)
    // compute_xc_kernel_nuclear_gradient -- exported here (was file-local to
    // dft_gradient.cpp) rather than duplicated a third time, since this is
    // real shared Becke-weight-derivative algebra, not per-TU private
    // wiring like the small shell/atom bookkeeping helpers each of those
    // TUs still keeps its own copy of.
    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string> becke_partition_owner_derivatives(
        const MolecularGrid &grid,
        const HartreeFock::Molecule &mol,
        Eigen::Index ip);

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
