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
