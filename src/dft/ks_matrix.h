#ifndef DFT_KS_MATRIX_H
#define DFT_KS_MATRIX_H

#include <expected>
#include <string>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/grid.h"
#include "xc_grid.h"

namespace DFT
{

struct XCMatrixContribution
{
    bool polarized = false;
    Eigen::MatrixXd alpha;
    Eigen::MatrixXd beta;
};

struct KSPotentialMatrices
{
    bool polarized = false;

    Eigen::MatrixXd coulomb;
    Eigen::MatrixXd xc_alpha;
    Eigen::MatrixXd xc_beta;
    Eigen::MatrixXd alpha;
    Eigen::MatrixXd beta;
};

std::expected<XCMatrixContribution, std::string>
assemble_xc_matrix(
    const MolecularGrid& molecular_grid,
    const AOGridEvaluation& ao_grid,
    const XCGridEvaluation& xc_grid);

KSPotentialMatrices combine_ks_potential(
    const Eigen::Ref<const Eigen::MatrixXd>& coulomb,
    const XCMatrixContribution& xc_matrix);

} // namespace DFT

#endif // DFT_KS_MATRIX_H
