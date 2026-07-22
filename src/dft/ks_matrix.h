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
        Eigen::MatrixXd exact_exchange_alpha;
        Eigen::MatrixXd exact_exchange_beta;
        Eigen::MatrixXd alpha;
        Eigen::MatrixXd beta;
        double exact_exchange_coefficient = 0.0;
        double exact_exchange_energy = 0.0;
    };

    // point_begin/point_end restrict assembly to a contiguous grid-point slice
    // (for MPI rank partitioning); point_end < 0 (the default) means all points,
    // keeping every serial/test call site byte-identical.
    std::expected<XCMatrixContribution, std::string>
    assemble_xc_matrix(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const XCGridEvaluation &xc_grid,
        Eigen::Index point_begin = 0,
        Eigen::Index point_end = -1);

    KSPotentialMatrices combine_ks_potential(
        const Eigen::Ref<const Eigen::MatrixXd> &coulomb,
        const XCMatrixContribution &xc_matrix,
        double exact_exchange_coefficient = 0.0,
        const Eigen::MatrixXd &exact_exchange_alpha = Eigen::MatrixXd(),
        const Eigen::MatrixXd &exact_exchange_beta = Eigen::MatrixXd(),
        double exact_exchange_energy = 0.0);

} // namespace DFT

#endif // DFT_KS_MATRIX_H
