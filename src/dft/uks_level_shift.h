#ifndef DFT_UKS_LEVEL_SHIFT_H
#define DFT_UKS_LEVEL_SHIFT_H

#include <Eigen/Core>
#include <expected>
#include <string>

namespace DFT::Driver
{
    // Occupancy-one spin density. This raises only the virtual subspace
    // of an idempotent P: C^T shift C = lambda * diag(0_occ, 1_virt).
    // SCF iteration aid only; never an energy term or a PT2 level shift.
    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string>
    build_uks_level_shift_matrix(
        const Eigen::Ref<const Eigen::MatrixXd> &overlap,
        const Eigen::Ref<const Eigen::MatrixXd> &spin_density,
        double level_shift);

    // Final post-SCF boundary: check the physical, unshifted Fock eigenpair
    // and metric normalization. Supplying shifted orbital energies fails.
    [[nodiscard]] std::expected<void, std::string>
    validate_uks_unshifted_orbitals(
        const Eigen::Ref<const Eigen::MatrixXd> &physical_fock,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap,
        const Eigen::Ref<const Eigen::MatrixXd> &coefficients,
        const Eigen::Ref<const Eigen::VectorXd> &energies,
        double tolerance = 1e-9);
}

#endif
