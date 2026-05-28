#ifndef HF_HGP_SYMM_H
#define HF_HGP_SYMM_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"
#include "symmetry/group_operations.h"

namespace HartreeFock
{
    namespace HeadGordonPople
    {
        std::expected<std::vector<double>, std::string> _build_skeleton_eri_symm(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            std::size_t nbasis,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);

        std::expected<Eigen::MatrixXd, std::string> _compute_2e_fock_symm(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            const Eigen::MatrixXd &density,
            std::size_t nbasis,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);

        std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
        _compute_2e_fock_uhf_symm(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            const Eigen::MatrixXd &Pa,
            const Eigen::MatrixXd &Pb,
            std::size_t nbasis,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);

        std::expected<Eigen::MatrixXd, std::string> _compute_2e_fock_symm_spherical(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            const Eigen::MatrixXd &density,
            std::size_t nbasis_cart,
            const Eigen::MatrixXd &cart_to_sph,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);

        std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
        _compute_2e_fock_uhf_symm_spherical(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            const Eigen::MatrixXd &Pa,
            const Eigen::MatrixXd &Pb,
            std::size_t nbasis_cart,
            const Eigen::MatrixXd &cart_to_sph,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);
    } // namespace HeadGordonPople
} // namespace HartreeFock

#endif // !HF_HGP_SYMM_H
