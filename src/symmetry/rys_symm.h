#ifndef HF_RYS_SYMM_H
#define HF_RYS_SYMM_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"
#include "symmetry/group_operations.h"

// ─── Rys-quadrature direct Fock with FULL point-group symmetry reduction ────────
//
// Phase 2 of the full-symmetry ERI reduction (docs/FULL_SYMMETRY_ERI_DESIGN.md).
// Rys analog of os_symm.h, kept separate so the OS and Rys symmetry-reduced paths
// can be A/B benchmarked against each other and against their production engines.
// The skeleton / petite-list / multiplicity / symmetrization logic is shared via
// symmetry/skeleton_eri.h; this engine only supplies RysQuad::_rys_contracted_eri.

namespace HartreeFock
{
    namespace RysQuad
    {
        // Build ONLY the (orbit-weighted) skeleton ERI tensor — density-independent
        // first half of _compute_2e_fock_symm; built once, reused across iterations
        // (C1, docs/FULL_SYMMETRY_PERF_SCOPE.md). See ObaraSaika::_build_skeleton_eri_symm.
        std::expected<std::vector<double>, std::string> _build_skeleton_eri_symm(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            std::size_t nbasis,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);

        // RHF direct Fock G = J − ½K via the skeleton + symmetrization scheme.
        std::expected<Eigen::MatrixXd, std::string> _compute_2e_fock_symm(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            const Eigen::MatrixXd &density,
            std::size_t nbasis,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);

        // UHF direct Fock {G_alpha, G_beta} via the skeleton + symmetrization scheme.
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

        // Spherical-mode full-symmetry Fock (Step 2) — see os_symm.h for semantics.
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
    } // namespace RysQuad
} // namespace HartreeFock

#endif // !HF_RYS_SYMM_H
