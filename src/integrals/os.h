#ifndef HF_OS_H
#define HF_OS_H

#include <tuple>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "shellpair.h"
#include "base/types.h"

namespace HartreeFock
{
    namespace ObaraSaika
    {
        double _os_1d(const double gamma, const double distPA, const double distPB, const int lA, const int lB);
        std::tuple<double, double> _compute_3d_overlap_kinetic(const HartreeFock::ShellPair &shell_pair);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_1e(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis);
        Eigen::MatrixXd _compute_nuclear_attraction(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis, const HartreeFock::Molecule &molecule);

        // Build the two-electron Fock contribution G = J - 0.5*K.
        // Iterates over all unique shell-pair quartets and accumulates Coulomb
        // and exchange contributions using 8-fold permutation symmetry.
        Eigen::MatrixXd _compute_2e_fock(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                         const Eigen::MatrixXd& density,
                                         std::size_t nbasis);

        // UHF variant: returns {G_alpha, G_beta}.
        // Builds the ERI tensor once (spin-independent) and contracts with
        // Pa, Pb, and Pt = Pa+Pb to form both spin Fock contributions.
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                             const Eigen::MatrixXd& Pa,
                             const Eigen::MatrixXd& Pb,
                             std::size_t nbasis);

        // Compute the cross-overlap matrix S_cross(μ, ν) = <χ_μ^large | χ_ν^small>
        // between two basis sets centered on the same molecule.
        // Result has dimensions nbasis_large × nbasis_small.
        Eigen::MatrixXd _compute_cross_overlap(const HartreeFock::Basis& large_basis,
                                               const HartreeFock::Basis& small_basis);
    }
}

#endif // !HF_OS_H
