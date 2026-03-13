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
        // Build the full AO ERI tensor. Applies Schwarz screening:
        // quartets with Q(i,j)·Q(k,l) < tol_eri are skipped.
        std::vector <double> _compute_2e(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                         std::size_t nbasis,
                                         double tol_eri = 1e-10);

        Eigen::MatrixXd _compute_fock_rhf(const std::vector<double> &_eri,
                                          const Eigen::MatrixXd &density,
                                          const std::size_t nbasis);

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_fock_uhf(const std::vector<double>& _eri,
                          const Eigen::MatrixXd& Pa, const Eigen::MatrixXd& Pb,
                          std::size_t nbasis);

        // Build the two-electron Fock contribution G = J - 0.5*K (direct SCF).
        // Applies Schwarz screening before each _contracted_eri call.
        Eigen::MatrixXd _compute_2e_fock(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                         const Eigen::MatrixXd& density,
                                         std::size_t nbasis,
                                         double tol_eri = 1e-10);

        // UHF direct-SCF variant: returns {G_alpha, G_beta}.
        // Applies Schwarz screening; builds the ERI tensor once per call.
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                             const Eigen::MatrixXd& Pa,
                             const Eigen::MatrixXd& Pb,
                             std::size_t nbasis,
                             double tol_eri = 1e-10);

        // Compute the cross-overlap matrix S_cross(μ, ν) = <χ_μ^large | χ_ν^small>
        // between two basis sets centered on the same molecule.
        // Result has dimensions nbasis_large × nbasis_small.
        Eigen::MatrixXd _compute_cross_overlap(const HartreeFock::Basis& large_basis,
                                               const HartreeFock::Basis& small_basis);
    }
}

#endif // !HF_OS_H
