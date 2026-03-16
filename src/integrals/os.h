#ifndef HF_OS_H
#define HF_OS_H

#include <array>
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

        // ── Gradient derivative integrals ──────────────────────────────────────────

        // Returns {dS/dAx, dS/dAy, dS/dAz, dT/dAx, dT/dAy, dT/dAz}
        // GTO-centre derivative of one contracted (μ,ν) shell pair (AM shift rule).
        std::array<double,6> _compute_1e_deriv_A(const HartreeFock::ShellPair& sp);

        // Returns {dV/dAx, dV/dAy, dV/dAz}
        // Nuclear-attraction GTO-centre derivative (sums over all nuclei in mol).
        std::array<double,3> _compute_nuclear_deriv_A_elem(
            const HartreeFock::ShellPair& sp,
            const HartreeFock::Molecule& mol);

        // Returns contracted dV_μν/dC_{direction} for one nucleus at C with charge Z.
        // direction: 0=x, 1=y, 2=z
        double _compute_nuclear_deriv_C_elem(
            const HartreeFock::ShellPair& sp,
            const Eigen::Vector3d& C, double Z, int direction);

        // Returns flat array of ERI derivatives for one (μν|λσ) contracted quartet.
        // Layout: [cen*3 + dir], cen∈{0=A,1=B,2=C,3=D}, dir∈{0,1,2}
        std::array<double,12> _compute_eri_deriv_elem(
            const HartreeFock::ShellPair& spAB,
            const HartreeFock::ShellPair& spCD);

        // Compute the cross-overlap matrix S_cross(μ, ν) = <χ_μ^large | χ_ν^small>
        // between two basis sets centered on the same molecule.
        // Result has dimensions nbasis_large × nbasis_small.
        Eigen::MatrixXd _compute_cross_overlap(const HartreeFock::Basis& large_basis,
                                               const HartreeFock::Basis& small_basis);
    }
}

#endif // !HF_OS_H
