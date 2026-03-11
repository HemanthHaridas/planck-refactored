#ifndef HF_SCF_H
#define HF_SCF_H

#include <Eigen/Core>
#include <expected>
#include <string>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock
{
    namespace SCF
    {
        // Build the symmetric orthogonalization matrix X = S^{-1/2}.
        // Returns an error string if S is singular (smallest eigenvalue < threshold).
        std::expected<Eigen::MatrixXd, std::string>
        build_orthogonalizer(const Eigen::MatrixXd& S, double threshold = 1e-8);

        // Form the initial density matrix from the core Hamiltonian:
        //   diagonalize X^T * H * X, occupy the lowest n_occ orbitals.
        Eigen::MatrixXd initial_density(const Eigen::MatrixXd& H,
                                        const Eigen::MatrixXd& X,
                                        std::size_t n_occ);

        // Run the RHF SCF procedure.
        // Stores converged density, Fock, MO energies/coefficients in calculator._info._scf.
        // Returns an error string if convergence is not achieved.
        std::expected<void, std::string> run_rhf(HartreeFock::Calculator& calculator,
                                                  const std::vector<HartreeFock::ShellPair>& shell_pairs);

        // Run the UHF SCF procedure.
        // Uses molecule.multiplicity to derive n_alpha and n_beta.
        // Stores converged alpha and beta channels in calculator._info._scf.
        // Returns an error string if convergence is not achieved.
        std::expected<void, std::string> run_uhf(HartreeFock::Calculator& calculator,
                                                  const std::vector<HartreeFock::ShellPair>& shell_pairs);
    }
}

#endif // !HF_SCF_H
