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
        struct IterationMetrics
        {
            double delta_energy = 0.0;
            double delta_density_max = 0.0;
            double delta_density_rms = 0.0;
        };

        struct RestrictedIterationData
        {
            Eigen::MatrixXd density;
            Eigen::MatrixXd fock;
            Eigen::VectorXd mo_energies;
            Eigen::MatrixXd mo_coefficients;
            double electronic_energy = 0.0;
            double total_energy = 0.0;
        };

        struct UnrestrictedIterationData
        {
            Eigen::MatrixXd alpha_density;
            Eigen::MatrixXd beta_density;
            Eigen::MatrixXd alpha_fock;
            Eigen::MatrixXd beta_fock;
            Eigen::VectorXd alpha_mo_energies;
            Eigen::VectorXd beta_mo_energies;
            Eigen::MatrixXd alpha_mo_coefficients;
            Eigen::MatrixXd beta_mo_coefficients;
            double electronic_energy = 0.0;
            double total_energy = 0.0;
        };

        // Build the symmetric orthogonalization matrix X = S^{-1/2}.
        // Returns an error string if S is singular (smallest eigenvalue < threshold).
        std::expected<Eigen::MatrixXd, std::string>
        build_orthogonalizer(const Eigen::MatrixXd &S, double threshold = 1e-8);

        // Form the initial density matrix from the core Hamiltonian:
        //   diagonalize X^T * H * X, occupy the lowest n_occ orbitals.
        Eigen::MatrixXd initial_density(const Eigen::MatrixXd &H,
                                        const Eigen::MatrixXd &X,
                                        std::size_t n_occ);

        // Symmetry-adapted initial RHF density for the SAO-blocked path:
        // diagonalize the core Hamiltonian in each orthonormal SAO irrep block,
        // then occupy the globally lowest-energy symmetry-adapted orbitals.
        Eigen::MatrixXd initial_density_sao(const Eigen::MatrixXd &H,
                                            const Eigen::MatrixXd &U,
                                            const std::vector<int> &block_sizes,
                                            const std::vector<int> &block_offsets,
                                            std::size_t n_occ);

        IterationMetrics restricted_iteration_metrics(
            const Eigen::MatrixXd &previous_density,
            const Eigen::MatrixXd &next_density,
            double previous_total_energy,
            double total_energy);

        IterationMetrics unrestricted_iteration_metrics(
            const Eigen::MatrixXd &previous_alpha_density,
            const Eigen::MatrixXd &previous_beta_density,
            const Eigen::MatrixXd &next_alpha_density,
            const Eigen::MatrixXd &next_beta_density,
            double previous_total_energy,
            double total_energy);

        bool is_converged(
            const HartreeFock::OptionsSCF &scf_options,
            const IterationMetrics &metrics,
            unsigned int iteration) noexcept;

        void store_restricted_iteration(
            HartreeFock::Calculator &calculator,
            const RestrictedIterationData &iteration,
            const IterationMetrics &metrics);

        void store_unrestricted_iteration(
            HartreeFock::Calculator &calculator,
            const UnrestrictedIterationData &iteration,
            const IterationMetrics &metrics);

        // Run the RHF SCF procedure.
        // Stores converged density, Fock, MO energies/coefficients in calculator._info._scf.
        // Returns an error string if convergence is not achieved.
        std::expected<void, std::string> run_rhf(HartreeFock::Calculator &calculator,
                                                 const std::vector<HartreeFock::ShellPair> &shell_pairs);

        // Run the UHF SCF procedure.
        // Uses molecule.multiplicity to derive n_alpha and n_beta.
        // Stores converged alpha and beta channels in calculator._info._scf.
        // Returns an error string if convergence is not achieved.
        std::expected<void, std::string> run_uhf(HartreeFock::Calculator &calculator,
                                                 const std::vector<HartreeFock::ShellPair> &shell_pairs);

        // Run the ROHF SCF procedure.
        // Uses molecule.multiplicity to derive the closed/open-shell occupations.
        // Stores alpha/beta densities and one shared MO coefficient set in both channels.
        // Returns an error string if convergence is not achieved.
        std::expected<void, std::string> run_rohf(HartreeFock::Calculator &calculator,
                                                  const std::vector<HartreeFock::ShellPair> &shell_pairs);
    } // namespace SCF
} // namespace HartreeFock

#endif // !HF_SCF_H
