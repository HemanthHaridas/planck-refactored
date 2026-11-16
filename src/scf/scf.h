#ifndef HF_SCF_H
#define HF_SCF_H

#include <Eigen/Core>
#include <expected>
#include <string>

#include "base/types.h"
#include "integrals/shellpair.h"
#include "solvation/pcm.h"

namespace HartreeFock
{
    namespace SCF
    {
        struct IterationMetrics
        {
            double delta_energy = 0.0;
            double delta_density_max = 0.0;
            double delta_density_rms = 0.0;
            double diis_error = 0.0;  // 0 when DIIS inactive; then not gated on
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

        // Coulomb two-electron Fock G[P] (J - ½K for RHF; {J - K_α, J - K_β}
        // for UHF) of a density in the WORKING basis, returned in the working
        // basis: the SCF's own two-electron operator, fitted when the SCF ran
        // RI-JK (scf_ri_jk), exact otherwise. The one place response and
        // gradient code chooses between them. The integral builders are Cartesian: P is lifted with
        // to_cartesian and G brought back with to_working, both the identity
        // in Cartesian mode. For response and gradient code; the SCF's own
        // direct build adds incremental-Fock state on top of the same steps.
        std::expected<Eigen::MatrixXd, std::string> two_electron_fock(
            const HartreeFock::Calculator &calculator,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &density);
        std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string> two_electron_fock_uhf(
            const HartreeFock::Calculator &calculator,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &density_alpha,
            const Eigen::MatrixXd &density_beta);

        // diis_stall_floor: the DIIS residual below which a step is never
        // treated as stalled (see scf.cpp). 1e-10 for HF; the KS loops pass a
        // higher value, since their residual floor sits higher (XC grid noise,
        // and the SOSCF deadband stops at |g| ~ 2e-9).
        bool is_converged(
            const HartreeFock::OptionsSCF &scf_options,
            const IterationMetrics &metrics,
            unsigned int iteration,
            double diis_stall_floor = 1E-10) noexcept;

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
                                                 const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                                 const HartreeFock::Solvation::PCMState *pcm = nullptr);

        // Run the UHF SCF procedure.
        // Uses molecule.multiplicity to derive n_alpha and n_beta.
        // Stores converged alpha and beta channels in calculator._info._scf.
        // Returns an error string if convergence is not achieved.
        std::expected<void, std::string> run_uhf(HartreeFock::Calculator &calculator,
                                                 const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                                 const HartreeFock::Solvation::PCMState *pcm = nullptr);

        // Run the ROHF SCF procedure.
        // Uses molecule.multiplicity to derive the closed/open-shell occupations.
        // Stores alpha/beta densities and one shared MO coefficient set in both channels.
        // Returns an error string if convergence is not achieved.
        std::expected<void, std::string> run_rohf(HartreeFock::Calculator &calculator,
                                                  const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                                  const HartreeFock::Solvation::PCMState *pcm = nullptr);
    } // namespace SCF
} // namespace HartreeFock

#endif // !HF_SCF_H
